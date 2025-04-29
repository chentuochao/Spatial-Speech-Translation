import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Lambda(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        import types
        assert type(lambd) is types.LambdaType
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class LayerNormPermuted(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(LayerNormPermuted, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        Args:
            x: [B, C, T, F]
        """
        x = x.permute(0, 2, 3, 1) # [B, T, F, C]
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2) # [B, C, T, F]
        return x

class TFGridNet(nn.Module):
    def __init__(
        self,
        input_dim,
        n_srcs=2,
        n_fft=128,
        stride=64,
        window="hann",
        num_inputs=1,
        n_layers=6,
        lstm_hidden_units=192,
        lstm_down=4,
        attn_n_head=4,
        attn_approx_qk_dim=512,
        emb_dim=48,
        emb_ks=1,
        emb_hs=1,
        activation="prelu",
        eps=1.0e-5,
        ref_channel=-1,
        use_attn=False,
        chunk_causal=True,
        local_atten_len=100,
        spectral_masking=True,
        merge_method = "None",
        conv_lstm = True
    ):
        super().__init__()
        self.n_srcs = n_srcs
        self.n_layers = n_layers
        self.num_inputs = num_inputs
        assert n_fft % 2 == 0
        n_freqs = n_fft // 2 + 1
        self.n_freqs = n_freqs
        self.ref_channel = ref_channel
        self.emb_dim = emb_dim
        self.eps = eps
        self.chunk_size = stride
        self.spectral_masking = spectral_masking
        self.merge_method = merge_method
        
        self.n_fft = n_fft
        self.window = window

        t_ksize = 3
        self.t_ksize = t_ksize
        ks, padding = (t_ksize, 3), (0, 1)
        
        # Initialize first convolution
        self.conv = nn.Sequential(
            nn.Conv2d(num_inputs, emb_dim, ks, padding=padding),
            LayerNormPermuted(emb_dim)
        )

        self.blocks = nn.ModuleList([])
        for _ in range(n_layers):
            self.blocks.append(
                GridNetBlock(
                    emb_dim,
                    emb_ks,
                    emb_hs,
                    n_freqs,
                    lstm_hidden_units,
                    lstm_down,
                    n_head=attn_n_head,
                    approx_qk_dim=attn_approx_qk_dim,
                    activation=activation,
                    eps=eps,
                    use_attn=use_attn,
                    chunk_causal=chunk_causal,
                    local_atten_len=local_atten_len,
                    conv_lstm = conv_lstm
                )
            )

        self.deconv = nn.ConvTranspose2d(emb_dim, n_srcs * 2, ks, padding=( self.t_ksize - 1, 1))
    
    def init_buffers(self, batch_size, device):
        conv_buf = torch.zeros(batch_size, self.num_inputs, self.t_ksize - 1, self.n_freqs,
                device=device)
            
        deconv_buf = torch.zeros(batch_size, self.emb_dim, self.t_ksize - 1, self.n_freqs,
                                 device=device)

        gridnet_buffers = {}
        for i in range(len(self.blocks)):
            gridnet_buffers[f'buf{i}'] = self.blocks[i].init_buffers(batch_size, device)

        return dict(conv_buf=conv_buf, deconv_buf=deconv_buf,
                    gridnet_bufs=gridnet_buffers)

    def forward(self, input_stft: torch.Tensor, input_state) -> torch.Tensor:
        """
        B: batch, M: mic, R: real/imag, F: freq bin, T: time step (TF-domain)
        C: feature
        input_stft: (B, RM, T, F)
        output: (B, M, R*F, T)
        """
        n_batch, _, n_frames, n_freqs = input_stft.shape
        batch = input_stft

        if input_state is None:
            input_state = self.init_buffers(input_stft.shape[0], input_stft.device)
        
        conv_buf = input_state['conv_buf']
        deconv_buf = input_state['deconv_buf']
        gridnet_buf = input_state['gridnet_bufs']
            
        batch = torch.cat((conv_buf, batch), dim=2)
        conv_buf = batch[:, :,  -(self.t_ksize - 1):, :]
        
        batch = self.conv(batch)  # [B, -1, T, F]

        # BCTQ
        batch = batch.permute(0, 2, 3, 1)
        
        for ii in range(self.n_layers):
            batch, gridnet_buf[f'buf{ii}'] = self.blocks[ii](batch, gridnet_buf[f'buf{ii}']) # [B, T, Q, C]

        batch = batch.permute(0, 3, 1, 2) # [B, C, T, Q]
        
        batch = torch.cat(( deconv_buf, batch), dim=2)
        deconv_buf = batch[:, :,  -(self.t_ksize - 1):, :]
        
        batch = self.deconv(batch)  # [B, n_srcs*R, T, F]batch ] 
        batch = batch.view([n_batch, self.n_srcs, 2, n_frames, n_freqs]) # [B, n_srcs, R, n_frames, n_freqs]
        
        batch = batch.transpose(3, 4) # (B, n_srcs, R, n_fft//2 + 1, T)
        
        # Concat real and imaginary parts
        batch = torch.cat([batch[:, :, 0], batch[:, :, 1]], dim=2) # (B, n_srcs, nfft + 2, T)

        # Do spectral masking
        if self.spectral_masking:
            batch = batch * input_stft[:, :self.n_srcs] # First few channels only

        input_state['conv_buf'] = conv_buf
        input_state['deconv_buf'] = deconv_buf
        input_state['gridnet_bufs'] = gridnet_buf

        return batch, input_state


class GridNetBlock(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(
        self,
        emb_dim,
        emb_ks,
        emb_hs,
        n_freqs,
        hidden_channels,
        lstm_down,
        n_head=4,
        local_atten_len= 100,
        approx_qk_dim=512,
        activation="prelu",
        eps=1e-5,
        use_attn=True,
        chunk_causal = True,
        conv_lstm = True
    ):
        super().__init__()
        bidirectional = False # Causal
        self.local_atten_len = local_atten_len
        self.E = math.ceil(
                    approx_qk_dim * 1.0 / n_freqs
                )  # approx_qk_dim is only approximate
        
        self.n_head = n_head
        self.V_dim = emb_dim // n_head
        self.H = hidden_channels
        self.lstm_down = lstm_down
        
        in_channels = emb_dim
        self.in_channels = in_channels
        self.n_freqs = n_freqs

        ## intra RNN can be optimized by conv or linear because the frequence length are not very large
        self.conv_lstm = conv_lstm
        if conv_lstm:
            self.conv = nn.Conv1d(in_channels=emb_dim,
                                out_channels=emb_dim*lstm_down,
                                kernel_size=lstm_down,
                                stride=lstm_down)
            self.act = nn.PReLU()
            self.norm = LayerNormalization4D(emb_dim*lstm_down)
            
            self.intra_rnn = nn.LSTM(
                emb_dim*lstm_down, hidden_channels, 1, batch_first=True, bidirectional=True
            )
            
            self.deconv = nn.ConvTranspose1d(in_channels=hidden_channels * 2,
                                            out_channels=emb_dim,
                                            kernel_size=lstm_down,
                                            stride=lstm_down,
                                            output_padding=n_freqs - (n_freqs//lstm_down) * lstm_down)
        else:
            self.intra_norm = LayerNormalization4D(emb_dim, eps=eps)
            self.intra_rnn = nn.LSTM(
                in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
            )
            self.intra_linear = nn.Linear(
                hidden_channels*2, emb_dim,
            )

        self.inter_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM( 
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=bidirectional,
        )
        self.inter_linear = nn.Linear(
            hidden_channels*(bidirectional + 1), emb_dim
        )
        
        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head
    
    def init_buffers(self, batch_size, device):
        ctx_buf = {}

        c0 = torch.zeros((1,
                          batch_size * self.n_freqs,
                          self.H), device=device)
        ctx_buf['c0'] = c0

        h0 = torch.zeros((1,
                          batch_size * self.n_freqs,
                          self.H), device=device)
        ctx_buf['h0'] = h0

        return ctx_buf

    def forward(self, x, init_state = None, debug=False):
        """GridNetBlock Forward.

        Args:
            x: [B, T, Q, C]
            out: [B, T, Q, C]
        """
        
        if init_state is None:
            init_state = self.init_buffers(x.shape[0], Q.device)

        B, T, Q, C = x.shape
        
        # intra RNN
        input_ = x

        if self.conv_lstm:
            intra_rnn = input_.reshape(B * T, Q, C)  # [B * T, Q, C]
            
            intra_rnn = self.conv(intra_rnn.transpose(1, 2)) # [BT, C, K] K = Q // stride
            intra_rnn = self.act(intra_rnn)
            intra_rnn = self.norm(intra_rnn.transpose(1, 2)) # [BT, K, C]
            
            self.intra_rnn.flatten_parameters()
            
            intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, -1, H]
            
            intra_rnn = self.deconv(intra_rnn.transpose(1, 2)) # [BT, C, Q]

            intra_rnn = intra_rnn.transpose(1, 2)
        else:
            intra_rnn = self.intra_norm(input_) # [B, T, Q, C]
            intra_rnn = intra_rnn.reshape(B * T, Q, C)  # [B * T, Q, C]
            self.intra_rnn.flatten_parameters()

            intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, -1, H]
            intra_rnn = self.intra_linear(intra_rnn)  # [BT, Q, C]
        
        intra_rnn = intra_rnn.view(B, T, Q, C) # [B, T, Q, C]
        intra_rnn = intra_rnn + input_  # [B, T, Q, C]
        out = intra_rnn

        # inter RNN
        input_ = intra_rnn # [B, T, Q, C]
        
        inter_rnn = self.inter_norm(intra_rnn)  # [B, T, Q, C]
        inter_rnn = inter_rnn.transpose(1, 2).reshape(B * Q, T, C)  # [BQ, T, C]
        
        self.inter_rnn.flatten_parameters()
        
        h0 = init_state['h0']
        c0 = init_state['c0']

        inter_rnn, (h0, c0) = self.inter_rnn(inter_rnn, (h0, c0))  # [BQ, -1, H]
       
        init_state['h0'] = h0
        init_state['c0'] = c0
       
        inter_rnn = self.inter_linear(inter_rnn)  # [BQ, T, C]
        
        inter_rnn = inter_rnn.view([B, Q, T, C])
        inter_rnn = inter_rnn.transpose(1, 2) # [B, T, Q, C]
        inter_rnn = inter_rnn + input_  # [B, T, Q, C]
        
        out = inter_rnn

        return out, init_state


# Use native layernorm implementation
class LayerNormalization4D(nn.Module):
    def __init__(self, C, eps=1e-5, preserve_outdim=False):
        super().__init__()
        self.norm = nn.LayerNorm(C, eps=eps)
        self.preserve_outdim = preserve_outdim

    def forward(self, x: torch.Tensor):
        """
        input: (*, C)
        """
        x = self.norm(x)
        return x
    
class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        assert len(input_dimension) == 2
        Q, C = input_dimension
        super().__init__()
        self.norm = nn.LayerNorm((Q * C), eps=eps)

    def forward(self, x: torch.Tensor):
        """
        input: (B, T, Q * C)
        """
        x = self.norm(x)

        return x



if __name__ == "__main__":
    pass