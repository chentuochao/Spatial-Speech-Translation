import torch
import torch.nn as nn

from .tfgridnet_causal import TFGridNet
import torch.nn.functional as F
from  scipy.signal.windows import tukey
import numpy as np
from src.models.common.dsp import MC_features_ONNX

def mod_pad(x, chunk_size, pad):
    # Mod pad the input to perform integer number of
    # inferences
    mod = 0
    if (x.shape[-1] % chunk_size) != 0:
        mod = chunk_size - (x.shape[-1] % chunk_size)

    x = F.pad(x, (0, mod))
    x = F.pad(x, pad)

    return x, mod

class Net(nn.Module):
    def __init__(self, stft_chunk_size=64, stft_pad_size = 32, stft_back_pad = 32,
                 num_ch=2, D=64, B=6, I=1, J=1, L=0, H=128, local_atten_len=100,
                 E = 4, chunk_causal=False, num_src = 2,
                 spectral_masking=False, use_first_ln=False, merge_method = "None",
                 conv_lstm = True, lstm_down=5, use_sp_feats = True):
        super(Net, self).__init__()
        self.num_ch = num_ch
        num_separator_inputs = num_ch * 2 + use_sp_feats * (3 * (num_ch - 1))

        self.stft_chunk_size = stft_chunk_size
        self.stft_pad_size = stft_pad_size
        self.stft_back_pad = stft_back_pad
        self.layers = B
        self.n_srcs = num_src
        self.use_sp_feats = use_sp_feats

        self.embed_dim = D
        self.E = E

        # Input conv to convert input audio to a latent representation        
        self.nfft = stft_back_pad + stft_chunk_size + stft_pad_size
        
        self.nfreqs = self.nfft//2 + 1

        # Construct synthesis/analysis windows

        # window_fn = lambda x: tukey(x, alpha = ((stft_pad_size+stft_back_pad) / 2) /self.nfft)
        window_fn = lambda x: np.ones(x)
        
        # If possible, use perfect reconstruction windows
        if ((stft_pad_size) % stft_chunk_size) == 0:
            print("Using perfect STFT windows")
            # self.analysis_window = torch.from_numpy( window_fn(self.nfft) ).float()
            self.analysis_window = torch.from_numpy( window_fn(self.nfft) ).float()

            self.synthesis_window = torch.zeros(stft_pad_size + stft_chunk_size).float()
            
            A = self.synthesis_window.shape[0]
            B = self.stft_chunk_size
            N = self.analysis_window.shape[0]
            
            assert (A % B) == 0
            for i in range(A):
                num = self.analysis_window[N - A + i]
            
                denom = 0
                for k in range(A//B):
                    denom += (self.analysis_window[N - A + (i % B) + k * B] ** 2)
                
                self.synthesis_window[i] = num / denom
        else:
            print("Using imperfect STFT windows")
            self.analysis_window = torch.from_numpy( window_fn(self.nfft) ).float()
            self.synthesis_window = torch.from_numpy( window_fn(self.nfft) ).float()

        # Number of chunks to use for overlap & add
        self.istft_lookback = 1 + (self.synthesis_window.shape[0] - 1) // self.stft_chunk_size

        # TF-GridNet        
        self.tfgridnet = TFGridNet(None,
                                   n_srcs=num_src,
                                   n_fft=self.nfft,
                                   stride=stft_chunk_size,
                                   emb_dim=D,
                                   emb_ks=I,
                                   emb_hs=J,
                                   n_layers=self.layers,
                                   num_inputs=num_separator_inputs,
                                   attn_n_head=L,
                                   attn_approx_qk_dim=E*self.nfreqs,
                                   lstm_hidden_units=H,
                                   local_atten_len=local_atten_len,
                                   chunk_causal = chunk_causal,
                                   spectral_masking = spectral_masking,
                                   conv_lstm = conv_lstm,
                                   lstm_down=lstm_down)

    def init_buffers(self, batch_size, device):
        buffers = {}
        
        buffers['tfgridnet_bufs'] = self.tfgridnet.init_buffers(batch_size, device)
        buffers['istft_buf'] = torch.zeros(batch_size * self.n_srcs, self.synthesis_window.shape[0], self.istft_lookback, device=device)

        return buffers

    def extract_features(self, x):
        """
        x: (B, M, T)
        returns: (B, M, C*F, T)
        """
        B, M, T = x.shape

        x = x.reshape(B*M, T)
        x = torch.stft(x, n_fft = self.nfft, hop_length = self.stft_chunk_size,
                          win_length = self.nfft, window=self.analysis_window.to(x.device),
                          center=False, normalized=False, return_complex=True)
        x = torch.view_as_real(x) # [B*M, F, T, 2]
        BM, _F, T, C = x.shape
        
        x = x.reshape(B, M, _F, T, C) # [B, M, F, T, 2]

        x = x.permute(0, 4, 1, 3, 2) # [B, 2, M. T, F]

        if self.use_sp_feats:
            feats = MC_features_ONNX(x[:, 0], x[:, 1]) # [B, 3(M - 1), T, F]
            
            x = x.reshape(B, C*M, T, _F) # [B, 2M, T, F]
            x = torch.cat([x, feats], dim=1) # [B, 5M - 3, T, F]
        else:
            x = x.reshape(B, C*M, _F, T) # [B, 2M, T, F]

        return x

    def synthesis(self, x, input_state):
        """
        x: (B, S, C*F, T)
        returns: (B, S, t) 
        """
        istft_buf = input_state['istft_buf']

        B, S, CF, T = x.shape
        X = x.reshape(B*S, CF, T)
        X = X.reshape(B*S, 2, -1, T).permute(0, 2, 3, 1) # [BS, F, T, C]
        X = X[..., 0] + 1j * X[..., 1]

        x = torch.fft.irfft(X, dim=1) # [BS, iW, T]
        x = x[:, -self.synthesis_window.shape[0]:] # [BS, oW, T]

        # Apply synthesis window
        x = x * self.synthesis_window.unsqueeze(0).unsqueeze(-1).to(x.device)

        oW = self.synthesis_window.shape[0]

        # Concatenate blocks from previous IFFTs
        x = torch.cat([istft_buf, x], dim=-1)
        istft_buf = x[..., -istft_buf.shape[1]:] # Update buffer

        # Get full signal
        x = F.fold(x, output_size=(self.stft_chunk_size * x.shape[-1] + (oW - self.stft_chunk_size), 1),
                      kernel_size=(oW, 1), stride=(self.stft_chunk_size, 1)) # [BS, 1, t]
        
        # Drop samples from previous chunks and from pad
        x = x[:, :, -T * self.stft_chunk_size - self.stft_pad_size: - self.stft_pad_size] 
        x = x.reshape(B, S, -1) # [B, S, t]

        input_state['istft_buf'] = istft_buf

        return x, input_state

    def predict(self, x, input_state, pad=True):
        """
        B: batch
        M: mic
        t: time step (time-domain)
        x: (B, M, t)
        R: real or imaginary
        """

        mod = 0
        if pad:
            pad_size = (self.stft_back_pad, self.stft_pad_size)
            x, mod = mod_pad(x, chunk_size=self.stft_chunk_size, pad=pad_size)

        # Time-domain to TF-domain
        x = self.extract_features(x) # [B, RM, F, T]

        x, input_state['tfgridnet_bufs'] = self.tfgridnet(x, input_state['tfgridnet_bufs'])
        
        # TF-domain to time-domain
        x, next_state = self.synthesis(x, input_state) # [B, S, F, T]
        
        if mod != 0:
            x = x[:, :, :-mod]

        return x, next_state

    def forward(self, inputs, input_state = None, pad=True):
        x = inputs['mixture']

        if input_state is None:
            input_state = self.init_buffers(x.shape[0], x.device)

        x, next_state = self.predict(x, input_state, pad)

        return {'output': x, 'next_state': next_state}

if __name__ == "__main__":
    pass