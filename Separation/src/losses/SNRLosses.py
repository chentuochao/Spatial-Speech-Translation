import torch
import torch.nn as nn
from asteroid.losses.sdr import SingleSrcNegSDR


class SNRLosses(nn.Module):
    def __init__(self, name, **kwargs) -> None:
        super().__init__()
        self.name = name
        if name == 'sisdr':
            self.loss_fn = SingleSrcNegSDR('sisdr')
        elif name == 'snr':
            self.loss_fn = SingleSrcNegSDR('snr')
        elif name == 'sdsdr':
            self.loss_fn = SingleSrcNegSDR('sdsdr')
        else:
            assert 0, f"Invalid loss function used: Loss {name} not found"

    def forward(self, est: torch.Tensor, gt: torch.Tensor, **kwargs):
        """
        est: (B, C, T)
        gt: (B, C, T)
        """
        B, C, T = est.shape

        assert (torch.isnan(est).max() == 0), "Output tensor has nan!"
        assert (torch.isnan(gt).max() == 0), "GT tensor has nan!"

        est = est.reshape(B*C, T)
        gt = gt.reshape(B*C, T)
        
        return self.loss_fn(est_target=est, target=gt)
