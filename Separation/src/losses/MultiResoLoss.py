import torch
import torch.nn as nn
import auraloss


class MultiResoFuseLoss(nn.Module):
    def __init__(self, l1_ratio=0,  **kwargs) -> None:
        super().__init__()

        self.l1_ratio = l1_ratio
        self.l1 = nn.L1Loss()
        self.loss_fn = auraloss.freq.MultiResolutionSTFTLoss(**kwargs)


    def forward(self, est: torch.Tensor, gt: torch.Tensor, **kwargs):
        """
        est: (B, C, T)
        gt: (B, C, T)
        """
        B, C, T = est.shape

        if self.l1_ratio > 0:
            loss1 = self.loss_fn(est, gt) + self.l1_ratio * self.l1(est, gt)
        else:
            loss1 =  self.loss_fn(est, gt) 
        
        return loss1
