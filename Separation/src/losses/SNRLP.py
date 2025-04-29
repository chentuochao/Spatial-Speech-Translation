import torch
import torch.nn as nn
import math

from src.losses.SNRLosses import SNRLosses
from src.losses.LogPowerLoss import LogPowerLoss



class MyL1Loss(nn.Module):
    def __init__(self, use_diff = False) -> None:
        super().__init__()
        self.l1_loss = nn.L1Loss()#LogPowerLoss()

    def forward(self, est: torch.Tensor, gt: torch.Tensor, **kwargs):
        """
        input: (B, C, T) (B, C, T)
        """
        return self.l1_loss(est, gt)

class SNRLPLoss(nn.Module):
    def __init__(self, snr_loss_name = "snr", neg_weight = 1) -> None:
        super().__init__()
        self.snr_loss = SNRLosses(snr_loss_name)
        #self.lp_loss = LogPowerLoss()
        self.lp_loss = nn.L1Loss()#LogPowerLoss()
        self.neg_weight = neg_weight
    
    def forward(self, est: torch.Tensor, gt: torch.Tensor, **kwargs):
        """
        input: (B, C, T) (B, C, T)
        """
        
        neg_loss = 0
        pos_loss = 0

        comp_loss = torch.zeros((est.shape[0]), device=est.device)
        mask = (torch.max(torch.max(torch.abs(gt), dim=2)[0], dim=1)[0] == 0)
        # If there's at least one negative sample
        if any(mask):
            est_neg, gt_neg = est[mask], gt[mask]
            neg_loss = self.lp_loss(est_neg, gt_neg)
            comp_loss[mask] = neg_loss * self.neg_weight
            
        # If there's at least one positive sample
        if any((~ mask)):
            est_pos, gt_pos = est[~mask], gt[~mask]
            pos_loss = self.snr_loss(est_pos, gt_pos)
            
            # Compute_joint_loss
            comp_loss[~mask] = pos_loss
        
        return comp_loss