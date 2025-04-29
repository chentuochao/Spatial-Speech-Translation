import torch
import torch.nn as nn


class LogPowerLoss(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def forward(self, est: torch.Tensor, gt: torch.Tensor, **kwargs):
        """
        est: (B, C, T)
        gt: (B, C, T)

        return: (B)
        """
        B, C, T = est.shape

        assert torch.abs(gt).max() < 1e-6, "This loss must only be used when gt = 0"
        est = est.reshape(B*C, T) # [BC, T]
        loss = 10 * torch.log10(torch.sum(est ** 2, axis=-1) + 1e-3)
        loss = loss.reshape(B, C) # [B, C]
        loss = loss.mean(axis=-1) # [B]
        return loss