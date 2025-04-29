import torch
import torch.nn as nn

from torchaudio.functional import resample

from torchmetrics.functional import(
    scale_invariant_signal_distortion_ratio as si_sdr,
    scale_invariant_signal_noise_ratio as si_snr,
    signal_noise_ratio as snr)

from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility as STOI
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality as PESQ
import numpy as np
from .eval_utils import ild_diff_torch, itd_diff_torch

def compute_decay(est, mix):
    """
    [*, C, T]
    """
    types = type(est)
    assert type(mix) == types, "All arrays must be the same type"
    if types == np.ndarray:
        est, mix = torch.from_numpy(est), torch.from_numpy(mix)

    # Ensure that, no matter what, we do not modify the original arrays
    est = est.clone()
    mix = mix.clone()
    
    P_est = 10 * torch.log10(torch.sum(est ** 2, dim=-1)) # [*, C]
    P_mix = 10 * torch.log10(torch.sum(mix ** 2, dim=-1))

    return (P_mix - P_est).mean(dim=-1) # [*]

class Metrics(nn.Module):
    def __init__(self, name, fs = 16000, **kwargs) -> None:
        super().__init__()
        self.fs = fs
        self.func = None
        self.name=name 
        if name == 'snr':
            self.func = lambda est, gt, mix: snr(preds=est, target=gt)
        elif name == 'snr_i':
            self.func = lambda est, gt, mix: snr(preds=est, target=gt) - snr(preds=mix, target=gt)
        elif name == 'si_snr':
            self.func = lambda est, gt, mix: si_snr(preds=est, target=gt)
        elif name == 'si_snr_i':
            self.func = lambda est, gt, mix: si_snr(preds=est, target=gt) - si_snr(preds=mix, target=gt)
        elif name == 'si_sdr':
            self.func = lambda est, gt, mix: si_sdr(preds=est, target=gt)
        elif name == 'si_sdr_i':
            self.func = lambda est, gt, mix: si_sdr(preds=est, target=gt) - si_sdr(preds=mix, target=gt)
        elif name == 'STOI':
            self.func = lambda est, gt, mix: STOI(preds=est, target=gt, fs=fs)
        elif name == 'L1':
            self.func = lambda est, gt, mix: torch.nn.functional.l1_loss(input=est, target=gt, reduce = False).mean(dim=-1)
        elif name == 'L2':
            self.func = lambda est, gt, mix: torch.nn.functional.l2_loss(input=est, target=gt, reduce = False).mean(dim=-1)
        elif name == 'PESQ':
            fs_new = 16000
            self.func = lambda est, gt, mix: PESQ(preds=resample(est, fs, fs_new), target=resample(gt, fs, fs_new), fs=fs_new, mode = "nb")
        elif name == 'Multi_Reso_L1':
            mult_ireso_loss = MultiResoFuseLoss(**kwargs)
            self.func = lambda est, gt, mix: mult_ireso_loss(est = est, gt = gt)
        elif name == 'PLCPALoss':
            plcpa = PLCPALoss(**kwargs)
            self.func = lambda est, gt, mix: plcpa(est = est, gt = gt)
        elif name == "Hubert":
            huber = HubertLoss(**kwargs)
            self.func = lambda est, gt, mix: huber(est = est, gt = gt)
        elif name == "delta_ILD":
            self.func = lambda est, gt, mix: ild_diff_torch(s_est = est, s_gt = gt)
        elif name == "delta_ITD":
            self.func = lambda est, gt, mix: itd_diff_torch(s_est = est, s_gt = gt, sr = 16000)
        else:
            raise NotImplementedError(f"Metric {name} not implemented!")

    def forward(self, est, gt, mix):
        """
        input: (*, C, T)
        output: (*)
        """
        types = type(est)
        assert type(gt) == types and type(mix) == types, "All arrays must be the same type"
        if types == np.ndarray:
            est, gt, mix = torch.from_numpy(est), torch.from_numpy(gt), torch.from_numpy(mix)
        
        # Ensure that, no matter what, we do not modify the original arrays
        est = est.clone()
        gt = gt.clone()
        mix = mix.clone()

        per_channel_metrics = self.func(est=est, gt=gt, mix=mix) # [*, C]

        if self.name == "PLCPALoss":
            return per_channel_metrics[0].mean(dim=-1), per_channel_metrics[1].mean(dim=-1), per_channel_metrics[2].mean(dim=-1)
        else:
            return per_channel_metrics.mean(dim=-1) # [*]
