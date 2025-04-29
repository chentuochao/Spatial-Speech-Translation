import torch


def ILD(x1, x2, tol = 1e-6):
    # x - B, T
    # ILD - B, F, T
    ILD = torch.log10(torch.div(x1.abs() + tol, x2.abs() + tol))
    return ILD

def IPD(x1, x2, tol = 1e-6):
    # x - B, T
    # ILD - B, F, T
    IPD =  torch.angle(x1) -  torch.angle(x2)
    IPD_cos = torch.cos(IPD)
    IPD_sin = torch.sin(IPD)
    IPD_map = torch.cat((IPD_sin, IPD_cos), dim = 1)
    return IPD_map


def IPD_ONNX(real1, imag1, real2, imag2, norm, norm_ref, tol = 1e-6):
    B, _, f, T = real2.shape

    real2 = real2.repeat((1, real1.shape[1], 1, 1))#.reshape(B*(M-1), 1, f, T)
    imag2 = imag2.repeat((1, imag1.shape[1], 1, 1))#.reshape(B*(M-1), 1, f, T)

    IPD_cos = (real1 * real2 + imag1 * imag2) / (norm * norm_ref + tol)
    IPD_sin = (real2 * imag1 - imag2 * real1) / (norm * norm_ref + tol)
    
    IPD_cos = IPD_cos.reshape(-1, 1, f, T)
    IPD_sin = IPD_sin.reshape(-1, 1, f, T)
    
    IPD_map = torch.cat((IPD_sin, IPD_cos), dim = 1)

    IPD_map = IPD_map.reshape(B, 2 * imag1.shape[1], f, T)

    return IPD_map

def MC_features_ONNX(reals, imags, eps=1e-6):
    # Input: [B, M, F, T] or [B, M, T, F]
    r2, r1 = torch.split(reals, [1, reals.shape[1] - 1], dim=1)
    i2, i1 = torch.split(imags, [1, reals.shape[1] - 1], dim=1)
    
    # Compute magnitude
    norm = torch.sqrt(torch.square(reals) + torch.square(imags))
    norm_ref, norm = torch.split(norm, [1, norm.shape[1] - 1], dim=1)

    # Compute ILD
    ILD_m = torch.log10(torch.div(norm + eps, norm_ref + eps))

    # Compute IPD
    IPD_m = IPD_ONNX(r1, i1, r2, i2, norm, norm_ref)

    out = torch.cat([ILD_m, IPD_m], dim=1) # [B, 3M-3, f, T]
    
    return out