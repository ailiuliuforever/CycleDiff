import torch
import torch.nn.functional as F
from util.mse_psnr_ssim_mssim import ssim


def c_ms_ssim_loss(fake, target, win_size=5, scales=3, data_range=1.0):
    """
    Multi-scale SSIM loss for C-space [B, 3, 64, 64].

    For C-space 64x64 with win_size=5, 3-level downscale is supported:
      Level 1: 64x64 -> fine structure
      Level 2: 32x32 -> mid structure
      Level 3: 16x16 -> coarse structure

    Returns scalar = mean(1 - MS-SSIM) across batch.
    """
    loss = 0.0
    f, t = fake, target
    for i in range(scales):
        val = ssim(f, t, data_range=data_range, win_size=win_size,
                   size_average=True)
        loss += (1.0 - val)
        if i < scales - 1:
            f = F.avg_pool2d(f, 2)
            t = F.avg_pool2d(t, 2)
    return loss / scales
