"""Image-similarity metrics for quantifying reconstruction quality."""

import torch
import torch.nn.functional as F


def _to_4d(img):
    return img.unsqueeze(0) if img.dim() == 3 else img


def mse(a, b):
    return F.mse_loss(_to_4d(a).float(), _to_4d(b).float()).item()


def psnr(a, b, data_range=1.0):
    m = mse(a, b)
    if m == 0:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(data_range ** 2 / m)).item()


def _gaussian_window(window_size, sigma, channels, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(1)
    window_2d = (g @ g.t()).unsqueeze(0).unsqueeze(0)  # (1,1,ws,ws)
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(a, b, window_size=11, sigma=1.5, data_range=1.0):
    """Structural Similarity Index (mean over the image)."""
    a, b = _to_4d(a).float(), _to_4d(b).float()
    channels = a.shape[1]
    window = _gaussian_window(window_size, sigma, channels, a.device, a.dtype)
    pad = window_size // 2

    mu_a = F.conv2d(a, window, padding=pad, groups=channels)
    mu_b = F.conv2d(b, window, padding=pad, groups=channels)
    mu_a2, mu_b2, mu_ab = mu_a ** 2, mu_b ** 2, mu_a * mu_b

    sigma_a2 = F.conv2d(a * a, window, padding=pad, groups=channels) - mu_a2
    sigma_b2 = F.conv2d(b * b, window, padding=pad, groups=channels) - mu_b2
    sigma_ab = F.conv2d(a * b, window, padding=pad, groups=channels) - mu_ab

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / \
               ((mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2))
    return ssim_map.mean().item()


def all_metrics(reconstructed, original, data_range=1.0):
    """Return {'mse', 'psnr', 'ssim'} comparing a reconstruction to the truth."""
    recon = reconstructed.detach().clamp(0, data_range)
    orig = original.detach().clamp(0, data_range)
    return {
        "mse": mse(recon, orig),
        "psnr": psnr(recon, orig, data_range),
        "ssim": ssim(recon, orig, data_range=data_range),
    }
