"""Pure-PyTorch stand-in for fused_ssim (github.com/rahul-goel/fused-ssim), which gsplat's
simple_trainer imports. That package is a CUDA extension with no Windows build; this one is
slower but needs no compiler. run_job.py puts it on the trainer's path only when the real
package is missing.

Same call: fused_ssim(img1, img2, padding="same" | "valid", train=True) → mean SSIM,
images as (N, C, H, W) in 0..1, 11×11 Gaussian window (σ = 1.5).
"""
import torch
import torch.nn.functional as F

C1, C2 = 0.01 ** 2, 0.03 ** 2


def _window(channels, device, dtype, size=11, sigma=1.5):
    x = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(channels, 1, size, size).contiguous()


def fused_ssim(img1, img2, padding="same", train=True):
    channels = img1.shape[1]
    window = _window(channels, img1.device, img1.dtype)
    pad = 0 if padding == "valid" else window.shape[-1] // 2

    def blur(t):
        return F.conv2d(t, window, padding=pad, groups=channels)

    mu1, mu2 = blur(img1), blur(img2)
    var1 = blur(img1 * img1) - mu1 ** 2
    var2 = blur(img2 * img2) - mu2 ** 2
    cov = blur(img1 * img2) - mu1 * mu2
    ssim = ((2 * mu1 * mu2 + C1) * (2 * cov + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (var1 + var2 + C2))
    return ssim.mean()
