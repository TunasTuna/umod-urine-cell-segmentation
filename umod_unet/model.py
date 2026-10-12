from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


def build_model(cfg: Config) -> nn.Module:
    import segmentation_models_pytorch as smp

    weights = None if cfg.encoder_weights in {"none", "", "None"} else cfg.encoder_weights
    return smp.Unet(
        encoder_name=cfg.encoder,
        encoder_weights=weights,
        in_channels=3,
        classes=cfg.n_classes,
        encoder_depth=5,
        decoder_channels=(256, 128, 64, 32, 16),
    )


def pad_to_multiple(x: torch.Tensor, div: int = 32) -> torch.Tensor:
    h, w = x.shape[-2:]
    ph = (div - h % div) % div
    pw = (div - w % div) % div
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph))
    return x


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    img: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    tta: bool,
) -> torch.Tensor:
    """img: (1, 3, H, W) float tensor already normalized."""
    _, _, h, w = img.shape
    x = pad_to_multiple(img.to(device), 32)
    amp_on = use_amp and device.type == "cuda"

    def _fwd(inp: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_on):
            return model(inp).float()

    logits = _fwd(x)
    if tta:
        logits = logits + _fwd(x.flip(-1)).flip(-1)
        logits = logits + _fwd(x.flip(-2)).flip(-2)
        logits = logits / 3.0
    return logits[..., :h, :w]


def predict_full(
    model: nn.Module,
    img_chw: np.ndarray,
    device: torch.device,
    use_amp: bool,
    tta: bool,
) -> np.ndarray:
    x = torch.from_numpy(img_chw)[None]
    logits = predict_logits(model, x, device, use_amp, tta)
    return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
