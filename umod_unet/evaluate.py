from __future__ import annotations

import numpy as np
import torch

from .config import CLASS_NAMES, Config
from .data import normalize_image
from .model import predict_full


@torch.no_grad()
def per_image_conf(
    model: torch.nn.Module,
    imgs: np.ndarray,
    masks: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    cfg: Config,
) -> np.ndarray:
    model.eval()
    confs = np.zeros((len(imgs), 8, 8), np.int64)
    for i in range(len(imgs)):
        pred = predict_full(model, normalize_image(imgs[i], mean, std), device, cfg.amp, cfg.use_tta)
        confs[i] = np.bincount((masks[i].astype(np.int64) * 8 + pred).ravel(), minlength=64).reshape(8, 8)
    return confs


def scores(conf: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    tp = np.diag(conf).astype(np.float64)
    gt, pr = conf.sum(1).astype(np.float64), conf.sum(0).astype(np.float64)
    dice = np.where(gt > 0, 2 * tp / np.maximum(gt + pr, 1), np.nan)
    iou = np.where(gt > 0, tp / np.maximum(gt + pr - tp, 1), np.nan)
    fg = 2.0 * conf[1:, 1:].sum() / max(conf[1:, :].sum() + conf[:, 1:].sum(), 1)
    return dice, iou, float(fg)


def evaluate_split(
    model: torch.nn.Module,
    imgs: np.ndarray,
    masks: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    cfg: Config,
) -> dict:
    d, i, fg = scores(per_image_conf(model, imgs, masks, mean, std, device, cfg).sum(0))
    return {
        "macro_dice": float(np.nanmean(d[1:])),
        "macro_iou": float(np.nanmean(i[1:])),
        "fg_dice": float(fg),
        "dice": d,
        "iou": i,
    }


def bootstrap(confs: np.ndarray, B: int = 2000, seed: int = 0) -> dict:
    n = len(confs)
    d0, i0, fg0 = scores(confs.sum(0))
    idx = np.random.default_rng(seed).integers(0, n, size=(B, n))
    D, I, FG = np.empty((B, 8)), np.empty((B, 8)), np.empty(B)
    for b in range(B):
        D[b], I[b], FG[b] = scores(confs[idx[b]].sum(0))
    n_with = [int((confs[:, c, :].sum(1) > 0).sum()) for c in range(8)]
    return {"n": n, "dice": d0, "iou": i0, "fg": fg0, "D": D, "I": I, "FG": FG, "n_with": n_with}


def ci(a: np.ndarray) -> tuple[float, float]:
    return float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))


def markdown_table(res: dict) -> str:
    lines = [
        "| class | Dice [95% CI] | IoU [95% CI] | images containing it |",
        "|---|---|---|---|",
    ]
    for c in range(1, 8):
        dl, dh = ci(res["D"][:, c])
        il, ih = ci(res["I"][:, c])
        lines.append(
            "| %s | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %d |"
            % (CLASS_NAMES[c], res["dice"][c], dl, dh, res["iou"][c], il, ih, res["n_with"][c])
        )
    md_, mi_ = np.nanmean(res["D"][:, 1:], axis=1), np.nanmean(res["I"][:, 1:], axis=1)
    lines.append(
        "| **macro (classes 1-7)** | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | |"
        % (np.nanmean(res["dice"][1:]), *ci(md_), np.nanmean(res["iou"][1:]), *ci(mi_))
    )
    lines.append("")
    lines.append("Foreground-vs-background Dice: %.3f [%.3f, %.3f]" % (res["fg"], *ci(res["FG"])))
    return "\n".join(lines)
