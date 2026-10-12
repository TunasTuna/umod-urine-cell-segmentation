from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from .config import EXPECTED_TRAIN_COUNTS, RARE_PASTE_CLASSES, Config

try:
    import tifffile
except ImportError as exc:
    raise SystemExit("Install tifffile: pip install tifffile") from exc


SPLITS = ("train", "validation", "test")


def files_of(data_dir: Path, split: str, sub: str) -> list[Path]:
    return sorted(p for p in (data_dir / split / sub).rglob("*.tif*") if p.is_file())


def load_split(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    imgs = np.stack([tifffile.imread(p) for p in files_of(data_dir, split, "img")])
    masks = np.stack([tifffile.imread(p) for p in files_of(data_dir, split, "mult_mask")])
    return imgs, masks


def load_all(data_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    data = {}
    for split in SPLITS:
        imgs, masks = load_split(data_dir, split)
        print(
            f"{split} | images {imgs.shape} {imgs.dtype} | masks {masks.shape} {masks.dtype} "
            f"| class ids {np.unique(masks).tolist()}"
        )
        data[split] = (imgs, masks)
    tr_masks = data["train"][1]
    vals, cnts = np.unique(tr_masks, return_counts=True)
    matched = dict(zip(vals.tolist(), cnts.tolist())) == EXPECTED_TRAIN_COUNTS
    print("train class pixel counts match the expected values:", matched)
    return data


def channel_mean_std(imgs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_pix = 0
    s1 = np.zeros(3, np.float64)
    for im in imgs:
        a = im.astype(np.float64) / 65535.0
        s1 += a.sum(axis=(0, 1))
        n_pix += a.shape[0] * a.shape[1]
    mean64 = s1 / n_pix
    s2 = np.zeros(3, np.float64)
    for im in imgs:
        a = im.astype(np.float64) / 65535.0
        s2 += ((a - mean64) ** 2).sum(axis=(0, 1))
    mean = mean64.astype(np.float32)
    std = np.sqrt(s2 / n_pix).astype(np.float32)
    return mean, std


def class_pixels(masks: np.ndarray, max_per_class: int = 50_000, seed: int = 0) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {}
    for c in range(1, 8):
        idx = np.argwhere(masks == c)
        if len(idx) > max_per_class:
            idx = idx[rng.choice(len(idx), max_per_class, replace=False)]
        out[c] = idx
    return out


def ce_class_weights(counts: dict[int, int], n_classes: int = 8) -> torch.Tensor:
    """Log-inverse frequency, mean-normalized so the scale stays near 1."""
    w = np.ones(n_classes, np.float32)
    for c in range(n_classes):
        w[c] = 1.0 / np.log(1.02 + counts.get(c, 1))
    w /= w.mean()
    w[0] = min(w[0], 0.35)
    w[3] *= 1.4  # yeast
    w[1] *= 1.15  # rod
    w[4] *= 1.15  # misc
    return torch.from_numpy(w.astype(np.float32))


def extract_rare_instances(
    imgs: np.ndarray,
    masks: np.ndarray,
    classes: tuple[int, ...] = RARE_PASTE_CLASSES,
    min_area: int = 6,
    max_side: int = 160,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Connected components of rare classes, stored as tight RGB + label patches."""
    patches: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(len(masks)):
        for c in classes:
            labeled, n = ndimage.label(masks[i] == c)
            for k in range(1, n + 1):
                ys, xs = np.where(labeled == k)
                if ys.size < min_area:
                    continue
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                if (y1 - y0) > max_side or (x1 - x0) > max_side:
                    continue
                inst = np.zeros((y1 - y0, x1 - x0), np.uint8)
                inst[ys - y0, xs - x0] = c
                rgb = imgs[i, y0:y1, x0:x1].copy()
                patches.append((rgb, inst))
    return patches


def _paste(patch: np.ndarray, mask: np.ndarray, inst_rgb: np.ndarray, inst_m: np.ndarray, rng: np.random.Generator):
    ph, pw = inst_m.shape
    H, W = mask.shape
    if ph >= H or pw >= W:
        return
    y0 = int(rng.integers(0, H - ph + 1))
    x0 = int(rng.integers(0, W - pw + 1))
    sel = inst_m > 0
    patch[y0 : y0 + ph, x0 : x0 + pw][sel] = inst_rgb[sel].astype(np.float32) / 65535.0
    mask[y0 : y0 + ph, x0 : x0 + pw][sel] = inst_m[sel]


def sample_batch(
    imgs: np.ndarray,
    masks: np.ndarray,
    class_pix: dict[int, np.ndarray],
    cfg: Config,
    rng: np.random.Generator,
    mean: np.ndarray,
    std: np.ndarray,
    paste_bank: list[tuple[np.ndarray, np.ndarray]] | None = None,
    p_class: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    p_class = cfg.p_class if p_class is None else p_class
    bs, crop = cfg.batch_size, cfg.crop
    H, W = masks.shape[1:]
    classes = np.array(list(cfg.class_sample_weights.keys()), dtype=np.int64)
    weights = np.array([cfg.class_sample_weights[c] for c in classes], dtype=np.float64)
    weights /= weights.sum()

    xb = np.empty((bs, crop, crop, 3), np.float32)
    yb = np.empty((bs, crop, crop), np.int64)
    for k in range(bs):
        if rng.random() < p_class:
            c = int(rng.choice(classes, p=weights))
            pix = class_pix[c]
            i, cy, cx = pix[int(rng.integers(len(pix)))]
            y0 = int(np.clip(cy - rng.integers(crop // 8, crop - crop // 8), 0, H - crop))
            x0 = int(np.clip(cx - rng.integers(crop // 8, crop - crop // 8), 0, W - crop))
        else:
            i = int(rng.integers(len(imgs)))
            y0 = int(rng.integers(0, H - crop + 1))
            x0 = int(rng.integers(0, W - crop + 1))
        patch = imgs[i, y0 : y0 + crop, x0 : x0 + crop].astype(np.float32) / 65535.0
        m = masks[i, y0 : y0 + crop, x0 : x0 + crop].copy()

        if paste_bank and rng.random() < cfg.p_paste:
            n_paste = int(rng.integers(1, 4))
            for _ in range(n_paste):
                rgb, inst = paste_bank[int(rng.integers(len(paste_bank)))]
                _paste(patch, m, rgb, inst, rng)

        if rng.random() < 0.8:
            patch = patch * float(rng.uniform(0.85, 1.15)) + float(rng.uniform(-0.02, 0.02))
            patch = np.clip(patch, 0.0, 1.0)

        patch = (patch - mean) / std
        if rng.random() < 0.5:
            patch, m = patch[:, ::-1], m[:, ::-1]
        if rng.random() < 0.5:
            patch, m = patch[::-1], m[::-1]
        r = int(rng.integers(4))
        patch, m = np.rot90(patch, r), np.rot90(m, r)
        xb[k] = np.ascontiguousarray(patch)
        yb[k] = np.ascontiguousarray(m)
    return torch.from_numpy(xb).permute(0, 3, 1, 2).contiguous(), torch.from_numpy(yb)


def normalize_image(img: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32) / 65535.0
    return ((x - mean) / std).transpose(2, 0, 1).copy()
