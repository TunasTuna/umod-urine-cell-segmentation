from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import CLASS_NAMES
from .data import files_of, normalize_image
from .model import predict_full


PALETTE = ["#000000", "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628"]


def save_examples(
    model: torch.nn.Module,
    data_dir: Path,
    imgs: np.ndarray,
    masks: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    out_path: Path,
    use_amp: bool,
    tta: bool,
    seed: int = 123,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    cmap = ListedColormap(PALETTE)
    names = [p.name for p in files_of(data_dir, "test", "img")]
    pick = np.random.default_rng(seed).choice(len(imgs), 3, replace=False)
    fig, ax = plt.subplots(3, 3, figsize=(13, 10))
    model.eval()
    for r, i in enumerate(pick):
        pred = predict_full(model, normalize_image(imgs[i], mean, std), device, use_amp, tta)
        g = imgs[i][:, :, 0].astype(np.float32)
        lo, hi = np.percentile(g, 1), np.percentile(g, 99)
        ax[r, 0].imshow(np.clip((g - lo) / (hi - lo + 1e-9), 0, 1), cmap="gray")
        ax[r, 0].set_title("test image " + names[i] + " (contrast stretched)")
        ax[r, 1].imshow(masks[i], cmap=cmap, vmin=0, vmax=7, interpolation="nearest")
        ax[r, 1].set_title("annotation")
        ax[r, 2].imshow(pred, cmap=cmap, vmin=0, vmax=7, interpolation="nearest")
        ax[r, 2].set_title("prediction")
        for a in ax[r]:
            a.axis("off")
    fig.legend(handles=[Patch(color=PALETTE[c], label=CLASS_NAMES[c]) for c in range(8)], loc="lower center", ncol=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print("saved", out_path)
