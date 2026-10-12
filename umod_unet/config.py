from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


CLASS_NAMES = [
    "background",
    "rod",
    "RBC/WBC",
    "yeast",
    "misc",
    "single EPC",
    "small EPC sheet",
    "large EPC sheet",
]

EXPECTED_TRAIN_COUNTS = {
    0: 142_904_148,
    1: 28_386,
    2: 249_958,
    3: 2_846,
    4: 23_917,
    5: 1_032_510,
    6: 291_821,
    7: 234_414,
}

CLASS_SAMPLE_WEIGHTS = {
    1: 2.2,  # rod
    2: 1.0,  # RBC/WBC
    3: 3.2,  # yeast
    4: 2.4,  # misc
    5: 0.8,  # single EPC
    6: 1.6,  # small EPC sheet
    7: 2.0,  # large EPC sheet
}

RARE_PASTE_CLASSES = (1, 3, 4)  # rod, yeast, misc


@dataclass
class Config:
    data_dir: Path
    run_dir: Path = Path("runs/resnet34_unet")
    seed: int = 0

    encoder: str = "resnet34"
    encoder_weights: str = "imagenet"
    n_classes: int = 8

    total_epochs: int = 40
    steps_per_epoch: int = 100
    batch_size: int = 8
    crop: int = 384
    p_class: float = 0.8
    p_paste: float = 0.45
    lr: float = 8e-4
    weight_decay: float = 1e-4
    eval_every: int = 2

    use_tta: bool = True
    amp: bool = True

    class_sample_weights: dict[int, float] = field(default_factory=lambda: dict(CLASS_SAMPLE_WEIGHTS))
