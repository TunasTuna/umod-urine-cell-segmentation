# umod-unet

8-class pixel segmentation of urine microscopy images from the [UMOD dataset](https://rodare.hzdr.de/record/2563) (RODARE 2563, CC BY 4.0).

This is a stronger local rewrite of the original Colab scratch U-Net notebook. It is **not a medical device**.

## What changed vs the notebook

The notebook trained a 7.8M from-scratch U-Net. After 40 epochs it reached about **0.39 validation / 0.44 test macro Dice**, with yeast/misc/rods mostly missed and EPC size classes mixed up.

This project keeps the same splits and the same metrics, and changes the recipe:

| Piece | Original notebook | This project |
|---|---|---|
| Encoder | Scratch 4-level U-Net | ImageNet-pretrained **ResNet34 U-Net** |
| Loss | CE + mean Dice over all 8 classes (including background) | Weighted CE + Dice on classes 1–7 + Tversky on rod/yeast/misc |
| Sampling | Uniform among 7 cell types | Rare classes oversampled |
| Extra aug | Flips + 90° rotations | Same, plus brightness/contrast and **copy-paste** of rare cells |
| Inference | Whole 1040×1392, no TTA | Pad to /32 + horizontal/vertical flip TTA |

Yeast still has only a handful of images. Macro Dice will stay noisy on that class; the goal is a better cell finder and a higher chance of catching thin rods / debris.

## Data

Unzip `ds1.zip` so you have:

```
<data-dir>/train/img/*.tif
<data-dir>/train/mult_mask/*.tif
<data-dir>/validation/...
<data-dir>/test/...
```

Download: `https://rodare.hzdr.de/record/2563` (~2.3 GB). If you already ran the Colab notebook, the zip is on Drive at `MyDrive/umod_data/`.

## Train

Needs a NVIDIA GPU (T4 16 GB is enough). CPU will run but is impractically slow.

```bash
cd umod-unet
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python -m umod_unet.train --data-dir PATH\to\umod --run-dir runs\resnet34_unet
```

Useful flags:

- `--batch-size 16` — try this on a T4; default is 8 for smaller cards
- `--epochs 40` — same budget as the notebook
- `--eval-only` — reload `best.pt` and reprint val/test tables
- `--no-tta` — faster eval, slightly weaker rare-class recall

Checkpoints (`last.pt`, `best.pt`) are written every epoch so you can interrupt and rerun the same command.

## Train on Colab

1. Copy the whole `umod-unet` folder to Google Drive, e.g. `MyDrive/umod-unet`.
2. In Colab: **Runtime → Change runtime type → T4 GPU**.
3. Open `notebooks/train_colab.ipynb` from Drive and run all cells.

The notebook reuses `MyDrive/umod_data/ds1_download_test.zip` if you already downloaded UMOD. Checkpoints go to `MyDrive/umod_runs/resnet34_unet`. If Colab drops, rerun from the top — it continues from `last.pt`.

## Metrics

Same protocol as the notebook: pooled hard-argmax Dice/IoU on full images, 95% bootstrap over images. The checkpoint is chosen on **validation** macro Dice. Look at the test table only after you freeze the recipe.

## Cite

Please cite the UMOD dataset paper (see the RODARE record). This repository is independent of the original authors.
