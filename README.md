# umod-unet

Pixel-level segmentation of urine microscopy images into 8 classes (background + 7 cell types), trained and evaluated on the open [UMOD dataset](https://rodare.hzdr.de/record/2563). A ResNet34 U-Net with an ImageNet-pretrained encoder, written in PyTorch.

This is a personal learning project. It is **not a medical device** and has no clinical validation. The dataset comes from patients with symptomatic UTI at one clinic, and nothing here says how well microscopy segmentation would work for diagnosis.

## Results

Test split, whole images (padded to a multiple of 32, with horizontal/vertical flip TTA), hard argmax. Dice and IoU are pooled over all pixels of all images. Intervals are 95% bootstrap over images (2000 resamples). Checkpoint: epoch 22, chosen on validation macro Dice. One training run.

| class | Dice [95% CI] | IoU [95% CI] | test images containing it |
|---|---|---|---|
| rod | 0.495 [0.434, 0.550] | 0.329 [0.277, 0.380] | 45 |
| RBC/WBC | 0.656 [0.583, 0.758] | 0.488 [0.412, 0.610] | 73 |
| yeast | 0.208 [0.067, 0.329] | 0.116 [0.035, 0.197] | 12 |
| misc | 0.237 [0.156, 0.323] | 0.134 [0.084, 0.193] | 58 |
| single EPC | 0.895 [0.867, 0.917] | 0.810 [0.765, 0.846] | 67 |
| small EPC sheet | 0.692 [0.517, 0.801] | 0.529 [0.348, 0.667] | 18 |
| large EPC sheet | 0.893 [0.508, 0.943] | 0.806 [0.340, 0.891] | 5 |
| **macro (classes 1-7)** | 0.582 [0.517, 0.616] | 0.459 [0.384, 0.496] | |

Foreground-vs-background Dice: 0.890 [0.829, 0.930]. Validation macro Dice (the split used to pick the checkpoint): 0.512 [0.449, 0.577].

Without TTA the test macro Dice is the same (0.582 [0.524, 0.614]). TTA made no meaningful difference.

### Compared with the from-scratch baseline

The baseline is a 7.76M-parameter U-Net trained from scratch (test macro Dice 0.449 [0.385, 0.481], validation 0.402 [0.339, 0.478]; code: [LINK TO THE BASELINE NOTEBOOK]). Paired bootstrap over the same 100 test images, new minus baseline, Dice (new model without TTA):

| class | difference [95% CI] |
|---|---|
| rod | -0.001 [-0.049, +0.041] |
| RBC/WBC | -0.011 [-0.046, +0.041] |
| yeast | +0.073 [-0.038, +0.187] |
| misc | +0.020 [-0.041, +0.087] |
| single EPC | +0.144 [+0.101, +0.187] |
| small EPC sheet | +0.339 [+0.204, +0.452] |
| large EPC sheet | +0.364 [+0.228, +0.618] |
| **macro (1-7)** | **+0.133 [+0.090, +0.177]** |

The improvement comes from the three epithelial-cell classes. The size confusion between single cells, small sheets and large sheets is much reduced. Rod, RBC/WBC, yeast and misc show no detectable change, and foreground-vs-background Dice is unchanged (0.890 vs 0.891): the model does not find cells better, it labels them better. Rods, yeast and misc remain the weak classes (for example, 63% of misc pixels are still predicted as background).

## What changed relative to the baseline

| | Baseline | This project |
|---|---|---|
| Model | From-scratch 4-level U-Net, 7.76M parameters | ResNet34 U-Net, ImageNet-pretrained encoder, 24.44M parameters |
| Loss | Cross-entropy + Dice over all 8 classes | Class-weighted cross-entropy (label smoothing 0.04) + Dice over classes 1-7 + Tversky on rod/yeast/misc |
| Crop sampling | 70% cell-centred, uniform over classes | 80% cell-centred, rare classes weighted up |
| Augmentation | Flips, 90-degree rotations | Same, plus brightness/contrast jitter and copy-paste of rod/yeast/misc cells from the training images |
| Inference | Whole image | Whole image padded to a multiple of 32, with flip TTA |
| Learning rate, batch | 1e-3 peak, batch 16 | 8e-4 peak, gradient clipping, batch 16 on a T4 (default 8) |

Many things changed at once, so these results do not show which change is responsible for the gain. No ablation has been run.

## Limitations

- **One training run.** The intervals cover which images were in the test set, not run-to-run variation from training. The validation score moved between about 0.43 and 0.51 across epochs near the end of training, so the picked checkpoint's validation score is optimistic.
- **The test split is not clean for this recipe.** The recipe was designed after looking at the baseline's validation and test results, and the training script prints the test table at the end of every run.
- **Rare classes are not reliably measured.** Large EPC sheets appear in 5 test images and 2 validation images, and yeast in 12 test images, so their intervals are wide.
- **Annotation gaps.** Some structures that look like cells are not annotated in some images, and the model may be penalised for finding them. The baseline model also produced many false positives on one image with no annotated cells.
- **Metrics are not those of the original paper.** This project reports pooled hard-argmax Dice and IoU on whole images.
- Pretrained weights are downloaded from the internet on the first run.

## Data

Download `ds1.zip` from the [RODARE record](https://rodare.hzdr.de/record/2563) (about 2.3 GB, DOI 10.14278/rodare.2563, CC BY 4.0) and unzip it so you have:

```
<data-dir>/train/img/*.tif
<data-dir>/train/mult_mask/*.tif
<data-dir>/validation/...
<data-dir>/test/...
```

The data is not included in this repository.

## Train

Needs an NVIDIA GPU (a Colab T4 is enough; training took about 30 minutes at roughly 45 seconds per epoch). A CPU will run but is impractically slow.

```bash
pip install -r requirements.txt
python -m umod_unet.train --data-dir /path/to/umod --run-dir runs/resnet34_unet
```

Flags:

- `--batch-size 16` to use on a T4 (the default is 8)
- `--epochs 40`
- `--seed 0`
- `--eval-only` reloads `best.pt` and prints the validation and test tables again
- `--no-tta` skips flip TTA at evaluation

Checkpoints (`last.pt`, `best.pt`) are written every epoch, so an interrupted run continues from where it stopped when you rerun the same command. A run prints both the validation and the test tables at the end.

### On Colab

1. Copy the `umod-unet` folder to Google Drive (for example `MyDrive/umod-unet`).
2. Choose **Runtime > Change runtime type > T4 GPU**.
3. Open `notebooks/train_colab.ipynb` and run the cells in order. It downloads the dataset to `MyDrive/umod_data/ds1.zip` if it is not already there, and writes checkpoints and results to `MyDrive/umod_runs/resnet34_unet`.

## Cite

Please cite the dataset paper: Liou, N., De, T., Urbanski, A., Chieng, C., Kong, Q., David, A. L., Khasriya, R., Yakimovich, A., Horsley, H. "A clinical microscopy dataset to develop a deep learning diagnostic test for urinary tract infection." *Scientific Data* 11, 155 (2024). The dataset is also citable as RODARE, DOI 10.14278/rodare.2563. This repository is independent of the dataset authors.

## License

Code: [CHOOSE A LICENSE, e.g. MIT, AND ADD A LICENSE FILE]. The dataset is CC BY 4.0.
