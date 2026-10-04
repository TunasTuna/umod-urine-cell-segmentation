# UMOD 8-class U-Net, epoch-40 checkpoint

Whole-image evaluation (1040x1392, no tiling). Hard argmax, Dice and IoU pooled over all pixels of all images.
95% intervals: bootstrap over images (2000 resamples). They do NOT include run-to-run training variation (single seed).
Test split was evaluated once for this model, after the checkpoint was chosen on validation.

## Test
| class | Dice [95% CI] | IoU [95% CI] | images containing it |
|---|---|---|---|
| rod | 0.482 [0.399, 0.565] | 0.318 [0.249, 0.393] | 45 |
| RBC/WBC | 0.666 [0.610, 0.728] | 0.499 [0.439, 0.573] | 73 |
| yeast | 0.109 [0.016, 0.234] | 0.058 [0.008, 0.133] | 12 |
| misc | 0.225 [0.130, 0.313] | 0.127 [0.069, 0.186] | 58 |
| single EPC | 0.747 [0.691, 0.800] | 0.596 [0.528, 0.667] | 67 |
| small EPC sheet | 0.379 [0.215, 0.528] | 0.233 [0.121, 0.359] | 18 |
| large EPC sheet | 0.538 [0.092, 0.618] | 0.368 [0.048, 0.447] | 5 |
| **macro (classes 1-7)** | 0.449 [0.385, 0.481] | 0.314 [0.268, 0.342] | |

Foreground-vs-background Dice: 0.891 [0.854, 0.916]

## Validation
| class | Dice [95% CI] | IoU [95% CI] | images containing it |
|---|---|---|---|
| rod | 0.497 [0.360, 0.608] | 0.331 [0.220, 0.436] | 57 |
| RBC/WBC | 0.708 [0.624, 0.766] | 0.548 [0.453, 0.620] | 70 |
| yeast | 0.175 [0.000, 0.340] | 0.096 [0.000, 0.205] | 8 |
| misc | 0.107 [0.046, 0.238] | 0.057 [0.024, 0.135] | 57 |
| single EPC | 0.781 [0.703, 0.842] | 0.641 [0.542, 0.727] | 59 |
| small EPC sheet | 0.381 [0.190, 0.588] | 0.236 [0.105, 0.417] | 15 |
| large EPC sheet | 0.166 [0.000, 0.380] | 0.090 [0.000, 0.234] | 2 |
| **macro (classes 1-7)** | 0.402 [0.339, 0.478] | 0.285 [0.238, 0.351] | |

Foreground-vs-background Dice: 0.812 [0.710, 0.889]

## Recipe

- U-Net, base 32 filters, 4 down/up levels, BatchNorm, 7.76M parameters, 3-channel input (16-bit TIFF scaled by 65535, per-channel mean/std from train), 8 classes (0 = background)
- 40 epochs x 100 steps, batch 16, 384x384 random crops, 70% centred on a random cell class, flips and 90-degree rotations
- AdamW, one-cycle LR schedule (max 1e-3), fp16 mixed precision, loss = cross-entropy + (1 - mean multi-class Dice)
- Checkpoint = best validation macro Dice (epoch 40). About 33 minutes on a Colab T4
- torch 2.11.0+cu130, python 3.13.15, GPU Tesla T4
- Data: UMOD (RODARE record 2563), the provided train/validation/test split, 100 images each
