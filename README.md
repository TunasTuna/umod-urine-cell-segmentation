# UMOD cell segmentation: an 8-class U-Net in PyTorch

Segmenting and classifying cells in unstained urine microscopy images, using the open UMOD dataset. This is a personal learning project. It is **not** a diagnostic tool and has no clinical validation.

## What this is

The UMOD dataset contains 300 microscopy images of urine from patients with symptomatic urinary tract infection, with every cell annotated at pixel level. This repo trains a small U-Net that labels each pixel as background or one of seven cell types, and evaluates it on whole images with confidence intervals.

The original authors released a Keras code base that segments cells vs. background (`casus/UMOD`). This project is a separate PyTorch implementation that predicts all seven cell classes directly, trained and evaluated on the dataset's provided split.

## Dataset

- **UMOD**, RODARE record 2563 (HZDR), CC BY 4.0. [ADD the exact DOI from the record page.]
- Paper: *A clinical microscopy dataset to develop a deep learning diagnostic test for urinary tract infection*, Scientific Data, 2024. [ADD the full citation from the paper's page.]
- 300 images (1040 x 1392, 16-bit RGB, almost grey), split by the authors into 100 train / 100 validation / 100 test. Each has a multi-class mask with values 0-7.
- Classes: 0 background, 1 rod, 2 RBC/WBC, 3 yeast, 4 misc, 5 single EPC, 6 small EPC sheet, 7 large EPC sheet.
- The data is **not** included in this repo. Download `ds1.zip` from the RODARE record.

## Results

Test split, whole images (no tiling), hard argmax, Dice and IoU pooled over all pixels. Intervals are 95% bootstrap over images (2000 resamples).

| class | Dice [95% CI] | IoU [95% CI] | test images containing it |
|---|---|---|---|
| rod | 0.482 [0.399, 0.565] | 0.318 [0.249, 0.393] | 45 |
| RBC/WBC | 0.666 [0.610, 0.728] | 0.499 [0.439, 0.573] | 73 |
| yeast | 0.109 [0.016, 0.234] | 0.058 [0.008, 0.133] | 12 |
| misc | 0.225 [0.130, 0.313] | 0.127 [0.069, 0.186] | 58 |
| single EPC | 0.747 [0.691, 0.800] | 0.596 [0.528, 0.667] | 67 |
| small EPC sheet | 0.379 [0.215, 0.528] | 0.233 [0.121, 0.359] | 18 |
| large EPC sheet | 0.538 [0.092, 0.618] | 0.368 [0.048, 0.447] | 5 |
| **macro (classes 1-7)** | 0.449 [0.385, 0.481] | 0.314 [0.268, 0.342] | |

Foreground-vs-background Dice: 0.891 [0.854, 0.916].

The checkpoint was chosen on the validation split (macro Dice 0.402 [0.339, 0.478]), and the test split was evaluated once for this model.

## Method

- U-Net, 4 down/up levels, base 32 filters, BatchNorm, 7.76M parameters, 3-channel input.
- Input: 16-bit TIFF scaled to [0, 1], normalised with per-channel mean/std from the training images.
- Training: 40 epochs of 100 steps, batch 16, 384 x 384 random crops (70% centred on a randomly chosen cell class), flips and 90-degree rotations.
- Loss: cross-entropy + (1 - mean multi-class Dice). AdamW, one-cycle learning-rate schedule (max 1e-3), fp16 mixed precision.
- About 33 minutes on a Colab T4. Checkpoints are saved every epoch to Google Drive so a disconnected session loses at most one epoch.
- Inference runs on the whole 1040 x 1392 image at once, with no tiling.

## Limitations

- **One training run, one seed.** The intervals cover only which test images were drawn, not run-to-run variation from training.
- **Rare classes are not reliably measured.** Large EPC sheets appear in 5 test images and yeast in 12, so their intervals are wide. Misc and yeast are weak.
- **The three EPC classes are confused with each other by size.** I haven't tested whether this reflects ambiguity in the labels.
- **Some images look unannotated.** On validation, the model labels an elongated structure in image 0030 and a large irregular structure in image 0056 as cells, while the masks leave them out. I can't tell from the images whether these are model errors or annotation gaps.
- **Image 0075 (no annotated cells) produces many false positives.** It has a grainy texture unlike the other images, and a Keras baseline also failed on it.
- **Metrics are not comparable with the original paper.** This project reports pooled hard-argmax Dice and IoU on whole images. The original repo's evaluation script averages per-image soft Dice for a binary model.
- The dataset comes from symptomatic patients at a single clinic. Nothing here says how well microscopy segmentation would work for diagnosing UTI.

## Running it on Colab

1. Download `ds1.zip` from the RODARE record and put it at `MyDrive/umod_data/ds1.zip`.
2. Open `[NOTEBOOK NAME].ipynb` in Colab, select a T4 GPU, and run the cells in order.
3. The notebook unzips the data, loads it into memory, trains in resumable chunks, and writes checkpoints and a results file to `MyDrive/umod_runs/`.

## Notes on the original Keras code

I also ran the original `casus/UMOD` code on a current Colab (Python 3.13, TensorFlow 2.20) to look at its binary baseline. It needed several compatibility changes, which I reported upstream: [LINK TO YOUR ISSUE].

## License

Code: [CHOOSE, e.g. MIT]. The dataset is CC BY 4.0, so cite the dataset paper if you reuse it.
