# Scientific Image Forgery Detection with Mask R-CNN

A Mask R-CNN pipeline that locates copy-move forgeries in scientific images, trained on the [Recod.ai/LUC Scientific Image Forgery Detection](https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection) data from Kaggle. Built in November and December 2025 as part of a four-person university team project. The team's presentation placed 2nd at a university AI project contest on December 12, 2025.

**Research report:** [Research Report 02 (PDF)](https://www.davidlonski.com/research/mask-r-cnn-kaggle/report.pdf), which covers the runs and a re-analysis of the training data. Project page: [davidlonski.com/research/mask-r-cnn-kaggle](https://www.davidlonski.com/research/mask-r-cnn-kaggle).

## Team

A four-person team. David Lonski built the Mask R-CNN training pipeline, ran it and diagnosed the results. The team's second stage, a vision-transformer model built on DINOv2, and a teammate's separate classifier were the teammates' work. The team registered for the Kaggle competition but did not submit; the deliverable was the contest presentation.

## What the runs found

- **Model:** torchvision's `maskrcnn_resnet50_fpn` with COCO weights, a two-class box predictor (background and forged region) and a new mask predictor: 43.9 million parameters.
- **Data:** 2,751 forged images with pixel masks, split 80/20 into 2,200 training and 551 validation images. Authentic images were not used.
- **Runs:**

  | Run | Dates (2025) | GPU | Loader | Epochs | Best validation loss |
  |---|---|---|---|---|---|
  | A | Nov 24–25 | A100 40 GB (Colab) | November | 10 | 0.2727 (epoch 4) |
  | B | Dec 1–3 | RTX 5000 Ada (lab) | December | 13 | lowest at epoch 6 |
  | C | Dec 3–5 | RTX 5000 Ada (lab) | December | 24 | about 0.538 (epoch 12) |

- **At the time:** in runs B and C, training loss kept falling while validation loss stalled or rose, read as overfitting made worse by the mix of image types. Predictions were mostly large boxes covering most of the image.
- **Re-analysis (October 2026):** re-running the notebooks' own mask loaders on every mask found a more basic cause. The mask files store instances first (`count × height × width`), but both loaders expected a single channel:
  - The November loader produced an empty target for 2,408 of 2,751 images (87.5%).
  - The December loader handled single-instance masks but turned all 882 multi-instance masks (32.1%) into vertical stripes unrelated to the forged regions.
  - The December mask check passed because it tested shape, values and boxes, not content.

  This does not rule out overfitting; separating the two needs a re-run with correct targets. No detection metric (IoU, mAP or the competition score) was computed.

## Repository layout

| Path | What it holds |
|---|---|
| `notebooks/train_maskrcnn_pytorch_1.ipynb` | The latest version of the training code (January 5, 2026), without training outputs. Start here. |
| `notebooks/train_maskrcnn_pytorch (2).ipynb` | The December version as run on the lab GPU (December 3, 2025), with outputs: configuration, mask check, run B's loss plot and predictions, and the start of run C |
| `notebooks/train_maskrcnn_pytorch.ipynb` | The November version (run A on Colab), with per-epoch losses and predictions |
| `notebooks/train_maskrcnn_pytorch (3).ipynb` | A Colab copy from December 11, 2025: data cells only, no training |
| `notebooks/Data_Set_Info.ipynb` | Dataset inventory: file counts, sizes and a sample viewer |
| `notebooks/study/train_shapes.ipynb` | The Matterport Mask R-CNN "train on shapes" tutorial (MIT license), run as study material |
| `team/DataLoader.ipynb` | The team's shared data-exploration notebook: competition download, mask visualization and the planned vision-transformer approach |
| `team/CNN02.ipynb` | A teammate's transfer-learning classifier |
| `analysis/make_figures.py` | The re-analysis for Report 02: mask survey, loader re-runs, split reproduction and figures |
| `analysis/data/` | Its outputs: `masks.csv`, `loader_summary.csv`, `run_a_epochs.csv` |

Notebook file names are kept as they were saved, so they match the report's evidence list.

## Setup

The notebooks were run on Google Colab and on a Windows workstation with an NVIDIA RTX 5000 Ada (CUDA 12.x). Install:

```bash
pip install torch torchvision numpy opencv-python pillow matplotlib kaggle
```

On Windows, install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) first.

### Data

The dataset is not included; download it from Kaggle under the competition's rules. With a Kaggle API token (`kaggle.json`) in place:

```bash
kaggle competitions download -c recodai-luc-scientific-image-forgery-detection
```

The notebooks expect `train_images/forged/` and `train_masks/` inside the dataset folder. Set `DATA_DIR` in the dataset cell (it defaults to a Google Drive folder). On Colab, the notebooks mount Google Drive, ask you to upload `kaggle.json`, and download the data.

### Re-analysis

```bash
python analysis/make_figures.py <dataset_dir> [notebook_dir]
```

`notebook_dir` defaults to `notebooks/`. It writes the CSV files to `analysis/data/` and the figures to `analysis/figures/`.

## Known issues

These are left as they were, so the notebooks match the report:

- Both mask loaders read the masks as single-channel arrays (see the re-analysis above). A fix is to build one target per stored instance channel and check targets against the raw masks.
- Validation losses are computed with the model in training mode, which is how torchvision returns losses; there is no detection or segmentation metric.
- The train/validation split shuffles the file list in the order the operating system returns it, so it can differ between machines.

## Author

David Lonski, with three teammates.
