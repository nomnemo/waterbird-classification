Waterbird Species Classification
================================

This repository contains the classification stage of a two‑step pipeline for monitoring colonial waterbirds from UAV (drone) imagery. It builds on prior work that trained object detectors on rookery islands along the Texas coast; here, we take detector crops of individual birds and train an image classifier to assign each crop to a waterbird species or mixed/other class.

The goal is to obtain robust, per‑bird species predictions on a long‑tailed dataset (many Laughing Gulls and Mixed Terns, few rare species) while avoiding data leakage from overlapping tiles of the same source image.


Project Overview
----------------

- **Input data**: Cropped bird images (`data/crops/...`) and a metadata JSON/CSVs that describe:
  - `crop_path`: path to the crop image
  - `species_name`, `species_id`
  - `source_image`: parent UAV image (used for grouped splitting)
  - bounding boxes (`bbox_original`, `bbox_expanded`)
- **Task**: Single‑image multi‑class classification (one species label per crop).
- **Model**: ImageNet‑pretrained **Swin‑Tiny** (`swin_tiny_patch4_window7_224`) from `timm`, fine‑tuned on the crop dataset.
- **Key challenges**:
  - **Long‑tailed distribution** (few samples for some species).
  - **Overlapping tiles / duplicates** of the same bird.
  - Need for **grouped splits** so the same parent image does not leak across train/val/test.


Data and Splits
---------------

The main processed dataset lives under `data/`:

- `data/metadata_balanced_t100.json`
  - List of crop records after combining very rare species into an `OTHER` class and capping some classes.
  - Each entry includes class labels, `crop_path`, `source_image`, and bounding‑box metadata.
- `data/crops/`
  - Directory tree of cropped bird images, organized by species (e.g., `data/crops/BLSK/...`).
- `data/split_train.csv`, `data/split_val.csv`, `data/split_test.csv`
  - CSV files defining the train/validation/test splits.
  - Generated from `metadata_balanced_t100.json` using grouped splitting on the parent image ID inferred from `source_image` (so all crops from the same UAV image go to the same split).

Splits are approximately 80/10/10 at the **parent image** level to prevent leakage from near‑duplicate crops of the same bird.


Model and Training
------------------

The classifier is trained with:

- **Backbone**: `swin_tiny_patch4_window7_224` via `timm.create_model`.
- **Input size**: 224×224 RGB.
- **Augmentations** (training):
  - Resize/pad to preserve context around the bird, random resized crops, flips, small rotations/shifts/scales.
  - Color jitter and coarse dropout.
  - ImageNet mean/std normalization.
- **Augmentations** (validation/test):
  - Deterministic resize/pad, center crop, ImageNet normalization.
- **Optimization**:
  - Optimizer: `AdamW`.
  - Learning rate and weight decay tuned over several values.
  - Cosine learning rate schedule.
  - Optional gradient accumulation.
- **Imbalance handling**:
  - Per‑class caps (`max_per_class`) to limit dominant classes.
  - `WeightedRandomSampler` that samples each crop with weight `1 / class_count`, making rare species more likely to appear in each batch.
  - Optional class‑weight tensor for use in loss functions.

Training metrics and artifacts (curves, confusion matrices, per‑class scores) are written under `runs_swin/<run_name>/`.


Repository Structure
--------------------

- `data/`
  - `metadata_balanced_t100.json` – balanced metadata with final class list.
  - `crops/` – cropped bird images arranged by species.
  - `split_train.csv`, `split_val.csv`, `split_test.csv` – grouped splits used for classification.
- `data_exploration/`
  - Plots and notebooks/scripts for exploring species distributions and class balancing (e.g., bar charts, cumulative plots).
- `scripts/`
  - `0_group_rare_bird_species.py`  
    Group very rare species into an `OTHER` / mixed class to reduce extreme sparsity.
  - `1_get_dataset_distribution.py`  
    Compute and plot dataset statistics (per‑class counts, cumulative distributions) before and after grouping.
  - `3_split_data.py`  
    Create 80/10/10 train/val/test splits from `metadata_balanced_t100.json` using grouped splitting by parent image (derived from `source_image`).
  - `4_dataloader.py`  
    Build PyTorch `DataLoader`s for train/val/test:
    - Reads `split_*.csv`.
    - Applies per‑class caps (`max_per_class`) if requested.
    - Builds label maps.
    - Attaches `WeightedRandomSampler` for balanced training batches.
  - `5_train_swin.py`  
    Main training script for Swin‑Tiny:
    - CLI for hyperparameters (epochs, LR, weight decay, max_per_class, batch size, etc.).
    - Trains the model, saves best checkpoint, exports curves, confusion matrices, and per‑class metrics.
  - `bird_dataset.py`  
    `BirdDataset` class: wraps CSV rows and, when indexed, loads the corresponding crop image, applies transforms, and returns `(image_tensor, label_tensor)`.
  - `image_transformer.py`  
    Centralized Albumentations pipelines:
    - `build_train_transforms`, `build_eval_transforms`, and `get_transforms(img_size, train)` for training vs evaluation augmentation.
- `runs_swin/`
  - Output directory containing results from past Swin runs (organized by run name).
- `swin_runs_summary.csv`
  - Summary table of previous Swin experiments (hyperparameters and key metrics).
- `compressed_swintiny_experiment_report.pdf`
  - Compact report of earlier experiments, including confusion matrices and per‑class metrics.


Environment Setup
-----------------

The project assumes Python 3.9+ and a GPU‑enabled PyTorch installation. The exact versions are listed in `requirements.txt`.

1. **Create and activate a virtual environment**

   Using `venv`:

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # on macOS/Linux
   # .venv\Scripts\activate         # on Windows (PowerShell/cmd)
   ```

   Or using `conda`:

   ```bash
   conda create -n waterbirds python=3.10
   conda activate waterbirds
   ```

2. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   Make sure your PyTorch install matches your CUDA version if you plan to train on GPU (see [https://pytorch.org](https://pytorch.org) for the recommended command).


Running the Pipeline
--------------------

The full end‑to‑end process (assuming the processed metadata and crops already exist) looks like:

1. **(Optional) Recompute dataset statistics and class grouping**

   - Use `scripts/0_group_rare_bird_species.py` and `scripts/1_get_dataset_distribution.py` if you need to regenerate the balanced metadata and plots from raw exports.

2. **Create grouped train/val/test splits**

   ```bash
   python scripts/3_split_data.py
   ```

   This reads `data/metadata_balanced_t100.json` and writes `data/split_train.csv`, `data/split_val.csv`, `data/split_test.csv`, making sure all crops from the same parent image are kept in the same split.

3. **Train Swin‑Tiny**

   ```bash
   python scripts/5_train_swin.py \
       --epochs 20 \
       --lr 3e-4 \
       --weight-decay 0.01 \
       --max-per-class 100 \
       --accum-steps 1
   ```

   Outputs go to `runs_swin/<auto_generated_run_name>/`:
   - `best_swin.pt` (checkpoint),
   - `curves.png`, `metrics.csv` (training curves),
   - `*_cm.png`, `*_ap_per_class.json`, `summary.json` (evaluation metrics).


Evaluation and Metrics
----------------------

The training script reports:

- **Accuracy** on train/val.
- **Macro‑F1** and **macro mAP (one‑vs‑rest)** on validation and test.
- **Confusion matrices** (per split, and a combined view) to inspect which species pairs are consistently confused (e.g., ROTE vs MTRN, GREG vs MEGRT).

Because of the long‑tailed nature of the dataset and mixed classes (e.g., Mixed Terns, Mixed Egrets), macro‑averaged metrics and per‑class AP/F1 are more informative than overall accuracy alone.


Next Steps / Extensions
-----------------------

Possible extensions on top of this codebase:

- Compare Swin‑Tiny to other backbones (e.g., ViT, ConvNeXt) using the same pipeline.
- Explore tern‑specific strategies (e.g., separate “Tern vs Non‑Tern” head, then Royal vs Sandwich on high‑confidence subsets).
- Try different class balancing strategies (label smoothing, focal loss, alternative samplers).
- Integrate this classifier downstream of the existing bird detector to evaluate end‑to‑end detection + classification performance on full UAV scenes.
