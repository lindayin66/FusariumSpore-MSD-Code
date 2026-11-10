# FusariumSpore-MSD-Code

Minimal reproducibility code for **FusariumSpore-MSD v1.0**, a multi-scale RGB bright-field microscopy image dataset of *Fusarium* spores (10× and 40× objectives) with mixed-species background interference, for object detection and per-image spore counting.

## Dataset

- **Dataset**: FusariumSpore-MSD v1.0
- **Zenodo DOI**: https://doi.org/10.5281/zenodo.23827073
- **Code repository**: https://github.com/lindayin66/FusariumSpore-MSD-Code
- **Classes**: `0 = FG` (*F. graminearum*), `1 = FPG` (*F. pseudograminearum*), `2 = FI` (*F. incarnatum*)
- **Original microscopy images**: 1,731 RGB JPEG (2592 × 1944)
- **Annotated instances**: 46,244 YOLO-format bounding boxes
- **Predefined splits (image-level, stratified by group and magnification)**:
  - Train: 1,211 images / 31,782 instances
  - Val: 347 images / 9,168 instances
  - Test: 173 images / 5,294 instances
- **Objective magnifications**: 10× (950 images) and 40× (781 images)
- **Sample groups**: FG (526), FPG (499), FI (453), FGFO mixed (253)

The predefined splits are intended for baseline reproduction only. This repository contains code and configuration files only. **Trained model weights and dataset images are not distributed.**

## Repository structure

```text
FusariumSpore-MSD-Code/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── configs/
│   └── dataset_original.yaml
├── scripts/
│   ├── validate_dataset.py
│   ├── run_baseline.py
│   └── annotation_consistency.py
└── docs/
    └── REPRODUCIBILITY.md
```

## Installation

Reference environment used in the manuscript:

```text
Python       3.10.21
PyTorch      2.14.0+cu130
Ultralytics  8.4.153
CUDA         13.0
GPU          NVIDIA RTX 6000D
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the PyTorch build matching your local CUDA/CPU setup separately.

## Dataset location

A convenient layout is to place the dataset as a sibling of this repository:

```text
workspace/
├── FusariumSpore-MSD-Code/
└── FusariumSpore-MSD_v1.0/
    └── split_7_2_1/
        ├── train/  (images/ labels/)
        ├── val/    (images/ labels/)
        └── test/   (images/ labels/)
```

The provided YAML assumes the dataset sits next to this repository. Edit the `path:` entry in `configs/dataset_original.yaml` if your local location differs.

## 1. Validate the dataset and reproduce Table 3 statistics

```bash
python scripts/validate_dataset.py \
    --dataset-root  ../FusariumSpore-MSD_v1.0/split_7_2_1 \
    --output-dir    results/table3
```

The script checks:

- image–label pairing under train/val/test;
- YOLO field count, class_id validity, and coordinate range;
- expected split sizes and instance counts;
- JPEG readability;
- 2592 × 1944 image size.

It also computes per-split mean instances per image, median normalized bbox area, median bbox aspect ratio, median brightness (grayscale mean), median contrast (grayscale std), and median sharpness (variance of Laplacian).

Final public-release Table 3 values:

| metric | train | val | test |
|---|---|---|---|
| images | 1211 | 347 | 173 |
| instances | 31782 | 9168 | 5294 |
| mean instances/image | 26.24 | 26.42 | 30.60 |
| median normalized bbox area | 0.0009 | 0.0010 | 0.0009 |
| median bbox aspect ratio | 0.773 | 0.831 | 0.849 |
| median brightness | 114.27 | 114.24 | 114.52 |
| median contrast | 5.71 | 5.84 | 5.70 |
| median sharpness | 28.26 | 27.94 | 27.98 |

## 2. Baseline training and evaluation

All baseline functionality is integrated in `scripts/run_baseline.py`.

Reference training settings:

```text
imgsz        = 640
batch        = 8
epochs       = 300
patience     = 50
seed         = 42
workers      = 4
```

YOLOv8n and YOLO11n use `deterministic=True, AMP=False`;
RT-DETR-L uses `deterministic=False, AMP=True` (deformable attention has no deterministic CUDA backward).

### Train

YOLOv8n:

```bash
python scripts/run_baseline.py train \
    --weights     yolov8n.pt \
    --model-type  yolo \
    --data        configs/dataset_original.yaml \
    --name        yolov8n-split721
```

YOLO11n:

```bash
python scripts/run_baseline.py train \
    --weights     yolo11n.pt \
    --model-type  yolo \
    --data        configs/dataset_original.yaml \
    --name        yolo11n-split721
```

RT-DETR-L:

```bash
python scripts/run_baseline.py train \
    --weights     rtdetr-l.pt \
    --model-type  rtdetr \
    --data        configs/dataset_original.yaml \
    --name        rtdetr_l-split721
```

### Evaluate detection

```bash
python scripts/run_baseline.py detection \
    --weights     runs/yolov8n-split721/weights/best.pt \
    --model-type  yolo \
    --data        configs/dataset_original.yaml \
    --name        yolov8n-test \
    --output      results/detection_v8.json
```

Reports Precision, Recall, mAP@0.5, and mAP@0.5:0.95 on the predefined test split.

### Select the counting threshold on validation only

```bash
python scripts/run_baseline.py threshold \
    --weights      runs/yolov8n-split721/weights/best.pt \
    --model-type   yolo \
    --dataset-root ../FusariumSpore-MSD_v1.0/split_7_2_1 \
    --split        val \
    --output       results/threshold_v8.csv
```

Sweeps confidence from 0.10 to 0.90 (step 0.01) and picks the threshold that minimizes validation MAE. Reference thresholds:

| Model | best conf |
|---|---|
| YOLOv8n | 0.32 |
| YOLO11n | 0.35 |
| RT-DETR-L | 0.60 |

### Evaluate counting

```bash
python scripts/run_baseline.py counting \
    --weights      runs/yolov8n-split721/weights/best.pt \
    --model-type   yolo \
    --dataset-root ../FusariumSpore-MSD_v1.0/split_7_2_1 \
    --split        test \
    --conf         0.32 \
    --output       results/count_v8.csv
```

Reports MAE, RMSE, R², mean signed error, fraction of images within ±2 spores, and total predicted vs. ground-truth counts.

## 3. Annotation consistency calculation

For two YOLO label directories describing the same images:

```bash
python scripts/annotation_consistency.py \
    --reference   path/to/original_labels \
    --comparison  path/to/reannotation_labels \
    --iou-threshold 0.5 \
    --output      results/annotation_consistency.csv
```

Performs one-to-one Hungarian matching (same class + IoU ≥ 0.5) and reports Precision, Recall, instance-level F1, and matched-box IoU statistics. Manual review is a separate human step and is not automated here.

## Citation

If you use the dataset, cite:

```text
FusariumSpore-MSD v1.0. Zenodo.
https://doi.org/10.5281/zenodo.23827073
```

Cite the Data Descriptor paper once it is published. See `CITATION.cff` for this code repository.

## Licenses

- Code in this repository: MIT License.
- FusariumSpore-MSD dataset: CC BY 4.0.

## Notes

- This repository does not redistribute dataset images.
- Trained baseline weights are not distributed.
- Do not commit local datasets, model checkpoints, or experiment output directories.
