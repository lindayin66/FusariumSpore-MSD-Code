# Reproducibility checklist

This document maps every number in the manuscript to the command that produces it.

## Environment

| Component | Version |
|---|---|
| Python | 3.10.21 |
| PyTorch | 2.14.0+cu130 |
| Ultralytics | 8.4.153 |
| CUDA | 13.0 |
| GPU | NVIDIA RTX 6000D |

## Table 3 — dataset statistics

```bash
python scripts/validate_dataset.py \
    --dataset-root ../FusariumSpore-MSD_v1.0/split_7_2_1 \
    --output-dir   results/table3
```

Expected:

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

## Table 3 — detection mAP

```bash
python scripts/run_baseline.py detection \
    --weights    runs/yolov8n-split721/weights/best.pt \
    --model-type yolo \
    --data       configs/dataset_original.yaml \
    --name       yolov8n-test
# repeat for yolo11n and rtdetr-l (--model-type rtdetr)
```

## Table 4 / Figure 5 — counting

1. Sweep threshold on val (per model):

```bash
python scripts/run_baseline.py threshold \
    --weights      runs/yolov8n-split721/weights/best.pt \
    --model-type   yolo \
    --dataset-root ../FusariumSpore-MSD_v1.0/split_7_2_1 \
    --split        val \
    --output       results/threshold_v8.csv
```

Reference best conf (val MAE minimised):

| Model | best conf |
|---|---|
| YOLOv8n | 0.32 |
| YOLO11n | 0.35 |
| RT-DETR-L | 0.60 |

2. Count on test:

```bash
python scripts/run_baseline.py counting \
    --weights      runs/yolov8n-split721/weights/best.pt \
    --model-type   yolo \
    --dataset-root ../FusariumSpore-MSD_v1.0/split_7_2_1 \
    --split        test \
    --conf         0.32 \
    --output       results/count_v8.csv
```

Expected test MAE:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| YOLOv8n | 1.75 | 3.24 | 0.988 |
| YOLO11n | 1.64 | 2.91 | 0.990 |
| RT-DETR-L | 1.41 | 2.51 | 0.993 |

## Annotation consistency

```bash
python scripts/annotation_consistency.py \
    --reference  path/to/original_labels \
    --comparison path/to/reannotation_labels \
    --iou-threshold 0.5
```
