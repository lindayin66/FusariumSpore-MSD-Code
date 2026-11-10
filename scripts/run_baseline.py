#!/usr/bin/env python3
"""
run_baseline.py — unified entry point for training / detection eval /
threshold search / counting evaluation on FusariumSpore-MSD.

Subcommands
-----------
train       Train a YOLO or RT-DETR baseline.
detection   Report Precision / Recall / mAP@0.5 / mAP@0.5:0.95 on test.
threshold   Sweep confidence on val, pick threshold minimising MAE.
counting    Run counting with a fixed confidence, report MAE / RMSE / R².
"""

import argparse, csv
from pathlib import Path
import numpy as np, pandas as pd
from ultralytics import YOLO, RTDETR

CLASSES = {0: "FG", 1: "FPG", 2: "FI"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load_model(weights: str, model_type: str = "yolo"):
    return RTDETR(weights) if model_type == "rtdetr" else YOLO(weights)


def gt_counts(label_dir: Path):
    out = {}
    for lp in sorted(label_dir.glob("*.txt")):
        out[lp.stem + ".jpg"] = sum(1 for l in open(lp) if l.strip())
    return out


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def cmd_train(args):
    model = load_model(args.weights, args.model_type)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        deterministic=args.deterministic,
        amp=args.amp,
        project=args.project,
        name=args.name,
        plots=True,
    )


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------
def cmd_detection(args):
    model = load_model(args.weights, args.model_type)
    m = model.val(data=args.data, split="test", imgsz=640,
                  batch=8, device=args.device, plots=True,
                  project=args.project, name=args.name)
    print(m.results_dict)
    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(m.results_dict).to_csv(out)
    print(f"[OK] wrote {out}")


# ---------------------------------------------------------------------------
# threshold
# ---------------------------------------------------------------------------
def cmd_threshold(args):
    model = load_model(args.weights, args.model_type)
    root = Path(args.dataset_root).expanduser().resolve()
    gt = gt_counts(root / args.split / "labels")
    img_dir = root / args.split / "images"

    rows = []
    best_c, best_mae = None, float("inf")
    for c in np.arange(0.10, 0.91, 0.01):
        r = model.predict(source=str(img_dir), conf=float(c), imgsz=640,
                          device=args.device, verbose=False)
        pred = {Path(x.path).name: len(x.boxes) for x in r}
        mae = float(np.mean([abs(pred.get(n, 0) - v) for n, v in gt.items()]))
        rows.append({"conf": round(float(c), 2), "MAE": mae})
        if mae < best_mae:
            best_mae, best_c = mae, float(c)

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[best] conf={best_c:.2f}  MAE={best_mae:.3f}")
    print(f"[OK] wrote {out}")


# ---------------------------------------------------------------------------
# counting
# ---------------------------------------------------------------------------
def cmd_counting(args):
    model = load_model(args.weights, args.model_type)
    root = Path(args.dataset_root).expanduser().resolve()
    gt = gt_counts(root / args.split / "labels")
    img_dir = root / args.split / "images"

    r = model.predict(source=str(img_dir), conf=args.conf, imgsz=640,
                      device=args.device, verbose=False)
    pred = {Path(x.path).name: len(x.boxes) for x in r}
    names = sorted(gt.keys())
    yt = np.array([gt[n] for n in names], dtype=float)
    yp = np.array([pred.get(n, 0) for n in names], dtype=float)
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1 - float(np.sum(err ** 2)) / denom if denom > 0 else 0.0
    within2 = float(np.mean(np.abs(err) <= 2))

    print(f"n={len(names)}")
    print(f"GT total={int(yt.sum())}  Pred total={int(yp.sum())}")
    print(f"MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")
    print(f"mean signed error={float(err.mean()):.3f}")
    print(f"|err|<=2 images={within2*100:.1f}%")

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"image": names, "GT": yt.astype(int),
                  "Pred": yp.astype(int), "error": err.astype(int)}
                 ).to_csv(out, index=False)
    print(f"[OK] wrote {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    t = sub.add_parser("train")
    t.add_argument("--weights", required=True, help="e.g. yolov8n.pt / yolo11n.pt / rtdetr-l.pt")
    t.add_argument("--model-type", choices=["yolo", "rtdetr"], default="yolo")
    t.add_argument("--data", required=True)
    t.add_argument("--name", required=True)
    t.add_argument("--project", default="runs")
    t.add_argument("--epochs", type=int, default=300)
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--device", type=int, default=0)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--patience", type=int, default=50)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--deterministic", action="store_true")
    t.add_argument("--amp", action="store_true")
    t.set_defaults(func=cmd_train)

    # detection
    d = sub.add_parser("detection")
    d.add_argument("--weights", required=True)
    d.add_argument("--model-type", choices=["yolo", "rtdetr"], default="yolo")
    d.add_argument("--data", required=True)
    d.add_argument("--name", default="eval")
    d.add_argument("--project", default="runs/eval")
    d.add_argument("--device", type=int, default=0)
    d.add_argument("--output", default="results/detection.json")
    d.set_defaults(func=cmd_detection)

    # threshold
    h = sub.add_parser("threshold")
    h.add_argument("--weights", required=True)
    h.add_argument("--model-type", choices=["yolo", "rtdetr"], default="yolo")
    h.add_argument("--dataset-root", required=True)
    h.add_argument("--split", default="val")
    h.add_argument("--device", type=int, default=0)
    h.add_argument("--output", default="results/threshold_search.csv")
    h.set_defaults(func=cmd_threshold)

    # counting
    c = sub.add_parser("counting")
    c.add_argument("--weights", required=True)
    c.add_argument("--model-type", choices=["yolo", "rtdetr"], default="yolo")
    c.add_argument("--dataset-root", required=True)
    c.add_argument("--split", default="test")
    c.add_argument("--conf", type=float, required=True)
    c.add_argument("--device", type=int, default=0)
    c.add_argument("--output", default="results/counting.csv")
    c.set_defaults(func=cmd_counting)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
