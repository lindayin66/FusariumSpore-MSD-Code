#!/usr/bin/env python3
"""Annotation consistency: compare two YOLO label sets via IoU>=0.5 matching."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment


def read_labels(lp):
    boxes = []
    if not lp.is_file(): return boxes
    for line in open(lp):
        p = line.split()
        if len(p)==5:
            boxes.append((int(p[0]), float(p[1]), float(p[2]),
                          float(p[3]), float(p[4])))
    return boxes


def iou(a, b):
    ax1, ay1 = a[1]-a[3]/2, a[2]-a[4]/2
    ax2, ay2 = a[1]+a[3]/2, a[2]+a[4]/2
    bx1, by1 = b[1]-b[3]/2, b[2]-b[4]/2
    bx2, by2 = b[1]+b[3]/2, b[2]+b[4]/2
    iw = max(0, min(ax2,bx2)-max(ax1,bx1))
    ih = max(0, min(ay2,by2)-max(ay1,by1))
    inter = iw*ih
    union = a[3]*a[4] + b[3]*b[4] - inter
    return inter/union if union > 0 else 0.0


def match(ref, cmp, thr=0.5):
    nr, nc = len(ref), len(cmp)
    if nr==0 and nc==0: return 0,0,0,[]
    if nr==0: return 0,nc,0,[]
    if nc==0: return 0,0,nr,[]
    cost = np.ones((nr,nc))
    for i,r in enumerate(ref):
        for j,c in enumerate(cmp):
            if r[0]==c[0]:
                v = iou(r,c)
                if v >= thr: cost[i,j] = 1 - v
    ri, ci = linear_sum_assignment(cost)
    tp, ious = 0, []
    for i,j in zip(ri,ci):
        if cost[i,j] < 1:
            tp += 1; ious.append(1-cost[i,j])
    return tp, nc-tp, nr-tp, ious


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--comparison", required=True)
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--output", default="results/annotation_consistency.csv")
    args = ap.parse_args()

    ref_dir = Path(args.reference).expanduser().resolve()
    cmp_dir = Path(args.comparison).expanduser().resolve()
    tp_tot = fp_tot = fn_tot = 0
    all_iou = []
    rows = []
    for rl in sorted(ref_dir.glob("*.txt")):
        cl = cmp_dir/rl.name
        rb = read_labels(rl); cb = read_labels(cl)
        tp, fp, fn, ious = match(rb, cb, args.iou_threshold)
        tp_tot += tp; fp_tot += fp; fn_tot += fn
        all_iou.extend(ious)
        rows.append({"image": rl.name, "n_ref": len(rb), "n_cmp": len(cb),
                     "TP": tp, "FP": fp, "FN": fn})

    prec = tp_tot/(tp_tot+fp_tot) if (tp_tot+fp_tot) else 0
    rec = tp_tot/(tp_tot+fn_tot) if (tp_tot+fn_tot) else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
    print(f"images: {len(rows)}")
    print(f"reference boxes: {tp_tot+fn_tot}")
    print(f"comparison boxes: {tp_tot+fp_tot}")
    print(f"TP={tp_tot} FP={fp_tot} FN={fn_tot}")
    print(f"Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f}")
    if all_iou:
        print(f"matched IoU mean={np.mean(all_iou):.4f} "
              f"median={np.median(all_iou):.4f}")

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
