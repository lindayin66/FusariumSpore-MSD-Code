#!/usr/bin/env python3
"""Validate FusariumSpore-MSD and reproduce Table 3 statistics."""
import argparse, csv
from pathlib import Path
import cv2, numpy as np

SPLITS = ("train", "val", "test")
EXPECTED_CLASSES = {0, 1, 2}
EXPECTED_WH = (2592, 1944)


def validate_label(lp):
    errors, areas, aspects, n = [], [], [], 0
    for ln, raw in enumerate(open(lp), 1):
        line = raw.strip()
        if not line: continue
        p = line.split()
        if len(p) != 5:
            errors.append(f"{lp.name}:{ln} needs 5 fields"); continue
        try:
            cid = int(p[0]); xc, yc, w, h = map(float, p[1:])
        except ValueError:
            errors.append(f"{lp.name}:{ln} non-numeric"); continue
        if cid not in EXPECTED_CLASSES:
            errors.append(f"{lp.name}:{ln} bad class_id={cid}")
        if not (0 <= xc <= 1 and 0 <= yc <= 1):
            errors.append(f"{lp.name}:{ln} center out of [0,1]")
        if not (0 < w <= 1 and 0 < h <= 1):
            errors.append(f"{lp.name}:{ln} w/h out of (0,1]"); continue
        n += 1; areas.append(w*h); aspects.append(w/h)
    return n, errors, areas, aspects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--output-dir", default="results/table3")
    args = ap.parse_args()
    root = Path(args.dataset_root).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    all_reports, all_errors = [], []
    for sp in SPLITS:
        img_dir = root/sp/"images"; lbl_dir = root/sp/"labels"
        rep = {"split": sp, "n_images": 0, "n_instances": 0,
               "mean_per_image": None, "median_area": None,
               "median_aspect": None, "median_brightness": None,
               "median_contrast": None, "median_sharpness": None}
        errs = []
        cnt, areas, aspects, br, co, sh = [], [], [], [], [], []
        imgs = sorted(p for p in img_dir.iterdir()
                      if p.suffix.lower() in (".jpg",".jpeg",".png"))
        rep["n_images"] = len(imgs)
        for ip in imgs:
            lp = lbl_dir/(ip.stem+".txt")
            if not lp.is_file():
                errs.append(f"missing label: {ip.name}"); continue
            img = cv2.imread(str(ip))
            if img is None:
                errs.append(f"unable to read: {ip.name}"); continue
            h, w = img.shape[:2]
            if (w, h) != EXPECTED_WH:
                errs.append(f"{ip.name} size {w}x{h}")
            n, e, ar, asp = validate_label(lp)
            errs.extend(e); cnt.append(n); areas.extend(ar); aspects.extend(asp)
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            br.append(float(g.mean())); co.append(float(g.std()))
            sh.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
        rep["n_instances"] = int(sum(cnt))
        if cnt: rep["mean_per_image"] = float(np.mean(cnt))
        if areas:
            rep["median_area"] = float(np.median(areas))
            rep["median_aspect"] = float(np.median(aspects))
        if br:
            rep["median_brightness"] = float(np.median(br))
            rep["median_contrast"] = float(np.median(co))
            rep["median_sharpness"] = float(np.median(sh))
        all_reports.append(rep); all_errors.extend(errs)

    with open(out/"table3.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_reports[0].keys()))
        w.writeheader(); [w.writerow(r) for r in all_reports]
    with open(out/"validation_errors.txt", "w") as f:
        f.write("\n".join(all_errors) if all_errors else "No errors.\n")

    print(f"\n| metric | train | val | test |")
    print("|" + "---|"*4)
    for key in ["n_images","n_instances","mean_per_image","median_area",
                "median_aspect","median_brightness","median_contrast","median_sharpness"]:
        vals = [r[key] for r in all_reports]
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in vals]
        print(f"| {key} | " + " | ".join(cells) + " |")
    print(f"\n[OK] {len(all_errors)} issues -> {out/'validation_errors.txt'}")


if __name__ == "__main__":
    main()
