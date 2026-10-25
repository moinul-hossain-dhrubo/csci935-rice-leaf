"""Stage 1: dataset inspection.

Checks that labels are read correctly, counts images per disease x background,
measures image sizes and simple quality statistics, looks for duplicate
images, and saves contact sheets for viewing.

Outputs
  dataset-notes/01_inspect_log.txt     everything printed below
  dataset-notes/counts.csv             disease x background counts
  dataset-notes/size_summary.csv       width/height ranges per background
  dataset-notes/image_stats.csv        one row per image
  dataset-notes/similar_pairs.csv      most similar image pairs per background
  Dhan-Shomadhan/_interim/inspect/     contact sheets (interim, for viewing only)

Run from the folder that contains Dhan-Shomadhan/:
  python 01_inspect.py
"""
import csv
import hashlib
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import (BACKGROUNDS, CLASSES, DATASET_DIR, INTERIM_DIR,
                    list_images, read_label_csv)

NOTES_DIR = Path("dataset-notes")
SHEET_DIR = INTERIM_DIR / "inspect"
STAT_SIDE = 512          # longest side used for quality statistics
THUMB = 160              # contact-sheet thumbnail size in pixels
COPY_CORR = 0.98         # thumbnail correlation treated as "the same picture"
TOP_PAIRS = 12           # most similar pairs listed per background, for viewing

_log_lines = []


def log(msg=""):
    print(msg)
    _log_lines.append(str(msg))


def resize_longest(img, side):
    h, w = img.shape[:2]
    s = side / max(h, w)
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                      interpolation=cv2.INTER_AREA)


def key(r):
    return r["file"] + r["background"] + r["disease"]


def thumb_vector(gray):
    """24 x 48 grey thumbnail (upright), mean removed, length 1, for comparing images."""
    if gray.shape[1] > gray.shape[0]:
        gray = gray.T
    v = cv2.resize(gray, (24, 48), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-9)


def thumbnail(img, label):
    t = resize_longest(img, THUMB)
    canvas = np.full((THUMB + 18, THUMB, 3), 255, np.uint8)
    y0 = (THUMB - t.shape[0]) // 2
    x0 = (THUMB - t.shape[1]) // 2
    canvas[y0:y0 + t.shape[0], x0:x0 + t.shape[1]] = t
    cv2.putText(canvas, label[:22], (2, THUMB + 13), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def save_sheet(thumbs, path, cols=6):
    if not thumbs:
        return
    rows = []
    for i in range(0, len(thumbs), cols):
        row = thumbs[i:i + cols]
        row += [np.full_like(thumbs[0], 255)] * (cols - len(row))
        rows.append(np.hstack(row))
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def main():
    t0 = time.time()
    NOTES_DIR.mkdir(exist_ok=True)
    SHEET_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Folder structure and labels -------------------------------------
    log("== 1. Folder structure ==")
    for bg_dir in sorted(p for p in DATASET_DIR.iterdir() if p.is_dir()):
        if bg_dir.name.startswith("_"):
            continue
        for dz_dir in sorted(p for p in bg_dir.iterdir() if p.is_dir()):
            n = sum(1 for f in dz_dir.iterdir() if f.is_file())
            log(f"  {bg_dir.name!r:24} / {dz_dir.name!r:18} {n} files")
    records = list_images()
    log(f"Images found: {len(records)}")

    log("\n== 2. Label check: folder vs CSV vs file-name prefix ==")
    csv_labels = read_label_csv()
    problems = 0
    if csv_labels is None:
        log("  No CSV found in the dataset folder; CSV check skipped.")
    else:
        log(f"  CSV rows: {len(csv_labels)}")
        names = Counter(r["file"] for r in records)
        repeated = [n for n, c in names.items() if c > 1]
        log(f"  File names used more than once across folders: {len(repeated)} {repeated[:10]}")
        missing_in_csv = [r["file"] for r in records if r["file"] not in csv_labels]
        missing_on_disk = sorted(set(csv_labels) - set(names))
        log(f"  Images not in CSV: {len(missing_in_csv)} {missing_in_csv[:10]}")
        log(f"  CSV rows with no image: {len(missing_on_disk)} {missing_on_disk[:10]}")
        for r in records:
            lab = csv_labels.get(r["file"])
            if lab and lab != (r["disease"], r["background"]):
                problems += 1
                log(f"  MISMATCH {r['file']}: folder={r['disease']}/{r['background']} csv={lab}")
        log(f"  Folder/CSV label mismatches: {problems}")

    prefix_by_group = defaultdict(Counter)
    for r in records:
        prefix = "_".join(Path(r["file"]).stem.split("_")[:-1]) or Path(r["file"]).stem
        r["prefix"] = prefix
        prefix_by_group[(r["background"], r["disease"])][prefix] += 1
    log("  File-name prefixes per group:")
    odd_prefix = []
    for gk in sorted(prefix_by_group):
        log(f"    {gk[0]:5} {gk[1]:13} {dict(prefix_by_group[gk])}")
        main_prefix = prefix_by_group[gk].most_common(1)[0][0]
        odd_prefix += [r["file"] for r in records
                       if (r["background"], r["disease"]) == gk and r["prefix"] != main_prefix]
    log(f"  Images whose prefix differs from the rest of their folder: {len(odd_prefix)} {odd_prefix[:10]}")

    # 3. Counts ------------------------------------------------------------
    log("\n== 3. Counts per disease x background ==")
    counts = Counter((r["disease"], r["background"]) for r in records)
    rows = []
    log(f"  {'disease':14}" + "".join(f"{b:>8}" for b in BACKGROUNDS) + f"{'total':>8}")
    for d in CLASSES:
        vals = [counts[(d, b)] for b in BACKGROUNDS]
        rows.append([d] + vals + [sum(vals)])
        log(f"  {d:14}" + "".join(f"{v:>8}" for v in vals) + f"{sum(vals):>8}")
    totals = [sum(counts[(d, b)] for d in CLASSES) for b in BACKGROUNDS]
    rows.append(["Total"] + totals + [sum(totals)])
    log(f"  {'Total':14}" + "".join(f"{v:>8}" for v in totals) + f"{sum(totals):>8}")
    write_csv(NOTES_DIR / "counts.csv", ["disease"] + BACKGROUNDS + ["total"], rows)

    # 4. Per-image statistics ----------------------------------------------
    log("\n== 4. Reading every image (sizes, quality statistics, hashes) ==")
    unreadable = []
    thumbs = {}
    for i, r in enumerate(records):
        raw = r["path"].read_bytes()
        r["md5"] = hashlib.md5(raw).hexdigest()
        r["file_kb"] = round(len(raw) / 1024)
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            unreadable.append(r["file"])
            continue
        r["height"], r["width"] = img.shape[:2]
        small = resize_longest(img, STAT_SIDE)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        r["brightness"] = round(float(v.mean()), 1)               # 0 to 255
        r["contrast"] = round(float(gray.std()), 1)
        r["sharpness"] = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1)
        r["white_frac"] = round(float(((s < 40) & (v > 180)).mean()), 3)
        r["green_frac"] = round(float(((h >= 35) & (h <= 85) & (s > 60) & (v > 40)).mean()), 3)
        r["thumbvec"] = thumb_vector(gray)
        thumbs[key(r)] = thumbnail(small, r["file"])
        if (i + 1) % 200 == 0:
            log(f"  {i + 1}/{len(records)} read ({time.time() - t0:.0f} s)")
    log(f"  Unreadable images: {len(unreadable)} {unreadable[:10]}")
    ok = [r for r in records if "width" in r]

    stat_cols = ["width", "height", "file_kb", "brightness", "contrast",
                 "sharpness", "white_frac", "green_frac"]
    write_csv(NOTES_DIR / "image_stats.csv",
              ["file", "background", "disease"] + stat_cols + ["md5"],
              [[r["file"], r["background"], r["disease"]] + [r[c] for c in stat_cols] + [r["md5"]]
               for r in ok])

    log("\n== 5. Image sizes per background ==")
    size_rows = []
    for b in BACKGROUNDS:
        grp = [r for r in ok if r["background"] == b]
        if not grp:
            continue
        sizes = Counter((r["width"], r["height"]) for r in grp)
        W = np.array([r["width"] for r in grp]); H = np.array([r["height"] for r in grp])
        portrait = int((H > W).sum())
        log(f"  {b}: n={len(grp)} width {W.min()} to {W.max()} (median {int(np.median(W))}), "
            f"height {H.min()} to {H.max()} (median {int(np.median(H))}), "
            f"portrait {portrait}, landscape {len(grp) - portrait - int((H == W).sum())}, "
            f"square {int((H == W).sum())}")
        log(f"     most common sizes (w x h): {sizes.most_common(5)}")
        size_rows.append([b, len(grp), W.min(), W.max(), int(np.median(W)),
                          H.min(), H.max(), int(np.median(H)), portrait, len(sizes)])
    write_csv(NOTES_DIR / "size_summary.csv",
              ["background", "n", "w_min", "w_max", "w_median", "h_min", "h_max",
               "h_median", "n_portrait", "n_distinct_sizes"], size_rows)

    log("\n== 6. Quality statistics (median [min, max]) per background ==")
    for b in BACKGROUNDS:
        grp = [r for r in ok if r["background"] == b]
        parts = []
        for c in ["brightness", "contrast", "sharpness", "white_frac", "green_frac"]:
            a = np.array([r[c] for r in grp])
            parts.append(f"{c} {np.median(a):.3g} [{a.min():.3g}, {a.max():.3g}]")
        log(f"  {b}: " + "; ".join(parts))
    log("  (brightness = mean HSV value 0-255; sharpness = Laplacian variance at "
        f"{STAT_SIDE}px, higher = sharper; white_frac / green_frac = share of near-white / green pixels)")

    # 7. Duplicates and near-duplicates -----------------------------------
    # Exact copies: same file bytes (MD5). Near copies: each image shrunk to a
    # 24 x 48 grey thumbnail (landscape images turned upright), mean removed and
    # scaled to length 1; the dot product of two thumbnails is their
    # correlation (1.0 = same picture). A re-saved or resized copy scores about
    # 0.99 or more. Lower scores only mean "similar layout", which is common on
    # the white background, so they are listed for LOOKING at, not counted.
    log("\n== 7. Duplicate and most similar images ==")
    by_md5 = defaultdict(list)
    for r in ok:
        by_md5[r["md5"]].append(r["file"])
    exact = [g for g in by_md5.values() if len(g) > 1]
    log(f"  Exact duplicate groups (same bytes): {len(exact)} {exact[:5]}")
    V = np.array([r["thumbvec"] for r in ok])
    C = V @ V.T
    iu = np.triu_indices(len(ok), 1)
    corr = C[iu]
    log(f"  Pairs with thumbnail correlation >= {COPY_CORR}: {int((corr >= COPY_CORR).sum())}")
    log(f"  Highest correlation of any pair: {corr.max():.3f}")
    pair_rows = []
    for b in BACKGROUNDS:
        idx = [k for k, r in enumerate(ok) if r["background"] == b]
        sub = C[np.ix_(idx, idx)]
        su = np.triu_indices(len(idx), 1)
        order = np.argsort(-sub[su])[:TOP_PAIRS]
        log(f"  {TOP_PAIRS} most similar {b} pairs (correlation, files, labels):")
        sheet = []
        for k in order:
            ra, rb = ok[idx[su[0][k]]], ok[idx[su[1][k]]]
            c = float(sub[su][k])
            log(f"    {c:.3f}  {ra['file']} ({ra['disease']})  ~  {rb['file']} ({rb['disease']})")
            pair_rows.append([b, round(c, 3), ra["file"], ra["disease"], rb["file"], rb["disease"]])
            sheet += [thumbs[key(ra)], thumbs[key(rb)]]
        save_sheet(sheet, SHEET_DIR / f"similar_pairs_{b}.jpg", cols=6)
    write_csv(NOTES_DIR / "similar_pairs.csv",
              ["background", "correlation", "file_a", "disease_a", "file_b", "disease_b"], pair_rows)

    # 8. Contact sheets (for looking at, not used by the methods) -----------
    log("\n== 8. Contact sheets ==")
    rng = np.random.default_rng(0)
    for b in BACKGROUNDS:
        for d in CLASSES:
            grp = [r for r in ok if r["background"] == b and r["disease"] == d]
            pick = [grp[i] for i in sorted(rng.choice(len(grp), min(24, len(grp)), replace=False))]
            save_sheet([thumbs[key(r)] for r in pick],
                       SHEET_DIR / f"{b}_{d.replace(' ', '').lower()}.jpg")
    for name, col, reverse in [("blurriest", "sharpness", False), ("darkest", "brightness", False),
                               ("brightest", "brightness", True)]:
        for b in BACKGROUNDS:
            grp = sorted((r for r in ok if r["background"] == b), key=lambda r: r[col], reverse=reverse)
            save_sheet([thumbs[key(r)] for r in grp[:12]], SHEET_DIR / f"extreme_{name}_{b}.jpg")
    field = sorted((r for r in ok if r["background"] == "field"), key=lambda r: r["green_frac"], reverse=True)
    save_sheet([thumbs[key(r)] for r in field[:12]], SHEET_DIR / "extreme_mostgreen_field.jpg")
    white = sorted((r for r in ok if r["background"] == "white"), key=lambda r: r["white_frac"], reverse=True)
    save_sheet([thumbs[key(r)] for r in white[:12]], SHEET_DIR / "extreme_mostwhite_white.jpg")
    log(f"  Saved to {SHEET_DIR}")

    log(f"\nDone in {time.time() - t0:.0f} s.")
    (NOTES_DIR / "01_inspect_log.txt").write_text("\n".join(_log_lines) + "\n")


if __name__ == "__main__":
    main()
