"""Rice leaf disease classification on the Dhan-Shomadhan dataset (CSCI935/435 group project, Group 24).

Method: frozen pretrained ResNet-18 (ImageNet) features + SVM.
Baseline: colour histograms (HSV, YCrCb) + HOG features + SVM.
Evaluation: 5 seeded, stratified 80/20 splits; each model is tested on the
white-background, field-background and all (mixed) test images.

Run from the folder that contains the dataset folder Dhan-Shomadhan/:
    python group24.py all          run every stage below in order
    python group24.py inspect      stage 1: dataset checks and statistics
    python group24.py splits       stage 2: the 5 train/test splits
    python group24.py baseline     stage 3: colour + HOG features
    python group24.py resnet       stage 4: ResNet-18 features
    python group24.py evaluate     stage 5: tune, train, test, write results
Optional: --data PATH (dataset folder, default Dhan-Shomadhan),
          --splits 0 1 ... (evaluate only some splits).

Interim data (splits, cached features, contact sheets) is written under
Dhan-Shomadhan/_interim/. Results are written to results/, dataset notes to
dataset-notes/. ResNet-18 weights download automatically on first use.
"""
import argparse
import csv
import hashlib
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

# ===========================================================================
# Settings
# ===========================================================================
CLASSES = ["Brown Spot", "Leaf Scald", "Rice Blast", "Rice Tungro", "Sheath Blight"]
BACKGROUNDS = ["white", "field"]
SCENARIOS = ["white", "field", "mixed"]
SEEDS = [0, 1, 2, 3, 4]                  # one per split
TEST_FRACTION = 0.20
RESULTS_DIR = Path("results")
NOTES_DIR = Path("dataset-notes")

# The supplied folder names are misspelled and some have trailing spaces.
# Each known spelling (lower case, spaces trimmed) maps to one class.
DISEASE_FOLDER_MAP = {
    "brown spot": "Brown Spot", "browon spot": "Brown Spot",
    "leaf scald": "Leaf Scald", "leaf scaled": "Leaf Scald",
    "rice blast": "Rice Blast",
    "rice tungro": "Rice Tungro", "rice turgro": "Rice Tungro",
    "sheath blight": "Sheath Blight", "shath blight": "Sheath Blight",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Baseline features (stage 3)
BASE_SIZE = (128, 256)                   # (width, height)
HSV_BINS = [(0, 16, 180), (1, 8, 256), (2, 8, 256)]       # (channel, bins, range): H, S, V
YCRCB_BINS = [(1, 16, 256), (2, 16, 256)]                 # Cr, Cb
HOG_ARGS = dict(orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                block_norm="L2-Hys", feature_vector=True)

# ResNet-18 features (stage 4)
RESNET_SIZE = (224, 448)                 # (width, height): keeps the tall photo shape
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
RESNET_BATCH = 32

# Classifier (stage 5)
METHODS = ["colour_hog_svm", "resnet18_svm"]
MAIN = "resnet18_svm"
EXTRA = "resnet18_svm_whiteonly"         # main method trained on white images only
CV_FOLDS = 5


class Paths:
    """All dataset-relative paths, set once from --data."""
    def __init__(self, dataset):
        self.dataset = Path(dataset)
        self.interim = self.dataset / "_interim"
        self.splits = self.interim / "splits.csv"
        self.sheets = self.interim / "inspect"
        self.features = {"colour_hog_svm": self.interim / "features_colour_hog.npz",
                         "resnet18_svm": self.interim / "features_resnet18.npz"}


class Log:
    """Print a line and keep it, so the same text can be saved to a log file."""
    def __init__(self, path):
        self.path = Path(path)
        self.lines = []

    def __call__(self, msg=""):
        print(msg)
        self.lines.append(str(msg))

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(self.lines) + "\n")


def write_csv(path, header, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


# ===========================================================================
# Reading the dataset
# ===========================================================================
def background_from_folder(name):
    key = name.strip().lower()
    if key.startswith("white"):
        return "white"
    if key.startswith("field") or key.startswith("feild"):
        return "field"
    raise ValueError(f"Unknown background folder name: {name!r}")


def disease_from_folder(name):
    key = " ".join(name.strip().lower().split())
    if key not in DISEASE_FOLDER_MAP:
        raise ValueError(f"Unknown disease folder name: {name!r}; add it to DISEASE_FOLDER_MAP.")
    return DISEASE_FOLDER_MAP[key]


def list_images(P):
    """One dict per image (path, file, background, disease), sorted by name strings
    so the order is identical on Windows, macOS and Linux."""
    if not P.dataset.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {P.dataset.resolve()}. "
                                f"Run from the folder that contains Dhan-Shomadhan/.")
    by_name = lambda p: p.name
    records = []
    for bg_dir in sorted((p for p in P.dataset.iterdir() if p.is_dir()), key=by_name):
        if bg_dir.name.strip().startswith("_"):
            continue                                       # _interim
        background = background_from_folder(bg_dir.name)
        for dz_dir in sorted((p for p in bg_dir.iterdir() if p.is_dir()), key=by_name):
            disease = disease_from_folder(dz_dir.name)
            for f in sorted(dz_dir.iterdir(), key=by_name):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                    records.append({"path": f, "file": f.name,
                                    "background": background, "disease": disease})
    return records


def read_label_csv(P):
    matches = list(P.dataset.glob("*.csv"))
    if not matches:
        return None
    labels = {}
    with open(matches[0], newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            disease_part, bg_part = row["Diseases"].split("(", 1)  # "Browon Spot(Feild Background)"
            labels[row["pictureName"].strip()] = (disease_from_folder(disease_part),
                                                  background_from_folder(bg_part))
    return labels


def load_splits(P):
    if not P.splits.exists():
        raise FileNotFoundError(f"{P.splits} not found. Run: python group24.py splits")
    paths = {(r["file"], r["background"], r["disease"]): r["path"] for r in list_images(P)}
    rows = []
    with open(P.splits, newline="") as fh:
        for row in csv.DictReader(fh):
            row["path"] = paths[(row["file"], row["background"], row["disease"])]
            rows.append(row)
    return rows


def resize_longest(img, side):
    h, w = img.shape[:2]
    s = side / max(h, w)
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


# ===========================================================================
# Stage 1: dataset inspection
# ===========================================================================
STAT_SIDE, THUMB, COPY_CORR, TOP_PAIRS = 512, 160, 0.98, 12


def _thumb_vector(gray):
    if gray.shape[1] > gray.shape[0]:
        gray = gray.T
    v = cv2.resize(gray, (24, 48), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-9)


def _thumbnail(img, label):
    t = resize_longest(img, THUMB)
    canvas = np.full((THUMB + 18, THUMB, 3), 255, np.uint8)
    y0, x0 = (THUMB - t.shape[0]) // 2, (THUMB - t.shape[1]) // 2
    canvas[y0:y0 + t.shape[0], x0:x0 + t.shape[1]] = t
    cv2.putText(canvas, label[:22], (2, THUMB + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def _save_sheet(thumbs, path, cols=6):
    if not thumbs:
        return
    rows = []
    for i in range(0, len(thumbs), cols):
        row = thumbs[i:i + cols]
        row += [np.full_like(thumbs[0], 255)] * (cols - len(row))
        rows.append(np.hstack(row))
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def stage_inspect(P):
    log = Log(NOTES_DIR / "01_inspect_log.txt")
    t0 = time.time()
    P.sheets.mkdir(parents=True, exist_ok=True)
    key = lambda r: r["file"] + r["background"] + r["disease"]

    log("== 1. Folder structure ==")
    for bg_dir in sorted((p for p in P.dataset.iterdir() if p.is_dir()), key=lambda p: p.name):
        if bg_dir.name.startswith("_"):
            continue
        for dz_dir in sorted((p for p in bg_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
            n = sum(1 for f in dz_dir.iterdir() if f.is_file())
            log(f"  {bg_dir.name!r:24} / {dz_dir.name!r:18} {n} files")
    records = list_images(P)
    log(f"Images found: {len(records)}")

    log("\n== 2. Label check: folder vs CSV vs file-name prefix ==")
    csv_labels = read_label_csv(P)
    if csv_labels is None:
        log("  No CSV found in the dataset folder; CSV check skipped.")
    else:
        names = Counter(r["file"] for r in records)
        log(f"  CSV rows: {len(csv_labels)}")
        log(f"  File names used more than once across folders: {sum(c > 1 for c in names.values())}")
        log(f"  Images not in CSV: {sum(r['file'] not in csv_labels for r in records)}")
        log(f"  CSV rows with no image: {len(set(csv_labels) - set(names))}")
        mism = [r["file"] for r in records
                if r["file"] in csv_labels and csv_labels[r["file"]] != (r["disease"], r["background"])]
        log(f"  Folder/CSV label mismatches: {len(mism)} {mism[:10]}")
    prefix_by_group = defaultdict(Counter)
    for r in records:
        r["prefix"] = "_".join(Path(r["file"]).stem.split("_")[:-1]) or Path(r["file"]).stem
        prefix_by_group[(r["background"], r["disease"])][r["prefix"]] += 1
    log("  File-name prefixes per group:")
    odd = []
    for gk in sorted(prefix_by_group):
        log(f"    {gk[0]:5} {gk[1]:13} {dict(prefix_by_group[gk])}")
        main_prefix = prefix_by_group[gk].most_common(1)[0][0]
        odd += [r["file"] for r in records
                if (r["background"], r["disease"]) == gk and r["prefix"] != main_prefix]
    log(f"  Images whose prefix differs from the rest of their folder: {len(odd)} {odd[:10]}")

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

    log("\n== 4. Reading every image (sizes, quality statistics) ==")
    unreadable, thumbs = [], {}
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
        r["brightness"] = round(float(v.mean()), 1)
        r["contrast"] = round(float(gray.std()), 1)
        r["sharpness"] = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1)
        r["white_frac"] = round(float(((s < 40) & (v > 180)).mean()), 3)
        r["green_frac"] = round(float(((h >= 35) & (h <= 85) & (s > 60) & (v > 40)).mean()), 3)
        r["thumbvec"] = _thumb_vector(gray)
        thumbs[key(r)] = _thumbnail(small, r["file"])
        if (i + 1) % 200 == 0:
            log(f"  {i + 1}/{len(records)} read ({time.time() - t0:.0f} s)")
    log(f"  Unreadable images: {len(unreadable)} {unreadable[:10]}")
    ok = [r for r in records if "width" in r]
    cols = ["width", "height", "file_kb", "brightness", "contrast", "sharpness", "white_frac", "green_frac"]
    write_csv(NOTES_DIR / "image_stats.csv", ["file", "background", "disease"] + cols + ["md5"],
              [[r["file"], r["background"], r["disease"]] + [r[c] for c in cols] + [r["md5"]] for r in ok])

    log("\n== 5. Image sizes per background ==")
    size_rows = []
    for b in BACKGROUNDS:
        grp = [r for r in ok if r["background"] == b]
        if not grp:
            continue
        sizes = Counter((r["width"], r["height"]) for r in grp)
        W = np.array([r["width"] for r in grp]); H = np.array([r["height"] for r in grp])
        portrait, square = int((H > W).sum()), int((H == W).sum())
        log(f"  {b}: n={len(grp)} width {W.min()} to {W.max()} (median {int(np.median(W))}), "
            f"height {H.min()} to {H.max()} (median {int(np.median(H))}), portrait {portrait}, "
            f"landscape {len(grp) - portrait - square}, square {square}")
        log(f"     most common sizes (w x h): {sizes.most_common(5)}")
        size_rows.append([b, len(grp), W.min(), W.max(), int(np.median(W)), H.min(), H.max(),
                          int(np.median(H)), portrait, len(sizes)])
    write_csv(NOTES_DIR / "size_summary.csv", ["background", "n", "w_min", "w_max", "w_median", "h_min",
                                               "h_max", "h_median", "n_portrait", "n_distinct_sizes"], size_rows)

    log("\n== 6. Quality statistics (median [min, max]) per background ==")
    for b in BACKGROUNDS:
        grp = [r for r in ok if r["background"] == b]
        parts = []
        for c in ["brightness", "contrast", "sharpness", "white_frac", "green_frac"]:
            a = np.array([r[c] for r in grp])
            parts.append(f"{c} {np.median(a):.3g} [{a.min():.3g}, {a.max():.3g}]")
        log(f"  {b}: " + "; ".join(parts))
    log(f"  (brightness = mean HSV value 0-255; sharpness = Laplacian variance at {STAT_SIDE}px, "
        f"higher = sharper; white_frac / green_frac = share of near-white / green pixels)")

    log("\n== 7. Duplicate and most similar images ==")
    by_md5 = defaultdict(list)
    for r in ok:
        by_md5[r["md5"]].append(r["file"])
    exact = [g for g in by_md5.values() if len(g) > 1]
    log(f"  Exact duplicate groups (same bytes): {len(exact)} {exact[:5]}")
    V = np.array([r["thumbvec"] for r in ok])
    C = V @ V.T
    corr = C[np.triu_indices(len(ok), 1)]
    log(f"  Pairs with thumbnail correlation >= {COPY_CORR}: {int((corr >= COPY_CORR).sum())}")
    log(f"  Highest correlation of any pair: {corr.max():.3f}")
    pair_rows = []
    for b in BACKGROUNDS:
        idx = [k for k, r in enumerate(ok) if r["background"] == b]
        sub = C[np.ix_(idx, idx)]
        su = np.triu_indices(len(idx), 1)
        vals = sub[su]
        log(f"  {TOP_PAIRS} most similar {b} pairs (correlation, files, labels):")
        sheet = []
        for k in np.argsort(-vals)[:TOP_PAIRS]:
            ra, rb = ok[idx[su[0][k]]], ok[idx[su[1][k]]]
            log(f"    {vals[k]:.3f}  {ra['file']} ({ra['disease']})  ~  {rb['file']} ({rb['disease']})")
            pair_rows.append([b, round(float(vals[k]), 3), ra["file"], ra["disease"], rb["file"], rb["disease"]])
            sheet += [thumbs[key(ra)], thumbs[key(rb)]]
        _save_sheet(sheet, P.sheets / f"similar_pairs_{b}.jpg")
    write_csv(NOTES_DIR / "similar_pairs.csv",
              ["background", "correlation", "file_a", "disease_a", "file_b", "disease_b"], pair_rows)

    log("\n== 8. Contact sheets ==")
    rng = np.random.default_rng(0)
    for b in BACKGROUNDS:
        for d in CLASSES:
            grp = [r for r in ok if r["background"] == b and r["disease"] == d]
            pick = [grp[i] for i in sorted(rng.choice(len(grp), min(24, len(grp)), replace=False))]
            _save_sheet([thumbs[key(r)] for r in pick], P.sheets / f"{b}_{d.replace(' ', '').lower()}.jpg")
    for name, col, rev in [("blurriest", "sharpness", False), ("darkest", "brightness", False),
                           ("brightest", "brightness", True)]:
        for b in BACKGROUNDS:
            grp = sorted((r for r in ok if r["background"] == b), key=lambda r: r[col], reverse=rev)
            _save_sheet([thumbs[key(r)] for r in grp[:12]], P.sheets / f"extreme_{name}_{b}.jpg")
    for b, col, name in [("field", "green_frac", "mostgreen"), ("white", "white_frac", "mostwhite")]:
        grp = sorted((r for r in ok if r["background"] == b), key=lambda r: r[col], reverse=True)
        _save_sheet([thumbs[key(r)] for r in grp[:12]], P.sheets / f"extreme_{name}_{b}.jpg")
    log(f"  Saved to {P.sheets}")
    log(f"\nDone in {time.time() - t0:.0f} s.")
    log.save()


# ===========================================================================
# Stage 2: five seeded, stratified 80/20 splits
# ===========================================================================
def stage_splits(P):
    from sklearn.model_selection import train_test_split
    log = Log(RESULTS_DIR / "logs" / "02_splits_log.txt")
    P.interim.mkdir(parents=True, exist_ok=True)
    records = list_images(P)
    groups = [f"{r['disease']}|{r['background']}" for r in records]
    idx = list(range(len(records)))
    log(f"Images: {len(records)}; groups: {len(set(groups))}; test fraction {TEST_FRACTION}; seeds {SEEDS}")
    for k, seed in enumerate(SEEDS):
        _, test_idx = train_test_split(idx, test_size=TEST_FRACTION, stratify=groups, random_state=seed)
        test_set = set(test_idx)
        for i, r in enumerate(records):
            r[f"split{k}"] = "test" if i in test_set else "train"

    log("\nTest images per group (disease, background) in each split:")
    log(f"  {'group':28}" + "".join(f"{'s' + str(k):>6}" for k in range(len(SEEDS))) + f"{'of':>6}")
    for d in CLASSES:
        for b in BACKGROUNDS:
            members = [r for r in records if (r["disease"], r["background"]) == (d, b)]
            n = [sum(r[f"split{k}"] == "test" for r in members) for k in range(len(SEEDS))]
            log(f"  {d + ' / ' + b:28}" + "".join(f"{v:>6}" for v in n) + f"{len(members):>6}")
    log("\nPer split: train / test, and test white / field:")
    for k in range(len(SEEDS)):
        c = Counter(r[f"split{k}"] for r in records)
        assert c["train"] + c["test"] == len(records)
        tw = sum(r[f"split{k}"] == "test" and r["background"] == "white" for r in records)
        tf = sum(r[f"split{k}"] == "test" and r["background"] == "field" for r in records)
        log(f"  split{k}: train {c['train']}, test {c['test']} (white {tw}, field {tf})")
    times = Counter(sum(r[f"split{k}"] == "test" for k in range(len(SEEDS))) for r in records)
    log("\nHow many of the 5 splits each image is a test image in (0 = never tested, 5 = always):")
    log("  " + ", ".join(f"{t}: {times[t]} images" for t in range(len(SEEDS) + 1)))
    same = [(a, b) for a in range(len(SEEDS)) for b in range(a + 1, len(SEEDS))
            if all(r[f"split{a}"] == r[f"split{b}"] for r in records)]
    log(f"Identical splits (should be none): {same}")
    cols = ["file", "background", "disease"] + [f"split{k}" for k in range(len(SEEDS))]
    write_csv(P.splits, cols, [[r[c] for c in cols] for r in records])
    log(f"\nSaved {P.splits}")
    log.save()


# ===========================================================================
# Stage 3: baseline features (colour histograms + HOG)
# ===========================================================================
def load_upright(path, size, rgb=False):
    """Read at 1/4 resolution, optionally convert to RGB, rotate landscape to
    portrait, resize to size = (width, height)."""
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_4)
    if img is None:
        raise IOError(f"Cannot read {path}")
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[1] > img.shape[0]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def colour_features(img):
    parts = []
    for space, bins in [(cv2.COLOR_BGR2HSV, HSV_BINS), (cv2.COLOR_BGR2YCrCb, YCRCB_BINS)]:
        conv = cv2.cvtColor(img, space)
        for ch, n, top in bins:
            h = cv2.calcHist([conv], [ch], None, [n], [0, top]).ravel()
            parts.append(h / h.sum())
    return np.concatenate(parts)


def save_features(P, method, X, rows):
    np.savez_compressed(P.features[method], X=X,
                        files=np.array([r["file"] for r in rows]),
                        background=np.array([r["background"] for r in rows]),
                        disease=np.array([r["disease"] for r in rows]))


def stage_baseline_features(P):
    from skimage.feature import hog
    log = Log(RESULTS_DIR / "logs" / "03_baseline_features_log.txt")
    rows = load_splits(P)
    t0 = time.time()
    feats = []
    for i, r in enumerate(rows):
        img = load_upright(r["path"], BASE_SIZE)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feats.append(np.concatenate([colour_features(img), hog(grey, **HOG_ARGS)]).astype(np.float32))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(rows)} ({time.time() - t0:.0f} s)")
    X = np.stack(feats)
    save_features(P, "colour_hog_svm", X, rows)
    n_col = sum(b[1] for b in HSV_BINS) + sum(b[1] for b in YCRCB_BINS)
    log(f"Images: {X.shape[0]}; features per image: {X.shape[1]} (colour {n_col}, HOG {X.shape[1] - n_col})")
    log(f"Image size used: {BASE_SIZE[0]} x {BASE_SIZE[1]} (w x h); HOG {HOG_ARGS}")
    log(f"Any NaN or inf: {bool(~np.isfinite(X).all())}")
    log(f"Colour part: each of the 5 histograms sums to 1: {bool(np.allclose(X[:, :n_col].sum(1), 5, atol=1e-3))}")
    log(f"HOG part: min {X[:, n_col:].min():.3f}, max {X[:, n_col:].max():.3f}, mean {X[:, n_col:].mean():.3f}")
    log(f"Time: {time.time() - t0:.0f} s")
    log(f"Saved {P.features['colour_hog_svm']}")
    log.save()


# ===========================================================================
# Stage 4: ResNet-18 features (frozen, ImageNet weights)
# ===========================================================================
def stage_resnet_features(P):
    import torch
    from torchvision.models import ResNet18_Weights, resnet18
    log = Log(RESULTS_DIR / "logs" / "04_resnet_features_log.txt")
    rows = load_splits(P)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)   # downloads weights on first use
    model.fc = torch.nn.Identity()          # drop the ImageNet classifier: output = 512 pooled features
    model.eval().to(device)

    def prepare(path):
        img = load_upright(path, RESNET_SIZE, rgb=True).astype(np.float32) / 255.0
        return ((img - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)

    t0 = time.time()
    feats = []
    with torch.no_grad():
        for start in range(0, len(rows), RESNET_BATCH):
            batch = np.stack([prepare(r["path"]) for r in rows[start:start + RESNET_BATCH]])
            feats.append(model(torch.from_numpy(batch).to(device)).cpu().numpy().astype(np.float32))
            print(f"  {min(start + RESNET_BATCH, len(rows))}/{len(rows)} ({time.time() - t0:.0f} s)")
    X = np.concatenate(feats)
    save_features(P, "resnet18_svm", X, rows)
    log(f"Device: {device}; torch {torch.__version__}")
    log(f"Weights: ResNet18_Weights.IMAGENET1K_V1; input {RESNET_SIZE[0]} x {RESNET_SIZE[1]} (w x h)")
    log(f"Images: {X.shape[0]}; features per image: {X.shape[1]}")
    log(f"Any NaN or inf: {bool(~np.isfinite(X).all())}")
    log(f"Values: min {X.min():.3f}, max {X.max():.3f}, mean {X.mean():.3f}; share of exact zeros {(X == 0).mean():.3f}")
    log(f"Time: {time.time() - t0:.0f} s")
    log(f"Saved {P.features['resnet18_svm']}")
    log.save()


# ===========================================================================
# Stage 5: SVM tuning (training part only), testing, result files
# ===========================================================================
def param_grid(n_features):
    """Linear SVM (5 C values) and RBF SVM (5 C x 4 gamma); gamma relative to
    1/n_features, scikit-learn's 'scale' value for standardised features."""
    g = 1.0 / n_features
    return [{"svc__kernel": ["linear"], "svc__C": [0.0001, 0.001, 0.01, 0.1, 1]},
            {"svc__kernel": ["rbf"], "svc__C": [1, 10, 100, 1000, 10000],
             "svc__gamma": [g / 16, g / 4, g, 4 * g]}]


def tune_and_fit(X, y, groups, seed):
    """Grid search with 5-fold CV (stratified by disease x background) on the given
    TRAINING images only; scaler fitted inside each fold; best setting refitted."""
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    pipe = Pipeline([("scale", StandardScaler()), ("svc", SVC())])
    folds = list(StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed).split(X, groups))
    search = GridSearchCV(pipe, param_grid(X.shape[1]), cv=folds, scoring="f1_macro", n_jobs=-1, refit=True)
    search.fit(X, y)
    return search


def load_features(P, method, rows):
    data = np.load(P.features[method])
    keys = list(zip(data["files"], data["background"], data["disease"]))
    if keys != [(r["file"], r["background"], r["disease"]) for r in rows]:
        raise ValueError(f"{P.features[method]} does not match {P.splits}; rerun the feature stages.")
    return data["X"]


def stage_evaluate(P, split_ids):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
    log = Log(RESULTS_DIR / "logs" / "05_train_evaluate_log.txt")
    rows = load_splits(P)
    y = np.array([r["disease"] for r in rows])
    bg = np.array([r["background"] for r in rows])
    files = np.array([r["file"] for r in rows])
    groups = np.array([f"{d}|{b}" for d, b in zip(y, bg)])
    t0 = time.time()
    summary, per_class, preds, params = [], [], [], []
    conf = defaultdict(lambda: np.zeros((len(CLASSES), len(CLASSES)), int))

    def record(method, k, scen, mask, y_pred):
        yt, yp = y[mask], y_pred[mask]
        acc = accuracy_score(yt, yp)
        mf1 = f1_score(yt, yp, labels=CLASSES, average="macro", zero_division=0)
        summary.append([method, scen, k, round(acc, 4), round(mf1, 4)])
        p, r, f, s = precision_recall_fscore_support(yt, yp, labels=CLASSES, zero_division=0)
        for c, pc, rc, fc, sc in zip(CLASSES, p, r, f, s):
            per_class.append([method, scen, k, c, round(pc, 4), round(rc, 4), round(fc, 4), int(sc)])
        conf[(method, scen)] += confusion_matrix(yt, yp, labels=CLASSES)
        log(f"    {scen:6} n={mask.sum():3}  accuracy {acc:.4f}  macro F1 {mf1:.4f}")

    def fit_predict(method, k, X, train_mask, test_mask, trained_on):
        search = tune_and_fit(X[train_mask], y[train_mask], groups[train_mask], SEEDS[k])
        bp = search.best_params_
        params.append([method, k, trained_on, bp.get("svc__kernel"), bp.get("svc__C"),
                       bp.get("svc__gamma", ""), round(search.best_score_, 4)])
        y_pred = np.empty(len(rows), dtype=object)
        y_pred[test_mask] = search.predict(X[test_mask])
        for f_, b_, t_, p_ in zip(files[test_mask], bg[test_mask], y[test_mask], y_pred[test_mask]):
            preds.append([method, k, f_, b_, t_, p_])
        log(f"  split{k} ({trained_on}): chosen {bp}, CV macro F1 {search.best_score_:.4f} ({time.time() - t0:.0f} s)")
        return y_pred

    for method in METHODS:
        X = load_features(P, method, rows)
        log(f"\n== {method}: {X.shape[1]} features ==")
        for k in split_ids:
            train = np.array([r[f"split{k}"] == "train" for r in rows])
            test = ~train
            y_pred = fit_predict(method, k, X, train, test, "all_train")
            for scen in SCENARIOS:
                record(method, k, scen, test if scen == "mixed" else test & (bg == scen), y_pred)
            if method == MAIN:                       # extra: white-only training, field test
                field_test = test & (bg == "field")
                y2 = fit_predict(EXTRA, k, X, train & (bg == "white"), field_test, "white_train")
                record(EXTRA, k, "field", field_test, y2)

    write_csv(RESULTS_DIR / "summary.csv", ["method", "scenario", "split", "accuracy", "macro_f1"], summary)
    write_csv(RESULTS_DIR / "per_class.csv",
              ["method", "scenario", "split", "class", "precision", "recall", "f1", "support"], per_class)
    write_csv(RESULTS_DIR / "predictions.csv", ["method", "split", "file", "background", "true", "predicted"], preds)
    write_csv(RESULTS_DIR / "best_params.csv",
              ["method", "split", "trained_on", "kernel", "C", "gamma", "cv_macro_f1"], params)
    for (method, scen), m in conf.items():
        write_csv(RESULTS_DIR / f"confusion_{method}_{scen}.csv", ["true \\ predicted"] + CLASSES,
                  [[c] + list(row) for c, row in zip(CLASSES, m)])

    wrong, tested, info = defaultdict(Counter), Counter(), {}
    for method, k, f_, b_, t_, p_ in preds:
        if method == MAIN:
            tested[f_] += 1
            info[f_] = (b_, t_)
            if p_ != t_:
                wrong[f_][p_] += 1
    wrong_rows = sorted(([f_, info[f_][0], info[f_][1], c.most_common(1)[0][0], sum(c.values()), tested[f_]]
                         for f_, c in wrong.items()), key=lambda r: (r[1], -r[4], r[0]))
    write_csv(RESULTS_DIR / "wrong_predictions.csv",
              ["file", "background", "true", "predicted", "times_wrong", "times_tested"], wrong_rows)

    log(f"\n== Mean +/- std over splits {split_ids} (std = sample std, ddof=1) ==")
    log(f"  {'method':24}{'scenario':9}{'accuracy':>18}{'macro F1':>18}")
    sd = lambda v: v.std(ddof=1) if len(v) > 1 else float("nan")
    for method in METHODS + [EXTRA]:
        for scen in SCENARIOS:
            sel = [s for s in summary if s[0] == method and s[1] == scen]
            if sel:
                a = np.array([s[3] for s in sel]); f = np.array([s[4] for s in sel])
                log(f"  {method:24}{scen:9}{a.mean():>10.4f} +/- {sd(a):.4f}{f.mean():>10.4f} +/- {sd(f):.4f}")
    log("\n== Mean recall per class over the splits (share of each disease's test images found) ==")
    log(f"  {'method':24}{'scenario':9}" + "".join(f"{c:>15}" for c in CLASSES))
    for method in METHODS:
        for scen in SCENARIOS:
            vals = [np.mean([pc[5] for pc in per_class if pc[0] == method and pc[1] == scen and pc[3] == c])
                    for c in CLASSES]
            log(f"  {method:24}{scen:9}" + "".join(f"{v:>15.3f}" for v in vals))
    log(f"\nMain-method test images wrong at least once: {len(wrong_rows)} "
        f"(white {sum(r[1] == 'white' for r in wrong_rows)}, field {sum(r[1] == 'field' for r in wrong_rows)})")
    log(f"Time: {time.time() - t0:.0f} s")
    log.save()


# ===========================================================================
# Command line
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="Rice leaf disease classification (Dhan-Shomadhan).")
    ap.add_argument("stage", choices=["all", "inspect", "splits", "baseline", "resnet", "evaluate"])
    ap.add_argument("--data", default="Dhan-Shomadhan", help="dataset folder (default: Dhan-Shomadhan)")
    ap.add_argument("--splits", type=int, nargs="+", default=list(range(len(SEEDS))),
                    help="splits to evaluate (default: all 5)")
    args = ap.parse_args()
    P = Paths(args.data)
    t0 = time.time()
    if args.stage in ("all", "inspect"):
        stage_inspect(P)
    if args.stage in ("all", "splits"):
        stage_splits(P)
    if args.stage in ("all", "baseline"):
        stage_baseline_features(P)
    if args.stage in ("all", "resnet"):
        stage_resnet_features(P)
    if args.stage in ("all", "evaluate"):
        stage_evaluate(P, args.splits)
    print(f"\nFinished '{args.stage}' in {time.time() - t0:.0f} s.")


if __name__ == "__main__":
    main()
