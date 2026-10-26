"""Stage 3: baseline features (colour histograms + HOG) for every image, saved once.

Each image: read at 1/4 size (fast), turned upright if landscape, resized to
128 x 256 (w x h), then
  colour: HSV histograms (H 16, S 8, V 8 bins) + YCrCb histograms (Cr 16, Cb 16),
          each divided by the pixel count                       -> 64 numbers
  HOG:    grey image, 9 orientations, 16 x 16 cells, 2 x 2 blocks -> 3780 numbers
Total 3844 numbers per image. Fixed rules per image, so nothing is learned
from other images here (no leakage). Scaling is done later on training only.

Output
  Dhan-Shomadhan/_interim/features_colour_hog.npz   X (1106 x 3844), files,
                                                    background, disease
  results/logs/03_baseline_features_log.txt

Run from the folder that contains Dhan-Shomadhan/:
  python 03_baseline_features.py
"""
import time

import cv2
import numpy as np
from skimage.feature import hog

from common import INTERIM_DIR, RESULTS_DIR, load_splits

OUT_FILE = INTERIM_DIR / "features_colour_hog.npz"
LOG_FILE = RESULTS_DIR / "logs" / "03_baseline_features_log.txt"
SIZE = (128, 256)                     # (width, height), close to the 1 : 2.1 photos
HSV_BINS = {"H": (0, 16, 180), "S": (1, 8, 256), "V": (2, 8, 256)}   # channel, bins, range
YCRCB_BINS = {"Cr": (1, 16, 256), "Cb": (2, 16, 256)}
HOG_ARGS = dict(orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                block_norm="L2-Hys", feature_vector=True)


def load_upright(path, size=SIZE):
    """Read an image at 1/4 resolution, rotate landscape to portrait, resize to size."""
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_4)
    if img is None:
        raise IOError(f"Cannot read {path}")
    if img.shape[1] > img.shape[0]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def colour_features(img):
    parts = []
    for space, bins in [(cv2.COLOR_BGR2HSV, HSV_BINS), (cv2.COLOR_BGR2YCrCb, YCRCB_BINS)]:
        conv = cv2.cvtColor(img, space)
        for ch, n, top in bins.values():
            h = cv2.calcHist([conv], [ch], None, [n], [0, top]).ravel()
            parts.append(h / h.sum())
    return np.concatenate(parts)


def hog_features(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return hog(grey, **HOG_ARGS)


def main():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    rows = load_splits()
    t0 = time.time()
    feats = []
    for i, r in enumerate(rows):
        img = load_upright(r["path"])
        feats.append(np.concatenate([colour_features(img), hog_features(img)]).astype(np.float32))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(rows)} ({time.time() - t0:.0f} s)")
    X = np.stack(feats)
    np.savez_compressed(OUT_FILE, X=X,
                        files=np.array([r["file"] for r in rows]),
                        background=np.array([r["background"] for r in rows]),
                        disease=np.array([r["disease"] for r in rows]))
    n_col = sum(b[1] for b in HSV_BINS.values()) + sum(b[1] for b in YCRCB_BINS.values())
    lines = [
        f"Images: {X.shape[0]}; features per image: {X.shape[1]} "
        f"(colour {n_col}, HOG {X.shape[1] - n_col})",
        f"Image size used: {SIZE[0]} x {SIZE[1]} (w x h); HOG {HOG_ARGS}",
        f"Any NaN or inf: {bool(~np.isfinite(X).all())}",
        f"Colour part: each of the 5 histograms sums to 1: "
        f"{bool(np.allclose(X[:, :n_col].sum(1), 5, atol=1e-3))}",
        f"HOG part: min {X[:, n_col:].min():.3f}, max {X[:, n_col:].max():.3f}, "
        f"mean {X[:, n_col:].mean():.3f}",
        f"Time: {time.time() - t0:.0f} s",
        f"Saved {OUT_FILE}",
    ]
    for line in lines:
        print(line)
    LOG_FILE.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
