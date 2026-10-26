"""Report figures for the CSCI935 rice leaf project (Group 24). Author: Akram.

Three functions, each saving one 300 dpi PNG to results/figures/:
    plot_confusion_matrices(results_dir) -> confusion.png
    plot_sample_grid(samples_dir)        -> samples.png
    plot_failures(failures_dir, wrong_predictions_csv) -> failures.png

Every figure is 18 cm wide (full text width) and at most 6 cm tall, with text
at 9 pt or larger when printed at that size. No titles: the report captions
describe each figure. Nothing is typed in by hand; all values are read from files.

Uses matplotlib and numpy only (plus the standard library).
Run from the project folder:  python plots.py
"""
from pathlib import Path
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["Brown Spot", "Leaf Scald", "Rice Blast", "Rice Tungro", "Sheath Blight"]
SCENARIOS = [("white", "(a) White"), ("field", "(b) Field"), ("mixed", "(c) Mixed")]
# File-name stems used for the sample images in figures/samples/ (plan task A1).
SAMPLE_STEMS = ["brownspot", "leafscald", "riceblast", "tungro", "sheathblight"]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

CM = 1 / 2.54                      # centimetres -> inches
FIG_WIDTH = 18 * CM
FONT_PT = 9
DPI = 300
OUT_DIR = Path("results") / "figures"

plt.rcParams.update({
    # Arial/Helvetica are narrower than matplotlib's default, so labels fit the
    # small panels; DejaVu Sans (always installed) is the fallback.
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": FONT_PT,
    "axes.labelsize": FONT_PT,
    "xtick.labelsize": FONT_PT,
    "ytick.labelsize": FONT_PT,
    "savefig.dpi": DPI,
})


def _save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI)          # no bbox_inches="tight": keeps the exact size
    plt.close(fig)
    print(f"Saved {path}")
    return path


def _read_confusion(path):
    """Read a 5x5 confusion CSV: header row and first (class name) column are skipped."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header = [h.strip() for h in rows[0][1:]]
    if header != CLASSES:
        raise ValueError(f"{path}: unexpected class order {header}")
    return np.array([[int(v) for v in r[1:]] for r in rows[1:] if r], dtype=int)


def _wrap(name):
    """'Sheath Blight' -> 'Sheath\\nBlight' so labels fit narrow cells."""
    return name.replace(" ", "\n")


def _load_image(path, landscape=True, max_side=800):
    """Load an image with matplotlib, shrink it by striding (keeps memory and
    file size low), and turn it to match the tile shape (landscape or portrait)
    so that centre-cropping keeps as much of the photo as possible."""
    img = plt.imread(path)
    step = max(1, max(img.shape[:2]) // max_side)
    img = img[::step, ::step]
    if (img.shape[1] >= img.shape[0]) != landscape:
        img = np.rot90(img)
    return img


def _crop_to_aspect(img, aspect):
    """Centre-crop img to width/height = aspect."""
    h, w = img.shape[:2]
    if w / h > aspect:
        new_w = int(round(h * aspect))
        x0 = (w - new_w) // 2
        return img[:, x0:x0 + new_w]
    new_h = int(round(w / aspect))
    y0 = (h - new_h) // 2
    return img[y0:y0 + new_h, :]


def _failure_label(true, pred, max_chars=17):
    """'true: X / pred: Y' on two lines; a line longer than max_chars (too wide
    for a 3 cm tile at 9 pt) is broken after the colon."""
    lines = []
    for key, val in (("true", true), ("pred", pred)):
        line = f"{key}: {val}"
        lines.append(line if len(line) <= max_chars else f"{key}:\n{val}")
    return "\n".join(lines)


def _find_image(folder, stem):
    for ext in IMAGE_EXTENSIONS:
        for cand in (folder / f"{stem}{ext}", folder / f"{stem}{ext.upper()}"):
            if cand.is_file():
                return cand
    raise FileNotFoundError(f"No image named {stem}.jpg/.jpeg/.png in {folder}")


def plot_confusion_matrices(results_dir="results", method="resnet18_svm"):
    """One figure, 3 panels (white, field, mixed) for the main method.

    Cell colour = fraction of that true class (row-normalised), so the three
    scenarios share one colour bar even though their test sets differ in size.
    The number written in each cell is the raw count from the CSV.
    """
    results_dir = Path(results_dir)
    mats = [_read_confusion(results_dir / f"confusion_{method}_{s}.csv") for s, _ in SCENARIOS]

    fig = plt.figure(figsize=(FIG_WIDTH, 6 * CM))
    # Left margin holds the true-class names; right strip holds the colour bar.
    left, right, bottom, top = 0.085, 0.905, 0.255, 0.91
    gap = 0.012
    panel_w = (right - left - 2 * gap) / 3
    axes = [fig.add_axes([left + i * (panel_w + gap), bottom, panel_w, top - bottom])
            for i in range(3)]
    cax = fig.add_axes([right + 0.02, bottom, 0.012, top - bottom])

    ticks = np.arange(len(CLASSES))
    for i, (ax, m, (_, label)) in enumerate(zip(axes, mats, SCENARIOS)):
        frac = m / m.sum(axis=1, keepdims=True)
        im = ax.imshow(frac, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for r in range(m.shape[0]):
            for c in range(m.shape[1]):
                ax.text(c, r, str(m[r, c]), ha="center", va="center", fontsize=FONT_PT,
                        color="white" if frac[r, c] > 0.55 else "black")
        ax.set_xticks(ticks)
        ax.set_xticklabels([_wrap(c) for c in CLASSES], rotation=90, linespacing=0.9,
                           va="top", ha="center")
        ax.set_yticks(ticks)
        ax.tick_params(length=0, pad=2)
        if i == 0:
            ax.set_yticklabels([_wrap(c) for c in CLASSES], linespacing=0.9)
        else:
            ax.set_yticklabels([])
        ax.text(0.5, 1.02, label, transform=ax.transAxes, ha="center", va="bottom")
    axes[1].set_xlabel("Predicted class", labelpad=1)
    axes[0].set_ylabel("")
    fig.text(0.004, (bottom + top) / 2, "True class", rotation=90, va="center", ha="left")
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Fraction of true class", labelpad=3)
    cb.set_ticks([0, 0.5, 1])
    return _save(fig, "confusion.png")


def plot_sample_grid(samples_dir="figures/samples"):
    """2 rows x 5 columns: top row white background, bottom row field."""
    samples_dir = Path(samples_dir)
    fig_h = 6 * CM
    left, right, bottom, top = 0.045, 0.998, 0.01, 0.92
    gap_x, gap_y = 0.006, 0.02
    tile_w = (right - left - 4 * gap_x) / 5
    tile_h = (top - bottom - gap_y) / 2
    aspect = (tile_w * FIG_WIDTH) / (tile_h * fig_h)       # physical width/height of a tile

    fig = plt.figure(figsize=(FIG_WIDTH, fig_h))
    for r, (bg, row_label) in enumerate([("white", "White"), ("field", "Field")]):
        y0 = top - (r + 1) * tile_h - r * gap_y
        for c, (stem, disease) in enumerate(zip(SAMPLE_STEMS, CLASSES)):
            ax = fig.add_axes([left + c * (tile_w + gap_x), y0, tile_w, tile_h])
            img = _crop_to_aspect(_load_image(_find_image(samples_dir, f"{stem}_{bg}"),
                                              landscape=aspect >= 1), aspect)
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(disease, fontsize=FONT_PT, pad=3)
            if c == 0:
                ax.set_ylabel(row_label, fontsize=FONT_PT, labelpad=3)
    return _save(fig, "samples.png")


def plot_failures(failures_dir="figures/failures",
                  wrong_predictions_csv="results/wrong_predictions.csv"):
    """One strip of 6 misclassified images (3 white, then 3 field).

    The images in failures_dir keep their original dataset file names; the
    background, true and predicted classes are looked up in the CSV.
    """
    failures_dir = Path(failures_dir)
    with open(wrong_predictions_csv, newline="", encoding="utf-8") as f:
        lookup = {row["file"]: row for row in csv.DictReader(f)}

    picks = []
    for p in sorted(failures_dir.iterdir(), key=lambda p: p.name):
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if p.name not in lookup:
            raise ValueError(f"{p.name} is not listed in {wrong_predictions_csv}")
        picks.append((p, lookup[p.name]))
    white = [x for x in picks if x[1]["background"] == "white"]
    field = [x for x in picks if x[1]["background"] == "field"]
    if len(white) != 3 or len(field) != 3:
        raise ValueError(f"Need 3 white and 3 field images in {failures_dir}, "
                         f"found {len(white)} white and {len(field)} field")

    fig_h = 5.5 * CM
    left, right, bottom, top = 0.002, 0.998, 0.25, 0.995
    gap = 0.012
    tile_w = (right - left - 5 * gap) / 6
    tile_h = top - bottom
    aspect = (tile_w * FIG_WIDTH) / (tile_h * fig_h)

    fig = plt.figure(figsize=(FIG_WIDTH, fig_h))
    for i, (path, row) in enumerate(white + field):
        ax = fig.add_axes([left + i * (tile_w + gap), bottom, tile_w, tile_h])
        ax.imshow(_crop_to_aspect(_load_image(path, landscape=aspect >= 1), aspect))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(_failure_label(row['true'], row['predicted']),
                      fontsize=FONT_PT, labelpad=2, linespacing=1.0)
    return _save(fig, "failures.png")


if __name__ == "__main__":
    plot_confusion_matrices("results")
    plot_sample_grid("figures/samples")
    plot_failures("figures/failures", "results/wrong_predictions.csv")
