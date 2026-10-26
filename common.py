"""Shared settings and data listing for every stage (merged into group_name.py later).

Run every stage from the folder that contains Dhan-Shomadhan/ (the brief's
".\\Dhan-Shomadhan\\" path is relative to where the code is run).
"""
from pathlib import Path
import csv

DATASET_DIR = Path("Dhan-Shomadhan")
INTERIM_DIR = DATASET_DIR / "_interim"          # brief rule (b): interim data lives here
SPLITS_FILE = INTERIM_DIR / "splits.csv"
RESULTS_DIR = Path("results")

CLASSES = ["Brown Spot", "Leaf Scald", "Rice Blast", "Rice Tungro", "Sheath Blight"]
BACKGROUNDS = ["white", "field"]
SEEDS = [0, 1, 2, 3, 4]                          # one per split
TEST_FRACTION = 0.20

# Folder names in the supplied dataset are misspelled and some have trailing
# spaces. Each known spelling (lower case, spaces trimmed) maps to one class.
DISEASE_FOLDER_MAP = {
    "brown spot": "Brown Spot",
    "browon spot": "Brown Spot",
    "leaf scald": "Leaf Scald",
    "leaf scaled": "Leaf Scald",
    "rice blast": "Rice Blast",
    "rice tungro": "Rice Tungro",
    "rice turgro": "Rice Tungro",
    "sheath blight": "Sheath Blight",
    "shath blight": "Sheath Blight",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def background_from_folder(name):
    """'White Background ' -> 'white'; 'Field Background  ' -> 'field'."""
    key = name.strip().lower()
    if key.startswith("white"):
        return "white"
    if key.startswith("field") or key.startswith("feild"):
        return "field"
    raise ValueError(f"Unknown background folder name: {name!r}")


def disease_from_folder(name):
    """Map a (possibly misspelled) disease folder name to one of CLASSES."""
    key = " ".join(name.strip().lower().split())
    if key not in DISEASE_FOLDER_MAP:
        raise ValueError(f"Unknown disease folder name: {name!r}. "
                         f"Add its spelling to DISEASE_FOLDER_MAP in common.py.")
    return DISEASE_FOLDER_MAP[key]


def list_images(dataset_dir=DATASET_DIR):
    """Return one dict per image: path, file, background, disease.

    Only the two background folders are scanned, so anything in _interim/
    is never treated as a dataset image. Sorted, so the order is the same on
    every machine (sorted by the name string, because Windows compares paths
    without regard to upper/lower case and macOS/Linux do not).
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_dir.resolve()}. "
            f"Run the code from the folder that contains Dhan-Shomadhan/.")
    records = []
    for bg_dir in sorted((p for p in dataset_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
        if bg_dir.name.strip().lower().startswith("_"):
            continue                              # _interim and similar
        background = background_from_folder(bg_dir.name)
        for dz_dir in sorted((p for p in bg_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
            disease = disease_from_folder(dz_dir.name)
            for f in sorted(dz_dir.iterdir(), key=lambda p: p.name):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                    records.append({"path": f, "file": f.name,
                                    "background": background, "disease": disease})
    return records


def read_label_csv(dataset_dir=DATASET_DIR):
    """Read the supplied CSV: {file name: (disease, background)} using the same mapping."""
    matches = list(Path(dataset_dir).glob("*.csv"))
    if not matches:
        return None
    labels = {}
    with open(matches[0], newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raw = row["Diseases"]                 # e.g. "Browon Spot(Feild Background)"
            disease_part, bg_part = raw.split("(", 1)
            labels[row["pictureName"].strip()] = (disease_from_folder(disease_part),
                                                  background_from_folder(bg_part))
    return labels


def load_splits():
    """Read the saved splits: list of dicts with file, background, disease, path, split0..split4."""
    if not SPLITS_FILE.exists():
        raise FileNotFoundError(f"{SPLITS_FILE} not found. Run Stage 2 (02_splits.py) first.")
    paths = {(r["file"], r["background"], r["disease"]): r["path"] for r in list_images()}
    rows = []
    with open(SPLITS_FILE, newline="") as fh:
        for row in csv.DictReader(fh):
            row["path"] = paths[(row["file"], row["background"], row["disease"])]
            rows.append(row)
    return rows
