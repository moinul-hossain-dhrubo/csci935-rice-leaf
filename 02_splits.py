"""Stage 2: five seeded, stratified 80/20 train/test splits, saved once.

Every later stage reads Dhan-Shomadhan/_interim/splits.csv, so all methods
are tested on exactly the same images in each split. Stratified by the 10
disease x background groups (D4).

Outputs
  Dhan-Shomadhan/_interim/splits.csv   one row per image: file, background,
                                       disease, split0 ... split4 (train/test)
  results/logs/02_splits_log.txt       counts and checks printed below

Run from the folder that contains Dhan-Shomadhan/:
  python 02_splits.py
"""
import csv
from collections import Counter

from sklearn.model_selection import train_test_split

from common import (BACKGROUNDS, CLASSES, INTERIM_DIR, RESULTS_DIR, SEEDS,
                    SPLITS_FILE, TEST_FRACTION, list_images)

LOG_FILE = RESULTS_DIR / "logs" / "02_splits_log.txt"
_log_lines = []


def log(msg=""):
    print(msg)
    _log_lines.append(str(msg))


def main():
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    records = list_images()
    groups = [f"{r['disease']}|{r['background']}" for r in records]
    idx = list(range(len(records)))
    log(f"Images: {len(records)}; groups: {len(set(groups))}; "
        f"test fraction {TEST_FRACTION}; seeds {SEEDS}")

    for k, seed in enumerate(SEEDS):
        _, test_idx = train_test_split(idx, test_size=TEST_FRACTION,
                                       stratify=groups, random_state=seed)
        test_set = set(test_idx)
        for i, r in enumerate(records):
            r[f"split{k}"] = "test" if i in test_set else "train"

    # Checks and counts ---------------------------------------------------
    log("\nTest images per group (disease, background) in each split:")
    log(f"  {'group':28}" + "".join(f"{'s' + str(k):>6}" for k in range(len(SEEDS))) + f"{'of':>6}")
    for d in CLASSES:
        for b in BACKGROUNDS:
            total = sum(1 for r in records if (r["disease"], r["background"]) == (d, b))
            n = [sum(1 for r in records if (r["disease"], r["background"]) == (d, b)
                     and r[f"split{k}"] == "test") for k in range(len(SEEDS))]
            log(f"  {d + ' / ' + b:28}" + "".join(f"{v:>6}" for v in n) + f"{total:>6}")
    log("\nPer split: train / test, and test white / field:")
    for k in range(len(SEEDS)):
        c = Counter(r[f"split{k}"] for r in records)
        tw = sum(1 for r in records if r[f"split{k}"] == "test" and r["background"] == "white")
        tf = sum(1 for r in records if r[f"split{k}"] == "test" and r["background"] == "field")
        assert c["train"] + c["test"] == len(records)        # every image in exactly one part
        log(f"  split{k}: train {c['train']}, test {c['test']} (white {tw}, field {tf})")
    times_tested = Counter(sum(r[f"split{k}"] == "test" for k in range(len(SEEDS))) for r in records)
    log("\nHow many of the 5 splits each image is a test image in "
        "(0 = never tested, 5 = always):")
    log("  " + ", ".join(f"{t}: {times_tested[t]} images" for t in range(len(SEEDS) + 1)))
    same = [(a, b) for a in range(len(SEEDS)) for b in range(a + 1, len(SEEDS))
            if all(r[f"split{a}"] == r[f"split{b}"] for r in records)]
    log(f"Identical splits (should be none): {same}")

    cols = ["file", "background", "disease"] + [f"split{k}" for k in range(len(SEEDS))]
    with open(SPLITS_FILE, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in records:
            w.writerow([r[c] for c in cols])
    log(f"\nSaved {SPLITS_FILE}")
    LOG_FILE.write_text("\n".join(_log_lines) + "\n")


if __name__ == "__main__":
    main()
