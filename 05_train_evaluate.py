"""Stage 5: train and evaluate the SVM on both feature sets, all splits, all scenarios.

For each method (colour_hog_svm, resnet18_svm) and each split:
  1. Training part only: choose SVM settings by 5-fold cross-validation
     (folds stratified by disease x background), scored by macro F1 (D7, D15).
  2. Refit the chosen setting on the whole training part.
  3. Predict every test image once; score three ways (D5):
       white = white test images, field = field test images, mixed = all.
Extra run (D6), main method only: train on white training images, test on
field test images (method name resnet18_svm_whiteonly, scenario field).
Test images are never used for tuning or training.

Outputs (results/)
  summary.csv                   method, scenario, split, accuracy, macro_f1
  per_class.csv                 ... class, precision, recall, f1, support
  confusion_<method>_<scenario>.csv   counts summed over the splits run
  predictions.csv               every test prediction (method, split, file, ...)
  best_params.csv               chosen SVM setting per method x split
  wrong_predictions.csv         main method's mistakes, one row per image
  logs/05_train_evaluate_log.txt

Run from the folder that contains Dhan-Shomadhan/:
  python 05_train_evaluate.py              (all 5 splits)
  python 05_train_evaluate.py --splits 0   (split 0 only)
"""
import argparse
import csv
import time
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import CLASSES, INTERIM_DIR, RESULTS_DIR, SEEDS, load_splits

FEATURE_FILES = {"colour_hog_svm": INTERIM_DIR / "features_colour_hog.npz",
                 "resnet18_svm": INTERIM_DIR / "features_resnet18.npz"}
MAIN = "resnet18_svm"
EXTRA = "resnet18_svm_whiteonly"
SCENARIOS = ["white", "field", "mixed"]
CV_FOLDS = 5
LOG_FILE = RESULTS_DIR / "logs" / "05_train_evaluate_log.txt"
_log_lines = []


def log(msg=""):
    print(msg)
    _log_lines.append(str(msg))


def param_grid(n_features):
    """Linear SVM (5 values of C) and RBF SVM (5 C x 4 gamma); gamma relative to
    1/n_features, which is sklearn's 'scale' value for standardised features.
    Widened on 26 Oct (D17): the first grid's edges were chosen in every split."""
    g = 1.0 / n_features
    return [{"svc__kernel": ["linear"], "svc__C": [0.0001, 0.001, 0.01, 0.1, 1]},
            {"svc__kernel": ["rbf"], "svc__C": [1, 10, 100, 1000, 10000],
             "svc__gamma": [g / 16, g / 4, g, 4 * g]}]


def tune_and_fit(X, y, groups, seed):
    """Grid search on the given (training) images only; returns the refitted model."""
    pipe = Pipeline([("scale", StandardScaler()), ("svc", SVC())])
    folds = list(StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed).split(X, groups))
    search = GridSearchCV(pipe, param_grid(X.shape[1]), cv=folds, scoring="f1_macro",
                          n_jobs=-1, refit=True)
    search.fit(X, y)
    return search


def load_features(method, rows):
    data = np.load(FEATURE_FILES[method])
    keys = list(zip(data["files"], data["background"], data["disease"]))
    expected = [(r["file"], r["background"], r["disease"]) for r in rows]
    if keys != expected:
        raise ValueError(f"{FEATURE_FILES[method]} does not match splits.csv; rerun Stages 3 and 4.")
    return data["X"]


def score(y_true, y_pred):
    return (accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, nargs="+", default=list(range(len(SEEDS))))
    args = ap.parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    rows = load_splits()
    y = np.array([r["disease"] for r in rows])
    bg = np.array([r["background"] for r in rows])
    files = np.array([r["file"] for r in rows])
    groups = np.array([f"{d}|{b}" for d, b in zip(y, bg)])
    t0 = time.time()

    summary, per_class, preds, params = [], [], [], []
    conf = defaultdict(lambda: np.zeros((len(CLASSES), len(CLASSES)), int))

    def record(method, k, scen, mask, y_pred_all):
        yt, yp = y[mask], y_pred_all[mask]
        acc, mf1 = score(yt, yp)
        summary.append([method, scen, k, round(acc, 4), round(mf1, 4)])
        p, r, f, s = precision_recall_fscore_support(yt, yp, labels=CLASSES, zero_division=0)
        for c, pc, rc, fc, sc in zip(CLASSES, p, r, f, s):
            per_class.append([method, scen, k, c, round(pc, 4), round(rc, 4), round(fc, 4), int(sc)])
        conf[(method, scen)] += confusion_matrix(yt, yp, labels=CLASSES)
        log(f"    {scen:6} n={mask.sum():3}  accuracy {acc:.4f}  macro F1 {mf1:.4f}")

    for method in FEATURE_FILES:
        X = load_features(method, rows)
        log(f"\n== {method}: {X.shape[1]} features ==")
        for k in args.splits:
            seed = SEEDS[k]
            train = np.array([r[f"split{k}"] == "train" for r in rows])
            test = ~train
            search = tune_and_fit(X[train], y[train], groups[train], seed)
            bp = search.best_params_
            params.append([method, k, "all_train", bp.get("svc__kernel"), bp.get("svc__C"),
                           bp.get("svc__gamma", ""), round(search.best_score_, 4)])
            log(f"  split{k}: chosen {bp}, CV macro F1 {search.best_score_:.4f} "
                f"({time.time() - t0:.0f} s)")
            y_pred = np.empty(len(rows), dtype=object)
            y_pred[test] = search.predict(X[test])
            for f_, b_, t_, p_ in zip(files[test], bg[test], y[test], y_pred[test]):
                preds.append([method, k, f_, b_, t_, p_])
            for scen in SCENARIOS:
                mask = test & (bg == scen) if scen != "mixed" else test
                record(method, k, scen, mask, y_pred)

            if method == MAIN:                          # D6: white-only training, field test
                tr_w = train & (bg == "white")
                s2 = tune_and_fit(X[tr_w], y[tr_w], groups[tr_w], seed)
                bp2 = s2.best_params_
                params.append([EXTRA, k, "white_train", bp2.get("svc__kernel"), bp2.get("svc__C"),
                               bp2.get("svc__gamma", ""), round(s2.best_score_, 4)])
                yp2 = np.empty(len(rows), dtype=object)
                mask = test & (bg == "field")
                yp2[mask] = s2.predict(X[mask])
                for f_, b_, t_, p_ in zip(files[mask], bg[mask], y[mask], yp2[mask]):
                    preds.append([EXTRA, k, f_, b_, t_, p_])
                log(f"  split{k} extra (train white only): chosen {bp2}")
                record(EXTRA, k, "field", mask, yp2)

    # Write results -------------------------------------------------------
    def write(name, header, data):
        with open(RESULTS_DIR / name, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(data)

    write("summary.csv", ["method", "scenario", "split", "accuracy", "macro_f1"], summary)
    write("per_class.csv", ["method", "scenario", "split", "class", "precision", "recall",
                            "f1", "support"], per_class)
    write("predictions.csv", ["method", "split", "file", "background", "true", "predicted"], preds)
    write("best_params.csv", ["method", "split", "trained_on", "kernel", "C", "gamma",
                              "cv_macro_f1"], params)
    for (method, scen), m in conf.items():
        write(f"confusion_{method}_{scen}.csv", ["true \\ predicted"] + CLASSES,
              [[c] + list(row) for c, row in zip(CLASSES, m)])

    wrong = defaultdict(Counter)
    tested = Counter()
    info = {}
    for method, k, f_, b_, t_, p_ in preds:
        if method != MAIN:
            continue
        tested[f_] += 1
        info[f_] = (b_, t_)
        if p_ != t_:
            wrong[f_][p_] += 1
    wrong_rows = sorted(([f_, info[f_][0], info[f_][1], c.most_common(1)[0][0],
                          sum(c.values()), tested[f_]] for f_, c in wrong.items()),
                        key=lambda r: (r[1], -r[4], r[0]))
    write("wrong_predictions.csv", ["file", "background", "true", "predicted",
                                    "times_wrong", "times_tested"], wrong_rows)

    log(f"\n== Mean +/- std over splits {args.splits} (std = sample std, ddof=1) ==")
    log(f"  {'method':24}{'scenario':9}{'accuracy':>18}{'macro F1':>18}")
    for method in list(FEATURE_FILES) + [EXTRA]:
        for scen in SCENARIOS:
            sel = [s for s in summary if s[0] == method and s[1] == scen]
            if not sel:
                continue
            a = np.array([s[3] for s in sel]); f = np.array([s[4] for s in sel])
            sd = (lambda v: v.std(ddof=1) if len(v) > 1 else float("nan"))
            log(f"  {method:24}{scen:9}{a.mean():>10.4f} +/- {sd(a):.4f}{f.mean():>10.4f} +/- {sd(f):.4f}")
    log(f"\nMain-method test images wrong at least once: {len(wrong_rows)} "
        f"(white {sum(r[1] == 'white' for r in wrong_rows)}, field {sum(r[1] == 'field' for r in wrong_rows)})")
    log(f"Time: {time.time() - t0:.0f} s")
    LOG_FILE.write_text("\n".join(_log_lines) + "\n")


if __name__ == "__main__":
    main()
