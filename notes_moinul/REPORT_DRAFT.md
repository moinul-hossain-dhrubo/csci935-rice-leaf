# Report draft: Moinul's sections

Page budgets (12pt, about 450 words per full page; figures and tables count):
- Section 5, implementation part: about 1 page (Anika writes the method part, 2 pages)
- Section 6, Experimental Results and Analysis: 2 pages (6.1, 6.2, 6.3)
- Appendix User Manual, part b (running the code): 1 page (Sabir writes part a)

No Python code in the report: modules and pseudocode only (brief, Section 5).
Every number comes from a results file, never typed from memory.

## 5 Implementation (draft, about 1 page; about 430 words + 1 algorithm box)

The method is implemented as a single Python 3.12 program organised in five
modules, run in order from one command (Appendix). It uses NumPy, OpenCV,
scikit-image, scikit-learn and PyTorch with torchvision. Every intermediate
result is saved under the dataset folder (`Dhan-Shomadhan/_interim/`), so each
module can be rerun on its own.

**M1 Data indexing and inspection.** Walks the two background folders and
assigns each image a disease and a background label from its folder names.
The supplied folder names contain misspellings and trailing spaces, so a fixed
spelling table maps each folder to one of the five classes, and the program
stops on any unknown folder rather than guessing. Labels are cross-checked
against the supplied CSV file and the file-name prefixes. Files are listed in
sorted name order so the list, and therefore every random split, is identical
on any operating system.

**M2 Split generator.** Creates five train/test splits (80%/20%) with fixed
seeds 0 to 4, stratified by the ten disease x background groups, and saves
them to one file that all later modules read. Both methods are therefore
tested on exactly the same images.

**M3 Baseline feature extractor.** Each image is read at quarter resolution,
rotated upright if landscape and resized to 128 x 256 pixels. Colour: HSV
histograms (16 hue, 8 saturation, 8 value bins) and YCrCb histograms (16 Cr,
16 Cb bins), each normalised to sum to one (64 values). Shape: HOG on the grey
image (9 orientations, 16 x 16-pixel cells, 2 x 2-cell blocks, 3780 values).

**M4 Deep feature extractor.** ResNet-18 with ImageNet weights, final
classification layer removed. Each image is converted to RGB, rotated upright,
resized to 224 x 448 pixels to keep its tall shape, and normalised with the
ImageNet channel means and standard deviations. The 512 values from the
global average pooling layer form the feature vector. The network is never
trained; it runs once over all images (on the Apple-silicon GPU when present).

**M5 Classifier training and evaluation** (Algorithm 1). For each feature type
and split, an SVM is tuned on the training part only, then tested once.

```
Algorithm 1: training and evaluation (one feature type F)
for split k = 0 .. 4:
    TRAIN, TEST <- images marked train / test in split k
    for each candidate (kernel, C, gamma) in the grid:
        score <- mean macro-F1 over 5 stratified folds of TRAIN
                 (standardise features using the training folds only, fit SVM,
                  predict the held-out fold)
    model <- standardiser + SVM with the best candidate, fitted on all of TRAIN
    P <- model.predict(F(TEST))                      # each test image once
    record accuracy, macro-F1, per-class scores and confusion counts on
        TEST-white, TEST-field and all of TEST
    if F is ResNet-18:                               # background-shift check
        repeat the tuning and fitting on the white images of TRAIN only;
        record scores on TEST-field
report mean and sample standard deviation over the 5 splits
```

The grid contains a linear SVM (5 values of C) and an RBF SVM (5 values of C x
4 values of gamma, set relative to 1 / number of features). The module writes
summary, per-class, confusion-matrix and per-image prediction files, from
which all tables and figures in Section 6 are produced.

(Runtimes to add from the final logs: M1 48 s, M3 20 s, M4 23 s on MPS
(`results/logs/01..04`); M5 from the final run.)

## 6 Experimental Results and Analysis (2 pages)

### 6.1 Experimental Settings

### 6.2 Results

### 6.3 Discussion of the Results

## Appendix b) Running the code (1 page; draft, to combine with Sabir's install page)

1. **Place the files.** Put `group_name.py` in a folder, and the unzipped
   dataset folder `Dhan-Shomadhan` (exactly as supplied, folder names unchanged)
   in the same folder.
2. **Open a terminal in that folder** and activate the environment from part a).
   Every command below is run from this folder.
3. **Check the ResNet-18 weights.** They download automatically (about 45 MB)
   the first time step 4 runs, from
   https://download.pytorch.org/models/resnet18-f37072fd.pth. Without internet,
   download that file elsewhere and place it in the PyTorch cache folder:
   macOS/Linux `~/.cache/torch/hub/checkpoints/`, Windows
   `C:\Users\<your name>\.cache\torch\hub\checkpoints\` (create the folders
   if needed).
4. **Run everything** (about N minutes in total on a 2024 MacBook):
   `python group_name.py all`
   Or one stage at a time, in this order: `inspect`, `splits`, `baseline`,
   `resnet`, `evaluate`.
5. **Find the results.**

| What | Where |
|---|---|
| Table in Section 6.2 (mean +/- std) | end of `results/logs/05_train_evaluate_log.txt` |
| Per-split scores | `results/summary.csv` |
| Per-class scores, confusion matrices | `results/per_class.csv`, `results/confusion_*.csv` |
| Every test prediction | `results/predictions.csv` |
| Chosen SVM settings | `results/best_params.csv` |
| Dataset statistics (Section 3) | `dataset-notes/` |
| Interim data (splits, features) | `Dhan-Shomadhan/_interim/` |

Notes: results are reproducible because the splits use fixed seeds and a
sorted file list. Deep features computed on a different device (CPU instead
of the Apple GPU) can differ in the last decimal places, which may change an
occasional prediction. (To confirm with Akram's clean-machine test, which runs
on a different computer.)
To rerun only the classifier on some splits: `python group_name.py evaluate --splits 0 1`.
