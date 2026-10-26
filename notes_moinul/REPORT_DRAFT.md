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

On a 16 GB M-series MacBook the modules took 48 s (M1), 1 s (M2), 20 s (M3),
23 s (M4, Apple GPU) and 186 s (M5, all splits)
(`dataset-notes/01_inspect_log.txt`, `results/logs/02..05`; to update from the
final `group_name.py all` run).

## 6 Experimental Results and Analysis (2 pages)

Draft of 26 Oct from the Step 6 run (`results/`); numbers to be re-checked
against the final `group_name.py all` run. Budget: 2 pages = Table 1 + Figure
(confusion matrices, 6 cm) + Figure (failure strip, 6 cm) + about 550 words.

### 6.1 Experimental Settings

**Data and splits.** All 1106 images were used. Five random splits (seeds 0
to 4) divided them 80% training (884 images) and 20% testing (222: 154 white,
68 field), stratified by the ten disease x background groups. There is no
separate validation set: settings were chosen by 5-fold cross-validation
inside the training part, so test images were never used for training or
tuning.

**Scenarios.** In each split one model was trained on the training images of
both backgrounds and tested once; its predictions were scored on (a) the white
test images, (b) the field test images and (c) all test images (mixed).

**Hyperparameters.** Baseline features: 128 x 256 image, HSV (16/8/8 bins) and
YCrCb (16/16 bins) histograms, HOG (9 orientations, 16 x 16 cells, 2 x 2
blocks); 3844 values. Main features: ResNet-18 (ImageNet weights, frozen),
224 x 448 input, 512 values. Classifier: standardisation then SVM; grid of
linear (C = 10^-4 to 1) and RBF (C = 1 to 10^4, gamma = 1/16 to 4 times
1/number of features) kernels, chosen by mean macro F1 over 5 folds. Chosen:
RBF, C = 10, gamma = 1/512 (4 splits) or 1/2048 (1) for the main method; RBF,
C = 100 (4 splits) or 1000 (1) for the baseline (`results/best_params.csv`).

**Experiments.** E1 baseline (colour + HOG + SVM) and E2 main method
(ResNet-18 + SVM), each in the three scenarios; E3 the main method trained on
white training images only and tested on field test images, to measure the
effect of the background. Metrics: accuracy and macro F1, mean and sample
standard deviation over the 5 splits.

### 6.2 Results

Table 1. Accuracy and macro F1, mean +/- standard deviation over 5 splits
(`results/logs/05_train_evaluate_log.txt`).

| Experiment | Test images | Accuracy | Macro F1 |
|---|---|---|---|
| E1 Colour + HOG + SVM | white | 0.500 +/- 0.048 | 0.453 +/- 0.052 |
| | field | 0.412 +/- 0.088 | 0.398 +/- 0.081 |
| | mixed | 0.473 +/- 0.039 | 0.439 +/- 0.040 |
| E2 ResNet-18 + SVM | white | **0.747 +/- 0.018** | **0.710 +/- 0.023** |
| | field | **0.579 +/- 0.047** | **0.570 +/- 0.060** |
| | mixed | **0.696 +/- 0.016** | **0.667 +/- 0.027** |
| E3 E2 trained on white only | field | 0.221 +/- 0.000 | 0.072 +/- 0.000 |

Figure X (Akram): confusion matrices of E2 summed over the 5 splits, (a) white,
(b) field, (c) mixed.

### 6.3 Discussion of the Results

**Expectations.** The main method outperformed the baseline in every scenario,
by about 22 points of accuracy on mixed test images, and varied less between
splits (standard deviation 0.016 vs 0.039). This matches the expectation that
features learned from 1.2 million photographs describe lesion texture better
than fixed colour counts and edge directions. Absolute accuracy (0.696 mixed)
was lower than hoped: a frozen network sees each leaf only at 224 pixels wide,
and the classifier learns from about 880 images.

**Successful cases.** Sheath Blight was recognised best (mean recall 0.927 on
white images, `per_class.csv`), followed by Rice Blast (0.795) and Rice Tungro
(0.733): their large pale lesions, dense dark spots and orange-yellow colour
are visible at the reduced resolution.

**Failure cases.** Brown Spot (recall 0.478 white, 0.400 field) and Leaf Scald
(0.579, 0.440) failed most. Summed over the splits, Brown Spot test images were
predicted as Leaf Scald 36 times and Rice Blast 24 times, and Leaf Scald as Rice
Blast 37 times (`confusion_resnet18_svm_mixed.csv`). These diseases share small
brown spots and tip discolouration (Section 3), which shrink to a few pixels
after resizing. [Amit's six failure cases and reasons go here, Figure Y.]

**Challenging cases: field images.** Accuracy fell from 0.747 on white to
0.579 on field images for the same model. In field photos the diseased leaf is
surrounded by healthy leaves, soil and water, so most of the image (and of the
pooled features) describes the background. E3 shows how strong this effect is:
a model trained on white images only predicted a single class (Leaf Scald) for
every field image in every split. The field features lay far from all white
training examples, so the RBF SVM had no basis to separate them. Field
results also varied most between splits (0.047), because only 68 field images
are tested per split.

**Limitations.** Some field leaves were photographed twice with slightly
different framing (for example rtf_17 and rtf_18, Section 3); when the two
copies fall on different sides of a split, the test score may be slightly
optimistic.

**Improvements to the method.** (1) Segment the leaf from the background
(colour thresholding or a segmentation network) before extracting features,
directly targeting the field-image gap shown by E3. (2) Describe the leaf at
higher resolution, for example by extracting features from several
overlapping tiles along the leaf and pooling them, so small Brown Spot and
Rice Blast lesions keep more pixels. (3) Fine-tune the last ResNet block on
the training images, so the features adapt to rice lesions rather than
ImageNet objects. (4) Weight classes in the SVM to counter the imbalance that
penalises Brown Spot.

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
