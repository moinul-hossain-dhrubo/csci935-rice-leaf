# Decisions (Moinul, code and Section 6)

Each choice in the code and experiments, the reason, and the alternative
rejected. Numbered D1, D2, ... in the order they were made. Group-level
planning decisions (dates, roles) stay in `admin/decisions.md`.

---

**D1. Main method: frozen pretrained ResNet-18 as a feature extractor, then an SVM.**
Choice: torchvision ResNet-18 with ImageNet weights, final classification
layer removed, so each image becomes 512 numbers (the output of the global
average pooling layer). The network is never retrained.
Why: 1106 images is far too few to train a deep network from scratch, but a
network already trained on about 1.2 million everyday photos has learned
general visual patterns (edges, textures, colour blobs) that transfer to leaf
lesions. Runs in minutes on a laptop, no GPU needed, low risk on a short
timeline. Decided 14 Oct (`admin/decisions.md`).
Alternative rejected: fine-tuning ResNet-18 (retraining its weights on our
images). Often more accurate in the literature, but slower, needs careful
tuning, and is out of scope for this timeline.

**D2. Baseline: colour histograms (HSV and YCrCb) plus HOG, then an SVM.**
Why: a hand-designed baseline shows how much the learned ResNet-18 features
add. Colour matters because several diseases differ mainly in lesion colour
(brown spots versus yellow-orange Tungro); HOG captures lesion shape and leaf
edges.
Alternative rejected: no baseline (then there is nothing to compare the main
method against in 6.3); other hand-made features such as LBP texture (more
features to describe in a tight page budget, no clear gain expected).

**D3. The same classifier (SVM) for both methods.**
Why: with the classifier fixed, any difference in results comes only from the
features, which makes the comparison fair and easy to explain. SVMs work well
with few training examples and long feature vectors, and train in seconds.
Alternative rejected: also trying kNN, random forest or logistic regression
(each adds results and page space without answering the client's question).

**D4. Evaluation: 5 seeded random splits, 80% train / 20% test, stratified by disease x background.**
Why: the brief requires an average over 5 random splits with the same
percentages, reported as mean and standard deviation. Stratifying by the 10
disease x background groups (not only the 5 diseases) keeps the white/field
mix the same in every split. This matters because the groups are unequal
(Field Brown Spot has 49 images in the zip listing; White Sheath Blight 219).
Seeds are fixed so every run reproduces the same splits.
Alternative rejected: stratify by disease only (a split could by chance get
very few field images of one disease in its test part); k-fold
cross-validation (the brief asks for random splits with the same percentages).

**D5. Scenario design: one model per split, trained on both backgrounds, scored three ways.**
Choice: in each split one model is trained on all training images (white and
field). It predicts every test image once; accuracy is then counted on the
white test images (scenario a), the field test images (b) and all test images
(c, mixed).
Why: (1) the client wants one method that handles either kind of photo;
(2) the same model sits all three "exams", so a gap between white and field
comes from the background alone; (3) results table stays small (2 methods x
3 scenarios) inside the 2-page limit. The brief allows this as long as test
images are never used for training and the report states it.
Alternative rejected: a separate model per background (option B). The
field-only model would learn from about 270 images, so a low field score
could come from too little training data rather than the background, and 6.3
could not separate the two.

**D6. One extra experiment: main method trained on white images only, tested on field images.**
Why: gives 6.3 direct evidence of whether the background changes what the
features "see" (a background shift), and supports suggesting background
removal (segmenting the leaf) as an improvement. Costs seconds of compute and
one table row.
Rule: this is the first thing dropped if Section 6 runs over 2 pages or time
runs short; the baseline is dropped next (scope cut in `admin/plan-private.md`).
Alternative rejected: options B and D (per-background models, and comparing
them with D5's model). More rows and the data-size problem muddies the reading.

**D7. SVM settings tuned by cross-validation inside the training part only.**
Why: test images must never influence any choice, otherwise the test score is
no longer an honest measure (brief, Section 6). With about 885 training images
per split, a separate validation split would leave too few images per group,
so tuning uses stratified cross-validation on the training part.
Alternative rejected: a fixed validation split (too small); tuning on the test
part (not allowed).

**D8. Code written as separate stage files, merged into one `group_name.py` on Tue 27 morning.**
Choice: one file per stage, plus `common.py` holding the shared settings
(paths, label mapping, seeds) so every stage reads data the same way. All files
use the same function names and paths so the merge is mostly copy and paste.
Why: Moinul's preference: easier to build and test one stage at a time.
Merge set for Tue morning (not evening) so any merge bug shows up before the
User Manual goes to Akram for the clean-machine test.
Alternative rejected: one file with stage commands from the start (no merge
step, but less separation while building).

**D9. Labels come from the folder names through an explicit mapping, checked against two other sources.**
The dataset folders are misspelled and have hidden trailing spaces
(`Field Background  ` with two spaces, `White Background ` with one;
`Browon Spot`, `Leaf Scaled`, `Rice Turgro`, `Shath Blight`). `common.py` maps
each known spelling to one of the 5 classes and stops with an error on any
unknown folder name, rather than guessing. Stage 1 checks the folder label
against the supplied CSV (`Dhan-Shomadhan_picture_Information.csv`) and the
file name prefix (`bsf_`, `sb_wb_`, ...) for every image.
Why: the marker runs the code on the original folder, so the code must read it
as it is. A silent label mistake would give believable but wrong results.
Alternative rejected: renaming the folders (the marker's copy would not be
renamed); trusting the CSV alone (one source, nothing to check it against).

**D10. Interim data goes under `Dhan-Shomadhan/_interim/`.**
Why: brief rule (b): interim data must be under the main dataset folder and
created by our Python code. Cached features, saved splits and inspection
contact sheets go there. The image-listing code only scans the two background
folders, so files in `_interim/` are never mistaken for dataset images.

**D11. Metrics: accuracy and macro F1, each as mean +/- std over the 5 splits.**
Why: accuracy is what the client understands. The classes are unequal in
size, so macro F1 (the average of the per-class F1 scores, each class counting
equally) shows whether small classes such as Field Brown Spot are being missed.
Per-class precision, recall and F1, and confusion matrices summed over splits,
are also saved for 6.3 and Akram's figures (`results/README.md`).

**D12. Splits made once by Stage 2 and saved to `Dhan-Shomadhan/_interim/splits.csv`; every later stage reads that file.**
Why: both methods must sit exactly the same exam in each split. One saved
file removes any chance of a stage drifting to a different split. Files are
listed sorted by their name string (not by path object), because Windows
compares paths ignoring upper/lower case; this keeps the list order, and so
the seeded split, identical on Mac and on the marker's Windows machine.
Split made with scikit-learn `train_test_split(test_size=0.2,
stratify=disease|background, random_state=seed)`, seeds 0 to 4.
Alternative rejected: each stage splitting on the fly with the same seed
(works only if every stage lists files identically, a silent risk).

**D13. Baseline image preparation: upright, 128 x 256, features of the whole image.**
Landscape photos (11, all field) are rotated to portrait, since leaves in the
dataset run vertically. Images are resized to 128 x 256 (w x h), close to the
1 : 2.1 shape of the photos, so leaves are barely distorted. Colour: HSV
(H 16, S 8, V 8 bins) and YCrCb (Cr 16, Cb 16 bins), each normalised to sum
to 1: 64 numbers. HOG: 9 orientations, 16 x 16 cells, 2 x 2 blocks, L2-Hys:
3780 numbers. Total 3844. Read at 1/4 resolution for speed (22 s for all
images in testing).
Why: a standard, simple hand-made descriptor of colour (lesion colour) and
shape (edges), cheap to compute and easy to describe in pseudocode.
Alternative rejected: segmenting the leaf from the background first (better
features, but that is an improvement to discuss in 6.3, not a baseline);
larger HOG images (thousands more numbers for 884 training images, more
risk of overfitting).

**D14. ResNet-18 input: upright, 224 x 448 (w x h), RGB, ImageNet normalisation; 512 features from global average pooling.**
Why: every photo is tall and narrow (about 1 : 2.1, Step 1). ResNet-18 ends in
global average pooling, so it accepts non-square input. 224 x 448 keeps the
leaf's shape and keeps its width at the 224-pixel scale the network was
trained on. Weights: torchvision `ResNet18_Weights.IMAGENET1K_V1`, final
layer replaced by identity, `eval()` mode, no gradients. Runs on MPS when
available (device logged).
Alternatives rejected: squashing to 224 x 224 (leaf distorted to half its
width relative to its length, lesions squashed); centre-cropping a 224 x 224
square (loses about half of each leaf, possibly the lesions); padding to a
square (leaf becomes a narrow strip, fewer pixels on the disease).

**D15. SVM tuning: grid search with 5-fold CV on the training part, scored by macro F1; std reported as sample std.**
Pipeline: StandardScaler (fitted inside each fold, so on training folds only)
then SVC. Grid: linear kernel, C in {0.001, 0.01, 0.1, 1}; RBF kernel, C in
{1, 10, 100}, gamma in {0.25, 1, 4} x 1/n_features (1/n_features is
scikit-learn's "scale" default for standardised features). Folds stratified by
disease x background, shuffled with the split's seed. The best setting is
refitted on the whole training part, then the test part is predicted once.
Macro F1 as the tuning score because the classes are unequal (D11): it stops
the search from favouring settings that ignore small classes.
Mean +/- std over the 5 splits uses the sample standard deviation (ddof = 1).
Alternative rejected: tuning by accuracy (rewards favouring large classes); a
larger grid (more time, little expected gain; widen only if the chosen
value sits at the edge of the grid in most splits).

**D16. Results files extend the `results/README.md` contract.**
Confusion CSVs carry a header row and a first column of class names (same
order as the README), so they are readable by eye; `plots.py` should skip the
first row and column. Extra files: `predictions.csv` (every test prediction),
`best_params.csv`, `wrong_predictions.csv` (main method's mistakes per image,
with how many of its test appearances were wrong, for Amit's A3).

**D17. SVM grid widened after the first full run (training-only decision).**
In the first run (ANALYSIS_LOG Step 5) the baseline chose C = 100 (largest
tried) and gamma = 1/(4 x n_features) (smallest tried) in all 5 splits, so its
best setting may lie outside the grid. New grid, used for BOTH methods:
linear C in {0.0001, 0.001, 0.01, 0.1, 1}; RBF C in {1, 10, 100, 1000, 10000},
gamma in {1/16, 1/4, 1, 4} x 1/n_features (25 settings).
Why this is not tuning on the test set: the trigger was the CHOSEN settings
(picked by cross-validation on training parts), not the test scores, and
the grid is the same for both methods. The first run's numbers stay in
ANALYSIS_LOG Step 5 for the record; the report uses the rerun.
Alternative rejected: keeping the narrow grid (the baseline would look
worse than it is, an unfair comparison).

**D18. Submission file `group_name.py` built on 26 Oct evening (merge moved earlier than D8's Tue morning).**
One file, five stage functions, run with `python group_name.py all` or one
stage name. Stage files 01 to 05 stay in the repo as the development
history; from now on changes are made in `group_name.py` (and mirrored in a
stage file only if that stage is rerun separately). Checked in the sandbox:
splits file byte-identical, baseline features identical (max difference 0.0)
and split-0 main-method results identical to the stage files.
Akram's `plots.py` is merged on Tue when it arrives. Renamed to the real
group name once known.

**D19. Group name: Group 24 (Moodle). Submission files `group24.py`, `group24.pdf`, `group24.zip`.**
Lowercase, no spaces, so the marker can type it on any system. `group_name.py`
renamed to `group24.py` (git history kept); README gained the run commands.

**D20. Pinned package versions in `requirements.txt`; OpenCV 4.13 (the brief's version) instead of opencv-contrib-python 5.0.**
Versions are those of the Mac environment that produced the results (Python
3.12.12, numpy 2.5.4, matplotlib 3.11.2, scikit-learn 1.9.1, scikit-image
0.26.0, torch 2.14.1, torchvision 0.29.1), read from `.venv` on 26 Oct.
OpenCV: the brief permits "OpenCV 4.13"; the code needs no contrib modules.
Tested in the sandbox: with opencv-python 4.13.0.92 the baseline features are
identical to the Mac's OpenCV 5.0 features (max difference 0.0) and the
inspection gives the same counts and checks.
torchvision is required for the ResNet-18 model and weights; it is part of the
PyTorch project, and the User Manual states it explicitly.
Alternative rejected: unpinned packages (a future release could change results
or break the code on the marker's machine).

**D21. Akram's `plots.py` merged into `group24.py` as stage 6 (`plots`); figure images named in the code and read from the dataset folder.**
Layout, sizes and styling kept exactly as Akram wrote them (18 cm wide, at most
6 cm tall, 9 pt, no titles; confusion colour = share of the true class, number =
count). Change: the sample grid and failure strip no longer read the repo
folders `figures/samples/` and `figures/failures/`, which are not in the
submission zip; instead `SAMPLE_FILES` (10 names, from Amit's
`originals.txt`) and `FAILURE_FILES` (6 names, from Amit's `failures.txt`)
list the dataset files, read straight from `Dhan-Shomadhan/`. Until those
lists are filled, the two figures are skipped with a message, so
`python group24.py all` never crashes on the marker's machine. Images are read
with OpenCV instead of matplotlib's reader (same pixels, one fewer dependency
path). Tested in the sandbox with temporary example picks: all three figures
render correctly. `plots.py` stays in the repo as Akram's original.

**D22. User Manual installs packages by name and version, not from `requirements.txt`; figure image lists filled.**
The marker receives only `group24.zip` (`group24.py` + `group24.pdf`), so the
manual cannot rely on `requirements.txt` or the repository: step 6 is one
`pip install` line with every package and version (D20). Sabir's install page
(S2) kept otherwise: Python 3.12 link, Mac/Windows table, `pip check`, weights
location, common problems. Combined with the run part into
`docs/user_manual.docx` (2 pages at 10.5 pt). `SAMPLE_FILES` and
`FAILURE_FILES` in `group24.py` filled from Amit's `originals.txt` and
`failures.txt`; `python group24.py plots` rebuilds all three figures from the
dataset alone (tested in the sandbox).
