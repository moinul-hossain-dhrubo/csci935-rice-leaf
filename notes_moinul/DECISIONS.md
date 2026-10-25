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
