# Analysis log (Moinul, code)

Numbered steps in the order they were run. Each step: stage file, what was
done, key numbers (copied from the output file named, never retyped from
memory), and what they mean. Test-part runs are scored once per split with
settings fixed beforehand (D7).

---

## Step 0 (25 Oct): dataset received and unzipped

Source: lecturer's OneDrive folder linked in the brief (`Dhan-Shomashan.zip`,
1,960,579,997 bytes), unzipped to `code/Dhan-Shomadhan/` (2.1 GB).
Zip listing (Python `zipfile`): 1106 `.jpg` files and 1 `.csv`, in 2
background folders x 5 disease folders. Folder names are misspelled and two
have trailing spaces (see D9). Git ignores the folder (`git status` clean).

Meaning: the image count matches the brief (1106). Labels must be read through
an explicit spelling map, checked in Step 1.

## Step 1 (26 Oct): dataset inspection, run on the Mac

Stage: `01_inspect.py`. Output: `dataset-notes/01_inspect_log.txt`, `counts.csv`,
`size_summary.csv`, `image_stats.csv`, `similar_pairs.csv`; contact sheets in
`Dhan-Shomadhan/_interim/inspect/`. Run time 48 s.

- Images found 1106; CSV rows 1106; folder/CSV mismatches 0; images not in CSV 0;
  file-name prefix odd ones out 0; unreadable 0 (log sections 2, 4).
- Counts (white / field): Brown Spot 90 / 49, Leaf Scald 143 / 74, Rice Blast
  198 / 74, Rice Tungro 119 / 76, Sheath Blight 219 / 64; totals 769 / 337 (`counts.csv`).
- Sizes: white all 1952 x 4160 (769 portrait). Field: 320 at 1952 x 4160,
  11 at 4160 x 1952 (landscape), 6 at 1536 x 3264 (`size_summary.csv`, log section 5).
- Quality, median [min, max] (log section 6):
  white brightness 159 [133, 191], sharpness 84.3 [21.6, 258], green_frac 0.058 [0, 0.245];
  field brightness 141 [99.8, 172], sharpness 1.06e+03 [55.8, 4.57e+03], green_frac 0.62 [0.059, 0.978].
- Duplicates: 0 exact groups; 0 pairs at thumbnail correlation >= 0.98; highest 0.976
  (white pair rb_wb_108 / rb_wb_99, different leaves on inspection). Most similar
  field pair rtf_17 / rtf_18 at 0.907: same leaf, two shots (log section 7).

Meaning: labels are safe to build on. The 10 groups range from 49 to 219
images, which confirms stratifying by disease x background (D4) and reporting
macro F1 (D11). Every image is tall and narrow (about 1 : 2.1), so the
ResNet-18 resize choice (squash, crop or pad to 224 x 224) needs deciding in
Stage 3. Same-leaf pairs exist in the field set; the thumbnail measure misses
some (bsf_7 / bsf_8, seen by eye), so their effect will be measured with the
ResNet-18 features later and reported as a limitation.

## Step 2 (26 Oct): five stratified 80/20 splits

Stage: `02_splits.py`. Output: `Dhan-Shomadhan/_interim/splits.csv`,
`results/logs/02_splits_log.txt`.

- Every split: train 884, test 222 (white 154, field 68).
- Test images per group, identical in all 5 splits: Brown Spot 18 white / 10 field,
  Leaf Scald 28 / 15, Rice Blast 40 / 15, Rice Tungro 24 / 15, Sheath Blight 44 / 13.
- Times each image is a test image across the 5 splits: 0: 367, 1: 436, 2: 242,
  3: 54, 4: 7, 5: 0. Identical splits: none.
- The Mac run reproduced the sandbox (Linux) run exactly.

Meaning: stratification worked: every test part has the same group mix. The
smallest test group is Field Brown Spot (10 images), so one error there moves
its recall by 10 points. 367 images are never tested, as expected for
independent random splits (0.8^5 = 0.33 of 1106 = about 363); not a problem,
since each split is a separate fair test. Same output on two machines confirms
the fixed seed plus sorted file list is reproducible (D12).

## Step 3 (26 Oct): baseline features

Stage: `03_baseline_features.py`. Output: `Dhan-Shomadhan/_interim/features_colour_hog.npz`,
`results/logs/03_baseline_features_log.txt`.
- 1106 images x 3844 features (colour 64, HOG 3780); no NaN/inf; each of the 5
  colour histograms sums to 1; HOG values 0.000 to 0.860, mean 0.123; 20 s.

## Step 4 (26 Oct): ResNet-18 features

Stage: `04_resnet_features.py`. Output: `Dhan-Shomadhan/_interim/features_resnet18.npz`,
`results/logs/04_resnet_features_log.txt`.
- Device mps, torch 2.14.1, weights IMAGENET1K_V1, input 224 x 448.
- 1106 images x 512 features; no NaN/inf; values 0.000 to 8.463, mean 0.593;
  share of exact zeros 0.018; 23 s.

Meaning: both feature files are complete and clean. Values are never negative
because ResNet's last block ends in a ReLU before pooling.

## Step 5 (26 Oct): SVM training and evaluation, all 5 splits

Stage: `05_train_evaluate.py`. Output: `results/summary.csv`, `per_class.csv`,
`confusion_*.csv`, `predictions.csv`, `best_params.csv`, `wrong_predictions.csv`,
`results/logs/05_train_evaluate_log.txt`. Run time 99 s.

Mean +/- std over 5 splits (log, last table):

| method | scenario | accuracy | macro F1 |
|---|---|---|---|
| colour_hog_svm | white | 0.5078 +/- 0.0424 | 0.4572 +/- 0.0460 |
| colour_hog_svm | field | 0.4147 +/- 0.0866 | 0.4005 +/- 0.0798 |
| colour_hog_svm | mixed | 0.4793 +/- 0.0333 | 0.4436 +/- 0.0324 |
| resnet18_svm | white | 0.7467 +/- 0.0183 | 0.7099 +/- 0.0233 |
| resnet18_svm | field | 0.5794 +/- 0.0472 | 0.5699 +/- 0.0602 |
| resnet18_svm | mixed | 0.6955 +/- 0.0164 | 0.6674 +/- 0.0269 |
| resnet18_svm_whiteonly | field | 0.2206 +/- 0.0000 | 0.0723 +/- 0.0000 |

- Chosen settings (`best_params.csv`): baseline RBF, C = 100, gamma = 6.50e-05
  (= 1/(4 x 3844)) in all 5 splits; main RBF, C = 10 in all 5 splits, gamma
  0.001953 (= 1/512) in 4 splits and 0.000488 in 1.
- White-only model (`confusion_resnet18_svm_whiteonly_field.csv`): predicted
  Leaf Scald for every one of the 340 field test predictions (68 x 5); 0.2206 =
  15/68, the share of Leaf Scald among field test images.
- Main method, mean recall per class (`per_class.csv`, averaged over splits):
  white: Brown Spot 0.478, Leaf Scald 0.579, Rice Blast 0.795, Rice Tungro 0.733,
  Sheath Blight 0.927; field: 0.400, 0.440, 0.627, 0.707, 0.677.
- Main method, white, summed confusion (`confusion_resnet18_svm_white.csv`):
  Brown Spot 43 right, 21 as Rice Blast, 17 as Leaf Scald; Leaf Scald 81 right,
  28 as Rice Blast, 17 as Brown Spot.
- Main-method test images wrong at least once: 239 (white 139, field 100).

Meaning:
1. ResNet-18 features beat the hand-made baseline by about 22 points of
   accuracy on mixed test images (0.6955 vs 0.4793), and are more stable
   (std 0.0164 vs 0.0333).
2. Field is clearly harder for both methods: main method white 0.7467 vs field
   0.5794. Field std is larger (0.0472), partly because only 68 field images
   are tested per split.
3. The hardest classes are Brown Spot and Leaf Scald, and their main confusions
   (Brown Spot with Rice Blast and Leaf Scald) match dataset note 2 (similar
   spots and discolouration).
4. White-only training collapsed to a single class on field images: field
   features lie far from every white training image. An RBF SVM gives the same
   answer to all points far from its training data, so the collapse shows a
   large background shift, and the exact class it falls to (Leaf Scald) carries
   no meaning.
5. The baseline's C sits at the largest value tried (100) and its gamma at the
   smallest in every split: by the D15 rule the grid should be widened, so the
   baseline is not under-tuned.
