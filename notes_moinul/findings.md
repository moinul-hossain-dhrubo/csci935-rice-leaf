# Findings: code and experiments (Moinul)

For Anika and the report. Plain English, past tense. Every number is copied
from an output file (named in brackets). Step-by-step record:
`ANALYSIS_LOG.md`; every choice and its reason: `DECISIONS.md`.

Sections are filled as each stage runs.

## 1. Dataset facts (Stage 1)

- The dataset held 1106 images: 769 on a white background and 337 in the field
  (`dataset-notes/counts.csv`). Group sizes ranged from 49 (Field Brown Spot) to
  219 (White Sheath Blight).
- Labels from the folder names, the supplied CSV and the file names agreed for
  all 1106 images (`dataset-notes/01_inspect_log.txt`, section 2).
- All white images were 1952 x 4160 pixels; field images came in three sizes and
  11 were landscape (`dataset-notes/size_summary.csv`).
- Field images were far greener than white ones (median share of green pixels
  0.62 vs 0.058), showing how much surrounding vegetation the field photos
  contain (`01_inspect_log.txt`, section 6).
- No file was an exact copy of another, but some consecutively numbered field
  photos showed the same leaf twice, for example `rtf_17` and `rtf_18`
  (`dataset-notes/similar_pairs.csv`).
- Six hard-image notes with example files: `dataset-notes/dataset_notes.md`.

## 2. Experimental settings (for 6.1)

- Five splits, 80% training (884 images) / 20% test (222: 154 white, 68 field),
  stratified by disease x background, seeds 0 to 4 (`results/logs/02_splits_log.txt`).
- SVM settings were chosen by 5-fold cross-validation on the training part only;
  chosen per split in `results/best_params.csv` (main method: RBF, C = 10 in all
  5 splits).
- Full settings: `REPORT_DRAFT.md` 6.1 and `DECISIONS.md` D13 to D17.

## 3. Results (for 6.2)

From `results/logs/05_train_evaluate_log.txt` (run of 26 Oct with the D17 grid),
mean +/- std over 5 splits:
- The main method (ResNet-18 + SVM) reached accuracy 0.7467 +/- 0.0183 on white
  test images, 0.5794 +/- 0.0472 on field and 0.6955 +/- 0.0164 on mixed.
- The baseline (colour + HOG + SVM) reached 0.5000 +/- 0.0475 white,
  0.4118 +/- 0.0883 field and 0.4730 +/- 0.0394 mixed.
- The main method trained on white images only scored 0.2206 on field images in
  every split: it predicted Leaf Scald for every field image
  (`confusion_resnet18_svm_whiteonly_field.csv`).

## 4. Success and failure cases (for 6.3)

- Mean recall of the main method over 5 splits (`results/per_class.csv`):
  Sheath Blight 0.927 white / 0.677 field, Rice Blast 0.795 / 0.627, Rice Tungro
  0.733 / 0.707, Leaf Scald 0.579 / 0.440, Brown Spot 0.478 / 0.400.
- Most frequent confusions, main method, mixed, summed over 5 splits
  (`confusion_resnet18_svm_mixed.csv`): Leaf Scald predicted as Rice Blast 37
  times, Brown Spot as Leaf Scald 36, Leaf Scald as Brown Spot 24, Brown Spot
  as Rice Blast 24.
- 239 test images were misclassified by the main method at least once (139
  white, 100 field) (`wrong_predictions.csv`). Amit's review of 6 of them: to come.
