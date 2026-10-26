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

## 3. Results (for 6.2)

## 4. Success and failure cases (for 6.3)
