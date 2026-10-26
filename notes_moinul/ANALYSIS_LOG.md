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
