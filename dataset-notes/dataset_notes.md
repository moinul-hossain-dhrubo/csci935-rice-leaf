# Dataset inspection notes (for Section 3)

By Moinul, 26 Oct. Every number below is copied from the files in this folder,
produced by `01_inspect.py` (log: `01_inspect_log.txt`). Visual observations
were made on the contact sheets in `Dhan-Shomadhan/_interim/inspect/`; open
the named full-size images to confirm before quoting them.

## Counts per disease x background (`counts.csv`)

| Disease | White | Field | Total |
|---|---|---|---|
| Brown Spot | 90 | 49 | 139 |
| Leaf Scald | 143 | 74 | 217 |
| Rice Blast | 198 | 74 | 272 |
| Rice Tungro | 119 | 76 | 195 |
| Sheath Blight | 219 | 64 | 283 |
| **Total** | **769** | **337** | **1106** |

- White images outnumber field images by more than 2 to 1 (769 vs 337).
- Classes are unequal: largest group White Sheath Blight (219), smallest Field Brown Spot (49).
- Labels were checked three ways (folder name, supplied CSV, file-name prefix):
  all 1106 images agree, none unreadable, no exact duplicate files (`01_inspect_log.txt`, sections 2, 4, 7).
- The supplied folder names are misspelled (`Browon Spot`, `Leaf Scaled`,
  `Rice Turgro`, `Shath Blight`) and two have trailing spaces; the code maps
  them to the 5 correct names.

## Image sizes (`size_summary.csv`)

- White: all 769 images are 1952 x 4160 pixels (width x height), portrait.
- Field: 1536 to 4160 pixels wide, 1952 to 4160 high; 320 are 1952 x 4160,
  11 are landscape (4160 x 1952), 6 are 1536 x 3264.
- All images are tall and narrow (about 1 : 2.1). The pretrained network
  expects 224 x 224 squares, so resizing must either squash, crop or pad.

## Six things that make the images hard (with examples)

1. **Field images are cluttered: the diseased leaf is a small part of the frame.**
   Many healthy leaves surround it, and the lesion may be one pale streak.
   Examples: `sbf_5.jpg`, `sbf_17.jpg`, `sbf_32.jpg` (Sheath Blight). Measured:
   median share of green pixels 0.62 in field images vs 0.058 in white images
   (`01_inspect_log.txt`, section 6).

2. **Some diseases look alike.** Rice Blast and Brown Spot both show small
   brown or dark spots (`rb_wb_92.jpg` vs `bs_wb_64.jpg`); Leaf Scald and Rice
   Tungro both show yellow-orange discolouration along the leaf
   (`ls_wb_61.jpg` vs `rt_wb_39.jpg`). Colour alone will not separate them.

3. **Lesions can be tiny relative to the image.** On white images the leaf is a
   thin vertical strip, and some lesions are barely visible at thumbnail size
   (`rb_wb_68.jpg`, `rb_wb_15.jpg`). Shrinking a 1952 x 4160 photo to 224 x 224
   reduces it about 9 times in width and 19 times in height (arithmetic from the
   sizes above), so small spots may shrink to a pixel or two.

4. **The "white" background is not uniform.** Paper tone ranges from bright
   white to mid-grey, with shading gradients. Share of near-white pixels ranges
   from 0 to 0.877 across white images (`01_inspect_log.txt`, section 6);
   compare `rb_wb_108.jpg` (0.738) with `rb_wb_15.jpg` (0.001) in `image_stats.csv`.

5. **Field lighting, ground and framing vary.** The darkest field images include
   wet soil, water reflections and dead plant material (`rtf_9.jpg`,
   `rtf_39.jpg`; brightness 110.9 and 110.8 vs field median 141), and 11 field
   photos are landscape rather than portrait (`rtf_26.jpg`, brightness 99.8,
   the darkest field image). Field brightness ranges 99.8 to 172.

6. **The same leaf sometimes appears twice.** Consecutively numbered field photos
   can show the same leaf from a slightly different angle: `rtf_17.jpg` /
   `rtf_18.jpg` (thumbnail correlation 0.907, the most similar field pair),
   `rbf_21.jpg` / `rbf_22.jpg` (0.832), `bsf_4.jpg` / `bsf_5.jpg` (0.751), and
   `bsf_7.jpg` / `bsf_8.jpg` (seen by eye; not caught by the measure)
   (`similar_pairs.csv`). If one copy lands in training and the other in
   testing, the test score can be slightly optimistic.

## What this means for the method (for Section 3's argument)

- Small, unequal dataset (49 to 219 images per group): too few to train a deep
  network from scratch; supports using a pretrained network as a fixed feature
  extractor plus an SVM.
- Field clutter and similar-looking diseases: hand-made colour features are
  likely to struggle; learned features that capture texture and shape should
  help. This is what the baseline vs main comparison tests.
- Unequal groups: splits are stratified by disease x background and results
  include macro F1, not only accuracy.
