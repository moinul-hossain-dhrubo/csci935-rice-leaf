# Learnings (Moinul, code)

A study companion for the stage files in `code/`. One section per stage: the
big idea, the key concepts with everyday analogies, what to say in a demo, and
check-yourself questions (answers at the end of each section). Read the
section before running the stage, then again with its output.

---

## L1. Stage 1 (`01_inspect.py`): looking at the data before using it

### The big idea: silent failures
Code bugs usually crash with an error. Data bugs usually do not: they give a
believable but wrong number. If the misspelled folder `Browon Spot` were read
as its own class, the code would happily train a 6- or 7-class model, print an
accuracy, and nobody would notice. Inspection turns each hidden assumption
("labels are right", "no image appears twice", "all images are similar in
size") into a checked fact, before any model is trained. It also feeds
Section 3: the brief asks for "an in-depth analysis of the characteristics of
the images and the challenges", and that analysis must support the method.

### Concept 1: where the labels come from
Analogy: three people label the same parcel (the folder it sits in, a
shipping list, and the code written on the box). If all three agree on every
parcel, you can trust the labels. Here the three sources are the folder name,
the supplied CSV (`Dhan-Shomadhan_picture_Information.csv`) and the file-name
prefix (`bsf_` = Brown Spot field, `sb_wb_` = Sheath Blight white).
`common.py` maps every known misspelling to one of the 5 classes and stops
with an error on any spelling it does not know. Stopping is on purpose: a
loud error is better than a quiet guess.

### Concept 2: hidden trailing spaces
`"Field Background  "` has two spaces at the end that you cannot see in
Finder. Code that looks for `"Field Background"` finds nothing. The code
strips spaces before comparing, so it works on the original folder exactly as
the marker will have it.

### Concept 3: counts and imbalance
Analogy: a class photo where one row has 219 students and another 49. If you
pick 20% of the photo at random, the small row might end up with very few
people. That is why the splits are stratified by disease x background (D4):
each of the 10 groups gives 20% of its own images to the test part.
Imbalance also matters for the classifier: a lazy model can score well by
favouring big classes. Macro F1 (D11) treats every class equally, so it
exposes that.

### Concept 4: image size and shape
ResNet-18 expects small square inputs (224 x 224 pixels). A tall
1952 x 4160 photo shrunk straight to a square gets squashed sideways, and a
small lesion may shrink to a few pixels. Knowing the real sizes decides how
Stage 3 resizes: keep the shape and pad, or crop, or squash. The inspection
also counts portrait versus landscape photos, because a sideways photo of a
leaf looks very different to a network.

### Concept 5: simple quality numbers
- **Brightness:** mean of the HSV "value" channel (0 dark to 255 bright).
- **Contrast:** spread of grey levels (standard deviation).
- **Sharpness:** variance of the Laplacian (a filter that responds to edges).
  Higher = more edges. Careful: a plain white background has few edges, so a
  white image scores low even when the leaf is in focus. Compare sharpness
  within one background, not between them.
- **white_frac / green_frac:** share of near-white pixels (paper) and of green
  pixels (leaves and grass). On a white image, a high white_frac means the
  leaf fills little of the frame. On a field image, a high green_frac means
  lots of other plants around the diseased leaf: clutter.

### Concept 6: duplicates and near-duplicates (leakage)
Analogy: an exam where one question is the same as a homework question with
the numbers barely changed. A student who memorised the homework gets it
right without understanding. If the same leaf is photographed twice, and one
photo lands in training and the other in testing, the model can "recognise"
the leaf instead of the disease, and the test score is too optimistic. This is
called **leakage**.
- **Exact copies** are found by comparing a fingerprint of the file bytes (MD5).
- **Near copies** are found by shrinking each image to a 24 x 48 grey
  thumbnail and measuring how alike two thumbnails are (correlation: 1.0 =
  identical). A first attempt with a 9 x 8 "hash" flagged hundreds of
  different-disease white pairs: at that size every thin leaf on paper looks the
  same. Lesson: a check also needs checking, by looking at what it flags.

### Concept 7: interim data and contact sheets
Contact sheets (grids of small thumbnails) are for people to look at; the
methods never use them. They are interim data, so they live under
`Dhan-Shomadhan/_interim/inspect/` (brief rule (b), D10).

### What to say in a demo
"Before training anything we inspected the dataset with a script. It checked
that the folder names, the supplied CSV and the file names give the same label
for all 1106 images, despite misspelled folder names. It counted images per
disease and background, which showed the white set is more than twice the
field set and Field Brown Spot is the smallest group, so we stratified our
splits by both disease and background. It measured image sizes, which decided
how we resize for ResNet-18, and it looked for repeated images, finding pairs
of field photos of the same leaf taken seconds apart."

### Check yourself
1. Why does the code stop with an error on an unknown folder name instead of
   skipping that folder?
2. A white image has sharpness 30 and a field image 900. Is the white image blurry?
3. Two photos of the same diseased leaf, one in training and one in testing:
   does that make the reported accuracy higher or lower than it should be? Why?
4. Why are the contact sheets saved under the dataset folder and not in `results/`?
5. Field Brown Spot has 49 images. About how many land in each test part with
   an 80/20 split, and why does that make its per-class score jumpy?

### Answers
1. Skipping would silently drop images (and maybe a whole class); the results
   would still print and look fine. An error forces someone to look.
2. Not necessarily. The Laplacian responds to edges, and a plain background has
   almost none. Compare white images with white images.
3. Higher than it should be: the model may match the leaf it has already seen
   rather than recognise the disease, so the test is partly a memory test.
4. Brief rule (b): interim data must sit under the main dataset folder and be
   made by our code. They are not results.
5. About 10 (20% of 49). One wrong image moves that class's recall by about
   10 percentage points, so it swings a lot between splits.

### What the output showed
(To be written after the run on the Mac, from `dataset-notes/01_inspect_log.txt`.)
