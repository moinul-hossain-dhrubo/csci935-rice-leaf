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

### What the output showed (26 Oct, `dataset-notes/01_inspect_log.txt`)
1. **Labels are clean.** All three sources agree for all 1106 images, so the
   misspelled folders were mapped correctly. Nothing to fix.
2. **The groups are very unequal:** from 49 (Field Brown Spot) to 219 (White
   Sheath Blight). With an 80/20 split, Field Brown Spot has about 10 test
   images per split, so its score will jump around between splits.
3. **Every white image has exactly the same size** (1952 x 4160), field images
   come in 3 sizes and 11 are sideways. Same camera setup for white, more
   casual shooting in the field.
4. **Sharpness, read correctly:** white median 84.3, field median about 1060.
   This is the check-yourself question 2 trap: the white numbers are low
   because plain paper has no edges, not because the photos are blurry.
5. **Green share is the clearest difference between backgrounds:** median 0.058
   white vs 0.62 field. In the field the diseased leaf competes with lots of
   healthy green around it.
6. **No copied files, but some field leaves were photographed twice**
   (`rtf_17` / `rtf_18`). Worth measuring later with better features.

---

## L2. Stage 2 (`02_splits.py`): dividing the images fairly, five times

### The big idea: a fair exam, repeated
A test score only means something if the test questions were never seen
during study. Stage 2 decides, once and for all, which images are "study
material" (training, 80%) and which are "exam questions" (test, 20%), for
each of the 5 splits. It saves that decision to a file, and every later stage
reads the same file. No later stage is allowed to make its own split.

### Concept 1: a random seed
Analogy: shuffling a deck of cards, but writing down exactly how you shuffled
so anyone can repeat it. The seed (0, 1, 2, 3, 4) is that written-down
shuffle. Same seed = same split on any computer, so the marker reproduces our
exact numbers. Different seeds = different splits, which is what the brief's
"5 runs with different random splits" asks for.

### Concept 2: stratified, by disease x background
Analogy: picking a 20% sample from a school by taking 20% of each class, not
20% of whoever walks past first. There are 10 groups (5 diseases x 2
backgrounds), from 49 to 219 images. Each group gives 20% of its own images
to the test part, so every test part has the same mix: about 154 white and
67 field images, and roughly 10 Field Brown Spot images. Without this, one
split might by chance get 4 Field Brown Spot test images and another 16.

### Concept 3: why save the split to a file
If each stage made its own split, a tiny difference (a file listed in a
different order on Windows) could put an image in training for the baseline
and in testing for the main method. Then the two methods would sit different
exams and could not be compared. Saving once removes that risk. The file is
interim data, so it lives at `Dhan-Shomadhan/_interim/splits.csv` (D10), and
the code that makes it is part of the submission (brief rule b).

### Concept 4: images can move between splits, and that is fine
An image in the training part of split 0 can be in the test part of split 1.
That is not leakage: each split is a separate experiment with its own model,
trained from scratch. Leakage only happens when one model is tested on an
image it trained on. The check that matters is inside each split: training
and test share no image.

### Concept 5: a known weak spot, the same leaf twice
Stage 1 found field leaves photographed twice (`rtf_17` / `rtf_18`). A random
split can put one in training and the other in test. This is not fixed here
(the measure is not reliable enough to group them all), but Stage 5 will
count how often it happened, and 6.3 reports it as a limitation.

### What to say in a demo
"We split the 1106 images 80/20 into training and test, stratified by
disease and background, so every test set has the same mix of white and field
images of each disease. We did this 5 times with fixed seeds 0 to 4, saved
the splits to a file, and every method uses that same file, so both methods
are tested on exactly the same images in each split."

### Check yourself
1. Why must the baseline and the main method use the same saved splits?
2. Is it a problem that `bs_wb_12.jpg` is in training in split 0 and in test in split 2?
3. What would go wrong if the split were stratified by disease only?
4. The marker runs our code on Windows. Why does a fixed seed alone not
   guarantee the same split, and what in the code protects against that?

### Answers
1. So both sit the same exam. Otherwise a difference in scores could come from
   easier test images, not better features.
2. No. Each split trains its own model from scratch; within each split,
   training and test are separate.
3. A split could by chance get very few field images of one disease in its test
   part, making the field score for that split jumpy and the white/field
   comparison unfair.
4. The seed fixes the shuffle, but the shuffle is applied to a list; if the list
   came in a different order, the result would differ. `list_images()` sorts
   the files by name, so the list is the same on every computer.

### What the output showed (26 Oct, `results/logs/02_splits_log.txt`)
1. **Every test part is the same size and mix:** 222 images (154 white, 68
   field), with exactly the same count per group in all 5 splits. That is
   stratification doing its job.
2. **Field Brown Spot gets 10 test images per split**, the smallest group.
   Expect its per-class score to swing by 10 points per mistake.
3. **367 images were never tested in any split.** Each split draws its 20%
   independently, so the chance of an image being missed 5 times is
   0.8^5 = 0.33, about 363 images. 5-fold cross-validation would test every
   image once, but the brief asks for random splits.
4. **The Mac run matched the sandbox run exactly**, so the splits are
   reproducible across machines.

---

## L3. Stage 3 (`03_baseline_features.py`): describing images by hand-made rules

### The big idea: turning a picture into a list of numbers
An SVM cannot look at a photo. It needs every image described by the same
fixed-length list of numbers (a **feature vector**), so that similar images
get similar lists. The baseline uses rules a person designed: "how much of
each colour is there?" and "which way do the edges point?". Stage 4 will use
rules ResNet-18 learned from millions of photos. Comparing the two answers the
question: do learned features beat hand-made ones on this problem?

### Concept 1: a fixed size first
Analogy: to compare handwriting, you first scale every page to the same size.
Images are shrunk to 128 x 256 pixels (width x height), close to their real
1 : 2.1 shape, so leaves are barely squashed. The 11 sideways field photos are
turned upright first, since almost every leaf in the dataset runs vertically.

### Concept 2: colour histogram
Analogy: emptying a jar of beads and counting how many of each colour, without
caring where each bead was. A histogram counts how many pixels fall into
each colour range ("bin"). It ignores position, so a lesion at the top or the
bottom of the leaf counts the same.
- **HSV**: Hue (which paint colour), Saturation (how strong), Value (how
  bright). Hue stays fairly steady when the light changes, unlike raw RGB.
  Bins: 16 for hue, 8 for saturation, 8 for value.
- **YCrCb**: Y is brightness, Cr and Cb are "how red" and "how blue". Only Cr
  and Cb are used (16 bins each), so this part ignores brightness.
- Each histogram is divided by the pixel count, so it shows shares, not counts.
Total colour features: 16 + 8 + 8 + 16 + 16 = 64 numbers.

### Concept 3: HOG (histogram of oriented gradients)
Analogy: describing a drawing by the direction of its pencil strokes in each
small square of the page. The image is cut into 16 x 16 pixel cells. In each
cell, HOG measures edge directions (9 direction bins between 0 and 180
degrees) and how strong they are. Neighbouring cells are grouped in 2 x 2
blocks and normalised, so a darker or lighter photo gives similar values.
On a 128 x 256 image: 8 x 16 cells, 7 x 15 blocks, 4 cells x 9 bins each =
7 x 15 x 36 = 3780 numbers. HOG captures shape: the long edges of the leaf,
the outline of a lesion.

### Concept 4: why this step cannot leak test information
Each image's features are computed from that image alone, with fixed rules.
Nothing is "learned" from other images here. So computing features for all
1106 images at once is safe. Anything that learns from data (scaling the
features, tuning the SVM) happens later, on the training part only.

### Concept 5: the baseline's built-in weakness
The histograms count every pixel, background included. On a white image about
a quarter of the pixels are near-white paper; in the field, most pixels are
healthy leaves and soil. So the colour features partly describe the
background, not the disease. That is expected to hurt most on field images,
and it is one reason to expect the learned features to do better.

### What to say in a demo
"Our baseline describes each image with 3844 hand-made numbers: 64 for
colour (HSV and YCrCb histograms) and 3780 for edge directions (HOG), from
the image resized to 128 x 256. These rules are fixed, so they cannot leak
test information. Their weakness is that they describe the whole image,
background included, which is why we expected them to struggle on field
images."

### Check yourself
1. Why is hue more useful than raw red/green/blue when lighting changes?
2. Two Brown Spot leaves, one with spots near the top, one near the bottom.
   Do their colour histograms differ? Do their HOG features?
3. Why is the Y channel left out of the YCrCb histogram?
4. Why is it safe to compute features for all images before splitting, but not
   safe to fit a feature scaler on all images?

### Answers
1. Dimmer light lowers all three RGB values together, changing the "colour" in
   RGB; hue describes which colour it is and changes much less.
2. Colour histograms: hardly, they ignore position. HOG: yes, the edges sit in
   different cells.
3. Y is brightness, already covered by HSV's V; leaving it out keeps this part
   about colour only and less sensitive to lighting.
4. Features use fixed rules per image. A scaler learns a mean and spread from
   the images it sees; fitted on all images, it would learn from test images.

---

## L4. Stage 4 (`04_resnet_features.py`): borrowing a trained eye

### The big idea: transfer learning, used as a fixed feature extractor
Training a deep network needs many thousands of labelled images; we have
1106. But ResNet-18 has already been trained on about 1.2 million everyday
photos (ImageNet, 1000 classes). To get good at that, its early layers learned
to see edges and colours, its middle layers textures and spots, its later
layers parts and shapes. Those skills transfer to leaves. We cut off its last
layer (the one that says "dog", "car", ...) and keep what comes just before:
512 numbers that summarise what the network "sees" in the image. The SVM then
learns diseases from those 512 numbers.
Analogy: hiring an experienced art critic who has never seen rice. You do
not teach them to see; you ask them to describe each leaf in their own
vocabulary, and a junior clerk (the SVM) learns which descriptions mean which
disease.

### Concept 1: "frozen" and `eval()`
Frozen means the network's weights never change: no training, so no risk of
learning from test images, and it runs once for all 1106 images. `eval()`
switches layers that behave differently during training (batch
normalisation) to their fixed, inference behaviour. `torch.no_grad()` skips
the bookkeeping needed for training, saving memory and time.

### Concept 2: global average pooling, and why the input need not be square
ResNet-18's last convolution produces 512 "maps", each showing where one
learned pattern appears in the image. Global average pooling takes the
average of each map: 512 numbers, whatever the image size. So instead of
squashing the tall photo into a 224 x 224 square (leaf distorted) or cropping
the middle square (top and bottom of the leaf lost), we feed 224 x 448 (w x h),
which keeps the photo's shape (D14). The network was trained on 224 x 224, so
its patterns are tuned to that scale; 224 wide keeps the leaf at a similar
scale.

### Concept 3: preparing images the way the network expects
- **RGB, not BGR.** OpenCV reads colour as blue-green-red; the weights were
  trained on red-green-blue. Forgetting to swap gives a network that sees
  blue leaves. It would still run without any error: a silent failure.
- **ImageNet normalisation.** Pixel values scaled to 0-1, then each channel has
  the ImageNet average subtracted and is divided by its spread (mean 0.485,
  0.456, 0.406; std 0.229, 0.224, 0.225). The network learned on images
  prepared this way, so ours must match.

### Concept 4: MPS
On an M-series Mac, PyTorch can run on the built-in GPU through "MPS" (Metal
Performance Shaders). The code uses MPS if available, otherwise CPU. Results
can differ in the last decimal places between devices; that does not change
the SVM's decisions in any meaningful way, but the device is logged.

### Concept 5: downloadable weights (a brief rule)
The brief allows pretrained models if they can be downloaded, with
instructions. torchvision downloads `resnet18-f37072fd.pth` (about 45 MB)
automatically the first time, into the PyTorch cache folder. The User Manual
must say this, and say where the file goes, for a marker without internet
during the run.

### What to say in a demo
"We used ResNet-18 pretrained on ImageNet as a fixed feature extractor: we
removed its final classification layer and took the 512-number output of the
global average pooling layer for each image. The network is never trained on
our data. Images are converted to RGB, kept upright at 224 x 448 to preserve
their shape, and normalised with the ImageNet statistics. An SVM then
classifies these 512-number descriptions."

### Check yourself
1. Why can the network take a 224 x 448 image when it was trained on 224 x 224?
2. What would happen, visibly, if BGR were not converted to RGB?
3. Why does using a frozen network on all 1106 images not leak test information?
4. Why remove the final layer instead of using ResNet-18's own 1000-class prediction?

### Answers
1. Its layers are convolutions (they slide over any size) and global average
   pooling averages each map to one number, whatever its size.
2. Nothing visible: no error, features still produced. Only the results would
   be quietly worse, because red and blue are swapped (brown lesions look blue).
3. The weights never change; each image's features depend only on that image
   and on ImageNet, never on our labels or other images.
4. The 1000 ImageNet classes do not include rice diseases. The layer before it
   holds the general description, which the SVM can map to our 5 classes.

---

## L5. Stage 5 (`05_train_evaluate.py`): training, tuning and the exam

### The big idea: tune on the homework, then sit the exam once
The SVM has settings (which kernel, how large C and gamma) that change its
results. Choosing them by looking at test scores would be like a student
seeing the exam paper before revising: the score would no longer measure
anything. So settings are chosen using the training part only, then the
chosen model sits the test once.

### Concept 1: cross-validation on the training part
Analogy: practice exams made from your own homework. The 884 training images
are cut into 5 "folds". Train on 4 folds, test on the 5th, rotate 5 times,
average. Do this for every candidate setting; keep the best. Then train once
more on all 884 with that setting. The real test images were never looked at.

### Concept 2: the SVM settings
- **Linear kernel:** the SVM draws flat boundaries between classes.
- **RBF kernel:** boundaries can curve; **gamma** sets how wiggly (large gamma
  = each training image's influence is very local, risks memorising).
- **C:** how hard the SVM tries to classify every training image correctly.
  Large C = strict (risks memorising noise); small C = relaxed (smoother
  boundary, may underfit).
13 combinations are tried (D15).

### Concept 3: standardising, inside the folds
The scaler makes every feature have mean 0 and spread 1, so features with big
numbers do not dominate. It learns that mean and spread from data, so it is
part of the pipeline: fitted on the training folds only, every time. Fitting
it on all images first would let test images influence training (leakage).

### Concept 4: one prediction, three scores
Each test image is predicted once. Then the answers are counted three ways:
white test images, field test images, all. That is D5 in code.

### Concept 5: macro F1 versus accuracy
Accuracy = share of test images right. If a model ignored Brown Spot
completely, accuracy would barely drop (Brown Spot is a small class). Macro F1
computes a score per class (balancing "when it says Brown Spot, is it right?"
and "does it find the Brown Spot images?") and averages the 5 classes equally,
so ignoring a class is punished.

### Concept 6: the extra run (white-only training, field test)
Same main method, but it never sees a field image during training. If the
background does not matter, it should do about as well on field images as the
normal model. If it collapses, the features are describing the background as
much as the disease.

### What to say in a demo
"For each split, the SVM's kernel, C and gamma were chosen by 5-fold
cross-validation on the training part only, scored by macro F1, with feature
scaling fitted inside each fold. The chosen setting was retrained on the full
training part and each test image predicted once; we then counted accuracy
and macro F1 on the white, field and all test images."

### Check yourself
1. Why is the scaler inside the pipeline instead of applied once to all images?
2. In one split the chosen C is 100, the largest value tried. What does that suggest?
3. A model gets 80% accuracy but macro F1 0.55. What is probably happening?
4. Why does each test image get predicted only once, even though it gets three scores?

### Answers
1. It learns a mean and spread; applied to all images it would learn from test
   images. Inside the pipeline it is fitted on training folds only.
2. The best value may lie beyond the grid; if that happens in most splits,
   widen the grid (still using training data only).
3. It does well on large classes and badly on one or more small ones.
4. One model, one prediction per image; the three scenarios only choose which
   predictions to count.
