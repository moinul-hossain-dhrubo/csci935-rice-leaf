# Amit's source notes for Papers 1 and 5

The DOCX files contain one paragraph each (177 and 175 whitespace-delimited words), followed by a citation. The latest team message assigns both papers to Amit and changes the length to 150-200 words. Citation numbers [1] and [5] are placeholders for Anika to renumber.

These notes provide the six-question evidence trail and page references for checking the summaries. They are supporting notes, not extra text for the two-page Related Work section.

## Paper 1

Source: `paper 1.pdf`, Sobuj et al. (2024). Page numbers below are PDF pages.

1. **Problem and data:** Rice disease classification. Section III-A, pp. 2-3, gives 4,078 images: 613 Brown Spot, 1,488 Healthy, 977 Leaf Blast, 1,000 Neck Blast. Sources include Dhan-Shomadhan and Kaggle Rice Leafs. The paper does not quantify white versus field backgrounds; do not describe the entire collection as laboratory-only or field-only.
2. **Method:** Sections III-C/D and IV, pp. 3-4: 224 x 224 resizing, augmentation, seven pretrained CNNs, HOG and LBP variants. Section V-B, pp. 5-6: Grad-CAM visualisation. Grad-CAM explains attention; it is not an additional classifier.
3. **Main result:** p. 5, Tables I, II and IV: EfficientNet-B7 baseline 0.92; with HOG 0.97. State 97% accuracy as the authors' reported result, not a reproduced result.
4. **Strength:** Our assessment of Tables I-III: comparing feature variants across seven networks makes the representation comparison more informative than testing one network alone.
5. **Limitation:** Sections III-V do not specify a clear train/validation/test allocation or background-separated results. This limits interpretation and reproducibility; it does not prove leakage. The four labels differ from our five disease labels.
6. **Relevance:** Our interpretation: motivates comparing handcrafted and learned features, but does not directly validate fixed ResNet-18 plus SVM or performance under the project's white/field/mixed scenarios.

**Reporting caution:** p. 5 prose calls LBP an improvement, but Tables I/III/IV show EfficientNet-B7 falls from 0.92 to 0.90. The summary uses the consistent HOG result and does not repeat that claim.

## Paper 5

Source: `paper 5.pdf`, Ali et al. (2024).

1. **Problem and data:** Abstract p. 1 calls the 5,932 Mendeley images in-field photographs. Section III-A p. 2 gives Bacterial Blight 1,584, Blast 1,440, Brown Spot 1,600 and Tungro 1,308. These sum to 5,932.
2. **Method:** Sections III-C/F, pp. 2-3: 300 x 300 inputs, augmentation, five ImageNet-pretrained models, additional pooling/dropout/dense layers and softmax. Table III p. 3 gives 50 epochs, Adam and learning rate 0.0001. This is not the same classifier as our SVM.
3. **Main result:** p. 4 Table IV and p. 5 conclusion report Xception at 99.99% test accuracy. Section IV pp. 3-4 states 597 test images. Preserve attribution to the authors.
4. **Strength:** Our assessment: compares five pretrained architectures on the same rice-disease task rather than presenting only the selected network.
5. **Limitation:** No separate white/field evaluation; Leaf Scald and Sheath Blight are absent from the four labels. These restrict transfer of the result to our problem.
6. **Relevance:** Our interpretation: supports testing pretrained features on field imagery, with independent repeated-split evaluation on our own five classes and three scenarios.

**Reporting caution:** With 597 test images, a single error changes ordinary accuracy by about 0.1675 percentage points. Thus 99.99% cannot be a literal integer correct/597 fraction rounded to two decimals. Figure 3 shows a perfect diagonal and the text says 100% TPR. The paper does not explain this discrepancy; do not silently change the reported figure to 100% or claim the result was independently verified. Table I specifies vertical reversal False despite prose mentioning vertical flips; the summary avoids that detail.

## Citation verification

Titles, authors, conference names and DOIs were checked against the supplied PDFs. Page ranges and indexed author names were verified using publisher-deposited Crossref records on 28 October 2026:

- https://api.crossref.org/works/10.1109/iCACCESS61735.2024.10499603
- https://api.crossref.org/works/10.1109/COMPAS60761.2024.10796309

IEEE Xplore's citation interface was not accessible through the web tool, so the citations were formatted from verified metadata rather than represented as verbatim exports from Cite This. Paper 1's publisher metadata groups the first surname as “Islam Sobuj” and the last as “Islam Khan”; the reference preserves these indexed names.
