# Results file format (contract between experiments and plotting)

The experiment code writes these files into `results/`. The plotting script only reads them.
First real results (one split) are added on Wed 21 Oct so plotting can be built early.

## summary.csv  (one row per method x scenario x split)
| column | example | notes |
|---|---|---|
| method | resnet18_svm | or `colour_hog_svm` |
| scenario | white | `white`, `field`, `mixed` |
| split | 0 | 0 to 4 |
| accuracy | 0.91 | 0 to 1 |
| macro_f1 | 0.89 | 0 to 1 |

## per_class.csv  (one row per method x scenario x split x class)
| column | example |
|---|---|
| method, scenario, split | as above |
| class | Brown Spot |
| precision, recall, f1 | 0 to 1 |

## confusion_<method>_<scenario>.csv
5 x 5 counts summed over the 5 splits. Rows = true class, columns = predicted class,
order: Brown Spot, Leaf Scald, Rice Blast, Rice Tungro, Sheath Blight.

## Figures wanted for the report (page budget is tight)
1. Confusion matrices: one figure, 3 panels (white / field / mixed), main method.
2. Sample image grid (2 x 5) for Section 3.
3. Failure-case strip (6 images, true vs predicted labels).

Accuracy mean and std go in a table, not a chart.
Rules: matplotlib only, PNGs to `results/figures/`, full text width, max 6 cm tall, text at least 9pt printed.
