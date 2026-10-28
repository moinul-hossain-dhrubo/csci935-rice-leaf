# Amit's contribution

Prepared on 28 October 2026 on branch `Amit`, based on `main` at `0e3e4dc`.

## Deliverables

| Task | Files | Check |
|---|---|---|
| A1 sample images | `figures/samples/` and `originals.txt` | Exactly 10 JPEGs, one for each disease/background combination; required filenames preserved |
| A2 Paper 1 | `papers/paper1_summary_amit.docx` | One 177-word paragraph plus IEEE-style citation |
| A2 Paper 5, added in team update | `papers/paper5_summary_amit.docx` | One 175-word paragraph plus IEEE-style citation |
| A3 failure review | `figures/failures/` and `failures.txt` | Exactly 3 white and 3 field JPEGs; reasons 11-12 words; final pattern line 23 words including its label |

Paper source notes and reporting caveats are in `papers/paper_notes_amit.md`. The DOCX summaries are AI-assisted drafts; Amit should personally check the cited pages and revise wording as required by Section 7 of the team plan before submission. Reference numbers [1] and [5] are for Anika to renumber.

## Image selection and provenance

The source is the lecturer's `Dhan-Shomashan.zip` linked from the assignment brief. The extracted dataset contains 1,106 images; every folder label agrees with the supplied picture-information CSV. Original dataset files were left unchanged. The selected files were copied without resizing, cropping, recompression or metadata changes. `figures/amit_image_manifest.csv` records the original relative paths, labels and SHA-256 hashes of all 16 copies.

Samples were chosen visually for visible damage/discolouration and checked in the existing plot script's centre crops. They are illustrative examples, not a representative evaluation sample.

Every failure pick has `times_wrong = times_tested = 3` in `results/wrong_predictions.csv`. Cross-checking `results/predictions.csv` confirms that each was misclassified in all three test appearances, with the same predicted label each time. This does not mean they were tested in all five splits. The selected true classes are Brown Spot, Leaf Scald and Rice Blast in each background. Three cases predict Sheath Blight, two Leaf Scald and one Rice Blast.

Reasons in `failures.txt` describe visually plausible explanations, not experimentally proven model causes. The six selected failures cannot establish overall error rates or cover every disease's failure modes.

## Verification and handoff

- Checked image counts, all ten sample filenames, original-name mapping, dataset labels, CSV membership, failure counts, word limits and byte-for-byte copy hashes.
- Rendered both DOCX files and inspected their single-page layouts; each has one summary paragraph followed by its citation, without headings or figures.
- Ran all three functions in the unchanged `plots.py` using Python 3.12, matplotlib 3.11.2 and numpy 2.5.4. All succeeded. Inspected the sample and failure previews for visible lesions and readable, uncropped labels.
- Verification previews are local, ignored files under `Dhan-Shomadhan/_interim/amit_review/plots/`. Existing tracked `results/figures/` files were not replaced; Akram's K3 can rebuild them with these selections using `python plots.py`.
- Only the selected images and Amit's documents/notes belong in this branch. The dataset, downloaded archive, temporary dependencies, model weights and experiment outputs are not included.

No group-chat message was sent. The branch is ready for the team to review and merge; the final report, final figure handoff and Moodle submission remain with their assigned owners.
