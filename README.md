# Rice Leaf Disease Classification (CSCI935/435 Group Project)

Classifies five rice leaf diseases (Brown Spot, Leaf Scald, Rice Blast, Rice Tungro, Sheath Blight)
from the Dhan-Shomadhan dataset, on white-background and field-background images.

**Team plan and every member's tasks:** [TASKS.md](TASKS.md) (same content as the team plan PDF).
**Report draft (Google Doc):** link added by Anika.

## Folders
| Folder | What goes in it |
|---|---|
| `papers/` | Paper PDFs (Anika) and paper summaries (Amit, Akram, Sabir, Aziz) |
| `dataset-notes/` | Dataset inspection notes |
| `figures/samples/` | 10 sample images for Section 3 |
| `figures/failures/` | 6 failure-case images for Section 6 |
| `docs/` | Installation instructions, User Manual, clean-machine test |
| `report/` | Reference list and report exports |
| `results/` | Experiment results (CSV) and figures |

## How to upload a file (no Git needed)
1. Open the folder your task names.
2. **Add file** > **Upload files**, drag your file in.
3. Write a short message (e.g. "Paper 1 summary, Amit"), keep **Commit directly to the main branch**, click **Commit changes**.

Never delete or rename other people's files. Do not upload the dataset.

## Setup (code)
```
pip install -r requirements.txt
```
Place the dataset at `./Dhan-Shomadhan/` (ignored by git).
