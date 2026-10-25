<div class="section">

<div>

<div class="kicker">

CSCI935 / CSCI435 Computer Vision · Spring 2026

</div>

# Group Project Team Plan

<div class="sub">

Rice Leaf Disease Classification and Detection for Bangladeshi Local Rice

</div>

<div class="bigdates">

<div>

Work starts**Mon 26 Oct**Plan and links shared Sun 25

</div>

<div style="border-color:var(--accent);background:var(--accent-soft)">

Team submission**Sat 31 Oct**Everything finished and submitted

</div>

<div>

Official deadline**Sun 1 Nov, 11:30pm**Sun 1 Nov is emergency buffer only

</div>

</div>

| Member | Role in one line                                                                    |
|--------|-------------------------------------------------------------------------------------|
| Moinul | Code lead: dataset inspection, all code, experiments, Section 6 Results, submission |
| Anika  | Report lead: literature search, Sections 3, 4, 5, Executive Summary, final report   |
| Amit   | Sample images, Paper 1 summary, failure-case review                                 |
| Akram  | Paper 2 summary, report figures, clean-machine test of the User Manual              |
| Sabir  | Paper 3 summary, installation instructions                                          |
| Aziz   | Paper 4 summary, IEEE reference list                                                |

</div>

<div>

#### Contents

|                                                                     |
|---------------------------------------------------------------------|
| 1\. How to use this document (and how to use AI with it)            |
| 2\. The project in plain words, and key terms                       |
| 3\. Page limits                                                     |
| 4\. Ground rules and where files go                                 |
| 5\. Calendar                                                        |
| 6\. Everyone's tasks at a glance                                    |
| 7\. The paper summary task (Amit, Akram, Sabir, Aziz)               |
| 8\. Amit · 9. Akram · 10. Sabir · 11. Aziz · 12. Anika · 13. Moinul |

</div>

</div>

<div class="section page">

## 1. How to use this document

1.  **Everyone reads Sections 1 to 6** (about 5 pages). They explain the project, the page limits and the rules.
2.  **Then read your own chapter** (Sections 8 to 13). It has every task you own, with exact steps, size limits, file names and due dates.
3.  **Paper summary writers** (Amit, Akram, Sabir, Aziz) also read **Section 7**.
4.  Each task ends with a **"Done when"** checklist. Tick every box before you post that the task is finished.

<div class="box rule">

<span class="t">The plan is fixed</span> Due dates, size limits, file names and folders in this document are decided. Please don't change them, even if an AI tool suggests something different. If something looks wrong or impossible, ask in the group chat.

</div>

### Using AI to understand your tasks

You are welcome to use AI tools (ChatGPT, Claude, Gemini, etc.) to understand your tasks, the paper you are summarising, or a term you don't know. To get good answers:

1.  **Give the AI the context.** Upload this whole PDF, or copy the "Context for AI" box below plus your own chapter.
2.  **Use the ready-made prompt** at the end of your chapter. It tells the AI what you need and what it must not change.
3.  **Ask the AI to explain, not to do the task for you.** For example: "Explain what this paper's method does in simple words", "What does a confusion matrix show?", "Check my summary against the 6 questions".

<div class="box rule">

<span class="t">Three AI rules that protect our marks</span>

1.  **Write report text in your own words.** The brief treats copied text as plagiarism, and AI text often repeats a paper's own sentences. Use AI to understand, then write it yourself.
2.  **Check every number against the actual paper or file.** AI tools sometimes invent accuracy figures, dataset sizes and even whole references. Every number and citation you hand in must be one you have seen in the source yourself.
3.  **The AI does not override this plan.** If it suggests a longer summary, more figures, a different file name or a different method, ignore it and follow this document.

</div>

<div class="box ai">

<span class="t">Context for AI (copy this into your AI chat first)</span>

<div class="prompt">

I am a member of a 6-person university group project for CSCI935 Computer Vision (University of Wollongong). The task: build and evaluate a method that classifies rice leaf photos into 5 diseases (Brown Spot, Leaf Scald, Rice Blast, Rice Tungro, Sheath Blight) using the Dhan-Shomadhan dataset (1106 images, two kinds: white background and field background). Our method: a pretrained ResNet-18 network turns each image into a feature vector and an SVM classifies it; we compare it with a baseline of colour histograms + HOG features + SVM. We test 3 scenarios (white-only, field-only, mixed), each repeated over 5 random train/test splits, reporting mean and standard deviation. The report has strict page limits and anything over the limit is not marked. The plan document I will share with you is fixed: do not change due dates, size limits, file names or the method. Help me UNDERSTAND my task and check my work; do not write report text for me, and never invent numbers or references.

</div>

</div>

</div>

<div class="section page">

## 2. The project in plain words

A client wants software that looks at a photo of a rice leaf and says which of five diseases it has: **Brown Spot, Leaf Scald, Rice Blast, Rice Tungro or Sheath Blight**. The photos come in two kinds: leaves on a **white background** (clean, easy) and leaves in the **field** (soil, other plants, shadows, harder). We use the **Dhan-Shomadhan** dataset of 1106 images collected in the Dhaka Division of Bangladesh.

We have to deliver two things in one zip file:

- **One Python file** that runs the method and reproduces our results.
- **One PDF report** written as a consulting report for the client, with fixed sections and page limits.

### Our method, step by step

1.  **Feature extraction:** a neural network called **ResNet-18**, already trained by others on millions of everyday photos, turns each leaf photo into a list of 512 numbers that describes it. We don't train this network; we only use it.
2.  **Classification:** an **SVM** (support vector machine), a simple classifier, learns which lists of numbers belong to which disease.
3.  **Baseline for comparison:** an older approach using **colour histograms** (how much of each colour is in the image) and **HOG** features (directions of edges), also with an SVM.
4.  **Testing:** 3 scenarios (white background only, field only, both mixed). Each one is repeated with **5 different random splits** of the data into training and testing images. We report the **average and standard deviation** of accuracy.

### Key terms

| Term                     | Meaning in this project                                                                                        |
|--------------------------|----------------------------------------------------------------------------------------------------------------|
| Scenario                 | Which test images we use: *white*, *field* or *mixed*.                                                         |
| Split                    | One random division of the images into a training part (80%) and a testing part (20%). We do 5 splits.         |
| Accuracy                 | Percentage of test images given the correct disease.                                                           |
| Standard deviation (std) | How much the accuracy changes between the 5 splits. Small std = stable results.                                |
| Confusion matrix         | A 5 x 5 table: rows are the true disease, columns the predicted disease. It shows which diseases get mixed up. |
| Failure case             | A test image the method classified wrongly.                                                                    |
| IEEE style               | The citation format the brief requires: `[1]` in the text, full details in the reference list.                 |
| IEEE Xplore              | The paper database we must use, accessed through the UOW library.                                              |
| Clean machine            | A computer or fresh Python environment that has never run our code, used to check the instructions work.       |

### Report sections (from the assignment brief)

1 Coversheet · 2 Executive Summary · 3 Problem Description · 4 Related Work · 5 Method and Implementation · 6 Experimental Results and Analysis · 7 References · 8 Appendix: User Manual. Page limits for each are in Section 3.

</div>

<div class="section page">

## 3. Page limits

<div class="box rule">

<span class="t">Anything past a section's page limit is ignored by the marker</span> Report format: single column, 12pt font, 1.25 cm margins. **One full page is about 450 words.** Figures and tables count toward the page limit of the section they sit in.

</div>

| Section                               | Marks                      | Page limit                       | Owner                                                    |
|---------------------------------------|----------------------------|----------------------------------|----------------------------------------------------------|
| Coversheet                            | 0                          | 1 page                           | Anika                                                    |
| 2\. Executive Summary                 | 1                          | 0.5 page                         | Anika                                                    |
| 3\. Problem Description               | 5                          | 1 page                           | Anika                                                    |
| 4\. Related Work                      | 2                          | 2 pages (4 summaries x 0.5 page) | Amit, Akram, Sabir, Aziz write; Anika edits              |
| 5\. Method and Implementation         | 8                          | 3 pages                          | Anika (method, 2 pages), Moinul (implementation, 1 page) |
| 6\. Experimental Results and Analysis | 8                          | 2 pages                          | Moinul                                                   |
| 7\. References                        | 1                          | no limit                         | Aziz                                                     |
| 8\. Appendix: User Manual             | 0 (heavy penalty if wrong) | 2 pages                          | Sabir (install, 1 page), Moinul (run, 1 page)            |

### How we stay inside the limits

1.  **Each section starts on a new page** in the report, so checking a limit means counting pages.
2.  **Every task has its own size limit** (words, pages or number of images). Stay at or under it.
3.  **Figures are full text width and at most 6 cm tall.** Related images are combined into one figure with panels (a), (b), (c).
4.  **No figure repeats a table.** If numbers are in a table, they are not also charted.

</div>

<div class="section page">

## 4. Ground rules and where files go

1.  **Own words only.** Copying sentences from a paper, website or AI tool is plagiarism and costs the whole group.
2.  **Stay inside your size limit.** If your limit is 200 words, 230 words is not acceptable. Use Word count in Word or Google Docs (Tools \> Word count).
3.  **Use the exact file name and folder** given in your task.
4.  **Post in the group chat when a task is done**, with the file name.
5.  **If you will be late, say so before the due date.** Late tasks are reassigned the next day so the group stays on schedule.
6.  **Your finished tasks are listed as your contribution** on the report coversheet.

### Where files go

**Shared Google Drive folder** `CSCI935 Group Project` (link posted in the group chat with this plan):

| Folder              | What goes in it                                            | Who uses it                     |
|---------------------|------------------------------------------------------------|---------------------------------|
| `papers/`           | Paper PDFs and paper summaries                             | Anika, Amit, Akram, Sabir, Aziz |
| `dataset-notes/`    | Dataset inspection notes                                   | Moinul, Anika                   |
| `figures/samples/`  | 10 sample images for Section 3                             | Amit, Akram                     |
| `figures/failures/` | 6 failure-case images                                      | Amit, Akram                     |
| `docs/`             | Installation instructions, User Manual, clean-machine test | Sabir, Moinul, Akram            |
| `report/`           | Report draft and reference list                            | Anika, Aziz, Moinul             |

**GitHub repository** (link posted with this plan): code only. Akram submits `plots.py` there.

</div>

<div class="section page">

## 5. Calendar

| Date             | Who                          | What is due                                                                           |
|------------------|------------------------------|---------------------------------------------------------------------------------------|
| Sun 25 Oct       | Moinul                       | Plan shared; Google Drive and GitHub links posted                                     |
| Mon 26 Oct       | Anika                        | Report template; papers found and assigned                                            |
| Mon 26 Oct       | Amit                         | A1 Sample images                                                                      |
| Mon 26 Oct       | Moinul                       | Dataset inspection; baseline method running                                           |
| Tue 27 Oct       | Anika                        | Section 3 Problem Description                                                         |
| Tue 27 Oct       | Moinul                       | Main method running; first results shared for Akram                                   |
| **Wed 28 Oct**   | **Amit, Akram, Sabir, Aziz** | **Paper summaries (200 words each)**                                                  |
| Wed 28 Oct       | Sabir                        | S2 Installation instructions                                                          |
| Wed 28 Oct       | Anika                        | Section 5 Method part                                                                 |
| Wed 28 Oct       | Moinul                       | All experiments finished                                                              |
| Thu 29 Oct       | Akram                        | K2 Figures script                                                                     |
| Thu 29 Oct       | Aziz                         | Z2 Reference list                                                                     |
| Thu 29 Oct       | Anika                        | Section 4 Related Work                                                                |
| Thu 29 Oct       | Moinul                       | Implementation part; wrong-prediction list for Amit; User Manual ready for Akram      |
| Fri 30 Oct, 12pm | Amit                         | A3 Failure-case review                                                                |
| Fri 30 Oct       | Akram                        | K4 Clean-machine test; K3 Final figures (evening)                                     |
| Fri 30 Oct       | Moinul                       | Section 6.1 and 6.2 draft; fix test issues                                            |
| Sat 31 Oct       | Moinul, Anika                | Code freeze; Section 6 final and reviewed; Executive Summary, coversheet, final merge |
| **Sat 31 Oct**   | **Moinul**                   | **Submit on Moodle**                                                                  |
| Sun 1 Nov        |                              | **Emergency buffer. No planned work.**                                                |

</div>

<div class="section page">

## 6. Everyone's tasks at a glance

| Member     | Task                         | Size limit               | Due                  |
|------------|------------------------------|--------------------------|----------------------|
| **Amit**   | A1 Sample images             | 10 images                | Mon 26 Oct           |
|            | A2 Summary of Paper 1        | 200 words (0.5 page)     | Wed 28 Oct           |
|            | A3 Failure-case review       | 6 images, 15 words each  | Fri 30 Oct, 12pm     |
| **Akram**  | K1 Summary of Paper 2        | 200 words (0.5 page)     | Wed 28 Oct           |
|            | K2 Figures script            | 3 figures, max 6 cm tall | Thu 29 Oct           |
|            | K4 Clean-machine test        | Test log                 | Fri 30 Oct           |
|            | K3 Final figures             | 3 figures, max 6 cm tall | Fri 30 Oct (evening) |
| **Sabir**  | S1 Summary of Paper 3        | 200 words (0.5 page)     | Wed 28 Oct           |
|            | S2 Installation instructions | 1 page                   | Wed 28 Oct           |
| **Aziz**   | Z1 Summary of Paper 4        | 200 words (0.5 page)     | Wed 28 Oct           |
|            | Z2 Reference list            | No limit                 | Thu 29 Oct           |
| **Anika**  | See Section 12               |                          |                      |
| **Moinul** | See Section 13               |                          |                      |

</div>

<div class="section page">

## 7. The paper summary task

<div class="meta">

**Who**Amit = Paper 1 · Akram = Paper 2 · Sabir = Paper 3 · Aziz = Paper 4 **Due****Wed 28 Oct** **Size limit****200 words maximum** (half a page). One paragraph. No headings, no bullet points, no figures. **Input**Your paper PDF, which Anika puts in `papers/` by Mon 26 Oct **Hand in**`papers/paper<N>_summary_<yourname>.docx`, for example `papers/paper1_summary_amit.docx`

</div>

#### Why this task matters

Section 4 of the report must summarise at least 4 IEEE papers from 2022 or later, and it has **only 2 pages**. Four summaries of half a page each fill it exactly. If a summary is longer, the end of the section is not marked.

<div class="box note">

<span class="t">You do not search for papers</span> Anika finds them. By **Mon 26 Oct** she puts the PDFs in `papers/` as `paper1.pdf` to `paper4.pdf` and posts in the group chat who has which. Nothing to do before then.

</div>

#### Steps

1.  **Read** the paper's abstract, method section, results section and conclusion. You don't need every page.
2.  **Answer these 6 questions** in rough notes:
    1.  What problem did they solve, on what data? (how many images, which crop, field or lab photos)
    2.  What method did they use? (which features or which model)
    3.  What was their main result? (one number, for example "96% accuracy on 4 diseases")
    4.  What is one real strength of the approach?
    5.  What is one real limitation? (for example: only clean backgrounds, small dataset, slow)
    6.  What does it mean for **our** problem? (white vs field backgrounds, our 5 rice diseases)
3.  **Write one paragraph** answering all 6 questions in that order. Use this skeleton:
    <div class="skel">

    \[Author surname\] et al. \[N\] addressed \[problem\] using \[dataset: size, crop, field or lab photos\]. They \[method, in 1 to 2 sentences\]. They reported \[main result with a number\]. A strength of this approach is \[strength\]. However, \[limitation\]. For our problem, this suggests \[relevance to white vs field backgrounds or our 5 diseases\].

    </div>

    Replace `[N]` with your paper number (Paper 1 = \[1\]). Anika corrects numbering later.
4.  **Be concrete.** Use names and numbers from the paper.
    | Too vague (avoid)            | Concrete (do this)                                                      |
    |------------------------------|-------------------------------------------------------------------------|
    | "The method performed well." | "The method reached 94.2% accuracy on 3,355 images."                    |
    | "They used deep learning."   | "They fine-tuned an EfficientNet-B0 network."                           |
    | "It has some limitations."   | "It was only tested on leaves photographed against a plain background." |

    <span class="small">(The numbers in this table are examples only. Use your paper's real numbers.)</span>
5.  **Count the words.** 200 maximum.
6.  **Below the paragraph**, paste the paper's IEEE citation: on the paper's IEEE Xplore page click **Cite This**, choose **IEEE**, copy. This line does not count toward the 200 words.

#### Done when

- One paragraph, 200 words or fewer
- All 6 questions answered, in order
- At least 2 numbers taken from the paper, each checked against the PDF yourself
- No sentence copied from the paper or from an AI tool
- IEEE citation pasted underneath
- Saved with the correct file name in `papers/`, posted in the group chat

<div class="box ai">

<span class="t">Prompt for your AI tool (attach the paper PDF and this plan)</span>

<div class="prompt">

I need to write a 200-word summary of the attached paper for Section 4 (Related Work) of our group report. Please follow Section 7 of the attached plan. First, explain the paper to me in simple words. Then answer the 6 questions in Section 7 step 2 as short notes, quoting the page where each number appears so I can check it. Do NOT write the summary paragraph for me. After I write my own paragraph, I will paste it back: then check it against the 6 questions, the 200-word limit and the "Done when" list, and point out anything vague.

</div>

</div>

</div>

<div class="section page">

<div class="member-head">

## 8. Amit

3 tasks · A1 Mon 26 · A2 Wed 28 · A3 Fri 30

</div>

<div class="task">

### A1. Sample images

<div class="meta">

**Due****Mon 26 Oct** **Size limit**10 images **Hand in**`figures/samples/`: 10 images + `originals.txt`

</div>

#### Why

Section 3 shows one example of each disease on each background, in one figure that Akram builds from your images.

#### Steps

1.  Download the dataset from the link on the assignment's Moodle page and unzip it. **Do not move, rename or edit any file inside it.**
2.  Pick **10 images**: for each of the 5 diseases, 1 white-background and 1 field image. Choose images where the disease is **clearly visible**.
3.  **Copy** them (never move the originals) and rename the copies exactly:  
    `brownspot_white.jpg` `brownspot_field.jpg` `leafscald_white.jpg` `leafscald_field.jpg` `riceblast_white.jpg` `riceblast_field.jpg` `tungro_white.jpg` `tungro_field.jpg` `sheathblight_white.jpg` `sheathblight_field.jpg`
4.  In `originals.txt`, write each new name next to the original file name, one per line. Example: `brownspot_white.jpg = BS_W_017.jpg`

#### Done when

- 10 images, all 5 diseases x 2 backgrounds, named exactly as above
- `originals.txt` lists all 10 original names

</div>

<div class="task">

### A2. Summary of Paper 1

<div class="meta">

**Due****Wed 28 Oct** **Size limit**200 words **Instructions**Follow **Section 7** exactly

</div>

</div>

<div class="task">

### A3. Failure-case review

<div class="meta">

**Due****Fri 30 Oct, 12pm** **Size limit**6 images, one line of 15 words or fewer each, plus one pattern line of 25 words or fewer **Input**`wrong_predictions.csv` from Moinul on **Thu 29 Oct** (columns: file name, background, true disease, predicted disease) **Hand in**`figures/failures/`: 6 images + `failures.txt`

</div>

#### Why

Section 6 must explain when and why the method gets things wrong. It has only 2 pages, so every word counts.

#### Steps

1.  Open the images listed in `wrong_predictions.csv`.
2.  Pick **6**: **3 white-background and 3 field**. Prefer ones where you can see a likely reason for the mistake.
3.  Copy the 6 images into `figures/failures/` (keep their original names).
4.  In `failures.txt`, write one line per image: `file name | true | predicted | reason (15 words max)`. Example reasons: "Small dark spots look like Brown Spot", "Most of the frame is soil", "Photo is blurred".
5.  Add one last line (25 words max) with any pattern you notice, for example "4 of 6 mistakes confuse Rice Blast with Brown Spot".

#### Done when

- 6 images: 3 white, 3 field
- 6 lines, each reason 15 words or fewer
- 1 pattern line, 25 words or fewer

</div>

<div class="box ai">

<span class="t">Prompt for your AI tool (attach this plan)</span>

<div class="prompt">

I am Amit. My tasks are in Section 8 of the attached plan (A1 to A3), and A2 follows Section 7. Explain each of my tasks in simple words, tell me what I should do first, and what a good result looks like. Do not change any due date, size limit or file name. When I paste my work back, check it against the "Done when" list for that task. Do not write the reasons or summary for me.

</div>

</div>

</div>

<div class="section page">

<div class="member-head">

## 9. Akram

4 tasks · K1 Wed 28 · K2 Thu 29 · K4 Fri 30 · K3 Fri 30

</div>

<div class="task">

### K1. Summary of Paper 2

<div class="meta">

**Due****Wed 28 Oct** **Size limit**200 words **Instructions**Follow **Section 7** exactly

</div>

</div>

<div class="task">

### K2. Figures script

<div class="meta">

**Due****Thu 29 Oct** **Size limit**3 figures, each 18 cm wide (full text width) and **at most 6 cm tall** **Input**First real results from Moinul in the GitHub repo's `results/` folder on **Tue 27 Oct**; Amit's 10 images in `figures/samples/` **Hand in**`plots.py` as a GitHub pull request, plus the 3 test PNGs posted in the group chat

</div>

#### Why

The report needs 3 compact figures. Section 6 has only 2 pages, so each figure must be small and readable. Your script is kept separate from the experiment code; Moinul merges it into the main file afterwards.

#### Input file you will read

`results/confusion_resnet18_svm_<scenario>.csv`, where scenario is `white`, `field` or `mixed`. Each is a 5 x 5 table of counts: rows = true disease, columns = predicted disease, in this order: Brown Spot, Leaf Scald, Rice Blast, Rice Tungro, Sheath Blight. Full details are in `results/README.md` in the repo.

#### Steps

Write `plots.py` using **matplotlib only** (plus numpy and the standard library). It has these 3 functions. Each saves a 300 dpi PNG to `results/figures/`:

1.  `plot_confusion_matrices(results_dir)` saves `confusion.png`
    - ONE figure with **3 panels side by side**: (a) White, (b) Field, (c) Mixed
    - Disease names on both axes, the count written inside each cell, **one shared colour bar**
2.  `plot_sample_grid(samples_dir)` saves `samples.png`
    - ONE figure, **2 rows x 5 columns**: top row white background, bottom row field
    - Disease name above each column; row labels "White" and "Field" on the left
3.  `plot_failures(failures_dir, wrong_predictions_csv)` saves `failures.png`
    - ONE strip of 6 images: 3 white, then 3 field
    - Under each image: `true: X / pred: Y`
    - Build and test it now with any 6 images; the real ones arrive Fri 30

#### Rules

- No numbers typed in by hand. Everything is read from the files.
- Text at least 9pt when printed at 18 cm width.
- No chart titles (the report caption does that).
- Accuracy is reported in a table by Moinul, so **no accuracy bar chart**.

#### Done when

- All 3 functions run without errors on Moinul's first results and Amit's samples
- Each PNG is 18 cm wide and 6 cm tall or less
- Pull request opened, test PNGs posted in the group chat

</div>

<div class="task">

### K4. Clean-machine test of the User Manual

<div class="meta">

**Due****Fri 30 Oct** **Input**`docs/user_manual.docx` from Moinul on **Thu 29 Oct** **Hand in**`docs/clean_machine_test.docx`

</div>

#### Why

If the marker cannot run our code, our results count as not reproducible and we lose heavy marks. You act as the marker.

#### Steps

1.  Use a computer, or a brand new Python virtual environment, that has **never** run our code.
2.  Follow the User Manual **exactly as written**. Do not fix problems using your own knowledge; the marker won't either.
3.  For every numbered step, record in a table: **step number · pass or fail · exact error message if it failed**.
4.  Record the total run time.
5.  Post the result in the group chat on the same day, so Moinul can fix problems the same day.

#### Done when

- Every step of the manual tested and logged
- Every error copied exactly, total run time recorded

</div>

<div class="task">

### K3. Final figures

<div class="meta">

**Due****Fri 30 Oct (evening)** **Size limit**Same 3 figures, at most 6 cm tall each **Hand in**`results/figures/confusion.png`, `samples.png`, `failures.png`, posted in the group chat

</div>

#### Steps

1.  Run `plots.py` on Moinul's full results (ready Wed 28) and Amit's failure images (ready Fri 30, 12pm).
2.  Place each PNG in a Word page at 18 cm width and print it, or print to PDF at 100%.
3.  Check: every label readable, nothing overlapping, nothing cut off, colour bar visible.

</div>

<div class="box ai">

<span class="t">Prompt for your AI tool (attach this plan)</span>

<div class="prompt">

I am Akram. My tasks are in Section 9 of the attached plan (K1 to K4), and K1 follows Section 7. Explain each task in simple words and what a good result looks like. For K2, help me understand matplotlib subplots, shared colour bars and how to size a figure in centimetres, and help me debug my own code from error messages. Keep to the rules in K2 (matplotlib only, 3 functions with the given names, max 6 cm tall, no accuracy chart). Do not change any due date, size limit or file name.

</div>

</div>

</div>

<div class="section page">

<div class="member-head">

## 10. Sabir

2 tasks · S1 Wed 28 · S2 Wed 28

</div>

<div class="task">

### S1. Summary of Paper 3

<div class="meta">

**Due****Wed 28 Oct** **Size limit**200 words **Instructions**Follow **Section 7** exactly

</div>

</div>

<div class="task">

### S2. Installation instructions

<div class="meta">

**Due****Wed 28 Oct** **Size limit****1 page** (the whole User Manual is limited to 2 pages; Moinul writes the other page) **Input**`requirements.txt` in the GitHub repo **Hand in**`docs/install_instructions.docx`

</div>

#### Why

The report appendix must explain exactly how to install everything. If a step is missing, the marker cannot run our code and we lose heavy marks.

#### Steps

1.  Install **Python 3.12** from python.org if you don't have it. Note the exact download page.
2.  Create a fresh virtual environment and activate it:
    |          | Mac                        | Windows                 |
    |----------|----------------------------|-------------------------|
    | Create   | `python3.12 -m venv venv`  | `py -3.12 -m venv venv` |
    | Activate | `source venv/bin/activate` | `venv\Scripts\activate` |
3.  Download `requirements.txt` from the GitHub repo and run `pip install -r requirements.txt`
4.  Run `pip freeze` and write down the **exact version** of these 7 packages: numpy, matplotlib, opencv-contrib-python, scikit-learn, scikit-image, torch, torchvision.
5.  Find out where the ResNet-18 weights are saved when they download for the first time (search "torch hub cache directory"). Write down that folder for Mac and for Windows.
6.  Write the instructions as **numbered steps, one command per step**, with Mac and Windows commands side by side in a table. **No screenshots.**
7.  Add a **"Common problems"** list, 3 items maximum, for anything that went wrong for you and how you fixed it.

#### Done when

- Someone who has never used Python could follow it
- All 7 package versions listed
- Weight download folder given for Mac and Windows
- Fits on 1 page

</div>

<div class="box ai">

<span class="t">Prompt for your AI tool (attach this plan)</span>

<div class="prompt">

I am Sabir. My tasks are in Section 10 of the attached plan (S1 and S2), and S1 follows Section 7. Explain each task in simple words. For S2, explain what a virtual environment is and help me fix any installation errors I paste in. The instructions must be written by me, fit on 1 page, list the exact package versions I get from pip freeze (do not guess versions), and contain no screenshots. Do not change any due date, size limit or file name.

</div>

</div>

</div>

<div class="section page">

<div class="member-head">

## 11. Aziz

2 tasks · Z1 Wed 28 · Z2 Thu 29

</div>

<div class="task">

### Z1. Summary of Paper 4

<div class="meta">

**Due****Wed 28 Oct** **Size limit**200 words **Instructions**Follow **Section 7** exactly

</div>

</div>

<div class="task">

### Z2. Reference list in IEEE style

<div class="meta">

**Due****Thu 29 Oct** **Size limit**No limit **Input**The 4 paper PDFs in `papers/`, plus extra items Anika posts in the group chat (for example the dataset paper and the ResNet paper) **Hand in**`report/references.docx`

</div>

#### Why

References are worth 1 mark and must follow IEEE style exactly, as shown in Section 7 of the assignment brief.

#### Steps

1.  Collect every item to be referenced (the 4 papers and Anika's extra items).
2.  Get each paper's details from its IEEE Xplore page (**Cite This** \> **IEEE**), then make sure it matches these patterns from the brief:
    | Type             | Pattern                                                                                                         |
    |------------------|-----------------------------------------------------------------------------------------------------------------|
    | Conference paper | A. Author, B. Author, and C. Author, "Title of paper," in *Proceedings of Name of Conference*, Year, pp. xx-yy. |
    | Journal paper    | A. Author and B. Author, "Title of paper," *Journal Name*, vol. x, pp. xx-yy, Year.                             |
3.  Check every reference against these 5 points:
    1.  Author initials come **before** surnames (D. G. Bailey, not Bailey D.)
    2.  Paper title in double quotes, with a comma inside the closing quote
    3.  Conference or journal name included
    4.  Year included
    5.  Page numbers included (if the paper has them)
4.  List the references in any order. Anika numbers them to match the order they appear in the report.

#### Done when

- Every paper and extra item listed
- Every reference checked against all 5 points
- Every detail copied from the real source, not typed from memory or AI

</div>

<div class="box ai">

<span class="t">Prompt for your AI tool (attach this plan)</span>

<div class="prompt">

I am Aziz. My tasks are in Section 11 of the attached plan (Z1 and Z2), and Z1 follows Section 7. Explain each task in simple words. For Z2, explain IEEE reference style and check the references I paste against the 5 points in Z2 step 3. Do not add, invent or complete any reference details yourself: if something is missing, tell me what is missing so I can find it in the paper. Do not change any due date, size limit or file name.

</div>

</div>

</div>

<div class="section page">

<div class="member-head">

## 12. Anika (report lead)

Report sections 2 to 5, final report

</div>

| Due        | Task                                      | Size limit | Details                                                                                                                                                                                                                                         |
|------------|-------------------------------------------|------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Mon 26 Oct | Report template                           | n/a        | Google Doc with all 8 section headings, **each section starting on a new page**, 12pt font, 1.25 cm margins, page numbers starting from 1 after the coversheet                                                                                  |
| Mon 26 Oct | Literature search                         | n/a        | Find 6 IEEE Xplore papers (2022 or later) on rice or leaf disease classification from images. Save as `papers/paper1.pdf` to `paper6.pdf`. Assign Papers 1 to 4 in the group chat (Amit 1, Akram 2, Sabir 3, Aziz 4). Papers 5 and 6 are spares |
| Tue 27 Oct | Section 3 Problem Description             | 1 page     | About 330 words plus the sample-image figure. Uses Moinul's dataset inspection. Frames the task as image classification and argues for the method                                                                                               |
| Wed 28 Oct | Section 5 Method part                     | 2 pages    | Rationale for the method, block diagram, step-by-step algorithm in pseudocode                                                                                                                                                                   |
| Thu 29 Oct | Section 4 Related Work                    | 2 pages    | Edit the 4 summaries for consistent style; fix citation numbers. If a summary is missing or below standard, write one from Paper 5 or 6                                                                                                         |
| Sat 31 Oct | Review Section 6                          | n/a        | Check it against the brief's 6.1, 6.2 and 6.3 requirements                                                                                                                                                                                      |
| Sat 31 Oct | Executive Summary, coversheet, formatting | 0.5 page   | Summary of about 220 words covering Sections 3 to 6. Coversheet lists each member's finished tasks. Check every section is within its page limit                                                                                                |

<div class="member-head" style="margin-top:18pt">

## 13. Moinul (code lead)

All code, Section 6, submission

</div>

| Due            | Task                                                    | Size limit  | Details                                                                                                                                                                                                                                  |
|----------------|---------------------------------------------------------|-------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Sun 25 Oct     | Setup                                                   | n/a         | Share this plan; create the Google Drive folder and GitHub repo and post both links; download the dataset and check its folder structure                                                                                                 |
| Mon 26 Oct     | Dataset inspection, baseline                            | 1 page      | Image counts per disease and background, image sizes, 6 notes on what makes the images hard, saved to `dataset-notes/` for Anika. Data loading, 5 seeded stratified splits, colour + HOG + SVM on one split                              |
| Tue 27 Oct     | Main method                                             | n/a         | ResNet-18 features + SVM on one split; first results in `results/` for Akram                                                                                                                                                             |
| Wed 28 Oct     | All experiments                                         | n/a         | 2 methods x 3 scenarios x 5 splits                                                                                                                                                                                                       |
| Thu 29 Oct     | Implementation part, wrong-prediction list, User Manual | 1 page each | Implementation described in modules and pseudocode (no Python code in the report). `wrong_predictions.csv` for Amit. Merge Akram's `plots.py`. Combine run steps with Sabir's install page into `docs/user_manual.docx` for Akram's test |
| Fri 30 Oct     | Section 6.1 and 6.2 draft, fix test issues              |             | Settings table; results table (mean and std for each method and scenario). Fix anything Akram's clean-machine test found                                                                                                                 |
| Sat 31 Oct     | Code freeze, Section 6 final, final merge               | 2 pages     | Section 6.3 discussion using Amit's failure cases and Akram's figures. Final merge with Anika                                                                                                                                            |
| **Sat 31 Oct** | **Submit on Moodle**                                    |             | Zip containing only the `.py` file and the `.pdf` report (no images, no interim data)                                                                                                                                                    |

<div class="box note" style="margin-top:14pt">

<span class="t">Questions?</span> Ask in the group chat. If a task is unclear, ask on the day you receive it, not on the due date.

</div>

</div>
