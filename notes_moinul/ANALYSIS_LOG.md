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
