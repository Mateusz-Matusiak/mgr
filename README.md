# Master's thesis — automatic detection of lumbar spine pathologies

This repository holds two things that belong to one project:

1. **The thesis itself** — a LaTeX document written in Polish (`main.tex` + `rozdzialy/`).
2. **The algorithms it describes** — Python code in `spine-ls-mgr/` that measures scoliosis
   (Cobb angle) and spondylolisthesis / retrolisthesis (sagittal slip) from 3D NIfTI
   segmentations of the lumbar spine.

Everything user-facing (thesis text, working notes, code output labels) is in Polish;
this README is the English entry point.

---

## Repository map

```
praca_magisterska/
├── main.tex                      # LaTeX root: preamble, page layout, \input of chapters
├── rozdzialy/                    # Chapters
│   ├── 1_wstep.tex               # 1. Introduction (topic, significance, medical imaging)
│   ├── 2_skolioza.tex            # 2. Scoliosis algorithm (clinical background, method, complexity)
│   ├── 3_kregozmyk_tylozmyk.tex  # 3. Spondylolisthesis / retrolisthesis algorithm
│   ├── 4_analiza_rezultatow.tex  # 4. Results analysis (stub)
│   └── 5_podsumowanie.tex        # 5. Conclusions (stub)
├── bibliography.bib              # Bibliography (biblatex/biber)
├── obrazy/                       # Figures used by the chapters (mostly 3D Slicer screenshots)
├── strona_tytulowa.pdf           # Title page, included verbatim via \includepdf
├── main.pdf                      # Last build output (git-ignored)
│
├── spine-ls-mgr/                 # Python component — see "Algorithms" below
│
├── algorithms_results.xlsx       # Batch measurement results for all cases
├── todo.txt                      # Open items: code fixes, missing figures, known limitations
├── uwagi_promotora.md            # Supervisor's PDF annotations, exported to markdown
├── sugestie_praca.md / .html     # Review of the whole thesis: gaps, suggestions
├── wymagania_kregozmyk.md        # Draft of section 3.2 (functional/non-functional requirements)
├── obrazy_do_zrobienia_kregozmyk.md  # Which 3D Slicer screenshots still need to be captured
├── rozdzial_1_1_rozbudowa.md     # Drafts / expansions, one file per section
├── rozdzial_1_2_rozbudowa.md
├── rozdzial_1_3_rozbudowa.md
├── rozdzial_2_1_poprawki.md      # Corrections, one file per section
├── rozdzial_2_2_poprawki.md
├── rozdzial_2_3_poprawki.md
├── rozdzial_3_1_poprawki.md
├── rozdzial_3_2_poprawki.md
└── kregozmyk_diff.html           # Rendered diff of an earlier revision of chapter 3
```

The `*_rozbudowa.md` / `*_poprawki.md` files are **staging documents**, not part of the
thesis. Each contains a rewritten version of one section, marked up with inline comments
(`% --- NOWE ---` for added text, `% FIX` for corrections, `% DO WERYFIKACJI` for things
needing the author's decision), ready to be pasted into the corresponding `.tex` file once
accepted.

---

## Building the thesis

```bash
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

Requires a TeX distribution with `biblatex`/`biber`, `polski`/`babel`, `pdfpages`,
`subcaption`, `listings`. Auxiliary files and `main.pdf` are git-ignored.

Two useful switches live in the preamble of `main.tex`: `\SHOWCOMMENTS` and
`\SHOWMODIFICATIONS`. While defined, `\comment{...}` renders in violet and
`\replace{old}{new}` shows the old text struck through in red next to the new text in blue.
Comment those two `\newcommand*` lines out to get a clean reading copy.

---

## Algorithms (`spine-ls-mgr/`)

```
spine-ls-mgr/
├── algorithms/
│   ├── scoliosis.py          # Cobb angle — takes the NIfTI path as argv[1]
│   ├── kregozmyk.py          # Current spondylolisthesis/retrolisthesis algorithm
│   ├── legacy_kregozmyk.py   # Earlier revision of the same algorithm, kept for reference
│   ├── slicer.py             # Helper: serialise points/vectors to 3D Slicer Markups JSON
│   ├── tonifti.py            # DICOM → NIfTI conversion (paths hardcoded, edit before use)
│   └── target/               # Output directory (JSON markups, debug volumes) — git-ignored
├── batch_analysis.py         # Runs scoliosis.py over every data/case*/seg.nii.gz
├── requirements.txt
└── similar-solutions.txt     # Links to comparable commercial products
```

### Input data

The NIfTI segmentations are **not** in this repository. They live outside it, at:

```
~/Desktop/Projects/spine-ls-mgr/data/case<N>/seg.nii.gz    # 69 cases
```

`kregozmyk.py` and `legacy_kregozmyk.py` hardcode a path to that location (case 11 and case
3 respectively) — edit the `sitk.ReadImage(...)` line near the bottom of the file before
running. `batch_analysis.py` instead expects a `data/` directory next to itself, so symlink
it once:

```bash
ln -s ~/Desktop/Projects/spine-ls-mgr/data spine-ls-mgr/data
```

### Setup

```bash
cd spine-ls-mgr
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install scikit-learn      # used by scoliosis.py, missing from requirements.txt
```

### Running

```bash
# Single case, Cobb angle — prints e.g. "th12-l3: 14.207 prawostronna"
cd spine-ls-mgr/algorithms
python3 scoliosis.py ~/Desktop/Projects/spine-ls-mgr/data/case64/seg.nii.gz

# Single case, slip between consecutive vertebrae (path hardcoded inside the file)
python3 kregozmyk.py

# All cases, Cobb angle
cd spine-ls-mgr
python3 batch_analysis.py
```

Both scripts must be run **from inside `algorithms/`** — they write to a relative
`./target/` path.

### What the algorithms do

Both share the first three stages of the pipeline; they diverge only at the measurement step.

1. **Spinal canal centreline** (`prepare_spinal_canal_points`) — labels 31/32/33 are merged
   into one mask, morphologically closed, and skeletonized (Lee's method, scikit-image).
   The skeleton is resampled along its oriented bounding box; the centroid of each slice
   gives one 3D point. Missing slices are filled by cubic spline interpolation.
2. **Per-vertebra local frame** — for each vertebra the nearest canal point is found, and an
   orthonormal frame is built from a PCA of the surrounding canal points plus the
   centroid→canal vector: `x` anterior-posterior, `y` superior-inferior, `z` medial-lateral.
3. **Endplate corner extraction** — the vertebra is resampled into that local plane; Canny
   edges → convex hull → `approxPolyDP` yield four corner points, back-projected into
   physical (LPS) space.
4. **Measurement:**
   - *Scoliosis* (`scoliosis.py`) — every vertebra pair from th11 to l5 is evaluated;
     endplate vectors are projected onto a common SVD-fitted plane, and the largest signed
     Cobb angle wins.
   - *Slip* (`kregozmyk.py`) — for each consecutive pair, the anterior slip
     (*kręgozmyk*) or posterior slip (*tyłozmyk*) is reported as a normalised scalar
     projection of the offset onto the lower endplate; the sign of the dot product decides
     which of the two it is.

### Label constants

| Structure | Label |
|---|---|
| th11 | 1 |
| th12 | 5 |
| l1 | 9 |
| l2 | 13 |
| l3 | 17 |
| l4 | 21 |
| l5 | 25 |
| s1 | 29 |
| dural_sac | 31 |
| spinal_canal | 32 |
| cauda_equina | 33 |

### Inspecting the output

Every intermediate geometric object is written to `algorithms/target/` as a
[3D Slicer Markups](https://slicer.readthedocs.io/) JSON file in the **LPS** coordinate
system, so it overlays directly on the source `seg.nii.gz`. Per vertebra you get
`contour.json`, `key_points.json`, `x_vector.json` / `y_vector.json` / `z_vector.json`,
`AB_VECTOR.json`, `AP_VECTOR.json`, `mapped_point.json`; at the top level,
`canal_points.json`, `approx.json` and `skeleton.nii.gz`. Load them in 3D Slicer via
*Add data*. `obrazy_do_zrobienia_kregozmyk.md` lists which of these screenshots the thesis
still needs.

---

## Where to start

| If you want to… | Go to |
|---|---|
| Read the finished text | `main.pdf`, or build it from `main.tex` |
| See what is still open | `todo.txt` |
| See what the supervisor asked for | `uwagi_promotora.md` |
| Understand the scoliosis method | chapter 2 (`rozdzialy/2_skolioza.tex`) + `algorithms/scoliosis.py` |
| Understand the slip method | chapter 3 (`rozdzialy/3_kregozmyk_tylozmyk.tex`) + `algorithms/kregozmyk.py` |
| Look at the measured numbers | `algorithms_results.xlsx` |
