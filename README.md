# Citrus infestation — deep-learning pipeline (leakage-free, tuned to your real data)

Step-by-step notebooks + a shared `nbpkg/` package, built and tested against your
actual folder tree (`tree_citrus.txt`).

## Definitive inventory (your data, ground truth)

Parsed from all your folders — this is the answer to the reviewer's count question:

* **189 individual fruit scanned**, each with **exactly 72 slices** → **13,608 images**.
* **141 infested / 48 control.**
* **2 fly-cohort batches** (scan-ID early 231xx–232xx / late 233xx–234xx), balanced
  ~23–24 per cohort × dose (50/100/500/ctrl).
* Per day: Day4=48, Day5=48, Day7=47, Day10=46.

Design is **cross-sectional**: *"six fruit randomly selected from each density and each
incubation period were scanned; each scan corresponded to one individual fruit."* So
**one folder = one distinct fruit** (not the same fruit over time). See `dataset_inventory.csv`.

Reconciliation of the three numbers that circulated: **189** scanned (what you have) →
**111** curated in the manuscript (subset after QC — criteria must be documented) →
**48** in the presentation was a miscount (cohort×dose×rep cells collapsed over days;
48×4×72=13,824 is the idealised image count, real is 13,608). The text also claims **3
replicates** but only **2 cohorts** are present in the data — clarify this for comment 2.

## Package

```
nbpkg/
  config.py         <- EDIT: data_root, curated_root, img_size(=128), fruit_key(="scan")
  dataset.py        <- parses your real names, one fruit = one scan (cross-sectional)
  slice_curation.py <- reproducible training-only slice cleaning (replaces manual step)
  citrus_dl.py      <- TensorFlow: data pipeline, backbones, training, Optuna
  eval_core.py      <- split, voting, threshold, metrics, cluster-bootstrap CIs
  selftest_core.py
00_data_prep.ipynb         [CPU] index, QC (72 slices), class balance, view slices
01_partitioning.ipynb      [CPU] fruit-level split + repeated group CV, leakage checks
02a_slice_curation.ipynb   [CPU] damage-score slice cleaning (train only) + human review
02_train.ipynb             [GPU] Optuna + train 4 backbones, save per-scan scores
03_voting_and_table3.ipynb [CPU] threshold on val, Table 3 with CIs
04_repeated_cv.ipynb       [GPU] 5x3 group CV, mean ± SD
```

## Run order

```
00 -> 01 -> 02a -> (set CFG.curated_root) -> 02 -> 03      # single split -> Table 3
00 -> 01 -> 02a -> 04                                      # repeated CV -> mean ± SD
```

## The slice-cleaning step (02a) — read this

The student's good Table 3 numbers depended on **manually deleting infested slices with
no visible signs**. Manual deletion is subjective and not reproducible — a weakness a
code-reading reviewer will flag. `02a` rebuilds it objectively:

* a per-slice **damage score** = fraction of interior pulp pixels that are cavity-dark
  (background and the central artificial column excluded);
* threshold = a percentile of **training-control** slice scores (calibrated without any
  test fruit or infested labels → no leakage);
* among **training infested** fruit only, slices below threshold are dropped; controls
  keep all; infested fruit with too few damaged slices are kept via a top-k fallback and
  **flagged as early-detection** cases;
* every decision is logged to `curation_manifest.csv` — edit the `kept` column to
  override, then re-run the last cell (human-in-the-loop).

Evaluation always uses the full 72 slices; curation touches training only. A more
rigorous long-term option is Multiple-Instance Learning (removes slice cleaning
entirely); noted in `slice_curation.py`.

## How each reviewer inconsistency is handled

1. **Protocol** — 03 (single split, point [95% CI]) vs 04 (repeated 5x3 CV, mean ± SD).
2. **Counts / perfect recall** — `dataset_inventory.csv` gives exact numbers; metrics carry CIs.
3. **Leakage / three-way partition** — split by fruit; 01 asserts no scan of a test fruit
   in train/val; threshold fixed on validation only.
4. **Clustering** — CIs can be cluster-bootstrapped; note the fly-cohort/box structure for
   the logistic-regression analysis.

## Validated

00/01/02a/03 were executed end-to-end against a reconstruction of your real tree
(189 scans -> 189 fruit -> 105/27/57 split; curation produced a manifest + curated tree;
Table 3 with CIs). `python nbpkg/selftest_core.py` re-checks the statistics.
