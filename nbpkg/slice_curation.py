"""
slice_curation.py — reproducible, training-only slice cleaning.

Replaces the student's manual "delete slices that don't show infestation" step
with an objective, documented criterion so it survives a code-reading reviewer.

Principle
---------
Infestation appears as dark cavities/tunnels inside the bright pulp. For each
slice we compute a `damage_score` = fraction of *interior* fruit pixels that are
markedly darker than the pulp (excluding background and the artificial central
column). We calibrate a threshold on CONTROL fruit from the TRAINING partition
only, then—among TRAINING infested fruit—keep for training only the slices whose
score exceeds that threshold. Control training fruit keep all slices.

Integrity guardrails
--------------------
* Curation touches TRAINING fruit only. Validation/test are always evaluated on
  the full 72 slices (this module never rewrites them).
* The threshold is calibrated on TRAIN control slices only — no infested labels
  and no test fruit are used, so there is no leakage.
* Every kept/dropped decision is logged to a manifest for human review/override.

A more rigorous long-term alternative is Multiple-Instance Learning (a fruit is
positive iff >=1 slice is positive), which removes the need for slice cleaning
altogether; noted here but not implemented.
"""
from __future__ import annotations
import os
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi


# --------------------------------------------------------------------------- #
def fruit_region(img, *, otsu_frac=0.6, min_area_frac=0.02, central_band_frac=0.12):
    """Boolean mask of the fruit interior (pulp + enclosed cavities), with the
    background and the artificial central black column removed."""
    H, W = img.shape
    thr = threshold_otsu(img) * otsu_frac
    bright = img > thr
    region = ndi.binary_fill_holes(bright)          # cavities become interior
    lbl, n = ndi.label(region)
    if n:
        sizes = ndi.sum(np.ones_like(lbl), lbl, range(1, n + 1))
        region = np.isin(lbl, 1 + np.where(sizes > min_area_frac * H * W)[0])
    band = int(central_band_frac * W); cx = W // 2
    region[:, cx - band: cx + band] = False
    return region


def damage_score(img, *, dark_frac_of_pulp=0.55, **mask_kw):
    """Fraction of interior fruit pixels that are cavity-dark. 0 = clean."""
    img = np.asarray(img, dtype=float)
    region = fruit_region(img, **mask_kw)
    if region.sum() == 0:
        return 0.0
    pulp_med = np.median(img[region])
    dark = region & (img < dark_frac_of_pulp * pulp_med)
    return float(dark.sum() / region.sum())


def _slice_files(scan_dir):
    p = Path(scan_dir)
    return sorted(p.glob("*.jpg")) or sorted(p.glob("*.png"))


def score_fruit(fruit, *, img_loader=None, **score_kw):
    """Return list of (slice_path, score) for every slice of a fruit's scans."""
    out = []
    load = img_loader or (lambda p: np.array(Image.open(p).convert("L"), float))
    for sc in fruit.scans:
        for sp in _slice_files(sc.directory):
            out.append((str(sp), damage_score(load(sp), **score_kw)))
    return out


# --------------------------------------------------------------------------- #
def calibrate_threshold(train_control_fruits, *, percentile=95, **score_kw):
    """Damage-score ceiling of 'normal', from TRAINING control slices only."""
    scores = []
    for f in train_control_fruits:
        scores += [s for _, s in score_fruit(f, **score_kw)]
    if not scores:
        raise ValueError("No control slices to calibrate on.")
    return float(np.percentile(scores, percentile)), np.asarray(scores)


def curate_training(train_fruits, threshold, *, min_keep=3, **score_kw):
    """Decide kept/dropped slices for TRAINING fruit.

    Control fruit: keep all. Infested fruit: keep slices with score > threshold;
    if fewer than `min_keep` pass, keep the top-`min_keep` (so no infested fruit
    is silently removed — early-detection cases are flagged instead).
    Returns a list of dict rows (one per slice) for the manifest.
    """
    rows = []
    for f in train_fruits:
        scored = score_fruit(f, **score_kw)
        if f.label == 0:
            for sp, sc in scored:
                rows.append(dict(fruit_id=f.fruit_id, slice=sp, score=sc,
                                 label=0, kept=True, reason="control_keep_all"))
            continue
        kept = [(sp, sc) for sp, sc in scored if sc > threshold]
        reason = "above_threshold"
        if len(kept) < min_keep:                    # early-detection fallback
            kept = sorted(scored, key=lambda t: t[1], reverse=True)[:min_keep]
            reason = "topk_fallback_flag_early_detection"
        keptset = {sp for sp, _ in kept}
        for sp, sc in scored:
            rows.append(dict(fruit_id=f.fruit_id, slice=sp, score=sc, label=1,
                             kept=sp in keptset,
                             reason=reason if sp in keptset else "below_threshold"))
    return rows


def build_curated_tree(manifest_rows, data_root, curated_root):
    """Materialise a curated tree (symlinks to KEPT slices only) mirroring the
    data_root/Day*/scan layout, so the training loader picks up only kept slices.
    Fruit/scans not present here fall back to full slices automatically."""
    data_root = Path(data_root); curated_root = Path(curated_root)
    n = 0
    for r in manifest_rows:
        if not r["kept"]:
            continue
        src = Path(r["slice"])
        rel = src.relative_to(data_root)
        dst = curated_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            try:
                os.symlink(src.resolve(), dst)
            except OSError:
                import shutil; shutil.copy(src, dst)
            n += 1
    return n
