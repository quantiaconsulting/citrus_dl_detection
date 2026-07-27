"""
eval_core.py
============
Framework-agnostic evaluation logic for the citrus infestation study.

This module deliberately contains NO deep-learning code so that the parts the
reviewer is worried about -- (i) fruit-level partitioning without leakage,
(ii) vote-threshold selection on the validation set only, and (iii) confidence
intervals on every metric -- can be unit-tested independently of TensorFlow and
without the image data.

Nomenclature
------------
A "fruit" is the unit of analysis. Each fruit has up to 72 axial slices.
* Training uses the *curated* (manually cleaned) slices of the TRAIN fruits.
* Early stopping monitors slice-level validation AUC on the VALIDATION fruits.
* The voting threshold is chosen on the VALIDATION fruits using the *full*
  72-slice protocol (identical to how the test set is scored) -> no test leakage.
* Final metrics + CIs are computed on the held-out TEST fruits only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Sequence
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


# --------------------------------------------------------------------------- #
# 1. Fruit-level partitioning (no slice of a fruit may cross partitions)
# --------------------------------------------------------------------------- #
def make_three_way_split(fruit_ids, labels, *, test_frac=0.30,
                         val_frac_of_trainval=0.20, seed=42):
    """Split *whole fruits* into train / val / test.

    Stratification is on the binary `labels` (infested/control) so the class
    balance is preserved across partitions. We deliberately do NOT stratify on
    the finer day x dose cells: with ~48-111 fruits those cells hold too few
    members to stratify reliably (this small-sample fragility is itself worth
    stating in the paper). Because we split fruit IDs -- not slices -- no fruit
    can appear in more than one partition, so there is no leakage.

    Returns three arrays of fruit_ids.
    """
    fruit_ids = np.asarray(fruit_ids)
    labels = np.asarray(labels)

    # test split
    tv_idx, te_idx = train_test_split(
        np.arange(len(fruit_ids)), test_size=test_frac,
        stratify=labels, random_state=seed)
    # val split carved out of the remaining train+val fruits (never touches test)
    tr_idx, va_idx = train_test_split(
        tv_idx, test_size=val_frac_of_trainval,
        stratify=labels[tv_idx], random_state=seed)

    return fruit_ids[tr_idx], fruit_ids[va_idx], fruit_ids[te_idx]


def two_way_split(fruit_ids, labels, *, val_frac=0.20, seed=42):
    """Simple stratified train/val split of whole fruits (used for the inner
    validation set inside each CV fold: early stopping + threshold selection)."""
    fruit_ids = np.asarray(fruit_ids); labels = np.asarray(labels)
    tr_idx, va_idx = train_test_split(
        np.arange(len(fruit_ids)), test_size=val_frac,
        stratify=labels, random_state=seed)
    return fruit_ids[tr_idx], fruit_ids[va_idx]


def repeated_group_folds(fruit_ids, labels, *, n_splits=5, n_repeats=3, seed=42):
    """Yield (train_fruit_ids, test_fruit_ids) for repeated stratified *group*
    k-fold, grouping by fruit. This is the statistically sound protocol for the
    small sample here and produces the mean +/- SD the reviewer asks for.

    Groups == fruit_ids guarantees all slices of a fruit stay together.
    """
    fruit_ids = np.asarray(fruit_ids)
    labels = np.asarray(labels)
    for rep in range(n_repeats):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    random_state=seed + rep)
        # X is a dummy; grouping is by fruit index so each fruit is its own group
        for tr, te in sgkf.split(np.zeros(len(fruit_ids)), labels,
                                 groups=np.arange(len(fruit_ids))):
            yield rep, fruit_ids[tr], fruit_ids[te]


def assert_no_leakage(*partitions):
    """Raise if any fruit appears in more than one partition."""
    seen = {}
    for name, ids in partitions:
        for f in ids:
            if f in seen:
                raise AssertionError(
                    f"Leakage: fruit {f!r} in both {seen[f]!r} and {name!r}")
            seen[f] = name


# --------------------------------------------------------------------------- #
# 2. Slice -> fruit aggregation (the voting system)
# --------------------------------------------------------------------------- #
@dataclass
class FruitScores:
    """Per-unit aggregated scores over slices. A "unit" is one scan (fruit x day)
    when evaluating per scan; `group` carries the PHYSICAL-fruit id so confidence
    intervals can be bootstrapped by fruit (cluster-aware) rather than by scan."""
    fruit_id: object
    y_true: int                 # 1 = infested, 0 = control
    n_slices: int
    n_votes_infested: int       # slices with p >= slice_decision_threshold
    mean_prob: float            # mean sigmoid probability across the unit
    group: object = None        # physical-fruit id (for cluster bootstrap)


def aggregate_fruit(fruit_id, y_true, slice_probs, *, slice_decision=0.5, group=None):
    slice_probs = np.asarray(slice_probs, dtype=float)
    votes = int(np.sum(slice_probs >= slice_decision))
    return FruitScores(fruit_id, int(y_true), len(slice_probs),
                       votes, float(slice_probs.mean()), group)


# --------------------------------------------------------------------------- #
# 3. Vote-threshold selection -- VALIDATION ONLY
# --------------------------------------------------------------------------- #
def select_vote_threshold(val_scores: Sequence[FruitScores], *, objective="f1"):
    """Choose the integer vote threshold that maximises `objective` on the
    VALIDATION fruits. A fruit is called infested if n_votes_infested > threshold.

    Ties are broken towards the *larger* threshold (more conservative about
    calling a fruit infested). Returns (best_threshold, best_value).
    """
    y = np.array([s.y_true for s in val_scores])
    votes = np.array([s.n_votes_infested for s in val_scores])
    max_votes = int(votes.max()) if len(votes) else 1

    best_t, best_v = 0, -1.0
    for t in range(0, max_votes + 1):
        pred = (votes > t).astype(int)
        v = _binary_objective(y, pred, objective)
        if v >= best_v:          # >= -> prefers larger t on ties
            best_v, best_t = v, t
    return best_t, best_v


def _binary_objective(y_true, y_pred, objective):
    tp, fp, tn, fn = _confusion(y_true, y_pred)
    if objective == "f1":
        return _f1(tp, fp, fn)
    if objective == "youden":            # sensitivity + specificity - 1
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        return sens + spec - 1.0
    if objective == "accuracy":
        n = tp + fp + tn + fn
        return (tp + tn) / n if n else 0.0
    raise ValueError(objective)


# --------------------------------------------------------------------------- #
# 4. Metrics + confidence intervals
# --------------------------------------------------------------------------- #
def _confusion(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp, fp, tn, fn


def _f1(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return (2 * tp) / denom if denom else 0.0


def _acc(y_true, y_pred):
    tp, fp, tn, fn = _confusion(y_true, y_pred); n = tp + fp + tn + fn
    return (tp + tn) / n if n else 0.0


def _prec(y_true, y_pred):
    tp, fp, tn, fn = _confusion(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) else 0.0


def _rec(y_true, y_pred):
    tp, fp, tn, fn = _confusion(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) else 0.0


def wilson_ci(successes, n, *, alpha=0.05):
    """Wilson score interval for a binomial proportion.

    Appropriate for accuracy, precision, recall (each is #successes / #trials).
    Returns (point, low, high). If n == 0 returns (nan, nan, nan).
    """
    if n == 0:
        return (float("nan"),) * 3
    z = stats.norm.ppf(1 - alpha / 2)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_ci(y_true, scores, metric_fn, *, n_boot=5000, alpha=0.05, seed=0):
    """Percentile bootstrap CI by resampling *units*.

    Use for AUC and F1, whose sampling distribution is not a simple binomial.
    `metric_fn(y_true_b, scores_b) -> float`. Resamples that yield a single
    class (metric undefined, e.g. AUC) are skipped.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); scores = np.asarray(scores)
    n = len(y_true)
    point = metric_fn(y_true, scores)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, sb = y_true[idx], scores[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            boots.append(metric_fn(yb, sb))
        except ValueError:
            continue
    if not boots:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def cluster_bootstrap_ci(y_true, scores, groups, metric_fn, *,
                         n_boot=5000, alpha=0.05, seed=0):
    """Cluster (block) bootstrap: resample whole GROUPS (physical fruits) with
    replacement, keeping all of a fruit's scans together. This is the honest CI
    when the evaluation units (scans) are clustered within fruit — it is wider
    than the naive bootstrap and reflects the repeated-measures structure the
    reviewer raised. `metric_fn(y_true_b, scores_b) -> float`."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); scores = np.asarray(scores)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    point = metric_fn(y_true, scores)
    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in chosen])
        yb, sb = y_true[idx], scores[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            boots.append(metric_fn(yb, sb))
        except ValueError:
            continue
    if not boots:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def evaluate_test(test_scores: Sequence[FruitScores], vote_threshold, *,
                  with_ci=True, n_boot=5000, cluster=True):
    """Compute the full metric set on the TEST units.

    Binary metrics use the voting threshold fixed on validation.
    AUC uses the continuous per-unit mean probability (threshold-independent).

    If the units carry a `.group` (physical-fruit id) and `cluster=True`, ALL
    confidence intervals are computed by cluster bootstrap over fruits (honest
    under repeated measures). Otherwise Wilson intervals (acc/prec/recall) and a
    naive bootstrap (AUC/F1) are used. `with_ci=False` returns point estimates.
    """
    y = np.array([s.y_true for s in test_scores])
    votes = np.array([s.n_votes_infested for s in test_scores])
    mean_prob = np.array([s.mean_prob for s in test_scores])
    groups = np.array([s.group for s in test_scores], dtype=object)
    pred = (votes > vote_threshold).astype(int)
    has_groups = cluster and any(g is not None for g in groups)

    tp, fp, tn, fn = _confusion(y, pred)
    n = tp + fp + tn + fn

    def pt(key):
        if key == "accuracy": return (tp + tn) / n if n else float("nan")
        if key == "precision": return tp / (tp + fp) if (tp + fp) else float("nan")
        if key == "recall": return tp / (tp + fn) if (tp + fn) else float("nan")

    if not with_ci:
        acc = (pt("accuracy"), float("nan"), float("nan"))
        prec = (pt("precision"), float("nan"), float("nan"))
        rec = (pt("recall"), float("nan"), float("nan"))
        f1v = (_f1(tp, fp, fn), float("nan"), float("nan"))
        try: aucv = (roc_auc_score(y, mean_prob), float("nan"), float("nan"))
        except ValueError: aucv = (float("nan"),) * 3
    elif has_groups:
        acc_fn = lambda a, p: _acc(a, p)
        prec_fn = lambda a, p: _prec(a, p)
        rec_fn = lambda a, p: _rec(a, p)
        f1_fn = lambda a, p: _f1(*_confusion(a, p)[:2], _confusion(a, p)[3])
        acc = cluster_bootstrap_ci(y, pred, groups, acc_fn, n_boot=n_boot)
        prec = cluster_bootstrap_ci(y, pred, groups, prec_fn, n_boot=n_boot)
        rec = cluster_bootstrap_ci(y, pred, groups, rec_fn, n_boot=n_boot)
        f1v = cluster_bootstrap_ci(y, pred, groups, f1_fn, n_boot=n_boot)
        aucv = cluster_bootstrap_ci(y, mean_prob, groups, roc_auc_score, n_boot=n_boot)
    else:
        acc = wilson_ci(tp + tn, n)
        prec = wilson_ci(tp, tp + fp)
        rec = wilson_ci(tp, tp + fn)
        f1v = bootstrap_ci(y, pred,
                           lambda a, b: _f1(*_confusion(a, b)[:2], _confusion(a, b)[3]),
                           n_boot=n_boot)
        try: aucv = bootstrap_ci(y, mean_prob, roc_auc_score, n_boot=n_boot)
        except ValueError: aucv = (float("nan"),) * 3

    return {
        "vote_threshold": vote_threshold,
        "n_test_unit": len(y),
        "n_infested": int(y.sum()),
        "n_control": int((y == 0).sum()),
        "n_fruit_groups": int(len({g for g in groups if g is not None})) or len(y),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "AUC": aucv,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "F1": f1v,
    }


# --------------------------------------------------------------------------- #
# 5. Reporting
# --------------------------------------------------------------------------- #
def _fmt(triple, nd=3):
    p, lo, hi = triple
    if any(np.isnan(x) for x in (p, lo, hi)):
        return f"{p:.{nd}f}" if not np.isnan(p) else "n/a"
    return f"{p:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def results_to_frame(named_results: dict) -> pd.DataFrame:
    """named_results: {model_name: evaluate_test(...) dict}."""
    rows = []
    for model, r in named_results.items():
        rows.append({
            "Model": model,
            "Vote threshold": r["vote_threshold"],
            "Test scans (inf/ctrl)": f"{r['n_test_unit']} ({r['n_infested']}/{r['n_control']})",
            "Test fruits": r.get("n_fruit_groups", r["n_test_unit"]),
            "AUC [95% CI]": _fmt(r["AUC"]),
            "Accuracy [95% CI]": _fmt(r["accuracy"]),
            "Precision [95% CI]": _fmt(r["precision"]),
            "Recall [95% CI]": _fmt(r["recall"]),
            "F1 [95% CI]": _fmt(r["F1"]),
        })
    return pd.DataFrame(rows)


def cv_results_to_frame(cv_records: dict) -> pd.DataFrame:
    """cv_records: {model_name: list_of_per_fold_metric_dicts}.
    Reports mean +/- SD across folds for each metric (reviewer comment 1)."""
    rows = []
    for model, folds in cv_records.items():
        def col(key):
            vals = [f[key] for f in folds if not np.isnan(f[key])]
            return f"{np.mean(vals):.3f} +/- {np.std(vals, ddof=1):.3f}" if vals else "n/a"
        rows.append({
            "Model": model, "n_folds": len(folds),
            "AUC": col("AUC"), "Accuracy": col("accuracy"),
            "Precision": col("precision"), "Recall": col("recall"),
            "F1": col("F1"),
        })
    return pd.DataFrame(rows)
