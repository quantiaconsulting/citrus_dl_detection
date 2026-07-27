"""config.py — single source of truth. Edit paths + the few flagged values."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Config:
    # ---- paths (EDIT) -------------------------------------------------------
    # Parent folder that contains Day4/ Day5/ Day7/ Day10/, each with scan folders.
    data_root: Path = Path("data")
    # Optional: parent of a curated copy (same Day*/scan layout) with cleaned
    # slices for TRAINING only. Leave as None to train on all slices.
    curated_root: Path | None = None
    out_dir: Path = Path("outputs")

    # ---- fruit identity / leakage control ----------------------------------
    # "cohort_dose_rep" (default, safe): one physical fruit = (cohort,dose,rep),
    #   all its day-scans grouped -> no identity leakage across timepoints.
    # "scan": one fruit == one scan folder (OLD leaky behaviour; comparison only).
    fruit_key: str = "scan"   # manuscript: each scan = one individual fruit (cross-sectional)

    # ---- image / model ------------------------------------------------------
    img_size: int = 128          # matches Methods (raw 256x256 are resized to 128)
    slices_per_fruit: int = 72

    candidate_backbones: tuple = ("EfficientNetB0", "MobileNetV2", "ResNet50",
                                  "NASNetMobile", "DenseNet121", "InceptionV3")
    final_backbones: tuple = ("MobileNetV2", "NASNetMobile",
                              "DenseNet121", "InceptionV3")

    # ---- training -----------------------------------------------------------
    batch_size: int = 32
    max_epochs: int = 20
    patience: int = 3
    finetune_unfreeze_last: int = 20
    slice_decision: float = 0.5
    threshold_objective: str = "f1"

    # ---- HPO ----------------------------------------------------------------
    hpo_n_trials: int = 25
    hpo_epochs: int = 12

    # ---- partitioning -------------------------------------------------------
    test_frac: float = 0.30
    val_frac_of_trainval: float = 0.20
    cv_n_splits: int = 5
    cv_n_repeats: int = 3
    n_boot: int = 5000
    seed: int = 42

    # ---- slice curation (training-only cleaning) ----------------------------
    curation_percentile: float = 95   # control-slice damage percentile = threshold
    curation_min_keep: int = 3         # min slices kept per infested fruit
    curated_dirname: str = 'curated_slices'   # written under out_dir

    hparams: dict = field(default_factory=dict)

CFG = Config()
