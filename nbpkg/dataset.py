"""
dataset.py — parse the REAL folder names and build a leakage-safe fruit index.

Discovery from the real tree: the data is longitudinal. One physical fruit was
scanned by (non-destructive) µCT at up to 4 days. Folder layout:

    data_root/Day{4,5,7,10}/<scan_folder>/vertical_section_*.jpg   (72 slices)

A <scan_folder> is ONE scan == one (fruit × day). Example names handled:
    CRI_D10_100_1_23306_01   CRI_D10_100-1_23144_01   CRI_4_100_1_23378_01
    CRI_D4_50_2_23223.01     CRI_D5_Ctrl_3_23206_01   CRI_D7_cntrl_5_23322_01

Two scan-ID cohorts (early 231xx-232xx, late 233xx-234xx) = two independent
groups of fruits. A physical fruit is therefore identified by (cohort, dose, rep)
and appears across ~4 day-scans. Splitting by scan/day would leak a fruit's
identity across timepoints, so we split by the physical-fruit key.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

CONTROL_KEYS = ("cntrl", "contr", "ctrl", "control")
# CRI_[D]<day>_<dose|ctrl>[-_]<rep>_<scanid>...
SCAN_RE = re.compile(
    r"CRI_D?(?P<day>\d+)_(?P<dose>\d+|cntrl|contr|ctrl|control)[-_](?P<rep>\d+)_(?P<scan>\d+)",
    re.IGNORECASE)

# scan-ID threshold separating the two cohorts (gap sits between 23233 and 23294)
COHORT_SPLIT_SCANID = 23260


@dataclass
class Scan:
    day: int
    scan_id: int
    directory: Path


@dataclass
class Fruit:
    fruit_id: str          # e.g. "A_early|dose100|rep1"  — the split/grouping key
    cohort: str
    dose: str              # "50" | "100" | "500" | "ctrl"
    rep: int
    label: int             # 1 infested, 0 control
    scans: list = field(default_factory=list)   # list[Scan], one per day


def parse_scan_name(name: str):
    m = SCAN_RE.match(name)
    if not m:
        return None
    dose_tok = m.group("dose").lower()
    is_ctrl = dose_tok in CONTROL_KEYS
    scan_id = int(m.group("scan"))
    return {
        "day": int(m.group("day")),
        "dose": "ctrl" if is_ctrl else dose_tok,
        "rep": int(m.group("rep")),
        "scan_id": scan_id,
        "label": 0 if is_ctrl else 1,
        "cohort": "B_late" if scan_id >= COHORT_SPLIT_SCANID else "A_early",
    }


def build_fruit_index(data_root, *, day_glob="Day*", fruit_key="cohort_dose_rep",
                      verbose=True):
    """Scan data_root/Day*/<scan_folder> and group scans into physical fruits.

    fruit_key selects the grouping (i.e. the unit kept whole across a split):
      * "cohort_dose_rep" (default, safest): one fruit = (cohort, dose, rep),
        all its day-scans grouped -> no identity leakage across timepoints.
      * "scan": one fruit == one scan folder (the OLD, leaky behaviour; kept only
        for comparison — do NOT use for the reported evaluation).
    """
    data_root = Path(data_root)
    groups = defaultdict(list)
    unparsed = []
    for day_dir in sorted(data_root.glob(day_glob)):
        if not day_dir.is_dir():
            continue
        for scan_dir in sorted(p for p in day_dir.iterdir() if p.is_dir()):
            info = parse_scan_name(scan_dir.name)
            if info is None:
                unparsed.append(scan_dir.name)
                continue
            if fruit_key == "cohort_dose_rep":
                fid = f"{info['cohort']}|dose{info['dose']}|rep{info['rep']}"
            elif fruit_key == "scan":
                fid = scan_dir.name
            else:
                raise ValueError(fruit_key)
            groups[fid].append((info, Scan(info["day"], info["scan_id"], scan_dir)))

    fruits = []
    for fid, items in groups.items():
        info0 = items[0][0]
        fruits.append(Fruit(
            fruit_id=fid, cohort=info0["cohort"], dose=info0["dose"],
            rep=info0["rep"], label=info0["label"],
            scans=sorted((s for _, s in items), key=lambda s: s.day)))

    if verbose:
        n_inf = sum(f.label for f in fruits)
        n_scan = sum(len(f.scans) for f in fruits)
        print(f"parsed {n_scan} scans -> {len(fruits)} physical fruits "
              f"({n_inf} infested / {len(fruits) - n_inf} control)")
        if unparsed:
            print(f"  UNPARSED ({len(unparsed)}):", unparsed[:5],
                  "..." if len(unparsed) > 5 else "")
    return fruits, unparsed


def slice_paths(scan_dir):
    """All slice files of a scan (order irrelevant for voting)."""
    p = Path(scan_dir)
    return sorted(str(x) for x in p.glob("*.jpg")) or \
           sorted(str(x) for x in p.glob("*.png"))


def fruit_scan_slices(fruit, *, days=None):
    """Yield (slice_path, label, scan_id) for a fruit, optionally restricted to
    certain days. Used for training (all days pooled) and per-scan evaluation."""
    for sc in fruit.scans:
        if days is not None and sc.day not in days:
            continue
        for sp in slice_paths(sc.directory):
            yield sp, fruit.label, sc.scan_id
