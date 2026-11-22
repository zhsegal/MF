"""The lineage call for T/NK cells: one place that decides what a cell *is*.

Why this module exists. `cell_type_final`/`cell_type_T` is an else-branch. nb21 (skin) and nb31
(blood) hand-mapped Leiden clusters off dot plots and gave `CD4` to every T cluster that was not
positively called something else -- in blood the `CD4` transcript was not even in the object's
10k HVG when the call was made. `reports/ccc52_mvb/qc_benign_arm/` measured the damage: 21.9% of
the skin benign CD4 arm sat in clusters with CD8A/B detection >= 0.25, plus a Treg admixture, and
the contaminants are real cells rather than ambient or doublets, so no filter on those clusters
could remove them.

`jobs/run_nonmal_T_reannot.py` rebuilt the call from a scVI embedding that the malignant cells
never entered, and wrote `data/atlas_joint/nonmal_T_labels_v1.parquet`. This module is the only
thing that reads it, so "which lineage is this cell" has exactly one answer in the codebase.

Three facts that shape the API:

1. **Only non-malignant cells are relabelled.** The parquet holds 576,188 cells (257,297 skin,
   318,891 blood) and malignant cells are absent by construction. They keep their old `CD4`, so
   `"CD4"` stays in `CD4_ARM` -- without it the malignant arm would vanish.
2. **The new vocabulary contains no `"CD4"`.** Anything that tested `== "CD4"` sees zero cells
   once the overlay lands. That is why `malignancy.cd4_level` takes its CD4 set from here
   instead of hardcoding a literal, and why both changes had to ship together.
3. **`mal_cluster` must not see these labels.** `cluster_malig_helpers.CELL_TYPES` and `MIN_CD4`
   define a benign cluster by its CD4 *composition*, and `jobs/run_cluster_malig.py` masks on
   `cell_type_T == "CD4"`. Feeding the new labels back in would change the malignancy call that
   the new labels were derived from. `read_obs(..., lineage=False)` is the opt-out, and the
   cluster-malignancy jobs use it.

Arm definition. The benign CD4 arm is `CD4_conv + CD4_naive` in skin and `CD4_conv` alone in
blood -- the same asymmetry v1 had, where the blood clean-up dropped naive clusters and skin kept
them. `Mixed`, `LowQC` and `Suspect_malignant` are never part of any arm: they are the cells the
re-annotation could not resolve, or that look clone-like by donor and TRBV dominance.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

NB_DIR = Path(__file__).resolve().parent.parent
ATLAS_JOINT = NB_DIR / "data" / "atlas_joint"

LABELS_PQ = ATLAS_JOINT / "nonmal_T_labels_v1.parquet"
LABELS_VERSION = "nonmal_T_labels_v1"
OLD_SOURCE = "nb21_nb31"

# the vocabulary jobs/run_nonmal_T_reannot.py emits
VOCAB = ("CD4_conv", "CD4_naive", "CD4_Treg", "CD8", "gdT", "MAIT", "NK_ILC",
         "Prolif", "LowQC", "Mixed", "Suspect_malignant")

# the benign CD4 arm, per tissue (see the module docstring)
CD4_BENIGN = {"skin": ("CD4_conv", "CD4_naive"), "blood": ("CD4_conv",)}

# "CD4" is the OLD label, still carried by the malignant cells, which are absent from the parquet
CD4_ARM = {t: frozenset({"CD4"} | set(v)) for t, v in CD4_BENIGN.items()}

# never an arm: unresolved, low quality, or clone-like
NOT_AN_ARM = frozenset({"Mixed", "LowQC", "Suspect_malignant"})

# every conventional CD4 cell, ignoring the per-tissue naive convention. Use this where the
# question is "is this a CD4 cell" (healthy-donor references, lineage masks) rather than
# "is this in the benign arm of the malignant-vs-benign contrast".
CD4_ANY = frozenset({"CD4", "CD4_conv", "CD4_naive"})

EXPECTED_CELLS = {"skin": 257_297, "blood": 318_891}

_CACHE = None


def load_labels():
    """The label table, indexed by cell_id. Cached: it is 576,188 rows and never changes."""
    global _CACHE
    if _CACHE is None:
        if not LABELS_PQ.exists():
            raise FileNotFoundError(
                f"{LABELS_PQ} is missing; it is written by jobs/run_nonmal_T_reannot.py")
        _CACHE = pd.read_parquet(
            LABELS_PQ, columns=["cell_id", "tissue", "cluster_new", "label_new"]
        ).set_index("cell_id")
    return _CACHE


def cd4_set(tissue):
    """The `cell_type_T` values that count as CD4 for arm construction in this tissue."""
    if tissue not in CD4_ARM:
        raise ValueError(f"unknown tissue {tissue!r}; have {sorted(CD4_ARM)}")
    return CD4_ARM[tissue]


def apply_lineage(obs, tissue, col="cell_type_T", strict=False):
    """Overlay the re-annotated lineage onto `col`, keeping the old call alongside it.

    Mirrors `cluster_malig_helpers.apply_celltype_fix`: cell_id-keyed, non-destructive, and a
    no-op on a frame that does not carry the column. The original lands in
    `cell_type_T_pre_lineage` and every row gets a `lineage_source` so any table can show which
    call it used and how many cells moved.

    `strict=True` asserts the full expected overlap for the tissue -- use it when the frame is
    meant to be the whole compartment, not a subset.
    """
    if col not in obs.columns:
        return obs
    lab = load_labels()
    lab = lab[lab["tissue"].astype(str) == str(tissue)]
    hit = obs.index.intersection(lab.index)
    if strict and len(hit) != EXPECTED_CELLS.get(tissue, len(hit)):
        raise AssertionError(
            f"{tissue}: {len(hit):,} cells matched the lineage table, expected "
            f"{EXPECTED_CELLS[tissue]:,} -- is this frame the whole compartment?")
    obs = obs.copy()
    # the h5ad stores cell_type_T as a Categorical, and the new vocabulary introduces categories
    # it does not have ("Cannot setitem on a Categorical with a new category"). Everything
    # downstream does .astype(str) on this column anyway, so settle it as strings here.
    obs[col] = obs[col].astype(str)
    obs["cell_type_T_pre_lineage"] = obs[col].astype(str)
    obs["lineage_source"] = OLD_SOURCE
    if len(hit):
        obs.loc[hit, col] = lab.loc[hit, "label_new"].to_numpy()
        obs.loc[hit, "lineage_source"] = LABELS_VERSION
        obs["cluster_new"] = pd.Series(lab["cluster_new"]).reindex(obs.index).to_numpy()
    return obs


def arm_mask(cell_type, tissue, benign_only=False):
    """Boolean mask over a `cell_type_T` series: is this cell in the CD4 arm of `tissue`?"""
    ct = pd.Series(cell_type).astype(str).to_numpy()
    want = set(CD4_BENIGN[tissue]) if benign_only else set(cd4_set(tissue))
    return np.isin(ct, list(want))


def summarize(obs, col="cell_type_T"):
    """Cells per label and per source -- what every job should print after the overlay."""
    if "lineage_source" not in obs.columns:
        return obs[col].astype(str).value_counts().rename("n_cells").to_frame()
    return (obs.groupby(["lineage_source", obs[col].astype(str)]).size()
            .rename("n_cells").reset_index().rename(columns={"level_1": col}))
