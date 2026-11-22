"""The four malignancy calls for skin T cells, assembled in one place.

There are exactly four, they are measured by independent means, and they are never OR-ed
together:

  `mal_tcr`  ALICE TCR clonality. A sequence fact -- a CDR3 that recurs in a founder family.
             Defined only for cells with a recovered CDR3; a cell with no TCR is NOT benign,
             it is untested, which is why every call ships with a `_defined` mask.
  `mal_cnv`  inferCNV arm consensus (nb23 Step 6). Defined only for donors where the arm pass
             found a signature (`cnv_callable`).
  `mal_knn`  Label propagation over the MrVI u latent (retired nb26, now
             `notebooks/_archive/old/26_transcriptome_malignancy_holdout.ipynb`): LabelSpreading(
             kernel="knn", k=20, alpha=0.2), anchored on ALICE, thresholded at a fixed 0.5.
             Defined for every CD4 cell that has expression. It was the headline until
             2026-10-22; it is now a sensitivity arm only.
  `mal_cluster` Per-sample Leiden (res 1.0) over all T cells, one call per cluster (nb26b):
             ALICE family share where the sample has TCR, else a gene program learned on the
             ALICE-labelled clusters when its held-out score is clear, else a manual call.
             Defined on every T type; `review_pending` clusters stay undefined.
             **This is the headline.** It is defined on every T type rather than CD4 only, it
             is anchored on a sequence fact per sample rather than on a latent geometry, and
             its HC samples are called benign by construction (see below).

Why not a combined flag. The retired pipeline carried `combined_malignant = tcr | (cnv & ok)`
and a nine-level `malignant_evidence` string. Both hide which measurement fired, and the CNV
call derives from smoothed regional expression -- partly the same signal a downstream
expression analysis then attributes. Keeping the three apart is what lets a result be stated as
"holds on the TCR call as well", which is a claim about replication rather than about a
disjunction.

The HC false-positive floor of `mal_knn` is ~23% and is not a bug: the classifier was trained
on CTCL anchors and has never seen a healthy repertoire. Report it next to any figure that
splits on `mal_knn`.

`mal_cluster` has no such floor, but for the opposite reason and it is not a measurement
either: nb26b assigns HC samples benign as a rule, so a 0% HC rate is an input, not a result.
Never quote it as evidence that the call is specific.
"""

from __future__ import annotations

import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

NB_DIR = Path(__file__).resolve().parent.parent
ATLAS_JOINT = NB_DIR / "data" / "atlas_joint"

# ---------------------------------------------------------------- sources
# Skin and blood are different cohorts with different parquets and the same column names. Before
# this was tissue-keyed, a blood analysis that imported this module silently assembled SKIN calls
# and reindexed them onto blood cell_ids -- every call came back undefined, or worse, matched on
# the handful of ids that collide. Call `use_tissue("blood")` before `load_malignancy`.
CONFIG = {
    "skin": dict(
        ALICE_PQ=ATLAS_JOINT / "alice_malignancy_v4.parquet",            # cell_id, tcr_clonal
        ALICE_COL="tcr_clonal",
        CNV_PQ=ATLAS_JOINT / "skin_T_malignancy_v5.parquet",             # nb23 Step 8
        KNN_PQ=ATLAS_JOINT / "cd4_transcriptome_malignancy_v1.parquet",  # retired nb26
        CLUSTER_PQ=ATLAS_JOINT / "skin_T_malignancy_cluster_v1.parquet",  # nb26b
    ),
    "blood": dict(
        ALICE_PQ=ATLAS_JOINT / "alice_malignancy_blood_v2.parquet",      # nb32, on the v2 annot
        ALICE_COL="tcr_malignant_alice",
        CNV_PQ=ATLAS_JOINT / "blood_T_malignancy_v4.parquet",            # nb32
        KNN_PQ=None,                                                     # no mal_knn in blood
        CLUSTER_PQ=ATLAS_JOINT / "blood_T_malignancy_cluster_v1.parquet",  # nb32b
    ),
}
TISSUE = None


def use_tissue(tissue="skin"):
    """Point this module at one compartment's parquets. Skin is applied at import."""
    global TISSUE
    if tissue not in CONFIG:
        raise ValueError(f"unknown tissue {tissue!r}; have {sorted(CONFIG)}")
    globals().update(CONFIG[tissue])
    TISSUE = tissue
    return tissue


use_tissue("skin")          # default; call use_tissue("blood") before load_malignancy for blood

CALLS = ["mal_tcr", "mal_cnv", "mal_knn", "mal_cluster"]
HEADLINE = "mal_cluster"            # flipped from mal_knn 2026-10-22; see module docstring

# nb26 benchmarks three propagation arms. `mrvi10k` is carried; `mrvi1k` is the 1,000-gene
# panel sensitivity arm; `leiden_free` is label-free and actually beats mrvi10k on 3 of 4
# leave-one-study-out folds -- swap KNN_CALL if you want the strongest transcriptome arm
# rather than the MrVI one.
KNN_CALL, KNN_PROB = "call_mrvi10k", "prob_mrvi10k"


def _read(path, columns=None):
    """Read a parquet to pandas, with any named index materialised as a column.

    pandas, not polars: this module is imported by degradome_helpers, which runs inside the
    build job's mrvi_env (it needs liana via ccc_helpers) and that env has no polars. pyarrow
    and pandas are in every env here.

    The index handling is not cosmetic -- these three parquets are each written differently,
    and polars hid the difference by ignoring pandas index metadata entirely:

      alice_malignancy_v4      cell_id is a real column, index unnamed
      cd4_transcriptome_*_v1   cell_id is the INDEX, and is not a column at all
      atlas_obs_full_v2        cell_id is the index AND a duplicate column

    Callers here all do `df["cell_id"]`, so the first case must keep its column, the second
    must get one, and the third must not end up with `cell_id` as both an index level and a
    column (which makes any later .merge(on="cell_id") raise "which is ambiguous").
    """
    if path is None:          # a source this compartment does not have (blood has no mal_knn)
        return None
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=columns)
    name = df.index.name
    if name is not None and name not in df.columns:
        return df.reset_index()                 # named index -> column, as polars exposed it
    return df.reset_index(drop=True)             # unnamed, or already duplicated as a column


def _columns(path):
    """Column names without reading a row -- pyarrow reads only the footer."""
    return set(pq.ParquetFile(path).schema_arrow.names)


def load_malignancy(cell_ids, has_tcr=None, strict=False, verbose=True):
    """Assemble the four calls for `cell_ids`, each with its definedness mask.

    Returns a frame indexed by `cell_id` with columns
    ``mal_tcr, mal_cnv, mal_knn, mal_cluster`` (bool, False where undefined),
    ``mal_*_defined`` (bool), and ``mal_knn_prob`` / ``mal_cluster_prob``.

    TCR definedness. `tcr_clonal=False` means "no founder family"; for a cell with no recovered
    CDR3 that is untested, not benign, so the call needs a `has_tcr` mask. DO NOT take that
    mask from `atlas_obs_full_v2.parquet` -- its `has_tcr` is stale for li2024 (0.072 there
    against 0.476 in the V(D)J fold-in cache `skin_T_tcr_annotated_v4.h5ad`, i.e. 30k li2024
    cells that DO have a TCR look untested). This function therefore prefers the `has_tcr`
    carried by the propagation parquet, which was written from that cache, and falls back to
    the `has_tcr` argument only for cells the parquet does not cover.

    `strict=True` raises when a source is missing instead of returning it all-undefined. Use it
    in the build; leave it False while upstream notebooks are still being run.
    """
    idx = pd.Index(pd.Series(cell_ids).astype(str), name="cell_id")
    out = pd.DataFrame(index=idx)
    for c in CALLS:
        out[c] = False
        out[f"{c}_defined"] = False

    # ---- has_tcr, preferring the V(D)J fold-in cache over stale obs ----
    tcr_mask = (pd.Series(np.asarray(pd.Series(has_tcr).fillna(False).astype(bool)), index=idx)
                if has_tcr is not None else pd.Series(True, index=idx))
    _ht = _read(KNN_PQ, ["cell_id", "has_tcr"])
    if _ht is not None:
        good = _ht.set_index(_ht["cell_id"].astype(str))["has_tcr"].reindex(idx)
        n_rescued = int((good.fillna(False).astype(bool) & ~tcr_mask).sum())
        tcr_mask = good.where(good.notna(), tcr_mask).fillna(False).astype(bool)
        if verbose and n_rescued:
            print(f"has_tcr: {n_rescued:,} cells marked TCR-bearing by the V(D)J fold-in cache "
                  f"that atlas obs calls untested (li2024)")

    # ---- mal_tcr -------------------------------------------------------
    alice = _read(ALICE_PQ, ["cell_id", ALICE_COL])
    if alice is None:
        msg = f"missing {ALICE_PQ}"
        if strict:
            raise FileNotFoundError(msg)
        warnings.warn(msg + " — mal_tcr left undefined")
    else:
        s = alice.set_index(alice["cell_id"].astype(str))[ALICE_COL]
        r = s.reindex(idx)
        defined = r.notna().to_numpy() & tcr_mask.to_numpy()
        out["mal_tcr"] = (r.fillna(False).to_numpy().astype(bool) & defined)
        out["mal_tcr_defined"] = defined

    # ---- mal_cnv -------------------------------------------------------
    cnv = None
    if CNV_PQ is not None and Path(CNV_PQ).exists():
        have = _columns(CNV_PQ)
        key = "obs_name" if "obs_name" in have else "cell_id"
        want = [key] + [c for c in ("cnv_arm_malignant", "cnv_callable") if c in have]
        cnv = _read(CNV_PQ, want)
        if cnv is not None:
            cnv = cnv.rename(columns={key: "cell_id"})
    if cnv is None or "cnv_arm_malignant" not in cnv:
        msg = (f"missing {CNV_PQ} (or its cnv_arm_malignant column) — run "
               f"23_malignancy_tcr_cnv Steps 6+8")
        if strict:
            raise FileNotFoundError(msg)
        warnings.warn(msg + " — mal_cnv left undefined")
    else:
        cnv = cnv.set_index(cnv["cell_id"].astype(str))
        call = cnv["cnv_arm_malignant"].reindex(idx)
        ok = (cnv["cnv_callable"].reindex(idx) if "cnv_callable" in cnv
              else call.notna())
        defined = (call.notna() & ok.fillna(False).astype(bool)).to_numpy()
        out["mal_cnv"] = call.fillna(False).to_numpy().astype(bool) & defined
        out["mal_cnv_defined"] = defined

    # ---- mal_knn -------------------------------------------------------
    knn = _read(KNN_PQ, ["cell_id", KNN_CALL, KNN_PROB])
    if knn is None:
        # Blood has no label-propagation arm at all, by design -- that is not a missing file.
        if KNN_PQ is not None:
            msg = f"missing {KNN_PQ}"
            if strict:
                raise FileNotFoundError(msg)
            warnings.warn(msg + " — mal_knn left undefined")
    else:
        knn = knn.set_index(knn["cell_id"].astype(str))
        prob = knn[KNN_PROB].reindex(idx)
        call = knn[KNN_CALL].reindex(idx)
        defined = (prob.notna() & call.notna()).to_numpy()
        out["mal_knn"] = call.fillna(False).to_numpy().astype(bool) & defined
        out["mal_knn_defined"] = defined
        out["mal_knn_prob"] = prob.to_numpy()

    # ---- mal_cluster ---------------------------------------------------
    clu = _read(CLUSTER_PQ, ["cell_id", "mal_cluster", "mal_cluster_prob"])
    if clu is None:
        msg = f"missing {CLUSTER_PQ} — run 26b_cluster_malignancy"
        if strict:
            raise FileNotFoundError(msg)
        warnings.warn(msg + " — mal_cluster left undefined")
    else:
        clu = clu.set_index(clu["cell_id"].astype(str))
        call = clu["mal_cluster"].reindex(idx)
        defined = call.notna().to_numpy()
        out["mal_cluster"] = call.fillna(False).to_numpy().astype(bool) & defined
        out["mal_cluster_defined"] = defined
        out["mal_cluster_prob"] = clu["mal_cluster_prob"].reindex(idx).to_numpy()

    if verbose:
        print(coverage_report(out).to_string())
    return out


def coverage_report(mal):
    """Per call: how many cells it is defined on, and its positive rate where defined."""
    rows = {}
    for c in CALLS:
        d = mal[f"{c}_defined"].to_numpy()
        rows[c] = {
            "n_defined": int(d.sum()),
            "frac_defined": round(float(d.mean()), 4),
            "n_malignant": int(mal[c].to_numpy().sum()),
            "frac_malignant_where_defined": (round(float(mal[c].to_numpy()[d].mean()), 4)
                                             if d.any() else np.nan),
        }
    return pd.DataFrame(rows).T


def concordance(mal, by=None):
    """Pairwise agreement between the calls, on cells where both are defined.

    `by` is an optional Series aligned to `mal` (stage arm, study, ...) -- the concordance is
    reported within each of its levels, because a number pooled over cohorts that differ in TCR
    recovery is not a property of the methods.
    """
    pairs = list(combinations(CALLS, 2))
    groups = ([("all", np.ones(len(mal), bool))] if by is None
              else [(g, (np.asarray(by) == g)) for g in pd.unique(pd.Series(by).dropna())])
    rows = []
    for gname, gmask in groups:
        for a, b in pairs:
            m = gmask & mal[f"{a}_defined"].to_numpy() & mal[f"{b}_defined"].to_numpy()
            if not m.any():
                rows.append({"group": gname, "a": a, "b": b, "n_both": 0,
                             "agree": np.nan, "a_pos": np.nan, "b_pos": np.nan})
                continue
            av, bv = mal[a].to_numpy()[m], mal[b].to_numpy()[m]
            inter = int((av & bv).sum())
            union = int((av | bv).sum())
            rows.append({
                "group": gname, "a": a, "b": b, "n_both": int(m.sum()),
                "agree": round(float((av == bv).mean()), 4),
                "jaccard": round(inter / union, 4) if union else np.nan,
                "a_pos": round(float(av.mean()), 4),
                "b_pos": round(float(bv.mean()), 4),
            })
    return pd.DataFrame(rows)


def cd4_level(cell_type_final, mal, call=HEADLINE, cd4_set=None):
    """CD4 -> {CD4_malignant, CD4_reactive, CD4_unassessed}; other types pass through.

    `CD4_unassessed` is a real level, not a dustbin: it is the cells `call` could not be
    evaluated on. Folding them into reactive would make "non-malignant CD4" mean "malignant
    was not detected", which is a different statement.

    What counts as CD4 comes from `helpers/lineage.py`, not from a literal. The re-annotation
    (`nonmal_T_labels_v1`) renamed the benign CD4 cells to `CD4_conv`/`CD4_naive` and moved the
    CD8, NK and Treg cells that the old per-cluster `CD4` label had swept in; the malignant
    cells are absent from that table and keep the old `CD4`, so the set holds both. A literal
    `== "CD4"` here would silently return an empty CD4 arm.
    """
    ct = pd.Series(cell_type_final).astype(str).to_numpy().copy()
    if cd4_set is None:
        import lineage as LIN
        cd4_set = LIN.cd4_set(TISSUE or "skin")
    is_cd4 = np.isin(ct, list(cd4_set))
    if not is_cd4.any():
        raise AssertionError(
            f"cd4_level: no cell matched the CD4 set {sorted(cd4_set)} -- the frame's lineage "
            f"column is probably pre-overlay or from the wrong tissue (TISSUE={TISSUE!r})")
    d = mal[f"{call}_defined"].to_numpy()
    v = mal[call].to_numpy()
    ct[is_cd4 & ~d] = "CD4_unassessed"
    ct[is_cd4 & d & v] = "CD4_malignant"
    ct[is_cd4 & d & ~v] = "CD4_reactive"
    return pd.Categorical(ct)
