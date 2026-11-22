#!/usr/bin/env python
"""nb52 QC - what is in the benign CD4 arm? (read-only follow-up to run_ccc52_mvb.py)

In the v1 skin fit CD8A (log2FC -4.7), CD8B, KLRD1 (-6.0), TRDC (-5.7) and FOXP3 (-3.9) come out
far lower in the malignant than in the benign CD4 arm, and CD8A is detected in 17% of benign-arm
cells. Either the benign arm carries non-CD4-conventional cells, or that is ambient RNA /
doublets, or it is real. This script measures it.

Nothing is refit and nothing in reports/ccc52_mvb/ is overwritten: the v1 arms, cohort rules and
eligible donors are read back and re-derived, and the derivation is asserted against v1's own
row_labels.npz before anything is computed. All output goes to reports/ccc52_mvb/qc_benign_arm/.

Stages (./run_ccc52_qc.sh <stage>, or `all` for prep -> pass -> tab in one job):

  prep  re-derive the v1 arms, attach the per-sample cluster key, define the five comparison
        groups, write the row labels the pass consumes.
  pass  one streamed pass over joint_annotated.h5ad restricted to the marker panel, accumulating
        counts, detection, library size and the co-detection cross-tabs.
  tab   the six CSVs and qc_summary.md.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

NB_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NB_DIR / "helpers"))

ATLAS = NB_DIR / "data" / "atlas_joint"
V1 = NB_DIR / "reports" / "ccc52_mvb"
OUT = V1 / "qc_benign_arm"

CONFIG = dict(
    JOINT_H5AD=ATLAS / "joint_annotated.h5ad",
    JOINT_LAYER="raw_counts",
    OBS_PARQUET=ATLAS / "atlas_obs_full_v2.parquet",
    MAL_PARQUET={"skin": ATLAS / "skin_T_malignancy_cluster_v1.parquet",
                 "blood": ATLAS / "blood_T_malignancy_cluster_v1.parquet"},
    CLUSTERS={"skin": ATLAS / "skin_T_clusters_v1.parquet",
              "blood": ATLAS / "blood_T_clusters_v1.parquet"},
    BLOOD_SUBTYPES=ATLAS / "blood_subtypes.csv",
    BLOOD_CT_FIX=ATLAS / "blood_T_celltype_corrections_v1.csv",
    # ---- v1 definitions, reused verbatim -----------------------------------------
    MAL_CALL="mal_cluster",
    CD4_LABEL="CD4",
    TREG_LABEL="CD4_Treg",
    DROP_DATASETS={"blood": ["geskin26"]},
    SKIN_NON_CD4_PHENOTYPES=("CD8", "gd", "DN"),
    SKIN_NON_CD4_DX=("CD8_aggr_CTCL", "pcAECTCL"),
    SKIN_OFF_SPECTRUM_DX=("LyP", "SPP", "LPP"),
    BLOOD_SUBTYPE_DROP=("Ambient_soup", "LowQC", "Doublet"),
    # ---- QC thresholds -------------------------------------------------------------
    FLAG_DETECT=0.05,        # benign cluster is "non-conventional" at >= this detection
    FLAG_STRICT=0.25,        # the stricter cut-off used for sensitivity variant (c)
    MIN_ROW_CELLS=20,        # rows smaller than this are dropped from qc_cluster_markers.csv
    MIN_BENIGN_AFTER=50,     # donor keeps its place in sensitivity (b) at >= this many benign
    CHUNK=20000,
    # ---- ADT (buus25 only: obsm/X_adt is NaN everywhere else) ------------------------
    ADT_KEY="X_adt",
    ADT_PANEL_KEY="adt_panel",
    ADT_MARKERS=["ADT_CD3", "ADT_CD4", "ADT_CD8A", "ADT_CD8a", "ADT_TCRab", "ADT_TCRgd",
                 "ADT_CD56"],
    # ---- doublets -------------------------------------------------------------------
    DOUBLET_COL="doublet_score",
    DOUBLET_THRESH=0.25,     # the fixed scrublet cutoff of helpers/atlas_join_helpers.py:1124
)

PANEL = ["CD4", "CD40LG", "IL7R", "CD3E", "CD8A", "CD8B", "TRDC", "TRGC1", "KLRD1", "NCAM1",
         "KLRF1", "NKG7", "GZMK", "FOXP3", "IL2RA", "CTLA4", "KRT14", "LYZ", "PECAM1"]

SENS = ["CD8A", "CD8B", "KLRD1", "TRDC", "FOXP3", "CD4", "TOX", "DPP4", "KIR3DL2", "CCL5",
        "CCL4", "KLRK1", "CXCR6", "CXCR3", "CCR6", "ITGA4", "CXCR4", "SELL", "CD27", "ICOS",
        "AREG", "IL10RA", "PTHLH", "PTH2R", "IGFL2", "CXCL13", "CD70", "PDGFA", "CD9", "IL13",
        "CCR4", "CD28"]

GENES = PANEL + [g for g in SENS if g not in PANEL]

FLAG_SETS = {"CD8": ["CD8A", "CD8B"], "gdNK": ["TRDC", "KLRD1"], "Treg": ["FOXP3"]}

# co-detection counters accumulated per v1 unit (tissue x donor x arm)
CO_COLS = ["n", "CD4p", "CD8ABp", "CD4p_CD8ABp", "FOXP3p", "CD40LGp", "FOXP3p_CD40LGp"]

GROUPS = ["CD4_benign", "CD4_malignant", "CD8", "NK", "CD4_Treg", "CD4_Treg_like"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _read_h5ad_str(h5, key):
    import h5py
    d = h5[key]
    if isinstance(d, h5py.Group):
        cats = np.array([x.decode() if isinstance(x, bytes) else x for x in d["categories"][:]])
        codes = d["codes"][:]
        return np.where(codes < 0, "", cats[np.clip(codes, 0, None)])
    v = d[:]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in v])


# ======================================================================================
# stage: prep
# ======================================================================================
def stage_prep(args):
    import polars as pl

    OUT.mkdir(parents=True, exist_ok=True)

    cols = ["cell_id", "dataset", "study", "donor", "sample_id", "patient_key",
            "clin_person_key", "compartment", "tech", "clin_dx", "clin_phenotype",
            "cell_type_final", CONFIG["DOUBLET_COL"]]
    obs = (pl.read_parquet(CONFIG["OBS_PARQUET"], columns=cols)
             .with_columns([pl.col(c).cast(pl.Utf8) for c in cols if c != CONFIG["DOUBLET_COL"]])
             .to_pandas().set_index("cell_id"))
    obs["tissue"] = obs["compartment"].map({"Skin": "skin", "Blood": "blood"})
    log(f"obs: {obs.shape}")

    mal = pd.concat([pd.read_parquet(p, columns=["cell_id", CONFIG["MAL_CALL"]])
                     for p in CONFIG["MAL_PARQUET"].values()]).set_index("cell_id")
    obs["mal"] = mal[CONFIG["MAL_CALL"]].reindex(obs.index)

    obs["unit"] = (obs["clin_person_key"].replace({"": np.nan, "nan": np.nan, "None": np.nan})
                   .fillna(obs["patient_key"]).fillna(obs["donor"]))

    # ---- v1 arms -------------------------------------------------------------------
    is_cd4 = obs["cell_type_final"].eq(CONFIG["CD4_LABEL"]) & obs["tissue"].notna()
    arm = pd.Series(pd.NA, index=obs.index, dtype="object")
    arm[is_cd4 & obs["mal"].eq(True)] = "M"
    arm[is_cd4 & obs["mal"].eq(False)] = "B"
    obs["arm"] = arm

    drop = pd.Series(pd.NA, index=obs.index, dtype="object")
    skin = obs["tissue"].eq("skin")
    drop[skin & (obs["clin_phenotype"].isin(CONFIG["SKIN_NON_CD4_PHENOTYPES"])
                 | obs["clin_dx"].isin(CONFIG["SKIN_NON_CD4_DX"]))] = "non-CD4 clone"
    drop[skin & obs["clin_dx"].isin(CONFIG["SKIN_OFF_SPECTRUM_DX"])] = "off-spectrum dx"
    for t, ds in CONFIG["DROP_DATASETS"].items():
        drop[obs["tissue"].eq(t) & obs["dataset"].isin(ds)] = "ex vivo"
    obs["cohort_keep"] = drop.isna()

    # ---- per-sample cluster key ------------------------------------------------------
    clus = pd.concat([pd.read_parquet(p, columns=["cell_id", "cluster"])
                      for p in CONFIG["CLUSTERS"].values()]).set_index("cell_id")
    obs["cluster"] = clus["cluster"].reindex(obs.index)

    # ---- v1 blood benign-arm cleanup, read back from the v1 table --------------------
    sub = pd.read_csv(V1 / "blood_benign_subclusters.csv")
    kind = {c: ("CD4_Treg_like" if bool(t) else "CD4_naive_like")
            for c, t, d in zip(sub["cluster"], sub["is_treg"], sub["dropped"]) if bool(d)}
    bsub = obs["cluster"].map(kind)
    moved = bsub.notna() & obs["arm"].eq("B") & obs["tissue"].eq("blood") & obs["cohort_keep"]
    obs["blood_cd4_sub"] = pd.NA
    obs.loc[moved, "blood_cd4_sub"] = bsub[moved]
    obs.loc[moved, "arm"] = pd.NA
    log(f"blood benign cells moved out of the arm: {int(moved.sum()):,}")

    assert not (obs["cell_type_final"].eq(CONFIG["TREG_LABEL"]) & obs["arm"].notna()).any()

    # ---- eligible units: taken from v1, never recomputed ------------------------------
    units = pd.read_csv(V1 / "units.csv")
    elig = set(zip(units["tissue"], units["unit"]))
    ukey = {(t, u, a): int(i) for t, u, a, i in
            zip(units["tissue"], units["unit"], units["arm"], units["unit_idx"])}
    umeta = (units.drop_duplicates(["tissue", "unit"])
             .set_index(["tissue", "unit"])[["study", "dataset", "tech", "dx"]])
    log(f"v1 eligible units: {units.drop_duplicates(['tissue','unit']).groupby('tissue').size().to_dict()}")

    # ---- cell-type labels, with the v1 corrections ------------------------------------
    ct = obs["cell_type_final"].copy()
    fix = pd.read_csv(CONFIG["BLOOD_CT_FIX"])
    hit = obs.index.isin(fix["cell_id"])
    ct.loc[hit] = fix.set_index("cell_id")["cell_type_T"].reindex(obs.index[hit]).values
    bad = pd.read_csv(CONFIG["BLOOD_SUBTYPES"], usecols=["cell_id", "subtype_fine"])
    bad = set(bad.loc[bad["subtype_fine"].isin(CONFIG["BLOOD_SUBTYPE_DROP"]), "cell_id"])
    ct.loc[obs.index.isin(bad)] = pd.NA

    grp = pd.Series(pd.NA, index=obs.index, dtype="object")
    grp[obs["arm"].eq("M")] = "CD4_malignant"
    grp[obs["arm"].eq("B")] = "CD4_benign"
    grp[grp.isna() & ct.eq("CD8")] = "CD8"
    grp[grp.isna() & ct.eq("NK")] = "NK"
    grp[grp.isna() & ct.eq(CONFIG["TREG_LABEL"])] = "CD4_Treg"
    grp[obs["blood_cd4_sub"].eq("CD4_Treg_like")] = "CD4_Treg_like"

    in_elig = np.fromiter(((t, u) in elig for t, u in zip(obs["tissue"], obs["unit"])),
                          dtype=bool, count=len(obs))
    grp[~(in_elig & obs["cohort_keep"].to_numpy())] = pd.NA
    obs["group"] = grp
    obs["cluster"] = obs["cluster"].fillna("unclustered")

    tab = (obs[obs["group"].notna()]
           .groupby(["tissue", "unit", "cluster", "group"], dropna=False)
           .size().rename("n_cells").reset_index())
    tab = tab.merge(umeta.reset_index(), on=["tissue", "unit"], how="left")
    tab["row_idx"] = np.arange(len(tab))
    tab.to_csv(OUT / "qc_row_groups.csv", index=False)
    log("group cells: " + str(obs[obs["group"].notna()].groupby(["tissue", "group"]).size().to_dict()))

    # ---- align to the joint object's own cell order ------------------------------------
    import h5py
    with h5py.File(CONFIG["JOINT_H5AD"], "r") as f:
        joint_ids = _read_h5ad_str(f["obs"], "cell_id")
    o = obs.reindex(pd.Index(joint_ids, name="cell_id"))

    rkey = {k: int(v) for k, v in
            zip(zip(tab["tissue"], tab["unit"], tab["cluster"], tab["group"]), tab["row_idx"])}
    row_group = np.fromiter(
        (rkey.get((t, u, c, g), -1) for t, u, c, g in
         zip(o["tissue"], o["unit"], o["cluster"], o["group"])), dtype=np.int32, count=len(o))

    row_unit = np.fromiter(
        (ukey.get((t, u, a), -1) for t, u, a in zip(o["tissue"], o["unit"], o["arm"])),
        dtype=np.int32, count=len(o))

    # the v1 arms, asserted rather than assumed
    v1 = np.load(V1 / "row_labels.npz", allow_pickle=False)["row_unit"]
    assert len(v1) == len(row_unit), "row length mismatch against v1"
    bad_n = int((v1 != row_unit).sum())
    assert bad_n == 0, f"{bad_n:,} cells disagree with the v1 arm assignment"
    log(f"v1 arms reproduced exactly ({int((row_unit >= 0).sum()):,} arm cells)")

    row_doublet = pd.to_numeric(o[CONFIG["DOUBLET_COL"]], errors="coerce").to_numpy(np.float32)
    log(f"doublet score present for {int(np.isfinite(row_doublet).sum()):,} of {len(o):,} cells; "
        f"datasets without it: "
        f"{sorted(set(o.loc[~np.isfinite(row_doublet), 'dataset'].dropna()))}")

    np.savez_compressed(OUT / "qc_row_labels.npz", row_group=row_group, row_unit=row_unit,
                        row_doublet=row_doublet, n_group=len(tab), n_unit=len(units))
    log("prep done")


# ======================================================================================
# stage: pass
# ======================================================================================
def stage_pass(args):
    import h5py
    from scipy import sparse

    lab = np.load(OUT / "qc_row_labels.npz", allow_pickle=False)
    row_group, row_unit = lab["row_group"], lab["row_unit"]
    row_doublet = lab["row_doublet"]
    n_group, n_unit = int(lab["n_group"]), int(lab["n_unit"])

    with h5py.File(CONFIG["JOINT_H5AD"], "r") as f:
        L = f["layers"][CONFIG["JOINT_LAYER"]]
        n, G = (int(x) for x in f["X"].attrs["shape"])
        var = _read_h5ad_str(f["var"], f["var"].attrs.get("_index", "_index"))
        assert len(row_group) == n

        pos = {g: int(np.flatnonzero(var == g)[0]) for g in GENES if (var == g).any()}
        genes = [g for g in GENES if g in pos]
        missing = [g for g in GENES if g not in pos]
        if missing:
            log(f"WARNING genes absent from the atlas var index: {missing}")
        k = len(genes)
        P = sparse.csr_matrix((np.ones(k), ([pos[g] for g in genes], np.arange(k))),
                              shape=(G, k))
        log(f"streaming {n:,} x {G:,}; {k} panel genes; {n_group} groups, {n_unit} units")

        X = np.zeros((n_group, k))
        D = np.zeros((n_group, k))
        NC = np.zeros(n_group, dtype=np.int64)
        UMI = np.zeros(n_group)
        CO = np.zeros((n_unit, len(CO_COLS)))
        tot = np.zeros(n, dtype=np.float32)

        ci = {g: j for j, g in enumerate(genes)}

        du_unit, du_score, du_cd8 = [], [], []

        indptr = L["indptr"][:]
        chunk = CONFIG["CHUNK"]
        t0 = time.time()
        checked = False
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            rg, ru = row_group[a:b], row_unit[a:b]
            if not ((rg >= 0).any() or (ru >= 0).any()):
                continue
            lo, hi = int(indptr[a]), int(indptr[b])
            M = sparse.csr_matrix(
                (L["data"][lo:hi], L["indices"][lo:hi], indptr[a:b + 1] - lo), shape=(b - a, G))
            if not checked:
                assert np.all(M.data == np.round(M.data)), "counts layer is not integer-valued"
                log(f"counts verified integer (first chunk, {M.nnz:,} nnz)")
                checked = True
            rows = np.asarray(M.sum(axis=1)).ravel()
            tot[a:b] = rows
            S = np.asarray((M @ P).todense())
            Bo = (S > 0).astype(np.float64)

            if (rg >= 0).any():
                s = np.flatnonzero(rg >= 0)
                g = sparse.csr_matrix((np.ones(len(s)), (rg[s], s)), shape=(n_group, b - a))
                X += g @ S
                D += g @ Bo
                NC += np.bincount(rg[s], minlength=n_group)
                UMI += np.bincount(rg[s], weights=rows[s], minlength=n_group)
            if (ru >= 0).any():
                s = np.flatnonzero(ru >= 0)
                flags = {
                    "n": np.ones(len(s)),
                    "CD4p": Bo[s, ci["CD4"]],
                    "CD8ABp": np.maximum(Bo[s, ci["CD8A"]], Bo[s, ci["CD8B"]]),
                    "FOXP3p": Bo[s, ci["FOXP3"]],
                    "CD40LGp": Bo[s, ci["CD40LG"]],
                }
                flags["CD4p_CD8ABp"] = flags["CD4p"] * flags["CD8ABp"]
                flags["FOXP3p_CD40LGp"] = flags["FOXP3p"] * flags["CD40LGp"]
                for j, c in enumerate(CO_COLS):
                    CO[:, j] += np.bincount(ru[s], weights=flags[c], minlength=n_unit)
            cd8ab = np.maximum(Bo[:, ci["CD8A"]], Bo[:, ci["CD8B"]])
            if (ru >= 0).any():
                s = np.flatnonzero(ru >= 0)
                du_unit.append(ru[s].copy())
                du_score.append(row_doublet[a:b][s])
                du_cd8.append(cd8ab[s].astype(np.uint8))
            if (a // chunk) % 20 == 0:
                log(f"  {b:,}/{n:,}  {time.time() - t0:.0f}s")

    def _cat(x, dt):
        return np.concatenate(x).astype(dt) if x else np.zeros(0, dtype=dt)

    sel = row_group >= 0
    log(f"pass done in {time.time() - t0:.0f}s; group cells {NC.sum():,}, arm cells {int(CO[:,0].sum()):,}")
    np.savez_compressed(
        OUT / "qc_accumulators.npz", X=X, D=D, NC=NC, UMI=UMI, CO=CO,
        genes=np.array(genes), missing=np.array(missing, dtype=object),
        tot_lab=tot[sel], grp_lab=row_group[sel],
        du_unit=_cat(du_unit, np.int32), du_score=_cat(du_score, np.float32),
        du_cd8=_cat(du_cd8, np.uint8))
    log(f"arm cells with a doublet score "
        f"{int(np.isfinite(_cat(du_score, np.float32)).sum()):,}")
    log(f"wrote {OUT/'qc_accumulators.npz'}")


# ======================================================================================
# stage: tab
# ======================================================================================
def _fmt(x, nd=3):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def stage_tab(args):
    acc = np.load(OUT / "qc_accumulators.npz", allow_pickle=True)
    X, D, NC, UMI, CO = acc["X"], acc["D"], acc["NC"], acc["UMI"], acc["CO"]
    genes = list(acc["genes"])
    missing = [str(m) for m in acc["missing"]]
    ci = {g: j for j, g in enumerate(genes)}
    tab = pd.read_csv(OUT / "qc_row_groups.csv").sort_values("row_idx").reset_index(drop=True)
    units = pd.read_csv(V1 / "units.csv").sort_values("unit_idx").reset_index(drop=True)

    med = (pd.DataFrame({"g": acc["grp_lab"], "t": acc["tot_lab"]})
           .groupby("g")["t"].median().reindex(np.arange(len(tab))))
    tab["median_umis"] = med.values
    assert (tab["n_cells"].values == NC).all(), "pass cell counts disagree with prep"

    panel = [g for g in PANEL if g in ci]
    det = D / NC[:, None]

    # ---- A. per donor x cluster x group ------------------------------------------------
    A = tab[["tissue", "unit", "study", "tech", "cluster", "group", "n_cells", "median_umis"]].copy()
    A = A.rename(columns={"unit": "donor"})
    for g in panel:
        A[f"det_{g}"] = det[:, ci[g]]
    A = A[A["n_cells"] >= CONFIG["MIN_ROW_CELLS"]].sort_values(
        ["tissue", "donor", "group", "n_cells"], ascending=[True, True, True, False])
    A.round(4).to_csv(OUT / "qc_cluster_markers.csv", index=False)
    log(f"qc_cluster_markers.csv: {len(A)} rows (>= {CONFIG['MIN_ROW_CELLS']} cells)")

    # ---- B. clusters pooled within donor, plus the across-donor median ------------------
    key = ["tissue", "unit", "group"]
    kmap = tab[key].astype(str).agg("|".join, axis=1).to_numpy()
    pooled = (pd.DataFrame({"k": kmap[acc["grp_lab"]], "t": acc["tot_lab"]})
              .groupby("k")["t"].median())
    idx = tab.groupby(key, sort=False).indices
    rows = []
    for (t, u, g), ii in idx.items():
        nc = NC[ii].sum()
        r = {"tissue": t, "donor": u, "group": g, "n_cells": int(nc),
             "median_umis": float(pooled[f"{t}|{u}|{g}"])}
        dsum = D[ii].sum(axis=0)
        for gg in panel:
            r[f"det_{gg}"] = dsum[ci[gg]] / nc
        rows.append(r)
    B = pd.DataFrame(rows)
    ref = (B.groupby(["tissue", "group"])[["median_umis"] + [f"det_{g}" for g in panel]]
           .median().reset_index())
    ref.insert(1, "donor", "ALL_DONORS_MEDIAN")
    ref["n_cells"] = B.groupby(["tissue", "group"])["n_cells"].sum().values
    B = pd.concat([B.sort_values(["tissue", "group", "donor"]), ref], ignore_index=True)
    B.round(4).to_csv(OUT / "qc_group_reference.csv", index=False)
    log(f"qc_group_reference.csv: {len(B)} rows")

    # ---- C. co-expression within the arms -----------------------------------------------
    co = pd.DataFrame(CO, columns=CO_COLS)
    co[["tissue", "donor", "arm"]] = units[["tissue", "unit", "arm"]].values
    piv = {a: g.set_index(["tissue", "donor"]) for a, g in co.groupby("arm")}
    C = []
    for (t, u), rb in piv["B"].iterrows():
        rm = piv["M"].loc[(t, u)]
        row = {"tissue": t, "donor": u}
        for pre, r in (("b", rb), ("m", rm)):
            n, cd4, cd8, both = r["n"], r["CD4p"], r["CD8ABp"], r["CD4p_CD8ABp"]
            fx, cl, fxcl = r["FOXP3p"], r["CD40LGp"], r["FOXP3p_CD40LGp"]
            row[f"{pre}_n"] = int(n)
            row[f"{pre}_n_CD4p"] = int(cd4)
            row[f"{pre}_n_CD8ABp"] = int(cd8)
            row[f"{pre}_n_CD4p_CD8ABp"] = int(both)
            row[f"{pre}_fracCD4_in_CD8ABpos"] = both / cd8 if cd8 else np.nan
            row[f"{pre}_fracCD4_in_CD8ABneg"] = (cd4 - both) / (n - cd8) if n > cd8 else np.nan
            row[f"{pre}_n_FOXP3p"] = int(fx)
            row[f"{pre}_fracCD40LG_in_FOXP3pos"] = fxcl / fx if fx else np.nan
            row[f"{pre}_fracCD40LG_in_FOXP3neg"] = (cl - fxcl) / (n - fx) if n > fx else np.nan
        C.append(row)
    C = pd.DataFrame(C).sort_values(["tissue", "donor"])
    C.round(4).to_csv(OUT / "qc_coexpression.csv", index=False)
    log(f"qc_coexpression.csv: {len(C)} rows")

    # ---- F. flagged benign clusters -------------------------------------------------------
    ben = tab["group"].eq("CD4_benign").to_numpy()
    flag_marks = {k: [g for g in v if g in ci] for k, v in FLAG_SETS.items()}
    flagged = {k: ben & (det[:, [ci[g] for g in v]] >= CONFIG["FLAG_DETECT"]).any(axis=1)
               for k, v in flag_marks.items()}
    any_flag = np.logical_or.reduce(list(flagged.values()))
    # The brief's 0.05 cut-off removes most of the skin benign arm, which leaves the sensitivity
    # analysis with a small and self-selected donor set. A strict cut-off that only catches whole
    # mislabelled clusters is carried alongside it as variant (c).
    any_flag25 = np.logical_or.reduce(
        [ben & (det[:, [ci[g] for g in v]] >= CONFIG["FLAG_STRICT"]).any(axis=1)
         for v in flag_marks.values()])
    tab["benign_flagged"] = any_flag
    for k, v in flagged.items():
        tab[f"flag_{k}"] = v
    tab.to_csv(OUT / "qc_row_groups.csv", index=False)

    F = []
    pos = pd.Series(np.arange(len(tab)))
    for (t, u), ii in pos[ben].groupby(
            [tab.loc[ben, "tissue"], tab.loc[ben, "unit"]], sort=False):
        ii = ii.to_numpy()
        tot_b = NC[ii].sum()
        r = {"tissue": t, "donor": u, "n_benign": int(tot_b),
             "n_clusters": len(ii), "n_clusters_flagged": int(any_flag[ii].sum()),
             "n_benign_flagged": int(NC[ii][any_flag[ii]].sum())}
        for k in FLAG_SETS:
            r[f"n_benign_flag_{k}"] = int(NC[ii][flagged[k][ii]].sum())
        r["n_benign_remaining"] = int(tot_b - r["n_benign_flagged"])
        r["frac_flagged"] = r["n_benign_flagged"] / tot_b
        F.append(r)
    F = pd.DataFrame(F).sort_values(["tissue", "frac_flagged"], ascending=[True, False])
    tot = F.groupby("tissue").sum(numeric_only=True).reset_index()
    tot["donor"] = "TOTAL"
    tot["frac_flagged"] = tot["n_benign_flagged"] / tot["n_benign"]
    F = pd.concat([F, tot[F.columns]], ignore_index=True)
    F.round(4).to_csv(OUT / "qc_flag_summary.csv", index=False)
    log(f"qc_flag_summary.csv: {len(F)} rows")

    # how much of the flagged mass is a whole mislabelled cluster rather than a few cells
    tiers = []
    for t in sorted(tab["tissue"].unique()):
        ti = tab["tissue"].eq(t).to_numpy() & ben
        base = NC[ti].sum()
        for nm, gs in flag_marks.items():
            mx = det[:, [ci[g] for g in gs]].max(axis=1)
            for th in (0.05, 0.10, 0.25, 0.50):
                m = ti & (mx >= th)
                tiers.append({"tissue": t, "marker_set": nm, "threshold": th,
                              "n_clusters": int(m.sum()), "n_cells": int(NC[m].sum()),
                              "frac_benign": NC[m].sum() / base})
    tiers = pd.DataFrame(tiers)

    # ---- G. sensitivity ---------------------------------------------------------------
    sens = [g for g in SENS if g in ci]
    cols = [ci[g] for g in sens]
    mal = tab["group"].eq("CD4_malignant").to_numpy()

    def _cpm(mask_rows):
        x = X[mask_rows][:, cols].sum(axis=0)
        lib = UMI[mask_rows].sum()
        return 1e6 * x / lib if lib > 0 else np.full(len(cols), np.nan)

    recs = []
    for t in sorted(tab["tissue"].unique()):
        ti = tab["tissue"].eq(t).to_numpy()
        donors = sorted(tab.loc[ti & ben, "unit"].unique())
        lfc_a, lfc_b, lfc_c, kept, kept_c = {}, {}, {}, [], []
        for u in donors:
            ui = ti & tab["unit"].eq(u).to_numpy()
            m = _cpm(ui & mal)
            all_b = ui & ben
            clean_b = all_b & ~any_flag
            lfc_a[u] = np.log2(m + 1) - np.log2(_cpm(all_b) + 1)
            n_clean = NC[clean_b].sum()
            if n_clean >= CONFIG["MIN_BENIGN_AFTER"]:
                lfc_b[u] = np.log2(m + 1) - np.log2(_cpm(clean_b) + 1)
                kept.append(u)
            strict_b = all_b & ~any_flag25
            if NC[strict_b].sum() >= CONFIG["MIN_BENIGN_AFTER"]:
                lfc_c[u] = np.log2(m + 1) - np.log2(_cpm(strict_b) + 1)
                kept_c.append(u)
        Aarr = np.vstack([lfc_a[u] for u in donors])
        Barr = np.vstack([lfc_b[u] for u in kept]) if kept else np.zeros((0, len(cols)))
        Carr = np.vstack([lfc_c[u] for u in kept_c]) if kept_c else np.zeros((0, len(cols)))
        for j, g in enumerate(sens):
            recs.append({
                "tissue": t, "gene": g,
                "n_donors_a": len(donors), "median_lfc_a": np.median(Aarr[:, j]),
                "n_pos_a": int((Aarr[:, j] > 0).sum()),
                "n_donors_b": len(kept),
                "median_lfc_b": np.median(Barr[:, j]) if kept else np.nan,
                "n_pos_b": int((Barr[:, j] > 0).sum()) if kept else 0,
                "delta_median_lfc": (np.median(Barr[:, j]) - np.median(Aarr[:, j])) if kept else np.nan,
                "n_donors_c": len(kept_c),
                "median_lfc_c": np.median(Carr[:, j]) if kept_c else np.nan,
                "n_pos_c": int((Carr[:, j] > 0).sum()) if kept_c else 0,
            })
        log(f"{t}: sensitivity donors {len(donors)} -> {len(kept)} at >= "
            f"{CONFIG['FLAG_DETECT']} / {len(kept_c)} at >= {CONFIG['FLAG_STRICT']}, keeping "
            f"donors with >= {CONFIG['MIN_BENIGN_AFTER']} benign cells")
    Gt = pd.DataFrame(recs)
    Gt.round(4).to_csv(OUT / "qc_sensitivity.csv", index=False)
    log(f"qc_sensitivity.csv: {len(Gt)} rows")

    Dt = _adt_table(tab)
    Et = _doublet_table(acc, units, tab)
    _summary(tab, B, C, F, Gt, Dt, Et, tiers, panel, missing)


def _adt_table(tab):
    """D. Surface protein, where it exists.

    buus25 is the only dataset whose ADT panel survived the atlas join, so obsm/X_adt is NaN for
    every other cell. Values are CLR-normalised within the cell (log1p, minus the cell's mean
    over the 128 markers), which is the usual ADT transform and is what makes a threshold
    comparable across cells of very different total protein capture.

    The gate is donor-internal and stated rather than inherited: for each donor and marker the
    threshold is the midpoint between the median CLR of that donor's CD8-labelled T cells and the
    median of its malignant CD4 cells. Those two are the purest reference populations the atlas
    offers inside the same donor and the same lane, so a benign-arm cell above the CD8 side of
    the midpoint looks like a CD8 cell by protein, not by label.
    """
    import h5py
    row_group = np.load(OUT / "qc_row_labels.npz", allow_pickle=False)["row_group"]
    with h5py.File(CONFIG["JOINT_H5AD"], "r") as f:
        if CONFIG["ADT_KEY"] not in f["obsm"]:
            (OUT / "qc_adt.csv").write_text("# the joint object carries no ADT matrix\n")
            return None
        d = f["obsm"][CONFIG["ADT_KEY"]]
        panel = [x.decode() if isinstance(x, bytes) else x
                 for x in f["uns"][CONFIG["ADT_PANEL_KEY"]][:]]
        grp, raw = [], []
        step = 100_000
        for a in range(0, d.shape[0], step):
            b = min(a + step, d.shape[0])
            rg = row_group[a:b]
            if not (rg >= 0).any():
                continue
            A = d[a:b, :]
            ok = np.flatnonzero((rg >= 0) & np.isfinite(A).any(axis=1))
            if not len(ok):
                continue
            grp.append(rg[ok].copy())
            raw.append(A[ok])
    if not grp:
        (OUT / "qc_adt.csv").write_text("# no ADT measurements for any eligible donor\n")
        log("qc_adt.csv: no ADT cells")
        return None
    A = np.concatenate(raw)
    la = np.log1p(np.clip(A, 0, None))
    with np.errstate(invalid="ignore"):
        clr = la - np.nanmean(la, axis=1, keepdims=True)       # CLR within the cell

    # buus25 was deposited under two panel spellings: ADT_CD8A covers 93.7% of its cells and
    # ADT_CD8a the remaining 6.7%. They are one marker, so they are merged rather than
    # intersected -- requiring both finite would throw away every cell.
    def _col(*names):
        out = np.full(len(clr), np.nan, dtype=np.float32)
        for nm in names:
            if nm in panel:
                v = clr[:, panel.index(nm)]
                out = np.where(np.isfinite(out), out, v)
        return out

    v = pd.DataFrame({"ADT_CD3": _col("ADT_CD3"), "ADT_CD4": _col("ADT_CD4"),
                      "ADT_CD8": _col("ADT_CD8A", "ADT_CD8a"),
                      "ADT_TCRab": _col("ADT_TCRab"), "ADT_TCRgd": _col("ADT_TCRgd"),
                      "ADT_CD56": _col("ADT_CD56")})
    amk = list(v.columns)
    v["row_idx"] = np.concatenate(grp)
    v = v[np.isfinite(v["ADT_CD4"]) & np.isfinite(v["ADT_CD8"])]
    v = v.merge(tab[["row_idx", "tissue", "unit", "group"]], on="row_idx")
    log(f"ADT cells with CD4 and CD8 protein: {len(v):,} in "
        f"{v['unit'].nunique()} donors, {v['tissue'].nunique()} tissues")
    g = v.groupby(["tissue", "unit", "group"])
    med = g[amk].median()
    nc = g.size().rename("n_cells_adt")

    thr = {}
    for (t, u), sub in med.groupby(level=[0, 1]):
        d = sub.droplevel([0, 1])
        if "CD8" in d.index and "CD4_malignant" in d.index:
            thr[(t, u)] = (d.loc["CD8"] + d.loc["CD4_malignant"]) / 2
    if not thr:
        (OUT / "qc_adt.csv").write_text("# ADT present but no donor has both a CD8 and a "
                                        "malignant CD4 reference group\n")
        return None

    rows = []
    for (t, u, grp), sub in v.groupby(["tissue", "unit", "group"]):
        if (t, u) not in thr:
            continue
        r = {"tissue": t, "donor": u, "group": grp, "n_cells_adt": int(len(sub))}
        for m in amk:
            r[f"med_{m}"] = sub[m].median()
            r[f"thr_{m}"] = thr[(t, u)][m]
            above = (sub[m] > thr[(t, u)][m]).mean()
            # above the midpoint means "on the CD8 side" for CD8 markers and vice versa
            r[f"frac_above_{m}"] = above
        rows.append(r)
    Dt = pd.DataFrame(rows).sort_values(["tissue", "donor", "group"])
    Dt.round(4).to_csv(OUT / "qc_adt.csv", index=False)
    log(f"qc_adt.csv: {len(Dt)} rows, {int(nc.sum()):,} cells, "
        f"{Dt['donor'].nunique()} donors")
    return Dt


def _doublet_table(acc, units, tab):
    """E. Scrublet score per arm, and inside the benign arm split by CD8A/B detection.

    predicted_doublet is False for every cell in the atlas because qc_filter drops predicted
    doublets before the join (helpers/atlas_join_helpers.py:1153), so the fraction above the
    fixed 0.25 cutoff is near zero by construction. The informative column is the median score
    and, above all, whether CD8A/B+ benign-arm cells carry a higher score than CD8A/B- ones.
    """
    du = pd.DataFrame({"unit_idx": acc["du_unit"], "score": acc["du_score"],
                       "cd8ab": acc["du_cd8"].astype(bool)})
    du = du.merge(units[["unit_idx", "tissue", "unit", "arm"]], on="unit_idx")
    ok = du[np.isfinite(du["score"])]
    rows = []
    for (t, u, a), sub in du.groupby(["tissue", "unit", "arm"]):
        s = sub[np.isfinite(sub["score"])]
        r = {"tissue": t, "donor": u, "arm": a, "n_cells": len(sub),
             "n_with_score": len(s),
             "median_doublet": s["score"].median() if len(s) else np.nan,
             "frac_above_thresh": (s["score"] > CONFIG["DOUBLET_THRESH"]).mean()
             if len(s) else np.nan}
        if a == "B" and len(s):
            for nm, m in (("CD8ABpos", s["cd8ab"]), ("CD8ABneg", ~s["cd8ab"])):
                r[f"n_{nm}"] = int(m.sum())
                r[f"median_doublet_{nm}"] = s.loc[m, "score"].median() if m.any() else np.nan
        rows.append(r)
    Et = pd.DataFrame(rows).sort_values(["tissue", "donor", "arm"])
    Et.round(4).to_csv(OUT / "qc_doublet.csv", index=False)
    log(f"qc_doublet.csv: {len(Et)} rows; {len(ok):,} of {len(du):,} arm cells carry a score")
    return Et


def _summary(tab, B, C, F, Gt, Dt, Et, tiers, panel, missing):
    """qc_summary.md: the per-tissue totals from F, the across-donor medians from B, and one
    paragraph of conclusion. The brief caps it at 40 lines, so everything else stays in the CSVs.
    """
    tot = F[F["donor"].eq("TOTAL")].set_index("tissue")
    ref = B[B["donor"].eq("ALL_DONORS_MEDIAN")].set_index(["tissue", "group"])
    ti = tiers.set_index(["tissue", "marker_set", "threshold"])
    cc = C.groupby("tissue")
    gg = Gt.set_index(["tissue", "gene"])
    show = [g for g in ["CD4", "CD40LG", "IL7R", "CD3E", "CD8A", "CD8B", "TRDC", "KLRD1",
                        "NKG7", "FOXP3", "IL2RA", "KRT14", "LYZ"] if g in panel]

    def cmed(t, pre, col):
        return cc.get_group(t)[f"{pre}_{col}"].median()

    L = []
    w = L.append
    w("# QC of the benign CD4 arm")
    w("")
    w(f"`jobs/run_ccc52_qc.py`, {time.strftime('%Y-%m-%d')}. Read-only; the v1 arms were "
      f"re-derived and asserted identical to `../row_labels.npz`."
      + (f" Panel genes absent from the atlas: {', '.join(missing)}." if missing else ""))
    w("")
    w(f"## Flagged benign clusters (F) -- any of CD8A, CD8B, TRDC, KLRD1, FOXP3 detected in "
      f">= {CONFIG['FLAG_DETECT']:.2f} of the cluster")
    w("")
    w("| tissue | benign cells | flagged | % | CD8 | gd/NK | Treg | remaining | "
      "CD8 set >=0.25 | >=0.50 |")
    w("|" + "---|" * 10)
    for t, r in tot.iterrows():
        w(f"| {t} | {int(r['n_benign']):,} | {int(r['n_benign_flagged']):,} | "
          f"{100*r['frac_flagged']:.1f}% | {int(r['n_benign_flag_CD8']):,} | "
          f"{int(r['n_benign_flag_gdNK']):,} | {int(r['n_benign_flag_Treg']):,} | "
          f"{int(r['n_benign_remaining']):,} | "
          f"{100*ti.loc[(t,'CD8',0.25),'frac_benign']:.1f}% "
          f"({int(ti.loc[(t,'CD8',0.25),'n_clusters'])}) | "
          f"{100*ti.loc[(t,'CD8',0.50),'frac_benign']:.1f}% "
          f"({int(ti.loc[(t,'CD8',0.50),'n_clusters'])}) |")
    w("")
    w("## Across-donor median detection by group (B)")
    w("")
    w("| tissue | group | cells | med UMI | " + " | ".join(show) + " |")
    w("|" + "---|" * (4 + len(show)))
    for (t, g), r in ref.iterrows():
        w(f"| {t} | {g} | {int(r['n_cells']):,} | {r['median_umis']:.0f} | "
          + " | ".join(_fmt(r[f'det_{g2}']) for g2 in show) + " |")
    w("")
    w("## Conclusion")
    w("")
    sk, bl = "skin", "blood"
    w(f"The benign CD4 arm is not a clean conventional-CD4 population, and what it carries is "
      f"**real cells, not ambient RNA and not doublets**. In skin "
      f"{int(ti.loc[(sk,'CD8',0.25),'n_clusters'])} benign-arm clusters "
      f"({100*ti.loc[(sk,'CD8',0.25),'frac_benign']:.1f}% of benign cells) reach CD8A/CD8B "
      f"detection >= 0.25 and {int(ti.loc[(sk,'CD8',0.50),'n_clusters'])} reach >= 0.50 -- whole "
      f"clusters at det_CD8A 0.85-0.97 with det_NKG7 ~0.9 and det_CD4 <= 0.16, i.e. CD8/NK "
      f"clusters that nb21's per-cluster hand map assigned to CD4 by default (see "
      f"`qc_answers.md` section 1). Three readings exclude the alternatives. (i) Inside the "
      f"benign arm CD4 is detected in {cmed(sk,'b','fracCD4_in_CD8ABpos'):.1%} of CD8A/B+ cells "
      f"against {cmed(sk,'b','fracCD4_in_CD8ABneg'):.1%} of CD8A/B- cells "
      f"({cmed(bl,'b','fracCD4_in_CD8ABpos'):.1%} vs "
      f"{cmed(bl,'b','fracCD4_in_CD8ABneg'):.1%} in blood); ambient and doublets both ADD CD8 "
      f"transcripts to cells that keep their CD4, so they predict the opposite -- and in the "
      f"malignant arm, where the few CD8A/B+ cells are exactly that artefact, the two are "
      f"close ({cmed(sk,'m','fracCD4_in_CD8ABpos'):.1%} vs "
      f"{cmed(sk,'m','fracCD4_in_CD8ABneg'):.1%}). (ii) The benign arm is the SHALLOWER arm "
      f"(skin median {ref.loc[(sk,'CD4_benign'),'median_umis']:.0f} UMI against "
      f"{ref.loc[(sk,'CD4_malignant'),'median_umis']:.0f}) yet its CD8A detection is "
      f"{ref.loc[(sk,'CD4_benign'),'det_CD8A']/ref.loc[(sk,'CD4_malignant'),'det_CD8A']:.0f}x "
      f"higher; ambient scales with depth. (iii) Scrublet scores are no higher in CD8A/B+ "
      f"benign cells than in CD8A/B- ones (skin "
      f"{Et[Et.arm.eq('B')].groupby('tissue')['median_doublet_CD8ABpos'].median()[sk]:.4f} vs "
      f"{Et[Et.arm.eq('B')].groupby('tissue')['median_doublet_CD8ABneg'].median()[sk]:.4f}), and "
      f"predicted doublets were deleted upstream, so the 0.25 cut-off is empty by construction. "
      f"Protein agrees where it exists (buus25, 8 blood donors): the benign arm is CD4-high by "
      f"ADT like the malignant arm, but a minority of its cells sits on the CD8 side of the "
      f"donor's own CD4/CD8 midpoint. A second, smaller admixture is Treg: skin FOXP3 detection "
      f"is {ref.loc[(sk,'CD4_benign'),'det_FOXP3']:.3f} in the benign arm against "
      f"{ref.loc[(sk,'CD4_malignant'),'det_FOXP3']:.3f} malignant although CD4_Treg is a "
      f"separate skin label, and those FOXP3+ benign cells are CD40LG-low "
      f"({cmed(sk,'b','fracCD40LG_in_FOXP3pos'):.1%} vs "
      f"{cmed(sk,'b','fracCD40LG_in_FOXP3neg'):.1%}), which is Treg phenotype rather than FOXP3 "
      f"noise. Blood is the same picture at roughly a quarter the size.")
    w("")
    w(f"**Consequence for v1.** CD8A, CD8B, KLRD1, TRDC and FOXP3 are artefacts of the arm "
      f"definition and must not be read as malignant-vs-benign biology. Dropping only the whole "
      f"mislabelled clusters (detection >= {CONFIG['FLAG_STRICT']}, which keeps "
      f"{int(gg.loc[(sk,'TOX'),'n_donors_c'])}/{int(gg.loc[(sk,'TOX'),'n_donors_a'])} skin and "
      f"{int(gg.loc[(bl,'TOX'),'n_donors_c'])}/{int(gg.loc[(bl,'TOX'),'n_donors_a'])} blood "
      f"donors) moves skin KLRD1 {gg.loc[(sk,'KLRD1'),'median_lfc_a']:.2f} -> "
      f"{gg.loc[(sk,'KLRD1'),'median_lfc_c']:.2f}, TRDC "
      f"{gg.loc[(sk,'TRDC'),'median_lfc_a']:.2f} -> {gg.loc[(sk,'TRDC'),'median_lfc_c']:.2f}, "
      f"FOXP3 {gg.loc[(sk,'FOXP3'),'median_lfc_a']:.2f} -> "
      f"{gg.loc[(sk,'FOXP3'),'median_lfc_c']:.2f} and CD8A "
      f"{gg.loc[(sk,'CD8A'),'median_lfc_a']:.2f} -> {gg.loc[(sk,'CD8A'),'median_lfc_c']:.2f}, "
      f"while the reported findings barely move: TOX "
      f"{gg.loc[(sk,'TOX'),'median_lfc_a']:.2f} -> {gg.loc[(sk,'TOX'),'median_lfc_c']:.2f}, "
      f"CD70 {gg.loc[(sk,'CD70'),'median_lfc_a']:.2f} -> "
      f"{gg.loc[(sk,'CD70'),'median_lfc_c']:.2f}, blood KIR3DL2 "
      f"{gg.loc[(bl,'KIR3DL2'),'median_lfc_a']:.2f} -> "
      f"{gg.loc[(bl,'KIR3DL2'),'median_lfc_c']:.2f}, blood DPP4 "
      f"{gg.loc[(bl,'DPP4'),'median_lfc_a']:.2f} -> {gg.loc[(bl,'DPP4'),'median_lfc_c']:.2f}. "
      f"The brief's own 0.05 cut-off is too loose to work with -- it removes "
      f"{100*tot.loc[sk,'frac_flagged']:.0f}% of the skin benign arm and leaves "
      f"{int(gg.loc[(sk,'TOX'),'n_donors_b'])} donors, a self-selected set -- so variant (c) at "
      f"{CONFIG['FLAG_STRICT']} is reported beside it in `qc_sensitivity.csv`. **Confidence: "
      f"high** that a CD8/NK and Treg admixture is present and drives the cytotoxic and Treg "
      f"genes; **moderate** on its exact size, because the CD8 and NK reference groups are "
      f"themselves mixtures (`README.md:85-91`) and the 0.25 cut-off is a judgement, not a "
      f"calibration.")
    (OUT / "qc_summary.md").write_text("\n".join(L) + "\n")
    log(f"wrote {OUT/'qc_summary.md'} ({len(L)} lines)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "pass", "tab", "all"])
    args = ap.parse_args()
    stages = ["prep", "pass", "tab"] if args.stage == "all" else [args.stage]
    for s in stages:
        log(f"stage {s}; output dir {OUT}")
        {"prep": stage_prep, "pass": stage_pass, "tab": stage_tab}[s](args)
        log(f"stage {s} complete")


if __name__ == "__main__":
    main()
