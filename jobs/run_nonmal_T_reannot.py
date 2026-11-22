#!/usr/bin/env python
"""Re-annotate the non-malignant T/NK cells of the MF/SS atlas (skin and blood, separately).

Why. `cell_type_final == "CD4"` is an else-branch: nb21 (skin) and nb31 (blood) hand-mapped
Leiden clusters from dot plots and gave CD4 to every T cluster not positively called something
else. The QC in reports/ccc52_mvb/qc_benign_arm/ measured the damage -- 21.9% of the skin benign
CD4 arm sits in clusters with CD8A/B detection >= 0.25, plus a Treg admixture -- and showed the
contaminants are real cells inside mixed per-sample clusters, so no filter on those clusters can
separate them. This script rebuilds the lineage call from an embedding that the malignant cells
never enter.

Nothing existing is modified: `mal_cluster` is read, never written, and the labels land in a new
file, data/atlas_joint/nonmal_T_labels_v1.parquet.

Stages (./run_nonmal_T_reannot.sh <stage>):

  prep   cell selection, batch keys, the v1 arms for comparison, row labels for the passes
  passA  streamed pass 1: per-dataset detection (which genes a dataset actually measures),
         per-tissue mean/variance of log-normalised counts (HVG), per-cell library size
  passB  streamed pass 2: the counts submatrix for the selected cells over HVG + marker genes,
         plus the malignant-arm aggregates that tables G and H need
  embed  GPU: scVI per tissue (batch = sample in skin, donor in blood), Leiden over-clustering
  label  per-cluster marker scoring, one round of sub-clustering, the label parquet
  tables the seven CSVs and reannot_summary.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

NB_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NB_DIR / "helpers"))

ATLAS = NB_DIR / "data" / "atlas_joint"
OUT = NB_DIR / "reports" / "nonmal_T_reannot"
WORK = OUT / "_work"
LABELS_OUT = ATLAS / "nonmal_T_labels_v1.parquet"

CONFIG = dict(
    JOINT_H5AD=ATLAS / "joint_annotated.h5ad",
    JOINT_LAYER="raw_counts",
    OBS_PARQUET=ATLAS / "atlas_obs_full_v2.parquet",
    MAL_PARQUET={"skin": ATLAS / "skin_T_malignancy_cluster_v1.parquet",
                 "blood": ATLAS / "blood_T_malignancy_cluster_v1.parquet"},
    CLUSTERS={"skin": ATLAS / "skin_T_clusters_v1.parquet",
              "blood": ATLAS / "blood_T_clusters_v1.parquet"},
    BLOOD_SUBCLUSTERS=NB_DIR / "reports" / "ccc52_mvb" / "blood_benign_subclusters.csv",
    V1_UNITS=NB_DIR / "reports" / "ccc52_mvb" / "units.csv",
    # ---- cells ---------------------------------------------------------------------
    T_LABELS=("CD4", "CD4_Treg", "CD8", "NK"),
    BATCH_KEY={"skin": "sample_id", "blood": "donor"},   # blood clusters are per donor already
    # ---- genes ----------------------------------------------------------------------
    N_HVG=2000,
    MIN_DETECT_HVG=0.01,        # a HVG candidate must be detected in >= 1% of the tissue's cells.
                                # Without this floor the binned-dispersion z picks near-empty
                                # genes: the first run returned a set whose median detection was
                                # 1e-4, i.e. half of it was noise.
    MIN_EXPECTED_DETECT=10.0,   # a gene is "not measured" in a dataset if it has zero counts
                                # there while >= this many detections were expected
    EXCLUDE_RE=r"^(TR[ABGD][VDJ]|IG[HKL][VDJ]|IGHA\d|IGHG\d|IGHM|IGHE|IGHD|IGKC|IGLC\d|"
               r"MT-|MTRNR|RP[LS]\d|RPLP\d|RPSA|HB[ABDEGMQZ]\d?$)",
    # ---- embedding --------------------------------------------------------------------
    N_LATENT=30,
    N_LAYERS=2,
    MAX_EPOCHS=60,
    BATCH_SIZE=1024,
    LEIDEN_RES=2.0,
    LEIDEN_RES_SUB=0.5,
    N_NEIGHBORS=15,
    # ---- labelling ----------------------------------------------------------------------
    MIN_INFORM=0.10,     # a marker is informative in a tissue only if its best cluster reaches
                         # this detection; otherwise it is dropped from gate and score
    PROLIF_U=0.50,       # relative MKI67/TOP2A level that counts as proliferating
    MARGIN=0.10,         # best lineage score must beat the runner-up by this much
    MIN_ANCHOR_CELLS=200,  # clusters smaller than this do not set a marker's anchor level
    MIN_SUBCLUSTER=200,  # Mixed clusters smaller than this are not sub-clustered
    SUSPECT_DONOR=0.80,  # one donor contributes >= this fraction of the cluster ...
    SUSPECT_TRBV=0.50,   # ... and one TRBV >= this fraction of its TCR-bearing cells
    LOWQC_UMI=0.5,       # median UMI below this fraction of the tissue median
    # ---- arms / statistics ----------------------------------------------------------------
    MIN_CELLS_PER_ARM=50,
    DROP_DATASETS={"blood": ["geskin26"]},
    SKIN_NON_CD4_PHENOTYPES=("CD8", "gd", "DN"),
    SKIN_NON_CD4_DX=("CD8_aggr_CTCL", "pcAECTCL"),
    SKIN_OFF_SPECTRUM_DX=("LyP", "SPP", "LPP"),
    SEED=0,
    CHUNK=20000,
)

# detection columns reported in tables B and C
REPORT_B = ["CD4", "CD40LG", "IL7R", "CD3E", "CD8A", "CD8B", "TRDC", "TRGC1", "KLRD1", "KLRF1",
            "NCAM1", "NKG7", "GZMK", "SLC4A10", "KLRB1", "FOXP3", "IL2RA", "CTLA4", "CCR7",
            "SELL", "TCF7", "S100A4", "MKI67", "TOX", "KIR3DL2"]

SENS = ["CD8A", "CD8B", "KLRD1", "TRDC", "FOXP3", "CD4", "TOX", "DPP4", "KIR3DL2", "PLS3",
        "CD7", "CCL5", "CCL4", "KLRK1", "CXCR6", "CXCR3", "CCR6", "ITGA4", "CXCR4", "SELL",
        "CD27", "ICOS", "AREG", "IL10RA", "PTHLH", "PTH2R", "IGFL2", "CXCL13", "CD70", "PDGFA",
        "CD9", "IL13", "IL4", "CCR4", "CD28", "CTLA4", "PDCD1", "TNFSF11", "TNFSF9", "ITGB1",
        "CSF2"]

EXTRA = ["CD3D", "TRGC2", "TYROBP", "FCGR3A", "KIT", "IKZF2", "LEF1", "TOP2A", "TRAC", "TRBC1",
         "TRBC2", "GNLY", "GZMB", "PRF1", "PTPRC", "LYZ", "CD2", "CD5", "ZBTB16", "RORC"]

REPORT = list(dict.fromkeys(REPORT_B + SENS + EXTRA))

VOCAB = ["CD4_conv", "CD4_naive", "CD4_Treg", "CD8", "gdT", "MAIT", "NK_ILC", "Prolif",
         "LowQC", "Mixed", "Suspect_malignant"]

# Lineage definitions. Each is (positive markers, negative markers). Scores are means of robust
# z-scores of the per-cluster DETECTION fraction, computed across the clusters of that tissue, so
# nothing here is a fixed cut-off -- the comparison is always to the rest of the tissue.
# Lineage definitions. `gate` is what a cluster must satisfy before the label is even
# considered, `pos`/`neg` build the score among the labels whose gate passed. Every level is
# RELATIVE: a marker's detection is divided by the highest detection any cluster of that tissue
# reaches for it, so "u >= 0.5" means "at least half as positive as the most positive cluster
# here", never a fixed fraction of cells. A marker whose best cluster never reaches MIN_INFORM
# detection is declared uninformative in that tissue and drops out of both gate and score.
#   gate entries: (feature, ">=" | "<=", level), all of which must hold.
LINEAGE = {
    "NK_ILC":   dict(gate=[("NK_ANY", ">=", 0.50), ("frac_tcr", "<=", 0.50),
                           ("CD3E", "<=", 0.60)],
                     pos=["KLRF1", "NCAM1", "FCGR3A", "TYROBP", "KIT"],
                     neg=["CD3E", "CD3D", "CD4", "frac_tcr", "CD8B"]),
    "gdT":      dict(gate=[("TRDC", ">=", 0.50), ("CD3E", ">=", 0.40)],
                     pos=["TRDC", "TRGC1", "TRGC2"],
                     neg=["CD4", "CD8B", "KLRF1"]),
    "MAIT":     dict(gate=[("MAIT_ANY", ">=", 0.50), ("CD3E", ">=", 0.40)],
                     pos=["SLC4A10", "KLRB1", "trav12_frac"],
                     neg=["CCR7", "SELL", "KLRF1"]),
    "CD8":      dict(gate=[("CD8_ANY", ">=", 0.45), ("CD3E", ">=", 0.40)],
                     pos=["CD8B", "CD8A", "GZMK", "NKG7"],
                     neg=["CD4", "CD40LG", "KLRF1", "FOXP3"]),
    "CD4_Treg": dict(gate=[("FOXP3", ">=", 0.50), ("CD3E", ">=", 0.40),
                           ("CD8_ANY", "<=", 0.30)],
                     pos=["FOXP3", "IL2RA", "IKZF2", "CTLA4"],
                     neg=["CD40LG", "CD8B", "KLRF1"]),
    "CD4_naive": dict(gate=[("CCR7", ">=", 0.50), ("SELL", ">=", 0.50),
                            ("S100A4", "<=", 0.65), ("CD3E", ">=", 0.40),
                            ("CD4_ANY", ">=", 0.35), ("CD8_ANY", "<=", 0.25)],
                      pos=["CCR7", "SELL", "TCF7", "LEF1"],
                      neg=["S100A4", "NKG7", "CD8B", "FOXP3"]),
    "CD4_conv": dict(gate=[("CD4_ANY", ">=", 0.40), ("CD3E", ">=", 0.50),
                           ("CD8_ANY", "<=", 0.25), ("NK_ANY", "<=", 0.35)],
                     pos=["CD4", "CD40LG", "IL7R", "S100A4"],
                     neg=["CD8B", "KLRF1", "FOXP3", "TCF7"]),
}

# composite gate features: the best of a small set, because any single lineage marker drops out
# in some chemistry (CD8B is weak in 3-prime skin, CD4 is poorly captured everywhere)
# IL7R deliberately absent from CD4_ANY: naive CD8 cells are IL7R-high, and including it sent
# tens of thousands of them into the CD4 labels on the first run.
COMPOSITE = {"NK_ANY": ["KLRF1", "NCAM1", "TYROBP"], "MAIT_ANY": ["SLC4A10", "trav12_frac"],
             "CD8_ANY": ["CD8A", "CD8B"], "CD4_ANY": ["CD4", "CD40LG"]}

# the new benign arm. Skin keeps its naive CD4 (v1 did too); blood drops them, as v1's own blood
# clean-up did -- that is the only reading of "CD4_conv excluding CD4_naive in blood" that says
# anything, since the two labels are disjoint by construction.
NEW_BENIGN = {"skin": ("CD4_conv", "CD4_naive"), "blood": ("CD4_conv",)}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _read_h5ad_str(h5, key):
    import h5py
    d = h5[key]
    if isinstance(d, h5py.Group):
        cats = np.array([x.decode() if isinstance(x, bytes) else x for x in d["categories"][:]])
        codes = d["codes"][:]
        return np.where(codes < 0, "", cats[np.clip(codes, 0, None)])
    return np.array([x.decode() if isinstance(x, bytes) else x for x in d[:]])


def _versions():
    import importlib.metadata
    out = {}
    for p in ("numpy", "pandas", "scipy", "scanpy", "anndata", "scvi-tools", "torch",
              "leidenalg", "igraph", "polars", "h5py"):
        try:
            out[p] = importlib.metadata.version(p)
        except Exception:
            out[p] = None
    out["python"] = sys.version.split()[0]
    return out


# ======================================================================================
# stage: prep
# ======================================================================================
def stage_prep(args):
    import polars as pl
    import h5py

    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    cols = ["cell_id", "dataset", "study", "donor", "sample_id", "patient_key",
            "clin_person_key", "compartment", "tech", "clin_dx", "clin_phenotype",
            "cell_type_final", "disease"]
    obs = (pl.read_parquet(CONFIG["OBS_PARQUET"], columns=cols)
             .with_columns([pl.col(c).cast(pl.Utf8) for c in cols])
             .to_pandas().set_index("cell_id"))
    obs["tissue"] = obs["compartment"].map({"Skin": "skin", "Blood": "blood"})
    log(f"obs {obs.shape}")

    mal = pd.concat([pd.read_parquet(p, columns=["cell_id", "mal_cluster", "call_source"])
                     for p in CONFIG["MAL_PARQUET"].values()]).set_index("cell_id")
    obs["mal"] = mal["mal_cluster"].reindex(obs.index)
    obs["call_source"] = mal["call_source"].reindex(obs.index)

    clus = pd.concat([pd.read_parquet(p, columns=["cell_id", "cluster", "top_trbv", "top_trav",
                                                  "tcr_ok"])
                      for p in CONFIG["CLUSTERS"].values()]).set_index("cell_id")
    for c in ("cluster", "top_trbv", "top_trav", "tcr_ok"):
        obs[c] = clus[c].reindex(obs.index)

    obs["unit"] = (obs["clin_person_key"].replace({"": np.nan, "nan": np.nan, "None": np.nan})
                   .fillna(obs["patient_key"]).fillna(obs["donor"]))

    # ---- v1 arms, reproduced for every donor so the new arm can be compared ---------
    is_cd4 = obs["cell_type_final"].eq("CD4") & obs["tissue"].notna()
    v1 = pd.Series(pd.NA, index=obs.index, dtype="object")
    v1[is_cd4 & obs["mal"].eq(True)] = "M"
    v1[is_cd4 & obs["mal"].eq(False)] = "B"
    sub = pd.read_csv(CONFIG["BLOOD_SUBCLUSTERS"])
    dropped = set(sub.loc[sub["dropped"].astype(bool), "cluster"])
    moved = obs["cluster"].isin(dropped) & v1.eq("B") & obs["tissue"].eq("blood")
    v1[moved] = pd.NA
    obs["v1_arm"] = v1

    keep = pd.Series(True, index=obs.index)
    skin = obs["tissue"].eq("skin")
    keep[skin & (obs["clin_phenotype"].isin(CONFIG["SKIN_NON_CD4_PHENOTYPES"])
                 | obs["clin_dx"].isin(CONFIG["SKIN_NON_CD4_DX"])
                 | obs["clin_dx"].isin(CONFIG["SKIN_OFF_SPECTRUM_DX"]))] = False
    for t, ds in CONFIG["DROP_DATASETS"].items():
        keep[obs["tissue"].eq(t) & obs["dataset"].isin(ds)] = False
    obs["v1_cohort_keep"] = keep

    # ---- selection -------------------------------------------------------------------
    inT = obs["cell_type_final"].isin(CONFIG["T_LABELS"]) & obs["tissue"].notna()
    obs["mal_null"] = obs["mal"].isna()
    sel = inT & ~obs["mal"].eq(True)
    mal_arm = is_cd4 & obs["mal"].eq(True)
    log(f"T compartment {int(inT.sum()):,}; non-malignant selected {int(sel.sum()):,} "
        f"({int((sel & obs['mal_null']).sum()):,} with a null malignancy call); "
        f"malignant CD4 arm {int(mal_arm.sum()):,}")
    log("selected by tissue x cell_type_final:\n"
        + obs[sel].groupby(["tissue", "cell_type_final"]).size().unstack(fill_value=0).to_string())

    obs["batch"] = np.where(obs["tissue"].eq("skin"),
                            obs[CONFIG["BATCH_KEY"]["skin"]].astype(str),
                            obs[CONFIG["BATCH_KEY"]["blood"]].astype(str))

    with h5py.File(CONFIG["JOINT_H5AD"], "r") as f:
        joint_ids = _read_h5ad_str(f["obs"], "cell_id")
    o = obs.reindex(pd.Index(joint_ids, name="cell_id"))

    s = o.index.isin(obs.index[sel])
    m = o.index.isin(obs.index[mal_arm])
    cells = o[s].copy()
    cells["row"] = np.arange(len(cells))
    cells.reset_index().to_parquet(WORK / "cells.parquet", index=False)

    row_sel = np.full(len(o), -1, dtype=np.int32)
    row_sel[np.flatnonzero(s)] = np.arange(int(s.sum()), dtype=np.int32)

    # dataset key for the "which genes does this dataset measure" test
    dk = (o["tissue"].astype(str) + "|" + o["dataset"].astype(str))
    dkeys = sorted(set(dk[s]))
    dmap = {k: i for i, k in enumerate(dkeys)}
    row_ds = np.where(s, dk.map(dmap).fillna(-1).to_numpy(), -1).astype(np.int32)

    tkeys = ["blood", "skin"]
    row_tis = np.where(s, o["tissue"].map({t: i for i, t in enumerate(tkeys)})
                       .fillna(-1).to_numpy(), -1).astype(np.int32)

    # malignant arm, aggregated per (tissue, v1 unit)
    mu = (o["tissue"].astype(str) + "|" + o["unit"].astype(str))
    mkeys = sorted(set(mu[m]))
    mmap = {k: i for i, k in enumerate(mkeys)}
    row_mal = np.where(m, mu.map(mmap).fillna(-1).to_numpy(), -1).astype(np.int32)

    np.savez_compressed(WORK / "row_labels.npz", row_sel=row_sel, row_ds=row_ds,
                        row_tis=row_tis, row_mal=row_mal,
                        dkeys=np.array(dkeys), tkeys=np.array(tkeys), mkeys=np.array(mkeys))
    log(f"rows: selected {int((row_sel>=0).sum()):,}, malignant-arm {int((row_mal>=0).sum()):,}, "
        f"{len(dkeys)} tissue x dataset keys, {len(mkeys)} malignant units")
    (OUT / "_versions.json").write_text(json.dumps(_versions(), indent=1))
    log("prep done")


# ======================================================================================
# stage: passA
# ======================================================================================
def stage_passA(args):
    """Per-dataset detection, per-tissue log-normalised mean/variance, per-cell library size."""
    import h5py
    from scipy import sparse

    lab = np.load(WORK / "row_labels.npz", allow_pickle=False)
    row_sel, row_ds, row_tis = lab["row_sel"], lab["row_ds"], lab["row_tis"]
    n_ds, n_tis = len(lab["dkeys"]), len(lab["tkeys"])
    n_sel = int((row_sel >= 0).sum())

    with h5py.File(CONFIG["JOINT_H5AD"], "r") as f:
        L = f["layers"][CONFIG["JOINT_LAYER"]]
        n, G = (int(x) for x in f["X"].attrs["shape"])
        var = _read_h5ad_str(f["var"], f["var"].attrs.get("_index", "_index"))
        assert len(row_sel) == n

        DET = np.zeros((n_ds, G))
        NDS = np.zeros(n_ds, dtype=np.int64)
        SL = np.zeros((n_tis, G))
        SL2 = np.zeros((n_tis, G))
        NT = np.zeros(n_tis, dtype=np.int64)
        lib = np.zeros(n_sel, dtype=np.float32)

        indptr = L["indptr"][:]
        chunk = CONFIG["CHUNK"]
        t0 = time.time()
        checked = False
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            rs = row_sel[a:b]
            if not (rs >= 0).any():
                continue
            lo, hi = int(indptr[a]), int(indptr[b])
            M = sparse.csr_matrix(
                (L["data"][lo:hi], L["indices"][lo:hi], indptr[a:b + 1] - lo), shape=(b - a, G))
            if not checked:
                assert np.all(M.data == np.round(M.data)), "counts layer is not integer"
                log(f"counts verified integer ({M.nnz:,} nnz in the first chunk)")
                checked = True
            rows = np.asarray(M.sum(axis=1)).ravel()
            keep = np.flatnonzero(rs >= 0)
            lib[rs[keep]] = rows[keep]

            Bi = M.copy()
            Bi.data = (Bi.data > 0).astype(np.float64)
            rd = row_ds[a:b]
            g = sparse.csr_matrix((np.ones(len(keep)), (rd[keep], keep)), shape=(n_ds, b - a))
            DET += g @ Bi
            NDS += np.bincount(rd[keep], minlength=n_ds)

            sf = np.zeros(b - a)
            sf[keep] = 1e4 / np.maximum(rows[keep], 1)
            Ln = sparse.diags(sf) @ M
            Ln.data = np.log1p(Ln.data)
            L2 = Ln.copy()
            L2.data = L2.data ** 2
            rt = row_tis[a:b]
            gt = sparse.csr_matrix((np.ones(len(keep)), (rt[keep], keep)), shape=(n_tis, b - a))
            SL += gt @ Ln
            SL2 += gt @ L2
            NT += np.bincount(rt[keep], minlength=n_tis)
            if (a // chunk) % 25 == 0:
                log(f"  {b:,}/{n:,}  {time.time() - t0:.0f}s")

    log(f"passA done in {time.time() - t0:.0f}s; cells {NT.sum():,}")
    np.savez_compressed(WORK / "passA.npz", DET=DET, NDS=NDS, SL=SL, SL2=SL2, NT=NT,
                        lib=lib, var=var)
    log(f"wrote {WORK/'passA.npz'}")


def _choose_genes():
    """Genes a tissue actually measures, minus clonotype/quality drivers, ranked by dispersion."""
    acc = np.load(WORK / "passA.npz", allow_pickle=True)
    lab = np.load(WORK / "row_labels.npz", allow_pickle=True)
    var = np.array([str(v) for v in acc["var"]])
    dkeys = [str(x) for x in lab["dkeys"]]
    tkeys = [str(x) for x in lab["tkeys"]]
    DET, NDS, SL, SL2, NT = acc["DET"], acc["NDS"], acc["SL"], acc["SL2"], acc["NT"]
    bad = np.array([bool(re.match(CONFIG["EXCLUDE_RE"], g)) for g in var])

    present = set(var)
    rep_genes = [g for g in REPORT if g in present]
    out, report = {}, []
    for ti, t in enumerate(tkeys):
        rows = [i for i, k in enumerate(dkeys) if k.split("|")[0] == t]
        pooled = DET[rows].sum(axis=0) / max(NDS[rows].sum(), 1)
        # "not measured here" = zero counts in a dataset that should have seen the gene often
        unmeasured = np.zeros(len(var), dtype=bool)
        for i in rows:
            expect = NDS[i] * pooled
            miss = (DET[i] == 0) & (expect >= CONFIG["MIN_EXPECTED_DETECT"])
            unmeasured |= miss
            report.append({"tissue": t, "dataset": dkeys[i].split("|")[1], "n_cells": int(NDS[i]),
                           "n_genes_unmeasured": int(miss.sum())})
        mean = SL[ti] / NT[ti]
        var_ = np.maximum(SL2[ti] / NT[ti] - mean ** 2, 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            disp = np.where(mean > 0, var_ / mean, 0)
        cand = (~bad) & (~unmeasured) & (mean > 0) & (pooled >= CONFIG["MIN_DETECT_HVG"])
        # z-score dispersion inside 20 equal-count bins of mean expression (the `seurat` recipe)
        z = np.full(len(var), -np.inf)
        ci = np.flatnonzero(cand)
        if len(ci):
            qs = np.quantile(mean[ci], np.linspace(0, 1, 21))
            bins = np.clip(np.searchsorted(qs, mean[ci], side="right") - 1, 0, 19)
            for bq in range(20):
                s = ci[bins == bq]
                if len(s) < 5:
                    continue
                mu, sd = disp[s].mean(), disp[s].std()
                z[s] = (disp[s] - mu) / (sd if sd > 0 else 1)
        order = np.argsort(-z)[:CONFIG["N_HVG"]]
        # the marker panel is forced in: the embedding has to be able to separate the lineages
        # it will later be labelled by, and CD8B / IL7R / S100A4 are not dispersed enough in skin
        # to survive a pure HVG ranking
        hv = sorted(set(var[order].tolist()) | set(rep_genes))
        out[t] = {"hvg": hv, "n_hvg_ranked": int(len(order)),
                  "n_markers_added": int(len(set(rep_genes) - set(var[order].tolist()))),
                  "median_detection_hvg": float(np.median(pooled[order])),
                  "n_candidates": int(cand.sum()),
                  "n_unmeasured": int(unmeasured.sum())}
        log(f"{t}: {int(cand.sum()):,} candidate genes "
            f"({int(unmeasured.sum()):,} unmeasured in >= 1 dataset, {int(bad.sum()):,} excluded "
            f"by pattern, floor {CONFIG['MIN_DETECT_HVG']}) -> {len(hv)} genes "
            f"(median detection {np.median(pooled[order]):.3f}, "
            f"+{out[t]['n_markers_added']} markers forced in)")
    rep = rep_genes
    missing = [g for g in REPORT if g not in present]
    union = sorted(set(rep) | set().union(*[set(v["hvg"]) for v in out.values()]))
    out["_report"] = rep
    out["_report_missing"] = missing
    out["_union"] = union
    out["_dataset_coverage"] = report
    if missing:
        log(f"WARNING marker genes absent from the atlas: {missing}")
    log(f"union gene space for the counts extraction: {len(union)}")
    (OUT / "genes.json").write_text(json.dumps(out, indent=1))
    return out


# ======================================================================================
# stage: passB
# ======================================================================================
def stage_passB(args):
    import h5py
    from scipy import sparse

    genes = _choose_genes()
    union = genes["_union"]
    rep = genes["_report"]

    lab = np.load(WORK / "row_labels.npz", allow_pickle=True)
    row_sel, row_mal, row_tis = lab["row_sel"], lab["row_mal"], lab["row_tis"]
    mkeys = [str(x) for x in lab["mkeys"]]
    tkeys = [str(x) for x in lab["tkeys"]]
    n_sel = int((row_sel >= 0).sum())

    with h5py.File(CONFIG["JOINT_H5AD"], "r") as f:
        L = f["layers"][CONFIG["JOINT_LAYER"]]
        n, G = (int(x) for x in f["X"].attrs["shape"])
        var = _read_h5ad_str(f["var"], f["var"].attrs.get("_index", "_index"))
        pos = {g: i for i, g in enumerate(var)}
        ucols = [pos[g] for g in union]
        P = sparse.csr_matrix((np.ones(len(ucols)), (ucols, np.arange(len(ucols)))),
                              shape=(G, len(union)))
        rcols = [union.index(g) for g in rep]
        ridx = {g: j for j, g in enumerate(rep)}

        MS = np.zeros((len(mkeys), len(rep)))       # malignant arm: summed counts
        MD = np.zeros((len(mkeys), len(rep)))       # malignant arm: detection counts
        MLIB = np.zeros(len(mkeys))
        MN = np.zeros(len(mkeys), dtype=np.int64)
        MCO = np.zeros((len(tkeys), 7))             # malignant arm co-detection
        # row_tis is -1 outside the selected (non-malignant) cells, so the malignant arm takes
        # its tissue from its own unit key instead
        mal_tis = np.array([tkeys.index(k.split("|", 1)[0]) for k in mkeys], dtype=np.int64)
        blocks = [None] * (n // CONFIG["CHUNK"] + 2)
        order = []

        indptr = L["indptr"][:]
        chunk = CONFIG["CHUNK"]
        t0 = time.time()
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            rs, rm = row_sel[a:b], row_mal[a:b]
            if not ((rs >= 0).any() or (rm >= 0).any()):
                continue
            lo, hi = int(indptr[a]), int(indptr[b])
            M = sparse.csr_matrix(
                (L["data"][lo:hi], L["indices"][lo:hi], indptr[a:b + 1] - lo), shape=(b - a, G))
            S = (M @ P).tocsr()
            if (rs >= 0).any():
                k = np.flatnonzero(rs >= 0)
                blocks[a // chunk] = S[k].astype(np.float32)
                order.append(rs[k])
            if (rm >= 0).any():
                k = np.flatnonzero(rm >= 0)
                R = np.asarray(S[k][:, rcols].todense())
                g = sparse.csr_matrix((np.ones(len(k)), (rm[k], np.arange(len(k)))),
                                      shape=(len(mkeys), len(k)))
                MS += g @ R
                MD += g @ (R > 0).astype(float)
                MLIB += np.bincount(rm[k], weights=np.asarray(M[k].sum(axis=1)).ravel(),
                                    minlength=len(mkeys))
                MN += np.bincount(rm[k], minlength=len(mkeys))
                d = (R > 0).astype(float)
                cd4, cd8 = d[:, ridx["CD4"]], np.maximum(d[:, ridx["CD8A"]], d[:, ridx["CD8B"]])
                fx = d[:, ridx["FOXP3"]]
                rt = mal_tis[rm[k]]
                for j, v in enumerate([np.ones(len(k)), cd4, cd8, cd4 * cd8, fx,
                                       d[:, ridx["CD40LG"]], fx * d[:, ridx["CD40LG"]]]):
                    MCO[:, j] += np.bincount(rt, weights=v, minlength=len(tkeys))
            if (a // chunk) % 25 == 0:
                log(f"  {b:,}/{n:,}  {time.time() - t0:.0f}s")

    X = sparse.vstack([x for x in blocks if x is not None], format="csr")
    pos_all = np.concatenate(order)
    X = X[np.argsort(pos_all)]
    assert X.shape[0] == n_sel, f"{X.shape[0]} != {n_sel}"
    log(f"passB done in {time.time() - t0:.0f}s; counts {X.shape}, {X.nnz:,} nnz")

    import anndata as ad
    cells = pd.read_parquet(WORK / "cells.parquet")
    acc = np.load(WORK / "passA.npz", allow_pickle=True)
    cells["lib"] = acc["lib"]
    keep_obs = ["cell_id", "tissue", "batch", "donor", "sample_id", "unit", "dataset", "study",
                "tech", "cell_type_final", "v1_arm", "call_source", "top_trbv", "top_trav",
                "cluster"]
    meta = cells[keep_obs].copy()
    for c in keep_obs:
        meta[c] = meta[c].astype(str)
    meta["lib"] = cells["lib"].to_numpy()
    meta["tcr_ok"] = cells["tcr_ok"].astype(str).to_numpy()
    meta["mal_null"] = cells["mal_null"].astype(str).to_numpy()
    meta["v1_cohort_keep"] = cells["v1_cohort_keep"].astype(str).to_numpy()
    for t in tkeys:
        m = (cells["tissue"] == t).to_numpy()
        A = ad.AnnData(X=X[m], obs=meta[m].set_index("cell_id").copy(),
                       var=pd.DataFrame(index=pd.Index(union, name="gene")))
        A.write_h5ad(WORK / f"counts_{t}.h5ad")
        log(f"wrote counts_{t}.h5ad {A.shape}")
    np.savez_compressed(WORK / "mal_acc.npz", MS=MS, MD=MD, MLIB=MLIB, MN=MN, MCO=MCO,
                        mkeys=np.array(mkeys), rep=np.array(rep))


# ======================================================================================
# stage: embed
# ======================================================================================
def stage_embed(args):
    import anndata as ad
    import scanpy as sc
    import scvi
    import torch

    genes = json.loads((OUT / "genes.json").read_text())
    scvi.settings.seed = CONFIG["SEED"]
    log(f"cuda: {torch.cuda.is_available()} "
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")

    for t in ("skin", "blood"):
        if args.tissue and args.tissue != t:
            continue
        A = sc.read_h5ad(WORK / f"counts_{t}.h5ad")
        hvg = [g for g in genes[t]["hvg"] if g in set(A.var_names)]
        A = A[:, hvg].copy()
        A.X = A.X.astype(np.float32)
        A.obs["batch"] = A.obs["batch"].astype(str)
        log(f"{t}: {A.shape}, {A.obs['batch'].nunique()} batches")
        scvi.model.SCVI.setup_anndata(A, batch_key="batch")
        model = scvi.model.SCVI(A, n_latent=CONFIG["N_LATENT"], n_layers=CONFIG["N_LAYERS"],
                                gene_likelihood="nb")
        model.train(max_epochs=CONFIG["MAX_EPOCHS"], batch_size=CONFIG["BATCH_SIZE"],
                    early_stopping=True, early_stopping_patience=10)
        Z = model.get_latent_representation()
        np.save(WORK / f"latent_{t}.npy", Z)
        log(f"{t}: latent {Z.shape}; clustering")

        B = ad.AnnData(X=np.zeros((Z.shape[0], 1), dtype=np.float32))
        B.obsm["X_scvi"] = Z
        sc.pp.neighbors(B, use_rep="X_scvi", n_neighbors=CONFIG["N_NEIGHBORS"],
                        random_state=CONFIG["SEED"])
        sc.tl.leiden(B, resolution=CONFIG["LEIDEN_RES"], key_added="cl",
                     random_state=CONFIG["SEED"], flavor="igraph", n_iterations=2,
                     directed=False)
        cl = B.obs["cl"].astype(str).to_numpy()
        pd.DataFrame({"cell_id": A.obs_names, "cluster_raw": cl}).to_parquet(
            WORK / f"clusters_{t}.parquet", index=False)
        log(f"{t}: {len(set(cl))} Leiden clusters at resolution {CONFIG['LEIDEN_RES']}")


# ======================================================================================
# stage: label
# ======================================================================================
def _group_matrix(codes, n):
    from scipy import sparse
    k = np.flatnonzero(codes >= 0)
    return sparse.csr_matrix((np.ones(len(k)), (codes[k], k)), shape=(n, len(codes)))


def _features(A, cl, genes):
    """Per-cluster detection of the marker genes plus the TCR-derived features."""
    gi = {g: i for i, g in enumerate(A.var_names)}
    cols = [gi[g] for g in genes]
    D = A.X[:, cols].copy()
    D.data = (D.data > 0).astype(np.float64)
    cats = pd.Categorical(cl)
    G = _group_matrix(cats.codes.astype(int), len(cats.categories))
    n = np.bincount(cats.codes.astype(int), minlength=len(cats.categories)).astype(float)
    det = np.asarray((G @ D).todense()) / n[:, None]
    f = pd.DataFrame(det, index=cats.categories, columns=genes)
    o = A.obs
    tcr = (o["tcr_ok"].astype(str) == "True").to_numpy().astype(float)
    trav = (o["top_trav"].astype(str) == "TRAV1-2").to_numpy().astype(float)
    f["n_cells"] = n.astype(int)
    f["frac_tcr"] = np.asarray(G @ tcr).ravel() / n
    with np.errstate(invalid="ignore", divide="ignore"):
        f["trav12_frac"] = np.asarray(G @ (trav * tcr)).ravel() / np.maximum(
            np.asarray(G @ tcr).ravel(), 1)
    f["median_umi"] = pd.Series(o["lib"].to_numpy()).groupby(
        pd.Series(cats.codes)).median().reindex(range(len(cats.categories))).to_numpy()
    for col, key in (("n_donors", "donor"), ("top_donor_frac", "donor"),
                     ("top_trbv_frac", "top_trbv")):
        g = pd.Series(o[key].astype(str).to_numpy()).groupby(pd.Series(cl).to_numpy())
        if col == "n_donors":
            f[col] = g.nunique().reindex(f.index).to_numpy()
        elif col == "top_donor_frac":
            f[col] = (g.apply(lambda x: x.value_counts(normalize=True).iloc[0])
                      .reindex(f.index).to_numpy())
    tcr_mask = tcr > 0
    tb = pd.Series(o["top_trbv"].astype(str).to_numpy()[tcr_mask]).groupby(
        pd.Series(np.asarray(cl)[tcr_mask]))
    f["top_trbv_frac"] = (tb.apply(lambda x: x.value_counts(normalize=True).iloc[0])
                          .reindex(f.index).fillna(0).to_numpy())
    return f


def _anchors(f, feats):
    """A marker's anchor is the highest detection any sizeable cluster of this tissue reaches."""
    big = f[f["n_cells"] >= CONFIG["MIN_ANCHOR_CELLS"]]
    if not len(big):
        big = f
    mx = big[feats].max()
    inform = mx >= CONFIG["MIN_INFORM"]
    for k in ("frac_tcr", "trav12_frac"):
        if k in mx.index:
            inform[k] = mx[k] >= CONFIG["MIN_INFORM"]
    return mx.replace(0, np.nan), inform


def _u(f, mx, inform):
    """Detection relative to the tissue's most positive cluster, in [0, 1]."""
    cols = [c for c in mx.index if inform.get(c, False)]
    u = (f[cols] / mx[cols]).clip(0, 1.5)
    for name, members in COMPOSITE.items():
        m = [g for g in members if g in u.columns]
        u[name] = u[m].max(axis=1) if m else 0.0
    return u


def _assign(f, mx, inform, tissue_umi):
    u = _u(f, mx, inform)
    out, tab = [], []
    for c in f.index:
        passed = {}
        for lab, d in LINEAGE.items():
            ok = True
            for feat, op, lvl in d["gate"]:
                if feat not in u.columns:
                    ok = False if op == ">=" else ok
                    continue
                v = u.loc[c, feat]
                if (op == ">=" and not v >= lvl) or (op == "<=" and not v <= lvl):
                    ok = False
                    break
            if not ok:
                continue
            pos = [g for g in d["pos"] if g in u.columns]
            neg = [g for g in d["neg"] if g in u.columns]
            passed[lab] = (float(np.mean(u.loc[c, pos])) if pos else 0.0) \
                - (float(np.mean(u.loc[c, neg])) if neg else 0.0)
        prolif = max(u.loc[c, g] for g in ("MKI67", "TOP2A") if g in u.columns) \
            if any(g in u.columns for g in ("MKI67", "TOP2A")) else 0.0
        lowqc = (f.loc[c, "median_umi"] < CONFIG["LOWQC_UMI"] * tissue_umi
                 and ("CD3E" not in u.columns or u.loc[c, "CD3E"] < 0.5))
        if passed:
            order = sorted(passed.items(), key=lambda kv: -kv[1])
            best = order[0]
            second = order[1] if len(order) > 1 else (None, -9.9)
            lab = best[0] if (best[1] - second[1]) >= CONFIG["MARGIN"] else "Mixed"
        else:
            best, second = (None, np.nan), (None, np.nan)
            lab = "LowQC" if lowqc else ("Prolif" if prolif >= CONFIG["PROLIF_U"] else "Mixed")
        if lab in ("Mixed", "Prolif") and lowqc:
            lab = "LowQC"
        if (f.loc[c, "top_donor_frac"] >= CONFIG["SUSPECT_DONOR"]
                and f.loc[c, "top_trbv_frac"] >= CONFIG["SUSPECT_TRBV"]):
            lab = "Suspect_malignant"
        out.append(lab)
        tab.append({"cluster": c, "n_cells": int(f.loc[c, "n_cells"]), "label": lab,
                    "best": best[0], "best_score": best[1], "second": second[0],
                    "second_score": second[1], "prolif_u": prolif, "lowqc": lowqc,
                    "n_gates_passed": len(passed),
                    "gates_passed": ",".join(sorted(passed))})
    return pd.Series(out, index=f.index), pd.DataFrame(tab)


def stage_label(args):
    import anndata as ad
    import scanpy as sc

    genes_json = json.loads((OUT / "genes.json").read_text())
    present = set(genes_json["_report"])
    score_feats = sorted({g for d in LINEAGE.values() for g in d["pos"] + d["neg"]}
                         | {g for d in LINEAGE.values() for g, _, _ in d["gate"]}
                         | {g for m in COMPOSITE.values() for g in m}
                         | {"MKI67", "TOP2A", "CD3E"})
    score_feats = [g for g in score_feats
                   if g in present or g in ("frac_tcr", "trav12_frac")]

    all_lab, all_tab, all_scores = [], [], []
    for t in ("skin", "blood"):
        A = sc.read_h5ad(WORK / f"counts_{t}.h5ad")
        cl = pd.read_parquet(WORK / f"clusters_{t}.parquet").set_index("cell_id")
        cl = cl["cluster_raw"].reindex(A.obs_names).astype(str).to_numpy()
        Z = np.load(WORK / f"latent_{t}.npy")
        genes = [g for g in REPORT if g in set(A.var_names)]
        tissue_umi = float(np.median(A.obs["lib"].to_numpy()))

        f = _features(A, cl, genes)
        mx, inform = _anchors(f, score_feats)
        log(f"{t}: uninformative markers (best cluster under {CONFIG['MIN_INFORM']}): "
            + str([k for k in mx.index if not inform[k]]))
        lab, sctab = _assign(f, mx, inform, tissue_umi)
        log(f"{t}: first pass {len(f)} clusters -> "
            + str(lab.value_counts().to_dict()))

        # one round of sub-clustering for whatever is still Mixed
        final = pd.Series(lab.reindex(cl).to_numpy(), index=A.obs_names).astype(str)
        cluster_new = pd.Series(np.array(cl, dtype=object), index=A.obs_names)
        mixed = [c for c in f.index if lab[c] == "Mixed"
                 and f.loc[c, "n_cells"] >= CONFIG["MIN_SUBCLUSTER"]]
        log(f"{t}: sub-clustering {len(mixed)} Mixed clusters "
            f"({int(f.loc[mixed, 'n_cells'].sum()) if mixed else 0:,} cells)")
        for c in mixed:
            m = np.flatnonzero(cl == c)
            B = ad.AnnData(X=np.zeros((len(m), 1), dtype=np.float32))
            B.obsm["X_scvi"] = Z[m]
            sc.pp.neighbors(B, use_rep="X_scvi", n_neighbors=min(CONFIG["N_NEIGHBORS"],
                                                                 max(len(m) // 20, 3)),
                            random_state=CONFIG["SEED"])
            sc.tl.leiden(B, resolution=CONFIG["LEIDEN_RES_SUB"], key_added="cl",
                         random_state=CONFIG["SEED"], flavor="igraph", n_iterations=2,
                         directed=False)
            sub = np.array([f"{c}.{x}" for x in B.obs["cl"].astype(str)])
            fs = _features(A[m], sub, genes)
            ls, st = _assign(fs, mx, inform, tissue_umi)
            st["parent"] = c
            all_scores.append(st)
            cluster_new.iloc[m] = sub
            final.iloc[m] = ls.reindex(sub).to_numpy()
            f = pd.concat([f.drop(index=c), fs])
            lab = pd.concat([lab.drop(index=c), ls])
        sctab["parent"] = ""
        all_scores.append(sctab[sctab["cluster"].isin(f.index)])

        f = f.loc[sorted(set(cluster_new))]
        lab = lab.reindex(f.index)
        tb = f.copy()
        tb.insert(0, "label_new", lab.to_numpy())
        tb.insert(0, "cluster_new", [f"{t}_{c}" for c in tb.index])
        tb.insert(0, "tissue", t)
        all_tab.append(tb.reset_index(drop=True))

        out = A.obs[["tissue", "donor", "sample_id"]].copy()
        out["cluster_new"] = [f"{t}_{c}" for c in cluster_new.to_numpy()]
        out["label_new"] = final.to_numpy()
        all_lab.append(out.reset_index())
        log(f"{t}: final labels " + str(pd.Series(final).value_counts().to_dict()))

    lab_df = pd.concat(all_lab, ignore_index=True)
    lab_df.to_parquet(LABELS_OUT, index=False)
    log(f"wrote {LABELS_OUT} ({len(lab_df):,} cells)")

    tab = pd.concat(all_tab, ignore_index=True)
    cols = (["tissue", "cluster_new", "label_new", "n_cells", "n_donors", "top_donor_frac",
             "top_trbv_frac", "frac_tcr", "median_umi"]
            + [f"det_{g}" for g in REPORT_B])
    tab = tab.rename(columns={g: f"det_{g}" for g in REPORT_B})
    tab = tab[[c for c in cols if c in tab.columns]]
    tab.round(4).to_csv(OUT / "reannot_cluster_table.csv", index=False)
    pd.concat(all_scores, ignore_index=True).round(3).to_csv(
        WORK / "cluster_scores.csv", index=False)
    log(f"reannot_cluster_table.csv: {len(tab)} clusters")


# ======================================================================================
# stage: tables
# ======================================================================================
def _detect_by(A, keys, genes):
    """Detection fraction and summed counts per group, for the given genes."""
    gi = {g: i for i, g in enumerate(A.var_names)}
    cols = [gi[g] for g in genes]
    S = A.X[:, cols]
    D = S.copy()
    D.data = (D.data > 0).astype(np.float64)
    cats = pd.Categorical(keys)
    G = _group_matrix(cats.codes.astype(int), len(cats.categories))
    n = np.bincount(cats.codes.astype(int), minlength=len(cats.categories)).astype(float)
    det = np.asarray((G @ D).todense()) / n[:, None]
    tot = np.asarray((G @ S).todense())
    lib = np.asarray(G @ A.obs["lib"].to_numpy()).ravel()
    return (pd.DataFrame(det, index=cats.categories, columns=genes),
            pd.DataFrame(tot, index=cats.categories, columns=genes),
            pd.Series(n, index=cats.categories), pd.Series(lib, index=cats.categories))


def _kmeans2(x):
    """1-D two-means; returns (centre0, centre1, size0, size1) or None if it does not split."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 50:
        return None
    c = np.array([np.quantile(x, 0.1), np.quantile(x, 0.9)])
    for _ in range(50):
        a = np.abs(x[:, None] - c[None, :]).argmin(axis=1)
        if not a.any() or a.all():
            return None
        nc = np.array([x[a == 0].mean(), x[a == 1].mean()])
        if np.allclose(nc, c):
            break
        c = nc
    n0, n1 = int((a == 0).sum()), int((a == 1).sum())
    sd = np.sqrt((x[a == 0].var() * n0 + x[a == 1].var() * n1) / len(x))
    sep = (c[1] - c[0]) / (sd if sd > 0 else 1)
    if sep < 1.5 or min(n0, n1) / len(x) < 0.10:
        return None
    return float(c[0]), float(c[1]), n0, n1


def stage_tables(args):
    import scanpy as sc
    import h5py

    genes_json = json.loads((OUT / "genes.json").read_text())
    rep = genes_json["_report"]
    labels = pd.read_parquet(LABELS_OUT).set_index("cell_id")
    acc = np.load(WORK / "mal_acc.npz", allow_pickle=True)
    mkeys = [str(x) for x in acc["mkeys"]]
    mrep = [str(x) for x in acc["rep"]]
    mi = {g: j for j, g in enumerate(mrep)}
    MD, MS, MLIB, MN, MCO = acc["MD"], acc["MS"], acc["MLIB"], acc["MN"], acc["MCO"]
    mal = pd.DataFrame({"key": mkeys, "n": MN, "lib": MLIB})
    mal[["tissue", "unit"]] = pd.DataFrame([k.split("|", 1) for k in mkeys], index=mal.index)

    C, D, F, G, H, E = [], [], [], [], [], []
    repB = [g for g in REPORT_B if g in rep]
    sens = [g for g in SENS if g in rep]
    per_t = {}

    for t in ("skin", "blood"):
        A = sc.read_h5ad(WORK / f"counts_{t}.h5ad")
        A.obs["label_new"] = labels["label_new"].reindex(A.obs_names).to_numpy()
        A.obs["keep"] = A.obs["v1_cohort_keep"].astype(str) == "True"
        new_b = A.obs["label_new"].isin(NEW_BENIGN[t]).to_numpy()
        v1_b = (A.obs["v1_arm"].astype(str) == "B").to_numpy()
        per_t[t] = (A, new_b, v1_b)
        mt = mal["tissue"] == t

        # ---- C. per label, pooled and as the across-donor median -----------------------
        det, _, n, _ = _detect_by(A, A.obs["label_new"].astype(str).to_numpy(), repB)
        for lb in det.index:
            C.append({"tissue": t, "label_new": lb, "scope": "pooled",
                      "n_cells": int(n[lb]), "n_donors": int(
                          A.obs.loc[A.obs["label_new"].eq(lb), "unit"].nunique()),
                      **{f"det_{g}": det.loc[lb, g] for g in repB}})
        dd, _, nn, _ = _detect_by(
            A, (A.obs["label_new"].astype(str) + "@" + A.obs["unit"].astype(str)).to_numpy(), repB)
        dd = dd[nn.reindex(dd.index) >= 20]
        dd["label_new"] = [i.split("@")[0] for i in dd.index]
        med = dd.groupby("label_new")[repB].median()
        for lb in med.index:
            C.append({"tissue": t, "label_new": lb, "scope": "across_donor_median",
                      "n_cells": int(n.get(lb, 0)),
                      "n_donors": int((dd["label_new"] == lb).sum()),
                      **{f"det_{g}": med.loc[lb, g] for g in repB}})
        # the v1 benign arm, as its own reference row -- without it nothing in this table can
        # be compared against what the re-annotation replaced
        if v1_b.any():
            d1, _, n1, _ = _detect_by(A[v1_b], np.repeat("V1_BENIGN_CD4", int(v1_b.sum())), repB)
            C.append({"tissue": t, "label_new": "V1_BENIGN_CD4", "scope": "pooled",
                      "n_cells": int(v1_b.sum()),
                      "n_donors": int(A.obs.loc[v1_b, "unit"].nunique()),
                      **{f"det_{g}": d1.iloc[0][g] for g in repB}})
            dv, _, nv, _ = _detect_by(A[v1_b], A.obs["unit"].astype(str).to_numpy()[v1_b], repB)
            dv = dv[nv.reindex(dv.index) >= 20]
            C.append({"tissue": t, "label_new": "V1_BENIGN_CD4",
                      "scope": "across_donor_median", "n_cells": int(v1_b.sum()),
                      "n_donors": int(len(dv)),
                      **{f"det_{g}": float(dv[g].median()) for g in repB}})
        mdet = MD[mt.to_numpy()] / np.maximum(MN[mt.to_numpy()], 1)[:, None]
        C.append({"tissue": t, "label_new": "MALIGNANT_CD4_v1", "scope": "pooled",
                  "n_cells": int(MN[mt.to_numpy()].sum()),
                  "n_donors": int(mt.sum()),
                  **{f"det_{g}": MD[mt.to_numpy(), mi[g]].sum()
                     / max(MN[mt.to_numpy()].sum(), 1) for g in repB}})
        big = MN[mt.to_numpy()] >= 20
        C.append({"tissue": t, "label_new": "MALIGNANT_CD4_v1", "scope": "across_donor_median",
                  "n_cells": int(MN[mt.to_numpy()].sum()), "n_donors": int(big.sum()),
                  **{f"det_{g}": float(np.median(mdet[big, mi[g]])) for g in repB}})

        # ---- D. crosstab ---------------------------------------------------------------
        ct = (pd.crosstab([A.obs["cell_type_final"].astype(str),
                           A.obs["v1_arm"].astype(str)], A.obs["label_new"].astype(str))
              .reset_index())
        ct.insert(0, "tissue", t)
        ct = ct.rename(columns={"cell_type_final": "cell_type_final_old", "v1_arm": "v1_arm"})
        D.append(ct)

        # ---- G. co-expression ----------------------------------------------------------
        gi = {g: i for i, g in enumerate(A.var_names)}
        def _flag(g):
            v = A.X[:, gi[g]]
            return np.asarray(v.todense()).ravel() > 0
        cd4, cd8 = _flag("CD4"), _flag("CD8A") | _flag("CD8B")
        fx = _flag("FOXP3")
        for nm, m in (("new_benign", new_b), ("v1_benign", v1_b)):
            if m.sum() == 0:
                continue
            G.append({"tissue": t, "arm": nm, "n_cells": int(m.sum()),
                      "frac_CD8ABpos": cd8[m].mean(), "frac_FOXP3pos": fx[m].mean(),
                      "det_CD4_in_CD8ABpos": cd4[m & cd8].mean() if (m & cd8).any() else np.nan,
                      "det_CD4_in_CD8ABneg": cd4[m & ~cd8].mean() if (m & ~cd8).any() else np.nan})
        ti = ["blood", "skin"].index(t)
        n_, c4, c8, both, f3 = MCO[ti, 0], MCO[ti, 1], MCO[ti, 2], MCO[ti, 3], MCO[ti, 4]
        G.append({"tissue": t, "arm": "malignant_v1", "n_cells": int(n_),
                  "frac_CD8ABpos": c8 / n_, "frac_FOXP3pos": f3 / n_,
                  "det_CD4_in_CD8ABpos": both / c8 if c8 else np.nan,
                  "det_CD4_in_CD8ABneg": (c4 - both) / (n_ - c8) if n_ > c8 else np.nan})

        # ---- F. donor arms --------------------------------------------------------------
        keep = A.obs["keep"].to_numpy()
        du = A.obs["unit"].astype(str).to_numpy()
        mn = mal[mt].set_index("unit")["n"]
        meta = (A.obs.assign(unit=du)
                .groupby("unit")[["study", "tech", "call_source"]]
                .agg(lambda x: x.astype(str).value_counts().idxmax()))
        for u in sorted(set(du)):
            m = (du == u) & keep
            r = {"tissue": t, "donor": u, "study": meta.loc[u, "study"],
                 "tech": meta.loc[u, "tech"], "call_source": meta.loc[u, "call_source"],
                 "n_malignant": int(mn.get(u, 0)),
                 "n_benign_v1": int((m & v1_b).sum()),
                 "n_benign_new": int((m & new_b).sum())}
            r["eligible_v1"] = bool(r["n_malignant"] >= CONFIG["MIN_CELLS_PER_ARM"]
                                    and r["n_benign_v1"] >= CONFIG["MIN_CELLS_PER_ARM"])
            r["eligible_new"] = bool(r["n_malignant"] >= CONFIG["MIN_CELLS_PER_ARM"]
                                     and r["n_benign_new"] >= CONFIG["MIN_CELLS_PER_ARM"])
            r["cohort_keep"] = bool(keep[du == u].any())
            F.append(r)

        # ---- H. sensitivity ---------------------------------------------------------------
        key = pd.Series(du, index=A.obs_names)
        _, tot_v1, n_v1, lib_v1 = _detect_by(A[v1_b & keep], key[v1_b & keep].to_numpy(), sens)
        _, tot_nb, n_nb, lib_nb = _detect_by(A[new_b & keep], key[new_b & keep].to_numpy(), sens)
        mal_t = mal[mt].set_index("unit")
        cpm_m = pd.DataFrame(
            1e6 * MS[mt.to_numpy()][:, [mi[g] for g in sens]]
            / np.maximum(MLIB[mt.to_numpy()], 1)[:, None],
            index=mal_t.index, columns=sens)
        cpm_v1 = 1e6 * tot_v1.div(np.maximum(lib_v1, 1), axis=0)
        cpm_nb = 1e6 * tot_nb.div(np.maximum(lib_nb, 1), axis=0)

        elig_v1 = {u for u in n_v1.index if n_v1[u] >= CONFIG["MIN_CELLS_PER_ARM"]
                   and mal_t["n"].get(u, 0) >= CONFIG["MIN_CELLS_PER_ARM"]}
        elig_nb = {u for u in n_nb.index if n_nb[u] >= CONFIG["MIN_CELLS_PER_ARM"]
                   and mal_t["n"].get(u, 0) >= CONFIG["MIN_CELLS_PER_ARM"]}
        both_u = sorted(elig_v1 & elig_nb)
        src = meta["call_source"]

        def _lfc(cpm_b, us):
            if not len(us):
                return None
            return (np.log2(cpm_m.loc[us, sens].to_numpy() + 1)
                    - np.log2(cpm_b.loc[us, sens].to_numpy() + 1))
        # the only like-for-like detection comparison: both arms on the same donors
        for nm3, msk in (("V1_BENIGN_common", v1_b & keep), ("NEW_BENIGN_common", new_b & keep)):
            sub = msk & np.isin(du, both_u)
            if sub.any():
                dc, _, nc, _ = _detect_by(A[sub], du[sub], repB)
                dc = dc[nc.reindex(dc.index) >= 20]
                C.append({"tissue": t, "label_new": nm3, "scope": "across_donor_median",
                          "n_cells": int(sub.sum()), "n_donors": int(len(dc)),
                          **{f"det_{g}": float(dc[g].median()) for g in repB}})

        sets = {"a_v1_common": (cpm_v1, both_u), "b_new_common": (cpm_nb, both_u),
                "c_new_all": (cpm_nb, sorted(elig_nb))}
        for nm2, sgroup in (("tcr", "tcr"), ("auto", "auto")):
            sets[f"c_new_{nm2}"] = (cpm_nb, sorted(u for u in elig_nb
                                                   if src.get(u, "") == sgroup))
        for nm2, (cpm_b, us) in sets.items():
            Lm = _lfc(cpm_b, us)
            for j, g in enumerate(sens):
                H.append({"tissue": t, "gene": g, "set": nm2, "n_donors": len(us),
                          "median_lfc": float(np.median(Lm[:, j])) if Lm is not None else np.nan,
                          "n_pos": int((Lm[:, j] > 0).sum()) if Lm is not None else 0})

    pd.DataFrame(C).round(4).to_csv(OUT / "reannot_label_reference.csv", index=False)
    pd.concat(D, ignore_index=True).fillna(0).to_csv(OUT / "reannot_crosstab.csv", index=False)
    pd.DataFrame(F).to_csv(OUT / "reannot_donor_arms.csv", index=False)
    pd.DataFrame(G).round(4).to_csv(OUT / "reannot_coexpression.csv", index=False)
    Ht = pd.DataFrame(H)
    Ht.round(4).to_csv(OUT / "reannot_sensitivity.csv", index=False)
    log("wrote C, D, F, G, H")

    _adt_table(per_t, labels)
    _summary(per_t, pd.DataFrame(C), pd.DataFrame(F), pd.DataFrame(G), Ht, genes_json)


def _adt_table(per_t, labels):
    import h5py
    rows, dropped = [], []
    with h5py.File(CONFIG["JOINT_H5AD"], "r") as f:
        if "X_adt" not in f["obsm"]:
            (OUT / "reannot_adt.csv").write_text("# no ADT matrix in the joint object\n")
            return
        d = f["obsm"]["X_adt"]
        panel = [x.decode() if isinstance(x, bytes) else x for x in f["uns"]["adt_panel"][:]]
        ids = _read_h5ad_str(f["obs"], "cell_id")
        want = {"CD4": ["ADT_CD4"], "CD8": ["ADT_CD8A", "ADT_CD8a"],
                "TCRgd": ["ADT_TCRgd"], "CD56": ["ADT_CD56", "ADT_CD56_QA17A16"]}
        idx = pd.Index(ids)
        sel = idx.isin(labels.index)
        pos_rows = np.flatnonzero(sel)
        vals, kept_ids = [], []
        step = 100_000
        for a in range(0, len(ids), step):
            b = min(a + step, len(ids))
            k = pos_rows[(pos_rows >= a) & (pos_rows < b)]
            if not len(k):
                continue
            A = d[a:b, :]
            fin = np.isfinite(A).any(axis=1)
            kk = k[fin[k - a]]
            if not len(kk):
                continue
            sub = A[kk - a]
            la = np.log1p(np.clip(sub, 0, None))
            with np.errstate(invalid="ignore"):
                clr = la - np.nanmean(la, axis=1, keepdims=True)
            out = {}
            for nm, names in want.items():
                v = np.full(len(kk), np.nan)
                r = np.full(len(kk), np.nan)
                for s2 in names:
                    if s2 in panel:
                        j = panel.index(s2)
                        v = np.where(np.isfinite(v), v, clr[:, j])
                        r = np.where(np.isfinite(r), r, sub[:, j])
                out[nm] = v
                out[nm + "_raw"] = r
            vals.append(pd.DataFrame(out))
            kept_ids.append(idx[kk])
    if not vals:
        (OUT / "reannot_adt.csv").write_text("# no ADT cells among the re-annotated cells\n")
        return
    V = pd.concat(vals, ignore_index=True)
    V.index = pd.Index(np.concatenate([x.to_numpy() for x in kept_ids]), name="cell_id")
    V = V.join(labels[["tissue", "donor", "label_new"]])
    for (t, don), sub in V.groupby(["tissue", "donor"]):
        gates = {}
        for nm in want:
            # A marker with no counts at all still yields a CLR value -- minus the cell's mean
            # log1p -- which is identical for every such marker and tracks total protein
            # capture, so two-means "splits" it cleanly and measures library size. In skin
            # ADT_TCRgd and ADT_CD56 are exactly that. Require real counts first.
            if (sub[nm + "_raw"].fillna(0) > 0).mean() < 0.01:
                gates[nm] = None
                continue
            km = _kmeans2(sub[nm])
            gates[nm] = None if km is None else (km[0] + km[1]) / 2
        if gates["CD4"] is None or gates["CD8"] is None:
            dropped.append(f"{t}:{don}")
            continue
        for lb, g2 in sub.groupby("label_new"):
            r = {"tissue": t, "donor": don, "label_new": lb, "n_cells": len(g2)}
            for nm in want:
                r[f"frac_{nm}pos"] = ((g2[nm] > gates[nm]).mean()
                                      if gates[nm] is not None else np.nan)
                r[f"gate_{nm}"] = gates[nm]
            rows.append(r)
    out = pd.DataFrame(rows)
    out.round(4).to_csv(OUT / "reannot_adt.csv", index=False)
    (WORK / "adt_dropped_donors.txt").write_text("\n".join(dropped) + "\n")
    log(f"reannot_adt.csv: {len(out)} rows, {out['donor'].nunique() if len(out) else 0} donors; "
        f"{len(dropped)} donors dropped as not bimodal: {dropped}")


def _summary(per_t, C, F, G, H, genes_json):
    v = json.loads((OUT / "_versions.json").read_text())
    cl = pd.read_csv(OUT / "reannot_cluster_table.csv")
    tgt = ["CD8A", "CD8B", "TRDC", "KLRD1", "FOXP3"]
    L = []
    w = L.append
    w("# Re-annotation of the non-malignant T/NK cells")
    w("")
    w(f"`jobs/run_nonmal_T_reannot.py`, {time.strftime('%Y-%m-%d')}. Labels: "
      f"`data/atlas_joint/nonmal_T_labels_v1.parquet` (new file); nothing else was written. "
      f"scvi-tools {v.get('scvi-tools')}, torch {v.get('torch')}, scanpy {v.get('scanpy')}, "
      f"python {v.get('python')}; full parameters in `_versions.json` and the script's CONFIG.")
    w("")
    w("## Method actually used")
    w("")
    w(f"- Cells: every T-compartment cell (`cell_type_final` in {list(CONFIG['T_LABELS'])}) whose "
      f"`mal_cluster` is not True -- all donors, healthy controls and ineligible donors "
      f"included. No cell in the atlas has a null malignancy call, so the \"flag the nulls\" "
      f"branch never fires.")
    w(f"- Genes: a gene is dropped when a dataset of that tissue has zero counts for it while "
      f">= {CONFIG['MIN_EXPECTED_DETECT']:.0f} detections were expected from the tissue-wide "
      f"rate (this is what catches the Flex panel); TR/IG V-D-J, MT-, ribosomal and haemoglobin "
      f"genes excluded; candidates must be detected in >= {CONFIG['MIN_DETECT_HVG']:.0%} of "
      f"cells. HVG = top {CONFIG['N_HVG']} by binned dispersion of log-normalised counts, "
      f"union the marker panel.")
    w("  - " + "; ".join(
        f"{t}: {genes_json[t]['n_candidates']:,} candidates "
        f"({genes_json[t]['n_unmeasured']:,} unmeasured) -> {len(genes_json[t]['hvg'])} genes, "
        f"median detection {genes_json[t]['median_detection_hvg']:.3f}, "
        f"+{genes_json[t]['n_markers_added']} markers forced in"
        for t in ("skin", "blood")) + ".")
    w(f"- Embedding: scVI (n_latent {CONFIG['N_LATENT']}, {CONFIG['N_LAYERS']} layers, NB, "
      f"<= {CONFIG['MAX_EPOCHS']} epochs, early stopping), batch = "
      f"{CONFIG['BATCH_KEY']['skin']} in skin / {CONFIG['BATCH_KEY']['blood']} in blood. Leiden "
      f"{CONFIG['LEIDEN_RES']} on {CONFIG['N_NEIGHBORS']} neighbours; one round of "
      f"sub-clustering at {CONFIG['LEIDEN_RES_SUB']} for Mixed clusters >= "
      f"{CONFIG['MIN_SUBCLUSTER']} cells. Seed {CONFIG['SEED']}.")
    w(f"- Labels: every level is relative -- a marker's detection is divided by the highest any "
      f"cluster of >= {CONFIG['MIN_ANCHOR_CELLS']} cells in that tissue reaches, and a marker "
      f"whose best cluster stays under {CONFIG['MIN_INFORM']} detection is dropped as "
      f"uninformative there. Each lineage has a gate it must pass (NK needs a low TCR fraction "
      f"and low CD3E; MAIT needs SLC4A10/TRAV1-2; CD8 needs CD8A/CD8B; the CD4 labels need "
      f"CD4/CD40LG and a LOW CD8A/B) and then wins on score by >= {CONFIG['MARGIN']}, else "
      f"`Mixed`. SLC4A10 and TRAV1-2 are uninformative in skin, so skin gets no MAIT label. "
      f"`Suspect_malignant` overrides all: one donor holds >= {CONFIG['SUSPECT_DONOR']:.0%} of "
      f"a cluster and one TRBV >= {CONFIG['SUSPECT_TRBV']:.0%} of its TCR-bearing cells.")
    w("")
    w("## Cells per label")
    w("")
    piv = (cl.groupby(["tissue", "label_new"])["n_cells"].sum().unstack(fill_value=0))
    w("| tissue | " + " | ".join(piv.columns) + " |")
    w("|" + "---|" * (len(piv.columns) + 1))
    for t, r in piv.iterrows():
        w(f"| {t} | " + " | ".join(f"{int(x):,}" for x in r) + " |")
    w("")
    w("## Arms")
    w("")
    w("| tissue | donors eligible v1 | eligible new | median benign v1 | median benign new |")
    w("|" + "---|" * 5)
    for t, g in F.groupby("tissue"):
        g = g[g["cohort_keep"]]
        w(f"| {t} | {int(g['eligible_v1'].sum())} | {int(g['eligible_new'].sum())} | "
          f"{g.loc[g['eligible_v1'], 'n_benign_v1'].median():.0f} | "
          f"{g.loc[g['eligible_new'], 'n_benign_new'].median():.0f} |")
    w("")
    w("## Section-4 targets (reported, not asserted)")
    w("")
    hh = H.set_index(["tissue", "gene", "set"])["median_lfc"]
    w("**Target 1 cannot be met as stated**: the malignant arm is a clone, detecting these five "
      "genes in 0.0000-0.011 of its cells, so any arm of real polyclonal T cells is many times "
      "it; the informative column is `change`. **Target 2 is missed** -- every value stays below "
      "-1 -- but all ten move the right way in skin. Detection is the across-donor median with "
      "both arms on the SAME donors (eligible under both); lfc is the median per-donor log2FC, "
      "(a) v1 arm and (b) new arm on those same donors, (c) new arm on every donor eligible "
      "under it. The tcr/auto split of (c) is in `reannot_sensitivity.csv`.")
    w("")
    w("| tissue | gene | det v1 | det new | change | det malig | lfc (a) | lfc (b) | lfc (c) |")
    w("|" + "---|" * 9)
    for t in ("skin", "blood"):
        ct = C[(C.tissue == t) & (C.scope == "across_donor_median")].set_index("label_new")
        for g2 in tgt:
            v1 = ct.loc["V1_BENIGN_common", f"det_{g2}"] if "V1_BENIGN_common" in ct.index else np.nan
            nb = ct.loc["NEW_BENIGN_common", f"det_{g2}"] if "NEW_BENIGN_common" in ct.index else np.nan
            mm = ct.loc["MALIGNANT_CD4_v1", f"det_{g2}"]
            w(f"| {t} | {g2} | {v1:.4f} | {nb:.4f} | {(nb/v1 if v1 else np.nan):.2f}x | "
              f"{mm:.4f} | {hh[(t,g2,'a_v1_common')]:.2f} | {hh[(t,g2,'b_new_common')]:.2f} | "
              f"{hh[(t,g2,'c_new_all')]:.2f} |")
    w("")
    w("Positive controls, lfc (a) -> (b): " + ", ".join(
        f"{t} {g2} {hh[(t,g2,'a_v1_common')]:+.2f}->{hh[(t,g2,'b_new_common')]:+.2f}"
        for t in ("skin", "blood") for g2 in ("TOX", "CD70")) + ".")
    w("")
    w("Per-cell co-expression inside each arm, all cells (`reannot_coexpression.csv`):")
    w("")
    w("| tissue | arm | cells | CD8A/B+ | FOXP3+ |")
    w("|" + "---|" * 5)
    for _, r in G[G["arm"] != "malignant_v1"].iterrows():
        w(f"| {r['tissue']} | {r['arm']} | {int(r['n_cells']):,} | {r['frac_CD8ABpos']:.4f} | "
          f"{r['frac_FOXP3pos']:.4f} |")
    w("")
    w("## Unsure about")
    w("")
    w("- The new benign arm is CD4_conv + CD4_naive in skin and CD4_conv alone in blood; the "
      "brief's wording also admits CD4_conv alone in both, which would drop skin naive cells.")
    w("- `Suspect_malignant` is a heuristic on donor and TRBV dominance, not a malignancy call; "
      "it does not replace `mal_cluster` and those cells are simply kept out of the benign arm.")
    w("- Blood TRDC and KLRD1 detection RISES in the new arm (1.3x, 1.8x) while CD8A, CD8B and "
      "FOXP3 fall. That is composition, not contamination: the new blood arm excludes 37,829 "
      "naive CD4 cells, which are the KLRD1-lowest cells in it, so removing them raises what "
      "remains. The per-cell CD8A/B+ fraction still falls 4x, from 0.141 to 0.036.")
    w("- The CD8 and NK_ILC references in `reannot_label_reference.csv` come from this "
      "re-annotation, so they are no longer the mixtures the old labels were, but blood NK is "
      "small and its separation from CD8 rests on KLRF1/TYROBP and the absence of a TCR.")
    (OUT / "reannot_summary.md").write_text("\n".join(L) + "\n")
    log(f"wrote {OUT/'reannot_summary.md'} ({len(L)} lines)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "passA", "passB", "embed", "label", "tables"])
    ap.add_argument("--tissue", default=None)
    args = ap.parse_args()
    fn = {"prep": stage_prep, "passA": stage_passA, "passB": stage_passB,
          "embed": stage_embed, "label": stage_label, "tables": stage_tables}
    log(f"stage {args.stage}; output {OUT}")
    fn[args.stage](args)
    log(f"stage {args.stage} complete")


if __name__ == "__main__":
    main()
