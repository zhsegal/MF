#!/usr/bin/env python
"""Payload-selectivity screen (Gate 1-3) for a CD70-targeted TRUCK, MF/CTCL skin.

    python run_payload_screen.py                 # the real run
    python run_payload_screen.py --max-cells 50000   # smoke test, never a cohort result

The surface screen (nb 42/49, `tables/tgt49_*`) answered what is ON the clone. This job answers
who can RECEIVE a payload, which is the question that picks the cytokine a CD70 TRUCK secretes.

  Gate 1  Is the payload receptor depleted on the malignant clone relative to the patient's own
          reactive CD4? A receptor the clone carries feeds the tumour. Paired within-donor
          Wilcoxon on donor-level means and on donor-level detection, BH *within family*.
  Gate 2  Can the payload reach anything? The functional complex, not one chain: IL-12 needs
          IL12RB1 AND IL12RB2 in the same cell. Co-expression is counted per cell, then
          aggregated per donor and arm, and reported as donor prevalence, not a cohort mean.
  Gate 3  Is the endogenous ligand already there, and is the decoy there with it (IL18BP)?
          Pseudobulk per arm per donor across the whole skin compartment.

Same machinery as `run_target_screen_qc.py` -- same objects, same `mal_cluster` call, same
log1p CP10K on `obs.total_counts`, same >= 20-cells-per-donor-arm rule, same cohort filter
(`target_cohort`). New gene set, new arms. Two structural fixes that job does not have:

  1. MISSINGNESS. The integrated object's `var` is a 42,347-gene UNION over 14 cohorts whose own
     objects carry 15,790-32,496 genes each. A gene a cohort never measured is read back as a
     column of zeros and enters the pooled statistics as real zeros -- which is why
     `n_donors_paired` is exactly 75 for all 2,355 genes of the tgt49 tables. Here a
     (study, gene) measured mask is built from the `var` of each cohort's own source object;
     unmeasured is NaN, never 0, and never enters a test or a donor count.
  2. EMPTY COUNT ROWS. 1,083 of the 1,147 gaydosik2023 cells in the skin T object have no
     non-zero entries at all, while the same cells are intact in that cohort's source object --
     the counts were lost when the joint atlas was assembled. Their `obs` still carries
     plausible QC, so they aggregate as a donor of pure zeros. Any cell whose count row is empty
     is dropped from every arm here, which takes four CD4_HC donors below the cell floor.

Both are diagnosed and quantified in the report rather than silently corrected.

CPU only. Two passes over two matrices, one 58-gene column block each.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("TARGET_SCREEN_THREADS", "4"))
os.environ.setdefault("MPLBACKEND", "Agg")     # target_figs pulls matplotlib; no display here

NB_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NB_DIR / "helpers"))

import h5py                              # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
from scipy import stats                  # noqa: E402
from statsmodels.stats.multitest import multipletests   # noqa: E402

import cluster_malig_helpers as CM       # noqa: E402
import malignancy as MAL                 # noqa: E402
import target_cohort as TC               # noqa: E402
import target_figs as TF                 # noqa: E402
import target_screen as TS               # noqa: E402

TAB_DIR = NB_DIR / "tables"
REPORT = NB_DIR / "reports" / "51_payload_selectivity.md"
LONG_PQ = NB_DIR / "data" / "targets" / "pay51_long.parquet"
MANIFEST = NB_DIR / "data" / "targets" / "pay51_manifest.json"
PREFIX = "pay51_"

MIN_CELLS = 20            # cells per (arm, donor), as in run_target_screen_qc.py
MIN_DONORS = 5            # donors per gene for a paired test
DETECT_FLOOR = 0.10       # a donor "carries" a gene at >= 10% of cells
PSEUDO = 0.05             # pseudocount of the paired log ratio
DS_DEPTH = 2000           # binomial thinning target for the depth-matched sensitivity
SEED = 0
SMALL_ARM = 60            # sensitivity (ii): drop donor-arms below this
BLOCK = 256

CALL = "mal_cluster"
T_OBJ = NB_DIR / "data" / "atlas_joint" / "skin_T_tcr_annotated_v4.h5ad"
J_OBJ = NB_DIR / "data" / "atlas_joint" / "joint_annotated.h5ad"
J_OBS = NB_DIR / "data" / "atlas_joint" / "atlas_obs_full_v2.parquet"
SUBTYPE_CSV = NB_DIR / "data" / "atlas_joint" / "skin_myeloid_fibro_subtypes.csv"

# ------------------------------------------------------------------ gene set
FAMILY_A = ["IL12RB1", "IL12RB2", "IL18R1", "IL18RAP", "IL2RB", "IL2RG", "IL15RA", "IL7R",
            "IL10RA", "CD40", "TNFRSF9", "CD27", "IFNGR1", "IFNGR2", "IL23R", "TNFRSF1A",
            "TNFRSF1B"]
FAMILY_B = ["IL21R", "IL4R", "IL6R", "IL27RA", "IL9R", "IL10RB", "IL13RA1", "IL13RA2",
            "TNFRSF25", "TNFRSF4", "TNFRSF8", "TNFRSF18", "IL2RA"]
FAMILY_C = ["XCR1", "CXCR3", "CCR5", "CCR4", "CCR7", "S1PR1", "CX3CR1",
            "XCL1", "XCL2", "CXCL9", "CXCL10", "CXCL11", "CCL19", "CCL21",
            "IL12A", "IL12B", "IL18", "IL18BP", "IL15", "FLT3LG", "CD70", "TNFSF9", "TNFSF4"]
AUDIT = ["CD2", "CD3E", "PTPRC", "CD7", "GAPDH"]      # never pooled into a family
FAMILIES = {"A_permitted": FAMILY_A, "B_disqualified": FAMILY_B, "C_axis": FAMILY_C,
            "audit": AUDIT}
GENES = list(dict.fromkeys(FAMILY_A + FAMILY_B + FAMILY_C + AUDIT))
FAMILY_OF = {g: f for f, gs in FAMILIES.items() for g in gs}

C_RECEPTORS = ["XCR1", "CXCR3", "CCR5", "CCR4", "CCR7", "S1PR1", "CX3CR1"]
C_LIGANDS = [g for g in FAMILY_C if g not in C_RECEPTORS]

# Gate 2: the functional complex. A payload reaches a cell only if every chain is in THAT cell.
COMPLEX = {
    "IL-12": ("IL12RB1", "IL12RB2"),
    "IL-18": ("IL18R1", "IL18RAP"),
    "IL-15": ("IL2RB", "IL2RG"),
    "XCL1": ("XCR1",),
    "CXCL9/10": ("CXCR3",),
}
REACH_THRESHOLDS = (0.01, 0.05, 0.10)

# ------------------------------------------------------------------ arms
MALIG, REACT, HC = "CD4_malignant", "CD4_reactive", "CD4_HC"
T_ARMS = [MALIG, REACT, HC, "CD8", "CD4_Treg"]
NK_LABELS = ("NK", "ILC1_NK")                      # obs.cell_type; overlaps the T arms
MONO_MAC = ("Mono_CD14", "Mono_CD16", "Mac_FOLR2", "Mac_infl", "Mac_SPP1_TREM2")
MYE_ARMS = {"Mono_Mac": MONO_MAC, "cDC1": ("cDC1",), "cDC2": ("cDC2",), "pDC": ("pDC",)}
BROAD_ARMS = ("Fibroblast", "Keratinocyte", "Vascular")     # cell_type_final, Gate 3 context
ARM_ORDER = T_ARMS + ["NK"] + list(MYE_ARMS) + list(BROAD_ARMS)

# ------------------------------------------------------------------ per-cohort gene universes
# The donor key carries its cohort (`D14__SC320` -> `D14`), and cohort maps 1:1 onto `study`.
COHORT_SRC = {
    "D1": "data/D1_chennareddy2025/processed/standardized.h5ad",
    "D3": "data/D3_brunner2024/processed/standardized.h5ad",
    "D5": "data/D5_rindler2021frontimm_multitissue/processed/standardized.h5ad",
    "D6": "data/D6_gaydosik2019_skin/processed/standardized.h5ad",
    "D7": "data/D7_rindler2021mc_skin/processed/standardized.h5ad",
    "D8": "data/D8_alkon2024_parapsoriasis/processed/standardized.h5ad",
    "D9": "data/D9_lyp_vs_ctcl_skin/processed/standardized.h5ad",
    "D10": "data/D10_jonak2021_discordant/processed/standardized.h5ad",
    "D11": "data/D11_brentuximab_skin/processed/standardized.h5ad",
    "D13": "data/D13_gaydosik2022_skin_blood/processed/standardized.h5ad",
    "D14": "data/D14_gaydosik2023_skin/processed/standardized.h5ad",
    "D15": "data/D15_il4ra_blockade_skin/processed/standardized.h5ad",
    "B5": "data/B5_buus2025_blood_skin/processed/standardized.h5ad",
    "H": "data/H_herrera2021/processed/standardized.h5ad",
    "Li2024_atlas": "data/CTCL_all_final_portal_tags.h5ad",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-cells", type=int, default=None,
                   help="use only the first N cells of each object (smoke test)")
    p.add_argument("--skip-joint", action="store_true",
                   help="T arms only -- skips the 55 GB pass over joint_annotated.h5ad")
    return p.parse_args()


# ------------------------------------------------------------------ matrix I/O
def row_nnz(g, n_obs):
    """Non-zero entries per cell, from the CSR indptr alone. No data is read."""
    ip = g["indptr"][:n_obs + 1].astype(np.int64)
    return np.diff(ip)


def read_block(g, cols, n_obs, keep=None, step=50_000):
    """Columns `cols` of a CSR h5ad matrix group, dense (cells x len(cols)) float32.

    CSR is row-major, so a column slice walks every row. Row chunks with no wanted cell are
    skipped on the indptr alone, which is what makes the 2.2M-cell joint object affordable.
    """
    want = np.asarray(cols)
    lookup = -np.ones(int(g.attrs["shape"][1]), dtype=np.int64)
    lookup[want] = np.arange(len(want))
    out = np.zeros((n_obs, len(want)), dtype=np.float32)
    indptr = g["indptr"]
    nchunk = (n_obs + step - 1) // step
    for ci, a in enumerate(range(0, n_obs, step), 1):
        b = min(a + step, n_obs)
        if keep is not None and not keep[a:b].any():
            continue
        ip = indptr[a:b + 1].astype(np.int64)
        lo, hi = int(ip[0]), int(ip[-1])
        if hi == lo:
            continue
        idx = g["indices"][lo:hi]
        dat = g["data"][lo:hi]
        pos = lookup[idx]
        sel = pos >= 0
        if sel.any():
            rows = np.repeat(np.arange(a, b), np.diff(ip))
            out[rows[sel], pos[sel]] = dat[sel].astype(np.float32)
        if ci % 10 == 0 or ci == nchunk:
            log(f"    rows {b:,}/{n_obs:,}")
    return out


def thin(X, p, rng):
    """Binomial thinning of a dense count block to a common depth. Only non-zeros are drawn."""
    out = np.zeros_like(X)
    nz = np.nonzero(X)
    if len(nz[0]):
        out[nz] = rng.binomial(X[nz].astype(np.int64), np.clip(p[nz[0]], 0, 1))
    return out


# ------------------------------------------------------------------ aggregation
def aggregate(X, L, groups, names, lib, ngen, Xds=None, ds_ok=None, cplx=None):
    """(arm, donor) x gene statistics, plus the per-group Gate-2 reach.

    `X` raw counts, `L` log1p CP10K of the same cells. Everything is computed down the cell axis.
    """
    long_rows, reach_rows = [], []
    for (arm, donor), idx in groups.items():
        sub = X[idx]
        pos = sub > 0
        rec = {
            "arm": arm, "donor": donor, "gene": names, "n_cells": len(idx),
            "pct_expr": pos.mean(0),
            "mean_all": L[idx].mean(0),
            "sd_all": L[idx].std(0, ddof=1) if len(idx) > 1 else np.zeros(len(names)),
            "median_total_counts": float(np.median(lib[idx])),
            "median_n_genes": float(np.median(ngen[idx])),
        }
        if Xds is not None:
            d = idx[ds_ok[idx]]
            rec["n_cells_ds"] = len(d)
            rec["pct_expr_ds"] = ((Xds[d] > 0).mean(0) if len(d)
                                  else np.full(len(names), np.nan))
        long_rows.append(pd.DataFrame(rec))
        if cplx is not None:
            r = {"arm": arm, "donor": donor, "n_cells": len(idx)}
            for k, v in cplx.items():
                r[k] = float(v[idx].mean())
            reach_rows.append(r)
    return long_rows, reach_rows


def complex_masks(X, names):
    """Per cell: does it carry the complete receptor complex for each payload?"""
    pos = pd.Index(names)
    out = {}
    for payload, chains in COMPLEX.items():
        if not all(c in pos for c in chains):
            continue
        m = np.ones(X.shape[0], dtype=bool)
        for c in chains:
            m &= X[:, pos.get_loc(c)] > 0
        out[payload] = m
    return out


# ------------------------------------------------------------------ reporting
SECTIONS: list[dict] = []


def save(df, name, index=False):
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    p = TAB_DIR / f"{PREFIX}{name}.csv"
    df.to_csv(p, index=index)
    return str(p.relative_to(NB_DIR))


def _tidy(table):
    """A row index that carries no information must not print as a column of row numbers."""
    if not isinstance(table, pd.DataFrame):
        return table
    if table.index.name is None and not isinstance(table.index, pd.RangeIndex):
        table = table.reset_index(drop=True)
    return table


def report(name, what, how, table, file=None, note=None, max_rows=None):
    table = _tidy(table)
    SECTIONS.append({"name": name, "what": what, "how": how, "table": table,
                     "file": file, "note": note, "max_rows": max_rows})
    print(f"\n=== {name} ===")
    print(f"WHAT: {what}")
    print(f"HOW : {how}")
    if file:
        print(f"FILE: {file}")
    if table is not None:
        print(TF._as_csv(table, max_rows))
    if note:
        print(note)


def write_report(title, context):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    L = [f"# {title}", "",
         f"_Written by `jobs/run_payload_screen.py` on {time.strftime('%Y-%m-%d %H:%M')}._", "",
         "Every table below is a CSV block with a header row, printed at 3 significant figures. "
         "`q` is BH-adjusted unless the **How** line says otherwise. The on-disk table backing "
         "each block is named above it.", "",
         "## Context", "", context.strip(), ""]
    for s in SECTIONS:
        L += [f"## {s['name']}", "", f"**What:** {s['what']}", "", f"**How:** {s['how']}", ""]
        if s.get("file"):
            L += [f"**Table on disk:** `{s['file']}`", ""]
        if s.get("table") is not None:
            L += ["```csv", TF._as_csv(s["table"], s.get("max_rows")), "```", ""]
        if s.get("note"):
            L += [s["note"].strip(), ""]
    REPORT.write_text("\n".join(L))
    log(f"wrote {REPORT.relative_to(NB_DIR)}  ({REPORT.stat().st_size / 1024:.0f} KB)")


# ------------------------------------------------------------------ statistics
def paired_counts(long, genes, value):
    """Per gene: how many paired donors move up, down or not at all. Donor-level."""
    w = (long[long["arm"].isin([MALIG, REACT]) & long["gene"].isin(genes)]
         .pivot_table(index=["gene", "donor"], columns="arm", values=value))
    if MALIG not in w.columns or REACT not in w.columns:
        return pd.DataFrame(columns=["gene", "n_up", "n_down", "n_tie"])
    w = w.dropna(subset=[MALIG, REACT])
    d = (w[MALIG] - w[REACT]).groupby(level="gene")
    return pd.DataFrame({"n_up": d.apply(lambda s: int((s > 0).sum())),
                         "n_down": d.apply(lambda s: int((s < 0).sum())),
                         "n_tie": d.apply(lambda s: int((s == 0).sum()))}).reset_index()


def gate1(long, genes, label):
    """Paired malignant-vs-reactive CD4 on expression AND on detection, BH within this set."""
    sub = long[long["gene"].isin(genes)]
    exp = TS.paired_selectivity(sub, MALIG, REACT, value="mean_all",
                                min_donors=MIN_DONORS, pseudo=PSEUDO)
    det = TS.paired_selectivity(sub, MALIG, REACT, value="pct_expr",
                                min_donors=MIN_DONORS, pseudo=PSEUDO)
    det = det.rename(columns={"log2fc_paired": "log2fc_detect", "p": "p_detect",
                              "q": "q_detect", "frac_donors_up": "frac_donors_up_detect",
                              "n_donors_paired": "n_donors_paired_detect"})
    out = exp.merge(det, on="gene", how="outer")
    out = out.merge(paired_counts(sub, genes, "mean_all"), on="gene", how="left")
    out.insert(1, "family", label)
    return out


def ols_stage(lfc, donors, stage_col="clin_stage_class"):
    """Per gene: donor-level log2FC ~ stage + study dummies, by least squares.

    The stage coefficient is reported with study held fixed. It is NOT identified independently
    of study when a stratum is dominated by one cohort, which is the case here; the crosstab is
    printed beside it so the reader can see that.
    """
    import numpy.linalg as la
    d = donors.loc[donors[stage_col].isin(["early", "advanced"])]
    rows = []
    for gene in lfc.columns:
        y = lfc[gene].reindex(d.index)
        ok = y.notna()
        if ok.sum() < 8:
            continue
        yy = y[ok].to_numpy(float)
        dd = d.loc[ok]
        early = dd[stage_col].eq("early").to_numpy(float)
        dummies = pd.get_dummies(dd["study"], drop_first=True).to_numpy(float)
        Xd = np.column_stack([np.ones(len(yy)), early, dummies])
        if la.matrix_rank(Xd) < Xd.shape[1]:
            Xd = np.column_stack([np.ones(len(yy)), early])        # study absorbed by stage
            absorbed = True
        else:
            absorbed = False
        beta, *_ = la.lstsq(Xd, yy, rcond=None)
        resid = yy - Xd @ beta
        dof = len(yy) - Xd.shape[1]
        if dof <= 0:
            continue
        s2 = float(resid @ resid) / dof
        cov = s2 * la.pinv(Xd.T @ Xd)
        se = float(np.sqrt(max(cov[1, 1], 0)))
        t = beta[1] / se if se > 0 else np.nan
        rows.append({"gene": gene, "family": FAMILY_OF.get(gene, ""),
                     "beta_early_vs_advanced": float(beta[1]), "se": se, "t": float(t),
                     "p": float(2 * stats.t.sf(abs(t), dof)) if np.isfinite(t) else np.nan,
                     "n_early": int(early.sum()), "n_advanced": int((1 - early).sum()),
                     "study_absorbed": absorbed})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q"] = multipletests(out["p"].fillna(1), method="fdr_bh")[1]
    return out


# ------------------------------------------------------------------ passes
def pass_t(args, genes_all, gi, rng, min_cells=None):
    """Skin T/NK object: the CD4/CD8/Treg arms exactly as run_target_screen_qc.py builds them.

    `min_cells` defaults to MIN_CELLS; the donor-long export passes 0 so the sub-threshold
    donor-arms can be written out and flagged instead of dropped. Aggregation is per group,
    so the groups at or above the floor are bit-identical either way.
    """
    min_cells = MIN_CELLS if min_cells is None else min_cells
    long_rows, reach_rows, diag = [], [], {}
    with h5py.File(T_OBJ, "r") as h5:
        obs = CM.read_obs(h5, cols=["donor", "sample_id", "study", "disease", "cell_type_T",
                                    "cell_type", "has_tcr", "total_counts", "n_genes",
                                    "doublet_score"])
        if args.max_cells:
            obs = obs.iloc[:args.max_cells]
        n_obs = len(obs)
        g = h5["layers"]["raw_counts"]
        log(f"{T_OBJ.name}: {n_obs:,} cells x {len(genes_all):,} genes")

        nnz = row_nnz(g, n_obs)
        empty = nnz == 0
        diag["empty_cells"] = int(empty.sum())
        by_study = (pd.DataFrame({"study": obs["study"].astype(str).to_numpy(),
                                  "donor": obs["donor"].astype(str).to_numpy(),
                                  "empty": empty})
                    .groupby(["study", "donor"])["empty"].agg(["sum", "size"]))
        by_study = by_study[by_study["sum"] > 0].rename(
            columns={"sum": "cells_with_empty_count_row", "size": "cells_in_object"})
        diag["empty_table"] = by_study.reset_index()
        log(f"cells with an empty count row: {int(empty.sum()):,} "
            f"({empty.mean():.3%}) in {by_study.index.get_level_values(0).nunique()} study(s)")

        mal = MAL.load_malignancy(obs.index,
                                  has_tcr=obs["has_tcr"].astype(bool).to_numpy(), verbose=True)
        lvl = pd.Series(np.asarray(MAL.cd4_level(obs["cell_type_T"], mal, call=CALL)).astype(str),
                        index=obs.index)
        don_tbl = TC.skin_donor_table()
        donor = obs["donor"].astype(str)
        is_hc = donor.map(don_tbl["is_hc"]).fillna(False).to_numpy()
        import lineage as LIN
        is_cd4 = obs["cell_type_T"].astype(str).isin(LIN.CD4_ANY).to_numpy()
        lvl[is_cd4 & is_hc] = HC
        arm = lvl.to_numpy()

        ct = obs["cell_type"].astype(str).to_numpy()
        is_nk = np.isin(ct, NK_LABELS)
        diag["nk_overlap"] = (pd.crosstab(pd.Series(arm[is_nk], name="arm_of_NK_cells"),
                                          pd.Series(obs["study"].astype(str).to_numpy()[is_nk],
                                                    name="study"))
                              .reset_index())

        lib = pd.to_numeric(obs["total_counts"], errors="coerce").to_numpy(float)
        ngen = pd.to_numeric(obs["n_genes"], errors="coerce").to_numpy(float)
        lib[~np.isfinite(lib) | (lib <= 0)] = 1.0

        ok = ~empty
        groups = {}
        key = pd.MultiIndex.from_arrays([arm, donor.to_numpy()], names=["arm", "donor"])
        for k, v in pd.Series(np.arange(n_obs)[ok], index=key[ok]).groupby(level=[0, 1]):
            if k[0] in T_ARMS and len(v) >= min_cells:
                groups[k] = v.to_numpy()
        nk_key = pd.Series(np.arange(n_obs)[ok & is_nk], index=donor.to_numpy()[ok & is_nk])
        for d, v in nk_key.groupby(level=0):
            if len(v) >= min_cells:
                groups[("NK", d)] = v.to_numpy()
        log(f"{len(groups):,} (arm, donor) groups >= {min_cells} cells in the T object")

        log("  reading the gene block ...")
        X = read_block(g, gi, n_obs)
        L = np.log1p(X / lib[:, None] * 1e4)
        ds_ok = lib >= DS_DEPTH
        Xds = thin(X, np.where(ds_ok, DS_DEPTH / np.maximum(lib, 1), 0.0), rng)
        cplx = complex_masks(X, GENES)
        lr, rr = aggregate(X, L, groups, GENES, lib, ngen, Xds=Xds, ds_ok=ds_ok, cplx=cplx)
        long_rows += lr
        reach_rows += rr
        diag["frac_cells_ds"] = float(ds_ok[ok].mean())
        del X, L, Xds
    return long_rows, reach_rows, diag


def pass_joint(args, genes_all, gi, min_cells=None):
    """Myeloid, fibroblast and the two context compartments, from the 42,347-gene joint object.

    `min_cells` as in pass_t.
    """
    min_cells = MIN_CELLS if min_cells is None else min_cells
    long_rows, reach_rows, diag = [], [], {}
    ob = pd.read_parquet(J_OBS, columns=["cell_id", "compartment", "cell_type_final", "donor",
                                         "study", "total_counts", "n_genes"])
    with h5py.File(J_OBJ, "r") as h5:
        idx_key = h5["obs"].attrs["_index"]
        cid = CM._read_elem(h5["obs"][idx_key])
        cid = pd.Index(cid).astype(str)
        g = h5["layers"]["raw_counts"]
        n_obs = int(g.attrs["shape"][0])
        assert len(cid) == n_obs == len(ob), (len(cid), n_obs, len(ob))
        same = bool((cid.to_numpy() == ob["cell_id"].astype(str).to_numpy()).all())
        if not same:
            log("obs order differs from the parquet -- reindexing the parquet onto obs")
            ob = ob.set_index("cell_id").reindex(cid).reset_index()
            assert ob["donor"].notna().all(), "parquet does not cover every joint cell"
        diag["obs_order_matches_parquet"] = same
        if args.max_cells:
            ob = ob.iloc[:args.max_cells]
            n_obs = len(ob)
        log(f"{J_OBJ.name}: {n_obs:,} cells, parquet aligned (identical order: {same})")

        sub = pd.read_csv(SUBTYPE_CSV, usecols=["cell_id", "lineage", "subtype_fine"])
        fine = ob["cell_id"].astype(str).map(sub.set_index("cell_id")["subtype_fine"])
        diag["sidecar_joined"] = int(fine.notna().sum())

        skin = ob["compartment"].astype(str).eq("Skin").to_numpy()
        ctf = ob["cell_type_final"].astype(str).to_numpy()
        fine = fine.fillna("").to_numpy()
        arm = np.full(n_obs, "", dtype=object)
        for a, labels in MYE_ARMS.items():
            arm[skin & np.isin(fine, labels)] = a
        for a in BROAD_ARMS:
            arm[skin & (ctf == a) & (arm == "")] = a
        want = arm != ""
        log(f"wanted cells: {int(want.sum()):,}  "
            + str(pd.Series(arm[want]).value_counts().to_dict()))

        nnz = row_nnz(g, n_obs)
        empty = nnz == 0
        diag["empty_cells_joint"] = int((empty & want).sum())
        ok = want & ~empty

        lib = pd.to_numeric(ob["total_counts"], errors="coerce").to_numpy(float)
        ngen = pd.to_numeric(ob["n_genes"], errors="coerce").to_numpy(float)
        lib[~np.isfinite(lib) | (lib <= 0)] = 1.0

        donor = ob["donor"].astype(str).to_numpy()
        key = pd.MultiIndex.from_arrays([arm[ok].astype(str), donor[ok]], names=["arm", "donor"])
        groups = {k: v.to_numpy() for k, v in
                  pd.Series(np.arange(n_obs)[ok], index=key).groupby(level=[0, 1])}
        groups = {k: v for k, v in groups.items() if len(v) >= min_cells}
        log(f"{len(groups):,} (arm, donor) groups >= {min_cells} cells in the joint object")

        log("  reading the gene block (this is the 55 GB pass) ...")
        X = read_block(g, gi, n_obs, keep=ok)
        L = np.log1p(X / lib[:, None] * 1e4)
        cplx = complex_masks(X, GENES)
        lr, rr = aggregate(X, L, groups, GENES, lib, ngen, cplx=cplx)
        long_rows += lr
        reach_rows += rr
        del X, L
    return long_rows, reach_rows, diag


# ------------------------------------------------------------------ coverage mask
def coverage_mask(studies_by_cohort):
    """measured[(study, gene)] from the `var` of each cohort's own source object."""
    rows = []
    for cohort, study in studies_by_cohort.items():
        p = NB_DIR / COHORT_SRC[cohort]
        with h5py.File(p, "r") as h5:
            v = h5["var"]
            arr = v[v.attrs["_index"]][:]
            have = {x.decode() if isinstance(x, bytes) else str(x) for x in arr}
        for gene in GENES:
            rows.append({"study": study, "cohort": cohort, "source": str(p.relative_to(NB_DIR)),
                         "n_genes_in_source": len(have), "gene": gene,
                         "measured": gene in have})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main
def main():
    args = parse_args()
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    CM.use_tissue("skin")
    MAL.use_tissue("skin")
    assert MAL.CLUSTER_PQ.name.startswith("skin_"), MAL.CLUSTER_PQ

    with h5py.File(T_OBJ, "r") as h5:
        genes_all = pd.Index(CM._read_elem(h5["var"][h5["var"].attrs["_index"]])).astype(str)
    missing = [g for g in GENES if g not in set(genes_all)]
    assert not missing, f"genes absent from the object var: {missing}"
    gi = genes_all.get_indexer(GENES)
    assert (gi >= 0).all()
    log(f"gene set: {len(GENES)} genes "
        + ", ".join(f"{f} {len(gs)}" for f, gs in FAMILIES.items()) + "; none absent")

    long_rows, reach_rows, diag = pass_t(args, genes_all, gi, rng)
    if not args.skip_joint:
        lr, rr, d2 = pass_joint(args, genes_all, gi)
        long_rows += lr
        reach_rows += rr
        diag.update(d2)

    long_all = pd.concat(long_rows, ignore_index=True)
    reach_all = pd.DataFrame(reach_rows)
    LONG_PQ.parent.mkdir(parents=True, exist_ok=True)
    long_all.to_parquet(LONG_PQ, index=False)
    log(f"wrote {LONG_PQ.relative_to(NB_DIR)}  {long_all.shape}")

    # ---------------- cohort
    don = TC.skin_donor_table()
    with h5py.File(T_OBJ, "r") as h5:
        o = CM.read_obs(h5, cols=["donor", "study"])
    o = o.astype({"donor": str, "study": str}).drop_duplicates("donor")
    d2s = o.set_index("donor")["study"]
    long_all["study"] = long_all["donor"].map(d2s)
    assert long_all["study"].notna().all(), "a donor key has no study"
    # The donor key carries its cohort; the mask is keyed on study, so the two must be 1:1.
    long_all["cohort"] = long_all["donor"].str.split("__").str[0]
    chk = long_all.groupby("cohort")["study"].nunique()
    assert (chk == 1).all(), f"cohort prefix is not 1:1 with study: {chk[chk > 1].to_dict()}"
    c2s = long_all.drop_duplicates("cohort").set_index("cohort")["study"].to_dict()

    long_c = TC.filter_long(long_all, don)
    kept = set(long_c["donor"])
    log(f"cohort: {len(don):,} keys -> {int(don['cohort_keep'].sum())} independent "
        f"({don.loc[don['cohort_keep'], 'person'].nunique()} people); "
        f"{len(kept)} of them carry an arm here")

    # ---------------- coverage mask
    cov = coverage_mask({c: s for c, s in c2s.items() if c in COHORT_SRC})
    missing_cohorts = sorted(set(c2s) - set(COHORT_SRC))
    assert not missing_cohorts, f"no source object registered for {missing_cohorts}"
    meas = cov.set_index(["study", "gene"])["measured"]
    key = pd.MultiIndex.from_arrays([long_c["study"], long_c["gene"]])
    is_meas = meas.reindex(key).to_numpy()
    # The mask must predict the data: an unmeasured (study, gene) has to be observed all-zero.
    viol = int(((~is_meas) & (long_c["pct_expr"].to_numpy() > 0)).sum())
    n_masked = int((~is_meas).sum())
    log(f"coverage mask: {n_masked:,} of {len(long_c):,} (donor-arm, gene) cells unmeasured; "
        f"{viol} contradict the mask")
    assert viol == 0, "a gene the source object does not carry has non-zero counts"
    for c in ("pct_expr", "mean_all", "sd_all", "pct_expr_ds"):
        if c in long_c:
            long_c.loc[~is_meas, c] = np.nan
    long_c = long_c.dropna(subset=["mean_all"])

    n_don_meas = (cov.merge(long_c[["study", "donor"]].drop_duplicates(), on="study")
                  .groupby(["gene", "study"])
                  .agg(n_donors_measured=("donor", "nunique"), measured=("measured", "first"))
                  .reset_index())
    n_don_meas.loc[~n_don_meas["measured"], "n_donors_measured"] = 0
    f_cov = save(n_don_meas, "gene_coverage")

    # ---------------- §1 input and cohort
    paired_donors = sorted(set(long_c.loc[long_c["arm"].eq(MALIG), "donor"])
                           & set(long_c.loc[long_c["arm"].eq(REACT), "donor"]))
    arms = (long_c.drop_duplicates(["arm", "donor"])
            .groupby("arm")
            .agg(n_donors=("donor", "nunique"), n_cells=("n_cells", "sum"),
                 n_studies=("study", "nunique"),
                 median_total_counts=("median_total_counts", "median"),
                 median_n_genes=("median_n_genes", "median"))
            .reindex([a for a in ARM_ORDER if a in set(long_c["arm"])]).reset_index())
    f_arms = save(arms, "arms")

    manifest = {
        "written": time.strftime("%Y-%m-%d %H:%M"),
        "builder": "run_payload_screen.py",
        "object_T": str(T_OBJ.relative_to(NB_DIR)),
        "object_nonT": "-" if args.skip_joint else str(J_OBJ.relative_to(NB_DIR)),
        "layer": "raw_counts",
        "malignancy_call": CALL,
        "normalisation": "log1p(counts / obs.total_counts * 1e4), per cell",
        "min_cells_per_donor_arm": MIN_CELLS,
        "min_donors_per_test": MIN_DONORS,
        "pseudocount_log2fc": PSEUDO,
        "detect_floor": DETECT_FLOOR,
        "n_genes": len(GENES),
        "donor_keys_in_object": int(len(don)),
        "independent_donor_keys": int(don["cohort_keep"].sum()),
        "people": int(don.loc[don["cohort_keep"], "person"].nunique()),
        "paired_donors": len(paired_donors),
        "hc_donors": int(long_c.loc[long_c["arm"].eq(HC), "donor"].nunique()),
        "empty_count_rows_dropped_T": diag.get("empty_cells", 0),
        "empty_count_rows_dropped_joint": diag.get("empty_cells_joint", 0),
        "unmeasured_cells_masked": n_masked,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")

    report("§1a Run manifest",
           "The object, the call, the normalisation and the thresholds every number below rests on.",
           "Written by this job to `data/targets/pay51_manifest.json`; keys are literal run "
           "parameters, not derived quantities.",
           manifest)

    report("§1b Arms",
           "One row per arm: donors, cells, studies, and the sequencing depth of its cells.",
           "Donor-arms with >= 20 cells after the empty-row rule and the cohort filter. "
           "median_total_counts and median_n_genes are the median over donor-arms of the "
           "per-donor median UMI and genes per cell, from `obs`. "
           "Depth is NOT matched between arms in Gate 1 or 2: see §6d.",
           arms, file=f_arms,
           note="The malignant arm is sequenced deeper than the reactive arm, so every "
                "DETECTION comparison (pct_expr) is depth-confounded in the direction that "
                "favours the malignant arm. §6d repeats Gate 1 detection after binomial "
                "thinning to a common 2,000 UMI.")

    report("§1c Empty count rows",
           "Cells that exist in the integrated object with QC metadata but hold no counts at all.",
           "Non-zero entries per cell read from the CSR indptr of `layers/raw_counts`. "
           "A cell with zero non-zeros is dropped from every arm here. Their source cohort "
           "object has the same cells intact, so the counts were lost when the atlas was "
           "assembled, not at sequencing.",
           diag.get("empty_table"), max_rows=40,
           note=f"{diag.get('empty_cells', 0):,} cells dropped in the T object, "
                f"{diag.get('empty_cells_joint', 0):,} in the joint object. "
                "This removes four CD4_HC donors (SC320, SC348, SC353, SC354) entirely and "
                "trims SC405; the corrected healthy n is in §1a (`hc_donors`). The four were "
                "NOT repaired from the source object -- that is a second data path for one "
                "study and was left for a separate job.")

    report("§1d NK arm overlap",
           "Where the NK-annotated cells sit in the T arms. The NK arm is not disjoint from them.",
           "`obs.cell_type in (NK, ILC1_NK)` crossed with the arm the same cells already carry. "
           "The CD4/CD8/Treg arms are left exactly as `run_target_screen_qc.py` defines them, so "
           "Gate 1 stays comparable with tgt49; the NK arm is additional and overlapping.",
           diag.get("nk_overlap"),
           note="Every NK-annotated cell in this atlas comes from one study (li2024), so the NK "
                "arm is a single-cohort observation and carries no between-study replication.")

    # ---------------- §2 gene universe and coverage
    fam_tab = pd.DataFrame([{"family": f, "n_genes": len(gs),
                             "genes": " ".join(gs),
                             "absent_from_object": ""} for f, gs in FAMILIES.items()])
    report("§2a Gene families",
           "The genes tested, by family. Families are corrected separately from each other and "
           "from the 2,115-gene surface screen.",
           "Family membership is fixed a priori, not data-driven. `audit` genes are carried for "
           "diagnostics (CD2/CD3E/PTPRC for the empty-row check, GAPDH/CD7 for the coverage "
           "check) and are never pooled into a family or into a BH correction with one.",
           fam_tab,
           note="No gene of the set is absent from the object's 42,347-gene var.")

    cov_wide = (n_don_meas.pivot_table(index="gene", columns="study",
                                       values="n_donors_measured", aggfunc="first")
                .reindex(GENES))
    unmeas = n_don_meas[~n_don_meas["measured"]]
    report("§2b Per-study coverage",
           "For each gene and study, the number of cohort donors whose source object actually "
           "measured that gene. 0 means the gene is not in that cohort's gene list at all.",
           "The mask is built from the `var` index of each cohort's own source object "
           "(`data/<cohort>/processed/standardized.h5ad`, and "
           "`data/CTCL_all_final_portal_tags.h5ad` for li2024), joined on the cohort prefix of "
           "the donor key. Unmeasured (study, gene) pairs are set to NaN and dropped before any "
           "test; they never enter a donor count as a zero. The mask is then checked against the "
           "data: an unmeasured pair must be observed all-zero, and 0 of "
           f"{n_masked if n_masked else 0} contradicted it.",
           cov_wide, file=f_cov,
           note=("Every gene of this payload set is measured in every cohort that contributes a "
                 "skin donor, so the mask removes nothing here: "
                 f"{len(unmeas)} (study, gene) pairs are unmeasured. That is the result, not a "
                 "reason the check was unnecessary -- in the 2,115-gene surface screen the same "
                 "mask would remove 342 genes from li2024 alone (31 of the 75 paired donors), "
                 "which is why `n_donors_paired` is a constant 75 in every tgt49 table. "
                 "See §7."))

    # ---------------- §3 Gate 1
    g1 = pd.concat([gate1(long_c, gs, f) for f, gs in FAMILIES.items()], ignore_index=True)
    prev = {}
    for a, tag in ((MALIG, "malig"), (REACT, "react"), (HC, "hc")):
        p = TS.prevalence(long_c, a, detect_floor=DETECT_FLOOR)[
            ["gene", "median_pct_expr", "frac_donors_positive", "n_donors"]]
        prev[tag] = p.rename(columns={c: f"{tag}_{c}" for c in p.columns if c != "gene"})
    for tag in prev:
        g1 = g1.merge(prev[tag], on="gene", how="left")
    g1 = g1.sort_values(["family", "log2fc_detect"], ascending=[True, False])
    f_g1 = save(g1, "cd4_paired")

    PRINT1 = ["gene", "family", "n_donors_paired", "log2fc_paired", "q", "frac_donors_up",
              "n_up", "n_down", "log2fc_detect", "q_detect", "malig_median_pct_expr",
              "react_median_pct_expr", "hc_median_pct_expr", "malig_frac_donors_positive"]
    how1 = ("Paired Wilcoxon signed-rank over donors that carry both arms, on the donor-level "
            "value; donor-level only, no cell is a unit of analysis. `log2fc_paired` is the "
            "median over donors of log2((mean_all malignant + 0.05)/(mean_all reactive + 0.05)) "
            "-- a ratio of log-scale means, not a raw fold change. `log2fc_detect` is the same "
            "construction on `pct_expr` (fraction of cells with >= 1 count). BH within family "
            "over the genes of that family only. Positive = higher on the malignant clone.")
    for fam, lab in (("A_permitted", "§3a Gate 1 -- permitted payload receptors"),
                     ("B_disqualified", "§3b Gate 1 -- disqualified payload receptors"),
                     ("C_axis", "§3c Gate 1 -- chemoattractant axis"),
                     ("audit", "§3d Gate 1 -- audit genes (not a family)")):
        sub = g1[g1["family"].eq(fam)][PRINT1]
        what = {"A_permitted": "Receptors a payload is permitted to engage: the hypothesis is "
                               "that they are DEPLETED on the malignant clone, so the payload "
                               "acts on the reactive compartment and not on the tumour.",
                "B_disqualified": "Receptors that disqualify a payload: the hypothesis is that "
                                  "they are ENRICHED on the clone, so the payload would feed it. "
                                  "They are kept in and reported -- that is the result.",
                "C_axis": "The chemoattractant axis: receptors that decide who is recruited, and "
                          "ligands that decide whether the signal is already present.",
                "audit": "Diagnostic genes. CD2/CD3E/PTPRC must be high in every T arm (they are "
                         "what showed the empty-row defect); GAPDH and CD7 check the coverage "
                         "mask and the tgt49 loss-marker result."}[fam]
        report(lab, what, how1, sub, file=f_g1)

    arms_here = [a for a in ARM_ORDER if a in set(long_c["arm"])]
    pct_index = []
    for a in arms_here:
        w = (long_c[long_c["arm"].eq(a)]
             .pivot_table(index="donor", columns="gene", values="pct_expr")
             .reindex(columns=[g for g in GENES]))
        w.index.name = "donor"
        pct_index.append({"arm": a, "n_donors": int(len(w)),
                          "n_genes": int(w.shape[1]),
                          "median_pct_expr_over_all": float(np.nanmedian(w.to_numpy())),
                          "file": save(w.reset_index(), f"donor_pct_{a}")})
    report("§3e Per-donor detection, every arm",
           "One table per arm: donor x gene fraction of cells with >= 1 count, so any gene can be "
           "re-examined per patient without recomputing anything.",
           "`pct_expr` of each (arm, donor) group, pivoted. Columns are in family order. A blank "
           "cell is a donor whose arm has fewer than 20 cells, or a (study, gene) the coverage "
           "mask marked unmeasured -- blank is never 0. `mean_all` for the same groups is in "
           "`data/targets/pay51_long.parquet`.",
           pd.DataFrame(pct_index))

    # ---------------- §4 Gate 2
    reach_c = reach_all[reach_all["donor"].isin(kept)].copy()
    payloads = [p for p in COMPLEX if p in reach_c.columns]
    f_g2 = save(reach_c.sort_values(["arm", "donor"]), "gate2_reach")
    reach_long = reach_c.melt(id_vars=["arm", "donor", "n_cells"], value_vars=payloads,
                              var_name="payload", value_name="frac_cells_reachable")
    g2 = (reach_long.groupby(["payload", "arm"])
          .agg(n_donors=("donor", "nunique"),
               median_frac=("frac_cells_reachable", "median"),
               mean_frac=("frac_cells_reachable", "mean"),
               max_frac=("frac_cells_reachable", "max"))
          .reset_index())
    for thr in REACH_THRESHOLDS:
        g2 = g2.merge(reach_long.assign(_h=reach_long["frac_cells_reachable"] >= thr)
                      .groupby(["payload", "arm"])["_h"].mean()
                      .rename(f"donor_prev_ge{int(thr * 100)}pct").reset_index(),
                      on=["payload", "arm"], how="left")
    g2["arm"] = pd.Categorical(g2["arm"], [a for a in ARM_ORDER if a in set(g2["arm"])],
                               ordered=True)
    g2 = g2.sort_values(["payload", "arm"])
    f_g2s = save(g2, "gate2_summary")

    report("§4a Gate 2 -- reach, by payload and arm",
           "The fraction of cells that carry the COMPLETE receptor complex for each payload, per "
           "arm, and how many donors reach a usable level.",
           "Co-expression is evaluated per cell on raw counts (>= 1 count of every chain in the "
           "same cell), then averaged within each (arm, donor) group, then summarised over "
           "donors. `donor_prev_geX` is the fraction of donors in that arm whose group has >= X% "
           "reachable cells -- a donor prevalence, not a cohort mean, because a cohort mean hides "
           "whether the reach sits in a few patients. Complexes: IL-12 = IL12RB1 AND IL12RB2; "
           "IL-18 = IL18R1 AND IL18RAP; IL-15 = IL2RB AND IL2RG; XCL1 = XCR1; CXCL9/10 = CXCR3. "
           "Descriptive: no test is run here.",
           g2, file=f_g2s,
           note="Detection of a two-chain complex is the product of two sparse detections, so it "
                "is the quantity most sensitive to sequencing depth in this report. Read it "
                "beside the per-arm depth in §1b: an arm with half the UMI will show less than "
                "half the reach for the same biology.")

    report("§4b Gate 2 -- per donor",
           "Every (arm, donor) group's reachable fraction, for each payload.",
           "Same quantity as §4a before aggregation. `n_cells` is the group size; groups below "
           "20 cells are not built at all.",
           reach_c.sort_values(["arm", "donor"]), file=f_g2, max_rows=60)

    # ---------------- §5 Gate 3
    lig = long_c[long_c["gene"].isin(C_LIGANDS)]
    g3 = (lig.groupby(["gene", "arm"])
          .agg(n_donors=("donor", "nunique"),
               median_pct_expr=("pct_expr", "median"),
               mean_pct_expr=("pct_expr", "mean"),
               median_mean_all=("mean_all", "median"),
               frac_donors_pos=("pct_expr", lambda s: float((s >= DETECT_FLOOR).mean())))
          .reset_index())
    g3["arm"] = pd.Categorical(g3["arm"], [a for a in ARM_ORDER if a in set(g3["arm"])],
                               ordered=True)
    g3 = g3.sort_values(["gene", "arm"])
    f_g3 = save(lig.sort_values(["gene", "arm", "donor"])[
        ["gene", "arm", "donor", "study", "n_cells", "pct_expr", "mean_all"]], "gate3_ligand")
    f_g3s = save(g3, "gate3_summary")

    report("§5a Gate 3 -- endogenous ligand, by arm",
           "Whether each payload's ligand is already produced in the lesion, and by which "
           "population.",
           "Pseudobulk per (arm, donor): `mean_all` is the mean over that group's cells of "
           "log1p CP10K, `pct_expr` the fraction of cells with >= 1 count; the table reports the "
           "median and mean of those donor values and the fraction of donors above the 10% "
           "detection floor. Descriptive, no test. Myeloid, fibroblast, keratinocyte and "
           "vascular arms are included because CXCL9/10/11 are largely myeloid and CCL21 is "
           "lymphatic-endothelial -- without them an absent ligand cannot be told apart from an "
           "absent cell type. (Those two context arms are an addition to the brief.)",
           g3, file=f_g3s, max_rows=200)

    il18 = g3[g3["gene"].isin(["IL18", "IL18BP"])]
    report("§5b Gate 3 -- IL18BP, the decoy",
           "IL-18 beside its decoy receptor IL18BP, which is what decides between native IL-18 "
           "and a decoy-resistant variant as the payload.",
           "Same quantity as §5a, restricted to the two genes so the comparison is not buried. "
           "IL18BP present in the lesion means a native IL-18 payload is neutralised before it "
           "acts; the arm carrying it says which cell does the neutralising.",
           il18, file=f_g3s)

    # ---------------- §6 sensitivity
    head = g1[["gene", "family", "log2fc_paired", "q", "log2fc_detect", "q_detect",
               "n_donors_paired"]].rename(columns={c: f"head_{c}" for c in
                                                   ("log2fc_paired", "q", "log2fc_detect",
                                                    "q_detect", "n_donors_paired")})

    def sens(long_sub, label):
        out = pd.concat([gate1(long_sub, gs, f) for f, gs in FAMILIES.items()],
                        ignore_index=True)
        out = out[["gene", "family", "log2fc_paired", "q", "log2fc_detect", "q_detect",
                   "n_donors_paired"]]
        out = out.merge(head.drop(columns="family"), on="gene", how="outer")
        out["d_log2fc_paired"] = out["log2fc_paired"] - out["head_log2fc_paired"]
        out["d_log2fc_detect"] = out["log2fc_detect"] - out["head_log2fc_detect"]
        out["sign_flip_q05"] = ((out["q"] < 0.05) != (out["head_q"] < 0.05))
        return out.sort_values(["family", "gene"]), label

    cultured = sorted(don.index[don["is_cultured"]])
    cult_paired = [d for d in cultured if d in paired_donors]
    s1, _ = sens(long_c[~long_c["donor"].isin(cultured)], "cultured")
    f_s1 = save(s1, "sens_no_cultured")
    report("§6a Sensitivity -- excluding the cultured donors",
           "Gate 1 recomputed with the ex-vivo cultured donors removed, against the headline.",
           "Donors flagged `is_cultured` by `target_cohort` (built from `clin_ex_vivo`): "
           f"{', '.join(cultured) or 'none'}; of those, {', '.join(cult_paired) or 'none'} "
           "carry both CD4 arms and therefore affect the paired test. Columns prefixed `head_` "
           "are the headline values of §3; `d_` is this analysis minus the headline; "
           "`sign_flip_q05` flags a gene that crosses q = 0.05 in either direction.",
           s1, file=f_s1, max_rows=80)

    small = long_c[long_c["n_cells"] >= SMALL_ARM]
    s2, _ = sens(small, "small arms")
    f_s2 = save(s2, "sens_min60cells")
    report("§6b Sensitivity -- donor-arms below 60 cells removed",
           "Gate 1 recomputed with every donor-arm under 60 cells dropped, against the headline.",
           f"The headline floor is {MIN_CELLS} cells per donor-arm; here it is {SMALL_ARM}. A "
           "small arm estimates `pct_expr` from few cells, so it is both noisy and biased toward "
           "0 for a sparse gene. Same columns as §6a.",
           s2, file=f_s2, max_rows=80)

    lfc = (long_c[long_c["arm"].isin([MALIG, REACT])]
           .pivot_table(index="donor", columns=["gene", "arm"], values="mean_all"))
    lfc = pd.DataFrame({g: np.log2((lfc[(g, MALIG)] + PSEUDO) / (lfc[(g, REACT)] + PSEUDO))
                        for g in GENES if (g, MALIG) in lfc.columns
                        and (g, REACT) in lfc.columns})
    dsub = don.loc[[d for d in lfc.index if d in don.index]]
    dsub = dsub.assign(study=dsub.index.map(d2s))
    s3 = ols_stage(lfc, dsub)
    f_s3 = save(s3, "sens_stage_study")
    stage_x = pd.crosstab(dsub["clin_stage_class"].fillna("unknown"), dsub["study"]).reset_index()
    report("§6c Sensitivity -- stage, with study as a covariate",
           "Whether the malignant-vs-reactive effect differs between early and advanced disease "
           "once the cohort of origin is held fixed.",
           "Per gene, ordinary least squares of the donor-level log2FC on an early-vs-advanced "
           "indicator plus one dummy per study; `beta_early_vs_advanced` is the stage "
           "coefficient with study held fixed, `p` its two-sided t test, `q` BH over the genes "
           "of this table. `study_absorbed` is True where study and stage are collinear within "
           "the genes' donors and the study dummies had to be dropped.",
           s3, file=f_s3, max_rows=80,
           note="Stage and cohort are confounded in this atlas. The crosstab below is printed in "
                "§6c-i; 12 of the 18 early paired donors are li2024, so no early-vs-advanced "
                "claim here is separable from a li2024-vs-rest claim, and the coefficient should "
                "be read as an upper bound on a stage effect, not as an estimate of one.")
    report("§6c-i Stage x study, paired donors",
           "The confound itself, as counts.",
           "Independent paired donors only, from `target_cohort`'s `clin_stage_class`.",
           stage_x)

    if "pct_expr_ds" in long_c.columns:
        dsl = long_c.dropna(subset=["pct_expr_ds"]).copy()
        dsl["pct_expr"] = dsl["pct_expr_ds"]
        s4 = pd.concat([gate1(dsl, gs, f) for f, gs in FAMILIES.items()], ignore_index=True)
        s4 = s4[["gene", "family", "log2fc_detect", "q_detect", "n_donors_paired_detect"]]
        s4 = s4.rename(columns={"log2fc_detect": "log2fc_detect_depthmatched",
                                "q_detect": "q_detect_depthmatched"})
        s4 = s4.merge(g1[["gene", "log2fc_detect", "q_detect"]], on="gene", how="left")
        s4["d_log2fc_detect"] = (s4["log2fc_detect_depthmatched"] - s4["log2fc_detect"])
        s4 = s4.sort_values(["family", "gene"])
        f_s4 = save(s4, "sens_depth_matched")
        report("§6d Sensitivity -- depth-matched detection",
               "Gate 1 detection recomputed after equalising sequencing depth, against the "
               "headline detection.",
               f"Counts are binomially thinned per cell to a common {DS_DEPTH:,} UMI (seed "
               f"{SEED}); only cells at or above that depth take part "
               f"({diag.get('frac_cells_ds', float('nan')):.1%} of T cells). `pct_expr` is then "
               "recomputed on the thinned counts and the same paired Wilcoxon is run. This is "
               "the direct answer to the depth difference between arms in §1b, and it is an "
               "addition to the brief.",
               s4, file=f_s4, max_rows=80)

    # ---------------- §7 caveats
    caveats = pd.DataFrame([
        ("tgt49 is not fixed by this job",
         "The (study, gene) coverage mask built here removes nothing from this 58-gene set, but "
         "the same mask applied to the 2,115-gene surface screen removes 342 genes from li2024 "
         "(15,790 genes in its source object), which is 31 of the 75 paired donors. tgt49's "
         "KIR3DL2 row (rank 522, log2FC 0, q 9.5e-4) is computed with those 31 donors "
         "contributing structural zeros to both arms. Those tables were left untouched by "
         "decision."),
        ("The healthy arm lost four donors",
         "SC320/SC348/SC353/SC354 are dropped because their count rows are empty, not because "
         "of anything biological. The counts exist in "
         "data/D14_gaydosik2023_skin/processed/standardized.h5ad and were not recovered."),
        ("The NK arm overlaps the T arms and is one study",
         "NK-annotated cells are also labelled CD4 or CD8 in cell_type_T and remain inside those "
         "arms; and every one of them is li2024. XCL1 reach on NK is therefore a single-cohort "
         "number that is partly re-counted in the CD4 arms."),
        ("Detection is depth-confounded",
         "The malignant arm is sequenced deeper than the reactive arm (§1b), which inflates "
         "log2fc_detect and Gate 2 reach in the malignant direction. §6d is the matched version; "
         "where the two disagree, the matched one is the one to quote."),
        ("Expression and detection disagree for some genes",
         "log2fc_paired and log2fc_detect answer different questions -- how much per cell, versus "
         "in how many cells. Where they disagree (IL18R1/IL18RAP in the existing tables) both are "
         "reported and neither is chosen; a receptor that is detected in more cells at a lower "
         "level per cell is a different clinical object from one that is brighter on fewer."),
        ("Gate 2 and Gate 3 are descriptive",
         "No test is run on reach or on ligand pseudobulk. They carry no p value and no multiple "
         "testing correction, and a difference between arms there is an observation, not a "
         "result."),
        ("Myeloid subtypes come from a sidecar",
         "Mono_Mac/cDC1/cDC2/pDC labels are read from "
         "data/atlas_joint/skin_myeloid_fibro_subtypes.csv, assigned at cluster level with a "
         "re-clustering rescue step for the UNK cells; UNK is excluded here because some of it "
         "is T-cell doublets."),
        ("Family C mixes receptors and ligands in one BH correction",
         "Family C is corrected as one family of 23 genes as specified, although its receptors "
         "and its ligands are different hypotheses; the family column lets a reader re-correct "
         "the two subsets separately from pay51_cd4_paired.csv."),
    ], columns=["caveat", "detail"])
    report("§7 Contradictions and caveats",
           "Everything in the data that undercuts a gate, named rather than left to the reader.",
           "Assembled by hand from the diagnostics above; each row points at the section or the "
           "file that carries the number.",
           caveats)

    context = f"""
**Object.** T/NK arms from `{T_OBJ.name}` ({manifest['object_T']}), layer `raw_counts`;
myeloid, fibroblast and context arms from `{J_OBJ.name}`, whose `var` is the same
42,347-gene index, with subtype labels joined from `skin_myeloid_fibro_subtypes.csv`.

**Malignancy call.** `{CALL}` -- the project headline call. CD4 cells split into
`CD4_malignant` / `CD4_reactive` / `CD4_unassessed` by `malignancy.cd4_level`; `CD4_HC` is
then set from the curated diagnosis (`clin_dx`), not from the atlas `disease` column.
`CD4_unassessed` is not used here.

**Normalisation.** {manifest['normalisation']}. The denominator is the true 42k-gene library
`obs.total_counts`, not the sum over the columns read.

**Arms.** {', '.join(ARM_ORDER)}. Every arm uses the same >= {MIN_CELLS}-cell floor per donor
and the same donor keys, so arms are joinable per donor. NK overlaps the CD4/CD8 arms (§1d).

**Cohort.** `target_cohort.skin_donor_table()` + `filter_long`: {manifest['donor_keys_in_object']}
donor keys reduce to {manifest['independent_donor_keys']} independent keys
({manifest['people']} people) after dropping non-CD4 clones, diagnoses outside the MF/SS
spectrum and repeat biopsies. {manifest['paired_donors']} donors carry both CD4 arms and enter
the paired test; the healthy arm has {manifest['hc_donors']} donors after the empty-row rule.

**Gene families.** A permitted ({len(FAMILY_A)}), B disqualified ({len(FAMILY_B)}),
C chemoattractant axis ({len(FAMILY_C)}), plus {len(AUDIT)} audit genes. BH correction runs
**within family**, and separately from the 2,115-gene surface screen -- this is a targeted
hypothesis set, not a genome scan.

**Tests.** Paired Wilcoxon signed-rank on per-donor differences, malignant vs reactive CD4
within the same biopsy, run twice: on `mean_all` (expression) and on `pct_expr` (detection).
Donor-level only. Both are reported side by side and neither is preferred.

**What the gates can and cannot establish.**
- *Gate 1* establishes that a receptor's transcript is higher or lower on the clone than on the
  patient's own reactive CD4. It does not establish surface protein, and a depleted transcript
  is not proof the clone cannot respond to the cytokine.
- *Gate 2* establishes what fraction of cells carry every chain of a complex as detected mRNA.
  Co-detection of two sparse transcripts is a severe underestimate of the true co-expression,
  so Gate 2 is a lower bound, and a comparison between arms of different depth is confounded.
- *Gate 3* establishes whether the ligand transcript is already present in the lesion and in
  which population. It says nothing about protein, about secretion, or about the gradient a
  cell would actually see.
- None of the three establishes efficacy. They establish selectivity, reachability and
  redundancy, which is what chooses between payloads, not whether a payload works.
"""
    write_report("51 - Payload-selectivity screen (Gate 1-3), CD70 TRUCK, MF/CTCL skin", context)
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
