#!/usr/bin/env python
"""Build the regimprint (nb31) input tables from the nb30 cohort, then call scripts/build_inputs.py.

Reads   data/atlas_joint/pseq29_cd4_cells_v1.h5ad (raw counts, arm), pseq30_donor_table_v1.csv (30 donors),
        subclones_v4.parquet (nb24, mal_cluster arms), skin_T_arm_cnv_v5.parquet, GWCD4i_DE_Stim48hr.h5ad
Writes  data/regimprint_nb31/data/{cells_slim.h5ad, de_stim48hr.parquet, gene_arm.tsv, cellcycle_genes.txt,
        ffpe_donors.txt, cnv_calls.csv, cell_filter_summary.csv, inputs.npz, build_summary.txt}
CPU job (jobs/run_regimprint.sh prep). Not for the login node.
"""
import argparse, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "helpers"))
PKG = ROOT / "docs" / "regimprint_v0.1"
AJ = ROOT / "data" / "atlas_joint"

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / "data" / "regimprint_nb31" / "data"))
ap.add_argument("--subclones", default=str(AJ / "subclones_v4.parquet"))
ap.add_argument("--subclone-col", default="nested_subclone", choices=["nested_subclone", "cnv_subclone"])
ap.add_argument("--reuse-de", default=None, help="existing de_stim48hr.parquet to reuse instead of re-melting")
ap.add_argument("--cc-thr", type=float, default=0.1)
ap.add_argument("--min-cells", type=int, default=150)
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
LOG = []


def say(*x):
    s = " ".join(str(v) for v in x); print(s, flush=True); LOG.append(s)


import anndata as ad, scanpy as sc
import perturbseq_helpers as P, subclone_helpers as S

# ---------------------------------------------------------------- cells
dt = pd.read_csv(AJ / "pseq30_donor_table_v1.csv")
donors = dt.loc[dt["retained"].astype(bool), "donor"].astype(str).tolist()
assert len(donors) == 30, len(donors)
A = ad.read_h5ad(AJ / "pseq29_cd4_cells_v1.h5ad")
A = A[A.obs["donor"].astype(str).isin(donors)].copy()
A.obs["donor"] = A.obs["donor"].astype(str); A.obs["arm"] = A.obs["arm"].astype(str)
say("cohort cells:", A.shape, "| donors:", A.obs["donor"].nunique(), "| arms:", A.obs["arm"].value_counts().to_dict())

sub = pd.read_parquet(a.subclones)
if "cell_id" in sub.columns:
    sub = sub.set_index("cell_id")
lab = sub[a.subclone_col].astype(object).reindex(A.obs_names)
lab = lab.where(lab.notna() & (lab.astype(str) != ""), np.nan)
is_mal = (A.obs["arm"] == "malignant").to_numpy()
lab[~is_mal] = np.nan
has_dec = pd.Series(lab.notna().to_numpy() & is_mal).groupby(A.obs["donor"].to_numpy()).any()
dec_donors = sorted(has_dec.index[has_dec])
unl = is_mal & lab.isna().to_numpy() & A.obs["donor"].isin(dec_donors).to_numpy()
say(f"subclone labels: {Path(a.subclones).name} [{a.subclone_col}]; donors with a decomposition: {len(dec_donors)}/30")
say(f"malignant cells labelled: {int((is_mal & lab.notna().to_numpy()).sum())}/{int(is_mal.sum())}; "
    f"unlabelled malignant cells in decomposed donors (dropped): {int(unl.sum())}")
A.obs["subclone"] = lab.to_numpy()
A.obs["unlabelled_in_decomposed"] = unl

# cycling: S / G2M scores on a normalised copy, both arms
B = ad.AnnData(X=A.layers["raw_counts"].astype(np.float32), obs=A.obs[[]].copy(), var=A.var[[]].copy())
sc.pp.normalize_total(B, target_sum=1e4); sc.pp.log1p(B)
s_genes = [g for g in S.CC_S_GENES if g in B.var_names]; g2m_genes = [g for g in S.CC_G2M_GENES if g in B.var_names]
sc.tl.score_genes_cell_cycle(B, s_genes=s_genes, g2m_genes=g2m_genes)
A.obs["S_score"] = B.obs["S_score"].to_numpy(); A.obs["G2M_score"] = B.obs["G2M_score"].to_numpy()
A.obs["is_cycling"] = ((A.obs["S_score"] > a.cc_thr) | (A.obs["G2M_score"] > a.cc_thr)).to_numpy()
# Was hardcoded False on the claim that CD4_Treg is a separate label so Tregs were already
# excluded. That was only true of the label, not of the cells: the old per-cluster "CD4" carried
# a Treg admixture (reports/ccc52_mvb/qc_benign_arm/). Read it from the re-annotation instead.
import lineage as LIN
_lin = LIN.load_labels()
A.obs["is_treg"] = (pd.Series(_lin["label_new"]).reindex(A.obs_names).to_numpy() == "CD4_Treg")
del B
flt = (A.obs.assign(n=1, cyc=A.obs["is_cycling"].astype(int), unl=A.obs["unlabelled_in_decomposed"].astype(int))
       .groupby(["donor", "arm"])[["n", "cyc", "unl"]].sum().reset_index())
flt["frac_cycling"] = (flt["cyc"] / flt["n"]).round(3)
flt.to_csv(OUT / "cell_filter_summary.csv", index=False)
say(f"cycling (S_score or G2M_score > {a.cc_thr}): " + ", ".join(
    f"{k}={v:.1%}" for k, v in A.obs.groupby("arm")["is_cycling"].mean().items()))

A = A[~A.obs["unlabelled_in_decomposed"].to_numpy()].copy()
A.X = None
A.obs["subclone"] = A.obs["subclone"].astype(object)
keep_cols = ["donor", "arm", "study", "tech", "sample_id", "subclone", "is_cycling", "is_treg", "S_score", "G2M_score"]
A.obs = A.obs[[c for c in keep_cols if c in A.obs.columns]].copy()
for c in ("study", "tech", "sample_id"):
    if c in A.obs:
        A.obs[c] = A.obs[c].astype(str)
slim = OUT / "cells_slim.h5ad"
obs_w = A.obs.copy(); obs_w["subclone"] = obs_w["subclone"].astype("category")   # NaN survives as category
A.obs = obs_w
A.write_h5ad(slim)
say("wrote", slim.name, A.shape)

ffpe = sorted(A.obs.loc[A.obs["tech"].astype(str).str.contains("flex", case=False), "donor"].unique()) if "tech" in A.obs else []
(OUT / "ffpe_donors.txt").write_text("\n".join(ffpe) + "\n")
say("FFPE donors:", ffpe)
(OUT / "cellcycle_genes.txt").write_text("\n".join(sorted(set(S.CC_S_GENES) | set(S.CC_G2M_GENES))) + "\n")

# ---------------------------------------------------------------- gene -> arm
garm = P.gene_arms().str.replace("^chr", "", regex=True)
garm.rename_axis("gene").rename("arm").reset_index().to_csv(OUT / "gene_arm.tsv", sep="\t", index=False)
say("gene_arm.tsv:", len(garm), "genes,", garm.nunique(), "arms")

# ---------------------------------------------------------------- CNV calls per subclone and arm
arm = pd.read_parquet(AJ / "skin_T_arm_cnv_v5.parquet")
if "obs_name" in arm.columns:
    arm = arm.set_index("obs_name")
ARM_COLS = S.arm_order([c for c in arm.columns if c.startswith("chr")])
arm = arm.reindex(A.obs_names)[ARM_COLS]
use = ~A.obs["is_cycling"].to_numpy()
subl = A.obs["subclone"].astype(object).where(A.obs["subclone"].notna(), "clone1").astype(str).to_numpy()
rows = []
for d in donors:
    dm = use & (A.obs["donor"].to_numpy() == d)
    ben = arm[dm & (A.obs["arm"].to_numpy() == "reactive")].dropna(how="all")
    for s in sorted(set(subl[dm & (A.obs["arm"].to_numpy() == "malignant")])):
        As = arm[dm & (A.obs["arm"].to_numpy() == "malignant") & (subl == s)].dropna(how="all")
        if len(As) < 20 or len(ben) < 20:
            continue
        ev = S.call_arm_events(As, ben, z_thr=2.5, eps=0.01)
        rows += [dict(donor=d, subclone=s, arm=k.replace("chr", ""), call=int(v)) for k, v in ev.items()]
cnv = pd.DataFrame(rows); cnv.to_csv(OUT / "cnv_calls.csv", index=False)
say(f"cnv_calls.csv: {cnv.groupby(['donor', 'subclone']).ngroups} subclones; called arms per subclone (median): "
    f"{cnv.assign(c=cnv.call != 0).groupby(['donor', 'subclone']).c.sum().median():.0f}")
del arm, A

# ---------------------------------------------------------------- DE table (Stim48hr), long format
DE_PATH = Path(a.reuse_de) if a.reuse_de else OUT / "de_stim48hr.parquet"
if a.reuse_de:
    say("reusing DE table:", DE_PATH)
    BUILD_DE = False
else:
    BUILD_DE = True
if BUILD_DE:
  ps = P.load_pseq("Stim48hr", layers=("log_fc", "lfcSE", "adj_p_value"))
  o = ps.obs
  flag = np.zeros(len(o), bool)
  for c in ("distal_offtarget_flag", "neighboring_gene_KD"):
      if c in o:
          flag |= o[c].astype(str).str.lower().eq("true").to_numpy()
  keep = ~flag
  say(f"Perturb-seq Stim48hr: {len(o)} perturbations x {ps.n_vars} genes; off-target flagged (removed): {int(flag.sum())}")
  reg = o["regulator"].astype(str).to_numpy()
  order = np.argsort(-o["n_cells_target"].astype(float).fillna(0).to_numpy(), kind="stable") if "n_cells_target" in o else np.arange(len(o))
  first = np.zeros(len(o), bool); seen = set()
  for i in order:
      if keep[i] and reg[i] not in seen:
          seen.add(reg[i]); first[i] = True
  say(f"duplicate regulator rows dropped (kept the one with most cells): {int((keep & ~first).sum())}")
  idx = np.where(first)[0]; genes = np.asarray(ps.var_names).astype(str)
  lfc = ps.layers["log_fc"][idx]; se = ps.layers["lfcSE"][idx]; pj = ps.layers["adj_p_value"][idx]
  ok = np.isfinite(lfc) & np.isfinite(se)
  ri, gi = np.nonzero(ok)
  de = pd.DataFrame({"regulator": pd.Categorical.from_codes(ri, categories=pd.Index(reg[idx])),
                     "gene": pd.Categorical.from_codes(gi, categories=pd.Index(genes)),
                     "log2FoldChange": lfc[ok].astype(np.float32), "lfcSE": se[ok].astype(np.float32),
                     "padj": pj[ok].astype(np.float32)})
  on = de[de["regulator"].astype(str) == de["gene"].astype(str)]
  say(f"DE long table: {len(de):,} rows, {de['regulator'].nunique()} regulators; on-target rows: {len(on)} "
      f"(negative: {int((on.log2FoldChange < 0).sum())})")
  de["regulator"] = de["regulator"].astype(str); de["gene"] = de["gene"].astype(str)
  de.to_parquet(OUT / "de_stim48hr.parquet", index=False)
  del de, ps, lfc, se, pj

# ---------------------------------------------------------------- build_inputs.py (package script, unchanged)
npz = OUT / "inputs.npz"
cmd = [sys.executable, str(PKG / "scripts" / "build_inputs.py"), "--h5ad", str(slim), "--counts-layer", "raw_counts",
       "--donor-col", "donor", "--arm-col", "arm", "--reactive", "reactive", "--malignant", "malignant",
       "--subclone-col", "subclone", "--exclude-cols", "is_treg", "is_cycling", "--min-cells", str(a.min_cells),
       "--de", str(DE_PATH), "--gene-arm", str(OUT / "gene_arm.tsv"),
       "--exclude-genes", str(OUT / "cellcycle_genes.txt"), "--cnv", str(OUT / "cnv_calls.csv"),
       "--ffpe", str(OUT / "ffpe_donors.txt"), "--out", str(npz)]
say("$ " + " ".join(cmd))
r = subprocess.run(cmd, cwd=PKG, capture_output=True, text=True)
say(r.stdout.strip());
if r.returncode:
    say(r.stderr[-3000:]); (OUT / "build_summary.txt").write_text("\n".join(LOG) + "\n"); sys.exit(r.returncode)

z = np.load(npz, allow_pickle=True)
nsub = np.bincount(z["sub_donor"])
say(f"median library size: reference arm {np.median(z['y_ref'].sum(1)):,.0f}; malignant subclones {np.median(z['y_sub'].sum(1)):,.0f}")
say("donors:", len(z["donors"]), "| subclones:", len(z["sub_name"]), "| genes:", len(z["genes"]), "| regulators:", len(z["reg_names"]))
say("subclones per donor:", dict(zip(z["donors"].tolist(), nsub.tolist())))
n_multi = int((nsub > 1).sum())
say(f"donors with > 1 subclone: {n_multi}  -> gate (>= 5): {'PASS' if n_multi >= 5 else 'FAIL - STOP'}")
(OUT / "build_summary.txt").write_text("\n".join(LOG) + "\n")
sys.exit(0 if n_multi >= 5 else 3)
