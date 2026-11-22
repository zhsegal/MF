# 20_skin/29_perturbseq_reverse_engineering.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/20_skin/29_perturbseq_reverse_engineering.ipynb` and is never run._

# 29 — Reverse-engineering the MF malignant-CD4 program with genome-scale Perturb-seq (GWCD4i, Stim48hr), then crossing it with inferCNV

**Question.** Which gene knockdowns in a normal, TCR-stimulated CD4 T cell push its transcriptome toward the
MF malignant-CD4 state, which oppose it, what pathways they form, where the MF program sits on the
TCR-engagement axis — and do the regulators line up with the chromosome arms MF cells lose and gain?

**Resource.** Zhu, Dann *et al.* (Cell 2026, doi:10.1016/j.cell.2026.08.002; `docs/PIIS0092867426009293 (1).pdf`):
CRISPRi (ZIM3-dCas9) knockdown of 12,748 genes in primary human CD4+ T cells, 4 donors, three culture
conditions. We use **Stim48hr** — 48 h after CD3/CD28/CD2 restimulation (ImmunoCult, 12.5 µL/mL) — and only
the published per-perturbation DESeq2 estimates (`GWCD4i.DE_stats.h5ad`: 11,333 perturbations × 10,282 genes
in this condition). The cell-level data (~1.6 TB) are not needed and were not downloaded.

**Method = the authors' "pert2state" model, as published** (Methods *Cell state signature analysis*; package
`github.com/emdann/pert2state_model`, vendored verbatim in `helpers/pert2state_model.py`): the cell-state
signature (DESeq2 Wald z of malignant vs reactive CD4) is regressed on the perturbation z-score matrix
(genes × regulators, on-target entry masked) through a truncated SVD (D = 60) + elastic net, with 5-fold
cross-validation over genes, a scrambled-regulator null, and a held-out replication cohort. The regulator
weights $w_k$ are the result: **$w_k > 0$ — knocking down $k$ moves a normal CD4 cell toward the MF state
(loss-of-function-like); $w_k < 0$ — $k$'s activity is required for the MF state (dependency-like).**

**Cohort.** MF skin donors whose malignant clone is CD4 (`target_cohort.skin_donor_table`), CD4 cells only,
malignant = headline `mal_cluster` call, reactive = same-donor non-malignant CD4. Discovery / replication =
80/20 donor split stratified by study (the authors split OneK1K donors the same way).

**Our extension (§4, not in the paper).** Regulators are placed on chromosome arms and crossed with the v5
inferCNV calls: cohort-level (do $w>0$ regulators sit on recurrently lost arms, $w<0$ on gained arms),
per donor (does a donor's own deletion set reconstruct its own program — the authors' `subset_regulators`
reconstruction), and per cell (knockdown-mimicry score vs the cell's own arm dosage, cis genes removed).

**Run order.** `jobs/download_perturbseq_gwcd4i.sh` → `jobs/run_perturbseq_prep.sh` → this notebook.
§2 (first cell) and §4C read the 12 GB skin-T object: run those on the GPU/compute kernel; they cache to
`data/atlas_joint/pseq29_*` and every other cell works from the caches.

Deviations from the authors, all deliberate and listed in the plan: `l1_ratio` grid capped at 1 (sklearn ≥ 1.2
rejects their default); replication = within-cohort donor split, not an independent cohort; a signed pathway
test next to their unsigned one; the Arce 2024 TCR-activation signature used as a comparator, not only a
negative control.

```python  # cell 1
# ============================================================
# §0  Parameters
# ============================================================
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import sys
from pathlib import Path


def _resolve_nb_dir() -> Path:
    start = Path.cwd()
    for base in [start, *start.parents]:
        for sub in [Path("."), Path("MF")]:
            cand = base / sub
            if cand.name == "MF" and (cand / "data").exists():
                return cand.resolve()
    raise FileNotFoundError(f"could not locate MF/data from {start}")


NB_DIR = _resolve_nb_dir(); print("NB_DIR =", NB_DIR)
OUT_DIR = NB_DIR / "data" / "atlas_joint"
FIG_DIR = NB_DIR / "figures"; FIG_DIR.mkdir(exist_ok=True)
TAB_DIR = NB_DIR / "tables"; TAB_DIR.mkdir(exist_ok=True)
PSEQ_DIR = NB_DIR / "data" / "external_ref" / "perturbseq_gwcd4i"

# ---- inputs ----
CONDITION   = "Stim48hr"                                        # the only condition used (user's choice)
PSEQ_H5     = PSEQ_DIR / f"GWCD4i_DE_{CONDITION}.h5ad"          # jobs/run_perturbseq_prep.py
OBJ         = OUT_DIR / "skin_T_tcr_annotated_v4.h5ad"          # nb23 cache: nb21 labels + V(D)J fold-in; raw_counts
ARM         = OUT_DIR / "skin_T_arm_cnv_v5.parquet"             # nb23: per-cell 41-arm inferCNV matrix + diploid null rows
CNV_SUMMARY = TAB_DIR / "skin_cnv_v5_summary.csv"               # nb23: per-donor signed arm signature
CALL        = "mal_cluster"                                     # headline malignancy call (helpers/malignancy.py)
DISEASE     = "MF"

# ---- caches written by this notebook (heavy cells skip themselves when these exist) ----
CELLS_PQ  = OUT_DIR / "pseq29_cells_v1.parquet"                 # §0: cell_id, donor, study, arm, split
CD4_H5    = OUT_DIR / "pseq29_cd4_cells_v1.h5ad"                # §2: raw counts of the cohort's CD4 cells (HEAVY to build)
DE_DISC   = OUT_DIR / "pseq29_mf_cd4_de_disc_v1.csv"            # §2: pydeseq2 malignant vs reactive, discovery donors
DE_REPL   = OUT_DIR / "pseq29_mf_cd4_de_repl_v1.csv"            # §2: same, replication donors
DONOR_LFC = OUT_DIR / "pseq29_mf_cd4_donor_lfc_v1.parquet"      # §2: per-donor log2FC (genes x donors)
SCORES_PQ = OUT_DIR / "pseq29_cell_scores_v1.parquet"           # §4C: per-cell knockdown-mimicry scores

# ---- knobs (authors' values unless noted) ----
SEED          = 0
MIN_CELLS_ARM = 50      # donor needs >= this many malignant AND reactive CD4 cells
REPL_FRAC     = 0.2     # replication share of donors (authors: 80/20 OneK1K split)
MIN_DE_GENES  = 10      # regulator filter: significant effects on >= 10 measured genes (authors)
BASEMEAN_MIN  = 0.1     # gene filter: mean baseMean > 0.1 in every condition (authors)
N_PCS, N_SPLITS, N_REPEATS = 60, 5, 3   # authors: D = 60, 5-fold CV, 3 repeats (15 splits)
MIN_ARM_FRAC  = 0.25    # §4A: an arm is 'lost'/'gained' if the event recurs in >= 25 % of callable donors
TOP_K_CELL    = 20      # §4C: regulators on event arms scored per cell
N_PERM        = 1000
```

```
NB_DIR = /home/projects/nyosef/zvise/MF
```

```python  # cell 2
import importlib, gc, json, warnings
import numpy as np, pandas as pd
import scanpy as sc, h5py
import matplotlib.pyplot as plt
import scipy.stats as ss

sys.path.insert(0, str(NB_DIR / "helpers"))
import perturbseq_helpers as P
import malignancy as MAL
import target_cohort as TC
import subclone_helpers as S
import cluster_malig_helpers as CM
import tcr_signaling_helpers as TH
for _m in (P, MAL, TC, S, CM, TH):
    importlib.reload(_m)
try:
    from anndata.io import read_elem
except ImportError:
    from anndata.experimental import read_elem

np.random.seed(SEED)
sc.settings.verbosity = 1
plt.rcParams["figure.dpi"] = 120
warnings.filterwarnings("ignore", category=FutureWarning)


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"pseq29_{name}.png", bbox_inches="tight")
    plt.savefig(FIG_DIR / f"pseq29_{name}.pdf", bbox_inches="tight")
    plt.show()


print("pert2state_model vendored from", P.ROOT / "helpers" / "pert2state_model.py")
```

```
pert2state_model vendored from /home/projects/nyosef/zvise/MF/helpers/pert2state_model.py
```

## §0 — Cohort: CD4-malignant MF skin donors, CD4 cells, malignant vs reactive arms

Donor truth comes from `target_cohort.skin_donor_table()` (clinical fields of the object): a donor is kept
only if its malignant clone is **CD4** (`clin_phenotype`, not CD8/γδ/DN, not the CD8-aggressive / pcAECTCL
entities), the diagnosis is **MF**, and the patient is not already in the cohort under another key.
Cells are `cell_type_T == "CD4"`; **malignant** = `mal_cluster & mal_cluster_defined`, **reactive** =
`mal_cluster_defined & ~mal_cluster` in the same donor (`malignancy.cd4_level` logic). Donors need both arms
≥ `MIN_CELLS_ARM`. The discovery / replication split is random, stratified by study.
Light: reads obs only.

```python  # cell 4
OBS_COLS = ["donor", "study", "disease", "sample_id", "cell_type_T", "has_tcr", "tissue", "tech",
            "clin_phenotype", "clin_dx", "clin_stage_class", "stage_class"]
with h5py.File(OBJ, "r") as h5:
    obs = CM.read_obs(h5, cols=OBS_COLS)
print("skin T object obs:", obs.shape)

# the four malignancy calls, assembled by helpers/malignancy.py; has_tcr from THIS object (nb23 fold-in)
_has_tcr = obs["has_tcr"].astype(str).str.lower().eq("true").to_numpy()
mal = MAL.load_malignancy(obs.index, has_tcr=_has_tcr)
obs["malignant"] = mal[CALL].to_numpy() & mal[f"{CALL}_defined"].to_numpy()
obs["reactive"] = mal[f"{CALL}_defined"].to_numpy() & ~mal[CALL].to_numpy()
del mal

don = TC.skin_donor_table()
is_mf_cd4_donor = don["cohort_keep"] & don["disease"].astype(str).eq(DISEASE) & ~don["is_hc"]
print("\nMF donor keys dropped by the cohort rules:")
print(don.loc[don["disease"].astype(str).eq(DISEASE) & ~don["cohort_keep"], "drop_reason"].value_counts().to_string())

keep_cells = (obs["cell_type_T"].astype(str).eq("CD4") & obs["disease"].astype(str).eq(DISEASE)
              & obs["donor"].astype(str).isin(don.index[is_mf_cd4_donor]) & (obs["malignant"] | obs["reactive"]))
cells = obs.loc[keep_cells, ["donor", "study", "sample_id", "tech"]].astype(str).copy()
cells["arm"] = np.where(obs.loc[keep_cells, "malignant"], "malignant", "reactive")

ct = cells.groupby(["donor", "arm"]).size().unstack(fill_value=0).reindex(columns=["malignant", "reactive"], fill_value=0)
elig = ct.index[(ct >= MIN_CELLS_ARM).all(axis=1)]
cells = cells[cells["donor"].isin(elig)].copy()
dtab = ct.loc[elig].assign(study=don.loc[elig, "study"].astype(str), phenotype=don.loc[elig, "phenotype"],
                           stage=don.loc[elig, "clin_stage_class"] if "clin_stage_class" in don else "")
dtab["split"] = P.stratified_donor_split(dtab, by="study", frac=REPL_FRAC, seed=SEED)
cells["split"] = dtab["split"].reindex(cells["donor"]).to_numpy()
cells.index.name = "cell_id"
cells.reset_index().to_parquet(CELLS_PQ, index=False)

print(f"\neligible donors: {len(elig)}  (both arms >= {MIN_CELLS_ARM} CD4 cells)   cells: {len(cells):,}")
print(dtab["split"].value_counts().to_string())
print(dtab.groupby("study")[["malignant", "reactive"]].agg(["count", "sum"]).to_string())
dtab.sort_values(["split", "study"])
```

```
skin T object obs: (539916, 12)

             n_defined  frac_defined  n_malignant  frac_malignant_where_defined
mal_tcr       193830.0        0.3590      80328.0                        0.4144
mal_cnv       484787.0        0.8979     154024.0                        0.3177
mal_knn       401217.0        0.7431     259612.0                        0.6471
mal_cluster   493312.0        0.9137     236015.0                        0.4784

/home/projects/nyosef/zvise/MF/helpers/target_cohort.py:179: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  don[f"ever_{k}"] = don[f"tx_{k}"] | don[f"prior_{k}"]
/home/projects/nyosef/zvise/MF/helpers/target_cohort.py:179: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  don[f"ever_{k}"] = don[f"tx_{k}"] | don[f"prior_{k}"]
/home/projects/nyosef/zvise/MF/helpers/target_cohort.py:179: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  don[f"ever_{k}"] = don[f"tx_{k}"] | don[f"prior_{k}"]
/home/projects/nyosef/zvise/MF/helpers/target_cohort.py:179: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  don[f"ever_{k}"] = don[f"tx_{k}"] | don[f"prior_{k}"]
/home/projects/nyosef/zvise/MF/helpers/target_cohort.py:179: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  don[f"ever_{k}"] = don[f"tx_{k}"] | don[f"prior_{k}"]
/home/projects/nyosef/zvise/MF/helpers/target_cohort.py:179: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  don[f"ever_{k}"] = don[f"tx_{k}"] | don[f"prior_{k}"]
/home/projects/nyosef/zvise/MF/helpers/target_cohort.py:179: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  don[f"ever_{k}"] = don[f"tx_{k}"] | don[f"prior_{k}"]
/home/projects/nyosef/zvise/MF/helpers/target_cohort.py:179: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  don[f"ever_{k}"] = don[f"tx_{k}"] | don[f"prior_{k}"]
/home/projects/nyosef/zvise/MF/helpers/target_cohort.py:179: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  don[f"ever_{k}"] = don[f"tx_{k}"] | don[f"prior_{k}"]
/home/projects/nyose
... [8061 more chars truncated]
```

## §1 — The Perturb-seq resource (Stim48hr)

What the screen measured and which regulators enter the model. The authors' regulator filter keeps
perturbations with significant effects on ≥ 10 measured genes; the gene filter keeps genes with mean
baseMean > 0.1 in every condition. Their 111 functional regulator clusters (Suppl. Table 13) are the
pathway vocabulary; **cluster 99 "TCR signaling" is the Stim48hr-specific one** (195 regulators) and is the
map of what 48 h of TCR engagement does to a normal CD4 cell.

```python  # cell 6
pseq = P.load_pseq(CONDITION, PSEQ_H5)
gmask = P.gene_mask(pseq, BASEMEAN_MIN)
rmask = P.regulator_mask(pseq, MIN_DE_GENES)
print(f"{CONDITION}: {pseq.n_obs:,} perturbations x {pseq.n_vars:,} measured genes")
print(f"genes with mean baseMean > {BASEMEAN_MIN} in all conditions: {gmask.sum():,}")
print(f"regulators with >= {MIN_DE_GENES} DE genes (authors' model filter): {rmask.sum():,}")
print(pd.read_csv(TAB_DIR / "pseq29_perturbseq_qc.csv").to_string(index=False))

clusters = P.load_clusters(P.CLUSTERS_CSV)
reg2cl = P.regulator_cluster_map(clusters)
print("\nregulator clusters annotated as TCR signaling:")
print(clusters[clusters["manual_annotation"].str.contains("TCR", na=False)]
      [["cluster", "manual_annotation", "cluster_gene_size", "rest_count", "stim8hr_count", "stim48hr_count", "condition_specificity"]]
      .to_string(index=False))
c99 = clusters.set_index("cluster").loc[99, "members"]
print(f"\ncluster 99 members ({len(c99)}):", ", ".join(sorted(c99)))
ds99 = P.load_downstream(99, CONDITION)
print(f"\ncluster 99 downstream genes in {CONDITION}: {len(ds99):,}; most-shared targets:")
print(ds99.sort_values("num_of_upstream", ascending=False).head(15)
      [["downstream_gene", "num_of_upstream", "sign_coherence"]].to_string(index=False))
```

```
Stim48hr: 11,281 perturbations x 10,282 measured genes
genes with mean baseMean > 0.1 in all conditions: 10,282
regulators with >= 10 DE genes (authors' model filter): 2,582
condition                                                tier  n_regulators
 Stim48hr                                              tested         11281
 Stim48hr        n_total_de_genes>=10 (authors' model filter)          2582
 Stim48hr >75 DE genes & >50 cells (authors' clustering tier)          1117
 Stim48hr                                ontarget_significant          7195
 Stim48hr                               distal_offtarget_flag           127
 Stim48hr                                      measured genes         10282
 Stim48hr    genes with mean baseMean>0.1 in all 3 conditions         10282

regulator clusters annotated as TCR signaling:
 cluster manual_annotation  cluster_gene_size  rest_count  stim8hr_count  stim48hr_count condition_specificity
      21     TCR signaling                135           2            135               0               Stim8hr
      22     TCR signaling                 11           0             11               0      Stim8hr_Stim48hr
      99     TCR signaling                195           0              0             195              Stim48hr

cluster 99 members (195): AASDH, ADAT2, ADAT3, ANKRD49, ANKRD61, APEX1, ARL6IP1, ATP1B3, ATRX, ATXN7L3, BCL10, BCL11B, BICDL1, BRAT1, BRD9, C11orf54, C1D, CALCOCO2, CAMSAP1, CARMIL2, CBLL1, CCDC134, CCN1, CD2, CD247, CD28, CD3D, CD3E, CD3G, CEP135, CEP89, CLCC1, CLDN7, CNOT6L, COG6, COLGALT1, COPS9, CREBBP, CTPS1, CUL4A, CYB5R4, DCAF4, DCAF5, DNAAF5, DNAJC3, DNAJC8, DOLPP1, DOT1L, EIF1, ELOB, EML2, EP300, ERCC1, ERCC4, EWSR1, EXOC8, FANCB, FANCL, FBRS, FBXW7, FCHO1, FDX1, FIGNL1, FMR1, FOXO1, GALNT6, GDAP2, GNE, GPALPP1, GPI, GPS2, GSS, HAPSTR1, HELT, HS3ST3B1, HTT, ICE2, IKBKB, IL12RB2, IMPDH2, ING5, INPP5D, IRF4, ITK, JUNB, KAT6A, KGD4, KIF14, KMT2E, L2HGDH, LAT, LCK, LCP2, LETM2, LRP5, LSM12, LSS, MAF, MALT1, MAP3K5, MAX, MED14, MED23, MEN1, METTL23, MTMR10, MTX2, MYO9A, N6AMT1, NADK2, NCAPD3, NCAPG2, NDFIP2, NFIX, NKAPD1, NR4A3, NUDCD1, NUP188, NUP37, NUP42, ORAI1, OSR1, OST4, OTUB1, PARP3, PCYT2, PDCD5, PEX5, PGAP2, PHF23, PHF6, PIBF1, PKN3, PLCG1, PPDPF, PPP3R1, PRKAR1B, PRKCQ, PRKDC, PTPRC, PYGM, RAB1A, RAC2, RAD51B, RAVER1, RIDA, RNASEH2A, RNF19B, RNF8, RPE, RPS6KA1, SAXO5, SCAF4, SENP1, SET, SHOC2, SIK3, SKA2, SLC16A1, SLC9A5, SMARCD1, SNAP23, SND1, SOCS4, SRR, SRSF11, SS18L2, ST8SIA1, STAT5A, SZRD1, TBCA, TBX21, TBX22, TESPA1, TFAP4, TLN1, TMEM140, TMEM263, TMX1, TNRC6A, TP73, TPGS2, UBIAD1, UBR4, USP7, VAV1, VEZT, WNK1, XRCC2, YIPF3, ZAP70, ZC3H10, ZNF703, ZNF837, ZRANB2


cluster 99 downstream genes in Stim48hr: 10,144; most-shared targets:
downstream_gene  num_of_upstream  sign_coherence
            B2M              113        1.000000
          HSPD1              108       -1.000000
       HSP90AB1              100       -1.000000
           CD52               95        1.000000
           MCL1               91        1.000000
          TRAP1               88       -1.000000
        DCUN1D5               87       -1.000000
            EVL               87        0.977011
           CYCS               86       -1.000000
           FYB1               86        1.000000
            SRM               85       -1.000000
       EBNA1BP2               84       -1.000000
          PSAT1               84       -1.000000
         SLC1A5               83       -1.000000
         EIF4A1               83       -1.000000
```

## §2 — The MF signature: malignant vs reactive CD4, pseudobulk DESeq2 (the response vector)

The authors' response is the DESeq2 Wald z-score (log2FC / lfcSE) of the state vs its reference, from
pseudobulks with covariates. Here: one pseudobulk per donor × arm, pydeseq2 design `~ donor + arm`
(donor blocks the comparison, exactly like the paired within-donor tests elsewhere in the project), run
separately on the discovery and the replication donors. The per-donor log2FC (CP10K, malignant vs reactive)
is kept for §4B.

> **HEAVY (first cell only):** reads the cohort's CD4 rows out of the 12 GB object with the CSR row reader
> and caches them as `pseq29_cd4_cells_v1.h5ad`. Skips itself when the cache exists.

```python  # cell 8
cells = pd.read_parquet(CELLS_PQ).set_index("cell_id")
if CD4_H5.exists():
    print("cache exists:", CD4_H5.name)
else:
    if not os.environ.get("LSB_JOBID"):
        print("⚠️  no LSB_JOBID — if this is the login node, stop: reading ~100k CSR rows of the 12 GB object can crash it.")
    import anndata as ad
    with h5py.File(OBJ, "r") as h5:
        names = pd.Index(np.asarray(read_elem(h5["obs"][h5["obs"].attrs["_index"]])).astype(str))
        var_names = pd.Index(np.asarray(read_elem(h5["var"][h5["var"].attrs["_index"]])).astype(str))
        pos = names.get_indexer(cells.index)
        assert (pos >= 0).all(), "cells missing from the object"
        order = np.argsort(pos)
        Xr = CM.read_csr_rows(h5["layers"]["raw_counts"], pos[order])
    a = ad.AnnData(X=Xr, obs=cells.iloc[order].copy(), var=pd.DataFrame(index=var_names))
    a.layers["raw_counts"] = a.X
    a.write_h5ad(CD4_H5)
    print("wrote", CD4_H5.name, a.shape)
    del a, Xr; gc.collect()
```

```
cache exists: pseq29_cd4_cells_v1.h5ad
```

```python  # cell 9
if DE_DISC.exists() and DE_REPL.exists() and DONOR_LFC.exists():
    print("DE caches exist")
else:
    a = sc.read_h5ad(CD4_H5)
    a.obs["pb"] = a.obs["donor"].astype(str) + "|" + a.obs["arm"].astype(str)
    counts, ncell = S.pseudobulk_counts(a, "pb", layer="raw_counts", min_cells=MIN_CELLS_ARM)
    meta = pd.DataFrame({"donor": [s.split("|")[0] for s in counts.index],
                         "arm": [s.split("|")[1] for s in counts.index]}, index=counts.index)
    split_of = a.obs.groupby("donor")["split"].first()
    for name, path in (("discovery", DE_DISC), ("replication", DE_REPL)):
        sel = meta.index[meta["donor"].map(split_of).eq(name)]
        print(f"{name}: {meta.loc[sel, 'donor'].nunique()} donors, {len(sel)} pseudobulks")
        res = S.run_pydeseq2(counts.loc[sel], meta.loc[sel], "~ donor + arm",
                             contrast=["arm", "malignant", "reactive"], n_cpus=8)
        res.index.name = "gene"
        res.to_csv(path)
    # per-donor log2FC on CP10K pseudobulks (for the dosage-predicted program, §4B)
    cpm = counts.div(counts.sum(axis=1), axis=0) * 1e4
    Yd = {}
    for d in meta["donor"].unique():
        m_, r_ = f"{d}|malignant", f"{d}|reactive"
        if m_ in cpm.index and r_ in cpm.index:
            Yd[d] = np.log2((cpm.loc[m_] + 1) / (cpm.loc[r_] + 1))
    pd.DataFrame(Yd).rename_axis("gene").to_parquet(DONOR_LFC)
    del a; gc.collect()

res_disc = pd.read_csv(DE_DISC, index_col=0)
res_repl = pd.read_csv(DE_REPL, index_col=0)
Y_donor = pd.read_parquet(DONOR_LFC)
y_disc = P.signature_from_deseq(res_disc)
y_repl = P.signature_from_deseq(res_repl)
common = y_disc.index.intersection(y_repl.index)
print(f"discovery z: {len(y_disc):,} genes; replication z: {len(y_repl):,}; "
      f"discovery-vs-replication Pearson r = {np.corrcoef(y_disc[common], y_repl[common])[0, 1]:.3f} "
      f"(the ceiling for any reconstruction)")
print(f"mean log2FC discovery = {res_disc['log2FoldChange'].mean():+.3f} (the authors centre the signature; prepare_y standardizes it)")
```

```
DE caches exist

discovery z: 27,363 genes; replication z: 21,490; discovery-vs-replication Pearson r = 0.862 (the ceiling for any reconstruction)
mean log2FC discovery = -0.756 (the authors centre the signature; prepare_y standardizes it)
```

```python  # cell 10
# volcano with the known MF genes; sanity check of the signature
r = res_disc.dropna(subset=["padj"]).copy()
r["nlp"] = -np.log10(r["padj"].clip(1e-300))
fig, ax = plt.subplots(figsize=(5, 3))
ax.scatter(r["log2FoldChange"], r["nlp"], s=2, c="lightgrey", rasterized=True)
for side, col in (("up", "firebrick"), ("down", "navy")):
    g = [x for x in P.MF_MARKERS[side] if x in r.index]
    ax.scatter(r.loc[g, "log2FoldChange"], r.loc[g, "nlp"], s=12, c=col)
    for x in g:
        ax.annotate(x, (r.loc[x, "log2FoldChange"], r.loc[x, "nlp"]), fontsize=6, xytext=(2, 2), textcoords="offset points")
n_sig = int((r["padj"] < 0.05).sum())
ax.set_xlabel("log2FC malignant vs reactive CD4 (discovery donors)"); ax.set_ylabel("-log10 FDR")
ax.set_title(f"MF signature: {n_sig:,} genes at FDR<5%; CD7/CD2 down, TOX/GATA3 up", fontsize=8)
savefig("mf_signature_volcano")
print(r.loc[[x for x in P.MF_MARKERS["down"] + P.MF_MARKERS["up"] if x in r.index], ["log2FoldChange", "padj"]].round(4).to_string())
```

```
log2FoldChange    padj
gene                           
CD7             -4.1675  0.0000
CD2             -0.6919  0.0116
CD5             -0.9993  0.0014
DPP4            -2.2790  0.0000
IL7R            -2.2455  0.0000
KLF2            -2.3359  0.0000
SELL            -2.5041  0.0000
TOX              1.0537  0.0000
GATA3            0.4131  0.0077
PLS3            -0.1717  0.7279
KIR3DL2         -0.7623  0.4379
TWIST1           2.5484  0.0000
TFAP2A           2.9400  0.0000
CCR4            -0.4388  0.1874
TNFRSF8          0.9678  0.0025
DNM3             1.1972  0.0000
```

## §3 — Reverse engineering: which knockdowns reproduce the MF program?

### 3a The model (authors', verbatim): CV over genes, scrambled null, replication donors

`Perturb2StateModel(pca_transform=True, n_pcs=60, n_splits=5, n_repeats=3)`: per-regulator standardization →
TruncatedSVD(60) → ElasticNetCV (inner 4-fold; α grid logspace(−3, 3, 30), l1_ratio logspace(−2, 0, 10)).
Genes = discovery ∩ replication ∩ Perturb-seq genes passing the baseMean filter; regulators = Stim48hr
perturbations with ≥ 10 DE genes; the on-target entry is masked (column-mean imputed). Performance = Pearson
r between predicted and observed z on **held-out genes**; the scrambled control permutes each gene's z across
regulators; the replication arm correlates the held-out-gene prediction with the replication donors'
signature, with the discovery–replication correlation as the ceiling.

```python  # cell 12
genes = pseq.var_names[gmask].intersection(y_disc.index).intersection(y_repl.index)
regs = pseq.obs_names[rmask]
X = P.build_X(pseq, genes, regulators=regs, layer="zscore_masked")
y = y_disc.loc[X.index]
print(f"X: {X.shape[0]:,} genes x {X.shape[1]:,} regulators; y: {len(y):,} genes")

models = {}
models["MF Stim48hr"] = P.fit_pert2state(X, y, n_pcs=N_PCS, n_splits=N_SPLITS, n_repeats=N_REPEATS, seed=SEED + 42, model_id="MF")
models["scrambled"] = P.fit_pert2state(X, y, n_pcs=N_PCS, n_splits=N_SPLITS, n_repeats=N_REPEATS, seed=SEED + 42,
                                       scramble=True, model_id="MF_scrambled")
m_real = models["MF Stim48hr"]
summ = P.heldout_summary(models)
repl = P.replication_eval(m_real, P.prepare_y(y_repl.reindex(X.index).dropna()))
summ.to_csv(TAB_DIR / f"pseq29_model_fit_{CONDITION}.csv"); repl.to_csv(TAB_DIR / f"pseq29_model_replication_{CONDITION}.csv", index=False)
print(summ[["n_genes", "n_regulators", "test_pearson", "test_pearson_se", "train_pearson", "alpha", "l1_ratio"]].round(4).to_string())
print("\nreplication (held-out genes, replication donors):")
print(repl.round(3).to_string(index=False))

fig, axs = plt.subplots(1, 2, figsize=(5, 3))
for k, (name, m) in enumerate(models.items()):
    v = m.eval["test_pearson"]
    axs[0].scatter(np.full(len(v), k) + np.random.uniform(-0.08, 0.08, len(v)), v, s=10, alpha=0.7)
    axs[0].errorbar(k, v.mean(), yerr=v.sem(), fmt="o", color="k", capsize=3)
axs[0].set_xticks(range(len(models))); axs[0].set_xticklabels(list(models), fontsize=7)
axs[0].axhline(0, ls=":", c="k"); axs[0].set_ylabel("Pearson r, held-out genes", fontsize=8)
axs[1].scatter(np.random.uniform(-0.08, 0.08, len(repl)), repl["correlation"], s=10, alpha=0.7)
axs[1].errorbar(0, repl["correlation"].mean(), yerr=repl["correlation"].sem(), fmt="o", color="k", capsize=3)
axs[1].axhline(repl["upper_correlation"].mean(), ls="--", c="grey"); axs[1].axhline(0, ls=":", c="k")
axs[1].set_xticks([0]); axs[1].set_xticklabels(["replication donors"], fontsize=7); axs[1].set_ylabel("Pearson r vs replication signature", fontsize=8)
axs[1].text(0.05, repl["upper_correlation"].mean(), "discovery-vs-replication ceiling", fontsize=6, va="bottom", color="grey")
fig.suptitle(f"Stim48hr knockdowns reconstruct the MF program: held-out r = {summ.loc['MF Stim48hr','test_pearson']:.2f} "
             f"(scrambled {summ.loc['scrambled','test_pearson']:.2f}); replication r = {repl['correlation'].mean():.2f}", fontsize=7)
savefig("model_fit")
```

```
X: 10,096 genes x 2,582 regulators; y: 10,096 genes

             n_genes  n_regulators  test_pearson  test_pearson_se  train_pearson   alpha  l1_ratio
model                                                                                             
MF Stim48hr    10096          2582        0.6113           0.0040         0.6193  0.2300    0.0154
scrambled      10096          2582        0.2627           0.0057         0.2647  0.2183    0.5347

replication (held-out genes, replication donors):
 split  n_test_genes  correlation  upper_correlation  pred_input_correlation
     0          2020        0.609              0.884                   0.631
     1          2019        0.554              0.868                   0.589
     2          2019        0.581              0.872                   0.601
     3          2019        0.608              0.884                   0.626
     4          2019        0.580              0.882                   0.610
     5          2020        0.601              0.879                   0.624
     6          2019        0.598              0.881                   0.622
     7          2019        0.593              0.879                   0.626
     8          2019        0.567              0.876                   0.579
     9          2019        0.576              0.876                   0.604
    10          2020        0.610              0.884                   0.630
    11          2019        0.568              0.877                   0.598
    12          2019        0.587              0.883                   0.613
    13          2019        0.595              0.872                   0.614
    14          2019        0.581              0.874                   0.606
```

### 3b Regulator weights $w_k$ — the regulators of the MF program

Top and bottom 20 by $w$ (mean ± SEM over the 15 fits), the authors' "Long Island City" plot (all
regulators, x-axis ordered by Ward clustering of their effects on the significant signature genes), and the
**control panel**: CTCL genes known to be lost / mutated (expect $w>0$) or gained / gain-of-function /
dependencies (expect $w<0$). The panel plays the role of the authors' `known_regulators`; a fit that ranks it
at random is reported as such.

```python  # cell 14
coefs = P.coef_table(m_real, clusters=reg2cl)
coefs.to_csv(TAB_DIR / f"pseq29_regulator_weights_{CONDITION}.csv")
ctrl = P.control_ranks(coefs)
ctrl.to_csv(TAB_DIR / f"pseq29_control_panel_{CONDITION}.csv", index=False)
print("control panel (sign_ok = sign of w matches the literature expectation; abs_rank of", len(coefs), "regulators):")
print(ctrl.round(4).to_string(index=False))
tested = ctrl.dropna(subset=["coef_mean"])
print(f"\nsign agreement: {int(tested['sign_ok'].sum())}/{len(tested)} tested controls "
      f"(binomial p = {ss.binomtest(int(tested['sign_ok'].sum()), len(tested), 0.5, alternative='greater').pvalue:.3f}); "
      f"median |w| rank of controls = {tested['abs_rank'].median():.0f}")

top = pd.concat([coefs.head(20), coefs.tail(20)])
ctrl_genes = {g for v in P.CONTROL_PANEL.values() for g in v}
fig, ax = plt.subplots(figsize=(5, 6))
yy = np.arange(len(top))[::-1]
ax.errorbar(top["coef_mean"], yy, xerr=top["coef_sem"], fmt="o", ms=3, color="k", elinewidth=0.8)
ax.set_yticks(yy); ax.set_yticklabels(top.index, fontsize=6)
for lab in ax.get_yticklabels():
    if lab.get_text() in ctrl_genes: lab.set_color("firebrick"); lab.set_fontweight("bold")
    if lab.get_text() in P.TCR_CASCADE: lab.set_color("darkgreen"); lab.set_fontweight("bold")
ax.axvline(0, ls="--", c="grey", lw=0.5)
ax.set_xlabel("regulator weight $w$  (>0: knockdown mimics MF; <0: knockdown opposes MF)", fontsize=7)
ax.set_title("Top regulators of the MF CD4 program (red = CTCL control panel, green = TCR cascade)", fontsize=7)
savefig("regulator_weights_top")
```

```
control panel (sign_ok = sign of w matches the literature expectation; abs_rank of 2582 regulators):
                        group regulator  coef_mean  coef_sem  coef_rank  abs_rank sign_ok
     LoF in CTCL (expect w>0)      PTEN    -0.0032    0.0003     0.8896     581.0   False
     LoF in CTCL (expect w>0)   TNFAIP3        NaN       NaN        NaN       NaN     NaN
     LoF in CTCL (expect w>0)      ZEB1    -0.0065    0.0002     0.9888      77.0   False
     LoF in CTCL (expect w>0)    CDKN2A        NaN       NaN        NaN       NaN     NaN
     LoF in CTCL (expect w>0)      TP53    -0.0036    0.0002     0.9120     461.0   False
     LoF in CTCL (expect w>0)    ARID1A     0.0001    0.0001     0.5025    2527.0    True
     LoF in CTCL (expect w>0)    DNMT3A        NaN       NaN        NaN       NaN     NaN
     LoF in CTCL (expect w>0)      TET2     0.0020    0.0001     0.2216    1119.0    True
     LoF in CTCL (expect w>0)     KMT2D        NaN       NaN        NaN       NaN     NaN
     LoF in CTCL (expect w>0)      CD58        NaN       NaN        NaN       NaN     NaN
     LoF in CTCL (expect w>0)       FAS        NaN       NaN        NaN       NaN     NaN
     LoF in CTCL (expect w>0)    NFKBIA        NaN       NaN        NaN       NaN     NaN
     LoF in CTCL (expect w>0)       RB1        NaN       NaN        NaN       NaN     NaN
     LoF in CTCL (expect w>0)      CYLD        NaN       NaN        NaN       NaN     NaN
     LoF in CTCL (expect w>0)     PRDM1        NaN       NaN        NaN       NaN     NaN
GoF / dependency (expect w<0)     PLCG1    -0.0077    0.0002     0.9930      43.0    True
GoF / dependency (expect w<0)    CARD11    -0.0034    0.0001     0.9031     511.0    True
GoF / dependency (expect w<0)     STAT3    -0.0010    0.0002     0.6559    1831.0    True
GoF / dependency (expect w<0)    STAT5B     0.0000    0.0001     0.5095    2566.0   False
GoF / dependency (expect w<0)      JAK3        NaN       NaN        NaN       NaN     NaN
GoF / dependency (expect w<0)      JAK1        NaN       NaN        NaN       NaN     NaN
GoF / dependency (expect w<0)      CD28    -0.0003    0.0001     0.5548    2338.0    True
GoF / dependency (expect w<0)     NFKB2     0.0023    0.0002     0.1906     971.0   False
GoF / dependency (expect w<0)       MYC     0.0030    0.0002     0.1286     649.0   False
GoF / dependency (expect w<0)       TOX        NaN       NaN        NaN       NaN     NaN
GoF / dependency (expect w<0)     GATA3     0.0038    0.0002     0.0759     394.0   False
GoF / dependency (expect w<0)      IRF4    -0.0004    0.0001     0.5684    2272.0    True
GoF / dependency (expect w<0)      BATF     0.0016    0.0002     0.2770    1410.0   False
GoF / dependency (expect w<0)  TNFRSF1B        NaN       NaN        NaN       NaN     NaN

sign agreement: 7/15 tested controls (binomial p = 0.696); median |w| rank of controls = 971
```

```python  # cell 15
# the authors' "Long Island City" plot: every regulator, ordered by clustering of its effects on signature genes
sig_genes = res_disc.index[res_disc["padj"] < 0.01].intersection(X.index)
order = P.lic_order(X, sig_genes)
lic = coefs.reindex(order).reset_index()
lic["rank"] = np.arange(len(lic))
fig, ax = plt.subplots(figsize=(5, 3))
ax.errorbar(lic["rank"], lic["coef_mean"], yerr=lic["coef_sem"], fmt=".", color="grey", ms=2, alpha=0.4, elinewidth=0.5)
for grp, col in (("LoF in CTCL (expect w>0)", "firebrick"), ("GoF / dependency (expect w<0)", "navy")):
    g = lic[lic["regulator"].isin(P.CONTROL_PANEL[grp])]
    ax.scatter(g["rank"], g["coef_mean"], s=14, c=col, zorder=3, label=grp)
    for _, row in g.iterrows():
        ax.annotate(row["regulator"], (row["rank"], row["coef_mean"]), fontsize=5, xytext=(1, 2), textcoords="offset points", color=col)
for _, row in pd.concat([lic.nlargest(8, "coef_mean"), lic.nsmallest(8, "coef_mean")]).iterrows():
    if row["regulator"] not in ctrl_genes:
        ax.annotate(row["regulator"], (row["rank"], row["coef_mean"]), fontsize=5, xytext=(1, 2), textcoords="offset points")
ax.axhline(0, ls="--", c="k", lw=0.5); ax.legend(fontsize=5, frameon=False)
ax.set_xlabel(f"{len(lic):,} regulators, ordered by clustering of effects on {len(sig_genes):,} signature genes", fontsize=7)
ax.set_ylabel("$w$", fontsize=8)
ax.set_title("Regulator weights across the screen; CTCL controls sit where their biology predicts" if tested["sign_ok"].mean() > 0.7
             else "Regulator weights across the screen; CTCL controls are NOT recovered", fontsize=7)
savefig("regulator_weights_lic")
```

### 3c Pathways: the authors' regulator clusters and the project's TCR/costim modules

The authors test whether a pathway's members carry larger $|w|$ than the rest (one-sided Mann–Whitney on
$|w|$). Direction matters for us, so the signed tests are added: members' $w$ > rest means "knocking down
this pathway mimics MF" (the pathway restrains the program), < rest means "the MF program depends on it".
Sets = the 111 clusters (Suppl. Table 13) + `tcr_signaling_helpers.TCR_MODULES`.

```python  # cell 17
sets = {**P.cluster_sets(clusters), **{f"TCRmod:{k}": v for k, v in TH.TCR_MODULES.items()}}
ptab = P.pathway_tests(coefs, sets, min_n=5)
ptab.to_csv(TAB_DIR / f"pseq29_pathway_tests_{CONDITION}.csv", index=False)
print("pathways with larger |w| than the rest (authors' test), BH q < 0.1:")
print(ptab[ptab["q_abs_greater"] < 0.1][["set", "n", "mean_w", "mean_abs_w", "q_abs_greater", "q_signed_greater", "q_signed_less"]].round(4).to_string(index=False))
print("\nsigned: knockdown MIMICS MF (q < 0.1):")
print(ptab[ptab["q_signed_greater"] < 0.1][["set", "n", "mean_w", "delta_signed", "q_signed_greater", "members"]].round(4).to_string(index=False))
print("\nsigned: knockdown OPPOSES MF (q < 0.1):")
print(ptab[ptab["q_signed_less"] < 0.1][["set", "n", "mean_w", "delta_signed", "q_signed_less", "members"]].round(4).to_string(index=False))

show = ptab.sort_values("p_abs_greater").head(25).iloc[::-1]
fig, ax = plt.subplots(figsize=(5, 5))
sc_ = ax.scatter(show["mean_w"], np.arange(len(show)), s=8 + 2 * show["n"], c=show["mean_w"], cmap="RdBu_r",
                 vmin=-show["mean_w"].abs().max(), vmax=show["mean_w"].abs().max(), edgecolor="k", linewidth=0.3)
ax.set_yticks(np.arange(len(show))); ax.set_yticklabels([s[:45] for s in show["set"]], fontsize=6)
ax.axvline(0, ls="--", c="grey", lw=0.5); ax.set_xlabel("mean $w$ of members (size = n regulators)", fontsize=7)
ax.set_title("Pathways carrying the MF program (top 25 by the authors' |w| test)", fontsize=7)
savefig("pathway_dotplot")
```

```
pathways with larger |w| than the rest (authors' test), BH q < 0.1:
                       set  n  mean_w  mean_abs_w  q_abs_greater  q_signed_greater  q_signed_less
         c22 TCR signaling 11 -0.0080      0.0080         0.0000            1.0000         0.0000
          c2 Ino80 complex 11  0.0061      0.0061         0.0000            0.0000         1.0000
 c32 Ragulator-Rag complex 11 -0.0037      0.0037         0.0103            1.0000         0.0000
c7 nonsense-mediated decay 10 -0.0034      0.0040         0.0141            1.0000         0.0037
 c31 Ragulator-Rag complex 12 -0.0032      0.0032         0.0506            1.0000         0.0001
               c93 unknown  8 -0.0051      0.0051         0.0506            1.0000         0.0001
    c36 citrate metabolism  8  0.0034      0.0034         0.0787            0.0006         1.0000
               c64 unknown 17 -0.0023      0.0033         0.0787            1.0000         0.0051

signed: knockdown MIMICS MF (q < 0.1):
                   set  n  mean_w  delta_signed  q_signed_greater                                                                         members
      c2 Ino80 complex 11  0.0061        0.9293            0.0000             RO60,NFRKB,HIRIP3,INO80E,INO80C,HEY1,UCHL5,INO80,ACTR5,ACTR8,INO80B
c36 citrate metabolism  8  0.0034        0.7491            0.0006                                     ACLY,MDH2,CS,SLC25A1,DLAT,STYXL1,PDHA1,PDHB
c19 NCOR/HDAC3 complex  6  0.0034        0.6823            0.0152                                          NCOR1,RELL2,ATXN2L,LSM12,TBL1XR1,HDAC3
           c54 unknown 33  0.0017        0.3620            0.0030 APOBEC3B,STIP1,LOXL4,ARIH1,SP100,RIPK1,MVB12B,NRBP1,RXRB,CAB39,NCKAP1L,HIP1R...
           c77 unknown 84  0.0017        0.3202            0.0000        PAN3,GCN1,UBA5,KAT7,HNRNPD,PDCD10,TSC2,TIMM10,DDA1,NKX2-2,H3C11,DDIT4...
           c26 unknown 15  0.0013        0.3475            0.0962         CASK,ZBTB1,MAU2,JOSD2,PLOD3,RBM33,SPOP,HOXB4,RAB27B,PHF3,METRN,CDCA3...
           c27 unknown 89  0.0013        0.2775            0.0001   BAZ1A,STIP1,LOXL4,CMTR1,SP100,NLRX1,ZNF414,DRAP1,RIPK1,MVB12B,PRSS21,NRBP1...
 c33 elongator complex 12  0.0025        0.4918            0.0199             NR2C2AP,KTI12,ELP2,ELP1,ELP3,MOCS3,CTU1,ELP5,ELP4,SLC39A6,URM1,CTU2
      c9 mediator-SAGA 34  0.0012        0.2755            0.0307  MED19,MED1,SCYL3,TSPYL5,TADA2B,SUPT7L,EID2,POGLUT3,SUPT20H,TADA1,CCNC,ECHS1...

signed: knockdown OPPOSES MF (q < 0.1):
                          set   n  mean_w  delta_signed  q_signed_less                                                                           members
            c22 TCR signaling  11 -0.0080       -0.9774         0.0000                        VAV1,CD247,BCAT2,LAT,CD3G,ZBED3,PLCG1,CD3D,CD3E,ZAP70,LCP2
    c32 Ragulator-Rag complex  11 -0.0037       -0.8038         0.0000          RRAGA,TMEM214,RRAGC,LAMTOR4,FOXK1,RPTOR,WDR59,MIOS,LAMTOR2,LAMTOR3,WDR24
   c7 nonsense-mediated decay  10 -0.0034       -0.6282         0.0037                               UPF2,TMEM79,SMG7,XRN1,SMG9,CASC3,SMG8,SMG1,SRR,SMG5
    c31 Ragulator-Rag complex  12 -0.0032       -0.7401         0.0001  LAMTOR5,RRAGA,TMEM214,RRAGC,VPS35,LAMTOR4,VPS29,RPTOR,WDR59,MIOS,LAMTOR2,LAMTOR3
                  c93 unknown   8 -0.0051       -0.7849         0.0001                                    USO1,NSD2,MBD5,LRBA,ARHGAP27,BCAT2,ZBED3,FITM2
                  c64 unknown  17 -0.0023       -0.4648         0.0051     DDX31,VPS45,ERCC6L,DHX9,SSB,ZFYVE21,IPO9,PRPF38B,SLC35B1,ZNF829,RTRAF,NAF1...
            c21 TCR signaling 100 -0.0012       -0.2812         0.0000        TSC2,COG5,COG6,SPTLC2,NATD1,MED23,TSC1,UBIAD1,STIM1,NUDCD1,COP1,PPP1R10...
          c48 tRNA processing   9 -0.0023       -0.5746         0.0109                     NDUFA8,RNASEH1,NUP188,PGS1,METTL1,ADAT3,RTRAF,TRMT61A,LAMTOR2
           c17 genome defense   5 -0.0032       -0.7164         0.0109                                                  SETDB1,UBR1,
... [785 more chars truncated]
```

### 3d The TCR-engagement axis

Three independent readings of where the MF program sits relative to TCR signalling:

1. **Comparator fit.** The same model on the authors' TCR-activation signature (Arce 2024, Teff stimulated vs
   resting — their negative control for polarization) gives $w^{act}$; regulators shared by MF and activation
   vs MF-specific ones, and the TCR-cascade components (CD3, LCK, ZAP70, LAT, LCP2, PLCG1, VAV1, CARD11, …).
2. **Projection on cluster 99.** The sign-coherent downstream program of the Stim48hr TCR-signalling cluster
   (genes that go *up* when TCR-signalling regulators are knocked down vs *down*). A **positive** projection =
   the MF program resembles *loss* of TCR signalling (below the stimulated state); **negative** = it resembles
   sustained engagement. The activation signature itself is the reference (expected negative).
3. The same for the Stim8hr clusters 21/22 (early TCR programme).

```python  # cell 19
y_act = P.load_activation_signature(P.ARCE_CSV)
m_act = P.fit_pert2state(X, y_act, n_pcs=N_PCS, n_splits=N_SPLITS, n_repeats=1, seed=SEED + 42, model_id="activation")
w_act = P.coef_table(m_act)
print("activation-signature fit:", P.heldout_summary({"Arce activation": m_act})[["n_genes", "test_pearson", "test_pearson_se"]].round(3).to_string())
both = coefs[["coef_mean", "coef_sem", "cluster"]].join(w_act[["coef_mean", "coef_sem"]], rsuffix="_act").dropna(subset=["coef_mean_act"])
rho, p = ss.spearmanr(both["coef_mean"], both["coef_mean_act"])
both["mf_specific"] = (both["coef_mean"].abs().rank(ascending=False) <= 100) & (both["coef_mean_act"].abs().rank(ascending=False) > 500)
both.to_csv(TAB_DIR / f"pseq29_mf_vs_activation_weights_{CONDITION}.csv")
print(f"\nSpearman rho(w_MF, w_activation) = {rho:.3f} (p = {p:.1e}) over {len(both):,} regulators")
print("\nTCR cascade components:")
print(both.reindex([g for g in P.TCR_CASCADE if g in both.index])[["coef_mean", "coef_mean_act", "cluster"]].round(4).to_string())
print("\nMF-specific regulators (top-100 |w_MF|, outside the top-500 |w_act|):")
print(both[both["mf_specific"]].sort_values("coef_mean")[["coef_mean", "coef_mean_act", "cluster"]].round(4).to_string())

fig, ax = plt.subplots(figsize=(5, 3))
ax.scatter(both["coef_mean_act"], both["coef_mean"], s=3, c="lightgrey", rasterized=True)
cas = both.reindex([g for g in P.TCR_CASCADE if g in both.index])
ax.scatter(cas["coef_mean_act"], cas["coef_mean"], s=12, c="darkgreen")
for g, row in cas.iterrows():
    ax.annotate(g, (row["coef_mean_act"], row["coef_mean"]), fontsize=5, xytext=(2, 2), textcoords="offset points", color="darkgreen")
for g, row in both[both["mf_specific"]].iterrows():
    ax.annotate(g, (row["coef_mean_act"], row["coef_mean"]), fontsize=4, xytext=(2, 2), textcoords="offset points", color="firebrick")
ax.axhline(0, ls=":", c="k", lw=0.5); ax.axvline(0, ls=":", c="k", lw=0.5)
ax.set_xlabel("$w$ for TCR activation (Arce 2024 stim vs rest)", fontsize=7); ax.set_ylabel("$w$ for MF malignant vs reactive", fontsize=7)
ax.set_title(f"MF regulators vs TCR-activation regulators: rho = {rho:.2f}; cascade in green, MF-specific in red", fontsize=7)
savefig("mf_vs_activation_weights")
```

```
activation-signature fit:                  n_genes  test_pearson  test_pearson_se
model                                                  
Arce activation     9972         0.732            0.007

Spearman rho(w_MF, w_activation) = 0.278 (p = 4.9e-47) over 2,582 regulators

TCR cascade components:
           coef_mean  coef_mean_act            cluster
regulator                                             
CD3D         -0.0081        -0.0126  c22 TCR signaling
CD3E         -0.0097        -0.0109  c22 TCR signaling
CD3G         -0.0074        -0.0103  c22 TCR signaling
CD247        -0.0046        -0.0103  c22 TCR signaling
LCK          -0.0044        -0.0088  c21 TCR signaling
ZAP70        -0.0123        -0.0156  c22 TCR signaling
LAT          -0.0072        -0.0136  c22 TCR signaling
LCP2         -0.0127        -0.0156  c22 TCR signaling
ITK          -0.0011        -0.0060  c21 TCR signaling
PLCG1        -0.0077        -0.0123  c22 TCR signaling
VAV1         -0.0045        -0.0111  c22 TCR signaling
PRKCQ        -0.0021        -0.0035  c99 TCR signaling
CARD11       -0.0034        -0.0062  c21 TCR signaling
BCL10        -0.0021        -0.0037  c21 TCR signaling
MALT1        -0.0010        -0.0022  c21 TCR signaling
PPP3R1       -0.0028        -0.0033  c21 TCR signaling
NFATC2       -0.0016         0.0011  c21 TCR signaling
RASGRP1       0.0003        -0.0028        c35 unknown
IKBKB        -0.0024        -0.0057  c99 TCR signaling
NFKB1         0.0025         0.0014        c30 unknown
RELA          0.0011        -0.0051                NaN
CD28         -0.0003        -0.0038  c21 TCR signaling
ICOS         -0.0010        -0.0032        c27 unknown
PTPRC        -0.0037        -0.0005  c21 TCR signaling
CD2           0.0018        -0.0032  c21 TCR signaling

MF-specific regulators (top-100 |w_MF|, outside the top-500 |w_act|):
           coef_mean  coef_mean_act                             cluster
regulator                                                              
CTBP1        -0.0112        -0.0034               c47 CtBP core complex
PHF20        -0.0109        -0.0008                                 NaN
LARP4        -0.0087        -0.0037                                 NaN
HELLS        -0.0084         0.0012                                 NaN
TFDP1        -0.0077        -0.0021                         c42 unknown
DCUN1D3      -0.0066         0.0011                                 NaN
ZEB1         -0.0065        -0.0029               c47 CtBP core complex
PCGF1        -0.0065         0.0037                                 NaN
B4GALT5      -0.0065        -0.0027                                 NaN
EZH2         -0.0064         0.0003                         c42 unknown
MXI1         -0.0063         0.0018                                 NaN
WDR24        -0.0063        -0.0012           c32 Ragulator-Rag complex
LRP5         -0.0062        -0.0042                   c99 TCR signaling
PEX3         -0.0061         0.0032                                 NaN
NFE2L1       -0.0061        -0.0030                         c64 unknown
GCSAM         0.0061         0.0032                                 NaN
TSC2          0.0062        -0.0008                   c21 TCR signaling
CSTF2         0.0062         0.0009                                 NaN
INO80C        0.0063         0.0024                    c2 Ino80 complex
APOBEC3B      0.0064         0.0015                         c42 unknown
SARNP         0.0064         0.0014                                 NaN
NCOR1         0.0065        -0.0004              c19 NCOR/HDAC3 complex
NR2C2AP       0.0065         0.0001               c33 elongator complex
TNRC6A        0.0065         0.0025                   c99 TCR signaling
PCBP1         0.0066        -0.0008                         c53 unknown
INO80E        0.0066         0.0021                    c2 Ino80 complex
BAZ1A         0.0067         0.0031                         c27 unknown
C11orf68      0.0068         0.0014                        
... [1956 more chars truncated]
```

```python  # cell 20
proj = {}
for label, (cl, cond) in {"cluster 99 TCR signaling (Stim48hr)": (99, "Stim48hr"),
                          "cluster 21 TCR signaling (Stim8hr)": (21, "Stim8hr"),
                          "cluster 22 TCR signaling core (Stim8hr)": (22, "Stim8hr")}.items():
    ds = P.load_downstream(cl, cond)
    proj[(label, "MF signature")] = P.program_projection(y_disc, ds, n_perm=N_PERM, seed=SEED)
    proj[(label, "MF replication")] = P.program_projection(y_repl, ds, n_perm=N_PERM, seed=SEED)
    proj[(label, "TCR activation (Arce)")] = P.program_projection(y_act, ds, n_perm=N_PERM, seed=SEED)
proj = pd.DataFrame(proj).T.rename_axis(["program", "signature"])
proj.to_csv(TAB_DIR / f"pseq29_tcr_program_projection.csv")
print(proj[["n_genes", "pearson_y_vs_kd_sign", "mean_y_kd_up", "mean_y_kd_down", "diff", "p_perm"]].round(3).to_string())

fig, ax = plt.subplots(figsize=(5, 3))
pp = proj.reset_index()
for k, sig in enumerate(["MF signature", "MF replication", "TCR activation (Arce)"]):
    g = pp[pp["signature"] == sig]
    ax.bar(np.arange(len(g)) + k * 0.27 - 0.27, g["pearson_y_vs_kd_sign"].astype(float), width=0.25, label=sig)
ax.set_xticks(np.arange(3)); ax.set_xticklabels([s.replace(" TCR signaling", "") for s in pp["program"].unique()], fontsize=6)
ax.axhline(0, c="k", lw=0.5); ax.legend(fontsize=6, frameon=False)
ax.set_ylabel("Pearson(signature, knockdown direction)", fontsize=7)
mf99 = float(proj.loc[("cluster 99 TCR signaling (Stim48hr)", "MF signature"), "pearson_y_vs_kd_sign"])
ax.set_title(("MF program resembles LOSS of 48 h TCR signalling" if mf99 > 0 else "MF program resembles SUSTAINED 48 h TCR signalling")
             + f" (r = {mf99:.2f}); activation signature is the reference", fontsize=7)
savefig("tcr_program_projection")
```

```
n_genes  pearson_y_vs_kd_sign  mean_y_kd_up  mean_y_kd_down   diff  p_perm
program                                 signature                                                                                        
cluster 99 TCR signaling (Stim48hr)     MF signature            6779.0                -0.371        -1.169           1.816 -2.985   0.001
                                        MF replication          6777.0                -0.371        -0.835           1.218 -2.054   0.001
                                        TCR activation (Arce)   6784.0                -0.437        -1.810           3.488 -5.298   0.001
cluster 21 TCR signaling (Stim8hr)      MF signature            6685.0                -0.369        -1.022           2.126 -3.147   0.001
                                        MF replication          6684.0                -0.388        -0.766           1.509 -2.275   0.001
                                        TCR activation (Arce)   6698.0                -0.370        -1.322           3.338 -4.659   0.001
cluster 22 TCR signaling core (Stim8hr) MF signature            6424.0                -0.327        -1.006           1.736 -2.742   0.001
                                        MF replication          6424.0                -0.333        -0.734           1.181 -1.915   0.001
                                        TCR activation (Arce)   6423.0                -0.387        -1.476           3.438 -4.914   0.001
```

## §4 — inferCNV cross: do the regulators sit on the arms MF cells lose and gain?

Hypothesis: a regulator whose knockdown mimics MF ($w>0$) should be enriched on **recurrently lost** arms —
its loss by deletion would do what the knockdown does — and a regulator whose knockdown opposes MF ($w<0$)
on **gained** arms. Arms from the same GTF the inferCNV pipeline used (`HG38_CENTROMERE` split); donor events
from the nb23 v5 per-donor signatures (`tables/skin_cnv_v5_summary.csv`, callable donors only).

Caveats stated once: the screen is acute CRISPRi knockdown in healthy activated CD4 cells; an arm loss is a
partial dosage change over hundreds of genes in a chronically stimulated tumour cell. "Loss ↔ knockdown" is
the assumption being tested, direction only.

### 4A Cohort level — regulator weights by arm class

```python  # cell 22
gene_arm = P.gene_arms(P.GTF, genes=set(pseq.var_names) | set(X.columns))
events = P.arm_event_table(donors=cells["donor"].unique(), summary_csv=CNV_SUMMARY)
classes = P.arm_classes(events, min_frac=MIN_ARM_FRAC)
print(f"callable donors with arm signatures: {len(events)} of {cells['donor'].nunique()} cohort donors")
print(classes[classes["class"] != "neutral"].round(3).to_string())

coefs["arm"] = gene_arm.reindex(coefs.index)
coefs["arm_class"] = classes["class"].reindex(coefs["arm"].values).fillna("unplaced").to_numpy()
act = P.arm_class_test(coefs, gene_arm, classes, n_perm=N_PERM, seed=SEED)
act.to_csv(TAB_DIR / f"pseq29_arm_class_tests_{CONDITION}.csv", index=False)
print("\n", act.round(4).to_string(index=False))

print("\ntop regulators on recurrently LOST arms (w>0 = deletion would mimic MF):")
lost = coefs[coefs["arm_class"] == "lost"].sort_values("coef_mean", ascending=False)
print(lost.head(15)[["coef_mean", "coef_sem", "arm", "cluster"]].round(4).to_string())
print("\ntop regulators on recurrently GAINED arms (w<0 = extra dosage would feed MF):")
gained = coefs[coefs["arm_class"] == "gained"].sort_values("coef_mean")
print(gained.head(15)[["coef_mean", "coef_sem", "arm", "cluster"]].round(4).to_string())

fig, ax = plt.subplots(figsize=(5, 3))
order_ = ["lost", "neutral", "gained"]
data = [coefs.loc[coefs["arm_class"] == k, "coef_mean"].to_numpy() for k in order_]
ax.boxplot(data, labels=[f"{k}\n(n={len(d)})" for k, d in zip(order_, data)], showfliers=False, widths=0.5)
for i, d in enumerate(data):
    ax.scatter(np.full(len(d), i + 1) + np.random.uniform(-0.15, 0.15, len(d)), d, s=2, alpha=0.3, c="grey", rasterized=True)
ax.axhline(0, ls="--", c="k", lw=0.5); ax.set_ylabel("$w$", fontsize=8); ax.set_xlabel("arm class in the MF cohort", fontsize=7)
p_lost = act.set_index("test").loc["lost vs neutral", "p_perm"]; p_gain = act.set_index("test").loc["gained vs neutral", "p_perm"]
ax.set_title(f"Regulators on lost arms: w {'>' if p_lost < 0.05 else '≈'} neutral (perm p = {p_lost:.3f}); "
             f"gained arms: w {'<' if p_gain < 0.05 else '≈'} neutral (p = {p_gain:.3f})", fontsize=7)
savefig("arm_class_weights")
```

```
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm

callable donors with arm signatures: 64 of 66 cohort donors
        frac_lost  frac_gained  n_donors   class
arm                                             
chr19p      0.094        0.391        64  gained
chr7p       0.047        0.484        64  gained
chr18q      0.016        0.281        64  gained
chr7q       0.000        0.594        64  gained
chr16p      0.000        0.562        64  gained
chr17q      0.000        0.453        64  gained
chr5p       0.609        0.000        64    lost
chr13q      0.453        0.062        64    lost
chr10q      0.453        0.016        64    lost
chr6q       0.422        0.109        64    lost
chr12p      0.375        0.047        64    lost
chr4q       0.344        0.094        64    lost
chr8p       0.328        0.047        64    lost
chr20q      0.312        0.031        64    lost
chr5q       0.297        0.078        64    lost
chr3q       0.297        0.062        64    lost
chr2p       0.266        0.062        64    lost
chr10p      0.250        0.203        64    lost


                                      test  n_set  n_neutral  mean_set  mean_neutral  cliffs_delta   p_mw  p_perm expected
                          lost vs neutral    635       1395    0.0001        0.0000        0.0095 0.3658  0.3866  greater
                        gained vs neutral    452       1395    0.0001        0.0000        0.0102 0.6273  0.7493     less
any event vs neutral (|w|, authors' test)   1087       1395    0.0022        0.0021        0.0126 0.2955  0.1439  greater

top regulators on recurrently LOST arms (w>0 = deletion would mimic MF):
           coef_mean  coef_sem     arm                             cluster
regulator                                                                 
UFM1          0.0114    0.0002  chr13q           c41 protein glycosylation
BTAF1         0.0099    0.0002  chr10q                         c88 unknown
PAN3          0.0098    0.0002  chr13q                         c25 unknown
IRF1          0.0087    0.0003   chr5q                                 NaN
TPT1          0.0086    0.0002  chr13q  c8 peptidyl-diphthamide metabolism
ACAD11        0.0085    0.0001   chr3q           c41 protein glycosylation
UBA5          0.0084    0.0002   chr3q           c41 protein glycosylation
UFL1          0.0078    0.0002   chr6q           c41 protein glycosylation
HNRNPD        0.0076    0.0001   chr4q                         c77 unknown
ZBTB2         0.0073    0.0002   chr6q                                 NaN
TMEM131L      0.0073    0.0003   chr4q           c41 protein glycosylation
PDCD10        0.0072    0.0003   chr3q                         c77 unknown
CBLB          0.0070    0.0002   chr3q                        c100 unknown
ATP6V1E2      0.0069    0.0002   chr2p                                 NaN
TNPO1         0.0068    0.0002   chr5q                         c88 unknown

top regulators on recurrently GAINED arms (w<0 = extra dosage would feed MF):
           coef_mean  coef_sem     arm                                    cluster
regulator                                                                        
HEXIM1       -0.0088    0.0003  chr17q  c58 ribosomal and mitochrondrial function
SMARCD3      -0.0085    0.0002   chr7q                                        NaN
LAT          -0.0072    0.0002  chr16p                          c22 TCR signaling
XPO6         -0.0071    0.0001  chr16p                                        NaN
DCUN1D3      -0.0066    0.0001  chr16p                                        NaN
EZH2         -0.0064    0.0001   chr7q                                c42 unknown
CALCOCO2     -0.0064    0.0002  chr17q                          c21 TCR signaling
W
... [973 more chars truncated]
```

### 4B Donor level — does a donor's own deletion set reconstruct its own program?

The authors' `get_prediction(X, subset_regulators=…)` reconstructs the signature from a chosen set of
regulators. For every callable donor we reconstruct from the regulators on **that donor's lost arms**
(and, separately, its gained arms) and correlate with the donor's own log2FC; the null is random regulator
sets of the same size. The summary is the share of donors whose own deletion set beats its null.

```python  # cell 24
dpp = P.dosage_predicted_program(m_real, gene_arm, events, Y_donor, n_null=200, seed=SEED)
dpp = dpp.merge(dtab[["study", "split"]], left_on="donor", right_index=True, how="left")
dpp.to_csv(TAB_DIR / f"pseq29_dosage_predicted_program_{CONDITION}.csv", index=False)
ok = dpp.dropna(subset=["r_obs"])
for cls in ("lost", "gained"):
    g = ok[ok["class"] == cls]
    if len(g):
        print(f"{cls} arms: {len(g)} donors; own-set r beats null (p<0.05) in {(g['p_null'] < 0.05).sum()}; "
              f"median r_obs = {g['r_obs'].median():.3f} vs null {g['null_mean'].median():.3f}; "
              f"paired Wilcoxon p = {ss.wilcoxon(g['r_obs'], g['null_mean'], alternative='greater').pvalue:.3g}")
print(ok.sort_values(["class", "p_null"]).round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(5, 3))
for k, cls in enumerate(("lost", "gained")):
    g = ok[ok["class"] == cls]
    ax.scatter(np.full(len(g), k) - 0.15 + np.random.uniform(-0.05, 0.05, len(g)), g["null_mean"], s=8, c="grey", label="random sets" if k == 0 else None)
    ax.scatter(np.full(len(g), k) + 0.15 + np.random.uniform(-0.05, 0.05, len(g)), g["r_obs"], s=8, c="firebrick", label="donor's own arms" if k == 0 else None)
    for _, row in g.iterrows():
        ax.plot([k - 0.15, k + 0.15], [row["null_mean"], row["r_obs"]], c="lightgrey", lw=0.4, zorder=0)
ax.set_xticks([0, 1]); ax.set_xticklabels(["regulators on lost arms", "regulators on gained arms"], fontsize=7)
ax.set_ylabel("r(reconstruction, donor log2FC)", fontsize=7); ax.legend(fontsize=6, frameon=False)
n_beat = int((ok.loc[ok["class"] == "lost", "p_null"] < 0.05).sum()); n_tot = int((ok["class"] == "lost").sum())
ax.set_title(f"A donor's own deleted-arm regulators reconstruct its program better than random in {n_beat}/{n_tot} donors", fontsize=7)
savefig("dosage_predicted_program")
```

```
lost arms: 64 donors; own-set r beats null (p<0.05) in 7; median r_obs = 0.244 vs null 0.229; paired Wilcoxon p = 0.00631
gained arms: 63 donors; own-set r beats null (p<0.05) in 0; median r_obs = 0.231 vs null 0.236; paired Wilcoxon p = 0.964
               donor  class  n_arms  n_regulators  r_obs  null_mean  null_sd  p_null           study       split
 Li2024_atlas__CTCL2 gained       5           381  0.227      0.202    0.017   0.060          li2024   discovery
           D13__MF21 gained       7           415  0.411      0.375    0.028   0.075    gaydosik2022   discovery
            D8__P138 gained       1            45  0.186      0.133    0.041   0.139       alkon2024   discovery
  Li2024_atlas__MF30 gained       6           502  0.310      0.297    0.014   0.189          li2024 replication
Li2024_atlas__CTCL14 gained       7           482  0.346      0.325    0.026   0.219          li2024 replication
            D8__P163 gained       6           443  0.242      0.232    0.013   0.219       alkon2024   discovery
 Li2024_atlas__CTCL4 gained       3           204  0.148      0.129    0.023   0.224          li2024   discovery
             D11__P1 gained       8           430  0.206      0.195    0.016   0.234 brentuximab2026   discovery
           D15__MF48 gained       7           355  0.271      0.259    0.016   0.239       il4ra2026 replication
             D11__P2 gained       9           586  0.326      0.319    0.010   0.254 brentuximab2026   discovery
           D6__SC157 gained       4           263  0.207      0.194    0.022   0.289    gaydosik2019   discovery
             D7__P84 gained       2           221  0.278      0.261    0.029   0.313  rindler2021_mc replication
           D13__MF19 gained       7           456  0.245      0.234    0.020   0.323    gaydosik2022   discovery
  Li2024_atlas__PT35 gained       7           591  0.205      0.198    0.014   0.338          li2024   discovery
             D1__PGS gained       9           706  0.220      0.216    0.011   0.348 chennareddy2025   discovery
            D1__P312 gained       9           436  0.132      0.128    0.012   0.368 chennareddy2025   discovery
 Li2024_atlas__CTCL6 gained       3           204  0.211      0.205    0.028   0.403          li2024   discovery
 Li2024_atlas__CTCL7 gained       3           293  0.124      0.121    0.034   0.408          li2024   discovery
           D6__SC158 gained       8           480  0.270      0.265    0.022   0.423    gaydosik2019   discovery
 Li2024_atlas__CTCL9 gained       7           520  0.372      0.369    0.015   0.448          li2024   discovery
             D11__P5 gained       8           424  0.193      0.189    0.016   0.463 brentuximab2026   discovery
             D11__P4 gained       5           125  0.215      0.205    0.040   0.463 brentuximab2026 replication
         D10__JONAK1 gained       2            74  0.244      0.236    0.050   0.478       jonak2021   discovery
           D13__MF25 gained       4           388  0.181      0.180    0.015   0.498    gaydosik2022   discovery
           D15__MF36 gained       6           257  0.375      0.371    0.028   0.507       il4ra2026   discovery
Li2024_atlas__CTCL10 gained       8           597  0.282      0.283    0.011   0.522          li2024   discovery
            D1__P303 gained       7           554  0.231      0.231    0.012   0.527 chennareddy2025 replication
  Li2024_atlas__MF22 gained       6           521  0.220      0.220    0.013   0.537          li2024 replication
  Li2024_atlas__PT52 gained       7           499  0.281      0.279    0.024   0.542          li2024   discovery
           D15__MF35 gained       7           572  0.364      0.366    0.011   0.557       il4ra2026   discovery
             D8__P90 gained       5           338  0.264      0.268    0.022   0.622       alkon2024 replication
           D13__MF17 gained       6           366  0.235      0.244    0.025   0.627    gaydosik2022   discovery
 Li2024_atlas__CTCL1 gained
... [10707 more chars truncated]
```

### 4C Per cell — knockdown-mimicry score vs the cell's own arm dosage

For the strongest regulators on event arms (and the strongest overall), a knockdown-mimicry score per
malignant cell: the authors' Fig 4G construction (top 50 up / 50 down genes of the knockdown at FDR < 0.1,
each gene standardized against reference cells — here the same donor's **reactive CD4** instead of NTC cells),
with **genes on the regulator's own arm removed** so the score shares no genes with the arm's inferCNV value.
Within each donor, Spearman ρ between the score and the per-cell arm z (`arm_z` against the held-out diploid
null); Stouffer across donors; the same test on donors **without** an event on that arm is the control.
Expected: deeper loss of the regulator's arm ↔ higher mimicry score for $w>0$ regulators.

> **HEAVY:** reads the cached CD4 object (§2) and the 1.2 M-row arm matrix; caches the scores.

```python  # cell 26
on_event = coefs.index[coefs["arm_class"].isin(["lost", "gained"])]
sel = coefs.loc[on_event, "coef_mean"].abs().sort_values(ascending=False).index[:TOP_K_CELL].tolist()
sel += [g for g in coefs.index[:5].tolist() + coefs.index[-5:].tolist() if g not in sel and g in gene_arm.index]
sets = P.kd_gene_sets(pseq, sel, gene_arm, n=50, fdr=0.1, drop_cis_arm=True)
sets = {k: v for k, v in sets.items() if len(v["up"]) + len(v["down"]) >= 10}
print(f"{len(sets)} regulators scored; gene-set sizes:", {k: (len(v["up"]), len(v["down"])) for k, v in list(sets.items())[:8]}, "...")

if SCORES_PQ.exists():
    scores = pd.read_parquet(SCORES_PQ)
    print("loaded cached scores", scores.shape)
else:
    if not os.environ.get("LSB_JOBID"):
        print("⚠️  no LSB_JOBID — this cell loads the cached CD4 object (several GB); run it on the compute kernel.")
    a = sc.read_h5ad(CD4_H5)
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)      # library size over all 42k genes, as nb24
    need = sorted({g for v in sets.values() for g in v["up"] + v["down"]} & set(a.var_names))
    expr = pd.DataFrame(a[:, need].X.toarray(), index=a.obs_names, columns=need)
    scores = P.kd_scores(expr, a.obs["donor"].astype(str), a.obs["arm"].astype(str).eq("reactive"), sets)
    scores["donor"] = a.obs["donor"].astype(str).to_numpy(); scores["arm_label"] = a.obs["arm"].astype(str).to_numpy()
    scores.to_parquet(SCORES_PQ)
    del a, expr; gc.collect()

arm_raw = pd.read_parquet(ARM).set_index("obs_name")
query, null = S.split_null_rows(arm_raw)
arm_cols = [c for c in query.columns if c.startswith("chr")]
nstats = S.arm_null_stats(null, arm_cols)
mal_cells = scores.index[scores["arm_label"] == "malignant"].intersection(query.index)
armz = S.arm_z(query.loc[mal_cells], nstats, arm_cols)
del arm_raw, query, null; gc.collect()
print(f"malignant cells with both a score and an arm profile: {len(mal_cells):,}")

long, summ_cell = P.per_cell_arm_test(scores.loc[mal_cells, list(sets)], armz, scores.loc[mal_cells, "donor"],
                                      gene_arm, events, min_cells=MIN_CELLS_ARM)
long.to_csv(TAB_DIR / f"pseq29_per_cell_arm_tests_long_{CONDITION}.csv", index=False)
summ_cell = summ_cell.merge(coefs[["coef_mean", "arm_class"]], left_on="regulator", right_index=True, how="left")
summ_cell.to_csv(TAB_DIR / f"pseq29_per_cell_arm_tests_{CONDITION}.csv", index=False)
print(summ_cell.round(4).to_string(index=False))
```

```
27 regulators scored; gene-set sizes: {'LCP2': (50, 50), 'UFM1': (50, 50), 'PHF20': (25, 50), 'SIN3B': (50, 50), 'BTAF1': (50, 50), 'PAN3': (50, 50), 'FITM2': (50, 50), 'HEXIM1': (50, 50)} ...

loaded cached scores (286748, 29)

malignant cells with both a score and an arm profile: 166,588

regulator    arm event_class  n_donors  mean_rho  median_rho  stouffer_z  p_stouffer  q_stouffer  coef_mean arm_class
     UFL1  chr6q        lost        27   -0.1852     -0.1812        -inf      0.0000      0.0000     0.0078      lost
     UFL1  chr6q    no event        32   -0.2047     -0.1678    -43.6918      0.0000      0.0000     0.0078      lost
     UFM1 chr13q        lost        29   -0.1587     -0.1699    -42.3485      0.0000      0.0000     0.0114      lost
     UBA5  chr3q        lost        19   -0.1296     -0.1013        -inf      0.0000      0.0000     0.0084      lost
   ACAD11  chr3q        lost        19   -0.1317     -0.1182    -33.7591      0.0000      0.0000     0.0085      lost
     UFL1  chr6q      gained         7   -0.3133     -0.3254    -30.5294      0.0000      0.0000     0.0078      lost
    PLCG1 chr20q    no event        44   -0.0907     -0.0789    -30.4079      0.0000      0.0000    -0.0077      lost
   DDRGK1 chr20p    no event        54    0.0851      0.0860     30.2564      0.0000      0.0000     0.0105   neutral
    ZAP70  chr2q    no event        57    0.0830      0.0729     26.6803      0.0000      0.0000    -0.0123   neutral
     PAN3 chr13q        lost        29   -0.1117     -0.0858    -26.6653      0.0000      0.0000     0.0098      lost
   IMMP2L  chr7q      gained        38    0.0746      0.0742     25.7456      0.0000      0.0000     0.0081    gained
     LCP2  chr5q        lost        19    0.1430      0.1292     25.6028      0.0000      0.0000    -0.0127      lost
    BTAF1 chr10q        lost        29   -0.1151     -0.0930    -25.4015      0.0000      0.0000     0.0099      lost
    KDM1A  chr1p    no event        60   -0.0671     -0.0663    -20.5824      0.0000      0.0000    -0.0126   neutral
     UBA5  chr3q    no event        43   -0.0764     -0.0742    -20.1650      0.0000      0.0000     0.0084      lost
    BTAF1 chr10q    no event        36   -0.0620     -0.0509    -18.0028      0.0000      0.0000     0.0099      lost
    BRMS1 chr11q    no event        53   -0.0387     -0.0311    -17.1218      0.0000      0.0000     0.0110   neutral
     PAN3 chr13q    no event        33   -0.0719     -0.0732    -17.0208      0.0000      0.0000     0.0098      lost
   ACAD11  chr3q    no event        43   -0.0631     -0.0583    -16.5628      0.0000      0.0000     0.0085      lost
   INTS12  chr4q        lost        22    0.0712      0.0839     16.3132      0.0000      0.0000    -0.0086      lost
     IRF1  chr5q    no event        42    0.0533      0.0432     15.7106      0.0000      0.0000     0.0087      lost
     UFM1 chr13q    no event        33   -0.0794     -0.0700    -15.5107      0.0000      0.0000     0.0114      lost
    SIN3B chr19p      gained        25   -0.0615     -0.0647    -15.4695      0.0000      0.0000     0.0101    gained
    ZAP70  chr2q      gained         6    0.1615      0.1842     15.4403      0.0000      0.0000    -0.0123   neutral
     LCP2  chr5q    no event        42    0.0781      0.0845     14.5042      0.0000      0.0000    -0.0127      lost
   SAP130  chr2q    no event        57   -0.0424     -0.0441    -13.8643      0.0000      0.0000     0.0113   neutral
    FITM2 chr20q    no event        44   -0.0350     -0.0275    -12.9475      0.0000      0.0000    -0.0093      lost
    PLCG1 chr20q        lost        20   -0.1209     -0.0916    -12.7925      0.0000      0.0000    -0.0077      lost
     TPT1 chr13q        lost        29   -0.0505     -0.0534    -12.6485      0.0000      0.0000     0.0086      lost
    KDM1A  chr1p      gained         2   -0.1739     -0.1739    -11.6704      0.0000      0.0000    -0.0126   neutral
    RCOR1 chr14q      gained         1    0.1118  
... [5377 more chars truncated]
```

```python  # cell 27
# forest: per-donor rho for each regulator, event donors vs no-event donors
show_regs = summ_cell[summ_cell["event_class"] != "no event"].sort_values("p_stouffer")["regulator"].unique()[:TOP_K_CELL]
fig, ax = plt.subplots(figsize=(5, 0.28 * len(show_regs) + 1))
for i, r_ in enumerate(show_regs[::-1]):
    g = long[long["regulator"] == r_]
    for ev, col, dx in ((0, "lightgrey", -0.12), (-1, "navy", 0.12), (1, "firebrick", 0.12)):
        gg = g[g["event"] == ev]
        ax.scatter(gg["rho"], np.full(len(gg), i + dx), s=10, c=col, alpha=0.8)
    arm_ = g["arm"].iloc[0] if len(g) else ""
    ax.text(ax.get_xlim()[1] if i == 0 else 0.62, i, f"{arm_}  w={coefs.loc[r_, 'coef_mean']:+.3f}", fontsize=5, va="center")
ax.set_yticks(range(len(show_regs))); ax.set_yticklabels(show_regs[::-1], fontsize=6)
ax.axvline(0, c="k", lw=0.5); ax.set_xlim(-0.6, 0.9)
ax.set_xlabel("Spearman rho(knockdown-mimicry score, arm z) within donor   grey = no event, blue = arm lost, red = arm gained", fontsize=6)
n_sig = int((summ_cell[(summ_cell["event_class"] != "no event")]["q_stouffer"] < 0.1).sum())
ax.set_title(f"Per-cell arm dosage tracks knockdown mimicry for {n_sig} regulator-arm pairs (Stouffer q<0.1) in event donors", fontsize=7)
savefig("per_cell_forest")

# one example scatter: the most significant event pair
if len(summ_cell[summ_cell["event_class"] != "no event"]):
    best = summ_cell[summ_cell["event_class"] != "no event"].iloc[0]
    g = long[(long["regulator"] == best["regulator"]) & (long["event"] != 0)].sort_values("p").iloc[0]
    cells_d = scores.index[(scores["donor"] == g["donor"]) & (scores["arm_label"] == "malignant")].intersection(armz.index)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(armz.loc[cells_d, g["arm"]], scores.loc[cells_d, g["regulator"]], s=3, alpha=0.4, rasterized=True)
    ax.set_xlabel(f"{g['arm']} inferCNV z (per cell, {g['donor']})", fontsize=7); ax.set_ylabel(f"{g['regulator']}-knockdown mimicry score", fontsize=7)
    ax.set_title(f"{g['regulator']} ({g['arm']} {'lost' if g['event'] < 0 else 'gained'}): rho = {g['rho']:.2f}, p = {g['p']:.1e}, n = {g['n_cells']:,} cells", fontsize=7)
    savefig("per_cell_example")
```

## §5 — Summary table and numerical export

```python  # cell 29
summary = coefs[["coef_mean", "coef_sem", "coef_rank", "abs_rank", "cluster", "arm", "arm_class"]].copy()
summary["w_activation"] = w_act["coef_mean"].reindex(summary.index)
summary["control_panel"] = [next((k for k, v in P.CONTROL_PANEL.items() if g in v), "") for g in summary.index]
pc = summ_cell[summ_cell["event_class"] != "no event"].groupby("regulator").agg(per_cell_stouffer_z=("stouffer_z", "first"), per_cell_q=("q_stouffer", "first"), per_cell_n_donors=("n_donors", "first"))
summary = summary.join(pc)
summary.to_csv(TAB_DIR / f"pseq29_summary_{CONDITION}.csv")

LLM_OPEN = "=" * 30 + " BEGIN pseq29 numerical result (paste as one block) " + "=" * 30
LLM_CLOSE = "=" * 30 + " END pseq29 " + "=" * 30
L = [LLM_OPEN, "", f"# 29_perturbseq_reverse_engineering — MF malignant CD4 (skin) reverse-engineered with GWCD4i {CONDITION} knockdowns, crossed with inferCNV", "",
     "## Context", f"Cohort: {len(dtab)} CD4-malignant MF skin donors ({(dtab['split']=='discovery').sum()} discovery / {(dtab['split']=='replication').sum()} replication), "
     f"{len(cells):,} CD4 cells (malignant vs same-donor reactive, call = {CALL}). Signature = pydeseq2 Wald z, ~ donor + arm. "
     f"Model = authors' pert2state (SVD D={N_PCS}, elastic net, {N_SPLITS}-fold x {N_REPEATS}); X = {X.shape[0]:,} genes x {X.shape[1]:,} regulators. "
     "w>0: knockdown mimics MF; w<0: knockdown opposes MF.", "",
     "## Model fit", summ[["n_genes", "n_regulators", "test_pearson", "test_pearson_se", "train_pearson"]].round(3).to_csv(), "",
     "## Replication (held-out genes vs replication donors)", repl.round(3).describe().loc[["mean", "std"]].to_csv(), "",
     "## Control panel", ctrl.round(4).to_csv(index=False), "",
     "## Top 25 w>0 (knockdown mimics MF)", coefs.head(25)[["coef_mean", "coef_sem", "cluster", "arm", "arm_class"]].round(4).to_csv(), "",
     "## Top 25 w<0 (knockdown opposes MF)", coefs.tail(25)[["coef_mean", "coef_sem", "cluster", "arm", "arm_class"]].round(4).to_csv(), "",
     "## Pathway tests (q<0.1, any column)", ptab[(ptab[["q_abs_greater", "q_signed_greater", "q_signed_less"]] < 0.1).any(axis=1)]
        [["set", "n", "mean_w", "q_abs_greater", "q_signed_greater", "q_signed_less"]].round(4).to_csv(index=False), "",
     "## TCR-engagement axis", f"Spearman rho(w_MF, w_activation) = {rho:.3f}", proj[["n_genes", "pearson_y_vs_kd_sign", "diff", "p_perm"]].round(3).to_csv(), "",
     "## 4A arm-class tests", act.round(4).to_csv(index=False), "",
     "## 4B dosage-predicted program (per donor)", ok.round(3).to_csv(index=False), "",
     "## 4C per-cell tests (regulator x arm x event class)", summ_cell.round(4).to_csv(index=False), "",
     LLM_CLOSE]
text = "\n".join(L)
(TAB_DIR / "pseq29_llm_export.md").write_text(text)
print(text)
print(f"\n[{len(text):,} characters; also at {TAB_DIR / 'pseq29_llm_export.md'}]")
```

```
============================== BEGIN pseq29 numerical result (paste as one block) ==============================

# 29_perturbseq_reverse_engineering — MF malignant CD4 (skin) reverse-engineered with GWCD4i Stim48hr knockdowns, crossed with inferCNV

## Context
Cohort: 66 CD4-malignant MF skin donors (52 discovery / 14 replication), 286,748 CD4 cells (malignant vs same-donor reactive, call = mal_cluster). Signature = pydeseq2 Wald z, ~ donor + arm. Model = authors' pert2state (SVD D=60, elastic net, 5-fold x 3); X = 10,096 genes x 2,582 regulators. w>0: knockdown mimics MF; w<0: knockdown opposes MF.

## Model fit
model,n_genes,n_regulators,test_pearson,test_pearson_se,train_pearson
MF Stim48hr,10096,2582,0.611,0.004,0.619
scrambled,10096,2582,0.263,0.006,0.265


## Replication (held-out genes vs replication donors)
,split,n_test_genes,correlation,upper_correlation,pred_input_correlation
mean,7.0,2019.2,0.5871999999999999,0.8780666666666668,0.6115333333333334
std,4.47213595499958,0.41403933560541256,0.016823453017397214,0.005049280948336916,0.015495928804812249


## Control panel
group,regulator,coef_mean,coef_sem,coef_rank,abs_rank,sign_ok
LoF in CTCL (expect w>0),PTEN,-0.0032,0.0003,0.8896,581.0,False
LoF in CTCL (expect w>0),TNFAIP3,,,,,
LoF in CTCL (expect w>0),ZEB1,-0.0065,0.0002,0.9888,77.0,False
LoF in CTCL (expect w>0),CDKN2A,,,,,
LoF in CTCL (expect w>0),TP53,-0.0036,0.0002,0.912,461.0,False
LoF in CTCL (expect w>0),ARID1A,0.0001,0.0001,0.5025,2527.0,True
LoF in CTCL (expect w>0),DNMT3A,,,,,
LoF in CTCL (expect w>0),TET2,0.002,0.0001,0.2216,1119.0,True
LoF in CTCL (expect w>0),KMT2D,,,,,
LoF in CTCL (expect w>0),CD58,,,,,
LoF in CTCL (expect w>0),FAS,,,,,
LoF in CTCL (expect w>0),NFKBIA,,,,,
LoF in CTCL (expect w>0),RB1,,,,,
LoF in CTCL (expect w>0),CYLD,,,,,
LoF in CTCL (expect w>0),PRDM1,,,,,
GoF / dependency (expect w<0),PLCG1,-0.0077,0.0002,0.993,43.0,True
GoF / dependency (expect w<0),CARD11,-0.0034,0.0001,0.9031,511.0,True
GoF / dependency (expect w<0),STAT3,-0.001,0.0002,0.6559,1831.0,True
GoF / dependency (expect w<0),STAT5B,0.0,0.0001,0.5095,2566.0,False
GoF / dependency (expect w<0),JAK3,,,,,
GoF / dependency (expect w<0),JAK1,,,,,
GoF / dependency (expect w<0),CD28,-0.0003,0.0001,0.5548,2338.0,True
GoF / dependency (expect w<0),NFKB2,0.0023,0.0002,0.1906,971.0,False
GoF / dependency (expect w<0),MYC,0.003,0.0002,0.1286,649.0,False
GoF / dependency (expect w<0),TOX,,,,,
GoF / dependency (expect w<0),GATA3,0.0038,0.0002,0.0759,394.0,False
GoF / dependency (expect w<0),IRF4,-0.0004,0.0001,0.5684,2272.0,True
GoF / dependency (expect w<0),BATF,0.0016,0.0002,0.277,1410.0,False
GoF / dependency (expect w<0),TNFRSF1B,,,,,


## Top 25 w>0 (knockdown mimics MF)
regulator,coef_mean,coef_sem,cluster,arm,arm_class
UFM1,0.0114,0.0002,c41 protein glycosylation,chr13q,lost
SAP130,0.0113,0.0002,c4 unknown,chr2q,neutral
BRMS1,0.011,0.0003,c4 unknown,chr11q,neutral
DDRGK1,0.0105,0.0002,c41 protein glycosylation,chr20p,neutral
SIN3B,0.0101,0.0002,c6 unknown,chr19p,gained
HDAC1,0.0101,0.0002,c60 unknown,chr1p,neutral
BTAF1,0.0099,0.0002,c88 unknown,chr10q,lost
RO60,0.0098,0.0003,c2 Ino80 complex,chr1q,neutral
PAN3,0.0098,0.0002,c25 unknown,chr13q,lost
KMT2B,0.0097,0.0003,,chr19q,neutral
GCN1,0.009,0.0002,c77 unknown,chr12q,neutral
IRF1,0.0087,0.0003,,chr5q,lost
TPT1,0.0086,0.0002,c8 peptidyl-diphthamide metabolism,chr13q,lost
TYK2,0.0085,0.0002,,chr19p,gained
ACAD11,0.0085,1e-04,c41 protein glycosylation,chr3q,lost
UBA5,0.0084,0.0002,c41 protein glycosylation,chr3q,lost
ZC3H10,0.0084,0.0002,c80 unknown,chr12q,neutral
NFRKB,0.0084,0.0004,c2 Ino80 complex,chr11q,neutral
POLR2M,0.0083,0.0003,,chr15q,neutral
IMMP2L,0.0081,1e-04,c96 unknown,chr7q,gained
IRF9,0.0081,0.0004,,chr14q,neutral
UBAP2L,0.0079,0.0002,c76 EMC complex,chr1q,neutral
MED24,0.0079,0.0002,c10 mediator-SAGA,chr17q,gained
UFL1,0.0078,0.0002,c41 protein glycosylation,chr6q,lost
KAT7,0.0077,0.0002,c30 unknown,chr17q,gained


## Top 25 w<0 (knockdown opposes MF)
regula
... [18078 more chars truncated]
```

## §6 — Full numerical report + methods narrative

Prints every number the notebook produced (cohort, resource QC, signature, model fit, replication, control panel, regulator weights, pathway tests, TCR axis, inferCNV cross at cohort / donor / cell level) preceded by a plain-language account of what was done, which data went in and how each quantity was computed. Each section is guarded, so a partial run prints the error for that block and continues. Also saved to `tables/pseq29_full_report_<CONDITION>.txt`.

```python  # cell 31
# ============================================================
# §6  Full report: every number the notebook produced + what was done, on what data, and how
# ============================================================
# Each section is wrapped so a partial run (missing variable) prints the error and continues.
import io as _io
_buf = _io.StringIO()
def _p(*a, **k):
    print(*a, **k); print(*a, **k, file=_buf)
def _hdr(t): _p("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)
def _sub(t): _p("\n" + "-" * 100 + f"\n{t}\n" + "-" * 100)
def _sec(title, fn):
    _sub(title)
    try: fn()
    except Exception as e: _p(f"[not available in this run: {type(e).__name__}: {e}]")
def _ts(df, **k): return df.to_string(**k)

_hdr(f"29_perturbseq_reverse_engineering — FULL REPORT ({CONDITION})")

# ---------------------------------------------------------------- 0. narrative
_p("""
WHAT THIS NOTEBOOK DOES
-----------------------
Goal: take the transcriptional program that separates MALIGNANT CD4 T cells from REACTIVE (non-malignant) CD4 T cells
in MF skin, and ask which single-gene CRISPRi knockdowns in a normal, TCR-stimulated CD4 T cell reproduce or oppose it.
This "reverse-engineers" the malignant state into candidate upstream regulators, groups them into pathways, places the
MF program on the TCR-engagement axis, and finally asks whether the regulators lie on the chromosome arms that MF cells
recurrently lose or gain (inferCNV), at cohort, donor, and single-cell resolution.

DATA USED
---------
1. Perturb-seq resource (external): Zhu, Dann et al., Cell 2026 (GWCD4i). CRISPRi (ZIM3-dCas9) knockdown of 12,748
   genes in primary human CD4+ T cells from 4 donors, three culture conditions. Only the condition "Stim48hr"
   (48 h after CD3/CD28/CD2 restimulation) and only the PUBLISHED per-perturbation DESeq2 summary statistics
   (knockdown vs non-targeting control: log2FC, lfcSE, Wald z, padj per measured gene) are used
   (data/external_ref/perturbseq_gwcd4i/GWCD4i_DE_Stim48hr.h5ad, built by jobs/run_perturbseq_prep.py).
   No cell-level Perturb-seq data were downloaded. Also from the paper: the 111 functional regulator clusters
   (Suppl. Table 13; cluster 99 = Stim48hr-specific "TCR signaling") and each cluster's downstream target genes.
2. MF skin T-cell atlas (this project, atlas v2): data/atlas_joint/skin_T_tcr_annotated_v4.h5ad (nb21 labels + V(D)J
   fold-in, raw counts). Cells restricted to cell_type_T == "CD4", disease == "MF".
3. Malignancy calls: helpers/malignancy.py, headline call `mal_cluster` (malignant = mal_cluster & defined;
   reactive = defined & ~mal_cluster, same donor).
4. Donor truth: helpers/target_cohort.skin_donor_table() — donors kept only if the malignant clone is CD4 (not CD8,
   DN, gamma-delta), diagnosis MF, not a duplicate of another donor key, not healthy control.
5. inferCNV (nb23, revision v5): per-cell 41-arm CNV matrix with diploid null rows
   (data/atlas_joint/skin_T_arm_cnv_v5.parquet) and per-donor signed arm signatures (tables/skin_cnv_v5_summary.csv).
6. Comparator signature: Arce 2024 TCR-activation signature (effector T stimulated vs resting), the authors' own
   negative control for polarization.
7. Gene-to-arm map: the same GTF the inferCNV pipeline used, split at hg38 centromeres.

CALCULATIONS, IN ORDER
----------------------
§0 Cohort. CD4 MF donors with >= MIN_CELLS_ARM malignant AND >= MIN_CELLS_ARM reactive CD4 cells. Random 80/20
   discovery/replication split of DONORS, stratified by study (same design as the authors' OneK1K split).
§1 Resource QC. Regulator filter = perturbations with significant effects on >= 10 measured genes (authors' model
   filter); gene filter = mean baseMean > 0.1 in every condition.
§2 MF signature (the response vector y). One pseudobulk per donor x arm (raw counts summed), pydeseq2 with design
   ~ donor + arm, contrast malignant vs reactive, run separately on discovery and replication donors. y = Wald
   z-score (log2FC / lfcSE) per gene, then centred/standardized by the authors' prepare_y. Per-donor log2FC on CP10K
   pseudobulks (log2((mal+1)/(react+1))) is kept for §4B. Pearson r between discovery and replication z = the ceiling
   any reconstruction can reach.
§3a pert2state model (authors' code, vendored verbatim in helpers/pert2state_model.py). X = genes x regulators matrix
   of Perturb-seq z-scores with the on-target entry masked (column-mean imputed); per-regulator standardization ->
   TruncatedSVD(D = 60) -> ElasticNetCV (inner 4-fold; alpha logspace(-3,3,30); l1_ratio logspace(-2,0,10)).
   Outer 5-fold CV over GENES, repeated 3x (15 fits). Performance = Pearson r(predicted, observed z) on held-out genes.
   Scrambled null = each gene's z permuted across regulators. Replication = held-out-gene predictions correlated with
   the replication donors' signature. Regulator weight w_k = mean coefficient over the 15 fits (+/- SEM), mapped back
   from SVD space. Sign convention: w > 0 -> knocking down k pushes a normal CD4 cell TOWARD the MF state
   (loss-of-function-like); w < 0 -> k's activity is REQUIRED for the MF state (dependency-like).
§3b Control panel. CTCL genes known to be lost/mutated (expect w > 0) and gained/GoF/dependencies (expect w < 0);
   sign agreement tested with a one-sided binomial test; rank of |w| among all regulators reported.
§3c Pathways. For each authors' cluster and each TCR module (tcr_signaling_helpers.TCR_MODULES) with >= 5 members:
   authors' unsigned test (one-sided Mann-Whitney, members' |w| > rest) plus two SIGNED tests (members' w > rest =
   knockdown mimics MF; members' w < rest = MF depends on the pathway). BH-FDR across sets.
§3d TCR-engagement axis. (1) Same model fitted to the Arce activation signature -> w_act; Spearman rho(w_MF, w_act);
   TCR-cascade components listed; "MF-specific" = top-100 |w_MF| but outside top-500 |w_act|. (2) Projection of each
   signature on the sign-coherent downstream program of TCR-signalling clusters 99 (Stim48hr), 21 and 22 (Stim8hr):
   Pearson r between the signature z and the knockdown direction, mean z of genes that go UP vs DOWN when TCR
   regulators are knocked down, difference, and a permutation p (N_PERM). Positive r = MF resembles LOSS of TCR
   signalling; negative = MF resembles SUSTAINED engagement. The activation signature is the reference.
§4A Cohort-level arm test. Arms classed "lost"/"gained" if the event recurs in >= MIN_ARM_FRAC of callable donors.
   Regulators on lost arms vs neutral arms (expect w greater), gained vs neutral (expect w less), any event vs neutral
   on |w| (authors' test): Mann-Whitney p, Cliff's delta, permutation p.
§4B Donor-level reconstruction. For each callable donor, the signature is reconstructed from ONLY the regulators on
   that donor's own lost arms (separately: gained arms) using the authors' get_prediction(subset_regulators=...), and
   correlated with the donor's own log2FC; null = 200 random regulator sets of the same size. Summary = share of
   donors whose own set beats its null (p < 0.05) and a paired Wilcoxon of r_obs vs null mean across donors.
§4C Per-cell test. For the strongest regulators on event arms (+ top/bottom 5 overall): knockdown-mimicry score per
   malignant cell (authors' Fig 4G construction: top 50 up / 50 down knockdown genes at FDR < 0.1, each gene
   standardized against the same donor's REACTIVE CD4 cells, genes on the regulator's own arm REMOVED). Within each
   donor, Spearman rho(score, per-cell arm z against the held-out diploid null); Stouffer combination across donors,
   separately for donors WITH the event on that arm and donors WITHOUT it (control). BH q over pairs.

DEVIATIONS FROM THE AUTHORS (deliberate): l1_ratio grid capped at 1 (sklearn >= 1.2); replication = within-cohort
donor split rather than an independent cohort; signed pathway tests added; Arce activation signature used as a
comparator as well as a negative control. §4 (inferCNV cross) is this project's extension, not in the paper.
""")

# ---------------------------------------------------------------- 1. parameters
def _params():
    for k in ["CONDITION", "CALL", "DISEASE", "SEED", "MIN_CELLS_ARM", "REPL_FRAC", "MIN_DE_GENES", "BASEMEAN_MIN",
              "N_PCS", "N_SPLITS", "N_REPEATS", "MIN_ARM_FRAC", "TOP_K_CELL", "N_PERM"]:
        _p(f"{k:14s} = {globals()[k]}")
    for k in ["PSEQ_H5", "OBJ", "ARM", "CNV_SUMMARY", "CELLS_PQ", "CD4_H5", "DE_DISC", "DE_REPL", "DONOR_LFC", "SCORES_PQ"]:
        _p(f"{k:14s} = {Path(globals()[k]).relative_to(NB_DIR)}")
_sec("PARAMETERS AND INPUT FILES", _params)

# ---------------------------------------------------------------- 2. cohort
def _cohort():
    _p(f"skin T object obs rows: {len(obs):,}" if "obs" in globals() else "")
    _p(f"eligible donors: {len(dtab)} (both arms >= {MIN_CELLS_ARM} CD4 cells); cells: {len(cells):,}")
    _p(f"malignant cells: {(cells['arm']=='malignant').sum():,}; reactive cells: {(cells['arm']=='reactive').sum():,}")
    _p("\nsplit:"); _p(_ts(dtab["split"].value_counts()))
    _p("\nper study (donors, cells):"); _p(_ts(dtab.groupby("study")[["malignant", "reactive"]].agg(["count", "sum"])))
    _p("\nMF donor keys dropped by the cohort rules:")
    _p(_ts(don.loc[don["disease"].astype(str).eq(DISEASE) & ~don["cohort_keep"], "drop_reason"].value_counts()))
    _p("\nfull donor table:"); _p(_ts(dtab.sort_values(["split", "study"])))
_sec("§0 COHORT", _cohort)

# ---------------------------------------------------------------- 3. resource
def _resource():
    _p(f"{CONDITION}: {pseq.n_obs:,} perturbations x {pseq.n_vars:,} measured genes")
    _p(f"genes passing baseMean > {BASEMEAN_MIN}: {int(gmask.sum()):,}; regulators with >= {MIN_DE_GENES} DE genes: {int(rmask.sum()):,}")
    _p(_ts(pd.read_csv(TAB_DIR / "pseq29_perturbseq_qc.csv"), index=False))
    _p("\nTCR-signalling clusters:")
    _p(_ts(clusters[clusters["manual_annotation"].str.contains("TCR", na=False)]
           [["cluster", "manual_annotation", "cluster_gene_size", "rest_count", "stim8hr_count", "stim48hr_count", "condition_specificity"]], index=False))
    _p(f"\ncluster 99 members ({len(c99)}): " + ", ".join(sorted(c99)))
    _p(f"\ncluster 99 downstream genes in {CONDITION}: {len(ds99):,}; top 15 most-shared targets:")
    _p(_ts(ds99.sort_values("num_of_upstream", ascending=False).head(15)[["downstream_gene", "num_of_upstream", "sign_coherence"]], index=False))
_sec("§1 PERTURB-SEQ RESOURCE", _resource)

# ---------------------------------------------------------------- 4. signature
def _signature():
    _p(f"discovery DE: {len(res_disc):,} genes tested; FDR<0.05: {int((res_disc['padj']<0.05).sum()):,}; "
       f"FDR<0.01: {int((res_disc['padj']<0.01).sum()):,}; up (lfc>0, FDR<0.05): {int(((res_disc['padj']<0.05)&(res_disc['log2FoldChange']>0)).sum()):,}; "
       f"down: {int(((res_disc['padj']<0.05)&(res_disc['log2FoldChange']<0)).sum()):,}")
    _p(f"replication DE: {len(res_repl):,} genes tested; FDR<0.05: {int((res_repl['padj']<0.05).sum()):,}")
    _p(f"discovery z: {len(y_disc):,} genes; replication z: {len(y_repl):,}; common: {len(common):,}; "
       f"Pearson r(discovery z, replication z) = {np.corrcoef(y_disc[common], y_repl[common])[0,1]:.4f} (reconstruction ceiling)")
    _p(f"mean log2FC discovery = {res_disc['log2FoldChange'].mean():+.4f}; median = {res_disc['log2FoldChange'].median():+.4f}")
    _p(f"per-donor log2FC matrix: {Y_donor.shape[0]:,} genes x {Y_donor.shape[1]} donors")
    rr = res_disc.dropna(subset=["padj"])
    mk = [x for x in P.MF_MARKERS["down"] + P.MF_MARKERS["up"] if x in rr.index]
    _p("\nknown MF marker genes (discovery):"); _p(_ts(rr.loc[mk, ["baseMean", "log2FoldChange", "lfcSE", "stat", "padj"]].round(4)))
    _p("\ntop 30 up in malignant (by Wald stat, FDR<0.05):")
    _p(_ts(rr[rr["padj"] < 0.05].sort_values("stat", ascending=False).head(30)[["log2FoldChange", "stat", "padj"]].round(4)))
    _p("\ntop 30 down in malignant:")
    _p(_ts(rr[rr["padj"] < 0.05].sort_values("stat").head(30)[["log2FoldChange", "stat", "padj"]].round(4)))
_sec("§2 MF SIGNATURE (pydeseq2 malignant vs reactive, ~ donor + arm)", _signature)

# ---------------------------------------------------------------- 5. model
def _model():
    _p(f"X: {X.shape[0]:,} genes x {X.shape[1]:,} regulators; y: {len(y):,} genes")
    _p("\nheld-out-gene performance (mean over 15 fits):")
    _p(_ts(summ[["n_genes", "n_regulators", "test_pearson", "test_pearson_se", "train_pearson", "alpha", "l1_ratio"]].round(4)))
    for name, m in models.items():
        v = m.eval["test_pearson"]
        _p(f"  {name}: per-fit test r = " + ", ".join(f"{x:.3f}" for x in v) + f"  (min {v.min():.3f}, max {v.max():.3f})")
    _p("\nreplication (held-out genes vs replication-donor signature), per split:")
    _p(_ts(repl.round(4), index=False))
    _p("\nreplication summary:")
    _p(_ts(repl[["correlation", "upper_correlation", "pred_input_correlation"]].agg(["mean", "std", "min", "max"]).round(4)))
_sec("§3a MODEL FIT (pert2state: SVD D=60 + elastic net, 5-fold x 3 over genes)", _model)

# ---------------------------------------------------------------- 6. weights
def _weights():
    _p(f"{len(coefs):,} regulator weights; w>0: {int((coefs['coef_mean']>0).sum()):,}; w<0: {int((coefs['coef_mean']<0).sum()):,}; "
       f"|w| quantiles 50/90/99%: {np.percentile(coefs['coef_mean'].abs(), [50, 90, 99]).round(5).tolist()}")
    _p("\ncontrol panel:"); _p(_ts(ctrl.round(4), index=False))
    _p(f"\nsign agreement: {int(tested['sign_ok'].sum())}/{len(tested)} tested controls; "
       f"binomial p (one-sided) = {ss.binomtest(int(tested['sign_ok'].sum()), len(tested), 0.5, alternative='greater').pvalue:.4f}; "
       f"median |w| rank of controls = {tested['abs_rank'].median():.0f} of {len(coefs)}; "
       f"{len(ctrl) - len(tested)} panel genes not in the regulator set")
    cols = [c for c in ["coef_mean", "coef_sem", "coef_rank", "abs_rank", "cluster", "arm", "arm_class"] if c in coefs.columns]
    _p("\nTOP 50 w>0 (knockdown MIMICS MF):"); _p(_ts(coefs.head(50)[cols].round(4)))
    _p("\nTOP 50 w<0 (knockdown OPPOSES MF / dependency):"); _p(_ts(coefs.tail(50)[cols].iloc[::-1].round(4)))
    _p(f"\nLong-Island-City ordering: {len(sig_genes):,} signature genes (FDR<0.01) used to cluster {len(lic):,} regulators")
_sec("§3b REGULATOR WEIGHTS", _weights)

# ---------------------------------------------------------------- 7. pathways
def _pathways():
    _p(f"{len(ptab)} sets tested (>= 5 members)")
    c = ["set", "n", "mean_w", "mean_abs_w", "delta_signed", "p_abs_greater", "q_abs_greater", "q_signed_greater", "q_signed_less"]
    c = [x for x in c if x in ptab.columns]
    _p("\nauthors' |w| test, q<0.1:"); _p(_ts(ptab[ptab["q_abs_greater"] < 0.1][c].round(4), index=False))
    _p("\nsigned: knockdown MIMICS MF (q<0.1):")
    _p(_ts(ptab[ptab["q_signed_greater"] < 0.1][c + ["members"]].round(4), index=False))
    _p("\nsigned: knockdown OPPOSES MF (q<0.1):")
    _p(_ts(ptab[ptab["q_signed_less"] < 0.1][c + ["members"]].round(4), index=False))
    _p("\nTCR modules (project):")
    _p(_ts(ptab[ptab["set"].str.startswith("TCRmod:")][c].round(4), index=False))
    _p("\nall sets, top 40 by p_abs_greater:")
    _p(_ts(ptab.sort_values("p_abs_greater").head(40)[c].round(4), index=False))
_sec("§3c PATHWAY TESTS", _pathways)

# ---------------------------------------------------------------- 8. TCR axis
def _tcr():
    _p("activation-signature (Arce 2024) fit:")
    _p(_ts(P.heldout_summary({"Arce activation": m_act})[["n_genes", "n_regulators", "test_pearson", "test_pearson_se", "train_pearson"]].round(4)))
    _p(f"\nSpearman rho(w_MF, w_activation) = {rho:.4f} (p = {p:.2e}) over {len(both):,} regulators")
    _p(f"Pearson r(w_MF, w_activation) = {np.corrcoef(both['coef_mean'], both['coef_mean_act'])[0,1]:.4f}")
    _p("\nTCR cascade components (w_MF, w_act):")
    _p(_ts(both.reindex([g for g in P.TCR_CASCADE if g in both.index])[["coef_mean", "coef_sem", "coef_mean_act", "coef_sem_act", "cluster"]].round(4)))
    _p(f"\nMF-specific regulators (top-100 |w_MF|, outside top-500 |w_act|): {int(both['mf_specific'].sum())}")
    _p(_ts(both[both["mf_specific"]].sort_values("coef_mean")[["coef_mean", "coef_mean_act", "cluster"]].round(4)))
    _p("\nprojection on TCR-signalling cluster programs (r>0: resembles LOSS of TCR signalling; r<0: SUSTAINED engagement):")
    _p(_ts(proj[["n_genes", "pearson_y_vs_kd_sign", "mean_y_kd_up", "mean_y_kd_down", "diff", "p_perm"]].astype(float).round(4)))
_sec("§3d TCR-ENGAGEMENT AXIS", _tcr)

# ---------------------------------------------------------------- 9. arms
def _arms():
    _p(f"callable donors with arm signatures: {len(events)} of {cells['donor'].nunique()} cohort donors")
    _p(f"regulators placed on arms: {int(coefs['arm'].notna().sum()):,} of {len(coefs):,}")
    _p("\narm classes (event in >= {:.0%} of callable donors):".format(MIN_ARM_FRAC)); _p(_ts(classes.round(3)))
    _p("\nregulators per arm class:"); _p(_ts(coefs["arm_class"].value_counts()))
    _p("\nmean / median w per arm class:")
    _p(_ts(coefs.groupby("arm_class")["coef_mean"].agg(["count", "mean", "median", "std"]).round(5)))
    _p("\n4A tests:"); _p(_ts(act.round(4), index=False))
    _p("\ntop 25 regulators on LOST arms (w>0 = deletion would mimic MF):")
    _p(_ts(lost.head(25)[["coef_mean", "coef_sem", "arm", "cluster"]].round(4)))
    _p("\ntop 25 regulators on GAINED arms (w<0 = extra dosage would feed MF):")
    _p(_ts(gained.head(25)[["coef_mean", "coef_sem", "arm", "cluster"]].round(4)))
    _p("\nper-arm mean w (placed regulators):")
    _p(_ts(coefs.dropna(subset=["arm"]).groupby("arm")["coef_mean"].agg(["count", "mean", "median"]).join(classes["class"]).round(5)))
_sec("§4A COHORT-LEVEL ARM TESTS", _arms)

# ---------------------------------------------------------------- 10. donor level
def _donor():
    for cls in ("lost", "gained"):
        g = ok[ok["class"] == cls]
        if len(g):
            _p(f"{cls} arms: {len(g)} donors; own-set r beats null (p<0.05) in {int((g['p_null']<0.05).sum())}; "
               f"median r_obs = {g['r_obs'].median():.4f} vs null {g['null_mean'].median():.4f}; "
               f"mean r_obs - null = {(g['r_obs']-g['null_mean']).mean():+.4f}; "
               f"paired Wilcoxon (greater) p = {ss.wilcoxon(g['r_obs'], g['null_mean'], alternative='greater').pvalue:.4g}; "
               f"sign test: {int((g['r_obs']>g['null_mean']).sum())}/{len(g)} donors above null")
    _p("\nper-donor table:"); _p(_ts(ok.sort_values(["class", "p_null"]).round(4), index=False))
_sec("§4B DONOR-LEVEL DOSAGE-PREDICTED PROGRAM", _donor)

# ---------------------------------------------------------------- 11. per cell
def _percell():
    _p(f"regulators scored: {len(sets)}; gene-set sizes (up, down): " + ", ".join(f"{k}({len(v['up'])},{len(v['down'])})" for k, v in sets.items()))
    _p(f"cells scored: {len(scores):,}; malignant cells with both a score and an arm profile: {len(mal_cells):,}")
    ev = summ_cell[summ_cell["event_class"] != "no event"]
    _p(f"\nregulator x arm x event-class rows: {len(summ_cell)}; event rows: {len(ev)}; event rows q<0.1: {int((ev['q_stouffer']<0.1).sum())}; "
       f"event rows with rho in the expected direction (w>0 & rho<0 or w<0 & rho>0): "
       f"{int(((np.sign(ev['coef_mean']) != np.sign(ev['mean_rho'])) & (ev['q_stouffer']<0.1)).sum())}")
    _p("\nsummary (all rows):"); _p(_ts(summ_cell.round(4), index=False))
    _p("\nper-donor rho, long table (first 60 rows by p):")
    _p(_ts(long.sort_values("p").head(60).round(4), index=False))
_sec("§4C PER-CELL KNOCKDOWN-MIMICRY vs ARM DOSAGE", _percell)

# ---------------------------------------------------------------- 12. outputs
def _outputs():
    _p("tables written (tables/):")
    for f in sorted(TAB_DIR.glob("pseq29_*")): _p(f"  {f.name}  ({f.stat().st_size/1e3:.0f} kB)")
    _p("figures written (figures/):")
    for f in sorted(FIG_DIR.glob("pseq29_*.png")): _p(f"  {f.name}")
    _p("caches (data/atlas_joint/):")
    for f in sorted(OUT_DIR.glob("pseq29_*")): _p(f"  {f.name}  ({f.stat().st_size/1e6:.0f} MB)")
_sec("FILES PRODUCED", _outputs)

_out = TAB_DIR / f"pseq29_full_report_{CONDITION}.txt"
_out.write_text(_buf.getvalue())
print(f"\n[report: {len(_buf.getvalue()):,} characters; saved to {_out.relative_to(NB_DIR)}]")
```

```
====================================================================================================
29_perturbseq_reverse_engineering — FULL REPORT (Stim48hr)
====================================================================================================

WHAT THIS NOTEBOOK DOES
-----------------------
Goal: take the transcriptional program that separates MALIGNANT CD4 T cells from REACTIVE (non-malignant) CD4 T cells
in MF skin, and ask which single-gene CRISPRi knockdowns in a normal, TCR-stimulated CD4 T cell reproduce or oppose it.
This "reverse-engineers" the malignant state into candidate upstream regulators, groups them into pathways, places the
MF program on the TCR-engagement axis, and finally asks whether the regulators lie on the chromosome arms that MF cells
recurrently lose or gain (inferCNV), at cohort, donor, and single-cell resolution.

DATA USED
---------
1. Perturb-seq resource (external): Zhu, Dann et al., Cell 2026 (GWCD4i). CRISPRi (ZIM3-dCas9) knockdown of 12,748
   genes in primary human CD4+ T cells from 4 donors, three culture conditions. Only the condition "Stim48hr"
   (48 h after CD3/CD28/CD2 restimulation) and only the PUBLISHED per-perturbation DESeq2 summary statistics
   (knockdown vs non-targeting control: log2FC, lfcSE, Wald z, padj per measured gene) are used
   (data/external_ref/perturbseq_gwcd4i/GWCD4i_DE_Stim48hr.h5ad, built by jobs/run_perturbseq_prep.py).
   No cell-level Perturb-seq data were downloaded. Also from the paper: the 111 functional regulator clusters
   (Suppl. Table 13; cluster 99 = Stim48hr-specific "TCR signaling") and each cluster's downstream target genes.
2. MF skin T-cell atlas (this project, atlas v2): data/atlas_joint/skin_T_tcr_annotated_v4.h5ad (nb21 labels + V(D)J
   fold-in, raw counts). Cells restricted to cell_type_T == "CD4", disease == "MF".
3. Malignancy calls: helpers/malignancy.py, headline call `mal_cluster` (malignant = mal_cluster & defined;
   reactive = defined & ~mal_cluster, same donor).
4. Donor truth: helpers/target_cohort.skin_donor_table() — donors kept only if the malignant clone is CD4 (not CD8,
   DN, gamma-delta), diagnosis MF, not a duplicate of another donor key, not healthy control.
5. inferCNV (nb23, revision v5): per-cell 41-arm CNV matrix with diploid null rows
   (data/atlas_joint/skin_T_arm_cnv_v5.parquet) and per-donor signed arm signatures (tables/skin_cnv_v5_summary.csv).
6. Comparator signature: Arce 2024 TCR-activation signature (effector T stimulated vs resting), the authors' own
   negative control for polarization.
7. Gene-to-arm map: the same GTF the inferCNV pipeline used, split at hg38 centromeres.

CALCULATIONS, IN ORDER
----------------------
§0 Cohort. CD4 MF donors with >= MIN_CELLS_ARM malignant AND >= MIN_CELLS_ARM reactive CD4 cells. Random 80/20
   discovery/replication split of DONORS, stratified by study (same design as the authors' OneK1K split).
§1 Resource QC. Regulator filter = perturbations with significant effects on >= 10 measured genes (authors' model
   filter); gene filter = mean baseMean > 0.1 in every condition.
§2 MF signature (the response vector y). One pseudobulk per donor x arm (raw counts summed), pydeseq2 with design
   ~ donor + arm, contrast malignant vs reactive, run separately on discovery and replication donors. y = Wald
   z-score (log2FC / lfcSE) per gene, then centred/standardized by the authors' prepare_y. Per-donor log2FC on CP10K
   pseudobulks (log2((mal+1)/(react+1))) is kept for §4B. Pearson r between discovery and replication z = the ceiling
   any reconstruction can reach.
§3a pert2state model (authors' code, vendored verbatim in helpers/pert2state_model.py). X = genes x regulators matrix
   of Perturb-seq z-scores with the on-target entry masked (column-mean imputed); per-regulator standardization ->
   TruncatedSVD(D = 60) -> ElasticNetCV (inner 4-fold; alpha logspace(-3,3,30); l1_ratio logspace(-2,0,10)).
   Outer 5-fold CV over GENES, repeated 3x (15 fits). Performan
... [102764 more chars truncated]
```
