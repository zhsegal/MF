# old/28b_tme_degradome_figures_mc_premerge.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/28b_tme_degradome_figures_mc_premerge.ipynb` and is never run._

# 28b_mc — figures for the cutaneous degradome, on the cluster-level malignancy call

A fresh copy of `28b_tme_degradome_figures`. It reads **`28_tme_degradome_mc`**'s tables
(`tables/deg28mc_*`), and its malignant-vs-reactive CD4 arms come from **`mal_cluster`**
(`26b_cluster_malignancy`) instead of `mal_tcr`. 28b is left untouched for a later comparison.
Outputs are `figures/degradome/deg28bmc_*` and `tables/deg28bmc_*`.

| fig | question | source |
|---|---|---|
| F1 | degrading in malignant vs reactive vs healthy CD4; paired test run here | object |
| F2 | proteases in fibroblasts (a) and myeloid cells (b) beside the CD4 arms, MF skin | object |
| FB | T-cell → stroma ligands vs what the clone does carry (inhibitors, lysosomal cathepsins, granzymes) | object, recomputed |
| FC | LIGHT → LTβR/HVEM → RELB, read as a chain | object, recomputed |
| FD | Th2 axis: TF on the clone, cytokines, receptor and output on the stroma | object, recomputed |
| FE | hyaluronan: synthesis, fragmentation, CD44 — stroma (MF vs healthy) and CD4 | object, recomputed |
| FF | the TGF-β loop in fibroblasts beside the same genes in CD4 | object, recomputed |
| **FG** *(new)* | does the stroma track tumour burden across MF samples? | 28_mc §D + object |
| **\*_stage** *(new)* | each figure above split by stage: early vs advanced MF (`stage_class`) | same, per stage |

## Arms

The malignancy call is `mal_cluster`:
- each sample's T cells are Leiden-clustered, and each cluster is called malignant or benign
  (ALICE where the sample has TCR, else a gene program + V-gene skew, else a manual call);
- CD4 with no call are excluded;
- healthy CD4 from our own atlas are a **fourth arm** (`CD4_HC`), and HC is benign by construction.

**Circularity:** TOX (FD) is part of the program that calls `auto` clusters, so its
malignant-vs-reactive difference is partly built into the arms. It is marked on the figure. Read
28_mc §A's `*_tcr_src` columns for the program-free result.

## FA–FG statistics

- **Cohort.** Every MF arm is restricted to the MF samples with a `mal_cluster` call (~100,
  against 38 on `mal_tcr`), so the clone and the stroma come from the same biopsies. Healthy =
  atlas HC.
- **Replicate value.** Donor **mean** of log1p CP10K over ≥ `cfg.MIN_CELLS` cells.
- **Tests.**
  - Malignant vs reactive CD4: paired Wilcoxon signed-rank on the per-biopsy difference
    (`DC.paired_within`).
  - Otherwise: donor-level Mann-Whitney.
  - *Not testable* below `cfg.MIN_DETECTION` in both arms, or with fewer than
    `cfg.MIN_DONORS_PER_ARM` donors.
- **Correction.** BH over the tests drawn in each figure. These are not 28_mc's q-values.
- **Titles** are descriptive. **Captions** that state a finding are computed from these tests, or
  checked against them, because the mal_tcr run's claims need not hold under the new arms.
- **Stage (`*_stage`).** `stage_class`: early = IA–IIA, advanced = IIB+; MF samples of unknown
  stage are left out. Within a stage each figure re-runs its own test. Advanced vs early is
  pooled across studies: donor-level Mann-Whitney on donor means, and for the clone on the
  per-biopsy malignant − reactive difference averaged per donor. Early is mostly li2024; each
  caption prints the study mix.

## Outputs

`figures/degradome/deg28bmc_*.png` and `tables/deg28bmc_*.csv`. Nothing under `deg28_*`,
`deg28b_*` or `deg28mc_*` is written.


```python  # cell 1
# ============================================================================
# §0  Setup — the same preamble and the same build gate as nb28 §0
# ============================================================================
# Figures only. The gate still runs: if the size factor was not summed over all 42,347 genes
# before the panel subset, every value drawn below is scaled by a cell-type-dependent factor.
import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


def _resolve_nb_dir() -> Path:
    start = Path.cwd()
    for base in [start, *start.parents]:
        for sub in [Path("."), Path("MF")]:
            cand = base / sub
            if cand.name == "MF" and (cand / "data").exists():
                return cand.resolve()
    raise FileNotFoundError(f"could not locate MF/data from {start}")


NB_DIR = _resolve_nb_dir()
sys.path.insert(0, str(NB_DIR / "helpers"))

import degradome_data as cfg          # config only; frozen
import degradome_helpers as D         # build + gate
import degradome_cells as DC          # scores, tests, the plotting layer
import malignancy as MAL              # the three calls
import external_ref as X              # reynolds2021
import cluster_malig_helpers as CM    # the program that calls auto clusters

# Inputs are 28_mc's tables (cfg.tab -> deg28mc_*); this notebook writes only deg28bmc_*.
cfg.TAB_PREFIX = cfg.FIG_PREFIX = "deg28mc_"
PREFIX = "deg28bmc_"


def fig28b(name, ext="png"):
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    return cfg.FIG_DIR / f"{PREFIX}{name}.{ext}"


def tab28b(name):
    cfg.TAB_DIR.mkdir(parents=True, exist_ok=True)
    return cfg.TAB_DIR / f"{PREFIX}{name}.csv"


ARM_CALL = "mal_cluster"
CALL_PROGRAM = set(pd.read_csv(CM.PROGRAM_CSV)["gene"])
CLONAL, NONCLONAL = cfg.CD4_MALIGNANT, cfg.CD4_REACTIVE
HC_CD4 = "CD4_HC"                     # healthy CD4 as an arm of its own; not an nb28 level

adata = D.load_degradome_adata()
eq = D.cached("equivalence",
              lambda: {"max_abs_diff": float(D.assert_build_equivalence(adata))})
assert eq["max_abs_diff"] < 1e-6, f"normalization gate failed: {eq}"
PANEL = [g for g in cfg.panel_genes() if g in set(map(str, adata.var_names))]

# 28_mc's tables. Nothing here recomputes them.
A = pd.read_csv(cfg.tab("A_paired_genes"))        # paired within biopsy, malignant vs reactive
B = pd.read_csv(cfg.tab("B_arm_tests"))           # donor-level, per block x row x gene x arm pair
CC = pd.read_csv(cfg.tab("C_ccc_edges"))          # curated ligand-receptor edges
COV = pd.read_csv(cfg.tab("coverage"))            # which levels may be spoken about

print(f"\nnormalization gate ok: max |lognorm(panel) - lognorm(full)| = {eq['max_abs_diff']:.3g}")
print(f"{adata.n_obs:,} cells | panel {len(PANEL)} genes | arms on {ARM_CALL}")
print(f"tables: A {len(A):,} genes | B {len(B):,} rows | C {len(CC):,} edge x level | "
      f"coverage {len(COV)} levels")
```

```
nb22 sidecar reconciled: 20 levels, 328,461 cells, zero drift
skin_degradome.h5ad: 985,273 cells x 503 genes, 112 donors, 123 samples, 12 studies
  scope ['MF', 'HC']: kept 1,002,481 of 1,345,527 skin cells; dropped 17,208 with no identity
  DEGRADOME-ONLY object: var is a hand-curated 503-gene panel. Not for HVG selection, PCA, UMAP, clustering or genome-wide DE. Use skin_degradome_pseudobulk_full.parquet for anything needing all genes.
[cache] equivalence <- equivalence.json

normalization gate ok: max |lognorm(panel) - lognorm(full)| = 0
985,273 cells | panel 503 genes | arms on mal_cluster
tables: A 503 genes | B 34,059 rows | C 394 edge x level | coverage 29 levels
```

```python  # cell 2
# ============================================================================
# §1  The arms — nb28 §2/§3, condensed, plus a healthy CD4 arm
# ============================================================================
# The object on disk was built with mal_knn as cfg.MALIG_HEADLINE. nb28 derives its own CD4
# split on mal_cluster and keeps the vocabulary identical, so writing that split straight into
# cfg.GROUPBY makes every plotting helper in degradome_cells -- all of which hardcode
# cfg.GROUPBY -- draw the mal_cluster arms with no helper edit.
mal = MAL.load_malignancy(adata.obs["cell_id"] if "cell_id" in adata.obs else adata.obs_names,
                          has_tcr=adata.obs.get("has_tcr"), verbose=False)
mal.index = adata.obs_names

IS_CD4 = (adata.obs[cfg.CELLTYPE_SRC].astype(str) == "CD4").to_numpy()
IS_MF = (adata.obs["disease"].astype(str) == "MF").to_numpy()
IS_HC = (adata.obs["disease"].astype(str) == cfg.HC_DISEASE).to_numpy()

LVL = pd.Series(adata.obs[cfg.GROUPBY].astype(str).to_numpy(), index=adata.obs_names)
LVL = LVL.mask(IS_CD4, pd.Series(MAL.cd4_level(adata.obs[cfg.CELLTYPE_SRC], mal,
                                               call=ARM_CALL).astype(str),
                                 index=adata.obs_names))
# Healthy CD4 is its own arm, not "reactive": malignant/reactive is a within-MF contrast and
# folding healthy cells into the reactive arm would make it mean two different things.
LVL = LVL.mask(IS_CD4 & IS_HC, HC_CD4)
adata.obs[cfg.GROUPBY] = pd.Categorical(LVL)
adata.obs["arm_level"] = adata.obs[cfg.GROUPBY]
LV = LVL.to_numpy()

# Stage for the *_stage companions: stage_class (early IA-IIA | advanced IIB+). stage_group needs
# an exact stage and calls gaydosik2019 / herrera2021 unknown although both cohorts are advanced.
STAGE_KEY = "stage_class"
STAGES = ["early", "advanced"]
STAGE_COL = {"early": "#e8a33d", "advanced": "#c0392b"}
STAGE = adata.obs[STAGE_KEY].astype(str).to_numpy()
_smp = adata.obs[cfg.SAMPLE_KEY].astype(str).to_numpy()
SAMPLE2DONOR = (pd.Series(adata.obs[cfg.DONOR_KEY].astype(str).to_numpy(), index=_smp)
                .groupby(level=0).first())
SAMPLE2STAGE = pd.Series(STAGE, index=_smp).groupby(level=0).first()

CLAIM = [lv for lv in cfg.CLAIM_LEVELS if lv in set(COV.loc[COV["claimable"], cfg.GROUPBY])]
FIBRO = [lv for lv in cfg.FIBRO_LEVELS if lv in CLAIM]
MYELOID = [lv for lv in cfg.MYELOID_LEVELS if lv in CLAIM]

ext = X.load_external()
EXT_T = ext.obs[X.TARGET_COL].astype(str).to_numpy()
ON_OBJ = set(LV)


def tme_rows(groups):
    """{row: {MF, HC, external HC}} for a {row_label: [fine levels]} map, plus per-row notes."""
    rows, notes = {}, {}
    for row, levels in groups.items():
        levels = [lv for lv in levels if lv in ON_OBJ]
        if not levels:
            continue
        m = np.isin(LV, levels)
        tg = cfg.EXT_ARM.get(row)
        rows[row] = {"MF": (adata, m & IS_MF), "HC": (adata, m & IS_HC),
                     "external HC": (ext, np.isin(EXT_T, tg)) if tg else None}
        note = []
        if row in cfg.EXT_POOLED:
            note.append(f"ext pooled: {'+'.join(tg)}")
        if not set(levels) & set(CLAIM):
            note.append("below donor gate")
        if note:
            notes[row] = "; ".join(note)
    return rows, notes


TME_ARMS = ["MF", "HC", "external HC"]
CD4_ARMS4 = ["malignant", "reactive", "HC CD4", "external HC"]
# DC.ARM_COLORS gives reactive and HC the same green; the healthy CD4 arm needs its own hue.
ARM_COLORS4 = {**DC.ARM_COLORS, "HC CD4": "#1f77b4"}

cd4_rows4 = {"CD4": {
    "malignant": (adata, (LV == CLONAL) & IS_MF),
    "reactive": (adata, (LV == NONCLONAL) & IS_MF),
    "HC CD4": (adata, LV == HC_CD4),
    "external HC": (ext, np.isin(EXT_T, cfg.EXT_ARM["CD4"])),
}}
mye_low, mye_low_notes = tme_rows(cfg.MYELOID_COARSE)
mye_high, mye_high_notes = tme_rows({lv: [lv] for lv in cfg.MYELOID_LEVELS})
fib_low, fib_low_notes = tme_rows(cfg.FIBRO_COARSE)
fib_high, fib_high_notes = tme_rows({lv: [lv] for lv in cfg.FIBRO_LEVELS})

FIB_FINE = [lv for lv in cfg.FIBRO_LEVELS if lv in ON_OBJ]
MYE_FINE = [lv for lv in cfg.MYELOID_LEVELS if lv in ON_OBJ]
STROMA_FINE = FIB_FINE + MYE_FINE

for lab, n in [("malignant", (LV == CLONAL) & IS_MF), ("reactive", (LV == NONCLONAL) & IS_MF),
               ("HC CD4", LV == HC_CD4), ("excluded, no call", (LV == cfg.CD4_UNASSESSED))]:
    print(f"  {lab}: {int(n.sum()):,} cells / "
          f"{adata.obs.loc[n, cfg.DONOR_KEY].nunique()} donors")
print(f"claimable {len(CLAIM)} levels | fibroblast {FIBRO} | myeloid {MYELOID}")
print(f"blocks: cd4 (1 row, 4 arms) | myeloid_low ({len(mye_low)}) | "
      f"myeloid_high ({len(mye_high)}) | fibro_low ({len(fib_low)}) | fibro_high ({len(fib_high)})")
```

```
malignant: 185,643 cells / 84 donors
  reactive: 137,664 cells / 86 donors
  HC CD4: 4,742 cells / 22 donors
  excluded, no call: 168 cells / 3 donors
claimable 18 levels | fibroblast ['F_papillary', 'F_reticular', 'F_mesenchymal', 'F_myofibroblast', 'F_inflammatory', 'F_apCAF'] | myeloid ['LC', 'cDC1', 'cDC2', 'DC_LAMP3', 'pDC', 'Mono_CD14', 'moDC', 'Mac_FOLR2', 'Mac_SPP1_TREM2', 'Mac_infl']
blocks: cd4 (1 row, 4 arms) | myeloid_low (4) | myeloid_high (12) | fibro_low (3) | fibro_high (6)
```

```python  # cell 3
# ============================================================================
# F0  One MF biopsy on a UMAP: cell types, with the malignant CD4 cells (mal_cluster) marked
# ============================================================================
# The degradome object carries no embedding. The sample's rows are pulled from the skin MRVI-u
# latent (sample-unaware, row-aligned to joint_mrvi_input_skin.h5ad; only obs is read from that
# file), and neighbors + UMAP are run on that one sample. Sample = UMAP_SAMPLE, or else the MF
# biopsy with a mal_cluster call that covers the most cell types (>= 20 cells each), ties broken
# by min(malignant, reactive).
import h5py
import scanpy as sc
from anndata.io import read_elem

UMAP_SAMPLE = None                    # set a sample_id to override the automatic pick
SKIN_INPUT = NB_DIR / "data" / "atlas_joint" / "joint_mrvi_input_skin.h5ad"
MRVI_U = NB_DIR / "data" / "atlas_joint" / "joint_X_mrvi_u_skin.npy"

_s = adata.obs[cfg.SAMPLE_KEY].astype(str).to_numpy()
_called = IS_MF & IS_CD4 & mal["mal_cluster_defined"].to_numpy()
_cand = pd.DataFrame({"sample": _s, "lv": LV})[np.isin(_s, np.unique(_s[_called]))]
_n = _cand.groupby(["sample", "lv"]).size().unstack(fill_value=0)
_rank = pd.DataFrame({"n_cells": _n.sum(1), "n_types": (_n >= 20).sum(1),
                      "n_mal": _n.get(CLONAL, 0), "n_react": _n.get(NONCLONAL, 0)})
_rank["balance"] = _rank[["n_mal", "n_react"]].min(1)
_rank = _rank[_rank["n_mal"] >= 20].sort_values(["n_types", "balance"], ascending=False)
SID = UMAP_SAMPLE or _rank.index[0]
print(_rank.head(5).to_string())

with h5py.File(SKIN_INPUT, "r") as f:
    _o = f["obs"]
    _ids = read_elem(_o["cell_id"] if "cell_id" in _o else _o[_o.attrs["_index"]])
_row = pd.Series(np.arange(len(_ids)), index=pd.Index(np.asarray(_ids).astype(str)))
_m = _s == SID
_cid = pd.Index(adata.obs["cell_id"] if "cell_id" in adata.obs else adata.obs_names)[_m]
_idx = _row.reindex(_cid.astype(str)).to_numpy()
assert not np.isnan(_idx).any(), f"{np.isnan(_idx).sum()} {SID} cells missing from the skin latent"
_idx = _idx.astype(int)
_srt = np.argsort(_idx)                                     # sorted rows: one pass over the mmap
_lat = np.empty((len(_idx), np.load(MRVI_U, mmap_mode="r").shape[1]), dtype=np.float32)
_lat[_srt] = np.load(MRVI_U, mmap_mode="r")[_idx[_srt]]

ad0 = sc.AnnData(obs=pd.DataFrame({"level": LV[_m]}, index=adata.obs_names[_m]))
ad0.obsm["X_mrvi_u"] = _lat
sc.pp.neighbors(ad0, use_rep="X_mrvi_u", n_neighbors=15, random_state=0)
sc.tl.umap(ad0, random_state=0)

_lab = {CLONAL: "Malignant CD4 (cluster call)", NONCLONAL: "Reactive CD4",
        cfg.CD4_UNASSESSED: "CD4, no call"}
_lvls = set(ad0.obs["level"])
_others = [lv for lv in cfg.CT_ORDER if lv in _lvls and lv not in (CLONAL, NONCLONAL)]
_others += sorted(_lvls - set(_others) - {CLONAL, NONCLONAL})
_cmap = plt.get_cmap("tab20")
_col = {lv: _cmap(i % 20) for i, lv in enumerate(_others)}
_col.update({CLONAL: DC.ARM_COLORS["clonal"], NONCLONAL: DC.ARM_COLORS["non-clonal"]})

U = ad0.obsm["X_umap"]
fig, ax = plt.subplots(figsize=(10, 6))
for lv in _others + [NONCLONAL, CLONAL]:                    # malignant drawn last, on top
    k = (ad0.obs["level"] == lv).to_numpy()
    if not k.any():
        continue
    hi = lv == CLONAL
    ax.scatter(U[k, 0], U[k, 1], s=4 if hi else 1.5, c=[_col[lv]], lw=0, rasterized=True,
               label=f"{_lab.get(lv, lv)} ({k.sum():,})", zorder=3 if hi else 2)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("UMAP1", fontsize=7); ax.set_ylabel("UMAP2", fontsize=7)
for _sp in ax.spines.values():
    _sp.set_visible(False)
_h, _l = ax.get_legend_handles_labels()
ax.legend(_h[::-1], _l[::-1], fontsize=5, markerscale=3, frameon=False, loc="center left",
          bbox_to_anchor=(1.0, 0.5), ncol=1 + (len(_l) > 14))
n_mal = int((ad0.obs["level"] == CLONAL).sum())
ax.set_title(f"{SID}: {n_mal:,} of {ad0.n_obs:,} cells are malignant CD4 (mal_cluster)",
             fontsize=8)
fig.savefig(fig28b("F0_umap_sample"), dpi=200, bbox_inches="tight")
plt.show()
print(ad0)

# F0a-c  The same UMAP three times: everything grey except one compartment.
def umap_highlight(levels, colors, title, tag):
    lv = ad0.obs["level"].to_numpy()
    on = np.isin(lv, levels)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(U[~on, 0], U[~on, 1], s=1, c="#d9d9d9", lw=0, rasterized=True, zorder=1,
               label=f"other ({(~on).sum():,})")
    for l in levels:
        k = lv == l
        if k.any():
            ax.scatter(U[k, 0], U[k, 1], s=3, c=[colors[l]], lw=0, rasterized=True, zorder=2,
                       label=f"{_lab.get(l, l)} ({k.sum():,})")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP1", fontsize=7); ax.set_ylabel("UMAP2", fontsize=7)
    for _sp in ax.spines.values():
        _sp.set_visible(False)
    ax.legend(fontsize=6, markerscale=3, frameon=False, loc="center left",
              bbox_to_anchor=(1.0, 0.5))
    ax.set_title(f"{SID}: {title}", fontsize=8)
    fig.savefig(fig28b(f"F0{tag}_umap_sample"), dpi=200, bbox_inches="tight")
    plt.show()


_fib = [l for l in FIB_FINE if l in _lvls]
_mye = [l for l in MYE_FINE if l in _lvls]
_tab = plt.get_cmap("tab10")
umap_highlight([CLONAL, NONCLONAL], _col,
               f"{n_mal:,} malignant vs {int((ad0.obs['level'] == NONCLONAL).sum()):,} "
               f"reactive CD4", "a")
umap_highlight(_fib, {l: _tab(i % 10) for i, l in enumerate(_fib)},
               f"{len(_fib)} fibroblast subtypes", "b")
umap_highlight(_mye, {l: plt.get_cmap("tab20")(i % 20) for i, l in enumerate(_mye)},
               f"{len(_mye)} myeloid subtypes", "c")
```

```
n_cells  n_types  n_mal  n_react  balance
sample                                                        
Li2024_atlas__CTCL8    34006       25   7379     2447     2447
Li2024_atlas__CTCL5    41011       24   7532     4073     4073
Li2024_atlas__CTCL7    77413       22   3658    38432     3658
D8__P163_LPD7_MF       10461       22    881      417      417
D7__P73_lesional       18145       22    189      344      189

<Figure size 1000x600 with 1 Axes>
AnnData object with n_obs × n_vars = 34006 × 0
    obs: 'level'
    uns: 'neighbors', 'umap'
    obsm: 'X_mrvi_u', 'X_umap'
    obsp: 'distances', 'connectivities'

<Figure size 1000x600 with 1 Axes>
<Figure size 1000x600 with 1 Axes>
<Figure size 1000x600 with 1 Axes>
```

```python  # cell 4
# ============================================================================
# §2  Scores, computed once
# ============================================================================
# Family score = plain mean of the family's genes on X (see the degradome_cells docstring for
# why not score_genes). Families below cfg.MIN_FAMILY_SIZE_FOR_SCORE get no mean and are read
# per gene instead -- ha_synthase and glycosidase are the two that matter here.
FAMS = DC.family_score_cells(adata)
BALS = DC.balance_cells(adata)
FAM_COLS = [f"fam_{k}" for k in FAMS]
BAL_COLS = [f"bal_{k}" for k in BALS]

# The replicate unit: one median per (level, donor), gated at cfg.MIN_CELLS.
DMED = DC.donor_median(adata, FAM_COLS + BAL_COLS)

# The external band. nb28 reads the medians the external build wrote rather than re-scoring
# 195,739 cells; a family whose gene set is not the same on both sides abstains entirely.
# The external build's artifacts keep nb28's deg28_ prefix; cfg.tab now points at deg28mc_.
EXT_DMED = X.load_external_donor_median(cfg.TAB_DIR / "deg28_external_donor_median.csv")
EXT_ABSTAIN, EXT_DROP = X.external_abstentions(cfg.TAB_DIR / "deg28_external_panel_coverage.csv")


def ext_band(scores, target):
    """{score: one value per external donor} for plot_split_violin / plot_external_band."""
    d = EXT_DMED[EXT_DMED["target"].astype(str) == target]
    return {s: d[s].dropna().to_numpy() for s in scores
            if s in EXT_DMED.columns and s not in set(EXT_DROP) and d[s].notna().any()}


# The paired slice: MF CD4 only, both arms, one row per (sample, arm). This is what §A tests
# and what plot_paired_slopes draws -- the sample is the replicate, so donor, study, chemistry,
# site and treatment are the same tube.
AXIS_CD4_GENES = ["TNFSF14", "LTA", "LTB", "CD40LG", "CXCR4", "CD44", "HYAL3", "HYAL2",
                  "IL7R", "SELL", "S1PR1", "ITGA4", "ITGA5", "ITGA6", "CCR7", "BSG"]
TOP_A = (A[A["q"].notna()].sort_values("q")["gene"].head(6).tolist())
SLICE_GENES = [g for g in dict.fromkeys(TOP_A + AXIS_CD4_GENES) if g in set(PANEL)]

cd4_mf = adata[IS_MF & np.isin(LV, [CLONAL, NONCLONAL])].copy()
SLICE_COLS = DC.gene_score_cells(cd4_mf, SLICE_GENES, verbose=False)
SM = DC.donor_median(cd4_mf, SLICE_COLS, groupby="arm_level", donor_key=cfg.SAMPLE_KEY)
N_PAIRS = int(SM[SM["eligible"]].pivot_table(index=cfg.SAMPLE_KEY, columns="arm_level",
                                             values=SLICE_COLS[0], aggfunc="median")
              .dropna().shape[0])

print(f"family scores: {len(FAM_COLS)} | balances: {len(BAL_COLS)} | "
      f"donor medians {DMED.shape} ({int(DMED['eligible'].sum()):,} eligible)")
print(f"external abstains: {EXT_ABSTAIN or 'none'}")
print(f"paired slice: {cd4_mf.n_obs:,} MF CD4 cells, {N_PAIRS} biopsies with both arms "
      f"above {cfg.MIN_CELLS} cells (28_mc §A reports {int(A['n_pairs'].max())})")
```

```python  # cell 5
# ============================================================================
# §3  Figure utilities — read nb28's tables, draw, never re-test
# ============================================================================
# B stores each pair as (arm_a, arm_b) with delta = donor_median_b - donor_median_a, so
# "MF vs HC" carries HC - MF. b_grid flips it to the direction a reader expects (arm_a minus
# arm_b) and keeps nb28's q untouched.
STAR = DC._stars
GREY = "#bdbdbd"


def b_grid(block, rows, genes, arm_pair="MF vs HC"):
    """(row x gene) donor medians, delta and q for one arm pair, straight out of deg28_B.

    `block` may be one block or several -- an axis read across fibroblasts and myeloid cells
    needs fibro_high and myeloid_high together, and their row names do not overlap.
    """
    blocks = [block] if isinstance(block, str) else list(block)
    d = B[B["block"].isin(blocks) & B["row"].isin(rows) & B["gene"].isin(genes)
          & (B["arm_pair"] == arm_pair)].copy()
    d["delta_ab"] = d["donor_median_a"] - d["donor_median_b"]     # arm_a minus arm_b
    return d


def heat(ax, d, rows, cols, value="delta_ab", row_col="row", col_col="gene", cmap="RdBu_r",
         vmax=None, title=None, cbar_label=None, star_col="q", annot=None, scale="shared"):
    """rows x cols heatmap with a BH star in each cell that has one. NaN cells are hatched.

    `scale="column"` gives every gene its own colour scale. Genes on this panel differ by an
    order of magnitude in absolute expression (CXCL12 moves by 1.8, CXCR4 by 0.2), and on one
    shared scale the smaller gene is a blank column -- which is not what the test says.
    """
    M = d.pivot_table(index=row_col, columns=col_col, values=value, aggfunc="mean")
    Q = d.pivot_table(index=row_col, columns=col_col, values=star_col, aggfunc="min")
    # rows/cols are kept as asked even when deg28_B has no row for them: a hatched cell says
    # "we could not look here", which an absent column does not.
    rows, cols = list(rows), list(cols)
    M, Q = M.reindex(index=rows, columns=cols), Q.reindex(index=rows, columns=cols)
    if scale == "column":
        cmax = np.nanmax(np.abs(M.to_numpy()), axis=0)
        cmax = np.where(np.isfinite(cmax) & (cmax > 0), cmax, 1.0)
        Z, v = M.to_numpy() / cmax, 1.0
    else:
        v = np.nanmax(np.abs(M.to_numpy())) if vmax is None else vmax
        v = 1.0 if not np.isfinite(v) or v == 0 else v
        Z = M.to_numpy()
    im = ax.imshow(Z, cmap=cmap, vmin=-v, vmax=v, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            if not np.isfinite(M.iat[i, j]):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=True, color="#f0f0f0",
                                           hatch="///", edgecolor="white", linewidth=0))
                continue
            q = Q.iat[i, j]
            s = STAR(q) if np.isfinite(q) else ""
            if annot is not None:
                s = f"{M.iat[i, j]:{annot}}\n{s}" if s else f"{M.iat[i, j]:{annot}}"
            if s:
                ax.text(j, i, s, ha="center", va="center", fontsize=5.5,
                        color="#111111" if abs(Z[i, j]) < 0.6 * v else "white")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=6, style="italic")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=6)
    ax.tick_params(length=2, pad=1)
    if title:
        ax.set_title(title, fontsize=8)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.ax.tick_params(labelsize=5)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=6)
    return M, Q


def arm_dots(ax, sl, arms, gene_idx, tests=None, row=None, gene=None, colors=None, ylabel=None):
    """One dot per donor per arm, from DC.arm_test_block's slices. The n a comparison has."""
    colors = colors or ARM_COLORS4
    rng = np.random.default_rng(0)
    tops = []
    for i, a in enumerate(arms):
        s = sl.get(a)
        if s is None:
            ax.text(i, 0, "no ext.\ncounterpart", ha="center", va="bottom", fontsize=5,
                    color="#888888", rotation=90)
            continue
        v = s["dmean"][:, gene_idx]
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        tops.append(np.max(v))
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=8, alpha=0.85,
                   color=colors.get(a, "#999999"), edgecolor="white", linewidth=0.3, zorder=3)
        ax.hlines(np.median(v), i - 0.28, i + 0.28, color="#111111", linewidth=1.2, zorder=4)
    top = max(tops) if tops else 1.0
    if tests is not None and row is not None and gene is not None:
        sig = tests[(tests["row"] == row) & (tests["gene"] == gene) & (tests["q"] < 0.05)]
        for k, t in enumerate(sorted(sig.itertuples(), key=lambda t: arms.index(t.arm_b)
                                     - arms.index(t.arm_a))):
            x0, x1 = arms.index(t.arm_a), arms.index(t.arm_b)
            h = top * (1.06 + 0.13 * k)
            ax.plot([x0, x0, x1, x1], [h - top * .03, h, h, h - top * .03], color="#333333",
                    linewidth=0.6, clip_on=False)
            ax.text((x0 + x1) / 2, h, STAR(t.q), ha="center", va="bottom", fontsize=6)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, rotation=90, fontsize=6)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_ylabel(ylabel or "", fontsize=6)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    return ax


def lollipop(ax, d, label_col="gene", value="pct_up", q_col="q", family_col="family",
             centre=0.5, xlabel=None, title=None, n_label=None):
    """One row per gene: dot at `value`, stem to `centre`, colour by family, grey if abstained.

    Direction is pct_up, not median_delta -- these panels are zero-inflated and the median
    paired delta is exactly 0 for most genes (nb28 §A).
    """
    d = d.reset_index(drop=True)
    fams = [f for f in dict.fromkeys(d[family_col].astype(str))]
    pal = DC._level_colors(fams)
    y = np.arange(len(d))[::-1]
    for yi, (_, r) in zip(y, d.iterrows()):
        v, q = r[value], r[q_col]
        ok = np.isfinite(v) and np.isfinite(q)
        c = pal[str(r[family_col])] if ok else GREY
        ax.plot([centre, v if ok else centre], [yi, yi], color=c, linewidth=1.0,
                alpha=0.8, zorder=2)
        ax.scatter([v if ok else centre], [yi], s=22 if ok else 10, color=c, zorder=3,
                   edgecolor="white", linewidth=0.4)
        lab = f"{r[label_col]}{' ' + STAR(q) if ok else ' (abstained)'}"
        ax.text(centre, yi, "  " + lab if (not ok or v >= centre) else lab + "  ",
                ha="left" if (not ok or v >= centre) else "right", va="center", fontsize=5.5,
                style="italic", zorder=4)
    ax.axvline(centre, color="#333333", linewidth=0.8, zorder=1)
    ax.set_yticks([])
    ax.set_ylim(-1, len(d))
    ax.tick_params(axis="x", labelsize=6)
    if xlabel or n_label:
        ax.set_xlabel("\n".join([x for x in (xlabel, n_label) if x]), fontsize=6)
    if title:
        ax.set_title(title, fontsize=8)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=pal[f], label=f,
                          markersize=4) for f in fams]
    return handles


def fam_arm_test(dmed, levels, score_cols, group="disease", arm_a="MF", arm_b="HC"):
    """Donor-level MWU on the FAMILY score, arm_a vs arm_b. BH over this grid only.

    nb28's tested unit is the gene, so a family-level star does not exist in deg28_B and has to
    be computed here. Same replicate (the donor), same test, same gate as DC.arm_tests.
    """
    out = []
    for lv in levels:
        d = dmed[(dmed[cfg.GROUPBY].astype(str) == lv) & dmed["eligible"]]
        for s in score_cols:
            x = d.loc[d[group].astype(str) == arm_a, s].dropna().to_numpy()
            y = d.loc[d[group].astype(str) == arm_b, s].dropna().to_numpy()
            p = np.nan
            if (len(x) >= cfg.MIN_DONORS_PER_ARM and len(y) >= cfg.MIN_DONORS_PER_ARM
                    and np.ptp(np.r_[x, y]) > 0):
                p = float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
            out.append({"row": lv, "gene": s, "n_donors_a": len(x), "n_donors_b": len(y),
                        "donor_median_a": np.median(x) if len(x) else np.nan,
                        "donor_median_b": np.median(y) if len(y) else np.nan, "p": p})
    r = pd.DataFrame(out)
    r["delta_ab"] = r["donor_median_a"] - r["donor_median_b"]
    r["q"] = np.nan
    m = r["p"].notna()
    if m.any():
        r.loc[m, "q"] = multipletests(r.loc[m, "p"], method="fdr_bh")[1]
    return r


FIG_INDEX = []


def logfig(fig_id, path, question, source, note=""):
    FIG_INDEX.append({"figure": fig_id, "file": Path(path).name, "question": question,
                      "source": source, "note": note})
    print(f"  {fig_id} -> {Path(path).name}")


print("figure utilities ready")
```

```python  # cell 6
# ============================================================================
# F1  Degrading in malignant vs reactive vs healthy CD4
# ============================================================================
# Two panels because they answer the question at two grains: the family score says how much
# degradative machinery a CD4 cell carries at all, the gene grid says which enzyme. The paired
# malignant-vs-reactive test (28_mc §A design) runs here on this call's arms; its one-line summary
# is printed, and the titles only name the figure.
F1_SCORES = [s for s in ["fam_mmp", "fam_cathepsin", "fam_serine_other", "fam_metallo_other",
                         "fam_granzyme", "fam_ha_catabolic", "fam_inhibitor",
                         "fam_matrix_receptor"] if s in DMED.columns]
CD4_LEVELS3 = [CLONAL, NONCLONAL, HC_CD4]
# The headline proteases only: the four MMPs CD4 cells express at all, the canonical sheddases
# ADAM10/17 plus the leukocyte ADAM8/19, uPAR and heparanase.
F1_GENES = [g for g in ["MMP2", "MMP9", "MMP14", "MMP25",
                        "ADAM8", "ADAM10", "ADAM17", "ADAM19",
                        "PLAUR", "HPSE"] if g in set(PANEL)]


def paired_title(cols, names, what, obj=None):
    """Summary + table from the paired within-biopsy test of `cols` on the MF CD4 arms (`obj`,
    default all of cd4_mf)."""
    sm = DC.donor_median(cd4_mf if obj is None else obj, cols, groupby="arm_level",
                         donor_key=cfg.SAMPLE_KEY)
    r = DC.paired_within(sm, cols, pair_key=cfg.SAMPLE_KEY, arm_col="arm_level",
                         ref=NONCLONAL, alt=CLONAL, fdr=True)
    r["name"] = r["score"].map(names)
    sig_ = r["q"] < 0.05
    up = r.loc[sig_ & (r["n_pos"] > r["n_neg"]), "name"].tolist()
    dn = r.loc[sig_ & (r["n_pos"] < r["n_neg"]), "name"].tolist()
    head = (f"Higher in malignant CD4: {', '.join(up)}" if up
            else f"No {what} is higher in malignant than reactive CD4")
    tail = f"; higher in reactive: {', '.join(dn)}" if dn else ""
    return f"{head}{tail} (paired within biopsy, {int(r['n_pairs'].max())} biopsies, BH)", r


F1A_TITLE, F1A_PAIRED = paired_title(F1_SCORES, {s: s.replace("fam_", "") for s in F1_SCORES},
                                     "degradome family")
_gcols = DC.gene_score_cells(cd4_mf, F1_GENES, verbose=False)
F1B_TITLE, F1B_PAIRED = paired_title(_gcols, dict(zip(_gcols, F1_GENES)), "headline protease")
cd4_mf.obs.drop(columns=[c for c in _gcols if c not in SLICE_COLS], inplace=True)
pd.concat([F1A_PAIRED.assign(panel="F1a"), F1B_PAIRED.assign(panel="F1b")]).to_csv(
    tab28b("F1_paired_tests"), index=False)

DC.plot_split_violin(
    adata, DMED, F1_SCORES, CD4_LEVELS3,
    title=("Degradome family scores in malignant, reactive and healthy CD4\n"
           "violin = cells, tick = median, dots = donors (≥ 25 cells); "
           "band = reynolds2021 healthy CD4, 5 donors"),
    external=ext_band(F1_SCORES, "CD4"), ncols=4,
    save=fig28b("F1a_cd4_family_scores"))
logfig("F1a", fig28b("F1a_cd4_family_scores"),
       "degradome family load, malignant vs reactive vs healthy CD4",
       "object (donor medians) + deg28_external_donor_median; paired test in F1_paired_tests")

f1b = DC.plot_family_arms(
    cd4_rows4, F1_GENES, CD4_ARMS4, arm_colors=ARM_COLORS4,
    title="Headline protease genes in malignant, reactive and healthy CD4",
    wrap=5, col_w=2.2, row_h=2.6, expressing_only=True, clean=True,
    save=fig28b("F1b_cd4_protease_genes"))
f1b.to_csv(tab28b("F1_source"), index=False)
logfig("F1b", fig28b("F1b_cd4_protease_genes"),
       "which protease, if any, the clone carries",
       "object; donor-level MWU recomputed over the drawn genes; paired test in F1_paired_tests",
       note="q recomputed figure-wide: the 4-arm set includes atlas HC CD4")
print(F1A_TITLE)
print(F1B_TITLE)
```

```python  # cell 7
# ============================================================================
# F1_stage  F1 split by stage: early vs advanced MF
# ============================================================================
# Stage = stage_class; MF samples of unknown stage are left out. Within a stage, F1's paired
# within-biopsy test runs on that stage's biopsies. Advanced vs early is pooled across studies: a
# donor-level Mann-Whitney on the per-biopsy malignant - reactive difference, averaged per donor,
# so study and chemistry cancel inside the difference.
_cd4_stage = cd4_mf.obs[STAGE_KEY].astype(str).to_numpy()


def stage_gap_test(cols, names):
    """Advanced vs early on the per-biopsy (median) malignant - reactive gap, donor-level MWU."""
    sm = DC.donor_median(cd4_mf, cols, groupby="arm_level", donor_key=cfg.SAMPLE_KEY)
    sm = sm[sm["eligible"]]
    out = []
    for c in cols:
        w = (sm.pivot_table(index=cfg.SAMPLE_KEY, columns="arm_level", values=c, aggfunc="median")
             .reindex(columns=[CLONAL, NONCLONAL]).dropna())
        donor = SAMPLE2DONOR.reindex(w.index).to_numpy()
        gap = (w[CLONAL] - w[NONCLONAL]).groupby(donor).mean()
        st = pd.Series(SAMPLE2STAGE.reindex(w.index).to_numpy()).groupby(donor).first()
        x, y = gap[st == "advanced"].to_numpy(), gap[st == "early"].to_numpy()
        p = (float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
             if min(len(x), len(y)) >= cfg.MIN_DONORS_PER_ARM and np.ptp(np.r_[x, y]) > 0
             else np.nan)
        out.append({"score": c, "name": names[c], "n_advanced": len(x), "n_early": len(y),
                    "gap_advanced": np.median(x) if len(x) else np.nan,
                    "gap_early": np.median(y) if len(y) else np.nan, "p": p})
    r = pd.DataFrame(out)
    r["q"] = np.nan
    m = r["p"].notna()
    if m.any():
        r.loc[m, "q"] = multipletests(r.loc[m, "p"], method="fdr_bh")[1]
    return r


def stage_title(cols, names, what):
    """F1's paired-test summary per stage, plus the advanced-vs-early gap line; and the tests."""
    lines, tabs = [], []
    for st in STAGES:
        t, r = paired_title(cols, names, what, obj=cd4_mf[_cd4_stage == st])
        lines.append(f"{st}: {t}")
        tabs.append(r.assign(stage=st))
    g = stage_gap_test(cols, names)
    ch = g[g["q"] < 0.05]
    moved = ", ".join(f"{r['name']} {'higher' if r['gap_advanced'] > r['gap_early'] else 'lower'}"
                      for _, r in ch.iterrows())
    lines.append(f"malignant - reactive gap, advanced vs early: {moved or 'no change'} "
                 f"(MWU on donor gap, {int(g['n_early'].max())} early | "
                 f"{int(g['n_advanced'].max())} advanced donors, BH)")
    return "\n".join(lines), pd.concat(tabs + [g.assign(stage="advanced vs early")])


# F1a_stage: the malignant and reactive arms of each stage side by side, healthy CD4 last.
F1S_NAME = {(CLONAL, "early"): "malignant early", (NONCLONAL, "early"): "reactive early",
            (CLONAL, "advanced"): "malignant adv.", (NONCLONAL, "advanced"): "reactive adv."}
_arm = np.full(adata.n_obs, "other", dtype=object)
for (lv, st), name in F1S_NAME.items():
    _arm[(LV == lv) & IS_MF & (STAGE == st)] = name
_arm[LV == HC_CD4] = "healthy CD4"
adata.obs["arm_stage"] = pd.Categorical(_arm)
F1S_LEVELS = list(F1S_NAME.values()) + ["healthy CD4"]
DMED_S = DC.donor_median(adata, F1_SCORES, groupby="arm_stage",
                         extra=("study", STAGE_KEY, "disease"))

F1AS_TITLE, F1AS_TESTS = stage_title(F1_SCORES, {s: s.replace("fam_", "") for s in F1_SCORES},
                                     "degradome family")
_gcols = DC.gene_score_cells(cd4_mf, F1_GENES, verbose=False)
F1BS_TITLE, F1BS_TESTS = stage_title(_gcols, dict(zip(_gcols, F1_GENES)), "headline protease")
cd4_mf.obs.drop(columns=[c for c in _gcols if c not in SLICE_COLS], inplace=True)
pd.concat([F1AS_TESTS.assign(panel="F1a"), F1BS_TESTS.assign(panel="F1b")]).to_csv(
    tab28b("F1_stage_tests"), index=False)

DC.plot_split_violin(
    adata, DMED_S, F1_SCORES, F1S_LEVELS, level_col="arm_stage",
    title=("Degradome family scores in malignant, reactive and healthy CD4, early and advanced MF\n"
           "violin = cells, tick = median, dots = donors (≥ 25 cells); stage = stage_class, "
           "unknown stage left out; band = reynolds2021 healthy CD4, 5 donors"),
    external=ext_band(F1_SCORES, "CD4"), ncols=4, panel=(2.4, 3.2),
    save=fig28b("F1a_stage_cd4_family_scores"))
logfig("F1a_stage", fig28b("F1a_stage_cd4_family_scores"),
       "degradome family load in malignant vs reactive CD4, early and advanced MF",
       "object (donor medians); paired test per stage + donor MWU on the gap (F1_stage_tests)")

# F1b_stage: F1b's four arms once per stage; plot_family_arms tests arm pairs within a row only.
cd4_rows_stage = {f"{st} MF": {
    "malignant": (adata, (LV == CLONAL) & IS_MF & (STAGE == st)),
    "reactive": (adata, (LV == NONCLONAL) & IS_MF & (STAGE == st)),
    "HC CD4": (adata, LV == HC_CD4),
    "external HC": (ext, np.isin(EXT_T, cfg.EXT_ARM["CD4"])),
} for st in STAGES}
f1bs = DC.plot_family_arms(
    cd4_rows_stage, F1_GENES, CD4_ARMS4, arm_colors=ARM_COLORS4,
    title="Headline protease genes in malignant, reactive and healthy CD4, early and advanced MF",
    wrap=5, col_w=2.2, row_h=2.6, expressing_only=True, clean=True,
    save=fig28b("F1b_stage_cd4_protease_genes"))
f1bs.to_csv(tab28b("F1_stage_source"), index=False)
logfig("F1b_stage", fig28b("F1b_stage_cd4_protease_genes"),
       "which protease the clone carries, early and advanced MF",
       "object; donor-level MWU within each stage row; paired test per stage (F1_stage_tests)",
       note="q recomputed figure-wide; healthy and external arms shared by both rows")
print(F1AS_TITLE)
print(F1BS_TITLE)
```

```python  # cell 8
# ============================================================================
# F2  Where the proteases are: fibroblasts and myeloid cells next to the CD4 arms, MF skin
# ============================================================================
# The two CD4 arms stay in both panels -- the contrast is the claim: near-empty CD4 violins for
# every protease beside stromal violins that are not. MF samples only; dots are donor medians
# (>= cfg.MIN_CELLS cells), the donor being the replicate. No test is drawn.
F2_CD4 = {"CD4_clonal": (LV == CLONAL) & IS_MF, "CD4_non_clonal": (LV == NONCLONAL) & IS_MF}
F2_PANELS = {
    "F2a": ("fibro",
            {"F_structural": np.isin(LV, cfg.FIBRO_COARSE["F_structural"]) & IS_MF,
             "F_inflammatory": (LV == "F_inflammatory") & IS_MF,
             "F_apCAF": (LV == "F_apCAF") & IS_MF},
            ["MMP2", "MMP14", "ADAMTS5", "CTSB", "CTSK", "PRSS23", "TIMP1", "TIMP3"],
            "Protease genes in fibroblasts vs malignant and reactive CD4, MF skin"),
    "F2b": ("myeloid",
            {lv: (LV == lv) & IS_MF for lv in ["Mac_FOLR2", "Mono_CD14", "cDC2"]},
            ["MMP12", "MMP9", "MMP14", "CTSB", "CTSC", "CTSS", "PLAUR", "SERPINA1"],
            "Protease genes in myeloid cells vs malignant and reactive CD4, MF skin"),
}
F2_COLORS = {"CD4_clonal": DC.ARM_COLORS["clonal"], "CD4_non_clonal": DC.ARM_COLORS["non-clonal"],
             "F_structural": "#08519c", "F_inflammatory": "#3182bd", "F_apCAF": "#9ecae1",
             "Mac_FOLR2": "#a63603", "Mono_CD14": "#e6550d", "cDC2": "#fdae6b"}
# Display names on the figure only; the source tables keep the atlas labels (methods).
F2_LABELS = {"CD4_clonal": "Malignant CD4 (cluster call)",
             "CD4_non_clonal": "Reactive CD4 (same biopsy)",
             "F_structural": "Structural fibroblast",
             "F_inflammatory": "Inflammatory fibroblast",
             "F_apCAF": "MHC-II+ fibroblast (apCAF)",
             "Mac_FOLR2": "Resident macrophage (FOLR2+)",
             "Mono_CD14": "Classical monocyte (CD14+)",
             "cDC2": "Conventional DC2"}

for tag, (lineage, stroma, genes, title) in F2_PANELS.items():
    d = DC.plot_group_violins(adata, {**F2_CD4, **stroma}, genes, title=title,
                              colors=F2_COLORS, labels=F2_LABELS,
                              save=fig28b(f"{tag}_{lineage}_vs_cd4"))
    d.to_csv(tab28b(f"{tag}_source_{lineage}"), index=False)
    logfig(tag, fig28b(f"{tag}_{lineage}_vs_cd4"),
           f"protease genes per cell type, CD4 arms vs {lineage}, MF skin",
           "object (per-cell log1p CP10K; donor medians)")
```

```python  # cell 9
# ============================================================================
# F2_stage  F2 split by stage: the same cell types in early and in advanced MF skin
# ============================================================================
# Descriptive, as F2: no test. Early block first, then advanced. The printed line counts, per
# stage, the proteases for which some stromal type's median donor mean is above both CD4 arms'.
from matplotlib.colors import to_rgb

_VAR = pd.Index(map(str, adata.var_names))
_DONOR = adata.obs[cfg.DONOR_KEY].astype(str).to_numpy()


def median_donor_mean(m, genes):
    """Median over donors (>= cfg.MIN_CELLS cells) of the donor mean log1p CP10K, per gene."""
    idx = np.flatnonzero(m)
    Xs = adata.X[idx][:, _VAR.get_indexer(genes)]
    Xs = Xs.toarray() if hasattr(Xs, "toarray") else np.asarray(Xs)
    g = pd.DataFrame(Xs, columns=genes).groupby(_DONOR[idx])
    return g.mean()[(g.size() >= cfg.MIN_CELLS).to_numpy()].median()


def pale(c, f=0.45):
    """`c` mixed toward white: the early block's shade of a cell type's colour."""
    return tuple(1 - f * (1 - v) for v in to_rgb(c))


F2S_WHO = {"fibro": "Fibroblasts", "myeloid": "Myeloid cells"}
for tag, (lineage, stroma, genes, _) in F2_PANELS.items():
    arms = {**F2_CD4, **stroma}
    groups = {f"{k}·{st}": m & (STAGE == st) for st in STAGES for k, m in arms.items()}
    med = pd.DataFrame({k: median_donor_mean(m, genes) for k, m in groups.items()}).T
    wins = {st: sum(bool(med.loc[[f"{k}·{st}" for k in stroma], g].max()
                         > med.loc[[f"{k}·{st}" for k in F2_CD4], g].max()) for g in genes)
            for st in STAGES}
    title = (f"Protease genes in {F2S_WHO[lineage].lower()} vs malignant and reactive CD4, "
             "early and advanced MF skin\n"
             "median of donor means; left block early (pale), right block advanced; "
             "stage = stage_class, unknown stage left out")
    d = DC.plot_group_violins(
        adata, groups, genes, title=title,
        colors={f"{k}·{st}": pale(F2_COLORS[k]) if st == "early" else F2_COLORS[k]
                for st in STAGES for k in arms},
        labels={f"{k}·{st}": f"{F2_LABELS[k]} · {st}" for st in STAGES for k in arms},
        panel=(3.8, 2.4), save=fig28b(f"{tag}_stage_{lineage}_vs_cd4"))
    d.to_csv(tab28b(f"{tag}_stage_source_{lineage}"), index=False)
    logfig(f"{tag}_stage", fig28b(f"{tag}_stage_{lineage}_vs_cd4"),
           f"protease genes per cell type, CD4 arms vs {lineage}, early and advanced MF skin",
           "object (per-cell log1p CP10K; donor medians)")
    print(f"{F2S_WHO[lineage]} are above both CD4 arms for {wins['early']}/{len(genes)} "
          f"proteases in early and {wins['advanced']}/{len(genes)} in advanced MF skin")
```

```python  # cell 10
# ============================================================================
# §4  FA–FE layer — mal_cluster cohort, donor means, every test recomputed here
# ============================================================================
# Every MF arm below is restricted to the MF samples whose CD4 cells carry a mal_cluster call,
# the sample set 28_mc §A and §C rest on -- so clone and stroma come from the same biopsies.
# The replicate value is the donor MEAN of log1p CP10K: these genes are zero-inflated and a
# per-cell median is exactly 0 in most donors. Both choices move the numbers, so no star here is
# nb28's q: each is recomputed on donor means and BH-corrected within its own figure.
import textwrap

import scipy.sparse as sp
from matplotlib.patches import FancyArrowPatch
from matplotlib.transforms import offset_copy

_samp = adata.obs[cfg.SAMPLE_KEY].astype(str).to_numpy()
CALL_SAMPLES = sorted(set(_samp[IS_MF & IS_CD4 & mal["mal_cluster_defined"].to_numpy()]))
IS_MF_CALL = IS_MF & np.isin(_samp, CALL_SAMPLES)
VAR = pd.Index(map(str, adata.var_names))

# Display names on the figures only; source tables keep the atlas labels (methods).
LABELS = {CLONAL: "Malignant CD4 (cluster call)", NONCLONAL: "Reactive CD4 (same biopsy)",
          HC_CD4: "Healthy-donor CD4",
          "F_structural": "Structural fibroblast", "F_inflammatory": "Inflammatory fibroblast",
          "F_apCAF": "MHC-II+ fibroblast (apCAF)", "F_reticular": "Reticular fibroblast",
          "F_mesenchymal": "Mesenchymal fibroblast",
          "Mac_FOLR2": "Resident macrophage (FOLR2+)", "Mono_CD14": "Classical monocyte (CD14+)",
          "cDC2": "Conventional DC2"}
UP, DOWN, NS = "#2a9d8f", "#b5542a", "#bdbdbd"
DIR_COLOR = {"up": UP, "down": DOWN, "ns": NS, "not tested": NS}
MALIG, REACT, HC_COL = "#7b3294", "#5e81ac", "#444444"
ARM_COL = {CLONAL: MALIG, NONCLONAL: REACT, HC_CD4: HC_COL}
XLAB = "donor mean, log1p CP10K"


def mask(ct, arm):
    """Cells of one cell type in one arm: 'MF' = called MF samples, 'HC' = atlas healthy skin,
    'early' / 'advanced' = the called MF samples of that stage (the *_stage figures).

    The healthy arm of either CD4 arm is healthy-donor CD4. F_structural is the coarse pool
    (papillary + reticular + mesenchymal + myofibroblast), as in F2.
    """
    if ct == HC_CD4 or (arm == "HC" and ct in (CLONAL, NONCLONAL)):
        return LV == HC_CD4
    m = np.isin(LV, cfg.FIBRO_COARSE.get(ct, [ct]))
    if arm == "HC":
        return m & IS_HC
    return m & IS_MF_CALL & (STAGE == arm if arm in STAGES else True)


_DM = {}


def donor_means(ct, arm, genes, key=cfg.DONOR_KEY):
    """(donor x gene) mean log1p CP10K, donors with >= cfg.MIN_CELLS cells; plus detection.

    Cached per (ct, arm, key); genes not yet cached are added on demand. `key=cfg.SAMPLE_KEY`
    gives the per-biopsy means the paired test runs on.
    """
    k = (ct, arm, key)
    genes = list(dict.fromkeys(genes))
    need = [g for g in genes if k not in _DM or g not in _DM[k][0].columns]
    if need:
        idx = np.flatnonzero(mask(ct, arm))
        Xs = adata.X[idx][:, VAR.get_indexer(need)]
        Xs = Xs.tocsr() if sp.issparse(Xs) else sp.csr_matrix(Xs)
        codes, uniq = pd.factorize(adata.obs[key].astype(str).to_numpy()[idx])
        cnt = np.bincount(codes, minlength=len(uniq))
        G = sp.csr_matrix((np.ones(len(idx)), (codes, np.arange(len(idx)))),
                          shape=(len(uniq), len(idx)))
        M = pd.DataFrame(np.asarray((G @ Xs).todense()) / np.maximum(cnt, 1)[:, None],
                         index=pd.Index(uniq, name=key), columns=need)
        det = pd.Series(np.asarray((Xs > 0).mean(axis=0)).ravel() if len(idx) else np.nan,
                        index=need)
        n = pd.Series(cnt, index=M.index)
        if k in _DM:
            M, det = _DM[k][0].join(M), pd.concat([_DM[k][1], det])
        _DM[k] = (M, det, n.reindex(M.index))
    M, det, cnt = _DM[k]
    return M.loc[cnt >= cfg.MIN_CELLS, genes], det[genes]


def prefetch(cts, genes, arms=("MF", "HC")):
    for ct in cts:
        for arm in arms:
            donor_means(ct, arm, genes)


def cell_values(ct, genes, max_cells=20_000, arm="MF"):
    """Per-cell log1p CP10K in one MF arm, subsampled for the violin KDE only (seed fixed)."""
    idx = np.flatnonzero(mask(ct, arm))
    if len(idx) > max_cells:
        idx = np.sort(np.random.default_rng(0).choice(idx, max_cells, replace=False))
    Xs = adata.X[idx][:, VAR.get_indexer(genes)]
    return pd.DataFrame(Xs.toarray() if sp.issparse(Xs) else np.asarray(Xs), columns=genes)


# ---------------------------------------------------------------- tests
def t_mwu(ct, gene, a="MF", b="HC", ct_b=None, label=None):
    """Donor-level Mann-Whitney on donor means: ct in arm a vs (ct_b or ct) in arm b."""
    ct_b = ct_b or ct
    ma, da = donor_means(ct, a, [gene])
    mb, db = donor_means(ct_b, b, [gene])
    x, y = ma[gene].to_numpy(), mb[gene].to_numpy()
    rec = {"label": label or LABELS.get(ct, ct), "gene": gene, "ct": ct, "arm_a": a,
           "ct_b": HC_CD4 if (b == "HC" and ct_b in (CLONAL, NONCLONAL)) else ct_b, "arm_b": b,
           "test": "MWU (donor)", "n_a": len(x), "n_b": len(y),
           "med_a": np.median(x) if len(x) else np.nan,
           "med_b": np.median(y) if len(y) else np.nan,
           "det_a": da[gene], "det_b": db[gene], "p": np.nan, "reason": ""}
    rec["delta"] = rec["med_a"] - rec["med_b"]
    rec["sign"] = np.sign(rec["delta"])
    if not np.nanmax([rec["det_a"], rec["det_b"]]) >= cfg.MIN_DETECTION:
        rec["reason"] = f"below {cfg.MIN_DETECTION:.0%} detection in both arms"
    elif min(len(x), len(y)) < cfg.MIN_DONORS_PER_ARM:
        rec["reason"] = f"too few donors ({len(x)}, {len(y)}; need {cfg.MIN_DONORS_PER_ARM})"
    elif np.ptp(np.r_[x, y]) == 0:
        rec["reason"] = "no variance"
    else:
        rec["p"] = float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
    rec["vals_a"], rec["vals_b"] = x, y
    return rec


def t_paired(genes, label="vs Reactive CD4 (same biopsy)", arm="MF"):
    """Clone vs reactive, paired within biopsy: nb28 §A's signed-rank, run on per-biopsy means.

    DC.paired_within is the §A function itself; only the per-biopsy summary changes (mean, not
    median), and the cohort is the same by construction (both arms need a mal_cluster call).
    `arm` = "MF", or one stage to pair only that stage's biopsies.
    """
    genes = list(dict.fromkeys(genes))
    a, da = donor_means(CLONAL, arm, genes, key=cfg.SAMPLE_KEY)
    b, db = donor_means(NONCLONAL, arm, genes, key=cfg.SAMPLE_KEY)
    long = (pd.concat([a.assign(arm=CLONAL), b.assign(arm=NONCLONAL)])
            .reset_index().assign(eligible=True))
    res = DC.paired_within(long, genes, pair_key=cfg.SAMPLE_KEY, arm_col="arm",
                           ref=NONCLONAL, alt=CLONAL, fdr=False).set_index("score")
    both = a.index.intersection(b.index)
    out = []
    for g in genes:
        r = res.loc[g]
        x, y = a.loc[both, g].to_numpy(), b.loc[both, g].to_numpy()
        det_ok = np.nanmax([da[g], db[g]]) >= cfg.MIN_DETECTION
        out.append({"label": label, "gene": g, "ct": CLONAL, "arm_a": arm, "ct_b": NONCLONAL,
                    "arm_b": arm, "test": "paired signed-rank (biopsy)",
                    "n_a": len(x), "n_b": len(y),
                    "med_a": np.median(x) if len(x) else np.nan,
                    "med_b": np.median(y) if len(y) else np.nan,
                    "det_a": da[g], "det_b": db[g],
                    "p": r["p"] if det_ok else np.nan,
                    "reason": r["reason"] if det_ok
                    else f"below {cfg.MIN_DETECTION:.0%} detection in both arms",
                    "delta": r["median_delta"], "sign": np.sign(r["n_pos"] - r["n_neg"]),
                    "n_pairs": r["n_pairs"], "n_pos": r["n_pos"], "n_neg": r["n_neg"],
                    "vals_a": x, "vals_b": y})
    return out


def bh(recs):
    """BH over every test drawn in one figure; direction = sign of a vs b where q < 0.05."""
    df = pd.DataFrame(recs).reset_index(drop=True)
    m = df["p"].notna()
    df["q"] = np.nan
    if m.any():
        df.loc[m, "q"] = multipletests(df.loc[m, "p"], method="fdr_bh")[1]
    df["direction"] = np.select([~m, df["q"] >= 0.05, df["sign"] > 0],
                                ["not tested", "ns", "up"], "down")
    return df


def save_source(df, name):
    df.drop(columns=["vals_a", "vals_b"], errors="ignore").to_csv(tab28b(name), index=False)


def show_tests(df, cols=("label", "gene", "n_a", "n_b", "med_a", "med_b", "q", "direction")):
    print(df[[c for c in cols if c in df.columns]].to_string(index=False, float_format="%.3g"))


# ---------------------------------------------------------------- drawing
def it(g):
    return rf"$\mathit{{{g}}}$"


def pct(x):
    if not np.isfinite(x):
        return "–"
    return f"{x:.0%}" if x >= 0.01 else f"{x:.1%}" if x >= 0.001 else "<0.1%"


def sig(q):
    return "n.t." if not np.isfinite(q) else (STAR(q) or "n.s.")


def _clean(ax):
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(labelsize=6, length=2)


def _violin(ax, v, pos, color, width=0.36, horizontal=False):
    v = np.asarray(v)[np.isfinite(v)]
    if len(v) and np.ptp(v) > 0:
        b = ax.violinplot([v], positions=[pos], widths=width, showextrema=False,
                          orientation="horizontal" if horizontal else "vertical")["bodies"][0]
        b.set_facecolor(color)
        b.set_alpha(0.3)
        b.set_edgecolor("none")


def _layout(df, group_col=None):
    """y per row (first row on top) plus one bold header line per group."""
    ys, heads, k, prev = [], [], 0, object()
    for g in (df[group_col] if group_col else [None] * len(df)):
        if group_col and g != prev:
            heads.append((k, g))
            k, prev = k + 1, g
        ys.append(k)
        k += 1
    return np.array([k - 1 - v for v in ys]), [(k - 1 - v, g) for v, g in heads], k


def _gapped(groups, gap=0.6):
    """y per row (first row on top) with `gap` rows between groups and no header rows, plus
    {group: (y_lo, y_hi)} for row_brackets."""
    ys, k, prev = [], 0.0, object()
    for g in groups:
        if ys and g != prev:
            k += gap
        ys.append(k)
        k, prev = k + 1, g
    top = ys[-1] if ys else 0.0
    y = np.array([top - v for v in ys])
    spans = {}
    for g, yi in zip(groups, y):
        lo, hi = spans.get(g, (yi, yi))
        spans[g] = (min(lo, yi), max(hi, yi))
    return y, spans, top + 1


def _heads(ax, heads):
    for yh, g in heads:
        ax.text(-0.02, yh, g, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=6.5, fontweight="bold")


def dumbbell(ax, df, group_col=None, note_col=None, brackets=False):
    """One row per test, df order top to bottom. Open dot = comparator (median of donor means),
    filled dot = MF / malignant arm coloured by direction, faint dots = donors, stars at right.
    `brackets=True` spaces groups apart instead of giving each a header row and returns their
    y spans for row_brackets, which the caller draws once the figure is final."""
    rng = np.random.default_rng(0)
    if brackets:
        y, spans, n = _gapped(list(df[group_col]))
        heads = []
    else:
        y, heads, n = _layout(df, group_col)
    for yi, (_, r) in zip(y, df.iterrows()):
        c = DIR_COLOR[r["direction"]]
        for v, col in ((r["vals_b"], HC_COL), (r["vals_a"], c)):
            v = np.asarray(v)
            ax.scatter(v, yi + rng.uniform(-0.16, 0.16, len(v)), s=3, color=col, alpha=0.3,
                       linewidth=0, zorder=2)
        ax.plot([r["med_b"], r["med_a"]], [yi, yi], color=c, linewidth=1.6, zorder=3)
        ax.scatter([r["med_b"]], [yi], s=28, facecolor="white", edgecolor=HC_COL,
                   linewidth=1.0, zorder=4)
        tested = r["direction"] != "not tested"
        ax.scatter([r["med_a"]], [yi], s=32, facecolor=c if tested else "white", edgecolor=c,
                   linewidth=1.0, zorder=5)
        note = r[note_col] if note_col and isinstance(r.get(note_col), str) else ""
        ax.text(1.01, yi, f"{sig(r['q'])}  {note}".rstrip(), transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=6)
    _heads(ax, heads)
    ax.set_yticks(y)
    ax.set_yticklabels(df["ylabel"], fontsize=6)
    ax.set_ylim(-0.7, n - 0.3)
    ax.axvline(0, color="#e0e0e0", linewidth=0.6, zorder=0)
    _clean(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return spans if brackets else None


def _x_annot(ax, i, r):
    ax.text(i, 0.97, sig(r["q"]), transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=7)
    if r["test"].startswith("paired") and np.isfinite(r["q"]):
        ax.text(i, 0.89, f"{r['n_pos']:.0f}↑/{r['n_neg']:.0f}↓", ha="center", va="top",
                transform=ax.get_xaxis_transform(), fontsize=5.5, color="#555555")


def _gene_ticks(ax, df):
    """Gene names as ticks; under each, % of cells detected (purple malignant, blue reactive)."""
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["gene"], fontsize=6.5, style="italic")
    ax.set_xlim(-0.6, len(df) - 0.4)
    for i, (_, r) in enumerate(df.iterrows()):
        for k, (v, col) in enumerate(((r["det_a"], MALIG), (r["det_b"], REACT))):
            ax.text(i, 0, pct(v), ha="center", va="top", fontsize=5.5, color=col,
                    transform=offset_copy(ax.get_xaxis_transform(), fig=ax.figure,
                                          y=-13 - 7 * k, units="points"))
    _clean(ax)


def det_label(ax):
    ax.text(0, 0, "% cells\ndetected", ha="right", va="top", fontsize=5.5, color="#555555",
            transform=offset_copy(ax.transAxes, fig=ax.figure, x=-4, y=-13, units="points"))


def paired(ax, df, violin=False, arm="MF"):
    """Clone vs reactive per gene: one line per biopsy (its two donor means), bar = median.
    Per-cell violins behind when `violin`. Returns the y top the caller should scale to."""
    genes = df["gene"].tolist()
    C = {ct: cell_values(ct, genes, arm=arm) for ct in (NONCLONAL, CLONAL)} if violin else {}
    top = 0.0
    for i, (_, r) in enumerate(df.iterrows()):
        xr, xm = i - 0.2, i + 0.2
        a, b = np.asarray(r["vals_a"]), np.asarray(r["vals_b"])
        if violin:
            for x, ct, col in ((xr, NONCLONAL, REACT), (xm, CLONAL, MALIG)):
                v = C[ct][r["gene"]].to_numpy()
                _violin(ax, v, x, col)
                if (v > 0).sum() >= 20:
                    top = max(top, float(np.quantile(v[v > 0], 0.98)))
        for va, vb in zip(a, b):
            ax.plot([xr, xm], [vb, va], color="#9e9e9e", linewidth=0.4, alpha=0.5, zorder=2)
        for x, v, col in ((xr, b, REACT), (xm, a, MALIG)):
            ax.scatter(np.full(len(v), x), v, s=6, color=col, linewidth=0, zorder=3)
            if len(v):
                ax.hlines(np.median(v), x - 0.12, x + 0.12, color="#111111", linewidth=1.2,
                          zorder=4)
        top = max(top, float(np.nanmax(np.r_[a, b, 0.0])))
        _x_annot(ax, i, r)
    _gene_ticks(ax, df)
    return top


def cell_violins(ax, df, arm="MF"):
    """Per-cell violins, reactive | malignant, donor means as dots. For genes the clone does not
    express the empty panel is the result, so it is drawn rather than dropped."""
    rng = np.random.default_rng(0)
    C = {ct: cell_values(ct, df["gene"].tolist(), arm=arm) for ct in (NONCLONAL, CLONAL)}
    for i, (_, r) in enumerate(df.iterrows()):
        for x, ct, col, v in ((i - 0.2, NONCLONAL, REACT, r["vals_b"]),
                              (i + 0.2, CLONAL, MALIG, r["vals_a"])):
            _violin(ax, C[ct][r["gene"]].to_numpy(), x, col)
            v = np.asarray(v)
            ax.scatter(x + rng.uniform(-0.06, 0.06, len(v)), v, s=6, color=col,
                       edgecolor="white", linewidth=0.2, zorder=3)
            if len(v):
                ax.hlines(np.median(v), x - 0.12, x + 0.12, color="#111111", linewidth=1.2,
                          zorder=4)
        _x_annot(ax, i, r)
    _gene_ticks(ax, df)


def group_brackets(ax, groups, y=1.03):
    """Label runs of genes above a panel: {name: [genes]} in plotted order."""
    i = 0
    for name, gs in groups.items():
        x0, x1 = i - 0.35, i + len(gs) - 1 + 0.35
        ax.plot([x0, x1], [y, y], transform=ax.get_xaxis_transform(), color="#333333",
                linewidth=0.7, clip_on=False)
        ax.text((x0 + x1) / 2, y + 0.015, name, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=6.5)
        i += len(gs)


def bracket(ax, x0, x1, y, text, h):
    ax.plot([x0, x0, x1, x1], [y - h, y, y, y - h], color="#333333", linewidth=0.6)
    ax.text((x0 + x1) / 2, y, text, ha="center", va="bottom", fontsize=6)


def row_brackets(ax, spans, sub=None, pad=6):
    """Bracket + bold label per row group ({name: (y_lo, y_hi)}) left of the y tick labels;
    `sub` = {name: grey second line}. The x is measured from the drawn tick labels, so call it
    once everything else is on the figure."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    x_px = min(t.get_window_extent(r).x0 for t in ax.get_yticklabels() if t.get_text())
    x = ax.transAxes.inverted().transform((x_px, 0))[0]
    w = 3 * fig.dpi / 72 / ax.get_window_extent(r).width
    tr = offset_copy(ax.get_yaxis_transform(), fig=fig, x=-pad, units="points")
    for name, (lo, hi) in spans.items():
        ax.plot([x, x - w, x - w, x], [lo - 0.35, lo - 0.35, hi + 0.35, hi + 0.35],
                transform=tr, color="#333333", linewidth=0.7, clip_on=False)
        mid, s = (lo + hi) / 2, (sub or {}).get(name)
        ax.text(x - w, mid, name, ha="right", va="center", fontsize=6.5, fontweight="bold",
                transform=offset_copy(tr, fig=fig, x=-3, y=3.5 if s else 0, units="points"))
        if s:
            ax.text(x - w, mid, s, ha="right", va="center", fontsize=5.5, color="#555555",
                    transform=offset_copy(tr, fig=fig, x=-3, y=-4, units="points"))


def _dot(**k):
    return plt.Line2D([], [], linestyle="", marker="o", markersize=5, **k)


def dir_handles(a="MF", b="healthy"):
    h = [_dot(color=UP, label=f"higher in {a} (q < 0.05)"),
         _dot(color=DOWN, label=f"lower in {a} (q < 0.05)"),
         _dot(color=NS, label="not significant"),
         _dot(markerfacecolor="white", markeredgecolor=NS, label="not testable")]
    return h + ([_dot(markerfacecolor="white", markeredgecolor=HC_COL, label=b)] if b else [])


ARM_HANDLES = [_dot(color=REACT, label=LABELS[NONCLONAL]), _dot(color=MALIG, label=LABELS[CLONAL])]


def caption(fig, text):
    """Wrapped caption under everything already drawn (tick labels, legends included)."""
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    fig.text(bb.x0 / fig.get_figwidth(), bb.y0 / fig.get_figheight() - 0.015,
             textwrap.fill(text, int(fig.get_figwidth() * 21)), fontsize=5.8, color="#555555",
             ha="left", va="top")


STATS_NOTE = (f"Call: mal_cluster. MF = the {len(CALL_SAMPLES)} MF samples with a call; healthy = atlas HC donors. "
              f"Replicate = donor mean log1p CP10K over >= {cfg.MIN_CELLS} cells. Malignant vs "
              f"reactive CD4: paired Wilcoxon signed-rank on the per-biopsy difference; otherwise "
              f"donor-level Mann-Whitney. BH over the tests in this figure (* < 0.05, ** < 0.01, "
              f"*** < 0.001); n.t. = not testable (< {cfg.MIN_DETECTION:.0%} of cells detected in "
              f"both arms, or < {cfg.MIN_DONORS_PER_ARM} donors per arm). Recomputed here: these "
              f"are not nb28's q-values.")

print(f"call cohort: {len(CALL_SAMPLES)} MF samples, "
      f"{adata.obs.loc[IS_MF_CALL, cfg.DONOR_KEY].nunique()} donors | "
      f"MF cells kept {int(IS_MF_CALL.sum()):,} of {int(IS_MF.sum()):,}")
```

```python  # cell 11
# ============================================================================
# §5  Stage layer — the *_stage companions of FB–FG: early vs advanced MF
# ============================================================================
# Stage = stage_class (early IA-IIA | advanced IIB+); MF samples of unknown stage are left out.
# Within a stage each figure re-runs its own test on that stage's samples: paired signed-rank for
# malignant vs reactive CD4, donor-level MWU against healthy otherwise. Advanced vs early is pooled
# across studies: a donor-level MWU on donor means for a stromal row and, for the clone, on the
# per-biopsy malignant - reactive difference averaged per donor, where study and chemistry cancel
# inside the difference. BH over every test in one figure, advanced-vs-early tests included.
_st_call = SAMPLE2STAGE.reindex(CALL_SAMPLES)
STAGE_SAMPLES = {st: sorted(_st_call.index[_st_call == st]) for st in STAGES}
N_NOSTAGE = int((~_st_call.isin(STAGES)).sum())
_s2study = (pd.Series(adata.obs[cfg.STUDY_KEY].astype(str).to_numpy(), index=_samp)
            .groupby(level=0).first())
STAGE_MIX = {st: _s2study.reindex(STAGE_SAMPLES[st]).value_counts() for st in STAGES}
EVA = "advanced vs early"
GAP_TEST = "MWU (donor, gap vs reactive)"
SHORT_ST = {"early": "early", "advanced": "adv."}


def t_gap_stage(genes, label="gap vs Reactive CD4, advanced vs early"):
    """Advanced vs early on the clone's excess over reactive in the same biopsy: per-biopsy
    difference of the two biopsy means, averaged per donor, donor-level Mann-Whitney."""
    genes = list(dict.fromkeys(genes))
    gap, det = {}, []
    for st in STAGES:
        a, da = donor_means(CLONAL, st, genes, key=cfg.SAMPLE_KEY)
        b, db = donor_means(NONCLONAL, st, genes, key=cfg.SAMPLE_KEY)
        both = a.index.intersection(b.index)
        gap[st] = (a.loc[both] - b.loc[both]).groupby(SAMPLE2DONOR.reindex(both).to_numpy()).mean()
        det += [da, db]
    det = pd.concat(det, axis=1).max(axis=1)
    out = []
    for g in genes:
        x, y = gap["advanced"][g].to_numpy(), gap["early"][g].to_numpy()
        rec = {"label": label, "gene": g, "ct": CLONAL, "arm_a": "advanced", "ct_b": CLONAL,
               "arm_b": "early", "test": GAP_TEST, "n_a": len(x), "n_b": len(y),
               "med_a": np.median(x) if len(x) else np.nan,
               "med_b": np.median(y) if len(y) else np.nan,
               "det_a": det[g], "det_b": det[g], "p": np.nan, "reason": ""}
        rec["delta"] = rec["med_a"] - rec["med_b"]
        rec["sign"] = np.sign(rec["delta"])
        if not det[g] >= cfg.MIN_DETECTION:
            rec["reason"] = f"below {cfg.MIN_DETECTION:.0%} detection in every arm"
        elif min(len(x), len(y)) < cfg.MIN_DONORS_PER_ARM:
            rec["reason"] = f"too few donors ({len(x)}, {len(y)}; need {cfg.MIN_DONORS_PER_ARM})"
        elif np.ptp(np.r_[x, y]) == 0:
            rec["reason"] = "no variance"
        else:
            rec["p"] = float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
        rec["vals_a"], rec["vals_b"] = x, y
        out.append(rec)
    return out


def stage_mwu(ct, gene, label=None, **extra):
    """ct's MF arm split by stage: early vs healthy, advanced vs healthy, advanced vs early."""
    item = f"{ct}|{gene}|HC"
    return ([dict(t_mwu(ct, gene, st, "HC", label=label), stage=st, item=item, **extra)
             for st in STAGES]
            + [dict(t_mwu(ct, gene, "advanced", "early", label=label), stage=EVA, item=item,
                    **extra)])


def stage_paired(genes, **extra):
    """Clone vs reactive in the same biopsy, early and advanced biopsies apart, then the gap
    advanced vs early."""
    return ([dict(r, stage=st, item=f"{r['gene']}|paired", **extra) for st in STAGES
             for r in t_paired(genes, arm=st)]
            + [dict(r, stage=EVA, item=f"{r['gene']}|paired", **extra)
               for r in t_gap_stage(genes)])


def stage_split(df):
    """(drawn rows, advanced-vs-early rows). Drawn = one row per item and stage, items in their
    first order, early above advanced; each advanced row carries its item's advanced-vs-early
    q and direction as `eva_q` / `eva_dir` -- the pooled test is written beside the row."""
    eva = df[df["stage"] == EVA].set_index("item")
    order = {k: i for i, k in enumerate(dict.fromkeys(df["item"]))}
    d = df[df["stage"].isin(STAGES)]
    d = (d.assign(_o=d["item"].map(order), _s=d["stage"].map(STAGES.index))
         .sort_values(["_o", "_s"], kind="stable").drop(columns=["_o", "_s"]))
    adv = (d["stage"] == "advanced").to_numpy()
    d["eva_q"] = np.where(adv, eva["q"].reindex(d["item"]).to_numpy(), np.nan)
    d["eva_dir"] = np.where(adv, eva["direction"].reindex(d["item"]).fillna("").to_numpy(), "")
    return d, df[df["stage"] == EVA]


def eva_text(q, direction, sep=" "):
    """'vs early ***↑' on an advanced row (↑ = higher in advanced); '' elsewhere."""
    if not isinstance(direction, str) or not direction:
        return ""
    arrow = {"up": "↑", "down": "↓"}.get(direction, "")
    return f"vs early{sep}{sig(q)}{arrow}"


def eva_color(direction):
    return {"up": STAGE_COL["advanced"], "down": STAGE_COL["early"]}.get(direction, "#555555")


def eva_x_annot(ax, df_eva, y=0.8):
    """Advanced-vs-early gap result above each gene of an advanced-stage paired panel."""
    for i, (_, r) in enumerate(df_eva.iterrows()):
        ax.text(i, y, eva_text(r["q"], r["direction"], sep="\n"),
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=5,
                color=eva_color(r["direction"]))


def stage_ticks(ax, stages, axis="y"):
    """Colour the tick labels by stage, in drawing order."""
    labs = ax.get_yticklabels() if axis == "y" else ax.get_xticklabels()
    for t, st in zip(labs, stages):
        t.set_color(STAGE_COL.get(st, "#111111"))


def with_stage(df):
    """ylabel + ' · early' / ' · adv.'."""
    return df.assign(ylabel=[f"{y} · {SHORT_ST[s]}" for y, s in zip(df["ylabel"], df["stage"])])


def stage_summary(df):
    """Malignant vs reactive (same biopsy) per stage, then the gap advanced vs early."""
    out = []
    for st in STAGES:
        d = df[(df["stage"] == st) & df["test"].astype(str).str.startswith("paired")]
        up = d.loc[d["direction"] == "up", "gene"].tolist()
        dn = d.loc[d["direction"] == "down", "gene"].tolist()
        n = d["n_pairs"].max() if len(d) else np.nan
        out.append(f"{st} ({f'{n:.0f}' if np.isfinite(n) else '?'} biopsies): higher in "
                   f"malignant {', '.join(up) or 'none'}; higher in reactive "
                   f"{', '.join(dn) or 'none'}")
    ch = df[(df["stage"] == EVA) & (df["test"] == GAP_TEST) & df["direction"].isin(["up", "down"])]
    out.append("malignant - reactive gap, advanced vs early: " + (", ".join(
        f"{g} {'higher' if d_ == 'up' else 'lower'} in advanced"
        for g, d_ in zip(ch["gene"], ch["direction"])) or "no change"))
    return out


STAGE_HANDLES = [_dot(color=STAGE_COL[st], label=f"{st} MF (row label colour)") for st in STAGES]
STAGE_NOTE = (
    "Stage = stage_class (early IA-IIA, advanced IIB+): "
    + "; ".join(f"{st} {len(STAGE_SAMPLES[st])} samples / "
                f"{SAMPLE2DONOR.reindex(STAGE_SAMPLES[st]).nunique()} donors ("
                + ", ".join(f"{s} {n}" for s, n in STAGE_MIX[st].items()) + ")"
                for st in STAGES)
    + f"; {N_NOSTAGE} called MF samples of unknown stage left out. Within a stage: the figure's "
    "own test on that stage's samples. Advanced vs early, pooled across studies: donor-level "
    "Mann-Whitney on donor means, and for malignant vs reactive CD4 on the per-biopsy difference "
    "averaged per donor ('vs early' beside an advanced row; ↑ = higher in advanced). BH over every "
    "test in this figure, both kinds.")

for st in STAGES:
    print(f"{st}: {len(STAGE_SAMPLES[st])} samples, "
          f"{SAMPLE2DONOR.reindex(STAGE_SAMPLES[st]).nunique()} donors | "
          + ", ".join(f"{s} {n}" for s, n in STAGE_MIX[st].items()))
print(f"unknown stage, left out: {N_NOSTAGE} of {len(CALL_SAMPLES)} called MF samples")
```

```python  # cell 12
# ============================================================================
# FB  The clone does not instruct the microenvironment; what it does carry
# ============================================================================
# B1 is the T-cell -> stroma ligands the clone would need to instruct fibroblasts; B2, on the
# same y scale, is what separates clone from reactive in the same biopsy. B1 is drawn, not
# dropped: empty panels are the result.
FB1 = {"T-cell -> stroma ligands": ["CD40LG", "LTA", "IL17A", "IL22", "IL11", "TGFB1"]}
FB2 = {"protease inhibitors": ["TIMP1", "CSTB", "SERPINB1", "SERPINB9"],
       "lysosomal cathepsins": ["CTSA", "CTSD", "CTSH"],
       "granzymes": ["GZMB", "GZMM"]}
_g1 = [g for gs in FB1.values() for g in gs]
_g2 = [g for gs in FB2.values() for g in gs]

FB = bh(t_paired(_g1) + t_paired(_g2))
FB["panel"] = ["B1"] * len(_g1) + ["B2"] * len(_g2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.4), sharey=True,
                               gridspec_kw={"width_ratios": [len(_g1), len(_g2)],
                                            "wspace": 0.08})
cell_violins(ax1, FB[FB["panel"] == "B1"])
top = paired(ax2, FB[FB["panel"] == "B2"])
top = max(top, *(float(np.nanmax(np.r_[a, b, 0.0])) for a, b in
                 FB.loc[FB["panel"] == "B1", ["vals_a", "vals_b"]].itertuples(index=False)))
ax1.set_ylim(-0.04 * top, top / 0.82)
ax1.set_ylabel(XLAB, fontsize=7)
ax2.tick_params(labelleft=False)
det_label(ax1)
group_brackets(ax1, FB1)
group_brackets(ax2, FB2)
fig.legend(handles=ARM_HANDLES, fontsize=6.5, frameon=False, ncol=2, loc="lower right",
           bbox_to_anchor=(0.99, 0.99))
DC._finish(fig, fig28b("FB_what_the_clone_carries"), True)
save_source(FB, "FB_source")
logfig("FB", fig28b("FB_what_the_clone_carries"),
       "T-cell -> stroma ligands the clone lacks vs its inhibitor / lysosomal / granzyme cargo",
       "object, mal_cluster cohort; paired signed-rank on per-biopsy means recomputed, BH within figure")
show_tests(FB, ("panel", "gene", "n_pairs", "med_a", "med_b", "n_pos", "n_neg", "det_a",
                "det_b", "q", "direction"))
```

```python  # cell 13
# ============================================================================
# FB_stage  FB split by stage: early biopsies on top, advanced below
# ============================================================================
FBS = bh(stage_paired(_g1, panel="B1") + stage_paired(_g2, panel="B2"))

fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.8), sharey=True,
                         gridspec_kw={"width_ratios": [len(_g1), len(_g2)], "wspace": 0.08,
                                      "hspace": 0.6})
top = 0.0
for i, st in enumerate(STAGES):
    ax1, ax2 = axes[i]
    d1 = FBS[(FBS["panel"] == "B1") & (FBS["stage"] == st)]
    d2 = FBS[(FBS["panel"] == "B2") & (FBS["stage"] == st)]
    cell_violins(ax1, d1, arm=st)
    top = max(top, paired(ax2, d2, arm=st),
              *(float(np.nanmax(np.r_[a, b, 0.0]))
                for a, b in d1[["vals_a", "vals_b"]].itertuples(index=False)))
    ax1.set_ylabel(f"{st.upper()} MF\n{XLAB}", fontsize=7, color=STAGE_COL[st])
    ax2.tick_params(labelleft=False)
    det_label(ax1)
    group_brackets(ax1, FB1)
    group_brackets(ax2, FB2)
for ax, panel in zip(axes[1], ("B1", "B2")):
    eva_x_annot(ax, FBS[(FBS["panel"] == panel) & (FBS["stage"] == EVA)])
axes[0][0].set_ylim(-0.04 * top, top / 0.72)
fig.subplots_adjust(top=0.8)
fig.legend(handles=ARM_HANDLES, fontsize=6.5, frameon=False, ncol=2, loc="upper right",
           bbox_to_anchor=(0.99, 0.03))
fig.suptitle("T-cell → stroma ligands, protease inhibitors, lysosomal cathepsins and granzymes "
             "in malignant vs reactive CD4, early and advanced MF", fontsize=9, x=0.0, ha="left",
             y=0.97)
print("\n".join(stage_summary(FBS)))
caption(fig, STATS_NOTE + " " + STAGE_NOTE)
DC._finish(fig, fig28b("FB_stage_what_the_clone_carries"), True)
save_source(FBS, "FB_stage_source")
logfig("FB_stage", fig28b("FB_stage_what_the_clone_carries"),
       "FB per stage: T-cell -> stroma ligands and the clone's cargo, early vs advanced MF",
       "object, mal_cluster cohort by stage_class; paired signed-rank per stage + donor MWU on "
       "the gap, BH within figure")
show_tests(FBS, ("panel", "stage", "gene", "n_a", "n_b", "med_a", "med_b", "n_pos", "n_neg",
                 "q", "direction"))
```

```python  # cell 14
# ============================================================================
# FC  LIGHT -> LTbetaR / HVEM -> RELB
# ============================================================================
# Read top to bottom as a chain: the ligand on the clone (1), its receptors on fibroblasts (2),
# the transcription factor they drive (3). One x axis, so the three rows are on one scale.
FC_FIBRO = ["F_apCAF", "F_inflammatory", "F_structural"]
prefetch(FC_FIBRO + ["F_mesenchymal"], ["TNFSF14", "TNFRSF14", "LTBR", "CD40", "RELB"])
FC_ROWS = {
    1: t_paired(["TNFSF14"])
       + [t_mwu(CLONAL, "TNFSF14", "MF", "MF", ct_b=NONCLONAL,
                label="vs Reactive CD4 (across donors)"),
          t_mwu(CLONAL, "TNFSF14", "MF", "HC", label="vs healthy-donor CD4")],
    2: [t_mwu(ct, "TNFRSF14") for ct in FC_FIBRO]
       + [t_mwu("F_structural", "LTBR")]
       + [t_mwu(ct, "CD40") for ct in FC_FIBRO],
    3: [t_mwu(ct, "RELB") for ct in ["F_mesenchymal", "F_inflammatory", "F_structural"]],
}
FC = bh([dict(r, row=k) for k, rs in FC_ROWS.items() for r in rs])
FC["ylabel"] = [r["label"] if r["row"] == 1 else f"{it(r['gene'])} · {LABELS[r['ct']]}"
                for _, r in FC.iterrows()]
# TNFSF14 is detected in few cells: say how few, and how many biopsies voted each way
FC["note"] = [("" if r["row"] != 1 else
               (f"{r['n_pos']:.0f}↑/{r['n_neg']:.0f}↓ · " if r["test"].startswith("paired")
                else "") + f"det {pct(r['det_a'])} | {pct(r['det_b'])}")
              for _, r in FC.iterrows()]
FC_TITLES = ["LIGAND  TNFSF14 (LIGHT) in malignant CD4 (filled) vs comparator (open)",
             "RECEPTORS  fibroblasts, MF vs healthy skin",
             "EFFECTOR  RELB in fibroblasts, MF vs healthy skin"]

nr = FC["row"].value_counts().reindex([1, 2, 3]).to_numpy()
fig = plt.figure(figsize=(7.0, 0.26 * nr.sum() + 2.4))
gs = fig.add_gridspec(3, 1, height_ratios=nr + 0.6, hspace=0.5,
                      left=0.40, right=0.76, top=0.9, bottom=0.12)
axes = []
for k in range(3):
    ax = fig.add_subplot(gs[k], sharex=axes[0] if axes else None)
    dumbbell(ax, FC[FC["row"] == k + 1], note_col="note")
    ax.set_title(FC_TITLES[k], fontsize=7, loc="left")
    if k < 2:
        ax.tick_params(labelbottom=False)
    axes.append(ax)
xmax = max(np.nanmax(np.r_[v, 0.0]) for col in ("vals_a", "vals_b") for v in FC[col])
axes[0].set_xlim(-0.02 * xmax, 1.05 * xmax)
axes[-1].set_xlabel(XLAB, fontsize=7)

# the chain: numbered rows joined by downward arrows in the left margin
cy = [(ax.get_position().y0 + ax.get_position().y1) / 2 for ax in axes]
for k, yc in enumerate(cy):
    fig.text(0.03, yc, str(k + 1), ha="center", va="center", fontsize=10, fontweight="bold",
             color="white", bbox=dict(boxstyle="circle,pad=0.35", facecolor="#333333",
                                      edgecolor="none"))
for y0, y1 in zip(cy[:-1], cy[1:]):
    fig.add_artist(FancyArrowPatch((0.03, y0 - 0.04), (0.03, y1 + 0.04),
                                   transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=10, color="#333333", linewidth=1.0))
fig.legend(handles=dir_handles("malignant / MF", "comparator (reactive or healthy)"),
           fontsize=6, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.55, 0.05))
fig.suptitle("LIGHT → LTβR / HVEM → RELB", fontsize=9.5, x=0.0, ha="left",
             y=0.99)
DC._finish(fig, fig28b("FC_light_ltbr_relb"), True)
save_source(FC, "FC_source")
logfig("FC", fig28b("FC_light_ltbr_relb"), "the LIGHT -> LTbetaR/HVEM -> RELB chain",
       "object, mal_cluster cohort; paired signed-rank + donor-level MWU recomputed, BH within figure")
show_tests(FC, ("row", "label", "gene", "n_a", "n_b", "med_a", "med_b", "det_a", "q",
                "direction"))
```

```python  # cell 15
# ============================================================================
# FC_stage  FC split by stage: the LIGHT chain in early and in advanced MF
# ============================================================================
# Every FC row twice, early above advanced (row label colour = stage). Row 1: the clone vs
# reactive in the same biopsy and vs healthy-donor CD4; rows 2-3: fibroblasts vs healthy skin.
# The advanced-vs-early result is written beside each advanced row.
FCS_ROWS = {
    1: stage_paired(["TNFSF14"]) + stage_mwu(CLONAL, "TNFSF14", label="vs healthy-donor CD4"),
    2: ([r for ct in FC_FIBRO for r in stage_mwu(ct, "TNFRSF14")]
        + stage_mwu("F_structural", "LTBR")
        + [r for ct in FC_FIBRO for r in stage_mwu(ct, "CD40")]),
    3: [r for ct in ["F_mesenchymal", "F_inflammatory", "F_structural"]
        for r in stage_mwu(ct, "RELB")],
}
FCS = bh([dict(r, row=k) for k, rs in FCS_ROWS.items() for r in rs])
FCS["ylabel"] = [r["label"] if r["row"] == 1 else f"{it(r['gene'])} · {LABELS[r['ct']]}"
                 for _, r in FCS.iterrows()]
FCD, _ = stage_split(FCS)
FCD = with_stage(FCD)
FCD["note"] = [" · ".join(x for x in (
    f"{r['n_pos']:.0f}↑/{r['n_neg']:.0f}↓" if str(r["test"]).startswith("paired")
    and np.isfinite(r["q"]) else "",
    eva_text(r["eva_q"], r["eva_dir"])) if x) for _, r in FCD.iterrows()]

nr = FCD["row"].value_counts().reindex([1, 2, 3]).to_numpy()
fig = plt.figure(figsize=(7.4, 0.26 * nr.sum() + 2.6))
gs = fig.add_gridspec(3, 1, height_ratios=nr + 0.6, hspace=0.5,
                      left=0.40, right=0.74, top=0.9, bottom=0.12)
axes = []
for k in range(3):
    ax = fig.add_subplot(gs[k], sharex=axes[0] if axes else None)
    d = FCD[FCD["row"] == k + 1]
    dumbbell(ax, d, note_col="note")
    stage_ticks(ax, d["stage"])
    ax.set_title(FC_TITLES[k], fontsize=7, loc="left")
    if k < 2:
        ax.tick_params(labelbottom=False)
    axes.append(ax)
xmax = max(np.nanmax(np.r_[v, 0.0]) for col in ("vals_a", "vals_b") for v in FCD[col])
axes[0].set_xlim(-0.02 * xmax, 1.05 * xmax)
axes[-1].set_xlabel(XLAB, fontsize=7)

cy = [(ax.get_position().y0 + ax.get_position().y1) / 2 for ax in axes]
for k, yc in enumerate(cy):
    fig.text(0.03, yc, str(k + 1), ha="center", va="center", fontsize=10, fontweight="bold",
             color="white", bbox=dict(boxstyle="circle,pad=0.35", facecolor="#333333",
                                      edgecolor="none"))
for y0, y1 in zip(cy[:-1], cy[1:]):
    fig.add_artist(FancyArrowPatch((0.03, y0 - 0.04), (0.03, y1 + 0.04),
                                   transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=10, color="#333333", linewidth=1.0))
fig.legend(handles=dir_handles("malignant / MF", "comparator (reactive or healthy)")
           + STAGE_HANDLES, fontsize=6, frameon=False, ncol=3, loc="upper center",
           bbox_to_anchor=(0.55, 0.05))
_nsig = {st: int(FCD.loc[FCD["stage"] == st, "direction"].isin(["up", "down"]).sum())
         for st in STAGES}
_nev = int(FCS.loc[FCS["stage"] == EVA, "direction"].isin(["up", "down"]).sum())
fig.suptitle("LIGHT → LTβR / HVEM → RELB, early and advanced MF", fontsize=9, x=0.0, ha="left",
             y=0.99)
print(f"{_nsig['early']} rows differ in early, {_nsig['advanced']} in advanced MF; "
      f"{_nev} differ advanced vs early")
caption(fig, STATS_NOTE + " " + STAGE_NOTE)
DC._finish(fig, fig28b("FC_stage_light_ltbr_relb"), True)
save_source(FCS, "FC_stage_source")
logfig("FC_stage", fig28b("FC_stage_light_ltbr_relb"),
       "the LIGHT -> LTbetaR/HVEM -> RELB chain, early vs advanced MF",
       "object, mal_cluster cohort by stage_class; paired signed-rank + donor-level MWU per "
       "stage, donor MWU advanced vs early, BH within figure")
show_tests(FCS, ("row", "stage", "label", "gene", "n_a", "n_b", "med_a", "med_b", "q",
                 "direction"))
```

```python  # cell 16
# ============================================================================
# FD  Th2 axis
# ============================================================================
# Left to right: the transcription factor is on the clone (D1), the cytokines are below
# detection in both CD4 arms (D2, drawn on D1's scale), the receptor and its output are on the
# stroma (D3). IL4R is shown in all five fibroblast subsets: the reproducibility is the point.
FD1 = ["GATA3", "BATF", "STAT3", "TOX"]
FD2 = ["IL4", "IL5", "IL13", "IL9", "IL31", "IL22"]
FD3 = ([("IL4R", ct, "receptor") for ct in ["F_structural", "F_inflammatory", "F_apCAF",
                                            "F_mesenchymal", "F_reticular"]]
       + [(g, "F_structural", "output") for g in ["HAS2", "POSTN", "TNC"]]
       + [("TSLP", "Mac_FOLR2", "alarmin")])
prefetch(["F_structural", "F_inflammatory", "F_apCAF", "F_mesenchymal", "F_reticular",
          "Mac_FOLR2"], ["IL4R", "HAS2", "POSTN", "TNC", "TSLP"])

FD = bh(t_paired(FD1) + t_paired(FD2) + [dict(t_mwu(ct, g), group=grp) for g, ct, grp in FD3])
FD["panel"] = ["D1"] * len(FD1) + ["D2"] * len(FD2) + ["D3"] * len(FD3)
FD["ylabel"] = [f"{it(r['gene'])} · {LABELS[r['ct']]}" for _, r in FD.iterrows()]

fig = plt.figure(figsize=(12.5, 3.6))
gs = fig.add_gridspec(1, 4, width_ratios=[len(FD1), len(FD2), 2.3, 4.2], wspace=0.12,
                      left=0.05, right=0.93, top=0.8, bottom=0.2)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharey=ax1)
ax3 = fig.add_subplot(gs[3])
top = paired(ax1, FD[FD["panel"] == "D1"], violin=True)
cell_violins(ax2, FD[FD["panel"] == "D2"])
ax1.set_ylim(-0.04 * top, top / 0.82)
ax1.set_ylabel("log1p CP10K (violin = cells, dots = donor means)", fontsize=6.5)
ax2.tick_params(labelleft=False)
det_label(ax1)
dumbbell(ax3, FD[FD["panel"] == "D3"], group_col="group")
ax3.set_xlabel(XLAB, fontsize=7)
_circ = [g for g in FD1 if g in CALL_PROGRAM]
ax1.set_title("D1  clone: transcription factors (paired)"
              + (f"\n† in the calling program: {', '.join(_circ)}" if _circ else ""),
              fontsize=7.5, loc="left")
ax2.set_title("D2  Th2 cytokines in CD4", fontsize=7.5, loc="left")
ax3.set_title("D3  stroma: receptor and output, MF vs healthy", fontsize=7.5, loc="left")
fig.legend(handles=ARM_HANDLES, fontsize=6.5, frameon=False, ncol=2, loc="upper left",
           bbox_to_anchor=(0.05, 0.02))
fig.legend(handles=dir_handles("MF", "healthy"), fontsize=6, frameon=False, ncol=3,
           loc="upper right", bbox_to_anchor=(0.97, 0.02))
fig.suptitle("Th2 axis", fontsize=9.5, x=0.0, ha="left", y=0.99)
DC._finish(fig, fig28b("FD_th2_axis"), True)
save_source(FD, "FD_source")
logfig("FD", fig28b("FD_th2_axis"), "the Th2 axis: TF on the clone, cytokines, stromal answer",
       "object, mal_cluster cohort; paired signed-rank + donor-level MWU recomputed, BH within figure")
show_tests(FD, ("panel", "gene", "ct", "n_a", "n_b", "med_a", "med_b", "det_a", "det_b", "q",
                "direction"))
```

```python  # cell 17
# ============================================================================
# FD_stage  FD split by stage: clone panels per stage (rows), stroma per stage (right)
# ============================================================================
FDS = bh(stage_paired(FD1, panel="D1") + stage_paired(FD2, panel="D2")
         + [r for g, ct, grp in FD3 for r in stage_mwu(ct, g, group=grp, panel="D3")])
FDS["ylabel"] = [f"{it(r['gene'])} · {LABELS[r['ct']]}" for _, r in FDS.iterrows()]

fig = plt.figure(figsize=(12.5, 6.8))
gs = fig.add_gridspec(2, 4, width_ratios=[len(FD1), len(FD2), 2.3, 4.2], wspace=0.12,
                      hspace=0.6, left=0.05, right=0.93, top=0.8, bottom=0.14)
top, row_axes = 0.0, []
for i, st in enumerate(STAGES):
    ax1 = fig.add_subplot(gs[i, 0], sharey=row_axes[0][0] if row_axes else None)
    ax2 = fig.add_subplot(gs[i, 1], sharey=ax1)
    top = max(top, paired(ax1, FDS[(FDS["panel"] == "D1") & (FDS["stage"] == st)],
                          violin=True, arm=st))
    cell_violins(ax2, FDS[(FDS["panel"] == "D2") & (FDS["stage"] == st)], arm=st)
    ax1.set_ylabel(f"{st.upper()} MF\nlog1p CP10K", fontsize=6.5, color=STAGE_COL[st])
    ax2.tick_params(labelleft=False)
    det_label(ax1)
    row_axes.append((ax1, ax2))
row_axes[0][0].set_ylim(-0.04 * top, top / 0.72)
for ax, panel in zip(row_axes[1], ("D1", "D2")):
    eva_x_annot(ax, FDS[(FDS["panel"] == panel) & (FDS["stage"] == EVA)])
row_axes[0][0].set_title("D1  clone: transcription factors (paired)"
                         + (f"\n† in the calling program: {', '.join(_circ)}" if _circ else ""),
                         fontsize=7.5, loc="left")
row_axes[0][1].set_title("D2  Th2 cytokines in CD4", fontsize=7.5, loc="left")

ax3 = fig.add_subplot(gs[:, 3])
D3S, _ = stage_split(FDS[FDS["panel"] == "D3"])
D3S = with_stage(D3S)
D3S["note"] = [eva_text(q, d) for q, d in zip(D3S["eva_q"], D3S["eva_dir"])]
dumbbell(ax3, D3S, group_col="group", note_col="note")
stage_ticks(ax3, D3S["stage"])
ax3.set_xlabel(XLAB, fontsize=7)
ax3.set_title("D3  stroma: receptor and output, MF stage vs healthy", fontsize=7.5, loc="left")

fig.legend(handles=ARM_HANDLES, fontsize=6.5, frameon=False, ncol=2, loc="upper left",
           bbox_to_anchor=(0.05, 0.03))
fig.legend(handles=dir_handles("MF", "healthy") + STAGE_HANDLES, fontsize=6, frameon=False,
           ncol=3, loc="upper right", bbox_to_anchor=(0.97, 0.03))
_d3 = {st: int((D3S.loc[D3S["stage"] == st, "direction"] == "up").sum()) for st in STAGES}
fig.suptitle("Th2 axis, early and advanced MF", fontsize=9.5, x=0.0, ha="left", y=0.99)
print("\n".join([f"stroma above healthy in {_d3['early']}/{len(FD3)} rows in early, "
                 f"{_d3['advanced']}/{len(FD3)} in advanced MF"]
                + stage_summary(FDS[FDS["panel"].isin(["D1", "D2"])])))
caption(fig, STATS_NOTE + " " + STAGE_NOTE)
DC._finish(fig, fig28b("FD_stage_th2_axis"), True)
save_source(FDS, "FD_stage_source")
logfig("FD_stage", fig28b("FD_stage_th2_axis"), "the Th2 axis, early vs advanced MF",
       "object, mal_cluster cohort by stage_class; paired signed-rank + donor-level MWU per "
       "stage, donor MWU advanced vs early, BH within figure")
show_tests(FDS, ("panel", "stage", "gene", "ct", "n_a", "n_b", "med_a", "med_b", "q",
                 "direction"))
```

```python  # cell 18
# ============================================================================
# FE  Hyaluronan — synthesis, fragmentation, receptor
# ============================================================================
# No curated ligand-receptor edge exists for HA, so the claim rests on who makes it (synthesis),
# who cuts it (fragmentation) and who can bind the fragments (CD44) -- on the stroma in E1 (MF vs
# healthy), on the CD4 arms in E2.
FE1 = ([("HAS2", ct, "synthesis") for ct in ["F_structural", "F_mesenchymal"]]
       + [("HAS1", "F_inflammatory", "synthesis")]
       + [("CEMIP", ct, "fragmentation") for ct in ["F_apCAF", "F_inflammatory", "F_reticular",
                                                    "F_structural"]]
       + [("HYAL2", ct, "fragmentation") for ct in ["F_mesenchymal", "Mac_FOLR2", "Mono_CD14",
                                                    "cDC2"]]
       + [("CD44", ct, "receptor") for ct in ["F_reticular", "F_inflammatory", "F_structural"]])
prefetch(sorted({ct for _, ct, _ in FE1}), ["HAS1", "HAS2", "CEMIP", "HYAL2", "CD44"])
prefetch([CLONAL, NONCLONAL], ["CD44", "HYAL3"])

_e1 = [dict(t_mwu(ct, g), group=grp) for g, ct, grp in FE1]

_e2 = (t_paired(["CD44", "HYAL3"])
       + [t_mwu(CLONAL, "CD44", "MF", "HC", label="Malignant vs healthy-donor CD4"),
          t_mwu(NONCLONAL, "CD44", "MF", "HC", label="Reactive vs healthy-donor CD4")])
FE = bh(_e1 + _e2)
FE["panel"] = ["E1"] * len(_e1) + ["E2"] * len(_e2)
FE["ylabel"] = [f"{it(r['gene'])} · {LABELS[r['ct']]}" for _, r in FE.iterrows()]
E2 = FE[FE["panel"] == "E2"]


def _q(gene, test, ct=CLONAL):
    r = E2[(E2["gene"] == gene) & E2["test"].str.startswith(test) & (E2["ct"] == ct)]
    return float(r["q"].iloc[0]) if len(r) else np.nan


def _bars(ax, gene, arms):
    """Bar = median of donor means, dots = donors. `arms` = [(cell type, 'MF' | 'HC')]."""
    rng = np.random.default_rng(0)
    top = 0.0
    for i, (ct, arm) in enumerate(arms):
        v = donor_means(ct, arm, [gene])[0][gene].to_numpy()
        if not len(v):
            continue
        ax.bar(i, np.median(v), width=0.62, color=ARM_COL[ct], alpha=0.3,
               edgecolor=ARM_COL[ct], linewidth=0.8)
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=6, color=ARM_COL[ct],
                   edgecolor="white", linewidth=0.2, zorder=3)
        top = max(top, float(v.max()))
    short = {CLONAL: "Malignant\nCD4", NONCLONAL: "Reactive\nCD4", HC_CD4: "Healthy-\ndonor CD4"}
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([short[ct] + (f"\n{arm}" if arm in STAGES else "") for ct, arm in arms],
                       fontsize=6)
    ax.set_title(it(gene), fontsize=8)
    _clean(ax)
    return top


def _stroma_bars(ax, df):
    """One slot per test: healthy bar (hollow) | MF bar (coloured by direction), dots = donors,
    bar = median of donor means, star = MF vs healthy."""
    rng = np.random.default_rng(0)
    top = 0.0
    for i, (_, r) in enumerate(df.iterrows()):
        c = DIR_COLOR[r["direction"]]
        for x, v, fc, ec in ((i - 0.2, r["vals_b"], "white", HC_COL), (i + 0.2, r["vals_a"], c, c)):
            v = np.asarray(v)
            if not len(v):
                continue
            ax.bar(x, np.median(v), width=0.36, color=fc, alpha=0.3 if fc != "white" else 1,
                   edgecolor=ec, linewidth=0.8)
            ax.scatter(x + rng.uniform(-0.08, 0.08, len(v)), v, s=4, color=ec,
                       edgecolor="white", linewidth=0.2, zorder=3)
            top = max(top, float(v.max()))
        ax.text(i, 0.97, sig(r["q"]), transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=7)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["ylabel"], rotation=45, ha="right", fontsize=6)
    ax.set_xlim(-0.6, len(df) - 0.4)
    _clean(ax)
    return top


E1 = FE[FE["panel"] == "E1"]
fig = plt.figure(figsize=(10.5, 3.8))
gs = fig.add_gridspec(1, 3, width_ratios=[len(E1), 2.4, 1.6], wspace=0.3,
                      left=0.06, right=0.98, top=0.82, bottom=0.34)
ax1, ax2, ax3 = (fig.add_subplot(gs[k]) for k in range(3))
t = _stroma_bars(ax1, E1)
ax1.set_ylim(0, t / 0.85)
ax1.set_ylabel(XLAB, fontsize=7)
group_brackets(ax1, {g: d for g, d in E1.groupby("group", sort=False)["gene"]})
ax1.set_title("E1  stroma and myeloid, MF vs healthy", fontsize=7.5, loc="left", pad=16)

t = _bars(ax2, "CD44", [(CLONAL, "MF"), (NONCLONAL, "MF"), (HC_CD4, "HC")])
for k, (x0, x1, q) in enumerate([(0, 1, _q("CD44", "paired")),
                                 (1, 2, _q("CD44", "MWU", NONCLONAL)),
                                 (0, 2, _q("CD44", "MWU", CLONAL))]):
    bracket(ax2, x0, x1, t * (1.08 + 0.13 * k), sig(q), h=0.03 * t)
ax2.set_ylim(0, t * 1.5)
ax2.set_ylabel(XLAB, fontsize=7)
t = _bars(ax3, "HYAL3", [(CLONAL, "MF"), (NONCLONAL, "MF")])
t = t if t > 0 else 1.0
bracket(ax3, 0, 1, t * 1.08, sig(_q("HYAL3", "paired")), h=0.03 * t)
ax3.set_ylim(0, t * 1.3)
ax2.text(0.0, 1.13, "E2  CD4", fontsize=7.5, transform=ax2.transAxes, ha="left")
fig.legend(handles=dir_handles("MF", "healthy"), fontsize=6, frameon=False, ncol=5,
           loc="upper left", bbox_to_anchor=(0.02, 0.04))
fig.suptitle("Hyaluronan: synthesis, fragmentation and CD44 in the stroma and CD4",
             fontsize=9.5, x=0.0, ha="left", y=0.99)
DC._finish(fig, fig28b("FE_hyaluronan"), True)
save_source(FE, "FE_source")
logfig("FE", fig28b("FE_hyaluronan"), "the hyaluronan axis: synthesis, fragmentation, CD44 in stroma and CD4",
       "object, mal_cluster cohort; paired signed-rank + donor-level MWU recomputed, BH within figure")
show_tests(FE, ("panel", "label", "gene", "n_a", "n_b", "med_a", "med_b", "q", "direction"))
```

```python  # cell 19
# ============================================================================
# FE_stage  FE split by stage
# ============================================================================
# E1: each stroma row twice (early, advanced) against healthy, with a bracket for advanced vs
# early. E2: CD4 arms per stage, brackets = the tests drawn (paired within stage, clone advanced
# vs early, clone per stage vs healthy CD4); the gap test is in each panel title.
FES = bh([r for g, ct, grp in FE1 for r in stage_mwu(ct, g, group=grp, panel="E1")]
         + stage_paired(["CD44", "HYAL3"], panel="E2")
         + stage_mwu(CLONAL, "CD44", label="Malignant vs healthy-donor CD4", panel="E2"))
FES["ylabel"] = [f"{it(r['gene'])} · {LABELS[r['ct']]}" for _, r in FES.iterrows()]
E1S, _ = stage_split(FES[FES["panel"] == "E1"])
E1S = with_stage(E1S)
E2S = FES[FES["panel"] == "E2"]


def _qs(gene, stage, kind):
    """q of one E2 test: kind = 'paired' | 'gap' | 'mwu' (the clone's own donor MWU)."""
    t = E2S["test"].astype(str)
    hit = {"paired": t.str.startswith("paired"), "gap": t == GAP_TEST,
           "mwu": t == "MWU (donor)"}[kind]
    r = E2S[(E2S["gene"] == gene) & (E2S["stage"] == stage) & hit]
    return float(r["q"].iloc[0]) if len(r) else np.nan


fig = plt.figure(figsize=(16.5, 5.2))
gs = fig.add_gridspec(1, 3, width_ratios=[len(E1S), 6.5, 5.2], wspace=0.25,
                      left=0.05, right=0.98, top=0.74, bottom=0.36)
ax1, ax2, ax3 = (fig.add_subplot(gs[k]) for k in range(3))
t = _stroma_bars(ax1, E1S)
for i, (_, r) in enumerate(E1S.iterrows()):
    if r["stage"] == "advanced":
        bracket(ax1, i - 0.8, i + 0.2, t * 1.1, eva_text(r["eva_q"], r["eva_dir"]).replace(
            "vs early ", ""), h=0.02 * t)
ax1.set_ylim(0, t * 1.3)
ax1.set_ylabel(XLAB, fontsize=7)
stage_ticks(ax1, E1S["stage"], axis="x")
group_brackets(ax1, {g: d for g, d in E1S.groupby("group", sort=False)["gene"]}, y=1.04)
ax1.set_title("E1  stroma and myeloid, MF stage vs healthy (bracket: advanced vs early)",
              fontsize=7.5, loc="left", pad=16)

_e2 = [(CLONAL, "early"), (NONCLONAL, "early"), (CLONAL, "advanced"), (NONCLONAL, "advanced")]
t = _bars(ax2, "CD44", _e2 + [(HC_CD4, "HC")])
for k, (x0, x1, q) in enumerate([(0, 1, _qs("CD44", "early", "paired")),
                                 (2, 3, _qs("CD44", "advanced", "paired")),
                                 (0, 2, _qs("CD44", EVA, "mwu")),
                                 (0, 4, _qs("CD44", "early", "mwu")),
                                 (2, 4, _qs("CD44", "advanced", "mwu"))]):
    bracket(ax2, x0, x1, t * (1.08 + 0.13 * k), sig(q), h=0.03 * t)
ax2.set_ylim(0, t * 1.8)
ax2.set_ylabel(XLAB, fontsize=7)
ax2.set_title(f"{it('CD44')}  (gap vs reactive, advanced vs early: "
              f"{sig(_qs('CD44', EVA, 'gap'))})", fontsize=6.5)
t = _bars(ax3, "HYAL3", _e2)
t = t if t > 0 else 1.0
for k, (x0, x1, q) in enumerate([(0, 1, _qs("HYAL3", "early", "paired")),
                                 (2, 3, _qs("HYAL3", "advanced", "paired"))]):
    bracket(ax3, x0, x1, t * 1.08, sig(q), h=0.03 * t)
ax3.set_ylim(0, t * 1.3)
ax3.set_title(f"{it('HYAL3')}  (gap, adv. vs early: {sig(_qs('HYAL3', EVA, 'gap'))})",
              fontsize=6.5)
ax2.text(0.0, 1.16, "E2  CD4 by stage", fontsize=7.5, transform=ax2.transAxes, ha="left")
fig.legend(handles=dir_handles("MF", "healthy") + STAGE_HANDLES, fontsize=6, frameon=False,
           ncol=6, loc="upper left", bbox_to_anchor=(0.02, -0.02))
_up = {st: int((E1S.loc[E1S["stage"] == st, "direction"] == "up").sum()) for st in STAGES}
_n1 = len(FE1)
_ev = int(FES.loc[(FES["panel"] == "E1") & (FES["stage"] == EVA), "direction"]
          .isin(["up", "down"]).sum())
fig.suptitle("Hyaluronan: synthesis, fragmentation and CD44 in the stroma and CD4, early and "
             "advanced MF", fontsize=9.5, x=0.0, ha="left", y=0.99)
print("\n".join([f"stroma above healthy in {_up['early']}/{_n1} rows in early, "
                 f"{_up['advanced']}/{_n1} in advanced MF; {_ev} rows differ advanced vs early"]
                + stage_summary(E2S)))
caption(fig, STATS_NOTE + " " + STAGE_NOTE)
DC._finish(fig, fig28b("FE_stage_hyaluronan"), True)
save_source(FES, "FE_stage_source")
logfig("FE_stage", fig28b("FE_stage_hyaluronan"),
       "the hyaluronan axis in stroma and CD4, early vs advanced MF",
       "object, mal_cluster cohort by stage_class; paired signed-rank + donor-level MWU per "
       "stage, donor MWU advanced vs early, BH within figure")
show_tests(FES, ("panel", "stage", "label", "gene", "n_a", "n_b", "med_a", "med_b", "q",
                 "direction"))
```

```python  # cell 20
# ============================================================================
# FF  The TGF-β loop runs inside the fibroblast compartment
# ============================================================================
# Left: every stage of the loop -- latent ligand, docking, activation, receptor, output -- in
# fibroblasts, MF vs healthy. Right: the ligands and activators in CD4 on the same x scale,
# clone vs reactive in the same biopsy. Myeloid TGFB1 is up too; it belongs with FB (the clone
# does not instruct), not here.
FF_STAGES = {
    "LATENT LIGAND": [("TGFB1", ["F_inflammatory", "F_apCAF"]),
                      ("TGFB3", ["F_inflammatory", "F_apCAF"])],
    "DOCKING": [("LRRC32", ["F_structural", "F_inflammatory", "F_apCAF"])],
    "ACTIVATION": [("ITGAV", ["F_structural", "F_apCAF"]), ("ITGB8", ["F_structural"]),
                   ("THBS1", ["F_structural", "F_inflammatory"]), ("TGM2", ["F_inflammatory"])],
    "RECEPTOR": [("TGFBR2", ["F_reticular", "F_structural", "F_mesenchymal"])],
    "OUTPUT": [(g, ["F_structural"]) for g in ["TNC", "POSTN", "PRSS23", "ADAMTS2", "FAP",
                                               "LOXL2", "SERPINE1", "LTBP2"]],
}
FF_CD4 = ["TGFB1", "TGFB3", "LRRC32", "ITGB8", "THBS1", "TGM2"]
_ff = [(st, g, ct) for st, gs in FF_STAGES.items() for g, cts in gs for ct in cts]
prefetch(sorted({ct for *_, ct in _ff}), sorted({g for _, g, _ in _ff}))

FF = bh([dict(t_mwu(ct, g), group=st, panel="F1") for st, g, ct in _ff]
        + [dict(r, panel="F2") for r in t_paired(FF_CD4)])
SHORT = {ct: LABELS[ct].replace(" fibroblast", "") for *_, ct in _ff}
FF["ylabel"] = [f"{it(r['gene'])} · {SHORT.get(r['ct'], '')}" for _, r in FF.iterrows()]
FF["note"] = [f"Δ {d:+.2f}" if np.isfinite(d) else "" for d in FF["delta"]]

# stages in the order above; within OUTPUT, largest MF - healthy shift on top
F1 = FF[FF["panel"] == "F1"]
F1 = (F1.assign(_s=F1["group"].map({s: i for i, s in enumerate(FF_STAGES)}),
                _e=np.where(F1["group"] == "OUTPUT", -F1["delta"], 0.0))
      .sort_values(["_s", "_e"], kind="stable").drop(columns=["_s", "_e"]))
F2 = FF[FF["panel"] == "F2"]
FF = pd.concat([F1, F2])

# The caption states these; print whether the data reproduces them rather than assume it.
print("claim checks:")
for st, d in F1.groupby("group", sort=False):
    print(f"  {st:<13} {'ok  ' if (d['direction'] == 'up').any() else 'FAIL'} "
          f"{int((d['direction'] == 'up').sum())}/{len(d)} rows up in MF | "
          f"Δ {d['delta'].min():+.2f} to {d['delta'].max():+.2f}")
_up, _dn = (F2.loc[F2["direction"] == k, "gene"].tolist() for k in ("up", "down"))
print(f"  CD4 flat    {'ok  ' if not _up else 'FAIL'} higher in clone: {_up or 'none'} | "
      f"lower in clone: {_dn or 'none'}")
_lr = F2.set_index("gene").loc["LRRC32"]
print(f"  LRRC32      {'ok  ' if _lr['direction'] == 'down' else 'FAIL'} reactive > clone in "
      f"{_lr['n_neg']:.0f}/{_lr['n_pairs']:.0f} biopsies, {sig(_lr['q'])}")


def cd4_hviolins(ax, df, arm="MF"):
    """Horizontal twin of cell_violins: one row per gene (first on top), malignant above
    reactive. Violin = cells, zeros included; dots = per-biopsy means; % of cells detected under
    each gene. Returns the 98th percentile of expressing cells, for the shared x limit."""
    rng = np.random.default_rng(0)
    C = {ct: cell_values(ct, df["gene"].tolist(), arm=arm) for ct in (NONCLONAL, CLONAL)}
    n, top = len(df), 0.0
    y = np.arange(n)[::-1]
    for yi, (_, r) in zip(y, df.iterrows()):
        for dy, ct, col, v in ((0.2, CLONAL, MALIG, r["vals_a"]),
                               (-0.2, NONCLONAL, REACT, r["vals_b"])):
            c = C[ct][r["gene"]].to_numpy()
            _violin(ax, c, yi + dy, col, horizontal=True)
            if (c > 0).sum() >= 20:
                top = max(top, float(np.quantile(c[c > 0], 0.98)))
            v = np.asarray(v)
            ax.scatter(v, yi + dy + rng.uniform(-0.06, 0.06, len(v)), s=6, color=col,
                       edgecolor="white", linewidth=0.2, zorder=3)
            if len(v):
                ax.vlines(np.median(v), yi + dy - 0.12, yi + dy + 0.12, color="#111111",
                          linewidth=1.2, zorder=4)
        votes = f"  {r['n_pos']:.0f}↑/{r['n_neg']:.0f}↓" if np.isfinite(r["q"]) else ""
        flip = "  reactive > clone" if r["direction"] == "down" else ""
        ax.text(1.01, yi, f"{sig(r['q'])}{votes}{flip}", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=6,
                color={"up": UP, "down": DOWN}.get(r["direction"], "#333333"))
        for k, (d, col) in enumerate(((r["det_a"], MALIG), (r["det_b"], REACT))):
            ax.text(0, yi, pct(d), ha="right", va="top", fontsize=5.5, color=col,
                    transform=offset_copy(ax.get_yaxis_transform(), fig=ax.figure, x=-3,
                                          y=-5 - 6.5 * k, units="points"))
    ax.set_yticks(y)
    ax.set_yticklabels([it(g) for g in df["gene"]], fontsize=6.5)
    ax.set_ylim(-0.7, n - 0.3)
    _clean(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return top


units = len(F1) + 0.6 * (F1["group"].nunique() - 1)
fig = plt.figure(figsize=(9.0, 0.2 * units + 1.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.85], wspace=0.55,
                      left=0.27, right=0.88, top=0.9, bottom=0.12)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)
spans = dumbbell(ax1, F1, group_col="group", note_col="note", brackets=True)
top = cd4_hviolins(ax2, F2)
xmax = max([top] + [float(np.nanmax(np.r_[v, 0.0])) for col in ("vals_a", "vals_b")
                    for v in FF[col]])
ax1.set_xlim(-0.02 * xmax, 1.05 * xmax)
ax1.set_xlabel(XLAB, fontsize=7)
ax2.set_xlabel("log1p CP10K (violin = cells, zeros included; dots = biopsy means)", fontsize=7)
ax1.set_title("Fibroblasts, MF vs healthy skin", fontsize=7.5, loc="left")
ax2.set_title("CD4, malignant vs reactive (same biopsy)\n"
              "under each gene: % cells detected, malignant | reactive", fontsize=7, loc="left")
fig.legend(handles=dir_handles("MF", "healthy"), fontsize=6, frameon=False, ncol=5,
           loc="upper left", bbox_to_anchor=(0.02, 0.05))
fig.legend(handles=ARM_HANDLES, fontsize=6, frameon=False, ncol=2, loc="upper right",
           bbox_to_anchor=(0.98, 0.05))
fig.suptitle("TGF-β loop genes in fibroblasts (MF vs healthy) and CD4 (malignant vs reactive)",
             fontsize=9.5, x=0.0, ha="left", y=0.99)
_span = F1.groupby("group", sort=False)["delta"].agg(["min", "max"])
row_brackets(ax1, spans, sub={st: f"Δ {r['min']:+.2f} to {r['max']:+.2f}"
                              for st, r in _span.iterrows()})
caption(fig,
        ("The loop is self-contained; no T-cell input is required. " if not _up
         else f"{', '.join(_up)} higher in malignant CD4, so the loop is not T-cell-free. ")
        + "Left: each stage of the loop "
        "in fibroblasts, MF vs healthy skin; output genes ordered by effect, largest on top. "
        "Right: the same ligands and activators in CD4 on the same x scale. The latent ligands "
        "and activators move by small amounts and the receptor and output by large ones (Δ = MF "
        "minus healthy, median of donor means, per row and per stage): the claim is a complete "
        "parts list at the detection limit, not a set of large effects. "
        f"LRRC32 is higher in reactive than malignant CD4 ({sig(_lr['q'])}; detected in "
        f"{pct(_lr['det_b'])} vs {pct(_lr['det_a'])} of cells) -- an internal positive control "
        "that CD4 cells are detected for this gene, so a flat clone is not a sensitivity failure. "
        + STATS_NOTE)
DC._finish(fig, fig28b("FF_tgfb_loop"), True)
save_source(FF, "FF_source")
logfig("FF", fig28b("FF_tgfb_loop"),
       "the TGF-beta loop (latent ligand, docking, activation, receptor, output) in fibroblasts "
       "vs the same genes in CD4",
       "object, mal_cluster cohort; donor-level MWU + paired signed-rank recomputed, BH within figure")
show_tests(FF, ("panel", "group", "gene", "ct", "n_a", "n_b", "med_a", "med_b", "det_a",
                "det_b", "q", "direction"))
```

```python  # cell 21
# ============================================================================
# FF_stage  FF split by stage
# ============================================================================
# Left: every loop row twice (early, advanced) vs healthy, rows in FF's order; right: the CD4
# ligands and activators, malignant vs reactive in early (top) and advanced (bottom) biopsies.
FFS = bh([r for grp, g, ct in _ff for r in stage_mwu(ct, g, group=grp, panel="F1")]
         + stage_paired(FF_CD4, panel="F2"))
FFS["ylabel"] = [f"{it(r['gene'])} · {SHORT.get(r['ct'], '')}" for _, r in FFS.iterrows()]
_order = {f"{r['ct']}|{r['gene']}|HC": i for i, (_, r) in enumerate(F1.iterrows())}
_f1 = FFS[FFS["panel"] == "F1"]
F1S, _ = stage_split(_f1.iloc[np.argsort(_f1["item"].map(_order).to_numpy(), kind="stable")])
F1S = with_stage(F1S)
F1S["note"] = [" · ".join(x for x in (f"Δ {d:+.2f}" if np.isfinite(d) else "",
                                      eva_text(q, e)) if x)
               for d, q, e in zip(F1S["delta"], F1S["eva_q"], F1S["eva_dir"])]
F2S = FFS[FFS["panel"] == "F2"]

print("claim checks by stage:")
for st in STAGES:
    for grp, d in F1S[F1S["stage"] == st].groupby("group", sort=False):
        print(f"  {st:<9} {grp:<13} {'ok  ' if (d['direction'] == 'up').any() else 'FAIL'} "
              f"{int((d['direction'] == 'up').sum())}/{len(d)} rows up in MF | "
              f"Δ {d['delta'].min():+.2f} to {d['delta'].max():+.2f}")
_cd4_up = {st: F2S.loc[(F2S["stage"] == st) & (F2S["direction"] == "up"), "gene"].tolist()
           for st in STAGES}
print(f"  CD4 flat   higher in clone: {_cd4_up}")

units = len(F1S) + 0.6 * (F1S["group"].nunique() - 1)
fig = plt.figure(figsize=(10.5, 0.2 * units + 2.2))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.85], wspace=0.9, hspace=0.25,
                      left=0.24, right=0.86, top=0.92, bottom=0.08)
ax1 = fig.add_subplot(gs[:, 0])
spans = dumbbell(ax1, F1S, group_col="group", note_col="note", brackets=True)
stage_ticks(ax1, F1S["stage"])
top, axr = 0.0, []
for i, st in enumerate(STAGES):
    ax = fig.add_subplot(gs[i, 1], sharex=ax1)
    top = max(top, cd4_hviolins(ax, F2S[F2S["stage"] == st], arm=st))
    ax.set_title(f"CD4 {st} MF, malignant vs reactive (same biopsy)", fontsize=7, loc="left",
                 color=STAGE_COL[st])
    axr.append(ax)
ev = F2S[F2S["stage"] == EVA]
for yi, (_, r) in zip(np.arange(len(ev))[::-1], ev.iterrows()):
    axr[1].text(1.01, yi - 0.32,
                eva_text(r["q"], r["direction"]).replace("vs early", "gap vs early"),
                transform=axr[1].get_yaxis_transform(), ha="left", va="center", fontsize=5,
                color=eva_color(r["direction"]))
xmax = max([top] + [float(np.nanmax(np.r_[v, 0.0])) for col in ("vals_a", "vals_b")
                    for v in pd.concat([F1S, F2S])[col]])
ax1.set_xlim(-0.02 * xmax, 1.05 * xmax)
ax1.set_xlabel(XLAB, fontsize=7)
axr[1].set_xlabel("log1p CP10K (violin = cells, zeros included; dots = biopsy means)", fontsize=7)
ax1.set_title("Fibroblasts, MF stage vs healthy skin", fontsize=7.5, loc="left")
fig.legend(handles=dir_handles("MF", "healthy") + STAGE_HANDLES, fontsize=6, frameon=False,
           ncol=4, loc="upper left", bbox_to_anchor=(0.02, 0.04))
fig.legend(handles=ARM_HANDLES, fontsize=6, frameon=False, ncol=2, loc="upper right",
           bbox_to_anchor=(0.98, 0.04))
_ok = {st: sum(bool((d["direction"] == "up").any())
               for _, d in F1S[F1S["stage"] == st].groupby("group", sort=False))
       for st in STAGES}
_ns = F1S["group"].nunique()
fig.suptitle("TGF-β loop genes in fibroblasts and CD4, early and advanced MF", fontsize=9,
             x=0.0, ha="left", y=0.99)
print(f"{_ok['early']}/{_ns} loop stages up in early, {_ok['advanced']}/{_ns} in advanced MF "
      "fibroblasts; " + ("CD4 flat in both" if not any(_cd4_up.values()) else
                         "higher in malignant CD4: " + "; ".join(f"{st} {', '.join(v)}"
                                                                 for st, v in _cd4_up.items() if v)))
row_brackets(ax1, spans)
caption(fig, STATS_NOTE + " " + STAGE_NOTE)
DC._finish(fig, fig28b("FF_stage_tgfb_loop"), True)
save_source(FFS, "FF_stage_source")
logfig("FF_stage", fig28b("FF_stage_tgfb_loop"),
       "the TGF-beta loop in fibroblasts vs the same genes in CD4, early vs advanced MF",
       "object, mal_cluster cohort by stage_class; donor-level MWU + paired signed-rank per "
       "stage, donor MWU advanced vs early, BH within figure")
show_tests(FFS, ("panel", "stage", "group", "gene", "ct", "n_a", "n_b", "med_a", "med_b", "q",
                 "direction"))
```

```python  # cell 22
# ============================================================================
# FG  Does the stroma track tumour burden? (28_mc §D)
# ============================================================================
# The strongest burden -> stroma association 28_mc §D found: robust rows first (pooled AND within
# study), else the smallest pooled q. One dot per MF sample, coloured by study; burden = the
# malignant share of the sample's mal_cluster-called T cells.
DT = pd.read_csv(cfg.tab("D_burden_stroma"))
BURDEN = pd.read_csv(cfg.tab("burden"), index_col=0)
_t = DT[DT["status"] == "tested"].sort_values(["robust", "q"], ascending=[False, True])
top_ = _t.iloc[0]
row_, readout_ = top_["stroma_row"], top_["readout"]
STROMA_COARSE = {**cfg.FIBRO_COARSE, **cfg.MYELOID_COARSE}
if readout_.startswith("share_of_"):
    _comp = cfg.FIBRO_COARSE if readout_ == "share_of_fibroblast" else cfg.MYELOID_COARSE
    _m = IS_MF & np.isin(LV, [lv for lvs in _comp.values() for lv in lvs])
    _all = pd.Series(_samp[_m]).value_counts()
    _row = pd.Series(_samp[_m & np.isin(LV, STROMA_COARSE[row_])]).value_counts()
    vals = (_row.reindex(_all.index, fill_value=0) / _all)[_all >= cfg.MIN_CELLS]
    ylab = f"{row_} share of {readout_.split('_')[-1]}s"
else:
    _m = IS_MF & np.isin(LV, STROMA_COARSE[row_])
    _g = pd.DataFrame({"s": _samp[_m], "v": adata.obs[readout_].to_numpy()[_m]}).groupby("s")["v"]
    vals = _g.median()[_g.size() >= cfg.MIN_CELLS]
    ylab = f"{readout_} in {row_}\n(sample median)"
pl = BURDEN.join(vals.rename("value"), how="inner")

fig, ax = plt.subplots(figsize=(5, 3))
_cmap = plt.get_cmap("tab20")
for k, s_ in enumerate(sorted(pl[cfg.STUDY_KEY].unique())):
    d_ = pl[pl[cfg.STUDY_KEY] == s_]
    ax.scatter(d_["burden"], d_["value"], s=12, color=_cmap(k % 20), label=s_, linewidth=0)
ax.set_xlabel("tumour burden (malignant share of T cells)", fontsize=7)
ax.set_ylabel(ylab, fontsize=7)
_name = readout_.replace("fam_", "").replace("bal_", "").replace("share_of_", "share of ")
if top_["q"] < 0.05:
    ttl = (f"{_name} in {row_} {'rises' if top_['rho'] > 0 else 'falls'} with tumour burden "
           f"(rho {top_['rho']:+.2f}, q {top_['q']:.1g}; within study rho "
           f"{top_['rho_within_study']:+.2f}{'' if top_['robust'] else ', n.s.'})")
else:
    ttl = (f"No stromal readout tracks tumour burden ({int(_t['n_samples'].max())} MF samples; "
           f"strongest: {_name} in {row_}, q {top_['q']:.2f})")
ax.set_title(f"{_name} in {row_} vs tumour burden, MF samples", fontsize=7, loc="left")
ax.legend(fontsize=5, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
_clean(ax)
DC._finish(fig, fig28b("FG_burden_stroma"), True)
pl.to_csv(tab28b("FG_source"))
logfig("FG", fig28b("FG_burden_stroma"), "does the stroma track tumour burden across MF samples",
       "28_mc §D (deg28mc_D_burden_stroma) + per-sample values recomputed from the object")
print(f"FG: {len(_t)} tests, {int(_t['robust'].sum())} robust | drawn: {row_} {readout_} | {ttl}")
```

```python  # cell 23
# ============================================================================
# FG_stage  FG split by stage: does the burden -> stroma association hold within a stage?
# ============================================================================
# FG's readout, dots coloured by stage. Spearman rho within early and within advanced samples;
# the readout advanced vs early is a donor-level Mann-Whitney on donor means (pooled across
# studies). BH over the three tests.
from scipy.stats import spearmanr

PLS = pl.assign(stage=SAMPLE2STAGE.reindex(pl.index).to_numpy())
_n_out = int((~PLS["stage"].isin(STAGES)).sum())
PLS = PLS[PLS["stage"].isin(STAGES)]
_rows = []
for st in STAGES:
    d_ = PLS[PLS["stage"] == st]
    ok = len(d_) >= cfg.MIN_DONORS_PER_ARM and d_["value"].nunique() > 1
    r_, p_ = spearmanr(d_["burden"], d_["value"]) if ok else (np.nan, np.nan)
    _rows.append({"test": f"spearman {st}", "n": len(d_), "rho": r_, "p": p_})
_dv = PLS.groupby("donor").agg(value=("value", "mean"), stage=("stage", "first"))
x_, y_ = (_dv.loc[_dv["stage"] == st, "value"].to_numpy() for st in ("advanced", "early"))
_p = (float(mannwhitneyu(x_, y_, alternative="two-sided").pvalue)
      if min(len(x_), len(y_)) >= cfg.MIN_DONORS_PER_ARM and np.ptp(np.r_[x_, y_]) > 0 else np.nan)
_rows.append({"test": EVA, "n": len(x_) + len(y_), "rho": np.nan, "p": _p,
              "med_advanced": np.median(x_) if len(x_) else np.nan,
              "med_early": np.median(y_) if len(y_) else np.nan})
FGS = pd.DataFrame(_rows)
FGS["q"] = np.nan
_m = FGS["p"].notna()
if _m.any():
    FGS.loc[_m, "q"] = multipletests(FGS.loc[_m, "p"], method="fdr_bh")[1]

fig, ax = plt.subplots(figsize=(5, 3))
for st in STAGES:
    d_ = PLS[PLS["stage"] == st]
    ax.scatter(d_["burden"], d_["value"], s=12, color=STAGE_COL[st], linewidth=0,
               label=f"{st} ({len(d_)} samples)")
ax.set_xlabel("tumour burden (malignant share of T cells)", fontsize=7)
ax.set_ylabel(ylab, fontsize=7)
ax.set_title(f"{_name} in {row_} vs tumour burden, early and advanced MF", fontsize=7,
             loc="left")
ax.legend(fontsize=5.5, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
_clean(ax)
caption(fig, f"Stage = stage_class (early IA-IIA, advanced IIB+); {_n_out} samples of unknown "
             "stage left out. One dot per MF sample. BH over the two within-stage Spearman tests "
             "and the advanced-vs-early Mann-Whitney (donor means, pooled across studies).")
DC._finish(fig, fig28b("FG_stage_burden_stroma"), True)
PLS.to_csv(tab28b("FG_stage_source"))
FGS.to_csv(tab28b("FG_stage_tests"), index=False)
logfig("FG_stage", fig28b("FG_stage_burden_stroma"),
       "does the stroma track tumour burden within early and within advanced MF",
       "28_mc §D's strongest readout; Spearman per stage + donor MWU advanced vs early, BH")
print(FGS.to_string(index=False, float_format="%.3g"))
```

```python  # cell 24
# ============================================================================
# §R  What was written
# ============================================================================
IDX = pd.DataFrame(FIG_INDEX)
IDX.to_csv(tab28b("figure_index"), index=False)
print(IDX.to_string(index=False))
print(f"\n{len(IDX)} figures -> {cfg.FIG_DIR}/{PREFIX}*.png")
print(f"source tables -> {cfg.TAB_DIR}/{PREFIX}*.csv")
print("nothing under deg28_* was written; nb28's tables are the input, not the output")
```
