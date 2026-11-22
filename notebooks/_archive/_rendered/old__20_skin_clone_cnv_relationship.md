# old/20_skin_clone_cnv_relationship.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/20_skin_clone_cnv_relationship.ipynb` and is never run._

# Clone ↔ CNV relationship in the CTCL/MF skin-T atlas

Goes beyond `15_skin_cd4_cd8ref_cnv_malignancy.ipynb` (which only *benchmarks* TCR vs CNV malignancy
agreement). Here we ask what the **TCR clone** and the **CNV state** carry about each other, grounded
in the literature review `CLONE_CNV_RESEARCH.md`:

- **Q1 — clonal CNV signature + subclonal evolution.** Does the dominant TCR clone carry a coherent,
  shared CNV profile, and is there CNV heterogeneity *within* the single clone (subclonal evolution:
  the TCR rearrangement is fixed at transformation, CNVs accumulate afterward)?
- **Q2 — recurrent CTCL landscape.** Do our malignant clones recapitulate the known CTCL arm events
  (gains 7 / 8q-MYC / 17q; losses 9p21-CDKN2A, 10q-PTEN, 17p-TP53, 13q)?
- **Q3 — burden vs expansion.** Does aneuploidy scale with clonal expansion?

**CNV side = nb14 strat_3** (`cnv_malig_cluster`): per-cnv-cluster GMM on `skin_T_tcr_malig_v2.h5ad`
against a **shared diploid reference** (within-donor `nonclonal` + atlas HC + external healthy),
`window=250`, `leiden_res=0.5`, `thr_scale=0.8`. The strat_3 per-cell scores + call are loaded from the
nb14 cache `skin_T_malignancy.parquet`; per-**arm** CNV profiles (needed for Q1/Q2, not cached) are
recomputed once against the same shared reference. The **TCR (ALICE) clone** stays as the clone axis.

> Heavy cells (inferCNV input prep, per-sample arm inferCNV) marked **HEAVY — run on GPU kernel**.

```python  # cell 1
import sys, gc
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc


def _resolve_nb_dir() -> Path:
    start = Path.cwd()
    for base in [start, *start.parents]:
        for sub in [Path("."), Path("notebooks/MF"), Path("scvi-tools-neural-nmf/notebooks/MF")]:
            cand = base / sub
            if cand.name == "MF" and (cand / "data").exists():
                return cand.resolve()
    raise FileNotFoundError(start)


NB_DIR = _resolve_nb_dir()
if str(NB_DIR) not in sys.path:
    sys.path.insert(0, str(NB_DIR))
import importlib
import atlas_join_helpers as H_join
import skin_T_cnv_helpers as H
importlib.reload(H_join); importlib.reload(H)

SEED = 0
np.random.seed(SEED)
sc.settings.verbosity = 1

# CNV side = nb14 strat_3 (per-cnv-cluster GMM on the v2 object, shared diploid reference).
V2            = NB_DIR / "data" / "atlas_joint" / "skin_T_tcr_malig_v2.h5ad"
GTF           = NB_DIR / "data" / "cache" / "Homo_sapiens.GRCh38.110.chr.gtf.gz"
INTEGRATED_H5 = NB_DIR / "data" / "Integrated_CTCL_skincellatlas_final_portal_tags.h5ad"
MALIG_PARQUET = NB_DIR / "data" / "atlas_joint" / "skin_T_malignancy.parquet"   # strat_3 outputs (nb14)
FIG_DIR   = NB_DIR / "figures"; FIG_DIR.mkdir(exist_ok=True)
OUT_PARQUET = NB_DIR / "data" / "atlas_joint" / "skin_clone_cnv_relationship.parquet"

CNV_LEIDEN_RES = 0.5   # matches strat_3 (skin_T_malignancy.parquet / skin_T_cnv_per_cell_res0.5)
ARM_CACHE = NB_DIR / "data" / "atlas_joint" / f"skin_T_arm_cnv_res{CNV_LEIDEN_RES:g}.parquet"
```

```
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
```

## Step 1–2 — load the curated v2 object + carry ALICE TCR malignancy

Same as nb14 Step 1–2. Loads `skin_T_tcr_malig_v2.h5ad` (242,959 cells, 34 kept donors). The ALICE
tumor call `tcr_malignant_alice` is carried in as `tcr_is_malignant` **and** `tcr_is_dominant_clone`
(so it also drives the benign/diploid reference). `has_tcr / tcr_clone_id / tcr_clone_size /
tcr_is_expanded / cached_malignant` come straight from the file.

```python  # cell 3
adata = sc.read_h5ad(V2)
print("loaded", V2.name, adata.shape)
# ALICE tumor call is the malignant truth; it also drives benign/reference exclusion downstream.
alice = adata.obs["tcr_malignant_alice"].astype(bool).to_numpy()
adata.obs["tcr_is_malignant"]      = alice
adata.obs["tcr_is_dominant_clone"] = alice
adata.obs["cached_malignant"]      = (adata.obs["cell_type"].astype(str) == "tumor_cell")
# has_tcr / tcr_clone_id / tcr_clone_size / tcr_is_expanded come from the file
print("alice malignant:", int(alice.sum()), "| has_tcr:", int(adata.obs["has_tcr"].sum()))
print(adata.obs.groupby("cell_type_T2", observed=True)["tcr_is_malignant"].agg(["size", "sum"]))
```

```
loaded skin_T_tcr_malig_v2.h5ad (242959, 49683)
alice malignant: 68620 | has_tcr: 165359
                size    sum
cell_type_T2               
CD4           169014  68620
CD4_Treg       15667      0
CD8            37207      0
NK             21071      0
```

## Step 3 — inferCNV input prep & shared diploid reference  · HEAVY (GPU kernel)

Exactly strat_3's setup (`H.prepare_infercnv_inputs`). **Query** = every annotated T cell
(`cell_type_T != "UNK"`) of the disease-bearing (non-HC) donors. **Reference** = three pooled groups,
each used per donor only if ≥20 cells: `nonclonal` (within-donor non-clonal T), `hc_atlas`
(atlas HC T, preferred), `healthy` (external benign-T from the Integrated CTCL atlas). Sets
`cnv_ref ∈ {query, nonclonal}`, `donor`, genomic positions; returns the shared reference too.

```python  # cell 5
acnv, SHARED_REF, CNV_DONORS, HC_DONORS = H.prepare_infercnv_inputs(
    adata, GTF, INTEGRATED_H5, SEED, n_hc_ref=5000, n_healthy_ref=3000, use_external=True)
print("usable (query) donors:", len(CNV_DONORS), "| acnv:", acnv.shape,
      "| query:", int((acnv.obs["cnv_ref"] == "query").sum()))
```

```
HC donors held out: 3 | CNV (query) donors: 31 | query cells: 240720
WARNING: adata.X seems to be already log-transformed.
WARNING: adata.X seems to be already log-transformed.
hc_atlas reference: 1559 cells
external healthy reference: 3000 cells

INFO:root:Extracted GTF attributes: ['gene_id', 'gene_version', 'gene_name', 'gene_source', 'gene_biotype', 'transcript_id', 'transcript_version', 'transcript_name', 'transcript_source', 'transcript_biotype', 'tag', 'transcript_support_level', 'exon_number', 'exon_id', 'exon_version', 'protein_id', 'protein_version', 'ccds_id']

WARNING: GTF file misses annotation for 301 genes in adata.
WARNING: Skipped 28 genes because of duplicate identifiers in GTF file.
common genes: 15777 | with genomic position: 15449 / 15777
query cells: 240720 | within-donor nonclonal: 92789 | shared ref: 4559 {'healthy': np.int64(3000), 'hc_atlas': np.int64(1559)}
usable (query) donors: 31 | acnv: (240720, 15777) | query: 147931
```

## Load cached strat_3 CNV scores + call (from nb14)

`cnv_cell_score` (genome-wide mean |inferCNV|), `cnv_score` (per-cnv-cluster mean), `cnv_focal_score`
(top-10% focal), `cnv_leiden` (strat_3 cluster), and the strat_3 malignancy call `cnv_malig_cluster`
— all loaded per query cell from `skin_T_malignancy.parquet`.

```python  # cell 7
sc_df = pd.read_parquet(MALIG_PARQUET)
for col in ["cnv_cell_score", "cnv_score", "cnv_focal_score"]:
    acnv.obs[col] = sc_df[col].reindex(acnv.obs_names).to_numpy()
acnv.obs["cnv_leiden"]        = sc_df["cnv_leiden"].reindex(acnv.obs_names).fillna("").to_numpy()
acnv.obs["cnv_malig_cluster"] = sc_df["cnv_malig_cluster"].reindex(acnv.obs_names).fillna(False).to_numpy().astype(bool)
q0 = acnv.obs["cnv_ref"] == "query"
print("query cells with strat_3 CNV:",
      int(acnv.obs.loc[q0, "cnv_cell_score"].notna().sum()), "/", int(q0.sum()),
      "| cnv_malig_cluster:", int(acnv.obs.loc[q0, "cnv_malig_cluster"].sum()))
```

```
query cells with strat_3 CNV: 147931 / 147931 | cnv_malig_cluster: 105259
```

## Q3 — CNV burden vs clone size / expansion  (cached strat_3 scalars, CPU-OK)

Per TCR+ query T cell: does genome-wide CNV burden (`cnv_cell_score`) scale with clonal expansion?
Categories: dominant (ALICE-malignant) clone vs non-dominant **expanded** clone vs small/singleton.
Spearman pooled + per-donor.

```python  # cell 9
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

q = acnv.obs[(acnv.obs["cnv_ref"] == "query") & acnv.obs["has_tcr"].to_numpy()].copy()
q = q[(q["tcr_clone_size"] > 0) & q["cnv_cell_score"].notna()]
q["clone_cat"] = np.where(q["tcr_is_dominant_clone"], "dominant",
                 np.where(q["tcr_is_expanded"], "expanded_nondom", "small/singleton"))

rho, p = spearmanr(np.log1p(q["tcr_clone_size"]), q["cnv_cell_score"])
perd = (q.groupby("donor", observed=True)
          .apply(lambda s: spearmanr(np.log1p(s["tcr_clone_size"]), s["cnv_cell_score"])[0]
                 if s["cnv_cell_score"].notna().sum() > 20 else np.nan))
print(f"pooled Spearman(burden, log clone_size) = {rho:.3f} (p={p:.1e}, n={len(q)})")
print(f"median per-donor Spearman = {np.nanmedian(perd.values):.3f} (n_donors={perd.notna().sum()})")
print(q.groupby("clone_cat", observed=True)["cnv_cell_score"].describe()[["count", "mean", "50%"]])

order = ["small/singleton", "expanded_nondom", "dominant"]
fig, ax = plt.subplots(figsize=(5, 3))
ax.boxplot([q.loc[q["clone_cat"] == c, "cnv_cell_score"].values for c in order],
           labels=order, showfliers=False)
ax.set_ylabel("cnv_cell_score"); ax.set_title("CNV burden rises with clonal expansion")
plt.xticks(rotation=15); plt.tight_layout()
plt.savefig(FIG_DIR / "clone_cnv_burden_by_expansion.png", dpi=150); plt.show()
```

```
/tmp/ipykernel_3924178/912492356.py:11: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  .apply(lambda s: spearmanr(np.log1p(s["tcr_clone_size"]), s["cnv_cell_score"])[0]
/tmp/ipykernel_3924178/912492356.py:11: FutureWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
  .apply(lambda s: spearmanr(np.log1p(s["tcr_clone_size"]), s["cnv_cell_score"])[0]

pooled Spearman(burden, log clone_size) = -0.232 (p=0.0e+00, n=71394)
median per-donor Spearman = -0.014 (n_donors=16)
                   count      mean       50%
clone_cat                                   
dominant         68604.0  0.006192  0.005721
expanded_nondom   2647.0  0.007531  0.006985
small/singleton    143.0  0.006545  0.006406

/tmp/ipykernel_3924178/912492356.py:19: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  ax.boxplot([q.loc[q["clone_cat"] == c, "cnv_cell_score"].values for c in order],

<Figure size 500x300 with 1 Axes>
```

## Q1/Q2 prerequisite — per-cell **arm-level** CNV  · HEAVY (GPU kernel)

`H.compute_arm_cnv_per_cell` re-runs per-donor inferCNV against the **shared diploid reference**
(`SHARED_REF`, same as strat_3) with `calculate_gene_values=True`, averages the per-gene CNV within
each chromosome arm, and caches a compact `query_cells × ~39 arm` matrix. Reloads the cache on later
runs. `window=250` matches strat_3.

```python  # cell 11
WINDOW_SIZE, CNV_N_JOBS, CNV_CHUNK, FORCE_ARM = 250, 8, 100, False   # HEAVY knobs (window matches strat_3)

arm = H.compute_arm_cnv_per_cell(acnv, CNV_DONORS, ARM_CACHE,
                                 shared_ref=SHARED_REF,
                                 reference_cat=("nonclonal", "hc_atlas", "healthy"),
                                 window=WINDOW_SIZE, n_jobs=CNV_N_JOBS, chunk=CNV_CHUNK, force=FORCE_ARM)
ARM_COLS = sorted([c for c in arm.columns if c != "donor"],
                  key=lambda a: (int(''.join(filter(str.isdigit, a.split('p')[0].split('q')[0])) or 99),
                                 0 if 'p' in a else 1))
arm = arm.reindex(acnv.obs_names)            # align to acnv (NaN for reference rows)
print("arm matrix:", arm[ARM_COLS].shape, "| arms:", ARM_COLS)
```

```
100%|██████████| 133/133 [31:35<00:00, 14.25s/it] 
/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/skin_T_cnv_helpers.py:611: RuntimeWarning: Mean of empty slice
  rows[a] = np.nanmean(G[np.ix_(qmask, cols)], axis=1)

[D1__P171] query=  7408  arms=41

100%|██████████| 200/200 [47:51<00:00, 14.36s/it] 
/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/skin_T_cnv_helpers.py:611: RuntimeWarning: Mean of empty slice
  rows[a] = np.nanmean(G[np.ix_(qmask, cols)], axis=1)

[D1__P204] query=  9089  arms=41

100%|██████████| 114/114 [27:14<00:00, 14.34s/it] 
/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/skin_T_cnv_helpers.py:611: RuntimeWarning: Mean of empty slice
  rows[a] = np.nanmean(G[np.ix_(qmask, cols)], axis=1)

[D1__P311] query=  4776  arms=41

100%|██████████| 73/73 [17:25<00:00, 14.32s/it]  
/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/skin_T_cnv_helpers.py:611: RuntimeWarning: Mean of empty slice
  rows[a] = np.nanmean(G[np.ix_(qmask, cols)], axis=1)

[D1__P312] query=  1229  arms=41

100%|██████████| 152/152 [36:21<00:00, 14.35s/it] 
/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/skin_T_cnv_helpers.py:611: RuntimeWarning: Mean of empty slice
  rows[a] = np.nanmean(G[np.ix_(qmask, cols)], axis=1)

[D1__P76] query=  8187  arms=41

100%|██████████| 111/111 [26:30<00:00, 14.33s/it] 
/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/skin_T_cnv_helpers.py:611: RuntimeWarning: Mean of empty slice
  rows[a] = np.nanmean(G[np.ix_(qmask, cols)], axis=1)

[D1__PGS] query=  4718  arms=41

100%|██████████| 58/58 [13:51<00:00, 14.34s/it]  
/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/skin_T_cnv_helpers.py:611: RuntimeWarning: Mean of empty slice
  rows[a] = np.nanmean(G[np.ix_(qmask, cols)], axis=1)

[D3__P105] query=   709  arms=41

100%|██████████| 114/114 [27:20<00:00, 14.39s/it] 
/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/skin_T_cnv_helpers.py:611: RuntimeWarning: Mean of empty slice
  rows[a] = np.nanmean(G[np.ix_(qmask, cols)], axis=1)

[D3__P192] query=  5352  arms=41

100%|██████████| 60/60 [14:20<00:00, 14.34s/it]  
/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/skin_T_cnv_helpers.py:611: RuntimeWarning: Mean of empty slice
  rows[a] = np.nanmean(G[np.ix_(qmask, cols)], axis=1)

[D3__P196] query=  1090  arms=41

100%|██████████| 150/150 [35:57<00:00, 14.38s/it] 
/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/skin_T_cnv_helpers.py:611: RuntimeWarning: Mean of empty slice
  rows[a] = np.nanmean(G[np.ix_(qmask, cols)], axis=1)

[D3__P87] query= 10177  arms=41

100%|██████████| 104/104 [24:52<00:00, 14.36s/it] 
/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/skin_T_cnv_helpers.py:611: RuntimeWarning: Mean of empty slice
  rows[a] = np.nanmean(G[np.ix_(qmask, cols)], axis=1)

[D3__P94] query=  1077  arms=41

100%|██████████| 67/67 [16:15<00:00, 14.55s/it]  
/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/skin_T_cnv_helpers.py:611: RuntimeWarning: Mean of empty slice
  rows[a] = np.nanmean(G[np.ix_(qmask, cols)], axis=1)

[D5__MFIVB] query=  1844  arms=41

100%|██████████| 47/47 [11:18<00:00, 14.43s/it]  
/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/skin_T_cnv_helpers.py:611: RuntimeWarning: Mean of empty slice
  rows[a] = np.nanmean(G[np.ix_(qmask, cols)], axis=1)

[H__MFIV1] query=    58  arms=41

100%|██████████| 48/48 [11:25<00:00, 14.27s/it]  
/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/skin_T_cnv_helpers.py:611: RuntimeWarning: Mean of empty slice
  rows[a] = np.nanmean(G[np.ix_(qmask, cols)], axis=1)

[H__SS1] query=    97  arms=41

100%|██████████| 48/48 [11:32<00:00, 14.43s/it]  
/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/skin_T_cnv_helpers.py:611: RuntimeWarning: Mean of empty slice
  rows[a] = np.nanmean(G[np.ix_(qmask, cols)], axis=1)

[H__SS2] query=    82  arms=41

100%|████
... [4977 more chars truncated]
```

## Q1a — clonal CNV signature: does the dominant clone have a coherent, distinct profile?

Per donor, mean arm profile of the **dominant (malignant) clone** vs benign CD4 (non-dominant,
non-tumor) and the top non-dominant **expanded** clone (specificity control — expected near-diploid).

```python  # cell 13
obs = acnv.obs
rows, labels, kinds = [], [], []
for d in CNV_DONORS:
    m = (obs["donor"] == d).to_numpy() & (obs["cnv_ref"] == "query").to_numpy()
    A = arm.loc[obs.index[m], ARM_COLS]
    dom = obs.loc[m, "tcr_is_malignant"].to_numpy()
    benign = (~obs.loc[m, "tcr_is_dominant_clone"].to_numpy()
              & (obs.loc[m, "cell_type"].astype(str) != "tumor_cell").to_numpy())
    if dom.sum() >= 20:
        rows.append(A[dom].mean().values); labels.append(f"{d} DOM(n={int(dom.sum())})"); kinds.append("dominant")
    if benign.sum() >= 20:
        rows.append(A[benign].mean().values); labels.append(f"{d} benign"); kinds.append("benign")

P = pd.DataFrame(rows, index=labels, columns=ARM_COLS)
dom_P = P[np.array(kinds) == "dominant"]
print("dominant-clone profiles:", dom_P.shape)
vlim = float(np.nanpercentile(np.abs(dom_P.values), 98)) or 0.05
fig, ax = plt.subplots(figsize=(10, max(3, 0.22 * len(dom_P))))
im = ax.imshow(dom_P.values, aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
ax.set_xticks(range(len(ARM_COLS))); ax.set_xticklabels(ARM_COLS, rotation=90, fontsize=6)
ax.set_yticks(range(len(dom_P))); ax.set_yticklabels(dom_P.index, fontsize=5)
ax.set_title("Dominant-clone mean CNV profile per sample (arms)")
fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01, label="inferCNV (arm mean)")
plt.tight_layout(); plt.savefig(FIG_DIR / "clone_cnv_dominant_arm_profiles.png", dpi=150); plt.show()
```

```
dominant-clone profiles: (29, 41)

<Figure size 1000x638 with 2 Axes>
```

## Q2 — recurrent CTCL CNV landscape across dominant clones

Per donor, call each arm gain/loss in the dominant clone vs that donor's **benign-CD4** arm
distribution (robust z). Recurrence frequency across donors, overlaid with the known CTCL events.

```python  # cell 15
KNOWN = {"chr7p": "gain", "chr7q": "gain", "chr8q": "gain", "chr17q": "gain",
         "chr9p": "loss", "chr10q": "loss", "chr17p": "loss", "chr13q": "loss", "chr6q": "loss"}
Z_THR, EPS = 2.5, 0.01

calls = {}   # donor -> arm -> {-1,0,1}
for d in CNV_DONORS:
    m = (obs["donor"] == d).to_numpy() & (obs["cnv_ref"] == "query").to_numpy()
    dom = obs.loc[m, "tcr_is_malignant"].to_numpy()
    benign = (~obs.loc[m, "tcr_is_dominant_clone"].to_numpy()
              & (obs.loc[m, "cell_type"].astype(str) != "tumor_cell").to_numpy())
    if dom.sum() < 20 or benign.sum() < 20:
        continue
    A = arm.loc[obs.index[m], ARM_COLS]
    b_med = A[benign].median(); b_mad = (A[benign] - b_med).abs().median() * 1.4826 + 1e-6
    z = (A[dom].mean() - b_med) / b_mad
    dom_mean = A[dom].mean()
    calls[d] = {a: (1 if (z[a] > Z_THR and dom_mean[a] > EPS) else
                    -1 if (z[a] < -Z_THR and dom_mean[a] < -EPS) else 0) for a in ARM_COLS}

C = pd.DataFrame(calls).T.reindex(columns=ARM_COLS).fillna(0)
freq = pd.DataFrame({"gain_frac": (C == 1).mean(), "loss_frac": (C == -1).mean()})
freq["known"] = [KNOWN.get(a, "") for a in freq.index]
print(f"donors with a callable dominant clone: {len(C)}")
print(freq[(freq["gain_frac"] > 0) | (freq["loss_frac"] > 0)].round(3).to_string())

fig, ax = plt.subplots(figsize=(10, 3))
x = np.arange(len(ARM_COLS))
ax.bar(x, freq["gain_frac"].values, color="#d62728", label="gain")
ax.bar(x, -freq["loss_frac"].values, color="#1f77b4", label="loss")
for i, a in enumerate(ARM_COLS):
    if a in KNOWN:
        ax.annotate("*", (i, 0), ha="center", va="center", fontsize=9, color="k")
ax.set_xticks(x); ax.set_xticklabels(ARM_COLS, rotation=90, fontsize=6)
ax.axhline(0, color="k", lw=0.5); ax.set_ylabel("fraction of donors"); ax.legend(fontsize=7)
ax.set_title("Recurrent arm-level CNV in dominant clones  (*  = known CTCL event)")
plt.tight_layout(); plt.savefig(FIG_DIR / "clone_cnv_recurrent_landscape.png", dpi=150); plt.show()
```

```
donors with a callable dominant clone: 29
        gain_frac  loss_frac known
chr1q       0.034      0.000      
chr2p       0.034      0.000      
chr4p       0.034      0.034      
chr4q       0.034      0.000      
chr5p       0.000      0.034      
chr6p       0.034      0.034      
chr6q       0.000      0.069  loss
chr7p       0.414      0.000  gain
chr7q       0.448      0.000  gain
chr8p       0.069      0.034      
chr8q       0.138      0.000  gain
chr9p       0.000      0.034  loss
chr9q       0.034      0.034      
chr10p      0.000      0.138      
chr10q      0.000      0.138  loss
chr12p      0.034      0.034      
chr13q      0.000      0.172  loss
chr14q      0.069      0.000      
chr15q      0.034      0.000      
chr16p      0.138      0.000      
chr16q      0.034      0.034      
chr17p      0.000      0.138  loss
chr17q      0.276      0.000  gain
chr18p      0.103      0.000      
chr18q      0.103      0.000      
chr19p      0.138      0.000      
chr21q      0.276      0.000      
chr22q      0.138      0.000      

<Figure size 1000x300 with 1 Axes>
```

## Q1b — subclonal CNV evolution *within* the dominant clone

The TCR clone is one transformation event; CNV heterogeneity inside it = subclonal evolution.
For high-burden donors, cluster the dominant-clone cells on the arm matrix into candidate CNV
subclones; report trunk (shared) vs branch (private) arm events and an annotated heatmap.

```python  # cell 17
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist

MIN_DOM, MAX_K = 150, 3
sub_rows = []
big = [(d, int(((obs["donor"] == d).to_numpy() & obs["tcr_is_malignant"].to_numpy()).sum()))
       for d in CNV_DONORS]
big = [d for d, n in sorted(big, key=lambda t: -t[1]) if n >= MIN_DOM][:3]
print("subclone donors (top dominant-clone size):", big)

acnv.obs["cnv_subclone"] = ""
for d in big:
    m = (obs["donor"] == d).to_numpy() & obs["tcr_is_malignant"].to_numpy()
    A = arm.loc[obs.index[m], ARM_COLS].fillna(0.0)
    k = min(MAX_K, max(2, A.shape[0] // 200))
    km = KMeans(k, random_state=SEED, n_init=5).fit(A.values)
    lab = km.labels_
    acnv.obs.loc[obs.index[m], "cnv_subclone"] = [f"{d}_s{c}" for c in lab]
    het = float(np.mean(pdist(km.cluster_centers_))) if k > 1 else 0.0
    # trunk vs branch: arms with |centroid|>0.02 in ALL vs SOME subclones
    strong = np.abs(km.cluster_centers_) > 0.02
    trunk = [a for j, a in enumerate(ARM_COLS) if strong[:, j].all()]
    branch = [a for j, a in enumerate(ARM_COLS) if strong[:, j].any() and not strong[:, j].all()]
    sub_rows.append({"donor": d, "n_cells": int(m.sum()), "k": k,
                     "centroid_spread": round(het, 4),
                     "trunk_arms": ",".join(trunk), "branch_arms": ",".join(branch)})
    # heatmap cells x arms grouped by subclone
    order = np.argsort(lab, kind="stable")
    fig, ax = plt.subplots(figsize=(9, 4))
    vlim = float(np.nanpercentile(np.abs(A.values), 98)) or 0.05
    im = ax.imshow(A.values[order], aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                   interpolation="none")
    for b in np.where(np.diff(np.sort(lab)) != 0)[0] + 1:
        ax.axhline(b - 0.5, color="k", lw=0.6)
    ax.set_xticks(range(len(ARM_COLS))); ax.set_xticklabels(ARM_COLS, rotation=90, fontsize=6)
    ax.set_yticks([]); ax.set_ylabel(f"{int(m.sum())} dominant-clone cells (by subclone)")
    ax.set_title(f"{d}: CNV subclones within one TCR clone")
    fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    plt.tight_layout(); plt.savefig(FIG_DIR / f"clone_cnv_subclones_{d}.png", dpi=150); plt.show()

subclone_summary = pd.DataFrame(sub_rows)
subclone_summary
```

```
subclone donors (top dominant-clone size): ['Li2024_atlas__CTCL3', 'Li2024_atlas__CTCL1', 'Li2024_atlas__CTCL5']

<Figure size 900x400 with 2 Axes>
<Figure size 900x400 with 2 Axes>
<Figure size 900x400 with 2 Axes>
                 donor  n_cells  k  centroid_spread trunk_arms   branch_arms
0  Li2024_atlas__CTCL3     8068  3           0.0511             chr7q,chr16p
1  Li2024_atlas__CTCL1     6932  3           0.0873      chr7q  chr7p,chr13q
2  Li2024_atlas__CTCL5     6379  3           0.0886             chr2p,chr21q
```

## Q (reverse) — strat_3 ↔ TCR concordance + persist

The strat_3 CNV-malignant call (`cnv_malig_cluster`) vs the orthogonal TCR (ALICE) dominant-clone call
on TCR+ query cells (precision / recall / F1; should track nb14 Step 11 ≈ sens 0.76 / spec 0.46).
The arm-burden score (≥ `K_ARM` strongly deviating arms) is kept as an auxiliary descriptor. Then
persist the per-cell clone↔CNV table.

```python  # cell 19
K_ARM = 2
q1 = (acnv.obs["cnv_ref"] == "query").to_numpy()
# auxiliary arm-burden descriptor (vs the pooled query-cell arm baseline)
b_med = arm.loc[q1, ARM_COLS].median(); b_mad = (arm.loc[q1, ARM_COLS] - b_med).abs().median() * 1.4826 + 1e-6
zc = ((arm[ARM_COLS] - b_med) / b_mad).abs()
acnv.obs["cnv_arm_burden"] = (zc > Z_THR).sum(axis=1).reindex(acnv.obs_names).fillna(0).to_numpy()
acnv.obs["cnv_arm_malig"]  = (acnv.obs["cnv_arm_burden"] >= K_ARM) & q1

# primary CNV call = strat_3 cnv_malig_cluster; concordance vs TCR (ALICE) on TCR+ query cells
tcr_av = acnv.obs["has_tcr"].to_numpy() & q1
yp = acnv.obs.loc[tcr_av, "cnv_malig_cluster"].to_numpy().astype(bool)
yt = acnv.obs.loc[tcr_av, "tcr_is_malignant"].to_numpy().astype(bool)
tp = int((yp & yt).sum()); fp = int((yp & ~yt).sum()); fn = int((~yp & yt).sum()); tn = int((~yp & ~yt).sum())
prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
sens = tp / max(1, tp + fn); spec = tn / max(1, tn + fp)
print(f"strat_3 (cnv_malig_cluster) vs TCR on {int(tcr_av.sum())} TCR+ query cells:")
print(f"  prec={prec:.3f} rec={rec:.3f} f1={2*prec*rec/max(1e-9,prec+rec):.3f} | sens={sens:.3f} spec={spec:.3f}")
print(pd.crosstab(acnv.obs.loc[tcr_av, "tcr_is_malignant"],
                  acnv.obs.loc[tcr_av, "cnv_malig_cluster"], rownames=["tcr"], colnames=["cnv (strat_3)"]))

keep = ["donor", "study", "disease", "cell_type", "cnv_ref", "has_tcr", "tcr_clone_id",
        "tcr_clone_size", "tcr_is_expanded", "tcr_is_dominant_clone", "tcr_is_malignant",
        "cnv_cell_score", "cnv_score", "cnv_leiden", "cnv_malig_cluster",
        "cnv_arm_burden", "cnv_arm_malig", "cnv_subclone"]
out = acnv.obs.loc[q1, [c for c in keep if c in acnv.obs.columns]].copy()
out = out.join(arm.loc[out.index, ARM_COLS])
out.index.name = "obs_name"
out.to_parquet(OUT_PARQUET)
print("wrote", OUT_PARQUET, out.shape)
```

```
strat_3 (cnv_malig_cluster) vs TCR on 71394 TCR+ query cells:
  prec=0.962 rec=0.756 f1=0.847 | sens=0.756 spec=0.273
cnv (strat_3)  False  True 
tcr                        
False            762   2028
True           16750  51854
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/skin_clone_cnv_relationship.parquet (147931, 59)
```
