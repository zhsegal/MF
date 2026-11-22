# old/04_infercnv.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/04_infercnv.ipynb` and is never run._

# inferCNV playground

Lifted out of `03_mrvi_replication.ipynb` §2 so the CNV inference + malignant-call logic can be iterated on independently. Reads the same `mrvi_ctcl_cache.h5ad` produced by `02_mrvi.ipynb`.

**Pipeline**
1. Load cached CTCL adata (raw counts in `.X`, MrVI latent in `obsm['X_mrvi']`).
2. Run `infercnvpy` against a stromal/KC diploid reference (no healthy donors in this h5ad).
3. Per-donor 2-component GMM on `cnv_score` to derive `is_malignant`.
4. Crosstab vs the paper's `cell_type == 'tumor_cell'` annotation as ground-truth check.

Requires `infercnvpy==0.6.1` (`pip install infercnvpy==0.6.1`).

```python  # cell 1
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

sc.settings.set_figure_params(dpi=80, facecolor="white", figsize=(5, 3))
print("scanpy", sc.__version__)

def _resolve_nb_dir():
    start = Path(__file__).parent.resolve() if "__file__" in globals() else Path.cwd()
    for base in [start, *start.parents]:
        for sub in [Path("."), Path("notebooks/MF"), Path("scvi-tools-neural-nmf/notebooks/MF")]:
            cand = base / sub
            if cand.name == "MF" and (cand / "data").exists():
                return cand.resolve()
    raise FileNotFoundError(f"could not locate MF/data starting from {start}")

NB_DIR    = _resolve_nb_dir()
CACHE_DIR = NB_DIR / "data" / "cache"
FIG_DIR   = NB_DIR / "figures"; FIG_DIR.mkdir(exist_ok=True)
TAB_DIR   = NB_DIR / "tables";  TAB_DIR.mkdir(exist_ok=True)
sc.settings.figdir = str(FIG_DIR)

CACHE_H5AD = CACHE_DIR / "mrvi_ctcl_cache.h5ad"
print("CACHE_H5AD =", CACHE_H5AD, "exists:", CACHE_H5AD.exists())
RNG = 0
```

## Load cache

Just the h5ad (no MrVI model needed for inferCNV). Stage groups recomputed for convenience downstream.

```python  # cell 3
adata = sc.read_h5ad(CACHE_H5AD)
print(adata)

EARLY = {"IA", "IB", "IIA"}
adata.obs["stage_group"] = (
    adata.obs["stage"].astype(str).map(lambda s: "early" if s in EARLY else "advanced")
    .astype("category").cat.reorder_categories(["early", "advanced"])
)
print("\ndonors per stage_group:")
print(adata.obs.drop_duplicates("donor").groupby("stage_group", observed=True)["donor"].nunique())
```

```python  # cell 4
adata.obs.shape
```

```
(419579, 18)
```

## inferCNV → malignant T-cell labels (spec Cell 3, Fig 3a)

**No healthy donors in this h5ad** — using stromal/keratinocyte cells as the diploid reference (standard infercnvpy workaround). Per-donor 2-component GMM on `cnv_score` defines `is_malignant`.

```python  # cell 6
import urllib.request
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import infercnvpy as cnv

GTF_URL = ("https://ftp.ensembl.org/pub/release-110/gtf/homo_sapiens/"
           "Homo_sapiens.GRCh38.110.chr.gtf.gz")
GTF_LOCAL = CACHE_DIR / "Homo_sapiens.GRCh38.110.chr.gtf.gz"
CNV_CACHE = CACHE_DIR / "infercnv_cnv_cache.h5ad"

# CRITICAL: Set to True so it ignores the broken cache and runs the new sorting logic.
# You can set this back to False after it completes successfully.
FORCE_RECOMPUTE_CNV = False

# infercnvpy expects log-normalised expression in .X (cache from notebook 02 stores raw counts).
# Idempotent via the "log1p" uns flag scanpy sets. Cheap (~30s) so always run.
if "log1p" not in adata.uns:
    assert "raw_counts" in adata.layers, "expected raw_counts layer from notebook 02 cache"
    adata.X = adata.layers["raw_counts"].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print("log-normalised adata.X (target_sum=1e4, log1p)")
else:
    print("adata.X already log1p-normalised; skipping")

# Diploid reference: stromal + KC + endothelial + pericytes + melanocytes
all_cts = set(adata.obs["cell_type"].astype(str).unique())
DIPLOID_CANDIDATES = [
    "Differentiated_KC", "Differentiated_KC*", "F2", "F3",
    "VE2", "Pericyte_1", "Pericyte_2", "Melanocyte",
]
DIPLOID = [c for c in DIPLOID_CANDIDATES if c in all_cts]
print("diploid reference:", DIPLOID)
assert len(DIPLOID) >= 3, "need at least 3 diploid reference cell types"

CNV_OBS_COLS  = ["cnv_score", "cnv_leiden"]
CNV_OBSM_KEYS = ["X_cnv", "X_cnv_pca"]

def _save_cnv_cache(a, path):
    side = ad.AnnData(
        X=np.zeros((a.n_obs, 1), dtype=np.float32),
        obs=pd.DataFrame(index=a.obs_names),
    )
    for k in CNV_OBSM_KEYS:
        if k in a.obsm:
            side.obsm[k] = a.obsm[k]
    for c in CNV_OBS_COLS:
        if c in a.obs:
            side.obs[c] = a.obs[c].values
    if "cnv" in a.uns:
        side.uns["cnv"] = a.uns["cnv"]
    side.write_h5ad(path, compression="gzip")

def _attach_cnv_cache(a, path):
    side = sc.read_h5ad(path)
    assert side.n_obs == a.n_obs and (side.obs_names == a.obs_names).all(), (
        "cnv cache obs mismatch — set FORCE_RECOMPUTE_CNV=True or delete the cache"
    )
    for k in CNV_OBSM_KEYS:
        if k in side.obsm:
            a.obsm[k] = side.obsm[k]
    for c in CNV_OBS_COLS:
        if c in side.obs:
            a.obs[c] = side.obs[c].values
    if "cnv" in side.uns:
        a.uns["cnv"] = side.uns["cnv"]

if (not FORCE_RECOMPUTE_CNV) and CNV_CACHE.exists():
    print(f"loading cnv cache <- {CNV_CACHE}  ({CNV_CACHE.stat().st_size/1e6:.1f} MB)")
    _attach_cnv_cache(adata, CNV_CACHE)
else:
    if not GTF_LOCAL.exists():
        print("downloading GTF (~50 MB)...")
        urllib.request.urlretrieve(GTF_URL, GTF_LOCAL)
    print("GTF:", GTF_LOCAL)
    
    # drop any stale annotations from a previous partial run so the merge in
    # genomic_position_from_gtf doesn't produce chromosome_x/_y collisions
    adata.var = adata.var.drop(columns=["chromosome", "start", "end", "gene_id", "gene_name"], errors="ignore")
    cnv.io.genomic_position_from_gtf(str(GTF_LOCAL), adata=adata)
    print("genes with positions before filter:", adata.var[["chromosome", "start", "end"]].notna().all(axis=1).sum())

    # --- CRITICAL FIX: Filter and Sort by Genomic Position ---
    # 1. Standardize chromosome column to string
    adata.var["chromosome"] = adata.var["chromosome"].astype(str)
    
    # 2. Drop genes without coordinates (unmapped contigs, MT, etc.)
    has_pos = adata.var[["chromosome", "start", "end"]].notna().all(axis=1)
    
    # 3. Filter for standard chromosomes only
    valid_chromosomes = [str(i) for i in range(1, 23)] + ["X", "Y"] + [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    valid_mask = adata.var["chromosome"].isin(valid_chromosomes)
    
    adata = adata[:, has_pos & valid_mask].copy()

    # 4. Make chromosome a categorical variable so it sorts strictly 1 to 22, X, Y
    chrom_order = [c for c in valid_chromosomes if c in adata.var["chromosome"].unique()]
    adata.var["chromosome"] = pd.Categorical(adata.var["chromosome"], categories=chrom_order, ordered=True)
    
    # 5. Execute the sort on the AnnData object
    adata = adata[:, adata.var.sort_values(["chromosome", "start"]).index].copy()
    print("genes after filter and sort:", adata.n_vars)
    print("Successfully sorted AnnData by genomic coordinates.")
    # ---------------------------------------------------------

    cnv.tl.infercnv(
        adata,
        reference_key="cell_type",
        reference_cat=DIPLOID,
        window_size=250,
        n_jobs=8,
    )
    cnv.tl.pca(adata)
    cnv.pp.neighbors(adata)
    cnv.tl.leiden(adata)
    cnv.tl.cnv_score(adata)
    
    _save_cnv_cache(adata, CNV_CACHE)
    print(f"saved cnv cache -> {CNV_CACHE}  ({CNV_CACHE.stat().st_size/1e6:.1f} MB)")

print("cnv_score: mean=%.4f, std=%.4f" % (adata.obs["cnv_score"].mean(), adata.obs["cnv_score"].std()))
```

```
adata.X already log1p-normalised; skipping
diploid reference: ['Differentiated_KC', 'Differentiated_KC*', 'F2', 'F3', 'VE2', 'Pericyte_1', 'Pericyte_2', 'Melanocyte']
GTF: /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/cache/Homo_sapiens.GRCh38.110.chr.gtf.gz

INFO:root:Extracted GTF attributes: ['gene_id', 'gene_version', 'gene_name', 'gene_source', 'gene_biotype', 'transcript_id', 'transcript_version', 'transcript_name', 'transcript_source', 'transcript_biotype', 'tag', 'transcript_support_level', 'exon_number', 'exon_id', 'exon_version', 'protein_id', 'protein_version', 'ccds_id']

WARNING: GTF file misses annotation for 186 genes in adata.
WARNING: Skipped 22 genes because of duplicate identifiers in GTF file.
genes with positions before filter: 9792
genes after filter and sort: 9785
Successfully sorted AnnData by genomic coordinates.

100%|██████████| 84/84 [03:38<00:00,  2.60s/it]
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/infercnvpy/tl/__init__.py:24: FutureWarning: In the future, the default backend for leiden will be igraph instead of leidenalg.

 To achieve the future defaults please pass: flavor="igraph" and n_iterations=2.  directed must also be False to work with igraph's implementation.
  return sc.tl.leiden(

saved cnv cache -> /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/cache/infercnv_cnv_cache.h5ad  (279.9 MB)
cnv_score: mean=0.0079, std=0.0049
```

## Per-sample inferCNV rerun (v2)

The pooled run above lets cross-donor technical variation contaminate the CNV signal. The v1 reference also included CTCL-enriched / disease-state subsets (`F2`, `F3`, `VE2`, `Differentiated_KC*`). Below: per-donor inference with a clean reference, resumable across crashes. v1 is preserved as `cnv_score_pooled_v1` / `cnv_leiden_pooled_v1` once v2 is promoted to canonical.

To use v1 from cache and skip the pooled recompute, set `FORCE_RECOMPUTE_CNV = False` in the cell above before running this section.

```python  # cell 8
# Cell A — v2 config

SAMPLE_KEY = "donor"
assert SAMPLE_KEY in adata.obs.columns, f"missing {SAMPLE_KEY!r} in obs"

# Disease-neutral reference: mesenchymal / endothelial / pericyte / melanocyte / nerve /
# non-asterisked keratinocyte subsets. Explicitly excludes CTCL-enriched F2/F3/VE2 and
# the asterisked Differentiated_KC* disease-state variant.
DIPLOID_REF_V2_CANDIDATES = [
    "F1",
    "VE1",
    "Pericyte_1", "Pericyte_2",
    "Melanocyte",
    "Schwann_1",
    "LE1", "LE2",
    "Sebaceous",
    "Undifferentiated_KC", "Proliferating_KC",
]
_all_cts = set(adata.obs["cell_type"].astype(str).unique())
DIPLOID_REF_V2 = [c for c in DIPLOID_REF_V2_CANDIDATES if c in _all_cts]
print("DIPLOID_REF_V2:", DIPLOID_REF_V2)
assert len(DIPLOID_REF_V2) >= 3, "need >=3 reference cell types in atlas; check cell type spellings"

# Fallback for PBMC-only donors with no stromal/epithelial cells (Sezary-syndrome leukemic samples).
PBMC_FALLBACK_REF_CANDIDATES = [
    "B_cell", "Mono_mac", "NK", "Mast_cell",
    "Macro_1", "Macro_2", "Inf_mac",
    "DC1", "DC2", "pDC", "MigDC",
]
PBMC_FALLBACK_REF = [c for c in PBMC_FALLBACK_REF_CANDIDATES if c in _all_cts]
print("PBMC_FALLBACK_REF:", PBMC_FALLBACK_REF)

CNV_CACHE_V2 = CACHE_DIR / "infercnv_cnv_cache_v2_per_sample.h5ad"
CNV_PER_SAMPLE_DIR = CACHE_DIR / "infercnv_per_sample"
CNV_PER_SAMPLE_DIR.mkdir(exist_ok=True)
print("CNV_CACHE_V2          =", CNV_CACHE_V2)
print("CNV_PER_SAMPLE_DIR    =", CNV_PER_SAMPLE_DIR)
print(f"donors: {adata.obs[SAMPLE_KEY].nunique()}  (will run inferCNV once per donor)")
```

```python  # cell 9
# Cell B — per-donor inferCNV loop (resumable)
import traceback

MIN_REF_CELLS = 50
WINDOW_SIZE   = 250
N_JOBS        = 8

donors = list(adata.obs[SAMPLE_KEY].astype(str).unique())
per_sample_obs = []
skipped = []

def _safe_donor_filename(d):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(d))

for d in donors:
    out_path = CNV_PER_SAMPLE_DIR / f"{_safe_donor_filename(d)}.h5ad"
    if out_path.exists():
        side = sc.read_h5ad(out_path)
        per_sample_obs.append(side.obs[["cnv_score", "cnv_leiden_unit"]])
        print(f"[{d}] cached -> {out_path.name}  (n={side.n_obs})")
        continue

    a = adata[adata.obs[SAMPLE_KEY].astype(str) == str(d)].copy()
    sample_cts = set(a.obs["cell_type"].astype(str).unique())
    ref = [c for c in DIPLOID_REF_V2 if c in sample_cts]
    n_ref = int(a.obs["cell_type"].astype(str).isin(ref).sum())

    if n_ref < MIN_REF_CELLS:
        ref_pbmc = [c for c in PBMC_FALLBACK_REF if c in sample_cts]
        n_ref_pbmc = int(a.obs["cell_type"].astype(str).isin(ref_pbmc).sum())
        if n_ref_pbmc >= MIN_REF_CELLS:
            print(f"[{d}] PBMC fallback ref={ref_pbmc}  (n_ref={n_ref_pbmc})")
            ref = ref_pbmc
            n_ref = n_ref_pbmc
        else:
            print(f"[{d}] SKIP — stromal={n_ref}  pbmc={n_ref_pbmc}  (need {MIN_REF_CELLS})")
            skipped.append((d, n_ref, n_ref_pbmc))
            continue

    print(f"[{d}] n_cells={a.n_obs:>6}  n_ref={n_ref:>5}  ref={ref}")
    try:
        cnv.tl.infercnv(
            a,
            reference_key="cell_type",
            reference_cat=ref,
            window_size=WINDOW_SIZE,
            n_jobs=N_JOBS,
        )
        cnv.tl.pca(a)
        cnv.pp.neighbors(a)
        cnv.tl.leiden(a, key_added="cnv_leiden_unit")
        cnv.tl.cnv_score(a)
    except Exception as e:
        print(f"[{d}] FAILED: {e}")
        traceback.print_exc()
        skipped.append((d, str(e), None))
        continue

    a.obs["cnv_leiden_unit"] = str(d) + "_" + a.obs["cnv_leiden_unit"].astype(str)

    side = ad.AnnData(
        X=np.zeros((a.n_obs, 1), dtype=np.float32),
        obs=a.obs[["cnv_score", "cnv_leiden_unit"]].copy(),
    )
    if "X_cnv" in a.obsm:
        side.obsm["X_cnv"] = a.obsm["X_cnv"]
    if "X_cnv_pca" in a.obsm:
        side.obsm["X_cnv_pca"] = a.obsm["X_cnv_pca"]
    if "cnv" in a.uns:
        side.uns["cnv"] = a.uns["cnv"]
    side.write_h5ad(out_path, compression="gzip")
    per_sample_obs.append(side.obs[["cnv_score", "cnv_leiden_unit"]])
    print(f"[{d}] saved -> {out_path.name}")

print(f"\nProcessed: {len(per_sample_obs)} donors")
print(f"Skipped:   {len(skipped)} donors")
for s in skipped:
    print(f"  {s}")
```

```python  # cell 10
# Cell C — stitch per-donor results, back up v1, promote v2 to canonical

assert len(per_sample_obs) > 0, "no per-donor results to stitch; rerun Cell B"

cnv_obs = pd.concat(per_sample_obs)
print(f"Total cells with per-donor CNV results: {len(cnv_obs):,} / {adata.n_obs:,}")

# Back up v1 pooled results before overwriting canonical names
adata.obs["cnv_score_pooled_v1"]  = adata.obs["cnv_score"].astype(float)
adata.obs["cnv_leiden_pooled_v1"] = adata.obs["cnv_leiden"].astype("category")

# Promote v2 to canonical
adata.obs["cnv_score"]  = cnv_obs["cnv_score"].reindex(adata.obs_names).astype(float)
adata.obs["cnv_leiden"] = (
    cnv_obs["cnv_leiden_unit"].reindex(adata.obs_names).astype("category")
)

n_missing = int(adata.obs["cnv_score"].isna().sum())
print(f"Cells without v2 CNV score (from skipped donors): {n_missing:,}")

# Sanity numbers (per the plan's verification step 3)
ref_mask_v2 = adata.obs["cell_type"].astype(str).isin(DIPLOID_REF_V2)
tcell_mask  = adata.obs["cell_type"].astype(str).isin(
    ["tumor_cell", "Th", "Tc", "Treg", "Tc17_Th17", "Tc_IL13_IL22"]
)
ref_mean    = adata.obs.loc[ref_mask_v2, "cnv_score"].mean()
tcell_mean  = adata.obs.loc[tcell_mask,  "cnv_score"].mean()
print(f"\nv2 cnv_score: mean={adata.obs['cnv_score'].mean():.4f}  std={adata.obs['cnv_score'].std():.4f}")
print(f"  reference cells (DIPLOID_REF_V2):  mean={ref_mean:.4f}")
print(f"  T/tumor cells:                     mean={tcell_mean:.4f}")
print(f"  T-vs-ref ratio:                    {tcell_mean/max(ref_mean, 1e-9):.2f}  (>1.5 = good separation)")
print(f"\n(v1 pooled was mean=0.0079, std=0.0049 with contaminated reference)")

# Save lean v2 cache (obs columns only — per-donor obsm lives in CNV_PER_SAMPLE_DIR/*.h5ad)
side = ad.AnnData(
    X=np.zeros((adata.n_obs, 1), dtype=np.float32),
    obs=adata.obs[
        ["cnv_score", "cnv_leiden", "cnv_score_pooled_v1", "cnv_leiden_pooled_v1"]
    ].copy(),
)
side.write_h5ad(CNV_CACHE_V2, compression="gzip")
print(f"\nSaved v2 cache -> {CNV_CACHE_V2}  ({CNV_CACHE_V2.stat().st_size/1e6:.1f} MB)")
```

```python  # cell 11
# Per-donor GMM threshold on T-cell cnv_score.
# Donor-local threshold is the right unit for the v2 (per-sample) cnv_score:
# the GMM means/midpoint scale with each donor's own background. Adds a NaN guard
# for donors that were skipped in Cell B and a percentile fallback when the GMM
# means collapse (no detectable bimodality).
T_LABELS = {"Th", "Tc", "Treg", "Tc17_Th17", "Tc_IL13_IL22", "tumor_cell"}
t_mask = adata.obs["cell_type"].astype(str).isin(T_LABELS)
adata.obs["is_T"] = t_mask
print("T-cell pool by cell_type:")
print(adata.obs.loc[t_mask, "cell_type"].value_counts())

# DIPLOID_REF_V2 defined in Cell A; fall back to v1 list if running standalone.
_ref_cts = set(globals().get("DIPLOID_REF_V2", DIPLOID))

def per_donor_threshold(scores, ref_scores=None, min_n=100, mean_eps=1e-4):
    """Donor-local cnv_score threshold for malignant call.

    GMM cross-weighted midpoint when the two components are distinguishable;
    otherwise fall back to the 99th percentile of within-donor reference cells
    (or the 90th percentile of all scores if no reference is provided).
    """
    scores = np.asarray(scores, dtype=float)
    if len(scores) < min_n:
        return float(np.percentile(scores, 90))
    gm = GaussianMixture(n_components=2, random_state=0).fit(scores.reshape(-1, 1))
    mu = gm.means_.ravel()
    w  = gm.weights_.ravel()
    if abs(mu[0] - mu[1]) < mean_eps:
        # GMM degenerated to one component — fall back to ref percentile if available
        if ref_scores is not None and len(ref_scores) >= 20:
            return float(np.percentile(ref_scores, 99))
        return float(np.percentile(scores, 90))
    return float((w[0] * mu[1] + w[1] * mu[0]) / w.sum())

t_view = adata[t_mask].copy()
is_mal = np.zeros(t_view.n_obs, dtype=bool)
skipped_donors = []
for d, sub_idx in t_view.obs.groupby("donor", observed=True).indices.items():
    s = t_view.obs["cnv_score"].iloc[sub_idx].values
    valid = ~np.isnan(s)
    if valid.sum() == 0:
        skipped_donors.append(d)
        continue  # donor was skipped in Cell B → leave is_mal=False
    # Within-donor reference scores (for fallback in degenerate-GMM case)
    donor_mask = adata.obs["donor"].astype(str) == str(d)
    ref_donor = adata.obs.loc[
        donor_mask & adata.obs["cell_type"].astype(str).isin(_ref_cts),
        "cnv_score",
    ].dropna().values
    thr = per_donor_threshold(s[valid], ref_scores=ref_donor)
    is_mal[sub_idx[valid]] = s[valid] > thr

t_view.obs["is_malignant"] = is_mal
adata.obs["is_malignant"] = False
adata.obs.loc[t_view.obs_names, "is_malignant"] = is_mal

rate = t_view.obs.groupby("donor", observed=True)["is_malignant"].mean()
print("per-donor malignant fraction:\n", rate.describe())
print(f"\nT cells: malignant={int(is_mal.sum())}  benign={int((~is_mal).sum())}")
if skipped_donors:
    print(f"Donors with no v2 cnv_score (skipped in Cell B): {skipped_donors}")
```

```python  # cell 12
import infercnvpy as cnv

# Canonical chromosome heatmap: genomic ordering across chromosomes, rows grouped
# into diploid reference / T-benign / T-malignant. Subsampled per group for legibility.
rng = np.random.default_rng(0)
N_PER_GROUP = 200

groups = pd.Series("other", index=adata.obs_names, dtype=object)
groups[adata.obs["cell_type"].astype(str).isin(DIPLOID)] = "ref"
groups[t_mask & ~adata.obs["is_malignant"]] = "T_benign"
groups[t_mask &  adata.obs["is_malignant"]] = "T_malignant"
adata.obs["cnv_group"] = pd.Categorical(groups, categories=["ref", "T_benign", "T_malignant", "other"])

pick = []
for g in ["ref", "T_benign", "T_malignant"]:
    sub = np.where(adata.obs["cnv_group"] == g)[0]
    take = sub if len(sub) <= N_PER_GROUP else rng.choice(sub, N_PER_GROUP, replace=False)
    pick.append(take)
pick = np.concatenate(pick)
ad_plot = adata[pick].copy()
ad_plot.obs["cnv_group"] = ad_plot.obs["cnv_group"].cat.remove_unused_categories()
print(ad_plot.obs["cnv_group"].value_counts())

# CNV magnitudes are small (~±0.05); clamp colormap so the bands are punchy instead of washed out.
X_cnv = ad_plot.obsm["X_cnv"]
data = X_cnv.data if hasattr(X_cnv, "data") else np.asarray(X_cnv).ravel()
vmag = float(np.nanpercentile(np.abs(data), 99))
print(f"X_cnv: shape={X_cnv.shape}  99th-pct |val|={vmag:.4f}")

# Per-cell heatmap (rows = cells). Tall figure so rows are visible.
axes = cnv.pl.chromosome_heatmap(
    ad_plot, groupby="cnv_group",
    figsize=(18, 14),
    vmin=-vmag, vmax=vmag,
    show=False,
)
fig = axes["heatmap_ax"].figure
fig.savefig(FIG_DIR / "cnv_chromosome_heatmap.png", dpi=200, bbox_inches="tight")
plt.show()

# Per-group summary heatmap (each group averaged → unmistakable visual comparison).
axes_s = cnv.pl.chromosome_heatmap_summary(
    ad_plot, groupby="cnv_group",
    figsize=(18, 4),
    vmin=-vmag/2, vmax=vmag/2,
    show=False,
)
fig_s = axes_s["heatmap_ax"].figure
fig_s.savefig(FIG_DIR / "cnv_chromosome_heatmap_summary.png", dpi=200, bbox_inches="tight")
plt.show()
```

```
cnv_group
ref            200
T_benign       200
T_malignant    200
Name: count, dtype: int64
X_cnv: shape=(600, 444)  99th-pct |val|=0.1750

<Figure size 1440x1120 with 4 Axes>
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/anndata/_core/anndata.py:252: ImplicitModificationWarning: X should not be a np.matrix, use np.ndarray instead.
  self._init_as_actual(
... storing 'cnv_group' as categorical

<Figure size 1440x320 with 4 Axes>
```

```python  # cell 13
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import infercnvpy as cnv

# Canonical chromosome heatmap: genomic ordering across chromosomes, rows grouped
# into diploid reference / T-benign / T-malignant. Subsampled per group for legibility.
rng = np.random.default_rng(0)
N_PER_GROUP = 200

groups = pd.Series("other", index=adata.obs_names, dtype=object)
groups[adata.obs["cell_type"].astype(str).isin(DIPLOID)] = "ref"
groups[t_mask & ~adata.obs["is_malignant"]] = "T_benign"
groups[t_mask &  adata.obs["is_malignant"]] = "T_malignant"
adata.obs["cnv_group"] = pd.Categorical(groups, categories=["ref", "T_benign", "T_malignant", "other"])

pick = []
for g in ["ref", "T_benign", "T_malignant"]:
    sub = np.where(adata.obs["cnv_group"] == g)[0]
    take = sub if len(sub) <= N_PER_GROUP else rng.choice(sub, N_PER_GROUP, replace=False)
    pick.append(take)
pick = np.concatenate(pick)
ad_plot = adata[pick].copy()
ad_plot.obs["cnv_group"] = ad_plot.obs["cnv_group"].cat.remove_unused_categories()
print(ad_plot.obs["cnv_group"].value_counts())

# Hardcode the limits for scRNA-seq CNVs to prevent noise from washing out the signal
vmag = 0.08  

# Per-cell heatmap (rows = cells). Tall figure so rows are visible.
axes = cnv.pl.chromosome_heatmap(
    ad_plot, groupby="cnv_group",
    figsize=(18, 14),
    vmin=-vmag, vmax=vmag,
    cmap="bwr",
    show=False,
)
fig = axes["heatmap_ax"].figure
fig.savefig(FIG_DIR / "cnv_chromosome_heatmap.png", dpi=200, bbox_inches="tight")
plt.show()

# Per-group summary heatmap (each group averaged → unmistakable visual comparison).
axes_s = cnv.pl.chromosome_heatmap_summary(
    ad_plot, groupby="cnv_group",
    figsize=(18, 4),
    vmin=-vmag, vmax=vmag,
    cmap="bwr",
    show=False,
)
# Both heatmap and summary use "heatmap_ax" in infercnvpy
fig_s = axes_s["heatmap_ax"].figure
fig_s.savefig(FIG_DIR / "cnv_chromosome_heatmap_summary.png", dpi=200, bbox_inches="tight")
plt.show()
```

```
cnv_group
ref            200
T_benign       200
T_malignant    200
Name: count, dtype: int64

<Figure size 1440x1120 with 4 Axes>
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/anndata/_core/anndata.py:252: ImplicitModificationWarning: X should not be a np.matrix, use np.ndarray instead.
  self._init_as_actual(

<Figure size 1440x320 with 4 Axes>
```

```python  # cell 14
ad_t = adata[t_mask].copy()
sc.pp.neighbors(ad_t, use_rep="X_mrvi", n_neighbors=30, random_state=RNG)
sc.tl.umap(ad_t, min_dist=0.3, random_state=RNG)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sc.pl.umap(ad_t, color="donor", ax=axes[0], show=False, legend_loc="none", size=3, title="donor")
sc.pl.umap(ad_t, color="is_malignant", ax=axes[1], show=False, size=3, title="is_malignant")
sc.pl.umap(ad_t, color="cnv_score", ax=axes[2], show=False, cmap="viridis", size=3, title="cnv_score")
fig.suptitle("Fig 3a — T cells (MrVI UMAP)")
fig.tight_layout(); fig.savefig(FIG_DIR / "Fig3a_Tcells_donor_malignant.png", dpi=200); plt.show()

TCELL_CACHE = CACHE_DIR / "tcells_cnv.h5ad"
ad_t.write_h5ad(TCELL_CACHE, compression="gzip")
print("cached ->", TCELL_CACHE)
```

```python  # cell 15
# Crosstab: new inferCNV-based is_malignant vs paper's cell_type == "tumor_cell"
prev_tumor = adata.obs["cell_type"].astype(str).eq("tumor_cell")
ct = pd.crosstab(
    prev_tumor.rename("prev_tumor_cell"),
    adata.obs["is_malignant"].rename("new_is_malignant"),
    margins=True,
)
print("all cells:")
print(ct)

ct_t = pd.crosstab(
    prev_tumor[t_mask].rename("prev_tumor_cell"),
    adata.obs.loc[t_mask, "is_malignant"].rename("new_is_malignant"),
    margins=True,
)
print("\nT cells only:")
print(ct_t)

both = int((prev_tumor & adata.obs["is_malignant"]).sum())
only_prev = int((prev_tumor & ~adata.obs["is_malignant"]).sum())
only_new  = int((~prev_tumor & adata.obs["is_malignant"]).sum())
union = both + only_prev + only_new
print(f"\nagreement (T cells): both={both}  only_prev={only_prev}  only_new={only_new}  Jaccard={both/max(1,union):.3f}")
```

```python  # cell 16
adata[adata.obs.cell_type=='tumor_cell'].obs.is_malignant.value_counts()
```

```python  # cell 17
adata.obs.head()
```

```python  # cell 18
# 1. Are cnv_leiden clusters confounded with sample/patient identity?
ct = pd.crosstab(adata.obs["cnv_leiden"], adata.obs['donor'])
ct_norm = ct.div(ct.sum(axis=1), axis=0)
# If any cluster has >70% of its cells from a single sample, you have batch confounding
max_dominance = ct_norm.max(axis=1)
print(max_dominance.describe())
print((max_dominance > 0.7).sum(), "of", len(max_dominance), "cnv clusters dominated by a single sample")

# 2. Does cnv_score correlate with sample identity more than with cell type?
import scipy.stats as st
print("ANOVA cnv_score ~ sample_id:", st.f_oneway(*[g["cnv_score"].values for _, g in adata.obs.groupby('donor')]))
print("ANOVA cnv_score ~ cell_type:", st.f_oneway(*[g["cnv_score"].values for _, g in adata.obs.groupby("cell_type")]))

# 3. For patients where TCR data exists, what's the agreement between dominant clone and high cnv_score?
# This is the most important biological sanity check — should be high (>50%) per patient.
```
