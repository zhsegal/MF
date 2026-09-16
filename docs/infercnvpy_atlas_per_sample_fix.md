# Fixing the infercnvpy Atlas Run — Per-Sample Loop

*Instructions for Claude Code. Goal: rerun infercnvpy on the CTCL atlas per-sample (rather than as a single pooled call across all 45 patients), with safer reference cell types, and validate the fix. Companion to `infercnvpy_mf_workflow.md` — this assumes that doc has been read.*

The previous run pooled all patients into one `cnv.tl.infercnv()` call, which lets cross-patient technical variation contaminate the CNV signal. This fix runs CNV inference one sample at a time, then stitches the per-sample results back onto the master AnnData.

**Do not delete the existing cache yet.** First confirm the diagnosis (Section 2), then recompute under a new cache name (Section 4), then compare (Section 5). Only delete the old cache once the new run has been validated.

---

## Step 0 — Working environment & assumptions

Assumed already in the namespace:
- `adata` — the master atlas AnnData (~420k cells).
- `CACHE_DIR` — path object pointing to the cache directory.
- `GTF_LOCAL` — path to the downloaded Ensembl GTF.
- The workflow doc has been read for context.

Required imports for this fix:

```python
import scanpy as sc
import anndata as ad
import infercnvpy as cnv
import numpy as np
import pandas as pd
import scipy.stats as st
from pathlib import Path
```

---

## Step 1 — Pre-flight inspection (no compute, no writes)

Before changing anything, inspect the AnnData to determine the correct column names and reference cell types. **Do not hardcode these from memory** — atlas labeling differs from notebook-to-notebook.

```python
# 1a. Find the sample/donor/patient key
candidate_keys = [c for c in adata.obs.columns
                  if any(s in c.lower() for s in ["sample", "donor", "patient", "subject", "library"])]
print("Candidate sample keys:", candidate_keys)
for c in candidate_keys:
    n_unique = adata.obs[c].nunique()
    print(f"  {c}: {n_unique} unique values")

# 1b. Find the technology/library-method key (for stratifying 10x 5' fresh vs 10x Flex FFPE etc.)
tech_keys = [c for c in adata.obs.columns
             if any(s in c.lower() for s in ["library", "tech", "chemistry", "method", "platform", "modality"])]
print("Candidate technology keys:", tech_keys)
for c in tech_keys:
    print(f"  {c}: {adata.obs[c].value_counts().to_dict()}")

# 1c. Find a TCR clonotype key if present (for downstream agreement check)
tcr_keys = [c for c in adata.obs.columns
            if any(s in c.lower() for s in ["clonotype", "clone", "tcr", "cta", "ctaa"])]
print("Candidate TCR keys:", tcr_keys)

# 1d. Enumerate cell types and flag disease-state-associated subsets
all_cts = sorted(adata.obs["cell_type"].astype(str).unique())
print(f"\n{len(all_cts)} cell types:")
for ct in all_cts:
    n = (adata.obs["cell_type"] == ct).sum()
    print(f"  {ct}: {n}")
```

**Decision point.** From the output above, choose:

- `SAMPLE_KEY`: the column with one value per biological sample (typically 30–60 unique values for a CTCL atlas). Prefer the most granular sample-level key over a coarser donor-level key — different lesions / time points from the same donor have different infiltrate composition and should be treated as separate inference units.
- `TECH_KEY`: the column distinguishing chemistries (10x 5′ vs 10x Flex vs published 10x 3′). If multiple chemistries are present, the loop in Step 4 should also be stratified by tech.
- `DIPLOID_REF`: a list of cell type names to use as reference. Apply the rules below.

### Reference cell type selection rules

Pick cell types that satisfy **all** of these:

1. Mesenchymal / epithelial / endothelial — not T or B lymphocytes (no shared lineage with malignancy).
2. Not preferentially expanded in CTCL — exclude any subset known to be CTCL-enriched.
3. Not labeled as a disease-state variant (no asterisk, no `_CTCL`, no `_lesional` suffix).
4. Present in ≥ ~50 cells per sample on average across the atlas (otherwise the loop will skip too many patients).

For the Li 2024 *Nat Immunol* atlas specifically:

| Likely safe to include | Likely unsafe — exclude |
|---|---|
| `F1` (canonical reticular fibroblast) | `F2` (MHC-II+ antigen-presenting fibroblast — **CTCL-enriched headline finding**) |
| `VE1` (canonical postcapillary venule) | `F3` (likely another disease-state fibroblast — verify) |
| `Pericyte_1`, `Pericyte_2` | `VE2`, `VE3` (verify whether disease-associated) |
| `Melanocyte` | `Differentiated_KC*` (the asterisk denotes a disease-modified variant) |
| `Basal_KC`, `Spinous_KC` (if present and not asterisked) | Anything labeled `*`, `_CTCL`, `_lesional` |
| `Schwann`, `LE` (lymphatic endothelial) if available | Any T cell, B cell, plasma cell, NK cell |

If unsure about a specific subset, check the original publication's Figure 1 / supplementary cell type annotations to see whether it's described as CTCL-enriched.

Codify the choice:

```python
SAMPLE_KEY = "..."   # fill in from 1a
TECH_KEY = "..."     # fill in from 1b, or None if uniform
TCR_KEY = "..."      # fill in from 1c, or None if no TCR data

DIPLOID_REF_CANDIDATES = [
    "F1", "VE1", "Pericyte_1", "Pericyte_2", "Melanocyte",
    "Basal_KC", "Spinous_KC", "LE", "Schwann",
    # add others that survive the rules above
]
DIPLOID_REF = [c for c in DIPLOID_REF_CANDIDATES if c in set(all_cts)]
print("Diploid reference shortlist:", DIPLOID_REF)
assert len(DIPLOID_REF) >= 3, "Need at least 3 reference cell types; review candidates above"
```

Persist these choices to a config file for reproducibility:

```python
import json
config = {
    "sample_key": SAMPLE_KEY,
    "tech_key": TECH_KEY,
    "tcr_key": TCR_KEY,
    "diploid_ref": DIPLOID_REF,
    "atlas_n_obs": int(adata.n_obs),
    "atlas_n_samples": int(adata.obs[SAMPLE_KEY].nunique()),
}
(CACHE_DIR / "cnv_v2_config.json").write_text(json.dumps(config, indent=2))
print("Saved config to cnv_v2_config.json")
```

---

## Step 2 — Diagnose the existing pooled run (confirm the problem)

Run these checks on the already-cached `adata.obs["cnv_score"]` / `adata.obs["cnv_leiden"]` from the pooled run. **If the problem isn't real, the rerun is wasted compute.**

```python
# 2a. Are cnv_leiden clusters confounded with sample identity?
ct = pd.crosstab(adata.obs["cnv_leiden"], adata.obs[SAMPLE_KEY])
ct_norm = ct.div(ct.sum(axis=1), axis=0)
max_dominance = ct_norm.max(axis=1)
print("Per-cluster sample-dominance (fraction of cluster from its single most common sample):")
print(max_dominance.describe())
n_dominated = (max_dominance > 0.7).sum()
print(f"{n_dominated} of {len(max_dominance)} cnv_leiden clusters are >70% from a single sample")

# 2b. ANOVA: does cnv_score vary more by sample or by cell type?
sample_groups = [g["cnv_score"].values for _, g in adata.obs.groupby(SAMPLE_KEY) if len(g) > 5]
celltype_groups = [g["cnv_score"].values for _, g in adata.obs.groupby("cell_type") if len(g) > 5]
F_sample,   p_sample   = st.f_oneway(*sample_groups)
F_celltype, p_celltype = st.f_oneway(*celltype_groups)
print(f"F(cnv_score ~ sample)    = {F_sample:.1f}   p={p_sample:.2e}")
print(f"F(cnv_score ~ cell_type) = {F_celltype:.1f}   p={p_celltype:.2e}")
print(f"Ratio F_sample / F_celltype = {F_sample/F_celltype:.2f}")

# 2c. Are reference cells themselves flagged as aneuploid in the pooled run?
ref_mask = adata.obs["cell_type"].isin(DIPLOID_REF)
ref_score = adata.obs.loc[ref_mask, "cnv_score"]
nonref_score = adata.obs.loc[~ref_mask, "cnv_score"]
print(f"Reference cells:    mean cnv_score = {ref_score.mean():.4f}  (n={ref_mask.sum()})")
print(f"Non-reference cells: mean cnv_score = {nonref_score.mean():.4f}  (n={(~ref_mask).sum()})")

# 2d. Per-sample reference contamination — for each sample, what fraction of its reference cells
#     are in cnv_leiden clusters that are >50% non-reference?
nonref_cluster_frac = 1 - ct_norm.loc[:, ref_mask].sum(axis=1) / ct.sum(axis=1)
contaminated_clusters = nonref_cluster_frac[nonref_cluster_frac > 0.5].index.tolist()
ref_in_contam = adata.obs.loc[ref_mask, "cnv_leiden"].isin(contaminated_clusters).mean()
print(f"Reference cells assigned to mostly-tumor cnv_leiden clusters: {ref_in_contam:.2%}")
```

**Interpretation gate.** Proceed to Step 3 if **any** of these hold:

- More than ~30% of cnv_leiden clusters are >70% dominated by a single sample (2a).
- F-statistic for sample is more than ~3× the F-statistic for cell_type (2b). Sample explaining more variance than cell type means batch effects dominate biological signal.
- Mean cnv_score for reference cells is within 30% of mean for non-reference cells (2c). Reference and non-reference should be well separated; if they're not, the inference failed.
- More than 10% of reference cells landed in mostly-tumor cnv_leiden clusters (2d).

If **none** of these hold, the pooled run may actually be fine and the "very different results" come from elsewhere (e.g. cell type label mismatch with published paper, different cell type aggregation, or different downstream thresholding). Stop here and re-examine what "different results" means.

If the diagnosis is confirmed, save the diagnostic output and proceed:

```python
diag_path = CACHE_DIR / "cnv_v1_pooled_diagnostics.txt"
diag_path.write_text(
    f"Pooled run diagnostics ({pd.Timestamp.now()}):\n"
    f"- Clusters dominated by 1 sample (>70%): {n_dominated}/{len(max_dominance)}\n"
    f"- F(sample)/F(cell_type) = {F_sample/F_celltype:.2f}\n"
    f"- Mean cnv_score reference: {ref_score.mean():.4f}\n"
    f"- Mean cnv_score non-reference: {nonref_score.mean():.4f}\n"
    f"- Reference cells in mostly-tumor clusters: {ref_in_contam:.2%}\n"
)
print(f"Saved diagnostics to {diag_path}")
```

---

## Step 3 — Set up the new (versioned) cache and gene annotation

Do **not** overwrite the existing cache. Use a new file name so both results coexist for comparison.

```python
CNV_CACHE_V2 = CACHE_DIR / "infercnv_cnv_cache_v2_per_sample.h5ad"
CNV_PER_SAMPLE_DIR = CACHE_DIR / "infercnv_per_sample"
CNV_PER_SAMPLE_DIR.mkdir(exist_ok=True)
print(f"New cache will be written to: {CNV_CACHE_V2}")
print(f"Per-sample intermediates in:  {CNV_PER_SAMPLE_DIR}")
```

Ensure `adata.X` is log-normalized (idempotent check from the original code):

```python
if "log1p" not in adata.uns:
    assert "raw_counts" in adata.layers, "expected raw_counts layer"
    adata.X = adata.layers["raw_counts"].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print("log-normalised adata.X")
else:
    print("adata.X already log1p-normalised")
```

Annotate genomic positions on the master adata once (this is global and safe to do across samples — each per-sample slice will inherit `var`):

```python
if not adata.var.get("chromosome", pd.Series(dtype=str)).notna().any():
    adata.var = adata.var.drop(
        columns=["chromosome", "start", "end", "gene_id", "gene_name"],
        errors="ignore",
    )
    cnv.io.genomic_position_from_gtf(str(GTF_LOCAL), adata=adata)
n_annot = adata.var[["chromosome", "start", "end"]].notna().all(axis=1).sum()
print(f"Genes with positions: {n_annot} / {adata.n_vars}")
assert n_annot > 10000, "Too few genes annotated — check GTF gene ID type (gene_name vs gene_id)"
```

---

## Step 4 — Per-sample inference loop

This is the core fix. Loop over `SAMPLE_KEY` (and `TECH_KEY` if present), running `cnv.tl.infercnv()` independently per sample with a within-sample reference.

```python
MIN_REF_CELLS = 50   # skip samples with fewer than this many reference cells
WINDOW_SIZE = 250    # match previous run for comparability; can revisit
N_JOBS = 8

# Build the iteration unit. If multiple technologies, stratify; else just sample.
if TECH_KEY is not None and adata.obs[TECH_KEY].nunique() > 1:
    iter_key = adata.obs[SAMPLE_KEY].astype(str) + "__" + adata.obs[TECH_KEY].astype(str)
    print(f"Stratifying by {SAMPLE_KEY} × {TECH_KEY}: {iter_key.nunique()} units")
else:
    iter_key = adata.obs[SAMPLE_KEY].astype(str)
    print(f"Stratifying by {SAMPLE_KEY}: {iter_key.nunique()} units")
adata.obs["_iter_unit"] = iter_key.values

per_sample_obs = []        # per-sample obs rows to stitch back
skipped = []               # samples that failed the reference threshold or other QC
fallback_pbmc_ref = ["B_cell", "Monocyte", "NK", "Mast", "Macrophage", "DC"]  # for PBMC-only samples

for unit in adata.obs["_iter_unit"].unique():
    out_path = CNV_PER_SAMPLE_DIR / f"{unit}.h5ad"
    if out_path.exists():
        # Already done in a previous pass; reload obs and continue
        side = sc.read_h5ad(out_path)
        per_sample_obs.append(side.obs[["cnv_score", "cnv_leiden_unit"]])
        print(f"[{unit}] cached, skipping recompute")
        continue

    a = adata[adata.obs["_iter_unit"] == unit].copy()
    sample_cts = set(a.obs["cell_type"].astype(str).unique())
    ref = [c for c in DIPLOID_REF if c in sample_cts]
    n_ref = a.obs["cell_type"].isin(ref).sum()

    # Fallback for PBMC-only samples with no stromal/epithelial cells
    if n_ref < MIN_REF_CELLS:
        ref_pbmc = [c for c in fallback_pbmc_ref if c in sample_cts]
        n_ref_pbmc = a.obs["cell_type"].isin(ref_pbmc).sum()
        if n_ref_pbmc >= MIN_REF_CELLS:
            print(f"[{unit}] using PBMC fallback reference: {ref_pbmc} (n={n_ref_pbmc})")
            ref = ref_pbmc
            n_ref = n_ref_pbmc
        else:
            print(f"[{unit}] SKIP — only {n_ref} stromal + {n_ref_pbmc} PBMC reference cells")
            skipped.append((unit, n_ref, n_ref_pbmc))
            continue

    print(f"[{unit}] n_cells={a.n_obs:>6}  n_ref={n_ref:>5}  ref={ref}")

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
        print(f"[{unit}] FAILED: {e}")
        skipped.append((unit, str(e), None))
        continue

    # Prefix the cluster label with the unit so labels don't collide downstream
    a.obs["cnv_leiden_unit"] = unit + "_" + a.obs["cnv_leiden_unit"].astype(str)

    # Persist just the obs columns + obsm we need; the per-sample h5ad is small
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
    per_sample_obs.append(side.obs)
    print(f"[{unit}] saved -> {out_path.name}")

print(f"\nProcessed: {len(per_sample_obs)} units")
print(f"Skipped:   {len(skipped)} units")
for s in skipped:
    print(f"  {s}")
```

This loop is resumable — if it crashes partway through, rerunning skips already-saved units.

---

## Step 5 — Stitch per-sample results back onto the master adata

```python
# Concatenate all per-sample obs results
cnv_obs = pd.concat(per_sample_obs)
print(f"Total cells with per-sample CNV results: {len(cnv_obs)} (out of {adata.n_obs} in atlas)")

# Overwrite the pooled-run columns with the per-sample ones, using v2 suffix for safety
adata.obs["cnv_score_v2"] = cnv_obs["cnv_score"].reindex(adata.obs_names)
adata.obs["cnv_leiden_v2"] = cnv_obs["cnv_leiden_unit"].reindex(adata.obs_names).astype("category")

# Cells in skipped units will have NaN cnv_score_v2 — handle explicitly
n_missing = adata.obs["cnv_score_v2"].isna().sum()
print(f"Cells without v2 CNV score (from skipped units): {n_missing}")

# Save the v2 cache
side = ad.AnnData(
    X=np.zeros((adata.n_obs, 1), dtype=np.float32),
    obs=adata.obs[["cnv_score_v2", "cnv_leiden_v2"]].copy(),
)
side.write_h5ad(CNV_CACHE_V2, compression="gzip")
print(f"Saved v2 cache: {CNV_CACHE_V2}  ({CNV_CACHE_V2.stat().st_size/1e6:.1f} MB)")
```

If you want the v2 results to live in the canonical column names (`cnv_score`, `cnv_leiden`) once you're confident the rerun is correct, do that explicitly at the end of Step 6 — not now.

---

## Step 6 — Validate the rerun (re-run Step 2 diagnostics on v2)

```python
# Restrict to cells that have v2 results
v2 = adata[~adata.obs["cnv_score_v2"].isna()].copy()

# 6a. Sample-dominance per cnv_leiden_v2 cluster
ct2 = pd.crosstab(v2.obs["cnv_leiden_v2"], v2.obs[SAMPLE_KEY])
ct2_norm = ct2.div(ct2.sum(axis=1), axis=0)
max_dom_v2 = ct2_norm.max(axis=1)
n_dom_v2 = (max_dom_v2 > 0.7).sum()
print(f"v2: {n_dom_v2}/{len(max_dom_v2)} clusters >70% from one sample (expected — clusters were prefixed by unit)")

# Better v2 test: within each sample's own clusters, do they distinguish biology?
# Drop the unit prefix and check intra-sample homogeneity of cell types
v2.obs["_clust_local"] = v2.obs["cnv_leiden_v2"].astype(str).str.split("_", n=1).str[-1]
# Within each sample, cell type composition per local cluster
sample_examples = v2.obs[SAMPLE_KEY].value_counts().head(3).index.tolist()
for s in sample_examples:
    sub = v2.obs[v2.obs[SAMPLE_KEY] == s]
    if len(sub) < 100:
        continue
    print(f"\nSample {s}: cell_type × local cnv cluster")
    print(pd.crosstab(sub["cell_type"], sub["_clust_local"]).head(10))

# 6b. ANOVA on v2 — sample F-statistic should drop relative to cell type
sample_groups_v2 = [g["cnv_score_v2"].dropna().values for _, g in v2.obs.groupby(SAMPLE_KEY) if g["cnv_score_v2"].notna().sum() > 5]
celltype_groups_v2 = [g["cnv_score_v2"].dropna().values for _, g in v2.obs.groupby("cell_type") if g["cnv_score_v2"].notna().sum() > 5]
F_sample_v2,   _ = st.f_oneway(*sample_groups_v2)
F_celltype_v2, _ = st.f_oneway(*celltype_groups_v2)
print(f"\nv2: F(cnv_score ~ sample)    = {F_sample_v2:.1f}")
print(f"v2: F(cnv_score ~ cell_type) = {F_celltype_v2:.1f}")
print(f"v2: ratio F_sample / F_celltype = {F_sample_v2/F_celltype_v2:.2f}")
print(f"(compare to v1: {F_sample/F_celltype:.2f}; ratio should drop substantially)")

# 6c. Reference cells should have low cnv_score_v2 within each sample
ref_mask_v2 = v2.obs["cell_type"].isin(DIPLOID_REF)
ref_v2 = v2.obs.loc[ref_mask_v2, "cnv_score_v2"]
nonref_v2 = v2.obs.loc[~ref_mask_v2, "cnv_score_v2"]
print(f"\nv2 reference cells:     mean={ref_v2.mean():.4f}")
print(f"v2 non-reference cells: mean={nonref_v2.mean():.4f}")
print(f"v2 ratio non-ref / ref: {nonref_v2.mean()/ref_v2.mean():.2f}")

# 6d. TCR-CNV agreement per sample (the most important biological check)
if TCR_KEY is not None:
    agreement = []
    for s in v2.obs[SAMPLE_KEY].unique():
        sub = v2.obs[v2.obs[SAMPLE_KEY] == s]
        if len(sub) < 100 or sub[TCR_KEY].nunique() < 2:
            continue
        # Find the dominant TCR clone within this sample
        clone_counts = sub[TCR_KEY].value_counts()
        if clone_counts.iloc[0] < 20:
            continue
        dominant = clone_counts.index[0]
        is_dom = sub[TCR_KEY] == dominant
        # Compare cnv_score distributions
        score_dom = sub.loc[is_dom, "cnv_score_v2"].dropna()
        score_other = sub.loc[~is_dom, "cnv_score_v2"].dropna()
        if len(score_dom) < 10 or len(score_other) < 10:
            continue
        # Higher cnv_score in dominant clone = good
        ratio = score_dom.mean() / score_other.mean() if score_other.mean() > 0 else np.nan
        agreement.append({
            "sample": s,
            "n_dominant": int(is_dom.sum()),
            "n_other": int((~is_dom).sum()),
            "mean_cnv_dominant": float(score_dom.mean()),
            "mean_cnv_other": float(score_other.mean()),
            "ratio": ratio,
        })
    agreement_df = pd.DataFrame(agreement)
    print(f"\nTCR-CNV agreement (n={len(agreement_df)} samples with usable TCR + CNV):")
    print(agreement_df["ratio"].describe())
    print(f"Samples where dominant clone has higher mean cnv_score: {(agreement_df['ratio'] > 1).sum()}/{len(agreement_df)}")
    agreement_df.to_csv(CACHE_DIR / "cnv_v2_tcr_agreement.tsv", sep="\t", index=False)
```

**Acceptance gate.** The v2 rerun is good if:

- F(sample)/F(cell_type) ratio dropped substantially compared to v1 (target: <1.5; v1 was likely >3).
- Reference cells have notably lower mean cnv_score than non-reference cells within each sample (ratio non-ref/ref > 1.5).
- ≥70% of samples with usable TCR data show dominant-clone cells having higher mean cnv_score than non-clonal cells.
- Per-sample crosstabs of cell_type × local cluster show one or two CNV-positive clusters dominated by T cells (the malignant clone), distinct from clusters dominated by reference cell types.

If acceptance criteria fail, see Step 7.

---

## Step 7 — Targeted inspection on representative samples

Pick 3 samples with very different profiles and inspect them in depth before accepting the rerun globally.

```python
# Choose representative samples
# - one early-stage MF skin (low burden)
# - one advanced-stage MF skin (high burden)
# - one leukemic SS PBMC (if PBMC samples are in the atlas)
inspect = ["..."]  # fill in 3 sample IDs from adata.obs[SAMPLE_KEY].unique()

for s in inspect:
    print(f"\n=== {s} ===")
    side_path = next(CNV_PER_SAMPLE_DIR.glob(f"{s}*.h5ad"), None)
    if side_path is None:
        print("  no per-sample file found — was this sample skipped?")
        continue

    # Reload the per-sample object with full obsm
    a = adata[adata.obs[SAMPLE_KEY] == s].copy()
    side = sc.read_h5ad(side_path)
    a.obsm["X_cnv"] = side.obsm["X_cnv"]
    a.obs["cnv_score"] = side.obs["cnv_score"].values
    a.obs["cnv_leiden"] = side.obs["cnv_leiden_unit"].values
    a.uns["cnv"] = side.uns["cnv"]

    print(f"  n_cells={a.n_obs}, cell types: {a.obs['cell_type'].value_counts().head().to_dict()}")
    print(f"  cnv_score: mean={a.obs['cnv_score'].mean():.4f}, std={a.obs['cnv_score'].std():.4f}")

    # Chromosome heatmap
    cnv.pl.chromosome_heatmap(
        a, groupby="cell_type",
        figsize=(16, 6),
        save=f"_v2_{s}_by_celltype.png",
    )
    cnv.pl.chromosome_heatmap(
        a, groupby="cnv_leiden",
        figsize=(16, 6),
        save=f"_v2_{s}_by_cnvcluster.png",
    )
```

When you view the saved PNGs:

- **MF-typical pattern to see in tumor-stage samples:** red (gain) bands on 7q, 8q, 17q; blue (loss) bands on 9p21, 10q, 17p, 13q. These should appear in T-cell clusters and **not** in reference cell clusters.
- **Reference cells should look mostly grey** (no consistent chromosome-scale deviation) on the heatmap. If reference cells show structured colored bands, the within-sample reference is contaminated or too small.
- **Patch-stage samples may show only subtle signal** — this is expected and is the known limitation of inferCNV-family methods on low-burden disease. Lean on TCR for these.

---

## Step 8 — Promote v2 to canonical (only after acceptance)

Once Step 6 acceptance criteria are met and Step 7 spot-checks look right:

```python
# Promote v2 columns to canonical names
adata.obs["cnv_score_pooled_v1"] = adata.obs["cnv_score"]   # keep v1 for audit trail
adata.obs["cnv_leiden_pooled_v1"] = adata.obs["cnv_leiden"]
adata.obs["cnv_score"] = adata.obs["cnv_score_v2"]
adata.obs["cnv_leiden"] = adata.obs["cnv_leiden_v2"]

# Drop temp columns
adata.obs = adata.obs.drop(columns=["_iter_unit"])

# Final summary
print("Promoted v2 -> canonical. v1 preserved as cnv_score_pooled_v1 / cnv_leiden_pooled_v1.")
print(f"cnv_score: mean={adata.obs['cnv_score'].mean():.4f}, std={adata.obs['cnv_score'].std():.4f}")
print(f"cnv_score missing: {adata.obs['cnv_score'].isna().sum()} cells (from skipped units)")
```

Do **not** delete the old pooled-run cache yet. Keep it for at least one more session in case you need to reconcile downstream analyses that referenced the v1 values.

---

## Step 9 — Update downstream malignancy calling

The per-sample loop changes how thresholding should work. With the pooled run, you could set one global cnv_score threshold. With per-sample, the score distribution differs per sample, so threshold per sample:

```python
# Per-sample reference-percentile threshold
adata.obs["is_aneuploid"] = False
for s in adata.obs[SAMPLE_KEY].unique():
    mask_s = adata.obs[SAMPLE_KEY] == s
    ref_s = mask_s & adata.obs["cell_type"].isin(DIPLOID_REF) & adata.obs["cnv_score"].notna()
    if ref_s.sum() < 20:
        continue
    threshold_s = np.percentile(adata.obs.loc[ref_s, "cnv_score"], 99)
    adata.obs.loc[mask_s & (adata.obs["cnv_score"] > threshold_s), "is_aneuploid"] = True
print(f"Aneuploid cells (per-sample threshold): {adata.obs['is_aneuploid'].sum()}")
```

Then redo the triple-criterion call (TCR + CNV + state) as in Step 8 of the main workflow doc, using the new `is_aneuploid`.

---

## Step 10 — Document the change in the methods record

Append to the methods notes / supplementary methods:

> CNV inference was performed per biological sample using infercnvpy (v.X), with reference cells drawn from disease-neutral mesenchymal, epithelial, and endothelial subsets within each sample (`F1`, `VE1`, `Pericyte_1/2`, `Melanocyte`, `Basal_KC`, `Spinous_KC`; disease-associated subsets `F2`, `F3`, `VE2`, `VE3`, `Differentiated_KC*` were explicitly excluded). For samples with insufficient stromal reference cells (predominantly PBMC samples from leukemic Sézary syndrome patients), B cells, monocytes, NK cells, and dendritic cells served as the fallback reference. CNV inference was stratified by library chemistry (10x 5′ vs 10x Flex FFPE vs published 10x 3′) to avoid technology-driven artifacts. A previous pooled run across all 45 patients produced cnv_leiden clusters confounded with sample identity (F(sample)/F(cell_type) = X.X) and was discarded.

---

## Decision tree summary

```
Step 1: pre-flight inspection — DO NOT compute, only print
   ↓
Step 2: diagnose v1 — confirm batch confounding before recomputing
   ↓                                          ↘
   diagnosis positive                          diagnosis negative
   ↓                                          ↓
   Step 3: set up versioned cache             STOP — investigate other causes
   ↓
   Step 4: per-sample loop (resumable)
   ↓
   Step 5: stitch results
   ↓
   Step 6: validate v2 with same diagnostics
   ↓                                          ↘
   acceptance criteria met                    criteria failed
   ↓                                          ↓
   Step 7: spot-check 3 samples               Re-examine ref cell choice,
   ↓                                          stratification, or window_size
   Step 8: promote v2 to canonical            and rerun
   ↓
   Step 9: per-sample malignancy thresholding
   ↓
   Step 10: methods documentation
```

---

## Notes on edge cases

- **Same patient, multiple lesions / time points.** Treat as separate samples. Different lesions even from the same patient have different infiltrate composition and may have CNV subclones that diverged. Pooling them defeats the point of per-sample inference.
- **Externally integrated published datasets in the atlas.** These were generated in other labs with different protocols. They count as separate technologies for stratification. If the atlas obs has a `dataset` or `source_publication` column, fold that into the iteration unit.
- **Samples with very few cells (<200).** Per-sample inference becomes unstable. Either skip (and document) or merge with samples from the same patient on the same chemistry only.
- **Samples where >80% of cells are malignant T cells.** The within-sample reference is small. The PBMC fallback (B, NK, monocyte, DC) usually applies here.
- **Reference cells skewed by inflammation.** Even "safe" reference cell types in lesional CTCL skin carry some inflammatory signature. The per-sample design is robust to this because the inflammation baseline is shared between reference and tumor cells within the same sample. The pooled design is not.

---

## What to commit / save

- `cnv_v2_config.json` — sample/tech/tcr key choices and reference cell list.
- `cnv_v1_pooled_diagnostics.txt` — record of why the pooled run was discarded.
- `infercnv_cnv_cache_v2_per_sample.h5ad` — the new canonical CNV cache.
- `infercnv_per_sample/*.h5ad` — per-sample intermediates (large; consider gitignoring).
- `cnv_v2_tcr_agreement.tsv` — per-sample TCR-CNV agreement metrics.
- The chromosome heatmap PNGs from Step 7.
