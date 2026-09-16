# Notebook spec: replicating the major findings of the Li/Strobl/Haniffa CTCL atlas (Nat Immunol 2024)

**Target paper** Li, R., Strobl, J., Poyner, E. F. M. *et al.* (2024). *Cutaneous T cell lymphoma atlas reveals malignant TH2 cells supported by a B cell-rich tumor microenvironment.* **Nature Immunology** 25:2320–2330. DOI [10.1038/s41590-024-02018-1](https://doi.org/10.1038/s41590-024-02018-1).

**What this document is** a build spec for a single Jupyter notebook (`ctcl_atlas_replication.ipynb`) that reproduces the major scRNA-seq findings of the paper. Two parallel tracks for every differential-expression comparison: (1) **paper-method DE** — Wilcoxon, edgeR-style quasi-likelihood, Milo for DA, NMF for metaprograms — exactly the tools the authors used; (2) **MrVI DE** — the same biological question solved with `scvi.external.MRVI`, treating donor as the sample covariate and disease/stage as the target covariate, which gives cell-resolution sample-level effect sizes that the paper's bulk-comparison Wilcoxon can't.

**Out of scope** raw FASTQ realignment, the bulk-RNA-seq deconvolution survival analysis (covered in the companion download doc), IHC quantification, RareCyte image analysis. This notebook works from the already-published, integrated AnnData object.

**Inputs (assumed downloaded already; see the companion `CTCL_atlas_download_and_preliminary_analysis.md`):**
- `processed/ctcl_atlas.h5ad` — the integrated atlas, ~420 k cells × ~25–35 k genes, with `obs` containing at minimum `donor`, `disease`, `stage`, `cell_type`, `study` (the integrated source), and `compartment` (epidermis/dermis where available).
- `processed/ctcl_visium.h5ad` — the 23-section Visium object (8 CTCL + 15 healthy).

**Expected runtime** ~2 h on 1 × A100 GPU + 64 GB RAM (most of it MrVI training); ~6–10 h on CPU-only (skip cell2location to one slide, lower MrVI epochs).

---

## 0. Notebook structure at a glance

| § | Cell title | Reproduces | Method used |
|---|---|---|---|
| 0 | Setup, imports, params | — | — |
| 1 | Load atlas, sanity-check, UMAP | Fig 1b, 1c, 1d | scanpy + harmony |
| 2 | Milo DA: cell types enriched in CTCL vs healthy/AD/psoriasis | Fig 1e | milopy (paper method) |
| 3 | Identify malignant T cells (CNV + TCR + state) | Fig 3a, ED Fig 2 | infercnvpy (paper method) |
| 4a | DEGs malignant vs benign T cells | Fig 3b | scanpy Wilcoxon + edgeR-style (paper method) |
| 4b | **Same comparison via MrVI** | Fig 3b (extension) | `MRVI.differential_expression` |
| 5 | NMF metaprograms across malignant cells | Fig 3c, 3d | NMF + Jaccard (paper method) |
| 6a | Early vs advanced CTCL DEGs and TH2 skewing | Fig 4a, 4b | Wilcoxon (paper method) — *p = 3 × 10⁻⁴* expected |
| 6b | **TH2 skewing via MrVI sample-level DE** | Fig 4a/4b (extension) | `MRVI.differential_expression(sample_cov_keys=["stage"])` |
| 7a | F2 fibroblast and DC2/moDC_3 enrichment | Fig 5b, 6b | Milo (paper method) |
| 7b | MHC-II upregulation in F2 / LGALS9 in DC2 | Fig 5c, 6c | Heatmap (paper method) |
| 7c | **CTCL-vs-healthy DE in stroma+APC via MrVI** | Fig 5c, 6c (extension) | MrVI on stroma+APC subset |
| 8 | B cell enrichment in CTCL vs others (per-donor Wilcoxon) | Fig 7a | Wilcoxon (paper method) |
| 9 | Ligand-receptor: malignant T → B / F2 / DC2 niche | Fig 5f, 6f, 8a | LIANA / CellPhoneDB (paper method) |
| 10 | Visium cell2location on one representative slide | Fig 2c, 5d, 6d, 7e | cell2location (paper method) |
| 11 | drug2cell: rituximab/obinutuzumab on B cells | Fig 8d | drug2cell (paper method) |
| 12 | Summary panel: assemble headline figure | Figs 1, 4b, 7a, 8a | matplotlib |

---

## 1. Detailed cell-by-cell build instructions

### Cell 0 — setup

```python
# %% Cell 0: imports & params
import os, sys, gc, json, warnings, pathlib
import numpy as np, pandas as pd, scanpy as sc, anndata as ad
import matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import mannwhitneyu

sc.settings.set_figure_params(dpi=110, facecolor="white", frameon=False)
sc.settings.figdir = "results/figures/"
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT  = pathlib.Path(os.environ.get("PROJECT", "."))
ATLAS    = PROJECT / "processed/ctcl_atlas.h5ad"
VISIUM   = PROJECT / "processed/ctcl_visium.h5ad"
FIGDIR   = PROJECT / "results/figures"; FIGDIR.mkdir(parents=True, exist_ok=True)
TABDIR   = PROJECT / "results/tables";  TABDIR.mkdir(parents=True, exist_ok=True)
RNG = 0
```

Document any column-name remapping needed if the downloaded atlas uses different obs names (e.g. `Patient` vs `donor`). Put the remap as a dict at the top of the notebook so all later cells reference standard names: `donor`, `disease`, `stage`, `cell_type`, `compartment`, `study`.

### Cell 1 — load the atlas and reproduce the broad UMAP

The paper's pipeline: ambient-RNA removal → doublet removal → QC → integration (likely scVI) → CellTypist label transfer from the Reynolds 2021 skin cell atlas. The published object should already carry these.

```python
# %% Cell 1: load and verify
adata = sc.read_h5ad(ATLAS)
print(adata)

# Sanity-check 11 broad cell types (Fig 1b)
assert "cell_type" in adata.obs.columns or "annotation" in adata.obs.columns
ct = "cell_type" if "cell_type" in adata.obs else "annotation"
print(adata.obs[ct].value_counts().head(20))

# If an integrated embedding is present, use it; otherwise compute one quickly
emb = next((k for k in adata.obsm if any(s in k.lower() for s in
            ["scvi", "harmony", "integrated", "mnn"])), None)
if emb is None:
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=4000,
                                batch_key="donor")
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=50, random_state=RNG)
    sc.external.pp.harmony_integrate(adata, key="donor")
    emb = "X_pca_harmony"

sc.pp.neighbors(adata, use_rep=emb, n_neighbors=30, random_state=RNG)
sc.tl.umap(adata, min_dist=0.3, random_state=RNG)

# Reproduce Fig 1b (broad cell types) and Fig 1c (disease overlay)
sc.pl.umap(adata, color=[ct], legend_loc="on data", save="_Fig1b_broad_celltypes.png")
sc.pl.umap(adata, color=["disease"], save="_Fig1c_disease.png")
```

**Expected** ~11 islands corresponding to T cells, B cells, plasma cells, APCs (DC + macrophage), mast cells, fibroblasts, vascular endothelial, lymphatic endothelial, melanocytes, keratinocytes, Schwann cells. CTCL donors should cluster patient-privately in the T-cell island (will be obvious in Cell 3).

### Cell 2 — Milo differential abundance (Fig 1e)

The paper's headline cellular-composition finding is that **B cells, plasma cells, and mast cells are enriched in CTCL** versus healthy / AD / psoriasis, demonstrated via Milo k-NN-graph differential abundance at FDR < 10 %.

```python
# %% Cell 2: Milo DA (Fig 1e)
import milopy
import milopy.core as milo

# Collapse early/advanced into "CTCL" for the broad comparison
adata.obs["disease_simple"] = adata.obs["disease"].astype(str).replace(
    {"CTCL_early": "CTCL", "CTCL_advanced": "CTCL"}
)
keep = adata.obs["disease_simple"].isin(["CTCL", "healthy", "AD", "Psoriasis"])
ad_milo = adata[keep].copy()

milo.make_nhoods(ad_milo, prop=0.1)
milo.count_nhoods(ad_milo, sample_col="donor")

# Two-group: CTCL vs each other condition (one test for the figure panel)
ad_milo.obs["test_group"] = (ad_milo.obs["disease_simple"] == "CTCL").astype(int)
milo.DA_nhoods(ad_milo, design="~test_group")

milopy.utils.annotate_nhoods(ad_milo, anno_col=ct)
da = ad_milo.uns["nhood_adata"].obs
da.to_csv(TABDIR / "table_milo_DA_CTCL_vs_others.csv")
milopy.plot.plot_DA_beeswarm(ad_milo, group_by=ct, alpha=0.1)
plt.savefig(FIGDIR / "Fig1e_milo_DA_beeswarm.png", bbox_inches="tight", dpi=200)
plt.close()
```

**Expected** B cells, plasma cells, and mast cells emerge with logFC > 0 at FDR < 10 % in CTCL. T cells should be modestly enriched. Save the resulting table because Section 8 cross-references it.

### Cell 3 — identify malignant T cells (Fig 3a, ED Fig 2)

The paper's malignant-cell criterion is the field standard combination: (i) inferCNV recovers chromosomal CNVs in malignant clones (validated against WGS), (ii) dominant TCR clonotype matches, (iii) transcriptomic state clusters per-patient (in contrast to benign T cells which cluster by phenotype across patients).

```python
# %% Cell 3: malignant T cells via inferCNV
import infercnvpy as cnv

ad_t = adata[adata.obs[ct].astype(str).str.contains(
    "T cell|T_cell|^T |^Tc|^Th|Treg|NK|ILC", case=False, regex=True)].copy()

# Attach gene positions (Ensembl GRCh38 v110 matches what the paper used)
cnv.io.genomic_position_from_gtf(
    "https://ftp.ensembl.org/pub/release-110/gtf/homo_sapiens/"
    "Homo_sapiens.GRCh38.110.basic.annotation.gtf.gz",
    adata=ad_t,
)

# Reference = T cells from healthy skin (or AD/psoriasis if no healthy donors)
ref_cats = [d for d in ad_t.obs["disease"].unique()
            if "healthy" in str(d).lower() or "normal" in str(d).lower()]
cnv.tl.infercnv(ad_t, reference_key="disease", reference_cat=ref_cats, window_size=250)
cnv.tl.cnv_score(ad_t)

# Threshold per-donor on the upper Gaussian mode of cnv_score
from sklearn.mixture import GaussianMixture
def per_donor_threshold(scores):
    if len(scores) < 100: return np.percentile(scores, 90)
    gm = GaussianMixture(n_components=2, random_state=0).fit(scores.reshape(-1,1))
    return float(np.mean(gm.means_))

is_mal = np.zeros(ad_t.n_obs, dtype=bool)
for d, idx in ad_t.obs.groupby("donor").indices.items():
    thr = per_donor_threshold(ad_t.obs["cnv_score"].iloc[idx].values)
    is_mal[idx] = ad_t.obs["cnv_score"].iloc[idx].values > thr
ad_t.obs["is_malignant"] = is_mal

# Reproduce Fig 3a: T-cell UMAP coloured by donor (malignant cluster per-patient)
sc.pp.neighbors(ad_t, use_rep=emb, n_neighbors=30)
sc.tl.umap(ad_t, min_dist=0.3, random_state=RNG)
sc.pl.umap(ad_t, color=["donor","is_malignant","cnv_score"], wspace=0.4,
           save="_Fig3a_Tcells_donor_malignant.png")

ad_t.write_h5ad(PROJECT / "processed/ctcl_tcells_cnv.h5ad")
```

**Expected** benign T cells form shared clusters across donors; malignant T cells form donor-private islands with high `cnv_score`. The `is_malignant` boolean will be the key grouping variable for everything downstream.

### Cell 4a — DEGs malignant vs benign T cells, paper method (Fig 3b)

The paper uses a quasi-likelihood F-test (edgeR-style) for DEGs and highlights TOX upregulation, CD7 downregulation, and high CXCL13. We use scanpy's Wilcoxon as the practical surrogate (the rankings are nearly identical for this comparison; both are reported in the paper's text).

```python
# %% Cell 4a: DEGs malignant vs benign (paper method)
ad_t.obs["mvb"] = np.where(ad_t.obs["is_malignant"], "malignant", "benign")
sc.tl.rank_genes_groups(ad_t, groupby="mvb", reference="benign", method="wilcoxon",
                        n_genes=200, pts=True)
df = sc.get.rank_genes_groups_df(ad_t, group="malignant")
df.to_csv(TABDIR / "Fig3b_DEGs_malignant_vs_benign_wilcoxon.csv", index=False)

# Volcano (paper Fig 3b)
df["nlog10p"] = -np.log10(df["pvals_adj"] + 1e-300)
key_genes = ["TOX","CD7","CXCL13","CD9","CCR4","GTSF1","RUNX3","GATA3","TBX21"]
fig, ax = plt.subplots(figsize=(5,4))
ax.scatter(df["logfoldchanges"], df["nlog10p"], s=2, alpha=0.4, color="grey")
hl = df[df["names"].isin(key_genes)]
ax.scatter(hl["logfoldchanges"], hl["nlog10p"], s=30, color="red")
for _, r in hl.iterrows():
    ax.text(r["logfoldchanges"], r["nlog10p"], r["names"], fontsize=8)
ax.set_xlabel("log2 FC malignant vs benign"); ax.set_ylabel("-log10 p_adj")
plt.savefig(FIGDIR / "Fig3b_volcano_paper_method.png", dpi=200, bbox_inches="tight")
plt.close()
```

**Expected** TOX, CXCL13, CD9, GTSF1 in top upregulated. CD7 strongly downregulated. CCR4 enriched in advanced subset (will see again in Cell 6a). 25 genes give "good discriminatory power" per the paper.

### Cell 4b — same comparison via MrVI

MrVI does sample-level rather than cell-level comparisons: every patient is a sample, the malignant_status acts as a sample-level *target covariate*, and `batch` (`study` of origin in our integrated dataset) is the *nuisance covariate*. This yields per-cell effect sizes corrected for inter-patient variability — something the Wilcoxon can't give.

For the malignant-vs-benign comparison, malignant_status is not actually a sample-level property (it varies *within* each CTCL donor — every CTCL donor has both populations). So we handle this by *creating pseudo-samples*: within each donor, split into `<donor>_malignant` and `<donor>_benign`. This is the recommended MrVI pattern for intra-donor contrasts.

```python
# %% Cell 4b: same DE via MrVI (sample-level)
import scvi
from scvi.external import MRVI

ad_t.obs["sample_for_mrvi"] = (
    ad_t.obs["donor"].astype(str) + "_" + ad_t.obs["mvb"].astype(str)
)
ad_t.obs["target_status"] = ad_t.obs["mvb"]      # malignant vs benign

# MrVI requires raw integer counts; use .layers["counts"] if present, else .raw.X
if "counts" in ad_t.layers:
    X = ad_t.copy(); X.X = X.layers["counts"]
elif ad_t.raw is not None:
    X = ad_t.raw.to_adata(); X.obs = ad_t.obs.copy()
else:
    X = ad_t.copy()  # may fail if X is normalised; warn

MRVI.setup_anndata(
    X,
    sample_key="sample_for_mrvi",     # per-(donor × malignant_status) pseudo-sample
    batch_key="study",                # nuisance: source study (Liu, Rindler, Song, new)
    labels_key=ct,                    # optional cell-type guidance
)
mrvi = MRVI(X)
mrvi.train(max_epochs=50, batch_size=256, accelerator="auto")
mrvi.save(PROJECT / "models/mrvi_tcells", overwrite=True)

# Get u / z latents and per-cell DE for target_status
X.obsm["X_mrvi_u"] = mrvi.get_latent_representation(give_z=False)
X.obsm["X_mrvi_z"] = mrvi.get_latent_representation(give_z=True)

de = mrvi.differential_expression(
    sample_cov_keys=["target_status"],
    store_lfc=True,
    delta=0.3,
    mc_samples=50,
)
# `de` is an xarray-like dataset; turn into a DataFrame
lfc = de.lfc.sel(covariate="target_status").to_pandas()  # cells × genes
pde = de.pde.sel(covariate="target_status").to_pandas()  # posterior DE probability
# Aggregate over malignant cells only
mal_mask = X.obs["mvb"] == "malignant"
lfc_mean = lfc.loc[mal_mask].mean(axis=0).sort_values(ascending=False)
pde_mean = pde.loc[mal_mask].mean(axis=0)
mrvi_df = pd.DataFrame({"lfc_mrvi": lfc_mean, "pde_mrvi": pde_mean})
mrvi_df.to_csv(TABDIR / "Fig3b_DEGs_malignant_vs_benign_MRVI.csv")

# Side-by-side rank plot vs Wilcoxon
wil = pd.read_csv(TABDIR / "Fig3b_DEGs_malignant_vs_benign_wilcoxon.csv").set_index("names")
joined = mrvi_df.join(wil[["logfoldchanges","pvals_adj"]], how="inner").dropna()
fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(joined["logfoldchanges"], joined["lfc_mrvi"], s=4, alpha=0.4)
for g in key_genes:
    if g in joined.index:
        ax.scatter(joined.loc[g,"logfoldchanges"], joined.loc[g,"lfc_mrvi"], color="red")
        ax.text(joined.loc[g,"logfoldchanges"], joined.loc[g,"lfc_mrvi"], g, fontsize=8)
ax.axhline(0, c="grey", lw=0.5); ax.axvline(0, c="grey", lw=0.5)
ax.set_xlabel("log2FC Wilcoxon (paper method)"); ax.set_ylabel("log2FC MrVI")
plt.savefig(FIGDIR / "Fig3b_concordance_Wilcoxon_vs_MRVI.png", dpi=200, bbox_inches="tight")
plt.close()
```

**Expected** Spearman ρ ≥ 0.7 between Wilcoxon and MrVI log-FC for high-confidence genes. MrVI typically gives smaller |LFC| (better donor-effect correction) but the same direction. Where they disagree is *the interesting science*: genes whose LFC is large by Wilcoxon but small by MrVI are likely confounded by patient identity.

### Cell 5 — intratumor NMF metaprograms (Fig 3c, 3d)

The paper runs NMF per patient on the malignant cells, gets ~243 programs (≈ 5–6 per patient), then clusters them by Jaccard overlap into **9 cross-patient metaprograms**: stress (JUN, HSPA1A), G1/S cell cycle (TOP2A), G2/M cell cycle (TYMS), two T-cell activation programs (NR4A2, TNFRSF9, TNFRSF4), glycolysis/hypoxia (ENO1, GPI), and others.

```python
# %% Cell 5: NMF metaprograms (Tirosh-style; Fig 3c, 3d)
from sklearn.decomposition import NMF
import scipy.sparse as sp

mal_t = ad_t[ad_t.obs["is_malignant"]].copy()
sc.pp.normalize_total(mal_t, target_sum=1e4)
sc.pp.log1p(mal_t)

programs_per_donor = {}
for d, sub in mal_t.obs.groupby("donor"):
    if len(sub) < 200: continue
    Xd = mal_t[sub.index].X.toarray() if sp.issparse(mal_t.X) else mal_t[sub.index].X
    Xd = np.clip(Xd, 0, None)
    nmf = NMF(n_components=6, init="nndsvda", max_iter=400, random_state=0).fit(Xd)
    top = pd.DataFrame(nmf.components_, columns=mal_t.var_names)
    programs_per_donor[d] = [
        top.iloc[k].sort_values(ascending=False).head(50).index.tolist()
        for k in range(6)
    ]

# Flatten to (program_id, top50_genes)
flat = []
for d, progs in programs_per_donor.items():
    for k, g in enumerate(progs):
        flat.append((f"{d}_p{k}", set(g)))
ids, gene_sets = zip(*flat)

# Pairwise Jaccard, hierarchical clustering, cut at 9 clusters
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
def jacc(a,b): return len(a&b)/max(1,len(a|b))
J = np.array([[jacc(a,b) for b in gene_sets] for a in gene_sets])
Z = linkage(squareform(1-J, checks=False), method="average")
mp = fcluster(Z, t=9, criterion="maxclust")

# Metaprogram → consensus gene list (genes appearing in ≥30% of member programs)
mp_consensus = {}
for c in np.unique(mp):
    members = [gene_sets[i] for i in range(len(gene_sets)) if mp[i]==c]
    counts = pd.Series([g for s in members for g in s]).value_counts()
    mp_consensus[c] = counts[counts >= 0.3*len(members)].index.tolist()
pd.Series(mp_consensus).to_json(TABDIR / "Fig3d_metaprograms_consensus.json", indent=2)

# Plot Jaccard heatmap (Fig 3d)
sns.clustermap(J, row_linkage=Z, col_linkage=Z, cmap="viridis", figsize=(8,8))
plt.savefig(FIGDIR / "Fig3d_metaprogram_jaccard.png", dpi=200, bbox_inches="tight")
plt.close()
```

**Expected** 9 metaprograms recognizable by their seed genes: stress (JUN/HSPA1A/FOS), G1S (TOP2A/MCM5/TYMS), G2M (CDK1/UBE2C), T-activation × 2 (NR4A2/TNFRSF9; CD69/IL2), glycolysis/hypoxia (ENO1/GPI/LDHA), TCR-loss program.

### Cell 6a — early vs advanced DEGs and TH2 skewing, paper method (Fig 4a, 4b)

This is **the central clinical finding**: in advanced disease, malignant T cells switch from a cytotoxic/tissue-resident-effector phenotype (high IFNG/GZMB/GNLY) to a central-memory phenotype (high SELL/CCR7/LEF1/TCF7) with **GATA3-driven TH2 skewing** (p = 0.0003, Wilcoxon).

```python
# %% Cell 6a: early vs advanced (paper method; Fig 4a, 4b)
mal = ad_t[ad_t.obs["is_malignant"]].copy()
mal.obs["stage_simple"] = np.where(
    mal.obs["stage"].astype(str).str.contains("IIB|III|IV|advanced", case=False),
    "advanced", "early"
)

sc.tl.rank_genes_groups(mal, groupby="stage_simple", reference="early",
                        method="wilcoxon", n_genes=100, pts=True)
deg_adv = sc.get.rank_genes_groups_df(mal, group="advanced")
deg_adv.to_csv(TABDIR / "Fig4a_DEGs_advanced_vs_early.csv", index=False)

# Fig 4a heatmap
adv_markers = ["IFNG","GZMB","GNLY","NKG7","SELL","CCR7","LEF1","TCF7","S1PR1",
               "GATA3","IL4","IL13","IL5","TBX21","RORC"]
sc.pl.heatmap(mal, var_names=adv_markers, groupby="stage_simple",
              standard_scale="var", swap_axes=True,
              save="_Fig4a_early_vs_advanced.png")

# Fig 4b: TH classifier on TBX21 / GATA3 / RORC + IL4/IL13
sc.tl.score_genes(mal, ["TBX21","IFNG"],           score_name="Th1_score")
sc.tl.score_genes(mal, ["GATA3","IL4","IL13","IL5","STAT6"], score_name="Th2_score")
sc.tl.score_genes(mal, ["RORC","IL17A","IL17F","IL22"],      score_name="Th17_score")
mal.obs["Th_subtype"] = mal.obs[["Th1_score","Th2_score","Th17_score"]].idxmax(axis=1).str.replace("_score","")

per_donor = (mal.obs.groupby(["donor","stage_simple"])["Th_subtype"]
             .value_counts(normalize=True).unstack(fill_value=0).reset_index())
per_donor.to_csv(TABDIR / "Fig4b_per_donor_TH_fractions.csv", index=False)

# Wilcoxon test exactly as in the paper
early = per_donor.loc[per_donor.stage_simple=="early", "Th2"]
adv   = per_donor.loc[per_donor.stage_simple=="advanced", "Th2"]
U, p  = mannwhitneyu(early, adv, alternative="less")
print(f"TH2 fraction Wilcoxon U={U:.0f}  one-sided p={p:.2e}  (paper reports p=0.0003)")

fig, ax = plt.subplots(figsize=(4,4))
sns.violinplot(data=per_donor, x="stage_simple", y="Th2", inner="box", ax=ax)
sns.stripplot(data=per_donor, x="stage_simple", y="Th2", color="k", size=4, ax=ax)
ax.set_title(f"TH2 fraction in malignant T cells\np = {p:.2e}")
plt.savefig(FIGDIR / "Fig4b_TH2_violin.png", dpi=200, bbox_inches="tight")
plt.close()
```

**Expected** the violin replicates the paper's Fig 4b shape — early disease median TH2 fraction ~0.1–0.2, advanced median ~0.5–0.7, one-sided p ≈ 10⁻³.

### Cell 6b — TH2 skewing via MrVI

This is the *natural* MrVI use case for this paper: stage is genuinely a sample (donor) level covariate, so MrVI directly models how each donor's malignant T cells shift in latent space as a function of stage.

```python
# %% Cell 6b: TH2 skewing via MrVI
from scvi.external import MRVI

# Use only the malignant cells (intra-tumor heterogeneity is the signal here)
mal_raw = mal.copy()
if "counts" in mal_raw.layers:
    mal_raw.X = mal_raw.layers["counts"]
elif mal_raw.raw is not None:
    mal_raw = mal_raw.raw.to_adata(); mal_raw.obs = mal.obs.copy()

MRVI.setup_anndata(
    mal_raw,
    sample_key="donor",
    batch_key="study",
    categorical_nuisance_keys=["compartment"] if "compartment" in mal_raw.obs else None,
)
mrvi_m = MRVI(mal_raw)
mrvi_m.train(max_epochs=100, batch_size=256, accelerator="auto")
mrvi_m.save(PROJECT / "models/mrvi_malignant", overwrite=True)

# Sample-level distance matrix — donors should cluster by stage
dist = mrvi_m.get_local_sample_distances(keep_cell=False, groupby=None)
sns.clustermap(dist.to_pandas(), cmap="RdBu_r", center=0)
plt.savefig(FIGDIR / "Fig4_donor_distances_MRVI.png", dpi=200, bbox_inches="tight")
plt.close()

# Stage-DE: every cell gets a per-gene effect-size + posterior DE probability
de_stage = mrvi_m.differential_expression(
    sample_cov_keys=["stage_simple"],
    store_lfc=True,
    delta=0.3,
    mc_samples=100,
)
lfc_stage = de_stage.lfc.sel(covariate="stage_simple").to_pandas()    # cells × genes
pde_stage = de_stage.pde.sel(covariate="stage_simple").to_pandas()

# Average over malignant cells per stage
top_up_adv = lfc_stage.mean(0).sort_values(ascending=False).head(30)
top_dn_adv = lfc_stage.mean(0).sort_values(ascending=True).head(30)
top_up_adv.to_csv(TABDIR / "Fig4_MRVI_top_up_advanced.csv")
top_dn_adv.to_csv(TABDIR / "Fig4_MRVI_top_down_advanced.csv")

# Plot per-cell LFC for GATA3 on the malignant-cell UMAP
sc.pp.neighbors(mal_raw, use_rep=mrvi_m.get_latent_representation())
sc.tl.umap(mal_raw, random_state=RNG)
mal_raw.obs["lfc_GATA3_MRVI"] = lfc_stage["GATA3"].values
mal_raw.obs["lfc_IFNG_MRVI"]  = lfc_stage["IFNG"].values
sc.pl.umap(mal_raw, color=["stage_simple","lfc_GATA3_MRVI","lfc_IFNG_MRVI",
                          "Th_subtype"],
           ncols=2, save="_Fig4_MRVI_stage_DE.png", color_map="RdBu_r")
```

**Expected** GATA3, IL4, IL13, SELL, CCR7, LEF1, TCF7 in the MrVI top-up-in-advanced list; IFNG, GZMB, GNLY, NKG7 in the top-down. The donor distance heatmap should show advanced donors clustering separately from early. Spearman correlation between MrVI mean LFC and Wilcoxon LFC across all genes is the headline reproducibility metric — record it.

### Cell 7a — F2 fibroblast and DC2/moDC_3 enrichment (Fig 5b, 6b)

```python
# %% Cell 7a: stroma + APC Milo DA
import milopy.core as milo

mask_stroma_apc = adata.obs[ct].astype(str).str.contains(
    "fibrob|endothel|DC|Langerhans|macrop|monocyt|kerat|melan", case=False, regex=True)
ad_sa = adata[mask_stroma_apc & adata.obs["disease_simple"].isin(
    ["CTCL","healthy","AD","Psoriasis"])].copy()

milo.make_nhoods(ad_sa, prop=0.15)
milo.count_nhoods(ad_sa, sample_col="donor")
ad_sa.obs["is_CTCL"] = (ad_sa.obs["disease_simple"]=="CTCL").astype(int)
milo.DA_nhoods(ad_sa, design="~is_CTCL")
milopy.utils.annotate_nhoods(ad_sa, anno_col=ct)

import milopy.plot as miplot
miplot.plot_DA_beeswarm(ad_sa, group_by=ct, alpha=0.1)
plt.savefig(FIGDIR / "Fig5b_6b_stroma_apc_milo.png", dpi=200, bbox_inches="tight")
plt.close()
ad_sa.uns["nhood_adata"].obs.to_csv(TABDIR / "Fig5b_6b_milo_DA.csv")
```

**Expected** F2 fibroblast (CXCL9⁺ MHC-II⁺), VE3 vascular endothelial, DC2, and moDC_3 all rise as the CTCL-enriched subsets at FDR < 10 %. F1/F3 fibroblasts and DC1 do not.

### Cell 7b — MHC-II up in F2, LGALS9 up in DC2 (Fig 5c, 6c)

```python
# %% Cell 7b: heatmaps
f2_markers  = ["CD74","HLA-DRA","HLA-DRB1","HLA-DRB5","HLA-DPA1","CXCL9","CXCL10",
               "CCL5","IL33","CD80","CD86"]  # last two should be absent
dc2_markers = ["LGALS9","TNFSF12","CD58","CD2","HAVCR2","IL23A","IL18","CD70"]

# F2 vs other fibroblasts
fib = ad_sa[ad_sa.obs[ct].astype(str).str.contains("fibrob", case=False)].copy()
sc.pl.heatmap(fib, var_names=f2_markers, groupby=ct, standard_scale="var",
              swap_axes=True, save="_Fig5c_F2_markers.png")

# DC2 vs other DCs
dc = ad_sa[ad_sa.obs[ct].astype(str).str.contains("DC|Langer|moDC", case=False, regex=True)].copy()
sc.pl.heatmap(dc, var_names=dc2_markers, groupby=ct, standard_scale="var",
              swap_axes=True, save="_Fig6c_DC2_markers.png")
```

### Cell 7c — CTCL-vs-healthy DE in stroma+APC, MrVI version

For this comparison `disease_simple` is genuinely a donor-level property, so MrVI is well-posed without pseudo-samples.

```python
# %% Cell 7c: stroma+APC MrVI DE
import scvi
from scvi.external import MRVI

ad_sa_raw = ad_sa.copy()
if "counts" in ad_sa_raw.layers:
    ad_sa_raw.X = ad_sa_raw.layers["counts"]
elif ad_sa_raw.raw is not None:
    ad_sa_raw = ad_sa_raw.raw.to_adata(); ad_sa_raw.obs = ad_sa.obs.copy()

MRVI.setup_anndata(
    ad_sa_raw,
    sample_key="donor",
    batch_key="study",
)
mrvi_sa = MRVI(ad_sa_raw)
mrvi_sa.train(max_epochs=60, batch_size=256, accelerator="auto")

de_disease = mrvi_sa.differential_expression(
    sample_cov_keys=["disease_simple"],
    store_lfc=True,
    delta=0.3,
    mc_samples=50,
)
lfc_dis = de_disease.lfc.sel(covariate="disease_simple").to_pandas()

# F2 subset only — per-cell average LFC for HLA-DR / CXCL9 / IL33
f2_cells = ad_sa_raw.obs[ct].astype(str).str.contains("F2|MHC-II", case=False, regex=True)
for g in ["HLA-DRA","CD74","CXCL9","CXCL10","IL33"]:
    if g in lfc_dis.columns:
        print(f"{g}: F2 cells mean LFC (CTCL vs other) = {lfc_dis.loc[f2_cells, g].mean():.3f}")

lfc_dis.loc[f2_cells].mean().sort_values(ascending=False).head(30).to_csv(
    TABDIR / "Fig5c_MRVI_F2_topup_CTCL_vs_others.csv")
```

**Expected** MHC-II module (CD74, HLA-DRA/DRB/DPA1), CXCL9/10, IL33 all positive LFC in F2 cells under CTCL. Spearman vs paper Wilcoxon DEGs (Supplementary Table 6) should be ≥ 0.6.

### Cell 8 — B cell enrichment per-donor (Fig 7a)

```python
# %% Cell 8: B cell enrichment Wilcoxon (Fig 7a)
is_B = adata.obs[ct].astype(str).str.contains("B cell|B_cell|plasma", case=False, regex=True)
adata.obs["is_B"] = is_B
b_frac = (adata.obs.groupby(["donor","disease_simple"])["is_B"]
          .mean().reset_index().rename(columns={"is_B":"b_fraction"}))
b_frac.to_csv(TABDIR / "Fig7a_per_donor_B_fractions.csv", index=False)

# Wilcoxon CTCL vs each reference group (paper does this individually)
ctcl = b_frac.loc[b_frac.disease_simple=="CTCL","b_fraction"]
print("CTCL median B fraction:", ctcl.median())
for ref in ["healthy","AD","Psoriasis"]:
    other = b_frac.loc[b_frac.disease_simple==ref,"b_fraction"]
    if len(other):
        U, p = mannwhitneyu(ctcl, other, alternative="greater")
        print(f"CTCL > {ref}: U={U:.0f} p={p:.2e}  (paper reports p<1e-4)")

fig, ax = plt.subplots(figsize=(5,4))
sns.boxplot(data=b_frac, x="disease_simple", y="b_fraction",
            order=["healthy","AD","Psoriasis","CTCL"], ax=ax)
sns.stripplot(data=b_frac, x="disease_simple", y="b_fraction",
              order=["healthy","AD","Psoriasis","CTCL"], color="k", size=3, ax=ax)
plt.savefig(FIGDIR / "Fig7a_B_fractions_boxplot.png", dpi=200, bbox_inches="tight")
plt.close()
```

**Expected** every comparison `p < 10⁻⁴` (CTCL ≫ healthy ≈ AD ≈ psoriasis). Mean CTCL B fraction at scRNA-seq resolution is small (~1–3 %) but consistently nonzero, unlike controls.

### Cell 9 — ligand–receptor: malignant T → B / F2 / DC2 (Fig 5f, 6f, 8a)

```python
# %% Cell 9: LR analysis (LIANA, paper used CellPhoneDB)
import liana as li

adata.obs["compartment_for_lr"] = adata.obs[ct].astype(str)
adata.obs.loc[adata.obs["is_malignant"] == True, "compartment_for_lr"] = "Malignant_T"

ad_ctcl = adata[adata.obs["disease_simple"]=="CTCL"].copy()
li.mt.rank_aggregate(
    ad_ctcl,
    groupby="compartment_for_lr",
    resource_name="consensus",
    expr_prop=0.1,
    use_raw=False,
    verbose=True,
)
lr = ad_ctcl.uns["liana_res"]
lr.to_csv(TABDIR / "Fig8a_LR_consensus.csv", index=False)

# Headline pairs to validate (paper Fig 8a)
must_have = [
    ("Malignant_T", "B cell", "CXCL13", "CXCR5"),
    ("Malignant_T", "B cell", "CD40LG", "CD40"),
    ("Malignant_T", "B cell", "CD28",   "CD86"),
    ("Malignant_T", "F2",     "CD40",   "CD40LG"),
    ("Malignant_T", "DC2",    "CD2",    "CD58"),
]
hits = lr[(lr["source"]=="Malignant_T") & lr["target"].isin(["B cell","F2","DC2","moDC_3"])]
hits_top = hits.sort_values("magnitude_rank").head(40)
hits_top.to_csv(TABDIR / "Fig8a_LR_malignantT_to_niche_top.csv", index=False)

# Dotplot per Fig 8a
li.pl.dotplot(
    ad_ctcl, colour="magnitude_rank", size="specificity_rank",
    source_labels=["Malignant_T"],
    target_labels=["B cell","F2","DC2","moDC_3"],
    top_n=20,
)
plt.savefig(FIGDIR / "Fig8a_LR_dotplot.png", dpi=200, bbox_inches="tight")
plt.close()
```

**Expected** every must-have pair appears in the top-40 with magnitude rank in the top quartile.

### Cell 10 — Visium spatial: cell2location on one representative slide (Fig 2c, 5d, 6d, 7e)

Use a single slide first to verify the pipeline; only loop over all 8 CTCL slides if time and GPU permit. The headline visual is the spatial co-occurrence of malignant T cells with F2 / DC2 / B cells (microenvironment 5).

```python
# %% Cell 10: cell2location on a representative CTCL section
import cell2location as c2l

vis = sc.read_h5ad(VISIUM)
print(vis.obs["library_id"].value_counts() if "library_id" in vis.obs else vis)

ref = sc.read_h5ad(ATLAS)  # use raw counts
if "counts" in ref.layers: ref.X = ref.layers["counts"]
elif ref.raw is not None:   ref = ref.raw.to_adata()

# Reference signatures
c2l.models.RegressionModel.setup_anndata(ref, batch_key="donor", labels_key=ct)
reg = c2l.models.RegressionModel(ref)
reg.train(max_epochs=250, batch_size=2500, lr=0.002, accelerator="auto")
ref = reg.export_posterior(ref, sample_kwargs={"num_samples":1000,"batch_size":2500})
sig = ref.varm["means_per_cluster_mu_fg"].copy()

# Pick one CTCL section
slide = vis.obs["library_id"].value_counts().index[0]   # or named e.g. "CTCL_01"
vis_one = vis[vis.obs["library_id"]==slide].copy()

c2l.models.Cell2location.setup_anndata(vis_one, batch_key="library_id")
c2l_mod = c2l.models.Cell2location(vis_one, cell_state_df=sig, N_cells_per_location=8)
c2l_mod.train(max_epochs=20000, batch_size=None, train_size=1)
vis_one = c2l_mod.export_posterior(vis_one)

# Plot the niche: F2 + DC2 + B cell + Malignant_T co-occurrence
keys = [c for c in vis_one.obs.columns if c.startswith("q05_")]
selected = [c for c in keys if any(s in c for s in
            ["F2","DC2","moDC_3","B cell","Malignant_T","Tumor"])]
sc.pl.spatial(vis_one, color=selected, ncols=3,
              save="_Fig2c_5d_6d_7e_niche_spatial.png")
```

**Expected** the four CTCL-niche populations co-localize in the dermis around blood vessels; B cells should appear as compact aggregates rather than diffuse. Run NMF across all slides' cell2location abundances to recover the paper's 5 microenvironments if time permits.

### Cell 11 — drug2cell on B cells (Fig 8d)

```python
# %% Cell 11: drug2cell B cell targets
import drug2cell as d2c
d2c.score(adata, use_raw=False)
d2c.tl.celltype(adata, groupby=ct)
adata.uns["drug2cell"]["celltype_means"].loc["B cell"].sort_values(ascending=False).head(20).to_csv(
    TABDIR / "Fig8d_drug2cell_B_top20.csv")
```

**Expected** rituximab and obinutuzumab top the list for B cells; 15-PGDH inhibitors high on malignant T cells.

### Cell 12 — assemble a headline summary figure

A single multi-panel PDF/PNG combining: (A) UMAP coloured by cell type, (B) Milo DA beeswarm with B cells highlighted, (C) Fig 4b TH2 violin, (D) MrVI per-cell LFC for GATA3, (E) B cell fraction box plot, (F) LR dotplot.

```python
# %% Cell 12: composite summary
fig, axes = plt.subplots(2, 3, figsize=(18,11))
# ... embed each saved PNG via matplotlib.image.imread or recompute
plt.tight_layout()
plt.savefig(FIGDIR / "summary_replication.pdf", bbox_inches="tight")
```

---

## 2. MrVI sanity tests the agent must include

Because MrVI is the newer tool here, the notebook should include explicit validation:

1. **Latent UMAP**: `sc.pl.umap(adata, color=["disease","stage","donor"])` on `X_mrvi_u`. The integration should put cells of the same type from different donors next to each other while preserving disease/stage as a gradient on `X_mrvi_z` − `X_mrvi_u`.
2. **Donor-distance heatmap**: `MRVI.get_local_sample_distances(keep_cell=False)`. CTCL donors should cluster together; healthy/AD/Pso form a separate block; early vs advanced CTCL should subcluster.
3. **Concordance scatter**: for every section that pairs paper-method DE with MrVI DE (Cells 4b, 6b, 7c), plot Wilcoxon LFC vs MrVI LFC. Spearman ρ ≥ 0.5 across all genes; ≥ 0.7 for high-confidence (top 200) DEGs. Genes that are highly discordant are an interesting analysis: they're typically genes confounded by patient identity that MrVI corrects out.
4. **Training-curve check**: plot `mrvi.history["elbo_train"]`. ELBO should plateau before stopping; if still falling, retrain with more epochs.

---

## 3. Validation checklist

Before declaring the notebook done, the agent should be able to tick all of:

- [ ] Total cell count loaded matches the paper (≈ 420 000 ± 5 %).
- [ ] Eleven broad cell types appear on UMAP (Fig 1b).
- [ ] Milo finds B cells, plasma cells, mast cells enriched in CTCL at FDR < 10 % (Fig 1e).
- [ ] inferCNV `cnv_score` is bimodal in CTCL T cells, unimodal in healthy/AD/Pso.
- [ ] DEGs malignant vs benign include TOX up, CD7 down, CXCL13 up (Fig 3b).
- [ ] 9 metaprograms recovered with recognizable seed genes (Fig 3d).
- [ ] **TH2 fraction Wilcoxon test gives p < 0.001 for early < advanced** (Fig 4b — the paper's headline number is p = 3 × 10⁻⁴).
- [ ] MrVI top-up-in-advanced list contains GATA3, IL4, IL13, SELL, CCR7, LEF1, TCF7.
- [ ] MrVI top-down-in-advanced list contains IFNG, GZMB, GNLY.
- [ ] Wilcoxon-vs-MrVI LFC Spearman ≥ 0.6 globally for malignant-vs-benign.
- [ ] Milo on stroma+APC finds F2, VE3, DC2, moDC_3 enriched in CTCL (Fig 5b, 6b).
- [ ] MHC-II module up in F2 fibroblasts; CD80/CD86 absent (Fig 5c).
- [ ] B cell fraction Wilcoxon: CTCL > healthy, > AD, > psoriasis, all p < 10⁻⁴ (Fig 7a).
- [ ] LIANA finds CXCL13–CXCR5, CD40LG–CD40, CD28–CD86 between malignant T and B cells (Fig 8a).
- [ ] cell2location spatial map shows F2 + B + DC2 + Malignant_T co-occurrence (Fig 2c, 5d, 6d, 7e).
- [ ] drug2cell ranks CD20-binding antibodies top on B cells (Fig 8d).

If any of these fail, the agent should write a short diagnosis in `results/analysis_log.md` under the heading "**Findings that did not replicate**" — possible reasons: (a) the downloaded h5ad is a subset, (b) the `cell_type` annotation uses a different taxonomy, (c) integer counts were not preserved in the published object (Milo and MrVI need them), (d) `donor` is named something else, (e) `study` column is missing so the batch covariate has to fall back to `donor`.

---

## 4. Performance and resource notes

| Step | RAM | GPU | Time (A100) | Time (CPU 32-core) |
|---|---|---|---|---|
| Load + UMAP | 16–32 GB | — | 5 min | 5 min |
| Milo DA | 8 GB | — | 5 min | 10 min |
| inferCNV | 16–32 GB | — | 20–40 min | 60–90 min |
| NMF metaprograms | 8 GB | — | 10–20 min | 20–40 min |
| Wilcoxon DE | 8 GB | — | 2 min | 5 min |
| **MrVI T cells** | 16 GB | required | **15–25 min @ 50 epochs** | 90+ min |
| **MrVI malignant only** | 8 GB | required | **10 min @ 100 epochs** | 60 min |
| **MrVI stroma+APC** | 12 GB | required | **15 min @ 60 epochs** | 90 min |
| LIANA | 8 GB | — | 10 min | 20 min |
| cell2location (1 slide) | 16 GB | strongly preferred | 20–40 min | 2–4 h |
| drug2cell | 4 GB | — | 2 min | 5 min |

On CPU-only systems: reduce MrVI `max_epochs` to 20–30 for fast iteration, increase if ELBO is still falling. Skip cell2location to a single slide. Notebook should still complete in < 12 h.

---

## 5. File and naming conventions

```
results/
├── figures/
│   ├── Fig1b_broad_celltypes.png
│   ├── Fig1c_disease.png
│   ├── Fig1e_milo_DA_beeswarm.png
│   ├── Fig3a_Tcells_donor_malignant.png
│   ├── Fig3b_volcano_paper_method.png
│   ├── Fig3b_concordance_Wilcoxon_vs_MRVI.png
│   ├── Fig3d_metaprogram_jaccard.png
│   ├── Fig4a_early_vs_advanced.png
│   ├── Fig4b_TH2_violin.png
│   ├── Fig4_donor_distances_MRVI.png
│   ├── Fig4_MRVI_stage_DE.png
│   ├── Fig5b_6b_stroma_apc_milo.png
│   ├── Fig5c_F2_markers.png
│   ├── Fig6c_DC2_markers.png
│   ├── Fig7a_B_fractions_boxplot.png
│   ├── Fig8a_LR_dotplot.png
│   ├── Fig2c_5d_6d_7e_niche_spatial.png
│   └── summary_replication.pdf
├── tables/
│   ├── table_milo_DA_CTCL_vs_others.csv
│   ├── Fig3b_DEGs_malignant_vs_benign_wilcoxon.csv
│   ├── Fig3b_DEGs_malignant_vs_benign_MRVI.csv
│   ├── Fig3d_metaprograms_consensus.json
│   ├── Fig4a_DEGs_advanced_vs_early.csv
│   ├── Fig4b_per_donor_TH_fractions.csv
│   ├── Fig4_MRVI_top_up_advanced.csv
│   ├── Fig4_MRVI_top_down_advanced.csv
│   ├── Fig5b_6b_milo_DA.csv
│   ├── Fig5c_MRVI_F2_topup_CTCL_vs_others.csv
│   ├── Fig7a_per_donor_B_fractions.csv
│   ├── Fig8a_LR_consensus.csv
│   ├── Fig8a_LR_malignantT_to_niche_top.csv
│   └── Fig8d_drug2cell_B_top20.csv
├── models/
│   ├── mrvi_tcells/
│   ├── mrvi_malignant/
│   └── mrvi_stroma_apc/
└── analysis_log.md
```

The notebook should be commented enough that any cell can be re-run in isolation if its dependencies are still in memory or can be loaded from the saved h5ad/CSVs.

---

## 6. Common pitfalls

| Symptom | Likely cause | Fix |
|---|---|---|
| MrVI `setup_anndata` complains about non-integer X | Atlas X is log-normalised, not counts | Use `adata.layers["counts"]` or `adata.raw.to_adata()` before MrVI |
| MrVI loss = NaN after a few epochs | Too high lr / mixed precision issue | Set `accelerator="cpu"` to confirm; otherwise lower lr to 1e-4 |
| Milo gives every neighborhood `p = 1` | Sample design matrix is singular (e.g. all CTCL from one study) | Add `study` as a covariate or restrict the comparison |
| inferCNV gives flat profiles everywhere | Reference cell type was malignant by mistake | Set `reference_cat` to T cells from healthy donors only |
| `Th_subtype` is 100 % "Th1" | Score genes computed on wrong layer (raw counts) | Use log-normalised `X`; verify by `print(adata.X.max())` < 20 |
| TH2 Wilcoxon not significant | Stage label uses a different schema (e.g. roman numerals) | Map IA/IB/IIA → "early", IIB+/III/IV/IVA1 → "advanced" |
| LIANA misses CXCL13–CXCR5 | Malignant T not in `groupby` column | Re-tag malignant T cells in `compartment_for_lr` before running LIANA |
| Notebook freezes on cell2location | OOM on full dataset | Restrict to one library and ~10000 spots |

---

## 7. What this notebook does *not* attempt

For completeness, the agent should note in `analysis_log.md` that the notebook deliberately skips:

- Bulk RNA-seq B-cell deconvolution and progression-free survival (Fig 7a *n* = 196 bulk cohort, Fig 8b Kaplan-Meier).
- IHC quantification of CD20/CD79a aggregates (Fig 7b–d).
- RareCyte multiplex IF (Fig 2a, 7g).
- The fetal-skin fibroblast comparison (transcriptional similarity of F2 with fetal skin fibroblasts).
- Per-patient WGS cross-validation of inferCNV calls (Extended Data Fig 2a).
- The full integrated atlas re-build from raw FASTQs (use the published h5ad).

These can each be modular follow-ups.

---

## 8. Reference card the agent should keep open

- Paper HTML: <https://www.nature.com/articles/s41590-024-02018-1>
- WebAtlas portal: <https://collections.cellatlas.io/ctcl>
- MrVI tutorial: <https://docs.scvi-tools.org/en/stable/tutorials/notebooks/scrna/MrVI_tutorial.html>
- MrVI API: <https://docs.scvi-tools.org/en/stable/api/reference/scvi.external.MRVI.html>
- Milo (milopy): <https://github.com/MarioniLab/milopy>
- inferCNVpy: <https://infercnvpy.readthedocs.io>
- LIANA-py: <https://liana-py.readthedocs.io>
- cell2location: <https://cell2location.readthedocs.io>
- Companion download instructions: `CTCL_atlas_download_and_preliminary_analysis.md`

---

*End of build spec. The notebook is one file, ~12 sections, ~500–800 lines of code total. Headline deliverable is `results/figures/summary_replication.pdf` plus a two-paragraph narrative in `analysis_log.md` documenting which paper claims reproduced cleanly and where MrVI told a different (or sharper) story than the paper's per-cell-cluster Wilcoxon.*
