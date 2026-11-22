# old/26_tcr_signaling_subclones.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/26_tcr_signaling_subclones.ipynb` and is never run._

# 26 — TCR / co-stimulation signaling across malignant clones & subclones

Continues `23_subclonal_evolution.ipynb`. Reads the TCR/costim gene programs from
`TCR_costim_gene_lists_and_interpretation.md` and asks **where in the TCR/costim cascade each
malignant population concentrates dysregulation**, at **two levels — module scores and individual
genes** — cross-checked against arm-level inferCNV.

**Comparisons**
1. Within sample: malignant vs reactive-CD4.
2. Within malignant: each `nested_subclone` (nb23) vs its own donor's reactive-CD4 control.
3. Global + compartment: all malignant vs all reactive-CD4, stratified by disease stage
   (`stage_class`) and skin layer (epidermis/dermis).

Reference = **reactive CD4 only** (lineage-matched). Scoring = `sc.tl.score_genes`.

```python  # cell 1
import importlib
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
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
    raise FileNotFoundError(f"could not locate MF/data from {start}")


NB_DIR = _resolve_nb_dir(); print("NB_DIR =", NB_DIR)
sys.path.insert(0, str(NB_DIR))
import tcr_signaling_helpers as th
import subclone_helpers as sh
for _m in (th, sh):
    importlib.reload(_m)

SEED = 0
np.random.seed(SEED)
sc.settings.verbosity = 1

DATA = NB_DIR / "data" / "atlas_joint"
FIG = NB_DIR / "figures"; FIG.mkdir(exist_ok=True)
DE_DIR = DATA / "tcr_signaling_de"; DE_DIR.mkdir(exist_ok=True)

H5AD = DATA / "skin_T_tcr_malig_v2.h5ad"
SUBCLONES = DATA / "subclones_v2.parquet"
ARM_CNV = DATA / "skin_T_arm_cnv_res0.5.parquet"
TRUNK_BRANCH = DATA / "subclone_trunk_branch_v2.csv"
MALIG_COL = "tcr_malignant_alice"
SUBCLONE_COL = "nested_subclone"
GROUPS = ("malignant", "reactive_CD4")
```

```
NB_DIR = /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF
```

## §0 Load & prep
Rebuild `X` from `raw_counts`, normalize+log1p; join subclones + arm-CNV; derive `malig_group`, `skin_layer`.

```python  # cell 3
adata = sc.read_h5ad(H5AD)
# X/counts are freed on load in this object -> rebuild from raw_counts (mirror nb23)
adata.X = adata.layers["raw_counts"].copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata
```

```python  # cell 4
# join subclone assignments (key: cell_id, else index) and per-cell arm-CNV scores
sub = pd.read_parquet(SUBCLONES)
if "cell_id" in sub.columns:
    sub = sub.set_index("cell_id")
for c in ["nested_subclone", "cnv_subclone", "is_dominant_subclone"]:
    if c in sub.columns:
        adata.obs[c] = sub[c].reindex(adata.obs_names)

arm = pd.read_parquet(ARM_CNV)
if "obs_name" in arm.columns:
    arm = arm.set_index("obs_name")
ARM_COLS = [c for c in arm.columns if c.startswith("chr")]
arm = arm.reindex(adata.obs_names)
print("arm cnv coverage:", arm[ARM_COLS].notna().any(axis=1).mean())
```

```python  # cell 5
# groups: malignant vs lineage-matched reactive CD4
malig = adata.obs[MALIG_COL].astype("boolean").fillna(False).to_numpy()
ct = adata.obs["cell_type_T2"].astype(str)
grp = np.where(malig, "malignant",
               np.where((~malig) & (ct == "CD4"), "reactive_CD4", "other"))
adata.obs["malig_group"] = pd.Categorical(grp, categories=["malignant", "reactive_CD4", "other"])

# skin layer from messy `tissue` (substring, same logic as subclone_helpers.layer_split_key)
tl = adata.obs["tissue"].astype(str).str.lower()
adata.obs["skin_layer"] = np.select(
    [tl.str.contains("epi"), tl.str.contains("derm")], ["epidermis", "dermis"], default="other")

print(adata.obs["malig_group"].value_counts())
print(adata.obs["stage_class"].value_counts())
print(adata.obs["skin_layer"].value_counts())
```

## §1 Score modules
Full scoring + a `_nolabile` re-score (dissociation-labile IEG genes dropped) for the sensitivity check. Coverage QC.

```python  # cell 7
cov = th.module_coverage(adata)
display(cov)

score_cols = th.score_tcr_modules(adata, seed=SEED, prefix="sig_")
score_cols_nl = th.score_tcr_modules(adata, seed=SEED, prefix="sig_nolabile_", drop=th.DISSOC_LABILE)
print(f"{len(score_cols)} module scores added")

# per-cell frames reused downstream
mod_scores = adata.obs[score_cols].copy()
genes_all = sorted({g for gs in th.TCR_MODULES.values() for g in gs} & set(adata.var_names))
gene_expr = th._dense_gene_frame(adata, genes_all)
ACT_COLS = [f"sig_{m}" for m in th.ACTIVITY_MODULES if f"sig_{m}" in score_cols]
print(len(genes_all), "module genes present;", len(ACT_COLS), "activity modules")
```

```python  # cell 8
# QC: labile vs no-labile agreement for the IEG-containing modules
for m in ["L4_ieg_acute", "L3c_ap1_mapk_activity"]:
    a, b = f"sig_{m}", f"sig_nolabile_{m}"
    if a in adata.obs and b in adata.obs:
        print(m, "spearman(full, no-labile) =",
              round(adata.obs[[a, b]].corr(method="spearman").iloc[0, 1], 3))
```

## §2 Within-sample — malignant vs reactive-CD4 (per donor)
Both levels. Module dotplot (module × donor); gene-level consistency lollipop. Context-only modules
are flagged (lineage/dose — not interpreted); `L0_pan_t_identity_loss` is an expected **drop**.

```python  # cell 10
grp_s = adata.obs["malig_group"]
donor_s = adata.obs["donor"].astype(str)
keep = grp_s.isin(GROUPS)

mod_stats = th.wilcoxon_two_group(mod_scores[keep.values], score_cols, grp_s[keep], GROUPS,
                                  level="module", donor_series=donor_s[keep])
gene_stats = th.wilcoxon_two_group(gene_expr[keep.values], genes_all, grp_s[keep], GROUPS,
                                   level="gene", donor_series=donor_s[keep])
mod_stats["module"] = mod_stats["feature"].str.replace("sig_", "", regex=False)
mod_stats["context_only"] = mod_stats["module"].isin(th.CONTEXT_ONLY)
mod_stats.to_csv(DE_DIR / "within_sample_module.csv", index=False)
gene_stats.to_csv(DE_DIR / "within_sample_gene.csv", index=False)
mod_stats.head()
```

```python  # cell 11
# module x donor dotplot (size=-log10 FDR, color=Cliff's delta; positive = up in malignant)
plot_df = mod_stats.copy()
plot_df["feature"] = plot_df["module"]
fig, ax = th.module_score_dotplot(plot_df, group_col="donor", feature_col="feature",
                                  title="Malignant vs reactive-CD4 module scores (per donor)")
fig.savefig(FIG / "tcr_signaling_within_sample_module_dotplot.png", dpi=150, bbox_inches="tight")
```

```python  # cell 12
# module x donor clustered heatmap: cols = donors clustered, rows = cascade-ordered modules.
# color = Cliff's delta (+ = up in malignant); bottom strips = study / disease stage / skin layer.
def _first(s): return s.astype(str).iloc[0]
def _dom_layer(s):
    s = s[s.isin(["epidermis", "dermis"])]
    return s.value_counts().idxmax() if len(s) else np.nan

_ob = adata.obs.groupby("donor", observed=True)
donor_annot = pd.DataFrame({"study": _ob["study"].agg(_first),
                            "stage": _ob["stage_class"].agg(_first),
                            "layer": _ob["skin_layer"].agg(_dom_layer)})
STRIP_PALS = {"stage": {"early": "#4daf4a", "advanced": "#e41a1c"},
              "layer": {"epidermis": "#377eb8", "dermis": "#ff7f00"}}
cg = th.module_donor_clustermap(mod_stats, group_col="donor", feature_col="module",
                                value_col="cliffs_delta", col_annot=donor_annot,
                                annot_palettes=STRIP_PALS,
                                title="Malignant vs reactive-CD4 module Δ (donors clustered)")
cg.savefig(FIG / "tcr_signaling_within_sample_module_clustermap.png", dpi=150, bbox_inches="tight")
```

```
<Figure size 900x700 with 5 Axes>
```

```python  # cell 13
# gene-level consistency across donors: mean Cliff's delta + # donors significant
g = (gene_stats.assign(sig=gene_stats["fdr"] < 0.05)
     .groupby("feature")
     .agg(cliffs_delta=("cliffs_delta", "mean"), n_sig=("sig", "sum"), n_donor=("donor", "nunique"))
     .reset_index())
g["module"] = g["feature"].map(lambda x: th.GENE_TO_MODULE.get(x, [""])[0])
top = g.reindex(g["cliffs_delta"].abs().sort_values(ascending=False).index).head(40)
ax = th.lollipop_effects(top, effect="cliffs_delta", label="feature", fdr_col=None,
                         title="Top genes: malignant vs reactive-CD4 (mean Cliff's δ across donors)",
                         figsize=(5, 8))
ax.figure.savefig(FIG / "tcr_signaling_within_sample_gene_lollipop.png", dpi=150, bbox_inches="tight")
```

```python  # cell 14
# gene-level lollipops, one figure per gene family (module): mean Cliff's δ across donors
fam_figs = th.lollipop_per_family(g, effect="cliffs_delta", feature_col="feature", fdr_col=None,
                                  title_prefix="mal vs reactive-CD4: ")
for _m, _fig in fam_figs.items():
    _fig.savefig(FIG / f"tcr_signaling_within_sample_lollipop_{_m}.png", dpi=150, bbox_inches="tight")
print(len(fam_figs), "family lollipops")
```

```python  # cell 15
# per-family violins: activity-module scores, malignant vs reactive-CD4 (pooled across donors)
obs_v = adata.obs.loc[keep.values, ["malig_group", *ACT_COLS]].copy()
for col in ACT_COLS:
    fig, _ = th.module_violin(obs_v, col, split_col="malig_group", order=list(GROUPS), title=col)
    fig.savefig(FIG / f"tcr_signaling_within_sample_violin_{col}.png", dpi=150, bbox_inches="tight")
print(len(ACT_COLS), "activity-module violins")
```

```python  # cell 16
# individual-gene dotplot for the largest-malignant donor, bracketed by module
n_mal = adata.obs.loc[malig, "donor"].value_counts()
top_donor = n_mal.index[0]
sub = adata[(donor_s == top_donor).values & keep.values].copy()
th.gene_dotplot(sub, group_col="malig_group",
                title=f"{top_donor}: malignant vs reactive-CD4 (genes by module)")
```

## §3 Within malignant — each subclone vs non-malignant control
Mirrors §2 on the `nested_subclone` axis (nb23 `subclones_v2.parquet`, joined in §0). Every nested
subclone is tested against **its own donor's reactive-CD4** (patient- + lineage-matched), at both
levels. Cohort-wide module × subclone dotplot + clustered heatmap; then per-subclone gene volcano
(colored by gene family) and per-family lollipops, saved to disk (one example shown inline).

```python  # cell 18
mal = adata[malig].copy()
mal_donor = mal.obs["donor"].astype(str)
mal_sub = mal.obs[SUBCLONE_COL].astype(str)
mal_mod = mal.obs[score_cols].copy()
mal_gene = th._dense_gene_frame(mal, genes_all)

# donors with >=2 subclones of adequate size
multi = []
for d in mal_donor.unique():
    subs = mal_sub[mal_donor == d].value_counts()
    if (subs >= 20).sum() >= 2:
        multi.append(d)
print(len(multi), "donors with >=2 sizeable subclones")
```

```python  # cell 19
# each nested subclone vs its OWN donor's reactive-CD4 control (both levels)
sub_series = adata.obs[SUBCLONE_COL].astype(str)
grp_all = adata.obs["malig_group"].astype(str)
don_all = adata.obs["donor"].astype(str)

subclone_mod = th.subclone_vs_control_stats(mod_scores, score_cols, sub_series, grp_all, don_all,
                                            level="module", control_group="reactive_CD4")
subclone_gene = th.subclone_vs_control_stats(gene_expr, genes_all, sub_series, grp_all, don_all,
                                             level="gene", control_group="reactive_CD4")
subclone_mod["module"] = subclone_mod["feature"].str.replace("sig_", "", regex=False)
subclone_mod["context_only"] = subclone_mod["module"].isin(th.CONTEXT_ONLY)
subclone_gene["module"] = subclone_gene["feature"].map(lambda x: th.GENE_TO_MODULE.get(x, [""])[0])
subclone_mod.to_csv(DE_DIR / "subclone_vs_control_module.csv", index=False)
subclone_gene.to_csv(DE_DIR / "subclone_vs_control_gene.csv", index=False)
print(subclone_mod["subclone"].nunique(), "subclones (module-level);",
      subclone_gene["subclone"].nunique(), "(gene-level) vs same-donor reactive-CD4")
subclone_mod.head()
```

```python  # cell 20
# cohort module x subclone dotplot: size=-log10 FDR, color=Cliff's delta (positive = up in subclone)
if not subclone_mod.empty:
    n_sub = subclone_mod["subclone"].nunique()
    fig, ax = th.module_score_dotplot(subclone_mod, group_col="subclone", feature_col="module",
                                      title="Subclone vs reactive-CD4 module scores (per subclone)",
                                      figsize=(max(6, 0.28 * n_sub + 3), 8))
    fig.savefig(FIG / "tcr_signaling_subclone_module_dotplot.png", dpi=150, bbox_inches="tight")
```

```python  # cell 21
# cohort module x subclone clustered heatmap: subclones clustered by their change profile.
# bottom strips = sample (donor) / study / disease stage / skin layer; per-subclone x labels off.
if not subclone_mod.empty:
    n_sub = subclone_mod["subclone"].nunique()
    def _first(s): return s.astype(str).iloc[0]
    def _dom_layer(s):
        s = s[s.isin(["epidermis", "dermis"])]
        return s.value_counts().idxmax() if len(s) else np.nan

    _gs = adata.obs[adata.obs[SUBCLONE_COL].astype(str) != ""].groupby(SUBCLONE_COL, observed=True)
    sub_meta = pd.DataFrame({"sample": _gs["donor"].agg(_first),
                             "study": _gs["study"].agg(_first),
                             "stage": _gs["stage_class"].agg(_first),
                             "layer": _gs["skin_layer"].agg(_dom_layer)})
    STRIP_PALS = {"stage": {"early": "#4daf4a", "advanced": "#e41a1c"},
                  "layer": {"epidermis": "#377eb8", "dermis": "#ff7f00"}}
    cg = th.module_donor_clustermap(subclone_mod, group_col="subclone", feature_col="module",
                                    value_col="cliffs_delta", col_annot=sub_meta,
                                    annot_palettes=STRIP_PALS, show_col_labels=False,
                                    title="Subclone vs reactive-CD4 module Δ (subclones clustered)",
                                    figsize=(max(9, 0.18 * n_sub + 4), 7))
    cg.savefig(FIG / "tcr_signaling_subclone_module_clustermap.png", dpi=150, bbox_inches="tight")
```

```
<Figure size 3100x700 with 5 Axes>
```

```python  # cell 22
# per-subclone gene volcano (Cliff's δ vs -log10 FDR, colored by gene family). Saved per subclone.
VOLC_DIR = FIG / "tcr_signaling_subclone_volcano"; VOLC_DIR.mkdir(exist_ok=True)
subs_all = sorted(subclone_gene["subclone"].unique()) if not subclone_gene.empty else []
for s in subs_all:
    d = subclone_gene[subclone_gene["subclone"] == s]
    ax = th.gene_volcano(d, effect="cliffs_delta", p_col="fdr", label="feature",
                         module_col="module", title=f"{s} vs reactive-CD4")
    ax.figure.savefig(VOLC_DIR / f"{s.replace('/', '_')}.png", dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
print(len(subs_all), "subclone volcanoes ->", VOLC_DIR)
# example inline: the largest subclone
if subs_all:
    ex = subclone_gene.loc[subclone_gene["n1"].idxmax(), "subclone"]
    th.gene_volcano(subclone_gene[subclone_gene["subclone"] == ex],
                    module_col="module", title=f"example — {ex} vs reactive-CD4");
```

```python  # cell 23
# per-family lollipops (subclone vs reactive-CD4) — SELECT samples to view (else ~150 subclones).
SAMPLES = ["Li2024_atlas__CTCL8"]   # <- one or more donors; [] = every subclone (huge)
sub2donor = subclone_gene.groupby("subclone")["donor"].first() if not subclone_gene.empty else {}
print("available donors:", sorted(set(sub2donor.values)))
sel = [s for s in subs_all if (not SAMPLES) or sub2donor.get(s) in SAMPLES]

LOLLI_DIR = FIG / "tcr_signaling_subclone_lollipop"; LOLLI_DIR.mkdir(exist_ok=True)
n_lolli = 0
for s in sel:
    d = subclone_gene[subclone_gene["subclone"] == s]
    sdir = LOLLI_DIR / s.replace("/", "_"); sdir.mkdir(exist_ok=True)
    figs = th.lollipop_per_family(d, effect="cliffs_delta", feature_col="feature", fdr_col="fdr",
                                  title_prefix=f"{s}: ")
    for _mod, _fig in figs.items():
        _fig.savefig(sdir / f"{_mod}.png", dpi=150, bbox_inches="tight")
        n_lolli += 1
    # figures for the selected samples are left open so they render inline
print(f"{n_lolli} per-family lollipops for {len(sel)} subclones (SAMPLES={SAMPLES}) -> {LOLLI_DIR}")
```

## §4 Global + compartment — all malignant vs all reactive-CD4
Pooled lollipops (module + genes), global gene dotplot, and stratification by disease stage / skin layer with violins.

```python  # cell 25
# pooled (across all samples) module-level effect sizes
mod_pooled = th.wilcoxon_two_group(mod_scores[keep.values], score_cols, grp_s[keep], GROUPS,
                                   level="module", donor_series=None)
mod_pooled["module"] = mod_pooled["feature"].str.replace("sig_", "", regex=False)
mod_pooled["context_only"] = mod_pooled["module"].isin(th.CONTEXT_ONLY)
mod_pooled.to_csv(DE_DIR / "global_module.csv", index=False)
ax = th.lollipop_effects(mod_pooled, label="module",
                         title="All malignant vs all reactive-CD4 (module Cliff's δ)", figsize=(5, 7))
ax.figure.savefig(FIG / "tcr_signaling_global_module_lollipop.png", dpi=150, bbox_inches="tight")
```

```python  # cell 26
# pooled gene-level, lollipop restricted to load-bearing activity modules
gene_pooled = th.wilcoxon_two_group(gene_expr[keep.values], genes_all, grp_s[keep], GROUPS,
                                    level="gene", donor_series=None)
gene_pooled["module"] = gene_pooled["feature"].map(lambda x: th.GENE_TO_MODULE.get(x, [""])[0])
gene_pooled.to_csv(DE_DIR / "global_gene.csv", index=False)
act_genes = gene_pooled[gene_pooled["module"].isin(th.ACTIVITY_MODULES)]
ax = th.lollipop_effects(act_genes, label="feature",
                         title="Activity-module genes: malignant vs reactive-CD4", figsize=(5, 9))
ax.figure.savefig(FIG / "tcr_signaling_global_activity_gene_lollipop.png", dpi=150, bbox_inches="tight")
```

```python  # cell 27
# global gene dotplot, malignant vs reactive-CD4, genes bracketed by module
sub_all = adata[keep.values].copy()
th.gene_dotplot(sub_all, group_col="malig_group",
                title="All samples: TCR/costim genes, malignant vs reactive-CD4")
```

```python  # cell 28
# stratified by compartment: module effect size per stratum
def stratified_module(strat_col, strata):
    out = []
    for s in strata:
        m = keep.values & (adata.obs[strat_col].astype(str) == s).values
        if m.sum() < 50:
            continue
        r = th.wilcoxon_two_group(mod_scores[m], score_cols, grp_s[m], GROUPS,
                                  level="module", donor_series=None)
        r["stratum"] = s
        out.append(r)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

stage_stats = stratified_module("stage_class", ["early", "advanced"])
layer_stats = stratified_module("skin_layer", ["epidermis", "dermis"])
for df, name in [(stage_stats, "stage"), (layer_stats, "layer")]:
    if df.empty:
        continue
    df["feature"] = df["feature"].str.replace("sig_", "", regex=False)
    fig, ax = th.module_score_dotplot(df, group_col="stratum", feature_col="feature",
                                      title=f"Malignant vs reactive-CD4 by {name}")
    fig.savefig(FIG / f"tcr_signaling_by_{name}_module_dotplot.png", dpi=150, bbox_inches="tight")
```

```python  # cell 29
# violins of key activity modules, split malignant/reactive, faceted by disease stage
obs_v = adata.obs[keep.values].copy()
key = ["sig_L3a_nfkb_activity", "sig_L3b_nfat_activity", "sig_SIGNAL3_jak_stat",
       "sig_COSTIM_pi3k_akt_activity", "sig_L5_chronicity_exhaustion"]
for col in [c for c in key if c in obs_v]:
    fig, _ = th.module_violin(obs_v, col, split_col="malig_group", facet_col="stage_class",
                              order=["early", "advanced"], title=col)
    fig.savefig(FIG / f"tcr_signaling_violin_{col}.png", dpi=150, bbox_inches="tight")
```

## §5 §2 signaling-mode localization
Per-subclone §2 contrast vector → clustered heatmap → mode-class label per subclone.

```python  # cell 31
contrast_df = th.subclone_contrast_vector(mal.obs, SUBCLONE_COL, score_prefix="sig_")
contrast_df = contrast_df[contrast_df.index.astype(str).str.len() > 0]
g = th.signaling_mode_heatmap(contrast_df, title="Subclone signaling-mode contrasts (§2)")
g.savefig(FIG / "tcr_signaling_mode_heatmap.png", dpi=150, bbox_inches="tight")
```

```python  # cell 32
from scipy.cluster.hierarchy import fcluster, linkage
contrast_names = [c for c in th.CONTRASTS if c in contrast_df.columns]
cvec = contrast_df[contrast_names].fillna(0.0)
Z = linkage(cvec, method="ward")
contrast_df["mode_cluster"] = fcluster(Z, t=4, criterion="maxclust")
# name each cluster by its most extreme mean contrast
names = {}
for cl, sub in contrast_df.groupby("mode_cluster"):
    mc = sub[contrast_names].mean()
    names[cl] = f"{mc.abs().idxmax()}{'+' if mc[mc.abs().idxmax()] > 0 else '-'}"
contrast_df["mode_label"] = contrast_df["mode_cluster"].map(names)
contrast_df["mode_label"].value_counts()
```

## §6 §3 arm-CNV × downstream activity cross
Discrete per-subclone arm events crossed against the matched activity module (independent dosage × function). Continuous per-cell arm score vs activity correlation as support.

```python  # cell 34
tb = pd.read_csv(TRUNK_BRANCH)
arm_events = th.parse_arm_events(tb)
# subclone-level activity means + donor label
sub_means = mal.obs.groupby(SUBCLONE_COL, observed=True)[ACT_COLS].mean()
sub_means["donor"] = mal.obs.groupby(SUBCLONE_COL, observed=True)["donor"].agg(
    lambda x: x.astype(str).iloc[0])
sub_means = sub_means[sub_means.index.astype(str).str.len() > 0].reset_index()
arm_events.head()
```

```python  # cell 35
for armn, direction, module, note in th.ARM_ACTIVITY_MAP:
    col = f"sig_{module}"
    if col not in sub_means.columns:
        continue
    fig, ax = th.arm_activity_box(sub_means, arm_events, armn, direction, module,
                                  title=f"{armn}{direction} -> {module}\n{note}")
    fig.savefig(FIG / f"tcr_signaling_cnv_{armn}{direction}_{module}.png", dpi=150, bbox_inches="tight")
```

```python  # cell 36
# support: per-donor Spearman(continuous arm score, matched activity module) over malignant cells
from scipy.stats import spearmanr
rows = []
arm_mal = arm.reindex(mal.obs_names)
for armn, direction, module, _ in th.ARM_ACTIVITY_MAP:
    acol = f"sig_{module}"
    if armn not in arm_mal.columns or acol not in mal.obs:
        continue
    for d in mal_donor.unique():
        m = (mal_donor == d).values
        x = arm_mal[armn].to_numpy()[m]
        y = mal.obs[acol].to_numpy()[m]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 30:
            continue
        rho, p = spearmanr(x[ok], y[ok])
        rows.append({"donor": d, "arm": armn, "direction": direction, "module": module,
                     "rho": rho, "p": p, "n": int(ok.sum())})
arm_corr = pd.DataFrame(rows)
arm_corr.to_csv(DE_DIR / "arm_activity_correlation.csv", index=False)
arm_corr.groupby(["arm", "module"])["rho"].median()
```

## §7 Sensitivity & caveats
Re-run the pooled malignant-vs-reactive contrast on the `_nolabile` IEG scores; compare 5′ vs FFPE-Flex (`tech`) as the dissociation negative control.

```python  # cell 38
labile_mods = [m for m in ["L4_ieg_acute", "L3c_ap1_mapk_activity"] if f"sig_{m}" in adata.obs]
comp = []
for m in labile_mods:
    full = th.wilcoxon_two_group(mod_scores[keep.values], [f"sig_{m}"], grp_s[keep], GROUPS,
                                 level="module", donor_series=None)
    nl = th.wilcoxon_two_group(adata.obs.loc[keep, [f"sig_nolabile_{m}"]], [f"sig_nolabile_{m}"],
                               grp_s[keep], GROUPS, level="module", donor_series=None)
    comp.append({"module": m, "cliffs_full": full["cliffs_delta"].iloc[0],
                 "cliffs_nolabile": nl["cliffs_delta"].iloc[0]})
pd.DataFrame(comp)
```

```python  # cell 39
# tech split for the acute-IEG module (dissociation negative control)
if "tech" in adata.obs and "sig_L4_ieg_acute" in adata.obs:
    obs_t = adata.obs[keep.values]
    fig, _ = th.module_violin(obs_t, "sig_L4_ieg_acute", split_col="malig_group",
                              facet_col="tech", title="Acute-IEG by chemistry (dissociation check)")
    fig.savefig(FIG / "tcr_signaling_ieg_by_tech.png", dpi=150, bbox_inches="tight")
```

**Caveats (doc §4)** — lineage confound handled via reactive-CD4 reference; `*_components`/coreceptor/kinase modules are context-only (flagged, not interpreted). inferCNV circularity: lean on arm-CNV × *activity* crosses; 20q↔PLCG1 read through NFAT activity, not PLCG1 RNA. IEG modules validated with the `_nolabile` re-score + tech split. GZMB/NFAT tests are on malignant (`tcr_malignant_alice`) cells only. `RCAN1` read as part of the NFAT set, not alone.

## §8 Save

```python  # cell 42
out = contrast_df.reset_index().rename(columns={"index": SUBCLONE_COL})
out.to_csv(DATA / "tcr_signaling_subclone_modes.csv", index=False)
print("wrote", DATA / "tcr_signaling_subclone_modes.csv")
print("DE tables in", DE_DIR)
print("figures in", FIG)
```
