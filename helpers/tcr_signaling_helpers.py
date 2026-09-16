"""TCR / co-stimulation signaling analysis helpers.

Gene modules + scoring + two-group / multi-group statistics + plots, transcribed from
`TCR_costim_gene_lists_and_interpretation.md`. Every comparison runs at two levels:
module score (sc.tl.score_genes) and individual genes.

Companion notebook: 26_tcr_signaling_subclones.ipynb.
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import sparse
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# 1. Gene modules (verbatim from TCR_costim_gene_lists_and_interpretation.md §1)
# ---------------------------------------------------------------------------
TCR_MODULES: dict[str, list[str]] = {
    # Level 0 — receptor complex & T-cell identity
    "L0_tcr_cd3_components": ["CD3D", "CD3E", "CD3G", "CD247", "TRAC", "TRBC2"],
    "L0_pan_t_identity_loss": ["CD2", "CD5", "CD7", "CD6", "CD52"],
    "L0_coreceptor": ["CD4", "CD8A", "CD8B"],
    # Level 1 — proximal kinases/phosphatases (lineage-confounded)
    "L1_proximal_kinases": ["LCK", "FYN", "ZAP70", "ITK", "TXK", "CSK", "PTPN6",
                            "PTPN22", "PTPRC", "UBASH3A", "UBASH3B"],
    # Level 2 — LAT signalosome, GEFs, PLCG1 pivot
    "L2_lat_signalosome_components": ["LAT", "LCP2", "GRAP2", "GRB2", "THEMIS", "NCK1", "WAS", "WIPF1"],
    "L2_gef_gtpase_components": ["VAV1", "VAV2", "VAV3", "SOS1", "RASGRP1", "RASGRP2", "RHOA", "RAC2", "CDC42"],
    "L2_plcg1_node_components": ["PLCG1", "PRKCB", "PRKCQ", "DGKA", "DGKZ"],
    # Level 3a — DAG → PKCθ → CBM → canonical NF-κB
    "L3a_cbm_nfkb_components": ["PRKCQ", "CARD11", "BCL10", "MALT1", "TRAF6", "TRAF2", "MAP3K7",
                                "TAB1", "TAB2", "CHUK", "IKBKB", "IKBKG", "NFKB1", "NFKB2",
                                "RELA", "RELB", "REL"],
    "L3a_nfkb_activity": ["NFKBIA", "NFKBIE", "NFKBIZ", "TNFAIP3", "TNFAIP8", "BIRC3", "BIRC2",
                          "TRAF1", "CFLAR", "BCL2A1", "CD83", "ICAM1", "CD40", "TNF", "LTB",
                          "CCL3", "CCL4", "NFKB2", "RELB"],
    # Level 3b — IP3 → Ca²⁺ → calcineurin → NFAT
    "L3b_calcium_nfat_components": ["PLCG1", "ITPR1", "ITPR2", "ITPR3", "STIM1", "STIM2", "ORAI1",
                                    "ORAI2", "ORAI3", "PPP3CA", "PPP3CB", "PPP3R1", "CAMK4",
                                    "NFATC1", "NFATC2", "NFATC3"],
    "L3b_nfat_activity": ["RCAN1", "NFATC1", "EGR2", "EGR3", "IRF4", "BATF", "IL2RA", "CD40LG",
                          "PDCD1", "CTLA4", "TNFRSF9", "TNFRSF18"],
    # Level 3c — RAS → MAPK → AP-1
    "L3c_ras_mapk_components": ["RASGRP1", "SOS1", "KRAS", "NRAS", "BRAF", "RAF1", "MAP2K1",
                                "MAP2K2", "MAPK1", "MAPK3", "MAPK8", "MAPK9", "MAPK14", "MAP3K8"],
    "L3c_ap1_mapk_activity": ["FOS", "FOSB", "JUN", "JUNB", "JUND", "ATF3", "EGR1", "DUSP1",
                              "DUSP2", "DUSP4", "DUSP5", "DUSP6", "IER2"],
    # Level 4 — integrated acute trigger (immediate-early)
    "L4_ieg_acute": ["FOS", "FOSB", "JUN", "JUNB", "JUND", "EGR1", "EGR2", "EGR3", "NR4A1",
                     "NR4A2", "NR4A3", "DUSP1", "DUSP2", "IER2", "IER3", "ZFP36", "CD69", "TNF", "MYC"],
    # Level 5 — sustained state & chronicity
    "L5_sustained_tf_layer": ["IRF4", "BATF", "BATF3", "TBX21", "GATA3", "RUNX3", "PRDM1", "ID2",
                              "ID3", "TOX", "TOX2", "ZEB1", "IKZF1", "IKZF2", "IKZF3", "MYB"],
    "L5_chronicity_exhaustion": ["TOX", "NR4A1", "NR4A2", "NR4A3", "PDCD1", "TIGIT", "HAVCR2",
                                 "LAG3", "ENTPD1", "CTLA4", "BATF", "IRF4"],
    # Co-stimulation
    "COSTIM_receptors_activating": ["CD28", "ICOS", "TNFRSF4", "TNFRSF9", "TNFRSF18", "CD27",
                                    "TNFRSF1B", "TNFRSF14", "CD2", "SLAMF1", "CD226", "CD40LG", "TNFRSF8"],
    "COSTIM_coinhibitory_receptors": ["CTLA4", "PDCD1", "BTLA", "LAG3", "HAVCR2", "TIGIT", "CD160",
                                      "VSIR", "ENTPD1", "KIR3DL2", "KIR2DL3", "KIR3DL1", "KLRG1"],
    "COSTIM_ligands_on_TME_partners": ["CD80", "CD86", "ICOSLG", "TNFSF4", "TNFSF9", "TNFSF18",
                                       "CD70", "TNF", "CD58", "PVR", "NECTIN2", "TNFSF14", "TNFSF8"],
    "COSTIM_pi3k_akt_components": ["PIK3CD", "PIK3R1", "PIK3CA", "PTEN", "AKT1", "AKT2", "PDPK1",
                                   "MTOR", "RICTOR", "RPTOR", "RHEB", "TSC1", "TSC2"],
    "COSTIM_pi3k_akt_activity": ["FOXO1", "RPS6KB1", "EIF4EBP1", "MYC", "CCND2", "SLC2A1", "BCL2L1"],
    # Signal 3 (JAK-STAT)
    "SIGNAL3_jak_stat": ["IL2RA", "IL2RB", "IL2RG", "JAK1", "JAK3", "TYK2", "STAT3", "STAT5A",
                         "STAT5B", "IL4R", "IL7R", "IL15RA", "SOCS1", "SOCS3", "CISH", "PIM1"],
    # Negative regulators
    "NEGREG_rheostat": ["CD5", "PTPN6", "PTPN22", "DGKA", "DGKZ", "CBL", "CBLB", "ITCH", "TNFAIP3",
                        "CYLD", "RC3H1", "ZC3H12A", "TNIP1", "NFKBIA", "RCAN1", "UBASH3A", "DUSP1",
                        "DUSP2", "SOCS1", "SOCS3", "CISH"],
}

# Load-bearing = the *_activity modules; these are context/dose only (doc §1, §4).
CONTEXT_ONLY: set[str] = {
    "L0_coreceptor", "L1_proximal_kinases",
    "L0_tcr_cd3_components", "L2_lat_signalosome_components", "L2_gef_gtpase_components",
    "L2_plcg1_node_components", "L3a_cbm_nfkb_components", "L3b_calcium_nfat_components",
    "L3c_ras_mapk_components", "COSTIM_pi3k_akt_components",
}

# Modules where a DROP is the signal (MF immunophenotype loss).
LOSS_IS_SIGNAL: set[str] = {"L0_pan_t_identity_loss"}

# Scored on DC/fibroblast/B/macrophage only — NOT the T-cell object. Defer to full-atlas notebook.
TME_LIGAND_MODULE: str = "COSTIM_ligands_on_TME_partners"

# Dissociation-labile IEG genes (doc §4) — removed for the _nolabile sensitivity re-score.
DISSOC_LABILE: set[str] = {"FOS", "FOSB", "JUN", "JUNB", "JUND", "EGR1", "NR4A1", "NR4A2",
                           "NR4A3", "DUSP1", "DUSP2", "ZFP36", "IER2", "CD69"}

# §2 signaling-mode contrasts: (numerator_module, denominator_module).
# Computed as z(num) - z(denom) across cells (robust analog of the doc's ratios, which are
# ill-defined on signed score_genes output).
CONTRASTS: dict[str, tuple[str, str]] = {
    "calcium_vs_cbm": ("L3b_nfat_activity", "L3a_nfkb_activity"),
    "autonomous_nfkb": ("L3a_nfkb_activity", "L4_ieg_acute"),
    "costim_amp": ("COSTIM_pi3k_akt_activity", "L4_ieg_acute"),
    "chronic_vs_acute": ("L5_chronicity_exhaustion", "L4_ieg_acute"),
    "ap1_vs_nfat": ("L3c_ap1_mapk_activity", "L3b_nfat_activity"),
}

# §3 arm-CNV × downstream activity crosses. (arm, direction '+'gain/'-'loss, activity_module, note)
ARM_ACTIVITY_MAP: list[tuple[str, str, str, str]] = [
    ("chr17q", "+", "SIGNAL3_jak_stat", "STAT3/5 gain → signal-3/JAK-STAT amplification (cleanest cross)"),
    ("chr10q", "-", "COSTIM_pi3k_akt_activity", "PTEN loss → PI3K/AKT amplification (FOXO1 down, mTOR up)"),
    ("chr6q", "-", "L3a_nfkb_activity", "A20/CYLD loss → de-repressed NF-κB"),
    ("chr8q", "+", "L5_chronicity_exhaustion", "TOX/MYC gain → chronicity/exhaustion + proliferation"),
    ("chr2q", "+", "COSTIM_receptors_activating", "CD28/ICOS/CTLA4 gain → costim amplified"),
    ("chr20q", "+", "L3b_nfat_activity", "PLCG1 dose → NFAT (circular vs PLCG1 RNA; read via activity)"),
    ("chr1p", "-", "COSTIM_receptors_activating", "loss removes TNFR2/costim + CD58 (tension w/ TNFR2 gains)"),
]

# Reverse map: gene -> list of modules it belongs to (for bracketing gene-level results).
GENE_TO_MODULE: dict[str, list[str]] = {}
for _mod, _genes in TCR_MODULES.items():
    for _g in _genes:
        GENE_TO_MODULE.setdefault(_g, []).append(_mod)

# Load-bearing activity modules — the ones to interpret / lead figures with.
ACTIVITY_MODULES: list[str] = [m for m in TCR_MODULES if m.endswith("_activity")]


# ---------------------------------------------------------------------------
# 2. Scoring
# ---------------------------------------------------------------------------
def score_tcr_modules(adata, modules=None, prefix="sig_", seed=0, min_genes=2, drop=None):
    """Score each module with sc.tl.score_genes. Returns list of added obs columns.

    `drop`: optional set of genes to exclude (e.g. DISSOC_LABILE for the sensitivity re-score);
    use a distinct `prefix` (e.g. 'sig_nolabile_') to keep both scorings.
    """
    modules = TCR_MODULES if modules is None else modules
    drop = set() if drop is None else set(drop)
    var_names = set(adata.var_names)
    added, skipped = [], {}
    for name, genes in modules.items():
        present = [g for g in genes if g in var_names and g not in drop]
        if len(present) < min_genes:
            skipped[name] = len(present)
            continue
        col = f"{prefix}{name}"
        sc.tl.score_genes(adata, present, score_name=col, random_state=seed)
        added.append(col)
    if skipped:
        warnings.warn(f"skipped {len(skipped)} modules with <{min_genes} present genes: {skipped}")
    return added


def module_coverage(adata, modules=None):
    """DataFrame of present/total gene counts per module (QC)."""
    modules = TCR_MODULES if modules is None else modules
    var_names = set(adata.var_names)
    rows = []
    for name, genes in modules.items():
        present = [g for g in genes if g in var_names]
        rows.append({"module": name, "n_present": len(present), "n_total": len(genes),
                     "missing": ",".join(g for g in genes if g not in var_names),
                     "context_only": name in CONTEXT_ONLY})
    return pd.DataFrame(rows).sort_values("module").reset_index(drop=True)


def _dense_gene_frame(adata, genes, index=None):
    """Per-cell expression DataFrame for `genes` (present only) from adata.X."""
    present = [g for g in genes if g in adata.var_names]
    if not present:
        return pd.DataFrame(index=adata.obs_names if index is None else index)
    X = adata[:, present].X
    X = X.toarray() if sparse.issparse(X) else np.asarray(X)
    return pd.DataFrame(X, columns=present, index=adata.obs_names)


# ---------------------------------------------------------------------------
# 3. Statistics
# ---------------------------------------------------------------------------
def _cliffs_delta(u, n1, n2):
    return (2.0 * u) / (n1 * n2) - 1.0


def _two_group_one_feature(a, b, feature, level, log2fc=True, eps=1e-9):
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n1, n2 = len(a), len(b)
    if n1 < 3 or n2 < 3:
        return None
    if np.ptp(np.concatenate([a, b])) == 0:  # all identical -> test undefined
        return None
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    ma, mb = float(np.mean(a)), float(np.mean(b))
    row = {"feature": feature, "level": level, "n1": n1, "n2": n2, "U": float(u), "p": float(p),
           "cliffs_delta": _cliffs_delta(u, n1, n2), "mean_a": ma, "mean_b": mb,
           "mean_diff": ma - mb}
    row["log2fc"] = float(np.log2((ma + eps) / (mb + eps))) if (log2fc and ma > 0 and mb > 0) else np.nan
    return row


def wilcoxon_two_group(values_df, feature_cols, group_series, groups, level,
                       donor_series=None, log2fc=True):
    """Mann-Whitney U per feature, group_a vs group_b. Pooled, or per-donor if donor_series given.

    values_df: per-cell features (module scores or gene expression), indexed like group_series.
    groups=(a, b): a is the 'foreground' (positive cliffs_delta => higher in a).
    Returns tidy DataFrame with BH-FDR (fdr computed within each donor stratum, or global if pooled).
    """
    ga, gb = groups
    g = group_series.reindex(values_df.index)
    rows = []
    if donor_series is None:
        strata = [("__all__", values_df.index)]
    else:
        d = donor_series.reindex(values_df.index)
        strata = [(dn, d.index[d == dn]) for dn in pd.unique(d.dropna())]
    for donor, idx in strata:
        sub_g = g.reindex(idx)
        idx_a = idx[sub_g == ga]
        idx_b = idx[sub_g == gb]
        if len(idx_a) < 3 or len(idx_b) < 3:
            continue
        stratum_rows = []
        for feat in feature_cols:
            a = values_df[feat].reindex(idx_a).to_numpy(dtype=float)
            b = values_df[feat].reindex(idx_b).to_numpy(dtype=float)
            r = _two_group_one_feature(a, b, feat, level, log2fc=log2fc)
            if r is None:
                continue
            r["donor"] = donor
            r["group_a"], r["group_b"] = ga, gb
            stratum_rows.append(r)
        if stratum_rows:
            pv = np.array([r["p"] for r in stratum_rows])
            fdr = multipletests(pv, method="fdr_bh")[1]
            for r, q in zip(stratum_rows, fdr):
                r["fdr"] = float(q)
            rows.extend(stratum_rows)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["direction"] = np.where(out["cliffs_delta"] >= 0, f"up_in_{ga}", f"up_in_{gb}")
    return out


def kruskal_multi(values_df, feature_cols, group_series, level, min_per_group=20):
    """Kruskal-Wallis across ≥3 groups (e.g. subclones) per feature, with BH-FDR."""
    g = group_series.reindex(values_df.index)
    counts = g.value_counts()
    keep = counts[counts >= min_per_group].index
    g = g[g.isin(keep)]
    if g.nunique() < 3:
        return pd.DataFrame()
    rows = []
    for feat in feature_cols:
        samples = [values_df[feat].reindex(g.index[g == k]).to_numpy(dtype=float) for k in keep]
        samples = [s[np.isfinite(s)] for s in samples]
        if any(len(s) < min_per_group for s in samples):
            continue
        if np.ptp(np.concatenate(samples)) == 0:  # all identical -> test undefined
            continue
        h, p = kruskal(*samples)
        means = {f"mean_{k}": float(np.mean(s)) for k, s in zip(keep, samples)}
        rng = max(m for m in means.values()) - min(m for m in means.values())
        rows.append({"feature": feat, "level": level, "H": float(h), "p": float(p),
                     "n_groups": len(keep), "mean_range": rng, **means})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr"] = multipletests(out["p"].to_numpy(), method="fdr_bh")[1]
    return out


def subclone_vs_control_stats(values_df, feature_cols, subclone_series, group_series,
                              donor_series, level, control_group="reactive_CD4",
                              min_subclone=20, min_control=20, log2fc=True):
    """Each malignant subclone vs its OWN donor's control group (patient + lineage matched).

    Mirror of `wilcoxon_two_group` on the nested-subclone axis (nb23). For every subclone the
    foreground = that subclone's cells, background = the same donor's `control_group` cells
    (reactive-CD4). positive cliffs_delta => higher in the subclone. BH-FDR computed within each
    subclone across features. Returns a tidy DataFrame with `subclone`/`donor` columns.
    """
    sc_s = subclone_series.reindex(values_df.index).astype(str)
    grp = group_series.reindex(values_df.index).astype(str)
    dn = donor_series.reindex(values_df.index).astype(str)
    subclones = [s for s in pd.unique(sc_s) if s and s.lower() != "nan"]
    rows = []
    for s in subclones:
        s_idx = values_df.index[sc_s == s]
        if len(s_idx) < min_subclone:
            continue
        donor = dn.reindex(s_idx).mode().iloc[0]
        c_idx = values_df.index[(grp == control_group) & (dn == donor)]
        if len(c_idx) < min_control:
            continue
        srows = []
        for feat in feature_cols:
            a = values_df[feat].reindex(s_idx).to_numpy(dtype=float)
            b = values_df[feat].reindex(c_idx).to_numpy(dtype=float)
            r = _two_group_one_feature(a, b, feat, level, log2fc=log2fc)
            if r is None:
                continue
            r["subclone"], r["donor"], r["n_control"] = s, donor, len(c_idx)
            srows.append(r)
        if srows:
            fdr = multipletests(np.array([r["p"] for r in srows]), method="fdr_bh")[1]
            for r, q in zip(srows, fdr):
                r["fdr"] = float(q)
            rows.extend(srows)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["direction"] = np.where(out["cliffs_delta"] >= 0, "up_in_subclone", "up_in_control")
    return out


def subclone_contrast_vector(obs, subclone_col, score_prefix="sig_"):
    """Per-subclone §2 contrast vector: z(num)-z(denom) of module means, z across subclones.

    Returns DataFrame subclone × contrast (+ raw activity-module means).
    """
    def z(x):
        s = x.std(ddof=0)
        return (x - x.mean()) / s if s > 0 else x * 0.0

    needed = set(ACTIVITY_MODULES)
    for num, den in CONTRASTS.values():
        needed.update((num, den))
    score_cols = [f"{score_prefix}{m}" for m in sorted(needed) if f"{score_prefix}{m}" in obs.columns]
    means = obs.groupby(subclone_col, observed=True)[score_cols].mean()
    zmeans = means.apply(z, axis=0)
    out = pd.DataFrame(index=means.index)
    for name, (num, den) in CONTRASTS.items():
        cn, cd = f"{score_prefix}{num}", f"{score_prefix}{den}"
        if cn in zmeans.columns and cd in zmeans.columns:
            out[name] = zmeans[cn] - zmeans[cd]
    return out.join(means)


# ---------------------------------------------------------------------------
# 4. Plots  (matplotlib/seaborn; title = finding; figsize small per repo convention)
# ---------------------------------------------------------------------------
def lollipop_effects(df, effect="cliffs_delta", label="feature", fdr_col="fdr", fdr_thr=0.05,
                     title=None, ax=None, figsize=(5, 6), color_sig="#c0392b", color_ns="#b0b0b0"):
    """Horizontal lollipop of effect sizes, sorted; dot colored by significance."""
    d = df.loc[:, ~df.columns.duplicated()].dropna(subset=[effect]).sort_values(effect)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(d))
    sig = d[fdr_col] < fdr_thr if (fdr_col and fdr_col in d) else np.ones(len(d), bool)
    colors = np.where(sig, color_sig, color_ns)
    ax.hlines(y, 0, d[effect], color="#888", lw=1, zorder=1)
    ax.scatter(d[effect], y, c=colors, s=40, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(d[label].astype(str), fontsize=8)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel(effect)
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def gene_volcano(df, effect="cliffs_delta", p_col="fdr", label="feature", module_col="module",
                 fdr_thr=0.05, top_n=12, title=None, ax=None, figsize=(5.5, 5), cmap="tab20"):
    """Volcano: effect size (x) vs -log10(FDR) (y), points colored by gene family (`module_col`).

    Labels the top_n features by |effect| * -log10(p). Dashed line at fdr_thr.
    """
    d = df.dropna(subset=[effect, p_col]).copy()
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    d["_y"] = -np.log10(d[p_col].clip(lower=1e-300))
    if module_col and module_col in d and d[module_col].notna().any():
        fams = sorted(d[module_col].dropna().astype(str).unique())
        cm = plt.get_cmap(cmap)
        cmap_d = {f: cm(i % cm.N) for i, f in enumerate(fams)}
        for f in fams:
            m = d[module_col].astype(str) == f
            ax.scatter(d.loc[m, effect], d.loc[m, "_y"], s=22, color=cmap_d[f], label=f,
                       edgecolor="k", linewidth=0.2)
        ax.legend(fontsize=5, markerscale=0.7, ncol=1, bbox_to_anchor=(1.01, 1), loc="upper left")
    else:
        ax.scatter(d[effect], d["_y"], s=22, color="#4477aa", edgecolor="k", linewidth=0.2)
    ax.axhline(-np.log10(fdr_thr), color="k", ls="--", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    d["_rank"] = d[effect].abs() * d["_y"]
    for _, r in d.nlargest(top_n, "_rank").iterrows():
        ax.annotate(str(r[label]), (r[effect], r["_y"]), fontsize=6,
                    xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel(effect)
    ax.set_ylabel("-log10 FDR")
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def module_donor_clustermap(stats_df, group_col="donor", feature_col="module",
                            value_col="cliffs_delta", title=None, figsize=(9, 7),
                            cmap="RdBu_r", row_cluster=False, col_cluster=True, row_order=None,
                            col_annot=None, annot_palettes=None, show_col_labels=True):
    """Heatmap module (rows) × donor (cols), colored by effect size.

    Column dendrogram groups the samples (donors) by their module-change profile
    (positive = up in malignant). Rows default to the TCR-cascade order (`row_cluster=False`,
    `row_order=None` -> `TCR_MODULES` key order) so the pathway reads L0->L5; set
    `row_cluster=True` to cluster rows instead. Companion to `module_score_dotplot` (§2).

    `col_annot`: optional DataFrame indexed by the column ids (donor / subclone) whose columns are
    categorical annotation tracks (e.g. `sample`, `study`). Rendered as colored strips BELOW the
    heatmap in clustered-column order, with a legend per track; `show_col_labels` writes the column
    names under the strips (turn off for many columns). `annot_palettes`: {track: {cat: color}}.
    """
    mat = stats_df.pivot_table(index=feature_col, columns=group_col, values=value_col)
    mat = mat.dropna(how="all")
    if not row_cluster:
        order = list(TCR_MODULES) if row_order is None else list(row_order)
        ordered = [m for m in order if m in mat.index] + [m for m in mat.index if m not in order]
        mat = mat.loc[ordered]
    n_don = mat.shape[1]
    filled = mat.fillna(0.0)
    vmax = float(np.nanmax(np.abs(filled.to_numpy()))) or 1.0
    do_col_cluster = col_cluster and n_don > 2
    g = sns.clustermap(filled, cmap=cmap, vmin=-vmax, vmax=vmax, center=0,
                       row_cluster=row_cluster, col_cluster=do_col_cluster,
                       figsize=figsize, xticklabels=True, yticklabels=True,
                       cbar_kws={"label": value_col})
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=7, rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=7)
    g.cax.set_visible(False)  # drop the colorbar; effect direction is self-evident (red=up)

    if col_annot is not None:
        from matplotlib.colors import to_rgba
        from matplotlib.patches import Patch
        order = (list(g.dendrogram_col.reordered_ind)
                 if do_col_cluster and g.dendrogram_col is not None else list(range(mat.shape[1])))
        cols_ordered = [mat.columns[i] for i in order]
        ann = col_annot.reindex(mat.columns).astype(str)
        tracks = list(ann.columns)
        palettes = {}
        for t in tracks:
            cats = sorted(v for v in ann[t].unique() if v and v.lower() != "nan")
            pal = (annot_palettes or {}).get(t)
            if pal is None:
                if t in ("sample", "donor"):
                    cols = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
                else:
                    cols = list(plt.get_cmap("Set2").colors) + list(plt.get_cmap("Set3").colors)
                pal = {c: cols[i % len(cols)] for i, c in enumerate(cats)}
            palettes[t] = pal
        strip = np.ones((len(tracks), len(cols_ordered), 4))
        for ti, t in enumerate(tracks):
            for ci, col in enumerate(cols_ordered):
                strip[ti, ci] = to_rgba(palettes[t].get(ann.loc[col, t], (0.87, 0.87, 0.87, 1.0)))
        g.ax_heatmap.set_xticks([]); g.ax_heatmap.set_xticklabels([])
        hm = g.ax_heatmap.get_position()
        row_h = min(0.03, 0.9 * hm.height / max(len(mat), 1))
        total_h = row_h * len(tracks)
        ax_s = g.fig.add_axes([hm.x0, hm.y0 - total_h - 0.006, hm.width, total_h])
        ax_s.imshow(strip, aspect="auto", interpolation="none",
                    extent=[0, len(cols_ordered), 0, len(tracks)], origin="upper")
        ax_s.set_yticks(np.arange(len(tracks)) + 0.5)
        ax_s.set_yticklabels(tracks[::-1], fontsize=7)
        ax_s.set_xticks([])
        for sp in ax_s.spines.values():
            sp.set_visible(False)
        if show_col_labels:
            ax_s.set_xticks(np.arange(len(cols_ordered)) + 0.5)
            ax_s.set_xticklabels(cols_ordered, rotation=90, fontsize=6)
        # row labels default to the right edge, where the annot legend also sits -> collision.
        # move the module row labels + ylabel to the (empty, row_cluster=False) left side.
        g.ax_heatmap.yaxis.set_ticks_position("left")
        g.ax_heatmap.yaxis.set_label_position("left")
        handles = []
        for t in tracks:
            handles.append(Patch(facecolor="white", edgecolor="white", label=f"{t}:"))
            handles += [Patch(facecolor=palettes[t][k], label=str(k)) for k in palettes[t]]
        g.ax_heatmap.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                            fontsize=5, frameon=False, handlelength=1.0)

    if title:
        g.fig.suptitle(title, fontsize=10)
    return g


def lollipop_per_family(gene_summary, modules=None, effect="cliffs_delta", feature_col="feature",
                        fdr_col=None, min_genes=2, width=4.5, title_prefix=""):
    """One lollipop figure per gene family (TCR module). Returns dict module -> fig.

    `gene_summary`: per-gene stats with `feature_col` (gene) + `effect` columns (e.g. the
    across-donor mean-Cliff's-δ frame). A gene shown in every family it belongs to.
    """
    modules = TCR_MODULES if modules is None else modules
    vals = gene_summary.set_index(feature_col)
    figs = {}
    for mod, genes in modules.items():
        present = [g for g in genes if g in vals.index]
        if len(present) < min_genes:
            continue
        d = vals.loc[present, :].reset_index()
        h = max(1.8, 0.28 * len(present) + 0.9)
        fig, ax = plt.subplots(figsize=(width, h))
        lollipop_effects(d, effect=effect, label=feature_col, fdr_col=fdr_col, ax=ax,
                         title=f"{title_prefix}{mod}")
        fig.tight_layout()
        figs[mod] = fig
    return figs


def module_score_dotplot(stats_df, group_col="donor", feature_col="feature",
                         size_col="fdr", color_col="cliffs_delta", title=None,
                         figsize=None, cmap="RdBu_r", vmax=None):
    """Grid dotplot from a tidy stats DataFrame: feature (y) × group (x).

    Dot size = -log10(FDR); color = effect (cliffs_delta). One glyph per (feature, group).
    """
    d = stats_df.copy()
    d["nlog_fdr"] = -np.log10(d[size_col].clip(lower=1e-300))
    feats = sorted(d[feature_col].unique())
    groups = sorted(d[group_col].unique())
    fi = {f: i for i, f in enumerate(feats)}
    gi = {g: i for i, g in enumerate(groups)}
    if figsize is None:
        figsize = (max(4, 0.5 * len(groups) + 3), max(3, 0.3 * len(feats) + 1))
    fig, ax = plt.subplots(figsize=figsize)
    vmax = vmax or float(np.nanmax(np.abs(d[color_col]))) or 1.0
    s = 20 + 60 * (d["nlog_fdr"] / (d["nlog_fdr"].max() or 1))
    sca = ax.scatter([gi[g] for g in d[group_col]], [fi[f] for f in d[feature_col]],
                     c=d[color_col], s=s, cmap=cmap, vmin=-vmax, vmax=vmax,
                     edgecolor="k", linewidth=0.3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=90, fontsize=7)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=7)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(-0.5, len(feats) - 0.5)
    fig.colorbar(sca, ax=ax, label=color_col, shrink=0.5)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig, ax


def gene_dotplot(adata, group_col, modules=None, standard_scale="var", title=None,
                 categories_order=None, figsize=None):
    """Individual-gene dotplot with genes bracketed by module (sc.pl.dotplot var_group)."""
    modules = TCR_MODULES if modules is None else modules
    var_names = set(adata.var_names)
    panel = {m: [g for g in genes if g in var_names] for m, genes in modules.items()}
    panel = {m: gs for m, gs in panel.items() if gs}
    return sc.pl.dotplot(adata, var_names=panel, groupby=group_col, standard_scale=standard_scale,
                         categories_order=categories_order, title=title, return_fig=True,
                         figsize=figsize)


def module_violin(obs, feature, split_col, facet_col=None, order=None, title=None,
                  figsize=(6, 3.5), kind="violin"):
    """Violin/box of one feature split by `split_col`, optionally faceted by `facet_col`."""
    plot_fn = sns.violinplot if kind == "violin" else sns.boxplot
    if facet_col is None:
        fig, ax = plt.subplots(figsize=figsize)
        plot_fn(data=obs, x=split_col, y=feature, order=order, ax=ax, cut=0 if kind == "violin" else None)
        ax.set_title(title or feature, fontsize=10)
        fig.tight_layout()
        return fig, [ax]
    facets = [f for f in (order or sorted(obs[facet_col].dropna().unique()))]
    fig, axes = plt.subplots(1, len(facets), figsize=(figsize[0] * len(facets) / 2, figsize[1]),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, fac in zip(axes, facets):
        sub = obs[obs[facet_col] == fac]
        plot_fn(data=sub, x=split_col, y=feature, ax=ax, cut=0 if kind == "violin" else None)
        ax.set_title(str(fac), fontsize=9)
        ax.set_xlabel("")
    axes[0].set_ylabel(feature)
    fig.suptitle(title or feature, fontsize=10)
    fig.tight_layout()
    return fig, list(axes)


def signaling_mode_heatmap(contrast_df, contrasts=None, title=None, figsize=(6, 8), cmap="RdBu_r"):
    """Clustered subclone × §2-contrast heatmap. Returns the seaborn ClusterGrid."""
    cols = list(CONTRASTS) if contrasts is None else contrasts
    cols = [c for c in cols if c in contrast_df.columns]
    d = contrast_df[cols].dropna(how="all")
    vmax = float(np.nanmax(np.abs(d.to_numpy()))) or 1.0
    g = sns.clustermap(d.fillna(0.0), cmap=cmap, vmin=-vmax, vmax=vmax, center=0,
                       col_cluster=False, figsize=figsize, yticklabels=True,
                       cbar_kws={"label": "z(num) - z(denom)"})
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=6)
    if title:
        g.fig.suptitle(title, fontsize=10)
    return g


def parse_arm_events(trunk_branch_df, donor_col="donor"):
    """Explode subclone_trunk_branch_v2.csv arm strings into tidy (donor, arm, direction, tier).

    Arm strings like 'chr17q+;chr8q+;chr17p-' in `trunk_arms` / `branch_arms`.
    """
    rows = []
    for _, r in trunk_branch_df.iterrows():
        for tier in ("trunk_arms", "branch_arms"):
            val = r.get(tier)
            if not isinstance(val, str) or not val.strip():
                continue
            for tok in val.split(";"):
                tok = tok.strip()
                if not tok:
                    continue
                direction = tok[-1]
                arm = tok[:-1] if direction in "+-" else tok
                rows.append({donor_col: r[donor_col], "arm": arm, "direction": direction,
                             "tier": tier.replace("_arms", "")})
    return pd.DataFrame(rows)


def arm_activity_box(subclone_means, arm_events, arm, direction, activity_module,
                     score_prefix="sig_", donor_col="donor", title=None, figsize=(4, 3.5)):
    """Box+strip of an activity-module mean in subclones WITH vs WITHOUT a given arm event.

    subclone_means: DataFrame indexed by subclone with a `donor` col and `{score_prefix}{module}` col.
    arm_events: tidy output of parse_arm_events (donor-level events applied to that donor's subclones).
    """
    col = f"{score_prefix}{activity_module}"
    hit_donors = set(arm_events.loc[(arm_events["arm"] == arm) &
                                    (arm_events["direction"] == direction), donor_col])
    d = subclone_means.copy()
    d["has_event"] = np.where(d[donor_col].isin(hit_donors), f"{arm}{direction}", "absent")
    fig, ax = plt.subplots(figsize=figsize)
    order = [f"{arm}{direction}", "absent"]
    sns.boxplot(data=d, x="has_event", y=col, order=order, ax=ax, showfliers=False)
    sns.stripplot(data=d, x="has_event", y=col, order=order, ax=ax, color="k", size=3, alpha=0.6)
    ax.set_title(title or f"{activity_module} vs {arm}{direction}", fontsize=9)
    ax.set_xlabel("")
    fig.tight_layout()
    return fig, ax
