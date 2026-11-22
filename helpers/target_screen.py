"""The five scoring axes of the surface-target screen (nb42 skin, nb43 blood).

One function per axis, each returning a per-gene frame, and `rank_table` to put them side by
side. There is no single composite score by design: a CAR target fails for a different reason
each time -- too rare, not selective, on a vital tissue, never reaches the surface, lost under
pressure -- and collapsing those into one number hides which one applies.

  1 prevalence   detection rate in malignant cells, per donor. A CAR needs the antigen in most
                 patients, not a high cohort mean carried by two deep-sequenced ones.
  2 selectivity  paired within-donor malignant vs reactive. The comparator that matters: every
                 pan-T antigen scores well against "other tissue" and still causes T aplasia.
  3 off_tumour   the worst non-target compartment, in this atlas and in healthy references,
                 plus the healthy-T burden stated separately as the aplasia cost.
  4 surface      the trafficking context: can this cell put the protein on the surface and keep
                 it there. A tier, not a correction factor -- see surface_targets.
  5 escape       heterogeneity within a donor (across the 26b/32b clusters) and between donors.

Everything is donor-level. Cells inside a donor are not independent, and a p-value computed over
cells is a statement about sequencing depth, not biology -- the project rule from
`degradome_data.CAVEATS["cell_level_pvalues"]`, applied here too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse, stats
from statsmodels.stats.multitest import multipletests

MIN_CELLS = 20          # a donor's arm must have this many cells to contribute a value
MIN_DONORS = 5          # a gene must have this many donors per arm to be tested
DETECT_FLOOR = 0.10     # "expressed" for prevalence purposes


# ---------------------------------------------------------------- helpers
def _dense(X):
    return np.asarray(X.todense()) if sparse.issparse(X) else np.asarray(X)


def per_donor(adata, genes, arm_col, donor_col, layer=None, min_cells=MIN_CELLS):
    """(arm, donor, gene) detection rate and mean log-expression.

    Returns a long frame: arm, donor, gene, pct_expr, mean_all, n_cells. `mean_all` includes the
    zeros -- that is the quantity a bulk measurement would see, and the one to compare across
    arms; `pct_expr` is the one that matters for a CAR, where what counts is the share of tumour
    cells carrying the antigen at all.
    """
    genes = [g for g in genes if g in set(map(str, adata.var_names))]
    sub = adata[:, genes]
    X = _dense(sub.layers[layer] if layer else sub.X)
    obs = adata.obs
    key = pd.MultiIndex.from_arrays([obs[arm_col].astype(str), obs[donor_col].astype(str)])

    rows = []
    for (arm, donor), idx in pd.Series(np.arange(len(obs)), index=key).groupby(level=[0, 1]):
        i = idx.to_numpy()
        if len(i) < min_cells:
            continue
        blk = X[i]
        rows.append(pd.DataFrame({
            "arm": arm, "donor": donor, "gene": genes,
            "pct_expr": (blk > 0).mean(0), "mean_all": blk.mean(0), "n_cells": len(i),
        }))
    if not rows:
        return pd.DataFrame(columns=["arm", "donor", "gene", "pct_expr", "mean_all", "n_cells"])
    return pd.concat(rows, ignore_index=True)


def from_cube(cube, meta, arm_col="level", donor_col="donor"):
    """Same long shape, but from a pre-built pseudobulk cube (ccc_utils.build_pseudobulk_cube)."""
    out = cube.rename(columns={arm_col: "arm", donor_col: "donor"})
    need = {"arm", "donor", "gene"}
    missing = need - set(out.columns)
    if missing:
        raise KeyError(f"cube is missing {sorted(missing)}")
    return out


# ---------------------------------------------------------------- 1. prevalence
def prevalence(long, arm, detect_floor=DETECT_FLOOR):
    """Per gene: how much of the tumour carries it, and in how many patients.

    `frac_donors_positive` is the headline. A target present at 90% in a third of patients is a
    biomarker-selected therapy; one at 40% in every patient is a different clinical product. The
    two are not ranked by the same number, so both are returned.
    """
    a = long[long["arm"].eq(arm)]
    if a.empty:
        return pd.DataFrame(columns=["gene", "median_pct_expr", "frac_donors_positive", "n_donors"])
    g = a.groupby("gene")
    out = pd.DataFrame({
        "median_pct_expr": g["pct_expr"].median(),
        "mean_pct_expr": g["pct_expr"].mean(),
        "median_mean_all": g["mean_all"].median(),
        "frac_donors_positive": a.assign(_p=a["pct_expr"] >= detect_floor)
                                 .groupby("gene")["_p"].mean(),
        "n_donors": g["donor"].nunique(),
    }).reset_index()
    return out.sort_values("frac_donors_positive", ascending=False)


# ---------------------------------------------------------------- 2. selectivity
def paired_selectivity(long, arm_pos, arm_neg, value="mean_all", min_donors=MIN_DONORS,
                       pseudo=0.05):
    """Within-donor malignant vs reactive: paired log2FC, Wilcoxon signed-rank, BH over genes.

    Paired because both arms come from the same biopsy and the same library prep -- the donor is
    the block, and an unpaired test across donors would spend most of its power on between-patient
    variation that the pairing removes.
    """
    w = (long[long["arm"].isin([arm_pos, arm_neg])]
         .pivot_table(index=["gene", "donor"], columns="arm", values=value))
    w = w.dropna(subset=[c for c in (arm_pos, arm_neg) if c in w.columns])
    if arm_pos not in w.columns or arm_neg not in w.columns:
        return pd.DataFrame(columns=["gene", "log2fc_paired", "n_donors_paired", "p", "q"])

    w["lfc"] = np.log2((w[arm_pos] + pseudo) / (w[arm_neg] + pseudo))
    rows = []
    for gene, d in w.groupby(level="gene"):
        n = len(d)
        if n < min_donors:
            continue
        diff = (d[arm_pos] - d[arm_neg]).to_numpy()
        p = (stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0)
        rows.append({"gene": gene, "log2fc_paired": float(d["lfc"].median()),
                     "n_donors_paired": int(n),
                     "frac_donors_up": float((diff > 0).mean()), "p": float(p)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q"] = multipletests(out["p"], method="fdr_bh")[1]
    return out.sort_values("log2fc_paired", ascending=False)


def blocked_selectivity(long, arm_pos, arm_neg, block, value="mean_all", min_per_arm=3):
    """Unpaired fallback, stratified by `block` (a donor->study map): van Elteren / stratified rank.

    For the donors that carry only one arm -- common in blood, where a patient can be almost
    entirely malignant -- the paired test has nothing to pair. Reported next to the paired result,
    never instead of it.
    """
    d = long[long["arm"].isin([arm_pos, arm_neg])].copy()
    d["block"] = d["donor"].map(block).fillna("NA")
    rows = []
    for gene, g in d.groupby("gene"):
        zs, ws = [], []
        for blk, gb in g.groupby("block"):
            x = gb.loc[gb["arm"].eq(arm_pos), value].to_numpy()
            y = gb.loc[gb["arm"].eq(arm_neg), value].to_numpy()
            if len(x) < min_per_arm or len(y) < min_per_arm:
                continue
            u = stats.mannwhitneyu(x, y, alternative="two-sided")
            n1, n2 = len(x), len(y)
            mu, sd = n1 * n2 / 2, np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
            zs.append((u.statistic - mu) / sd if sd > 0 else 0.0)
            ws.append(np.sqrt(n1 * n2))
        if not zs:
            continue
        z = float(np.sum(np.array(zs) * np.array(ws)) / np.sqrt(np.sum(np.array(ws) ** 2)))
        rows.append({"gene": gene, "z_blocked": z, "n_blocks": len(zs),
                     "p_blocked": float(2 * stats.norm.sf(abs(z)))})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_blocked"] = multipletests(out["p_blocked"], method="fdr_bh")[1]
    return out


# ---------------------------------------------------------------- 3. on-target off-tumour
def off_tumour(long, target_arms, healthy_t_arms=(), value="pct_expr"):
    """The worst non-target compartment per gene, and the healthy-T burden kept separate.

    Two different costs that must not be averaged. Expression on keratinocytes or endothelium is
    a tissue-toxicity risk; expression on healthy T cells is T-cell aplasia, which for CTCL is
    often accepted (CD7, CD5 and CD4 agents all cause it) but has to be stated, not hidden.
    `worst_offtumour_arm` names the compartment, because "which one" decides whether a candidate
    is dead or merely expensive.
    """
    target_arms, healthy_t_arms = set(target_arms), set(healthy_t_arms)
    other = long[~long["arm"].isin(target_arms | healthy_t_arms)]
    med = other.groupby(["gene", "arm"])[value].median().reset_index()
    if med.empty:
        worst = pd.DataFrame(columns=["gene", "worst_offtumour", "worst_offtumour_arm"])
    else:
        idx = med.groupby("gene")[value].idxmax()
        worst = (med.loc[idx].rename(columns={value: "worst_offtumour", "arm": "worst_offtumour_arm"})
                 .reset_index(drop=True))

    if healthy_t_arms:
        ht = (long[long["arm"].isin(healthy_t_arms)].groupby("gene")[value].median()
              .rename("healthy_t_burden").reset_index())
        worst = worst.merge(ht, on="gene", how="outer")
    return worst


# ---------------------------------------------------------------- 4. surface plausibility
def traffic_scores(adata, modules, arm_col, donor_col, layer=None, min_cells=MIN_CELLS):
    """Per (arm, donor) mean z-score of each trafficking module, genes present only.

    Not a per-gene correction. It answers "is this arm's secretory and recycling machinery in a
    different state from the comparator", which is the context in which a transcript-to-surface
    inference is or is not safe.
    """
    import scanpy as sc

    a = adata.copy()
    if layer:
        a.X = a.layers[layer].copy()
    present = set(map(str, a.var_names))
    cols = []
    for mod, genes in modules.items():
        g = [x for x in genes if x in present]
        if len(g) < 3:
            continue
        sc.tl.score_genes(a, g, score_name=f"traffic_{mod}", random_state=0)
        cols.append(f"traffic_{mod}")
    out = (a.obs.groupby([arm_col, donor_col], observed=True)[cols].mean()
           .join(a.obs.groupby([arm_col, donor_col], observed=True).size().rename("n_cells")))
    return out[out["n_cells"] >= min_cells].reset_index()


def surface_tier(gene, caveats, traffic_delta=None):
    """A three-way label for how far a transcript rank can be trusted for this gene.

    `direct`    no known post-transcriptional sink; the transcript is the best available proxy.
    `caveated`  a documented internalisation, shedding or lattice mechanism applies -- the
                caveat text says which.
    `unknown`   no information either way, which is the honest default and the common case.

    None of the three is a measurement. Every one of them needs flow or CITE-seq before a target
    is called.
    """
    if gene in caveats:
        return "caveated"
    if traffic_delta is not None and abs(traffic_delta) > 0.25:
        return "caveated"
    return "unknown"


# ---------------------------------------------------------------- 5. escape risk
def escape_risk(long, arm, cluster_long=None):
    """Between-donor spread, and within-donor spread across clusters when it is available.

    An antigen whose detection rate swings between subclones of one patient is the one that
    selects an antigen-negative escape clone under therapy -- the failure mode of every single-
    antigen CAR in a heterogeneous tumour.
    """
    a = long[long["arm"].eq(arm)]
    out = (a.groupby("gene")["pct_expr"].agg(between_donor_sd="std", min_donor_pct="min")
           .reset_index())
    if cluster_long is not None and not cluster_long.empty:
        c = cluster_long[cluster_long["arm"].eq(arm)]
        within = (c.groupby(["gene", "donor"])["pct_expr"].std()
                  .groupby("gene").median().rename("within_donor_sd").reset_index())
        out = out.merge(within, on="gene", how="left")
    return out


# ---------------------------------------------------------------- assembly
def rank_table(prevalence_df, selectivity_df, offtumour_df, escape_df, annotate_fn,
               surfaceome=None):
    """Put the axes side by side. No cutoff is applied and no composite score is computed.

    Sorted by selectivity then prevalence purely so the table opens somewhere sensible; the sort
    is not a ranking claim. Filtering is the reader's decision and belongs in the notebook, next
    to the sentence that justifies the threshold.
    """
    df = prevalence_df.merge(selectivity_df, on="gene", how="outer")
    for part in (offtumour_df, escape_df):
        if part is not None and not part.empty:
            df = df.merge(part, on="gene", how="left")
    df = df.merge(annotate_fn(df["gene"]), on="gene", how="left")
    if surfaceome is not None:
        keep = ["gene", "n_evidence", "surface_confident", "in_cspa", "cspa_high_confidence",
                "surfy_curated", "topology", "n_tm", "surfy_class"]
        df = df.merge(surfaceome[[c for c in keep if c in surfaceome.columns]],
                      on="gene", how="left")
    sort_cols = [c for c in ("log2fc_paired", "frac_donors_positive") if c in df.columns]
    return df.sort_values(sort_cols, ascending=False).reset_index(drop=True)


def control_check(df, controls, by="log2fc_paired"):
    """Where the clinically validated antigens landed. Run this before reading any new hit.

    CCR4, TNFRSF8, KIR3DL2 and IL2RA are approved or registrational in CTCL. If they are not near
    the top, the screen is miscalibrated and the novel hits below them are not evidence of
    anything.
    """
    d = df.dropna(subset=[by]).reset_index(drop=True)
    d["rank"] = d[by].rank(ascending=False, method="min").astype(int)
    hit = d[d["gene"].isin(controls)][["gene", "rank", by, "frac_donors_positive"]]
    missing = sorted(set(controls) - set(d["gene"]))
    return hit.sort_values("rank"), missing, len(d)


# ---------------------------------------------------------------- stratified selectivity
def strat_selectivity(long, arm_pos, arm_neg, groups, value="mean_all", min_donors=5,
                      pseudo=0.05, interaction=True):
    """`paired_selectivity` run inside each stratum, plus a test that the strata differ.

    `groups` maps donor -> stratum label (a dict or a Series). A stratum with fewer than
    `min_donors` paired donors is still reported, with its n, and simply carries no p: the point
    of a stratified table in this cohort is as much to show where there is no power as to show
    an effect, and silently dropping those rows is how an underpowered contrast gets read as a
    negative result.

    The interaction is a Mann-Whitney of the per-donor log2FC between the two largest strata --
    an honest two-sample comparison of effect sizes, not a model. With 8 MF against 40 SS it is
    a consistency check, and the returned `n_a`/`n_b` are there to be quoted beside it.
    """
    gmap = pd.Series(groups) if not isinstance(groups, pd.Series) else groups
    w = (long[long["arm"].isin([arm_pos, arm_neg])]
         .pivot_table(index=["gene", "donor"], columns="arm", values=value))
    w = w.dropna(subset=[c for c in (arm_pos, arm_neg) if c in w.columns])
    if arm_pos not in w.columns or arm_neg not in w.columns:
        return pd.DataFrame(columns=["gene", "stratum", "log2fc_paired", "n_donors_paired"])
    w["lfc"] = np.log2((w[arm_pos] + pseudo) / (w[arm_neg] + pseudo))
    w["stratum"] = w.index.get_level_values("donor").map(gmap).astype(object)
    w = w[w["stratum"].notna()]

    rows = []
    for (gene, stratum), d in w.groupby([w.index.get_level_values("gene"), "stratum"]):
        diff = (d[arm_pos] - d[arm_neg]).to_numpy()
        n = len(d)
        p = (float(stats.wilcoxon(diff).pvalue)
             if n >= min_donors and np.any(diff != 0) else np.nan)
        rows.append({"gene": gene, "stratum": stratum, "n_donors_paired": int(n),
                     "log2fc_paired": float(d["lfc"].median()),
                     "frac_donors_up": float((diff > 0).mean()), "p": p})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    m = out["p"].notna()
    out["q"] = np.nan
    if m.any():
        out.loc[m, "q"] = multipletests(out.loc[m, "p"], method="fdr_bh")[1]

    if not interaction:
        return out.sort_values(["gene", "stratum"])

    top = out.groupby("stratum")["n_donors_paired"].median().nlargest(2).index.tolist()
    if len(top) == 2:
        a, b = top
        inter = []
        for gene, d in w.groupby(level="gene"):
            xa = d.loc[d["stratum"].eq(a), "lfc"].to_numpy()
            xb = d.loc[d["stratum"].eq(b), "lfc"].to_numpy()
            if len(xa) < 3 or len(xb) < 3:
                continue
            inter.append({"gene": gene, "strata": f"{a} vs {b}",
                          "n_a": len(xa), "n_b": len(xb),
                          "delta_lfc": float(np.median(xa) - np.median(xb)),
                          "p_interaction": float(
                              stats.mannwhitneyu(xa, xb, alternative="two-sided").pvalue)})
        inter = pd.DataFrame(inter)
        if not inter.empty:
            inter["q_interaction"] = multipletests(inter["p_interaction"],
                                                   method="fdr_bh")[1]
            out = out.merge(inter, on="gene", how="left")
    return out.sort_values(["gene", "stratum"])
