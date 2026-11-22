"""Single-cell analysis layer for the degradome notebook (28).

The unit of analysis is the cell. That is a deliberate choice and it carries one obligation:
cells within a donor are not independent, so nothing here computes an unstratified p-value over
cells. Every test is blocked on study (`blocked_ranksum`, a van Elteren statistic) and every
effect is reported next to the median per-donor difference, so a result carried by one
deep-sequenced patient is visible rather than significant.

Why not `sc.tl.score_genes` for the family scores. It draws its control set from
expression-matched bins of the same AnnData. On a ~290-column hand-curated matrix panel the
control pool *is* the panel, so the background subtracted from the MMP score is the
cathepsin/collagen score. The number is uninterpretable and the artefact is invisible. Scoring
against the full 42,347-gene object instead would need a second streaming pass of the 112 GB
source for eight numbers. So a family score here is the plain mean of the family's genes on
`X` -- log1p CP10K with the size factor taken over all 42,347 genes *before* the column subset,
which is what makes it comparable across samples of very different depth
(`degradome_helpers.assert_build_equivalence` gates exactly that).

Small families are a trap: a 3-gene family is a spike at zero with a tail, not a distribution,
so `family_score_cells` refuses to score below `cfg.MIN_FAMILY_SIZE_FOR_SCORE` and the notebook
plots those per gene instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import degradome_data as cfg
from degradome_helpers import _finish, _level_colors  # noqa: F401  (shared palette + save)


# ================================================================ scores


def family_score_cells(adata, families=None, prefix="fam_", det_prefix="det_",
                       min_size=None, verbose=True):
    """Per-cell mean expression over each family's genes, plus the number detected.

    Adds `fam_<key>` (mean of `X` over the family) and `det_<key>` (count of the family's genes
    with a nonzero count in that cell) to `adata.obs`. Returns the list of families scored;
    families below `min_size` are skipped and named, because a family mean over three genes is
    not a distribution and a violin of one lies.

    `families` is either a list of keys into `cfg.DEGRADOME_FAMILIES` or a `{name: genes}` dict.
    The dict form is what `cfg.FIBROSIS_AXES` needs: those axes overlap the degradome families
    by 44 symbols and are kept out of `DEGRADOME_FAMILIES` so `GENE_TO_FAMILY` stays flat.
    """
    import scipy.sparse as sp

    if families is None:
        families = cfg.DEGRADOME_FAMILIES
    elif not isinstance(families, dict):
        families = {k: cfg.DEGRADOME_FAMILIES[k] for k in families}
    min_size = cfg.MIN_FAMILY_SIZE_FOR_SCORE if min_size is None else min_size

    var = pd.Index(map(str, adata.var_names))
    X = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    C = adata.layers.get(cfg.COUNTS_LAYER)
    C = (C.tocsr() if sp.issparse(C) else sp.csr_matrix(C)) if C is not None else X

    scored, skipped = [], {}
    for fam, genes in families.items():
        idx = var.get_indexer([g for g in genes if g in set(var)])
        idx = idx[idx >= 0]
        if len(idx) < min_size:
            skipped[fam] = len(idx)
            continue
        adata.obs[f"{prefix}{fam}"] = np.asarray(X[:, idx].mean(axis=1)).ravel()
        adata.obs[f"{det_prefix}{fam}"] = np.asarray((C[:, idx] > 0).sum(axis=1)).ravel()
        scored.append(fam)

    if verbose:
        print(f"scored {len(scored)} families per cell")
        if skipped:
            print(f"  too small to score (<{min_size} genes), plot per gene instead: {skipped}")
    return scored


def balance_cells(adata, pairs=None, prefix="bal_", verbose=True):
    """Per-cell log2((protease mean + eps) / (inhibitor mean + eps)) for each balance pair.

    Both sides are means on the same normalised layer, so the ratio is a within-cell contrast
    and does not move with library size at all. `remodel_tone` is the one that carries the
    question: degradative enzymes over the structural matrix they act on.
    """
    import scipy.sparse as sp

    pairs = cfg.BALANCE_PAIRS if pairs is None else pairs
    var = pd.Index(map(str, adata.var_names))
    X = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    eps = 0.01
    out = []
    for name, (num, den) in pairs.items():
        ni = var.get_indexer([g for g in cfg.family_genes(num) if g in set(var)])
        di = var.get_indexer([g for g in den if g in set(var)])
        ni, di = ni[ni >= 0], di[di >= 0]
        n = np.asarray(X[:, ni].mean(axis=1)).ravel()
        d = np.asarray(X[:, di].mean(axis=1)).ravel()
        adata.obs[f"{prefix}{name}"] = np.log2((n + eps) / (d + eps))
        out.append(name)
    if verbose:
        print(f"balance ratios: {out}")
    return out


def gene_score_cells(adata, genes=None, prefix="gene_", verbose=True):
    """Per-cell expression of each gene, copied onto `adata.obs` as `<prefix><gene>`.

    The per-donor and per-sample machinery in this module (`donor_median`, `paired_within`,
    `compare_levels`) takes score *columns* on `obs`, not genes on `X`. A family mean is such a
    column; a single gene is not, so it has to be materialised. That is all this does -- no
    scaling, no scoring, no control set: the value is `X` as built, log1p CP10K with the size
    factor over all 42,347 genes before the panel subset.

    Genes absent from `var` are skipped and named. Returns the obs column names, in the order
    the genes were given, so the caller can hand them straight to `donor_median`.
    """
    import scipy.sparse as sp

    genes = list(cfg.panel_genes()) if genes is None else list(dict.fromkeys(genes))
    var = pd.Index(map(str, adata.var_names))
    have = [g for g in genes if g in var]
    missing = [g for g in genes if g not in var]
    if not have:
        raise ValueError("none of the requested genes are on the object")
    X = adata[:, have].X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    for j, g in enumerate(have):
        adata.obs[f"{prefix}{g}"] = X[:, j]
    if verbose:
        print(f"gene columns: {len(have)} on obs as {prefix}<gene>"
              + (f"; absent from var: {missing}" if missing else ""))
    return [f"{prefix}{g}" for g in have]


# ================================================================ donor medians


def donor_median(adata, score_cols, groupby=None, donor_key=None, min_cells=None,
                 extra=("study", "stage_group", "disease")):
    """Median score per (level, donor), restricted to pairs that clear `min_cells`.

    This is the replicate unit. Cells give the distribution a violin draws; donors give the
    number a comparison is allowed to rest on, and the gate is what stops a level with two
    patients looking like a result.
    """
    groupby = cfg.GROUPBY if groupby is None else groupby
    donor_key = cfg.DONOR_KEY if donor_key is None else donor_key
    min_cells = cfg.MIN_CELLS if min_cells is None else min_cells

    score_cols = list(score_cols)
    keep = [groupby, donor_key] + [c for c in extra if c in adata.obs.columns]
    df = adata.obs[keep + score_cols].copy()
    df[groupby] = df[groupby].astype(str)
    df[donor_key] = df[donor_key].astype(str)

    g = df.groupby([groupby, donor_key], observed=True)
    med = g[score_cols].median()
    med["n_cells"] = g.size()
    meta = g[[c for c in extra if c in df.columns]].first()
    out = med.join(meta).reset_index()
    out["eligible"] = out["n_cells"] >= min_cells
    return out


# ================================================================ tests


def _ranksum_stratum(x, y):
    """Mann-Whitney W for `x` against `y`, with its tie-corrected mean and variance."""
    n1, n2 = len(x), len(y)
    N = n1 + n2
    allv = np.concatenate([x, y])
    r = pd.Series(allv).rank().to_numpy()
    W = r[:n1].sum()
    EW = n1 * (N + 1) / 2.0
    _, counts = np.unique(allv, return_counts=True)
    tie = float((counts ** 3 - counts).sum())
    VW = n1 * n2 * (N + 1) / 12.0
    if N > 1 and (N ** 3 - N) > 0:
        VW *= (1.0 - tie / (N ** 3 - N))
    return W, EW, max(VW, 0.0), n1, n2


def blocked_ranksum(df, value, group, ref, alt, block="study", min_per_arm=5):
    """van Elteren test: Wilcoxon rank-sum stratified by `block`.

    A pooled rank-sum over cells from 12 studies tests "does study composition differ", not
    "does the level differ". Stratifying removes the between-study shift by construction and
    weights each stratum by 1/(N+1), which is the design-free weighting -- it does not let one
    huge cohort decide the answer.

    Returns a dict with the z statistic, a two-sided p, the per-stratum arm sizes and the
    number of strata that contributed. Ties are corrected for; at these zero-inflated
    distributions that matters.
    """
    from scipy import stats

    d = df[[value, group, block]].dropna()
    d = d[d[group].astype(str).isin([str(ref), str(alt)])]
    num, den, strata, n_ref, n_alt = 0.0, 0.0, 0, 0, 0
    for _, s in d.groupby(block, observed=True):
        x = s.loc[s[group].astype(str) == str(alt), value].to_numpy()
        y = s.loc[s[group].astype(str) == str(ref), value].to_numpy()
        if len(x) < min_per_arm or len(y) < min_per_arm:
            continue
        W, EW, VW, n1, n2 = _ranksum_stratum(x, y)
        w = 1.0 / (n1 + n2 + 1.0)
        num += w * (W - EW)
        den += (w ** 2) * VW
        strata += 1
        n_ref += n2
        n_alt += n1
    if strata == 0 or den <= 0:
        return {"z": np.nan, "p": np.nan, "n_strata": strata,
                "n_ref": n_ref, "n_alt": n_alt}
    z = num / np.sqrt(den)
    return {"z": float(z), "p": float(2 * stats.norm.sf(abs(z))), "n_strata": int(strata),
            "n_ref": int(n_ref), "n_alt": int(n_alt)}


def compare_levels(adata, dmed, score_cols, group, ref, alt, levels=None,
                   block="study", min_donors=None, fdr=True):
    """For each level x score: the study-blocked cell-level test and the donor-level effect.

    Two numbers, always together. The p-value comes from cells (blocked on study); the effect
    size is the median of the per-donor medians in each arm and their difference. When they
    disagree -- a tiny p on a delta of 0.01 -- the effect is the honest one, and the pair is
    printed so that disagreement is visible instead of buried.

    Abstains (returns NaN, with the reason) when either arm has fewer than `min_donors`
    eligible donors, rather than testing a level this cohort cannot support.
    """
    from statsmodels.stats.multitest import multipletests

    min_donors = cfg.MIN_DONORS_PER_ARM if min_donors is None else min_donors
    groupby = cfg.GROUPBY
    obs = adata.obs
    levels = ([lv for lv in cfg.CT_ORDER if lv in set(obs[groupby].astype(str))]
              if levels is None else list(levels))

    rows = []
    for lv in levels:
        cells = obs[obs[groupby].astype(str) == lv]
        dm = dmed[(dmed[groupby] == lv) & dmed["eligible"]]
        n_ref_d = int((dm[group].astype(str) == str(ref)).sum())
        n_alt_d = int((dm[group].astype(str) == str(alt)).sum())
        testable = n_ref_d >= min_donors and n_alt_d >= min_donors
        for sc in score_cols:
            rec = {"level": lv, "score": sc, "axis": f"{ref}_vs_{alt}",
                   "n_donors_ref": n_ref_d, "n_donors_alt": n_alt_d, "testable": testable}
            a = dm.loc[dm[group].astype(str) == str(alt), sc]
            b = dm.loc[dm[group].astype(str) == str(ref), sc]
            rec["donor_median_ref"] = float(b.median()) if len(b) else np.nan
            rec["donor_median_alt"] = float(a.median()) if len(a) else np.nan
            rec["donor_delta"] = rec["donor_median_alt"] - rec["donor_median_ref"]
            if testable:
                rec.update(blocked_ranksum(cells, sc, group, ref, alt, block=block))
            else:
                rec.update({"z": np.nan, "p": np.nan, "n_strata": 0,
                            "n_ref": 0, "n_alt": 0})
                rec["reason"] = (f"{'ref' if n_ref_d < min_donors else 'alt'} arm has "
                                 f"{min(n_ref_d, n_alt_d)} eligible donors "
                                 f"(< {min_donors})")
            rows.append(rec)

    res = pd.DataFrame(rows)
    res["q"] = np.nan
    if fdr:
        m = res["p"].notna()
        if m.any():
            res.loc[m, "q"] = multipletests(res.loc[m, "p"], method="fdr_bh")[1]
    return res


def paired_within(dmed, score_cols, pair_key, arm_col, ref, alt, level=None,
                  level_col=None, fdr=True):
    """Wilcoxon signed-rank over paired (ref, alt) medians within the same `pair_key`.

    `compare_levels(block="sample_id")` already gives a within-sample test at the cell level.
    This is its replicate-level partner: one delta per sample that carries BOTH arms above
    `cfg.MIN_CELLS`, then a signed-rank over those deltas. It is the design that holds donor,
    study, chemistry, biopsy site and treatment fixed by construction -- the same patient, the
    same tube -- so it is the one number in this notebook that no study confound can reach.

    The price is n: a sample only contributes if both arms clear the cell gate, and `n_pairs`
    is reported next to every p precisely because it is usually much smaller than the donor
    count a pooled test would quote. A sign test runs alongside, because at small n the
    signed-rank's normal approximation is the weaker of the two.

    Returns one row per score: n_pairs, median_delta, w, p, n_pos, n_neg, sign_p, q.
    """
    from scipy import stats

    d = dmed[dmed["eligible"]].copy()
    if level is not None:
        level_col = cfg.GROUPBY if level_col is None else level_col
        d = d[d[level_col].astype(str) == str(level)]
    d[arm_col] = d[arm_col].astype(str)
    d = d[d[arm_col].isin([str(ref), str(alt)])]

    rows = []
    for sc in score_cols:
        w = (d.pivot_table(index=pair_key, columns=arm_col, values=sc, aggfunc="median")
             .dropna(subset=[str(ref), str(alt)]))
        delta = (w[str(alt)] - w[str(ref)]).to_numpy()
        rec = {"score": sc, "n_pairs": int(len(delta)),
               "median_delta": float(np.median(delta)) if len(delta) else np.nan,
               "n_pos": int((delta > 0).sum()), "n_neg": int((delta < 0).sum())}
        # wilcoxon needs at least one nonzero difference and refuses n < 1
        nz = delta[delta != 0]
        if len(nz) >= cfg.MIN_DONORS_PER_ARM:
            st = stats.wilcoxon(nz, alternative="two-sided")
            rec["w"], rec["p"] = float(st.statistic), float(st.pvalue)
            rec["sign_p"] = float(stats.binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue)
            rec["reason"] = ""
        else:
            rec["w"] = rec["p"] = rec["sign_p"] = np.nan
            rec["reason"] = (f"{len(nz)} sample(s) with a nonzero paired delta "
                             f"(< {cfg.MIN_DONORS_PER_ARM})")
        rows.append(rec)

    res = pd.DataFrame(rows)
    res["q"] = np.nan
    if fdr:
        from statsmodels.stats.multitest import multipletests
        m = res["p"].notna()
        if m.any():
            res.loc[m, "q"] = multipletests(res.loc[m, "p"], method="fdr_bh")[1]
    return res


# ================================================================ plots


def _donor_dots(ax, positions, dmed, level_col, levels, value, group=None, arm=None,
                color="#222222", seed=0):
    """Overlay one dot per eligible donor, jittered. The n a comparison actually has."""
    rng = np.random.default_rng(seed)
    for pos, lv in zip(positions, levels):
        d = dmed[(dmed[level_col] == lv) & dmed["eligible"]]
        if group is not None and arm is not None:
            d = d[d[group].astype(str) == str(arm)]
        v = d[value].dropna().to_numpy()
        if not len(v):
            continue
        ax.scatter(pos + rng.uniform(-0.09, 0.09, len(v)), v, s=7, zorder=3,
                   color=color, edgecolor="white", linewidth=0.3, alpha=0.9)


EXT_COLOR = "#7b4173"


def plot_external_band(ax, values, label=None, color=EXT_COLOR, xspan=None):
    """Draw the external healthy reference as a band, not as another violin.

    A band and not a box because the two sides are not exchangeable: the external donors come
    from one lab, one chemistry and one sex, so the spread being drawn is that cohort's spread,
    not a second sample of the same population. Drawing it behind our distributions says
    "here is where healthy skin sat in another study" -- which is what it can support -- rather
    than inviting the reader to read the gap as an effect size.

    `values` is one number per external donor (its median for this level x score).
    """
    v = np.asarray([x for x in np.ravel(values) if np.isfinite(x)], dtype=float)
    if not len(v):
        return ax
    lo, med, hi = np.percentile(v, [25, 50, 75])
    x0, x1 = (ax.get_xlim() if xspan is None else xspan)
    ax.axhspan(lo, hi, xmin=0, xmax=1, color=color, alpha=0.12, zorder=0, linewidth=0)
    ax.axhline(med, color=color, linewidth=1.0, linestyle=(0, (4, 2)), zorder=1)
    ax.scatter(np.full(len(v), x1), v, s=7, color=color, edgecolor="white", linewidth=0.3,
               zorder=3, clip_on=False)
    if label:
        ax.annotate(label, (x1, med), xytext=(4, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=5, color=color, annotation_clip=False)
    ax.set_xlim(x0, x1)
    return ax


def plot_paired_slopes(dmed, score, pair_key, arm_col, ref, alt, level=None, level_col=None,
                       title=None, ylabel=None, save=None, show=True, figsize=(5, 3),
                       color_up="#c51b7d", color_dn="#4d9221"):
    """One line per sample: its `ref` median to its `alt` median, both from the same biopsy.

    The figure for `paired_within`. Every line is a within-sample delta, so the reader counts
    lines rather than trusting a p, and a result carried by two samples cannot look like a
    trend.
    """
    import matplotlib.pyplot as plt

    d = dmed[dmed["eligible"]].copy()
    if level is not None:
        level_col = cfg.GROUPBY if level_col is None else level_col
        d = d[d[level_col].astype(str) == str(level)]
    d[arm_col] = d[arm_col].astype(str)
    w = (d.pivot_table(index=pair_key, columns=arm_col, values=score, aggfunc="median")
         .dropna(subset=[str(ref), str(alt)]))

    fig, ax = plt.subplots(figsize=figsize)
    for _, r in w.iterrows():
        a, b = float(r[str(ref)]), float(r[str(alt)])
        ax.plot([0, 1], [a, b], linewidth=0.8, alpha=0.75, zorder=2,
                color=(color_up if b > a else color_dn))
    ax.scatter(np.zeros(len(w)), w[str(ref)], s=9, color="#444444", zorder=3)
    ax.scatter(np.ones(len(w)), w[str(alt)], s=9, color="#444444", zorder=3)
    for x, arm in ((0, ref), (1, alt)):
        vals = w[str(arm)].to_numpy()
        if len(vals):
            ax.hlines(np.median(vals), x - 0.12, x + 0.12, color="#111111", linewidth=1.6,
                      zorder=4)

    n_up = int((w[str(alt)] > w[str(ref)]).sum())
    ax.set_xticks([0, 1])
    ax.set_xticklabels([str(ref), str(alt)], fontsize=8)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylabel(ylabel or score, fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title(title or f"{score}: up in {alt} in {n_up}/{len(w)} paired samples", fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return _finish(fig, save, show)


def plot_family_violin(adata, dmed, score, levels=None, title=None, ylabel=None,
                       save=None, show=True, ax=None, figsize=(9, 3.0), pct_detect=None):
    """Per-cell violin per level, with one dot per eligible donor on top.

    The violin is the cell distribution and is descriptive only; the dots are the replicate
    unit. Drawing both is the point -- a wide violin over three donors should not read as
    evidence, and side by side it does not.
    """
    import matplotlib.pyplot as plt

    groupby = cfg.GROUPBY
    obs = adata.obs
    levels = ([lv for lv in cfg.CT_ORDER if lv in set(obs[groupby].astype(str))]
              if levels is None else list(levels))
    data = [obs.loc[obs[groupby].astype(str) == lv, score].dropna().to_numpy()
            for lv in levels]
    colors = _level_colors(levels)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    pos = np.arange(len(levels))
    parts = ax.violinplot([d if len(d) else [0.0] for d in data], positions=pos,
                          showextrema=False, widths=0.85)
    for b, lv in zip(parts["bodies"], levels):
        b.set_facecolor(colors[lv])
        b.set_alpha(0.55 if lv in cfg.CLAIM_LEVELS else 0.25)
        b.set_edgecolor("none")
    for p, d in zip(pos, data):
        if len(d):
            ax.hlines(np.median(d), p - 0.28, p + 0.28, color="#111111", linewidth=1.1,
                      zorder=4)
    _donor_dots(ax, pos, dmed, groupby, levels, score)

    ax.set_xticks(pos)
    ax.set_xticklabels(levels, rotation=90, fontsize=7)
    ax.set_ylabel(ylabel or score, fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title(title or score, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if pct_detect is not None:
        for p, lv in zip(pos, levels):
            v = pct_detect.get(lv, np.nan)
            if np.isfinite(v):
                ax.annotate(f"{v:.0%}", (p, ax.get_ylim()[0]), xytext=(0, -1),
                            textcoords="offset points", ha="center", va="top",
                            fontsize=5, color="#666666", annotation_clip=False)
    return _finish(fig, save, show) if fig is not None else ax


def plot_stage_grid(adata, dmed, scores, levels, group="stage_group", arms=None,
                    cov=None, save=None, show=True, panel=(2.0, 1.9), ylabel_fmt="{}"):
    """scores x levels grid of per-arm violins, donor dots on top.

    A panel whose arm fails the donor gate is hatched and labelled "abstain (n=...)" rather
    than left blank: an empty axis reads as "no effect", and "we could not test this" is a
    different statement.
    """
    import matplotlib.pyplot as plt

    arms = cfg.STAGE_ARMS if arms is None else list(arms)
    groupby = cfg.GROUPBY
    obs = adata.obs
    nr, nc = len(scores), len(levels)
    fig, axes = plt.subplots(nr, nc, figsize=(panel[0] * nc, panel[1] * nr),
                             squeeze=False, sharex=True)
    arm_col = {"HC": "#4c9f70", "early": "#e8a33d", "advanced": "#c0392b"}

    for i, sc in enumerate(scores):
        for j, lv in enumerate(levels):
            ax = axes[i][j]
            sub = obs[obs[groupby].astype(str) == lv]
            data, ok = [], []
            for arm in arms:
                v = sub.loc[sub[group].astype(str) == arm, sc].dropna().to_numpy()
                nd = int(((dmed[groupby] == lv) & dmed["eligible"] &
                          (dmed[group].astype(str) == arm)).sum())
                data.append(v)
                ok.append(nd >= cfg.MIN_DONORS_PER_ARM)
            pos = np.arange(len(arms))
            parts = ax.violinplot([d if len(d) else [0.0] for d in data], positions=pos,
                                  showextrema=False, widths=0.8)
            for b, arm, good in zip(parts["bodies"], arms, ok):
                b.set_facecolor(arm_col.get(arm, "#999999"))
                b.set_alpha(0.6 if good else 0.15)
                b.set_edgecolor("none")
                if not good:
                    b.set_hatch("///")
            for p, d in zip(pos, data):
                if len(d):
                    ax.hlines(np.median(d), p - 0.26, p + 0.26, color="#111111",
                              linewidth=1.0, zorder=4)
            for p, arm in zip(pos, arms):
                _donor_dots(ax, [p], dmed, groupby, [lv], sc, group=group, arm=arm)
            for p, arm, good in zip(pos, arms, ok):
                if not good:
                    nd = int(((dmed[groupby] == lv) & dmed["eligible"] &
                              (dmed[group].astype(str) == arm)).sum())
                    ax.annotate(f"abstain\nn={nd}", (p, 0.5), xycoords=("data", "axes fraction"),
                                ha="center", va="center", fontsize=5, color="#888888")
            ax.set_xticks(pos)
            ax.set_xticklabels(arms, rotation=90, fontsize=6)
            ax.tick_params(axis="y", labelsize=6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if i == 0:
                ax.set_title(lv, fontsize=7.5)
            if j == 0:
                ax.set_ylabel(ylabel_fmt.format(sc), fontsize=7)
    fig.tight_layout()
    return _finish(fig, save, show)


def plot_split_violin(adata, dmed, scores, level_pair, title=None, save=None, show=True,
                      panel=(2.0, 2.0), ncols=4, level_col=None, external=None):
    """The same scores over two or more arms side by side (e.g. malignant vs reactive CD4).

    `level_col` is the obs/dmed column the arms live in; it defaults to `cfg.GROUPBY` but the
    arms are not always levels -- §7d splits one CD4 level into healthy/reactive/malignant, and
    those live in their own column. `external`, when given, maps a score to the external
    healthy donor medians for that arm set and draws them as a reference band behind each panel
    (see `plot_external_band` for why a band and not another violin).
    """
    import matplotlib.pyplot as plt

    groupby = cfg.GROUPBY if level_col is None else level_col
    obs = adata.obs
    n = len(scores)
    nr = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nr, ncols, figsize=(panel[0] * ncols, panel[1] * nr),
                             squeeze=False)
    colors = _level_colors(list(level_pair))
    for k, sc in enumerate(scores):
        ax = axes[k // ncols][k % ncols]
        data = [obs.loc[obs[groupby].astype(str) == lv, sc].dropna().to_numpy()
                for lv in level_pair]
        pos = np.arange(len(level_pair))
        parts = ax.violinplot([d if len(d) else [0.0] for d in data], positions=pos,
                              showextrema=False, widths=0.8)
        for b, lv in zip(parts["bodies"], level_pair):
            b.set_facecolor(colors[lv]); b.set_alpha(0.6); b.set_edgecolor("none")
        for p, d in zip(pos, data):
            if len(d):
                ax.hlines(np.median(d), p - 0.26, p + 0.26, color="#111111", linewidth=1.0,
                          zorder=4)
        _donor_dots(ax, pos, dmed, groupby, list(level_pair), sc)
        if external is not None and sc in external:
            plot_external_band(ax, external[sc], xspan=(pos[0] - 0.6, pos[-1] + 0.6))
        ax.set_xticks(pos)
        ax.set_xticklabels(level_pair, rotation=90, fontsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title(sc, fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for k in range(n, nr * ncols):
        axes[k // ncols][k % ncols].axis("off")
    if title:
        fig.suptitle(title, fontsize=10, y=1.01)
    fig.tight_layout()
    return _finish(fig, save, show)


def plot_degrade_vs_deposit(adata, x, y, levels=None, save=None, show=True,
                            figsize=(6.5, 5.5), max_cells=120_000, seed=0):
    """Per-cell scatter of degradative against structural output, centroid per level.

    One panel, deliberately: the question "who degrades more than they build" is a single
    two-dimensional fact and splitting it across facets hides the ordering.
    """
    import matplotlib.pyplot as plt

    groupby = cfg.GROUPBY
    obs = adata.obs
    levels = ([lv for lv in cfg.CT_ORDER if lv in set(obs[groupby].astype(str))]
              if levels is None else list(levels))
    colors = _level_colors(levels)
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=figsize)
    sub = obs[obs[groupby].astype(str).isin(levels)]
    if len(sub) > max_cells:
        sub = sub.iloc[rng.choice(len(sub), max_cells, replace=False)]
    ax.scatter(sub[x], sub[y], s=1.2, c=[colors[v] for v in sub[groupby].astype(str)],
               alpha=0.12, linewidths=0, rasterized=True)
    for lv in levels:
        d = obs[obs[groupby].astype(str) == lv]
        if not len(d):
            continue
        cx, cy = d[x].median(), d[y].median()
        ax.scatter([cx], [cy], s=60, color=colors[lv], edgecolor="black", linewidth=0.8,
                   zorder=5)
        ax.annotate(lv, (cx, cy), fontsize=6.5, zorder=6,
                    xytext=(4, 3), textcoords="offset points")
    lo = float(min(obs[x].min(), obs[y].min()))
    hi = float(max(obs[x].quantile(0.999), obs[y].quantile(0.999)))
    ax.plot([lo, hi], [lo, hi], ls="--", lw=0.8, color="#888888", zorder=1)
    ax.set_xlabel(x, fontsize=8)
    ax.set_ylabel(y, fontsize=8)
    ax.tick_params(labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return _finish(fig, save, show)


def plot_gene_panel(adata, dmed_genes, genes, levels, save=None, show=True,
                    panel=(2.0, 2.0), ncols=4, group=None, arms=None):
    """Per-gene violins over `levels` (or over `arms` within one level). For small families."""
    import matplotlib.pyplot as plt

    groupby = cfg.GROUPBY
    obs = adata.obs
    var = pd.Index(map(str, adata.var_names))
    genes = [g for g in genes if g in set(var)]
    nr = int(np.ceil(len(genes) / ncols))
    fig, axes = plt.subplots(nr, ncols, figsize=(panel[0] * ncols, panel[1] * nr),
                             squeeze=False)
    import scipy.sparse as sp
    X = adata.X.tocsc() if sp.issparse(adata.X) else sp.csc_matrix(adata.X)

    for k, g in enumerate(genes):
        ax = axes[k // ncols][k % ncols]
        col = np.asarray(X[:, var.get_loc(g)].todense()).ravel()
        s = pd.Series(col, index=obs.index)
        if group is None:
            cats, colors = list(levels), _level_colors(list(levels))
            data = [s[obs[groupby].astype(str) == lv].to_numpy() for lv in cats]
            cvals = [colors[c] for c in cats]
        else:
            cats = list(arms or cfg.STAGE_ARMS)
            m = obs[groupby].astype(str).isin(levels)
            data = [s[m & (obs[group].astype(str) == a)].to_numpy() for a in cats]
            cvals = [{"HC": "#4c9f70", "early": "#e8a33d",
                      "advanced": "#c0392b"}.get(a, "#999999") for a in cats]
        pos = np.arange(len(cats))
        parts = ax.violinplot([d if len(d) else [0.0] for d in data], positions=pos,
                              showextrema=False, widths=0.8)
        for b, c in zip(parts["bodies"], cvals):
            b.set_facecolor(c); b.set_alpha(0.6); b.set_edgecolor("none")
        for p, d in zip(pos, data):
            if len(d):
                ax.hlines(np.median(d), p - 0.26, p + 0.26, color="#111111", linewidth=1.0,
                          zorder=4)
        if dmed_genes is not None and g in dmed_genes.columns and group is None:
            _donor_dots(ax, pos, dmed_genes, groupby, cats, g)
        ax.set_xticks(pos)
        ax.set_xticklabels(cats, rotation=90, fontsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title(g, fontsize=8, style="italic")
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
    for k in range(len(genes), nr * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.tight_layout()
    return _finish(fig, save, show)


def plot_group_violins(adata, groups, genes, title=None, colors=None, save=None, show=True,
                       ncols=4, panel=(2.2, 2.2), max_cells=20_000, min_cells=None, seed=0,
                       ytop_q=0.98, labels=None):
    """One panel per gene, one violin per group, one dot per donor.

    groups : {label: bool mask over adata} in display order -- any cell set, not just a
             cfg.GROUPBY level (a pooled coarse type, a CD4 arm, a disease subset).
    labels : {group: display name} for the x axis only; the returned table keeps the keys.
    Violin = per-cell log1p CP10K over all cells, zeros included (subsampled to `max_cells` for
    the KDE only), black tick = median. Dots = per-donor median over that donor's cells in the
    group, donors with >= `min_cells` cells only. The y top is the `ytop_q` quantile of
    expressing cells, as in `plot_family_arms`.

    Returns the long donor table: group, donor, n_cells, one column per gene.
    """
    import matplotlib.pyplot as plt
    import scipy.sparse as sp

    min_cells = cfg.MIN_CELLS if min_cells is None else min_cells
    colors = colors or {}
    rng = np.random.default_rng(seed)
    var = pd.Index(map(str, adata.var_names))
    missing = [g for g in genes if g not in var]
    genes = [g for g in genes if g in var]
    if missing:
        print(f"not in var, skipped: {missing}")
    gi = var.get_indexer(genes)
    donors_all = adata.obs[cfg.DONOR_KEY].astype(str).to_numpy()

    vals, dmed = {}, []
    for lab, m in groups.items():
        idx = np.flatnonzero(np.asarray(m))
        if not len(idx):
            print(f"{lab}: no cells, skipped")
            continue
        sub = adata.X[idx][:, gi]
        sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
        d = pd.DataFrame(sub, columns=genes)
        d["donor"] = donors_all[idx]
        g = d.groupby("donor")
        med = g[genes].median()
        med.insert(0, "n_cells", g.size())
        med = med[med["n_cells"] >= min_cells].reset_index()
        med.insert(0, "group", lab)
        dmed.append(med)
        pick = (np.sort(rng.choice(len(idx), max_cells, replace=False))
                if len(idx) > max_cells else np.arange(len(idx)))
        vals[lab] = sub[pick]
        print(f"{lab}: {len(idx):,} cells / {len(med)} donors >= {min_cells} cells")
    dmed = pd.concat(dmed, ignore_index=True)
    labs = list(vals)
    pos = np.arange(len(labs))

    nr = int(np.ceil(len(genes) / ncols))
    fig, axes = plt.subplots(nr, ncols, figsize=(panel[0] * ncols, panel[1] * nr), squeeze=False)
    for k, g in enumerate(genes):
        ax = axes[k // ncols][k % ncols]
        tops = []
        for p, lab in zip(pos, labs):
            v = vals[lab][:, k]
            nz = v[v > 0]
            if len(nz):
                tops.append((len(nz) >= 20, np.quantile(nz, ytop_q)))
            if np.ptp(v) > 0:
                b = ax.violinplot([v], positions=[p], widths=0.8, showextrema=False)["bodies"][0]
                b.set_facecolor(colors.get(lab, "#999999")); b.set_alpha(0.6)
                b.set_edgecolor("none")
            ax.hlines(np.median(v), p - 0.26, p + 0.26, color="#111111", linewidth=1.0, zorder=4)
            dv = dmed.loc[dmed["group"] == lab, g].to_numpy()
            ax.scatter(p + rng.uniform(-0.12, 0.12, len(dv)), dv, s=9, zorder=5, color="#222222",
                       edgecolor="white", linewidth=0.3, alpha=0.9)
        top = (max([t for ok, t in tops if ok] or [t for _, t in tops]) if tops else 1.0)
        ax.set_ylim(-0.04 * top, 1.05 * top)
        ax.set_xlim(-0.6, len(labs) - 0.4)
        ax.set_title(g, fontsize=9, style="italic")
        ax.set_xticks([])
        if k % ncols == 0:
            ax.set_ylabel("log1p CP10K", fontsize=7)
        ax.tick_params(axis="y", labelsize=6.5)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    for k in range(len(genes), nr * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.tight_layout()
    # long display names go in a legend above the grid, not as rotated ticks under every panel
    from matplotlib.patches import Patch
    H, n_leg = panel[1] * nr, min(len(labs), 3)
    fig.legend(handles=[Patch(facecolor=colors.get(lab, "#999999"), alpha=0.6,
                              label=(labels or {}).get(lab, lab)) for lab in labs],
               loc="lower left", ncol=n_leg, fontsize=8, frameon=False,
               bbox_to_anchor=(0.0, 1.0 + 0.02 / H))
    if title:
        fig.suptitle(title, fontsize=11, x=0.01, ha="left", va="bottom",
                     y=1.0 + (0.12 + 0.22 * np.ceil(len(labs) / n_leg)) / H)
    _finish(fig, save, show)
    return dmed


def plot_by_study(adata, dmed, score, level, group="stage_group", arms=None,
                  save=None, show=True, panel=(1.7, 2.2)):
    """One panel per study: the confound made visible rather than asserted.

    In this cohort no study carries HC, early and advanced together, and the early arm is
    li2024-dominated at most levels. This figure is how the reader sees that, instead of
    taking the abstention table's word for it.
    """
    import matplotlib.pyplot as plt

    arms = cfg.STAGE_ARMS if arms is None else list(arms)
    groupby = cfg.GROUPBY
    obs = adata.obs[adata.obs[cfg.GROUPBY].astype(str) == level]
    studies = [s for s in sorted(obs[cfg.STUDY_KEY].astype(str).unique())
               if (obs[cfg.STUDY_KEY].astype(str) == s).sum() >= cfg.MIN_CELLS]
    arm_col = {"HC": "#4c9f70", "early": "#e8a33d", "advanced": "#c0392b"}
    fig, axes = plt.subplots(1, len(studies), figsize=(panel[0] * len(studies), panel[1]),
                             squeeze=False, sharey=True)
    for j, st in enumerate(studies):
        ax = axes[0][j]
        sub = obs[obs[cfg.STUDY_KEY].astype(str) == st]
        data = [sub.loc[sub[group].astype(str) == a, score].dropna().to_numpy() for a in arms]
        pos = np.arange(len(arms))
        parts = ax.violinplot([d if len(d) else [0.0] for d in data], positions=pos,
                              showextrema=False, widths=0.8)
        for b, a, d in zip(parts["bodies"], arms, data):
            b.set_facecolor(arm_col.get(a, "#999999"))
            b.set_alpha(0.65 if len(d) else 0.0)
            b.set_edgecolor("none")
        for p, d in zip(pos, data):
            if len(d):
                ax.hlines(np.median(d), p - 0.26, p + 0.26, color="#111111", linewidth=1.0,
                          zorder=4)
        ax.set_xticks(pos)
        ax.set_xticklabels(arms, rotation=90, fontsize=6)
        ax.set_title(st, fontsize=6.5)
        ax.tick_params(axis="y", labelsize=6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if j == 0:
            ax.set_ylabel(score, fontsize=7)
    fig.suptitle(f"{level} — {score} by study", fontsize=9, y=1.04)
    fig.tight_layout()
    return _finish(fig, save, show)


# ================================================================ per-gene, per-arm


ARM_COLORS = {"malignant": "#d62728", "reactive": "#2ca02c", "MF": "#d62728",
              "clonal": "#d62728", "non-clonal": "#2ca02c",
              "HC": "#2ca02c", "external HC": EXT_COLOR}


def _arm_slice(adata, mask, genes, max_cells, rng, min_cells=None):
    """Per-cell values of `genes` in one arm: dense subsample for the violin; detection, mean and
    per-donor means over all cells.

    Genes absent from this object's var come back as NaN columns and NaN stats -- absent is not
    zero, and the external reference does not carry the whole panel. `dmean` holds one row per
    donor with >= `min_cells` cells in the arm: the replicate unit the tests run on.
    """
    import scipy.sparse as sp

    min_cells = cfg.MIN_CELLS if min_cells is None else min_cells
    var = pd.Index(map(str, adata.var_names))
    have = np.array([g in var for g in genes], dtype=bool)
    idx = np.flatnonzero(np.asarray(mask))
    n = len(idx)
    k = min(n, max_cells)
    vals = np.full((k, len(genes)), np.nan, dtype=np.float32)
    det = np.full(len(genes), np.nan)
    mean = np.full(len(genes), np.nan)
    donor_key = cfg.DONOR_KEY if cfg.DONOR_KEY in adata.obs else cfg.EXT_DONOR_KEY
    donors = adata.obs[donor_key].astype(str).to_numpy()[idx]
    codes, uniq = pd.factorize(donors)
    cnt = np.bincount(codes, minlength=len(uniq))
    elig = cnt >= min_cells
    dmean = np.full((int(elig.sum()), len(genes)), np.nan)
    if n and have.any():
        sub = adata.X[idx][:, var.get_indexer(np.asarray(genes)[have])]
        sub = sp.csr_matrix(sub) if not sp.issparse(sub) else sub.tocsr()
        det[have] = np.asarray((sub > 0).mean(axis=0)).ravel()
        mean[have] = np.asarray(sub.mean(axis=0)).ravel()
        pick = np.sort(rng.choice(n, k, replace=False)) if n > k else np.arange(n)
        vals[:, have] = sub[pick].toarray()
        G = sp.csr_matrix((np.ones(n), (codes, np.arange(n))), shape=(len(uniq), n))
        dm = np.asarray((G @ sub).todense()) / cnt[:, None]
        dmean[:, have] = dm[elig]
    return {"vals": vals, "det": det, "mean": mean, "n": n, "n_donors": len(uniq),
            "dmean": dmean, "have": have}


def arm_tests(sl, genes, arms, only=None, min_donors=None):
    """Donor-level Mann-Whitney for every (row, gene, arm pair); BH over the whole figure.

    The unit is the donor mean, not the cell: at 10^4 cells per arm a cell-level rank test
    calls everything significant, and one deep patient can carry it
    (cfg.FORBIDDEN_CONTRASTS['cell_level_pvalues']). Unblocked by study -- MF vs HC mixes
    studies and the external arm cannot be blocked at all -- so a star means the donor
    distributions separate, not that study has been controlled for.
    """
    from itertools import combinations
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests

    min_donors = cfg.MIN_DONORS_PER_ARM if min_donors is None else min_donors
    out = []
    for r, arm_d in sl.items():
        for a, b in combinations(arms, 2):
            sa, sb = arm_d.get(a), arm_d.get(b)
            if sa is None or sb is None:
                continue
            for j, g in enumerate(genes):
                if only is not None and g not in only:
                    continue
                x, y = sa["dmean"][:, j], sb["dmean"][:, j]
                x, y = x[~np.isnan(x)], y[~np.isnan(y)]
                rec = {"row": r, "gene": g, "arm_a": a, "arm_b": b,
                       "n_donors_a": len(x), "n_donors_b": len(y),
                       "donor_median_a": np.median(x) if len(x) else np.nan,
                       "donor_median_b": np.median(y) if len(y) else np.nan,
                       "p": np.nan}
                rec["delta"] = rec["donor_median_b"] - rec["donor_median_a"]
                if len(x) >= min_donors and len(y) >= min_donors and np.ptp(np.r_[x, y]) > 0:
                    rec["p"] = float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
                out.append(rec)
    res = pd.DataFrame(out, columns=["row", "gene", "arm_a", "arm_b", "n_donors_a",
                                     "n_donors_b", "donor_median_a", "donor_median_b",
                                     "p", "delta"])
    res["q"] = np.nan
    m = res["p"].notna()
    if m.any():
        res.loc[m, "q"] = multipletests(res.loc[m, "p"], method="fdr_bh")[1]
    return res


def arm_test_block(rows, genes, arms, min_det=None, max_cells=0, seed=0, label=None,
                   verbose=True):
    """`arm_tests` with no figure attached: slice, drop undetected genes, test, explain.

    This is the headless half of `plot_family_arms`, split out so the notebook can run the
    donor-level comparison over the whole panel at once instead of one figure per family.
    `plot_family_arms` calls it too, so the two paths cannot drift apart.

    `max_cells=0` skips the violin subsample entirely -- the per-cell values are only ever used
    to draw, and a 474-gene panel over three arms is a lot of array to allocate for nothing.

    Returns `(tests, info)`. `tests` is `arm_tests`' table plus two columns the figure never
    needed and a reader does:

      status : 'significant' (q < 0.05) | 'ns' | 'not_tested'
      reason : why a pair was not tested -- an abstention is a coverage statement, not a null,
               and the two must never be printed alike.

    Arm pairs where one arm does not exist for a row (no external counterpart) are absent from
    `arm_tests` altogether; they are added back here as `not_tested`, because "we could not
    look" is a result.
    """
    min_det = cfg.MIN_DETECTION if min_det is None else min_det
    rng = np.random.default_rng(seed)
    genes = list(dict.fromkeys(genes))

    sl = {r: {a: (_arm_slice(ad, m, genes, max_cells, rng) if spec is not None else None)
              for a, spec in arm_d.items() for ad, m in [spec or (None, None)]}
          for r, arm_d in rows.items()}
    det = np.vstack([s["det"] for arm_d in sl.values() for s in arm_d.values()
                     if s is not None and s["n"]])
    keep = np.nanmax(np.where(np.isnan(det), -1, det), axis=0) >= min_det
    dropped = [g for g, k in zip(genes, keep) if not k]
    kept = [g for g, k in zip(genes, keep) if k]
    ext_missing = sorted({g for arm_d in sl.values() for a, s in arm_d.items()
                          if s is not None and a == "external HC"
                          for g, h in zip(genes, s["have"]) if not h})
    tests = arm_tests(sl, genes, arms, only=set(kept))
    tests = _explain_arm_tests(tests, sl, genes, arms, kept)
    info = {"sl": sl, "kept": kept, "dropped": dropped, "ext_missing": ext_missing,
            "min_det": min_det}
    if verbose:
        n_sig = int((tests["q"] < 0.05).sum())
        print(f"{label or 'block'}: {len(kept)}/{len(genes)} genes above {min_det:.0%} "
              f"detection; {n_sig}/{int(tests['p'].notna().sum())} donor-level tests at q < 0.05"
              + (f"; never detected: {len(dropped)}" if dropped else "")
              + (f"; absent from external var: {len(ext_missing)}" if ext_missing else ""))
    return tests, info


def _explain_arm_tests(tests, sl, genes, arms, kept, min_donors=None):
    """Add `status` / `reason` to an `arm_tests` table and restore the untestable pairs."""
    from itertools import combinations

    min_donors = cfg.MIN_DONORS_PER_ARM if min_donors is None else min_donors
    gi = {g: j for j, g in enumerate(genes)}
    kept_set = set(kept)
    if not len(tests):
        return tests.assign(reason="", status="", higher="")

    extra = []
    for r, arm_d in sl.items():
        for a, b in combinations(arms, 2):
            if arm_d.get(a) is not None and arm_d.get(b) is not None:
                continue
            gone = [x for x in (a, b) if arm_d.get(x) is None]
            for g in kept:
                extra.append({"row": r, "gene": g, "arm_a": a, "arm_b": b,
                              "reason": f"no {' and '.join(gone)} arm for this row"})
    if extra:
        tests = pd.concat([tests, pd.DataFrame(extra)], ignore_index=True)

    def _why(rec):
        if pd.notna(rec["p"]):
            return ""
        if isinstance(rec.get("reason"), str) and rec["reason"]:
            return rec["reason"]
        if rec["gene"] not in kept_set:
            return "below detection in every arm"
        j = gi[rec["gene"]]
        for arm in (rec["arm_a"], rec["arm_b"]):
            s = sl[rec["row"]].get(arm)
            if s is not None and not s["have"][j]:
                return f"gene absent from the {arm} object"
        if (rec["n_donors_a"] < min_donors) or (rec["n_donors_b"] < min_donors):
            return (f"too few donors ({rec['arm_a']} {rec['n_donors_a']:.0f}, "
                    f"{rec['arm_b']} {rec['n_donors_b']:.0f}; need >= {min_donors} each)")
        return "no variance (all donor means equal)"

    tests["reason"] = tests.apply(_why, axis=1)
    tests["status"] = np.where(tests["p"].isna(), "not_tested",
                               np.where(tests["q"] < 0.05, "significant", "ns"))
    tests["higher"] = np.where(tests["status"] == "not_tested", "",
                               np.where(tests["delta"] > 0, tests["arm_b"], tests["arm_a"]))
    return tests


def _stars(q):
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


def plot_family_arms(rows, genes, arms, title, notes=None, save=None, show=True,
                     arm_colors=None, max_cells=20_000, min_det=None, seed=0,
                     col_w=0.9, row_h=1.6, ytop_q=0.98, wrap=None, expressing_only=False,
                     clean=False):
    """Every gene of a family, per cell, with the arms side by side at each gene.

    rows : {row_label: {arm: (adata, bool_mask) or None}} -- one subplot row per cell type at
           the current resolution. None = the arm does not exist for that row (no external
           counterpart); it is named in the row title rather than drawn as an empty violin.
    arms : arm order, shared by every row.

    Grid = cell types x genes. Each gene column has its OWN y scale, shared down the column so
    rows stay comparable; the top is the `ytop_q` quantile of expressing cells, so a handful of
    outlier cells no longer flattens a low-signal gene (the violins above it are clipped).
    Violin = per-cell log1p CP10K (subsampled to `max_cells` per arm for the KDE only), black
    tick = median, white diamond = mean over all cells. Brackets mark arm pairs whose donor
    means differ at BH q < 0.05 (see `arm_tests`). Genes detected in fewer than `min_det` of
    cells in every arm of every row are dropped and printed.

    wrap            : genes per grid row (default: all in one); each block of `wrap` genes
                      repeats the cell-type rows.
    expressing_only : violin and median tick over expressing cells only, % expressing printed
                      under each violin, no mean diamond. For sparse genes, where the zero spike
                      flattens an all-cell violin to a line.
    clean           : no per-row cell/donor header and no bracket key (tested donors go in the
                      legend), larger fonts.

    Returns the test table (all pairs, significant or not).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    min_det = cfg.MIN_DETECTION if min_det is None else min_det
    colors = {**ARM_COLORS, **(arm_colors or {})}
    notes = notes or {}
    genes = list(dict.fromkeys(genes))

    tests, info = arm_test_block(rows, genes, arms, min_det=min_det, max_cells=max_cells,
                                 seed=seed, label=title, verbose=False)
    sl, kept = info["sl"], info["kept"]
    dropped, ext_missing = info["dropped"], info["ext_missing"]
    n_sig = int((tests["q"] < 0.05).sum())
    print(f"{title}: {len(kept)}/{len(genes)} genes drawn; {n_sig}/{int(tests['p'].notna().sum())}"
          f" donor-level tests at q < 0.05"
          + (f"; below {min_det:.0%} detection everywhere: {dropped}" if dropped else "")
          + (f"; absent from external var: {ext_missing}" if ext_missing else ""))
    if not kept:
        return tests

    nr, na, ng = len(rows), len(arms), len(kept)
    wrap = min(wrap or ng, ng)
    nb = -(-ng // wrap)
    fs = 1.3 if clean else 1.0
    fig, axes = plt.subplots(nr * nb, wrap, figsize=(max(5.0, col_w * wrap + 1.2), row_h * nr * nb),
                             squeeze=False)

    def ax_at(i, c):
        return axes[(c // wrap) * nr + i][c % wrap]

    for c in range(ng, nb * wrap):
        for i in range(nr):
            ax_at(i, c).set_visible(False)
    for c, g in enumerate(kept):
        j = genes.index(g)
        tops, n_br, pcts = [], 0, []
        for i, (r, arm_d) in enumerate(sl.items()):
            ax = ax_at(i, c)
            for a_i, a in enumerate(arms):
                s = arm_d.get(a)
                if s is None or not s["n"]:
                    continue
                v = s["vals"][:, j]
                if np.isnan(v).all():
                    continue
                nz = v[v > 0]
                if len(nz):
                    tops.append((len(nz) >= 20, np.quantile(nz, ytop_q)))
                d = nz if expressing_only else v
                if len(d) > 1 and np.ptp(d) > 0:
                    b = ax.violinplot([d], positions=[a_i], widths=0.85,
                                      showextrema=False)["bodies"][0]
                    b.set_facecolor(colors.get(a, "#999999")); b.set_alpha(0.6)
                    b.set_edgecolor("none")
                if len(d):
                    ax.hlines(np.median(d), a_i - 0.3, a_i + 0.3, color="#111111",
                              linewidth=0.9, zorder=4)
                if expressing_only:
                    pcts.append((ax, a_i, 100 * s["det"][j]))
                else:
                    ax.scatter([a_i], [s["mean"][j]], marker="D", s=8, color="white",
                               edgecolor="#111111", linewidth=0.5, zorder=5)
            sig = tests[(tests["row"] == r) & (tests["gene"] == g) & (tests["q"] < 0.05)]
            ax._sig = [(arms.index(t.arm_a), arms.index(t.arm_b), _stars(t.q))
                       for t in sig.itertuples()]
            n_br = max(n_br, len(ax._sig))
            ax.set_xticks([])
            ax.set_xlim(-0.6, na - 0.4)
            ax.tick_params(axis="y", labelsize=5 * fs, length=2, pad=1)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
        # arms with a handful of expressing cells would set the scale by their outliers
        top = (max([t for ok, t in tops if ok] or [t for _, t in tops]) if tops else 1.0)
        step = 0.13 * top
        for i in range(nr):
            ax = ax_at(i, c)
            for k, (x0, x1, st) in enumerate(sorted(ax._sig, key=lambda t: t[1] - t[0])):
                h = top + step * (0.4 + k)
                ax.plot([x0, x0, x1, x1], [h - step * 0.25, h, h, h - step * 0.25],
                        color="#333333", linewidth=0.6, clip_on=False)
                ax.text((x0 + x1) / 2, h, st, ha="center", va="bottom", fontsize=6 * fs)
            ax.set_ylim(-(0.16 if expressing_only else 0.04) * top, top + step * (0.6 + n_br))
            if expressing_only:
                ax.set_yticks([t for t in ax.get_yticks() if 0 <= t <= ax.get_ylim()[1]])
        for ax, x, p in pcts:
            ax.text(x, -0.08 * top, f"{p:.0f}%" if p >= 9.5 else f"{p:.1f}%",
                    ha="center", va="center", fontsize=5 * fs, color="#444444")
        ax_at(nr - 1, c).set_xlabel(g, fontsize=7 * fs, style="italic")

    labels = []
    for i, (r, arm_d) in enumerate(sl.items()):
        for blk in range(nb):
            axes[blk * nr + i][0].set_ylabel(r + (f"\n[{notes[r]}]" if clean and r in notes
                                                  else ""), fontsize=8 * fs)
        if clean:
            continue
        parts = []
        for a in arms:
            s = arm_d.get(a)
            parts.append(f"{a}: no ext. counterpart" if s is None
                         else f"{a} {s['n']:,} cells / {s['n_donors']} donors "
                              f"({len(s['dmean'])} tested)")
        t = axes[i][0].text(0, 1.06, f"{r}   |   " + "  ·  ".join(parts)
                            + (f"   [{notes[r]}]" if r in notes else ""),
                            transform=axes[i][0].transAxes, ha="left", va="bottom", fontsize=7)
        t.set_in_layout(False)
        labels.append(t)
    fig.tight_layout(h_pad=2.2, w_pad=0.4 if not clean else 1.0)
    for t in labels:              # out of tight_layout, back in for the saved bbox
        t.set_in_layout(True)
    H = row_h * nr * nb
    if clean:
        n_test = {a: [len(arm_d[a]["dmean"]) for arm_d in sl.values() if arm_d.get(a)]
                  for a in arms}
        handles = [Patch(facecolor=colors.get(a, "#999999"), alpha=0.6,
                         label=f"{a} (n={n_test[a][0]})" if nr == 1 and n_test[a] else a)
                   for a in arms]
        leg_y, title_y = 0.02, 0.4
    else:
        handles = ([Patch(facecolor=colors.get(a, "#999999"), alpha=0.6, label=a) for a in arms]
                   + [Patch(facecolor="none", edgecolor="none",
                            label="brackets: donor-level MWU, BH q<.05 * <.01 ** <.001 ***")])
        leg_y, title_y = 0.25, 0.6
    fig.legend(handles=handles, loc="lower left", ncol=len(handles), fontsize=7 * fs,
               frameon=False, bbox_to_anchor=(0.0, 1.0 + leg_y / H))
    fig.suptitle(title, fontsize=10 * fs, x=0.01, ha="left", va="bottom", y=1.0 + title_y / H)
    _finish(fig, save, show)
    if save is not None:          # tables/deg28_gene_tests/<figure stem>.csv
        out = cfg.TAB_DIR / f"{cfg.TAB_PREFIX}gene_tests" / f"{Path(save).stem}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        tests.to_csv(out, index=False)
    return tests


# ================================================================ agent-readable report


def agent_report(blocks, families, claim=(), ext_missing=(), context=None, min_det=None,
                 out=None, seed=0, arms_desc=None, title=None):
    """Print every number behind the §4-§6 figures as self-describing plain text.

    Written to be pasted into a chat with an agent that has no access to the notebook: every
    line names its block, family, cell type, arm and statistic, and the header states the
    units, the arm definitions, the test, and the caveats a reader needs to weigh a star.

    blocks   : {tag: rows}, the `rows` dicts passed to plot_family_arms (the notebook's BLOCKS).
    families : {family: [genes]} in display order.
    Per-arm statistics are recomputed over all cells; test rows are read back from
    tables/deg28_gene_tests/, so p/q are exactly the ones behind the brackets. Every gene the
    figure drew is listed for every row, and every arm pair gets a line: tested (delta, donors,
    p, q, direction) or NOT TESTED with the reason. Genes the figure dropped are named per
    family. Writes the long per-arm table to `out` (default tables/deg28_agent_report.csv) and
    every pair to `<out>_tests.csv`; returns the per-arm table.
    """
    min_det = cfg.MIN_DETECTION if min_det is None else min_det
    out = cfg.tab("agent_report") if out is None else Path(out)
    rng = np.random.default_rng(seed)
    claim = set(claim)
    allg = list(dict.fromkeys(g for gs in families.values() for g in gs))
    gi = {g: j for j, g in enumerate(allg)}
    L = []
    P = L.append

    P("#" * 100)
    P(title or "NB28 DEGRADOME — PER-GENE, PER-CELL-TYPE, PER-CONDITION NUMERIC REPORT "
               "(atlas v2, skin)")
    P("#" * 100)
    P("UNITS: expression = log1p(CP10K); library size summed over all 42,347 genes before the "
      "panel subset, so values are comparable across cell types and samples.")
    P("PER-ARM STATISTICS (all expression values are log1p CP10K): "
      "pct_expr = % of cells with expression > 0; "
      "mean_all = mean log expression over ALL cells (zeros included); "
      "mean_expr = mean log expression over EXPRESSING cells only (= mean_all / pct_expr); "
      "donor_median = median over donors of each donor's mean_all (donors with >= "
      f"{cfg.MIN_CELLS} cells in that arm only) -- this is the value the test compares; "
      "cells/donors = all cells and donors in the arm; tested = donors that enter the test.")
    P(f"ARMS: {arms_desc}" if arms_desc else
      "ARMS: CD4 block -> 'malignant' / 'reactive' = CD4 from MF samples split by the "
      f"{cfg.MALIG_HEADLINE} malignancy call; 'external HC' = healthy CD4 (Th) from Reynolds "
      "2021. Myeloid/fibroblast blocks -> 'MF' = cells of that type from MF-patient samples; "
      "'HC' = cells from healthy-control samples in this atlas; 'external HC' = Reynolds 2021 "
      "healthy skin (different lab, 10x 3' v2, 5 female donors).")
    P("RESOLUTION: *_low = nb22 subtype_coarse (Myeloid: Mono_Mac, DC, LC, Neutrophils; "
      "Fibroblast: F_structural, F_inflammatory, F_apCAF); *_high = nb22 subtype_fine.")
    P("EXTERNAL MATCHING (our level -> Reynolds targets pooled): "
      + "; ".join(f"{k}->{'+'.join(v)}" for k, v in cfg.EXT_ARM.items())
      + f". POOLED (nearest parent, not a matched subset): {sorted(cfg.EXT_POOLED)}. "
      "No external arm: pDC, Neutrophils.")
    P("TEST: two-sided Mann-Whitney U on per-donor means (unit = donor, not cell), each arm "
      f">= {cfg.MIN_DONORS_PER_ARM} tested donors; Benjamini-Hochberg over all tests in one "
      "figure (= one block x one family). delta_log1p = donor_median(arm_b) - donor_median(arm_a), in log1p CP10K units. SIGNIFICANT = "
      "q < 0.05.")
    P("CAVEATS: (1) MF vs HC is NOT blocked by study; HC and MF skin come partly from different "
      "cohorts, so a difference can carry study/chemistry effects. (2) The external arm cannot "
      "be blocked at all: different lab, chemistry, site and sex; shifts in genes with no "
      "biological reason to differ are the platform floor. (3) the CD4 split is MF-only: "
      + (f"{cfg.MALIG_HEADLINE} calls ~{cfg.MALIG_HC_FP[cfg.MALIG_HEADLINE]:.0%} of healthy CD4 "
         "malignant, having never seen a healthy repertoire. "
         if cfg.MALIG_HEADLINE == "mal_knn" else
         f"{cfg.MALIG_HEADLINE} assigns HC samples benign as a rule, so there is nothing to "
         "split inside HC and its 0% HC rate is an input, not a result. ") +
      "(4) malignant and reactive CD4 come from the same donors but are tested unpaired "
      "(conservative). (5) rows flagged 'below donor gate' do not have >= "
      f"{cfg.MIN_CELLS} cells in >= {cfg.MIN_SAMPLES} donors in any stage arm: describe, do "
      "not quote as a result.")
    for k, v in (context or {}).items():
        P(f"CONTEXT {k}: {v}")
    P(f"PANEL GENES ABSENT FROM THE EXTERNAL REFERENCE (external stats = NaN): "
      f"{list(ext_missing) or 'none'}")
    P(f"FAMILIES (display order): "
      + "; ".join(f"{f}=[{', '.join(g)}]" for f, g in families.items()))

    from itertools import combinations

    recs, sig_all, test_recs = [], [], []
    for tag, rows in blocks.items():
        arms = list(next(iter(rows.values())))
        P("")
        P("=" * 100)
        P(f"BLOCK {tag} | arms in order: {arms} | rows (cell types): {list(rows)}")
        P("=" * 100)
        st = {}
        for r, arm_d in rows.items():
            parts = []
            for a, spec in arm_d.items():
                st[(r, a)] = (None if spec is None
                              else _arm_slice(spec[0], spec[1], allg, 0, rng))
                s = st[(r, a)]
                parts.append(f"{a}: NO EXTERNAL COUNTERPART" if s is None else
                             f"{a}: {s['n']:,} cells, {s['n_donors']} donors, "
                             f"{len(s['dmean'])} tested")
            flags = []
            if r in cfg.EXT_POOLED:
                flags.append(f"external arm POOLED from {'+'.join(cfg.EXT_ARM[r])}")
            lv = (cfg.MYELOID_COARSE.get(r) or cfg.FIBRO_COARSE.get(r) or [r])
            if claim and r != "CD4" and not set(lv) & claim:
                flags.append("BELOW DONOR GATE")
            P(f"ROW {r} :: " + " | ".join(parts) + (f" :: {'; '.join(flags)}" if flags else ""))

        pairs = list(combinations(arms, 2))
        for fam, genes in families.items():
            tp = (cfg.TAB_DIR / f"{cfg.TAB_PREFIX}gene_tests"
                  / f"{cfg.FIG_PREFIX}{tag}_{fam}.csv")
            tests = pd.read_csv(tp) if tp.exists() else pd.DataFrame(
                columns=["row", "gene", "arm_a", "arm_b", "n_donors_a", "n_donors_b",
                         "p", "q", "delta"])
            tests["row"] = tests["row"].astype(str)
            tix = {(x.row, x.gene, x.arm_a, x.arm_b): x for x in tests.itertuples()}
            # the figure's gene set: detected >= min_det in any arm of any row
            drawn = [g for g in genes
                     if np.nanmax([st[k]["det"][gi[g]] for k in st
                                   if st[k] is not None] + [-1.0]) >= min_det
                     or g in set(tests["gene"])]
            n_sig = int((tests["q"] < 0.05).sum())
            P("")
            P(f"--- {tag} | family {fam} | {len(drawn)}/{len(genes)} genes in figure | "
              f"{n_sig}/{int(tests['p'].notna().sum())} tests significant"
              + ("" if tp.exists() else " | NO TEST TABLE (figure not run)"))
            gone = [g for g in genes if g not in drawn]
            if gone:
                P(f"    not in figure (< {min_det:.0%} of cells in every arm of every row): "
                  f"{gone}")
            for r in rows:
                P(f" [{tag} | {fam} | {r}]")
                for g in drawn:
                    j = gi[g]
                    cells = []
                    for a in arms:
                        s = st[(r, a)]
                        if s is None:
                            cells.append(f"{a}: n/a (no external counterpart)")
                            continue
                        dm = s["dmean"][:, j]
                        dmed = float(np.nanmedian(dm)) if np.isfinite(dm).any() else np.nan
                        mpos = (s["mean"][j] / s["det"][j] if s["det"][j] > 0 else 0.0)
                        cells.append(f"{a}: pct_expr={s['det'][j]:.1%} "
                                     f"mean_all={s['mean'][j]:.3f} mean_expr={mpos:.3f} "
                                     f"donor_median={dmed:.3f}" if s["have"][j]
                                     else f"{a}: GENE ABSENT")
                        recs.append({"block": tag, "family": fam, "row": r, "gene": g,
                                     "arm": a, "n_cells": s["n"], "n_donors": s["n_donors"],
                                     "n_donors_tested": int(np.isfinite(dm).sum()),
                                     "pct_expr": s["det"][j], "mean_all": s["mean"][j],
                                     "mean_expr": (s["mean"][j] / s["det"][j]
                                                   if s["det"][j] > 0 else 0.0),
                                     "donor_median": dmed})
                    P(f"  {g}: " + " | ".join(cells))
                    for a, b in pairs:
                        x = tix.get((r, g, a, b))
                        sa, sb = st[(r, a)], st[(r, b)]
                        rec = {"block": tag, "family": fam, "row": r, "gene": g,
                               "arm_a": a, "arm_b": b}
                        if x is not None and pd.notna(x.p):
                            hi = b if x.delta > 0 else a if x.delta < 0 else "neither"
                            sig = x.q < 0.05
                            P(f"      {a} vs {b}: donor_median {x.donor_median_a:.3f} vs "
                              f"{x.donor_median_b:.3f}, delta_log1p={x.delta:+.4f}, "
                              f"donors={x.n_donors_a}/{x.n_donors_b}, p={x.p:.2g}, q={x.q:.2g} -> "
                              + (f"SIGNIFICANT, higher in {hi}" if sig
                                 else f"ns (direction: {hi})"))
                            rec.update(status="significant" if sig else "not_significant",
                                       donor_median_a=x.donor_median_a,
                                       donor_median_b=x.donor_median_b, delta=x.delta, n_donors_a=x.n_donors_a,
                                       n_donors_b=x.n_donors_b, p=x.p, q=x.q, higher_in=hi)
                            if sig:
                                sig_all.append(rec)
                        else:
                            if sa is None or sb is None:
                                why = "no external counterpart"
                            elif not (sa["have"][j] and sb["have"][j]):
                                why = "gene absent from " + " and ".join(
                                    arm for arm, s_ in ((a, sa), (b, sb)) if not s_["have"][j])
                            else:
                                na_ = int(np.isfinite(sa["dmean"][:, j]).sum())
                                nb_ = int(np.isfinite(sb["dmean"][:, j]).sum())
                                why = (f"too few donors ({a} {na_}, {b} {nb_}; need "
                                       f">= {cfg.MIN_DONORS_PER_ARM} each)"
                                       if min(na_, nb_) < cfg.MIN_DONORS_PER_ARM
                                       else "no variance (all donor means equal)")
                            P(f"      {a} vs {b}: NOT TESTED - {why}")
                            rec.update(status="not_tested", reason=why)
                        test_recs.append(rec)

    P("")
    P("=" * 100)
    P("SUMMARY OF SIGNIFICANT DIFFERENCES (q < 0.05), per block and arm pair "
      "(every one is also listed in full above)")
    P("=" * 100)
    sig = pd.DataFrame(sig_all)
    if len(sig):
        for (tag, a, b), d in sig.groupby(["block", "arm_a", "arm_b"], sort=False):
            up = d.groupby("higher_in").size().to_dict()
            P(f"{tag} | {a} vs {b}: {len(d)} significant gene x row tests | higher in: {up}")
            P("   by family: " + ", ".join(f"{f}={n}" for f, n in
                                           d.groupby("family", sort=False).size().items()))
            top = d.reindex(d["delta"].abs().sort_values(ascending=False).index).head(15)
            P("   largest |delta|: " + "; ".join(
                f"{x.gene}@{x.row} {x.delta:+.2f} ({x.higher_in}, q={x.q:.1g})"
                for x in top.itertuples()))
    else:
        P("none")

    df = pd.DataFrame(recs)
    df.to_csv(out, index=False)
    tout = out.with_name(out.stem + "_tests.csv")
    tdf = pd.DataFrame(test_recs)
    tdf.to_csv(tout, index=False)
    P("")
    P(f"full per-arm table: {out} ({len(df):,} rows); every arm pair incl. not-tested with "
      f"reason: {tout} ({len(tdf):,} rows)")
    txt = out.with_suffix(".txt")
    P(f"this printout: {txt}")
    txt.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return df
