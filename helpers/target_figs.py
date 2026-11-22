"""The figure layer of the target screen, in the style of nb28 (`28_tme_degradome_mc`).

nb28's drawing code lives in notebook cells, not in `helpers/`, so it cannot be imported. This
is that layer rewritten against the target screen's long table, with the same conventions, so
the two notebooks' figures read as one set:

  - a paired slope plot is one line per donor joining its reactive and malignant means (red if
    up in malignant, blue if down, all one width),
    dots on both ends, a black median tick per arm, a BH star and the n-up / n-down donor vote
    above each gene, and the % of cells detected printed under each gene tick;
  - a dumbbell is one horizontal row per test, hollow dot for the comparator, filled dot
    coloured by direction, faint per-donor dots behind;
  - titles say what the panel is, never what it shows. Findings go in the printed table.

Everything takes the long table (`arm, donor, gene, <value>, pct_expr, n_cells`) and returns a
record frame that every figure in this module can draw, so a gene set is tested once and shown
three ways.

    import target_figs as TF
    TF.rc()
    rec = TF.t_paired(long, genes, "CD4_malignant", "CD4_reactive")
    fig, ax = plt.subplots(figsize=(10.2, 3.4))
    TF.paired_slopes(ax, rec)
    TF.save(fig, "FA_top_candidates"); TF.save_source(rec, "FA_top_candidates")
"""
from __future__ import annotations

import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import offset_copy
from scipy import stats
from statsmodels.stats.multitest import multipletests

import surface_targets as ST

# nb28's palette, unchanged.
MALIG = "#7b3294"
REACT = "#5e81ac"
COMPARATOR = "#444444"
UP, DOWN, NS = "#2a9d8f", "#b5542a", "#bdbdbd"
LINE_UP, LINE_DOWN = "#d62728", "#1f77b4"
GREY = "#9e9e9e"
STAGE_EARLY, STAGE_ADV = "#e8a33d", "#c0392b"
XLAB = "donor mean, log1p CP10K"

ARM_COLOR = {"CD4_malignant": MALIG, "CD4_reactive": REACT, "CD4_HC": COMPARATOR,
             "CD8": "#8c8c8c", "NK": "#b07aa1"}

FIG_INDEX: list[dict] = []


def _fonts(prefer=("Carlito", "Calibri", "Helvetica", "Arial", "DejaVu Sans")):
    """The preferred faces that this machine actually has, so matplotlib stops warning."""
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    return [f for f in prefer if f in have] or ["DejaVu Sans"]


def rc():
    """nb28 Part III's rcParams. Call once per notebook, before the first figure."""
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.family": _fonts(),
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": "tight", "savefig.dpi": 400,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def it(s):
    return f"$\\it{{{str(s).replace('_', chr(92) + '_')}}}$"


def stars(q):
    if not np.isfinite(q):
        return ""
    return "***" if q < 1e-3 else "**" if q < 1e-2 else "*" if q < 0.05 else "ns"


def pct(v):
    return "" if not np.isfinite(v) else f"{100 * v:.0f}%"


# ------------------------------------------------------------------ records
def t_paired(long, genes, arm_a="CD4_malignant", arm_b="CD4_reactive", value="mean_all",
             min_donors=5, label_col=None, pseudo=0.05):
    """One record per gene: the paired within-donor test plus everything a figure needs.

    `vals_a` / `vals_b` hold one value per donor, in matching order, so the slope plot can join
    them. Donors carrying only one arm are dropped, as in `target_screen.paired_selectivity`.
    """
    genes = [g for g in dict.fromkeys(genes)]
    d = long[long["arm"].isin([arm_a, arm_b]) & long["gene"].isin(genes)]
    w = d.pivot_table(index=["gene", "donor"], columns="arm", values=value)
    det = d.pivot_table(index=["gene", "donor"], columns="arm", values="pct_expr")
    w = w.dropna(subset=[c for c in (arm_a, arm_b) if c in w.columns])

    rows = []
    for gene in genes:
        if gene not in w.index.get_level_values("gene"):
            rows.append(dict(gene=gene, n_pairs=0, p=np.nan, reason="absent from the object"))
            continue
        g = w.xs(gene, level="gene")
        a, b = g[arm_a].to_numpy(), g[arm_b].to_numpy()
        diff = a - b
        dg = det.xs(gene, level="gene").reindex(g.index) if gene in \
            det.index.get_level_values("gene") else None
        p = (float(stats.wilcoxon(diff).pvalue)
             if len(g) >= min_donors and np.any(diff != 0) else np.nan)
        rows.append(dict(
            gene=gene, label=gene if label_col is None else label_col.get(gene, gene),
            arm_a=arm_a, arm_b=arm_b, test="wilcoxon paired",
            n_pairs=len(g), n_a=len(a), n_b=len(b),
            med_a=float(np.median(a)), med_b=float(np.median(b)),
            det_a=float(np.nanmedian(dg[arm_a])) if dg is not None else np.nan,
            det_b=float(np.nanmedian(dg[arm_b])) if dg is not None else np.nan,
            delta=float(np.median(diff)),
            log2fc=float(np.median(np.log2((a + pseudo) / (b + pseudo)))),
            sign=float(np.sign(np.median(diff))),
            n_pos=int((diff > 0).sum()), n_neg=int((diff < 0).sum()),
            p=p, reason="" if np.isfinite(p) else f"fewer than {min_donors} paired donors",
            vals_a=a, vals_b=b, donors=list(g.index)))
    return bh(pd.DataFrame(rows))


def bh(df, p_col="p", q_col="q"):
    """BH over the tested rows only; untested rows keep a NaN q and the 'not tested' direction."""
    df = df.copy()
    m = df[p_col].notna()
    df[q_col] = np.nan
    if m.any():
        df.loc[m, q_col] = multipletests(df.loc[m, p_col], method="fdr_bh")[1]
    df["direction"] = np.select(
        [~m, df[q_col] >= 0.05, df.get("sign", pd.Series(0, index=df.index)) > 0],
        ["not tested", "ns", "up"], "down")
    return df


def dir_color(d):
    return {"up": UP, "down": DOWN}.get(d, NS)


# ------------------------------------------------------------------ drawing
def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _room(ax, frac=0.72):
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, lo + (hi - lo) / frac)


def _x_annot(ax, i, r):
    ax.text(i, 0.97, stars(r.get("q", np.nan)), transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7)
    if r.get("n_pairs", 0):
        ax.text(i, 0.89, f"{int(r['n_pos'])}↑/{int(r['n_neg'])}↓",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=5.5, color="#555555")


def _gene_ticks(ax, df, show_detection=True):
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["label"] if "label" in df else df["gene"], fontsize=6.5,
                       style="italic")
    if not show_detection:
        return
    for i, (_, r) in enumerate(df.iterrows()):
        for k, (v, col) in enumerate(((r.get("det_a", np.nan), MALIG),
                                      (r.get("det_b", np.nan), REACT))):
            ax.text(i, 0, pct(v), ha="center", va="top", fontsize=5.5, color=col,
                    transform=offset_copy(ax.get_xaxis_transform(), fig=ax.figure,
                                          y=-13 - 7 * k, units="points"))


def _slope_pair(ax, i, a, b, dx=0.2):
    """One gene's donors at x = i: reactive at i - dx, malignant at i + dx, lines by direction."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    xm, xr = i + dx, i - dx
    for va, vb in zip(a, b):
        col = LINE_UP if va > vb else LINE_DOWN if va < vb else GREY
        ax.plot([xr, xm], [vb, va], color=col, linewidth=0.5, alpha=0.35, zorder=2,
                solid_capstyle="butt")
    for x, v, col in ((xr, b, REACT), (xm, a, MALIG)):
        if not len(v):
            continue
        ax.scatter(np.full(len(v), x), v, s=6, color=col, linewidth=0, zorder=3)
        ax.hlines(np.median(v), x - 0.12, x + 0.12, color="#111111", linewidth=1.2, zorder=4)


def paired_slopes(ax, df, ylabel=XLAB, show_detection=True, dx=0.2):
    """One line per donor joining its two arms, per gene: red if higher in malignant, blue if
    higher in reactive. nb28's `paired`."""
    for i, (_, r) in enumerate(df.iterrows()):
        _slope_pair(ax, i, r.get("vals_a", []), r.get("vals_b", []), dx)
        _x_annot(ax, i, r)
    ax.set_xlim(-0.6, len(df) - 0.4)
    _gene_ticks(ax, df, show_detection)
    ax.set_ylabel(ylabel)
    _clean(ax)
    _room(ax)
    return ax


def dumbbell(ax, df, note_col=None, label_col="ylabel", comparator_label="reactive",
             show_stars=True):
    """One horizontal row per gene: comparator median -> malignant median. nb28's `dumbbell`."""
    labels = df[label_col] if label_col in df else df["gene"].map(it)
    for yi, (_, r) in enumerate(df.iloc[::-1].iterrows()):
        c = dir_color(r.get("direction", "ns"))
        tested = r.get("n_pairs", 0) > 0
        ax.plot([r["med_b"], r["med_a"]], [yi, yi], color=c, linewidth=1.6, zorder=3)
        for v in np.asarray(r.get("vals_a", []), float):
            ax.scatter([v], [yi], s=3, color=c, alpha=0.3, linewidth=0, zorder=2)
        ax.scatter([r["med_b"]], [yi], s=28, facecolor="white", edgecolor=COMPARATOR,
                   linewidth=1.0, zorder=4)
        ax.scatter([r["med_a"]], [yi], s=32, facecolor=c if tested else "white", edgecolor=c,
                   linewidth=1.0, zorder=5)
        note = "" if note_col is None else str(r.get(note_col, "") or "")
        if show_stars or note:
            ax.text(1.01, yi, f"{stars(r.get('q', np.nan)) if show_stars else ''}  {note}"
                    .strip(), transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=6)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(list(labels)[::-1], fontsize=6.5)
    ax.set_ylim(-0.6, len(df) - 0.4)
    # The arm key goes on the axis label, not above the panel: at 1.01 axes-y it collides with
    # a left-aligned title every time.
    ax.set_xlabel(f"{XLAB}      ○ {comparator_label}   ● malignant")
    _clean(ax)
    return ax


ARM_SHORT = {"CD4_malignant": "malignant", "CD4_reactive": "reactive", "CD4_HC": "healthy CD4",
             "CD8": "CD8", "NK": "NK"}
OFF_ARMS = ("CD4_reactive", "CD8", "NK")
WORST = "#d62728"


def shortlist_panel(rec, label_col="ylabel", floor=0.10, gate=0.5, title=None):
    """Three panels sharing one gene axis: the effect, how many patients carry it, and the
    on- vs off-tumour burden.

    `rec` is a `t_paired` record frame (sorted as it should be drawn) carrying, in addition,
    `frac_donors_positive`, `n_donors_malig`, `median_pct_expr`, `worst_T_arm`,
    `worst_T_arm_pct` and `stage`.
    """
    n = len(rec)
    fig = plt.figure(figsize=(9.0, 0.24 * n + 1.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[3.0, 1.15, 1.75], wspace=0.22)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharey=ax0)
    ax2 = fig.add_subplot(gs[2], sharey=ax0)
    y = np.arange(n)[::-1]                        # row i of `rec` sits at y[i], as in dumbbell

    dumbbell(ax0, rec, label_col=label_col)
    ax0.set_xlabel(f"{XLAB}\n○ reactive   ● malignant (donor medians)")

    f = rec["frac_donors_positive"].to_numpy(float)
    nd = rec["n_donors_malig"].to_numpy(float)
    ax1.barh(y, f, height=0.62, color=UP, alpha=0.75, linewidth=0)
    for yi, fi, ni in zip(y, f, nd):
        if np.isfinite(fi) and np.isfinite(ni):
            ax1.text(fi + 0.02, yi, f"{round(fi * ni):.0f}/{ni:.0f}", va="center", fontsize=5.5,
                     color="#444444")
    ax1.axvline(gate, ls="--", lw=0.6, color="#888888")
    ax1.set_xlim(0, 1.32)
    ax1.set_xticks([0, 0.5, 1]); ax1.set_xticklabels(["0", "50%", "100%"])
    ax1.set_xlabel(f"patients positive\n(≥{floor:.0%} of malignant cells)")

    off = rec["worst_T_arm_pct"].to_numpy(float) * 100
    on = rec["median_pct_expr"].to_numpy(float) * 100
    arms = rec["worst_T_arm"].fillna("")
    ax2.barh(y, off, height=0.62, linewidth=0, alpha=0.8,
             color=[ARM_COLOR.get(a, GREY) for a in arms])
    ax2.scatter(on, y, s=22, color=MALIG, zorder=3, linewidth=0)
    xmax = np.nanmax(np.r_[off, on, 1.0])
    for yi, o, m, st in zip(y, off, on, rec.get("stage", pd.Series([""] * n)).fillna("")):
        if st:
            ax2.text(np.nanmax([o, m]) + 0.04 * xmax, yi, st, va="center", fontsize=5,
                     color="#777777")
    ax2.set_xlim(0, xmax * 1.45)
    ax2.set_xlabel("% cells detected\n● malignant   ▬ worst off-tumour arm")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Line2D([], [], marker="o", ls="", color=MALIG, markersize=4,
                label="malignant (median)")]
    h += [Patch(color=ARM_COLOR.get(a, GREY), alpha=0.8, label=f"off-tumour: {ARM_SHORT[a]}")
          for a in OFF_ARMS if a in set(arms)]
    ax2.legend(handles=h, fontsize=5.5, frameon=False, loc="lower left",
               bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2, columnspacing=0.8,
               handlelength=1.2)

    for ax in (ax1, ax2):
        ax.tick_params(axis="y", left=False, labelleft=False)
        _clean(ax)
    if title:
        ax0.set_title(title, loc="left", fontsize=7.5)
    return fig, (ax0, ax1, ax2)


def _facts_panel(ax, facts):
    ax.axis("off")
    step = min(0.12, 1.0 / max(len(facts), 1))
    for i, (k, v) in enumerate(facts.items()):
        ax.text(0, 1 - i * step, k, fontsize=6, color="#777777", va="top",
                transform=ax.transAxes)
        ax.text(0.52, 1 - i * step, str(v), fontsize=6, va="top", transform=ax.transAxes)


def gene_card(long, gene, facts=None, label=None, arm_a="CD4_malignant", arm_b="CD4_reactive",
              arms=("CD4_malignant", "CD4_reactive", "CD4_HC", "CD8", "NK"),
              thresholds=(0.10, 0.25, 0.50), min_donors=5, seed=0):
    """One-row summary of a single antigen: effect, patients positive, burden by arm, and,
    when `facts` is given, a text panel (an ordered dict of short strings). Returns (fig, rec, burden) where `burden` is the per-donor detection
    table the third panel draws.
    """
    label = label or gene
    rec = t_paired(long, [gene], arm_a, arm_b, value="mean_all", min_donors=min_donors)
    r = rec.iloc[0]
    d = long[long["gene"].eq(gene) & long["arm"].isin(arms)][["arm", "donor", "pct_expr"]]
    med = d.groupby("arm")["pct_expr"].median()
    offm = med.reindex([a for a in OFF_ARMS if a in med.index])
    worst = offm.idxmax() if len(offm) else None

    ratios = [1.0, 1.0, 1.6] + ([1.4] if facts else [])
    fig = plt.figure(figsize=(10.0 if facts else 7.4, 2.6))
    gs = fig.add_gridspec(1, len(ratios), width_ratios=ratios, wspace=0.45)
    ax = [fig.add_subplot(gs[i]) for i in range(len(ratios))]

    # 1 -- effect, paired within donor
    _slope_pair(ax[0], 0, r.get("vals_a", []), r.get("vals_b", []))
    ax[0].set_xlim(-0.5, 0.5)
    ax[0].set_xticks([-0.2, 0.2]); ax[0].set_xticklabels(["reactive", "malignant"])
    ax[0].set_ylabel(XLAB)
    ax[0].set_title(f"log2FC {r.get('log2fc', np.nan):+.2f}  {stars(r.get('q', np.nan))}",
                    loc="left", fontsize=7)
    _clean(ax[0])

    # 2 -- patients positive at increasing tumour-positivity cuts
    a = d[d["arm"].eq(arm_a)]["pct_expr"].to_numpy(float)
    fr = [float(np.mean(a >= t)) if len(a) else np.nan for t in thresholds]
    xs = np.arange(len(thresholds))
    ax[1].bar(xs, fr, width=0.62, color=UP, alpha=0.75, linewidth=0)
    for x, v in zip(xs, fr):
        if np.isfinite(v):
            ax[1].text(x, v + 0.03, f"{round(v * len(a)):.0f}/{len(a)}", ha="center",
                       fontsize=5.5, color="#444444")
    ax[1].set_xticks(xs); ax[1].set_xticklabels([f"≥{t:.0%}" for t in thresholds])
    ax[1].set_ylim(0, 1.12)
    ax[1].set_xlabel("malignant cells positive")
    ax[1].set_ylabel("fraction of patients")
    ax[1].set_title("patients positive", loc="left", fontsize=7)
    _clean(ax[1])

    # 3 -- burden by arm
    rng = np.random.default_rng(seed)
    present = [a_ for a_ in arms if a_ in set(d["arm"])]
    for i, a_ in enumerate(present):
        v = d.loc[d["arm"].eq(a_), "pct_expr"].to_numpy(float) * 100
        col = MALIG if a_ == arm_a else WORST if a_ == worst else GREY
        ax[2].scatter(i + rng.normal(0, 0.07, len(v)), v, s=5, color=col, alpha=0.6,
                      linewidth=0)
        ax[2].hlines(np.median(v), i - 0.25, i + 0.25, color="#111111", lw=1.2, zorder=3)
        ax[2].text(i, 1.0, f"{np.median(v):.0f}%", transform=ax[2].get_xaxis_transform(),
                   ha="center", va="bottom", fontsize=5.5, color=col)
    ax[2].set_xticks(range(len(present)))
    ax[2].set_xticklabels([ARM_SHORT.get(a_, a_) for a_ in present], rotation=30, ha="right")
    ax[2].set_ylabel("% cells detected (per donor)")
    ax[2].set_title("burden by arm", loc="left", fontsize=7, pad=9)
    _clean(ax[2])

    # 4 -- facts
    if facts:
        _facts_panel(ax[3], facts)

    wtxt = (f", worst off-tumour {ARM_SHORT.get(worst, worst)} {100 * med[worst]:.0f}%"
            if worst is not None else "")
    fig.suptitle(f"{label}: up in {int(r.get('n_pos', 0))}/{int(r.get('n_pairs', 0))} donors"
                 f"{wtxt}", x=0.01, y=1.0, ha="left", fontsize=8.5, style="italic")
    return fig, rec, d.assign(gene=gene)


def donor_gene_heat(ax, M, cmap="RdBu_r", vmax=None, cbar_label="log2FC malignant / reactive",
                    xlabel_italic=True, stars_by=None):
    """Genes x donors heatmap of a per-donor effect. nb28's `heat`, which nb28 never called.

    `M` is a DataFrame indexed by gene with one column per donor. NaN cells are hatched rather
    than coloured, so a donor that never had both arms is visibly absent instead of looking zero.
    """
    A = M.to_numpy(float)
    vmax = float(np.nanpercentile(np.abs(A), 98)) if vmax is None else vmax
    ax.imshow(np.where(np.isnan(A), 0, A), cmap=cmap, vmin=-vmax, vmax=vmax,
              aspect="auto", interpolation="nearest")
    ys, xs = np.nonzero(np.isnan(A))
    for y, x in zip(ys, xs):
        ax.add_patch(plt.Rectangle((x - .5, y - .5), 1, 1, facecolor="#f2f2f2",
                                   edgecolor="none", hatch="///", linewidth=0))
    ax.set_yticks(range(len(M)))
    ax.set_yticklabels(M.index, fontsize=6, style="italic" if xlabel_italic else "normal")
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(M.columns, fontsize=4.5, rotation=90)
    if stars_by is not None:
        for y, gene in enumerate(M.index):
            ax.text(M.shape[1] - 0.3, y, stars(stars_by.get(gene, np.nan)), fontsize=6,
                    va="center", ha="left")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-vmax, vmax))
    cb = ax.figure.colorbar(sm, ax=ax, fraction=0.02, pad=0.03)
    cb.set_label(cbar_label, fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    return ax


def caption(fig, text):
    """Wrapped small-print under the tight bounding box, as nb28 places it."""
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    fig.text(bb.x0 / fig.get_figwidth(),
             bb.y0 / fig.get_figheight() - 0.015,
             textwrap.fill(text, int(fig.get_figwidth() * 21)),
             fontsize=5.8, color="#555555", ha="left", va="top")


# ------------------------------------------------------------------ output
DROP_FROM_CSV = ("vals_a", "vals_b", "donors")
SHOW_COLS = ["gene", "n_pairs", "med_a", "med_b", "delta", "log2fc", "det_a", "det_b",
             "n_pos", "n_neg", "p", "q", "direction", "reason"]


def show_tests(df, cols=None, max_rows=None):
    """Print the complete test table under a figure -- every row, nothing elided."""
    cols = [c for c in (cols or SHOW_COLS) if c in df.columns]
    with pd.option_context("display.width", 240, "display.max_rows", max_rows,
                           "display.max_colwidth", 70):
        print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))


def save(fig, name, pdf=True, show=True, caption_text=None):
    """PNG (+ PDF) under the notebook's figure prefix, then show and close."""
    if caption_text:
        caption(fig, caption_text)
    paths = [ST.fig(name)]
    if pdf:
        paths.append(ST.fig(name, ext="pdf"))
    for p in paths:
        fig.savefig(p, bbox_inches="tight")
    print("wrote", *[str(p) for p in paths], sep="\n  ")
    if show:
        plt.show()
    plt.close(fig)
    return paths[0]


def save_source(df, name):
    out = df.drop(columns=[c for c in DROP_FROM_CSV if c in df.columns])
    p = ST.tab(name)
    out.to_csv(p, index=False)
    print("wrote", p)
    return p


def logfig(name, what, n_tests=None):
    FIG_INDEX.append({"figure": name, "shows": what, "n_tests": n_tests})


def write_index(name="figure_index"):
    if not FIG_INDEX:
        return None
    return save_source(pd.DataFrame(FIG_INDEX), name)


# ------------------------------------------------------------------ hand-off to another model
LLM_OPEN = "=" * 24 + " COPY FROM HERE " + "=" * 24
LLM_CLOSE = "=" * 24 + " COPY TO HERE " + "=" * 26
_LLM_HEAD = """\
This block is a complete, self-contained numerical report from one notebook of the MF/CTCL
surface-target project. Everything another model needs in order to reason about these numbers is
inside it: what was measured, on which cohort, with which test, and the result tables themselves.
No file outside this block has to be read. Numbers are given to 3 significant figures; tables are
CSV with a header row. A column named `q` is a Benjamini-Hochberg adjusted p-value unless the
section says otherwise.
"""


def _as_csv(obj, max_rows=None, float_fmt="%.3g") -> str:
    """A table as CSV text. A Series becomes two columns; a dict becomes key,value."""
    if isinstance(obj, str):
        return obj.rstrip()
    if isinstance(obj, dict):
        obj = pd.Series(obj)
    if isinstance(obj, pd.Series):
        obj = obj.rename(obj.name or "value").to_frame()
        obj.index.name = obj.index.name or "key"
    # A row label that carries information (gene, arm, donor) must survive into the CSV; a bare
    # RangeIndex must not become a column of row numbers.
    keep_index = bool(obj.index.name) or not isinstance(obj.index, pd.RangeIndex)
    df = obj.reset_index() if keep_index else obj
    note = ""
    if max_rows is not None and len(df) > max_rows:
        note = f"\n[{len(df) - max_rows} further rows omitted from this block]"
        df = df.head(max_rows)
    return df.to_csv(index=False, float_format=float_fmt).rstrip() + note


def llm_export(notebook, title, context, sections, prefix=None, print_block=True):
    """The last cell of every 40_targets notebook: one paste-able block of every result.

    `sections` is a list of dicts, in reading order:

        {"name":  "§3 malignant vs reactive CD4",
         "what":  "one sentence: the quantity",
         "how":   "one sentence: estimator, test, correction, n",
         "table": a DataFrame / Series / dict / preformatted string,
         "max_rows": 40,              # optional; states how many were omitted
         "file":  "tables/tgt42_candidates_skin.csv"}   # optional, where the full table lives

    The same text is written to `tables/<prefix>llm_export.md`, because a long block is scrolled
    out of a Jupyter output pane and the file is not.
    """
    L = [LLM_OPEN, "", f"# {notebook} — {title}", "", _LLM_HEAD, "## Context", "",
         textwrap.dedent(context).strip(), ""]
    for s in sections:
        L += [f"## {s['name']}", "", f"**What:** {s['what']}", "", f"**How:** {s['how']}", ""]
        if s.get("file"):
            L += [f"**Full table on disk:** `{s['file']}`", ""]
        if s.get("table") is not None:
            L += ["```csv", _as_csv(s["table"], s.get("max_rows")), "```", ""]
        if s.get("note"):
            L += [textwrap.dedent(s["note"]).strip(), ""]
    L += [LLM_CLOSE]
    text = "\n".join(L)

    prefix = prefix if prefix is not None else ST.TAB_PREFIX
    ST.TAB_DIR.mkdir(parents=True, exist_ok=True)
    path = ST.TAB_DIR / f"{prefix}llm_export.md"
    path.write_text(text)
    if print_block:
        print(text)
    print(f"\n[{len(text):,} characters. Paste everything between the two banner lines into the "
          f"other model. If the output above is truncated, copy {path} instead.]")
    return path
