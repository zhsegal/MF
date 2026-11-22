# 20_skin/28f_cosmx_figures.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/20_skin/28f_cosmx_figures.ipynb` and is never run._

# 28f — CosMx early-MF skin: figures (post-resolVI)

The spatial notebook of the 28 series: GSE337336, CosMx 1000-plex, 16 MF biopsies on 4 slides — slides 3, 4 patch; 7, 8 plaque; no tumor-stage lesions. Everything here is post-resolVI: cell types are resolVI's predictions (max probability ≥ 0.7, else Unassigned) and expression is resolVI's own-cell term, library 10⁴, log1p. Colours and violin design follow 28c_mc. Its predecessors 28d (raw-count degradome in situ) and 28e (T-cell compartments) are archived in `notebooks/old/`; the one step of theirs this pipeline needs — 28d's lineage calls, which resolVI trains on — is §P here.

| fig | shows |
|---|---|
| aa | two whole slides (4 patch, 7 plaque) by cell type |
| a | two windows per stage: cell types and the epidermis / dermis boundary |
| b | MMP2, MMP9, MMP14 by cell type |
| c | motility / residence gene families in epidermal vs dermal T cells |
| d | Hotspot: which genes each MMP co-localises with, and where T-cell genes fall |

Units: dots are per-piece means (a piece is a spatially contiguous group of fields of view within a slide; 41 pieces). Tests are Wilcoxon signed-rank over pieces holding both groups, BH within a figure. Stage = slide (2 vs 2): shown, not tested.

**Run order** (only to rebuild the inputs; they are on disk): §0 → §P with `REBUILD_LINEAGE = True` (writes `processed/cosmx_s1_cells_28d.parquet`) → `jobs/run_cosmx_resolvi.sh --force` (GPU) → `jobs/run_cosmx_resolvi.sh --export-only --force` → `jobs/run_cosmx_harreman.sh --resolvi` → §0b onward. Otherwise run top to bottom; §P skips itself.

Inputs: `processed/cosmx_s1.h5ad` (positions, cell areas), `processed/cosmx_s1_resolvi_all.h5ad` (`jobs/run_cosmx_resolvi.sh --export-only`), `tables/deg28e_hotspot_*_rvi.csv` (`jobs/run_cosmx_harreman.sh --resolvi`). Outputs: `figures/cosmx/cmx28f_*.png/.pdf`, `tables/cmx28f_*.csv`.

```python  # cell 1
# ============================================================================
# §0  Setup: 28c style, plot helpers, epidermis mask (no data; §0b loads it)
# ============================================================================
import warnings; warnings.filterwarnings("ignore")
import logging
import os
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"]:
    os.environ.setdefault(_v, "8")
import sys
import textwrap
from pathlib import Path

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter, label as ndi_label
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # Carlito absent on compute nodes


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
import cosmx_spatial as CS

PREFIX = "cmx28f_"
FIG_DIR = NB_DIR / "figures" / "cosmx"
TAB_DIR = NB_DIR / "tables"
RESOLVI_ALL = CS.S1_DIR / "processed" / "cosmx_s1_resolvi_all.h5ad"
P_CALL = 0.7          # resolVI probability for a cell-type call
MAX_CELLS = 20_000    # cells per violin (28c)
MIN_CELLS = 10        # cells per piece × group for a dot / a paired test
MIN_N = 5             # pieces for a test
EPI_STEP, EPI_SIGMA, EPI_MAX_HOLE = 5.0, 20.0, 30_000    # µm grid, µm smoothing, µm² largest filled hole
SEED = 0
FIG_INDEX = []

# ---- 28c palette ----
MALIG, MALIG_L = "#7b3294", "#c2a5cf"          # 28c malignant arm (full = epidermis, light = dermis)
UP, DOWN = "#2a9d8f", "#b5542a"                # left group higher | right group higher
TXT = {"up": UP, "down": DOWN}
STAGE_COL = {"patch": "#e8a33d", "plaque": "#c0392b"}   # 28c early / advanced
LIN4 = ["T", "Fibroblast", "Myeloid", "Keratinocyte"]
LIN_COL = {"T": MALIG, "Fibroblast": "#3182bd", "Myeloid": "#e6550d", "Keratinocyte": "#c9a66b",
           "Unassigned": "#d9d9d9"}
LIN_ORDER = LIN4 + ["Unassigned"]
LAYER_COL = {"epidermis": MALIG, "dermis": MALIG_L}
OFFSETS = {2: [-0.2, 0.2], 3: [-0.27, 0.0, 0.27], 4: [-0.33, -0.11, 0.11, 0.33]}
VW = {2: 0.36, 3: 0.25, 4: 0.2}

mpl.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": ["Carlito", "Calibri", "DejaVu Sans"], "font.size": 7,
    "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 400, "pdf.fonttype": 42, "svg.fonttype": "none",
})


def tab28f(name):
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    return TAB_DIR / f"{PREFIX}{name}.csv"


def save_fig(fig, name, what):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{PREFIX}{name}.{ext}")
    plt.show()
    plt.close(fig)
    FIG_INDEX.append({"figure": name, "file": f"{PREFIX}{name}.png", "what": what})
    print(f"  -> {FIG_DIR.name}/{PREFIX}{name}.png/.pdf")


def _stars(q):
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


def sig(q):
    return "n.t." if not np.isfinite(q) else (_stars(q) or "n.s.")


def _room(ax, lo, hi, frac=0.72):
    """Data in the bottom `frac` of the axes; stars and brackets go in the rest (28c)."""
    span = (hi - lo) or 1.0
    bottom = lo - 0.04 * span
    ax.set_ylim(bottom, bottom + (hi - bottom) / frac)


def caption(fig, text):
    """Wrapped caption under everything already drawn (28c)."""
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    fig.text(bb.x0 / fig.get_figwidth(), bb.y0 / fig.get_figheight() - 0.01,
             textwrap.fill(text, int(fig.get_figwidth() * 21)), fontsize=5.8, color="#555555",
             ha="left", va="top")


def _violin(ax, v, pos, color, width):
    v = np.asarray(v)[np.isfinite(v)]
    if len(v) and np.ptp(v) > 0:
        b = ax.violinplot([v], positions=[pos], widths=width, showextrema=False)["bodies"][0]
        b.set_facecolor(color)
        b.set_alpha(0.3)
        b.set_edgecolor("none")


def unit_means(v, groups, arms):
    """Per-piece mean of v in each group; pieces with < MIN_CELLS cells of a group are NaN."""
    df = pd.DataFrame({"piece": PIECE, "g": groups, "v": v})
    m = df[df.g.isin(arms)].groupby(["piece", "g"]).v.agg(["mean", "size"])
    return m["mean"].where(m["size"] >= MIN_CELLS).unstack().reindex(columns=arms)


def paired(U, a, b):
    d = U[[a, b]].dropna()
    if len(d) < MIN_N:
        return {"a": a, "b": b, "n": len(d), "p": np.nan, "diff": np.nan}
    return {"a": a, "b": b, "n": len(d), "p": wilcoxon(d[a], d[b]).pvalue, "diff": (d[a] - d[b]).median()}


def slot(ax, x, v, groups, arms, colors, U, tests):
    """28c slot at x: violin per group (≤ MAX_CELLS cells), per-piece dots, median tick, and one
    bracket per test (dict a, b, q, diff, level). Returns the (lo, hi) drawn."""
    w = VW[len(arms)]
    pos = {a: x + d for a, d in zip(arms, OFFSETS[len(arms)])}
    rng = np.random.default_rng(SEED)
    lo, hi = [], []
    for a in arms:
        c = v[groups == a]
        if len(c) > MAX_CELLS:
            c = rng.choice(c, MAX_CELLS, replace=False)
        _violin(ax, c, pos[a], colors[a], w)
        if len(c) >= 20:
            lo.append(float(np.quantile(c, 0.01))); hi.append(float(np.quantile(c, 0.98)))
        u = U[a].dropna().to_numpy()
        ax.scatter(pos[a] + rng.uniform(-0.2 * w, 0.2 * w, len(u)), u, s=5, color=colors[a],
                   edgecolor="white", linewidth=0.2, zorder=3)
        if len(u):
            ax.hlines(np.median(u), pos[a] - 0.35 * w, pos[a] + 0.35 * w, color="#111111",
                      linewidth=1.1, zorder=4)
            lo.append(float(u.min())); hi.append(float(u.max()))
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for t in tests:
        x0, x1 = pos[t["a"]], pos[t["b"]]
        y = 0.76 + 0.08 * t.get("level", 0)
        ax.plot([x0, x0, x1, x1], [y - 0.02, y, y, y - 0.02], transform=tr, color="#555555",
                linewidth=0.5, clip_on=False)
        col = TXT["up"] if t["diff"] > 0 else TXT["down"] if t["diff"] < 0 else "#333333"
        ax.text((x0 + x1) / 2, y, sig(t["q"]), transform=tr, ha="center", va="bottom", fontsize=5,
                color=col if np.isfinite(t["q"]) and t["q"] < 0.05 else "#333333")
    return (min(lo) if lo else 0.0), (max(hi) if hi else 1.0)


def epidermis_masks(step=EPI_STEP, sigma=EPI_SIGMA, max_hole_um2=EPI_MAX_HOLE):
    """Per piece, a grid mask of the epidermis: keratinocyte fraction ≥ 0.5 after `sigma` µm
    smoothing, closed, and with enclosed holes up to `max_hole_um2` filled -- an intraepidermal
    T-cell cluster (Pautrier-like), whose own neighbours are T cells, stays inside the epidermis.
    Pieces with < 50 keratinocytes are all dermis. Returns the per-cell layer and
    {piece: (x centres, y centres, mask)} for drawing the edge."""
    layer = np.full(N, "dermis", dtype=object)
    masks = {}
    is_k = LIN == "Keratinocyte"
    for p in np.unique(PIECE):
        idx = np.flatnonzero(PIECE == p)
        if is_k[idx].sum() < 50:
            continue
        lo_, hi_ = XY[idx].min(0) - 3 * sigma, XY[idx].max(0) + 3 * sigma
        ex, ey = np.arange(lo_[0], hi_[0] + step, step), np.arange(lo_[1], hi_[1] + step, step)
        A, _, _ = np.histogram2d(XY[idx, 0], XY[idx, 1], bins=[ex, ey])
        k = idx[is_k[idx]]
        K, _, _ = np.histogram2d(XY[k, 0], XY[k, 1], bins=[ex, ey])
        As, Ks = gaussian_filter(A, sigma / step), gaussian_filter(K, sigma / step)
        m = binary_closing((As > 0.01 * As.max()) & (Ks >= 0.5 * As), iterations=2)
        holes, n = ndi_label(binary_fill_holes(m) & ~m)
        if n:
            area = np.bincount(holes.ravel())[1:] * step * step
            m |= np.isin(holes, 1 + np.flatnonzero(area <= max_hole_um2))
        ix = np.clip(np.digitize(XY[idx, 0], ex) - 1, 0, len(ex) - 2)
        iy = np.clip(np.digitize(XY[idx, 1], ey) - 1, 0, len(ey) - 2)
        layer[idx] = np.where(m[ix, iy], "epidermis", "dermis")
        masks[p] = ((ex[:-1] + ex[1:]) / 2, (ey[:-1] + ey[1:]) / 2, m)
    return layer.astype(str), masks


def draw_edge(ax, piece, shift=(0.0, 0.0), lw=0.9):
    """Dashed epidermis edge of a piece (the mask boundary)."""
    if piece in MASKS:
        cx, cy, m = MASKS[piece]
        ax.contour(cx + shift[0], cy + shift[1], m.T.astype(float), levels=[0.5], colors="#111111",
                   linewidths=lw, linestyles="--")
```

```python  # cell 2
# ============================================================================
# §P  Lineage calls for resolVI training (was 28d §0–§1) — only to rebuild the inputs
# ============================================================================
# jobs/run_cosmx_resolvi.py trains semi-supervised on these calls (jobs/run_cosmx_transfer.py
# reads them too). Skipped while CS.CELLS_28D exists; REBUILD_LINEAGE = True redoes it.
# Leiden on the panel, each cluster called by marker score and checked against the stains.
# InSituType's labels are not used: its "B.cell" class is low-count cells with weak MS4A1/CD79A
# and its "Melanocyte" class is PanCK-high/KRT14-high. Melanocyte markers (PMEL, MLANA, TYRP1,
# DCT, MITF, SOX10) are off the panel, so melanocytes fall into Keratinocyte/Unassigned.
REBUILD_LINEAGE = False
MIN_COUNTS = 20     # CosMx cells below this are mostly segmentation debris
MANUAL = {}         # {"<leiden id>": "<lineage>"}: overrides, set after reviewing the table below
if REBUILD_LINEAGE or not CS.CELLS_28D.exists():
    import scanpy as sc

    a = CS.load_s1(cache=True)
    assert a.n_obs == CS.N_CELLS_EXPECTED, a.n_obs
    mix = CS.patient_mixing(a)
    print("GEO patient label vs space (per slide × FOV, median):")
    print(mix.groupby("slide")[["n_patients", "nn_same", "random"]].median().round(2))
    print("nn_same ≈ random → the GEO patient label carries no spatial structure; unit = piece.")
    a = a[a.obs.nCount_RNA >= MIN_COUNTS].copy()
    print(f"after nCount_RNA ≥ {MIN_COUNTS}: {a.n_obs:,} cells")

    a.X = CS.lognorm(a)
    scores, used = CS.lineage_scores(a)
    print("markers on panel:", used)
    sc.pp.pca(a, n_comps=30)
    sc.pp.neighbors(a, n_neighbors=15)
    sc.tl.leiden(a, resolution=1.0, key_added="leiden", flavor="igraph", n_iterations=2, random_state=0)

    CALLS = CS.call_clusters(a, scores, "leiden")
    CALLS["final"] = [MANUAL.get(c, v) for c, v in zip(CALLS.index, CALLS.call)]
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(CALLS[["n", "top", "margin", "call", "final", "PanCK_rel", "CD45_rel", "CD68_rel", "nCount",
                     "top_slide_frac", "flag_epi_immune", "insitutype_mode"]].round(2))
    CALLS.to_csv(tab28f("P_cluster_calls"))
    # Review before trusting the lineages:
    #  - flag_epi_immune: keratinocyte by RNA but T-level CD45 → likely intraepidermal T cells with
    #    keratinocyte spillover (epidermotropism). Kept as Keratinocyte unless overridden in MANUAL.
    #  - top_slide_frac > 0.9: a cluster that is one slide is batch until shown otherwise.
    print("flag_epi_immune:", CALLS.index[CALLS.flag_epi_immune].tolist(),
          "| single-slide clusters:", CALLS.index[CALLS.top_slide_frac > 0.9].tolist())

    a.obs["lineage"] = pd.Categorical(a.obs.leiden.map(CALLS.final).astype(str),
                                      categories=[l for l in CS.LINEAGE_COLORS if l in set(CALLS.final)])
    a.obs["t_sub"] = CS.t_subset(a)
    a.obs["depth_um"] = CS.dist_to_lineage(a, "Keratinocyte")
    print(pd.crosstab(a.obs.lineage, a.obs.insitutype))
    a.obs[["slide", "stage", "fov", "fov_key", "piece", "leiden", "lineage", "t_sub", "insitutype",
           "depth_um", "nCount_RNA"]].assign(x_um=a.obsm["spatial"][:, 0], y_um=a.obsm["spatial"][:, 1]
                                             ).to_parquet(CS.CELLS_28D)
    print(f"-> {CS.CELLS_28D}\nnext: jobs/run_cosmx_resolvi.sh --force; jobs/run_cosmx_resolvi.sh --export-only "
          "--force; jobs/run_cosmx_harreman.sh --resolvi; then run from §0b")
    del a
else:
    print(f"lineage calls present ({CS.CELLS_28D.name}); REBUILD_LINEAGE = True to redo them")
```

```python  # cell 3
# ============================================================================
# §0b  Data: positions from the CosMx object, everything else from resolVI
# ============================================================================
adata = CS.load_s1(cache=True, verbose=False)
R = ad.read_h5ad(RESOLVI_ALL)
assert R.obs_names.equals(adata.obs_names) and R.var_names.equals(adata.var_names), "resolVI export mismatch"
N = adata.n_obs
PROB = R.obs.filter(regex="^p_")
LIN = np.where(PROB.max(1).to_numpy() >= P_CALL, PROB.idxmax(1).str[2:].to_numpy(), "Unassigned")
EXPR = np.log1p(np.asarray(R.X, np.float32))                 # log1p CP10K (resolVI)
GI = {g: i for i, g in enumerate(R.var_names)}
expr = lambda g: EXPR[:, GI[g]]
YLAB = "log1p CP10K (resolVI)"
MEAN_LIN = pd.DataFrame(EXPR, columns=R.var_names).groupby(LIN).mean()

SLIDE = adata.obs.slide.astype(str).to_numpy()
PIECE = adata.obs.piece.astype(str).to_numpy()
SLIDES = list(adata.obs.slide.cat.categories)
SLIDE_STAGE = adata.obs.groupby("slide", observed=True).stage.first().astype(str).to_dict()
XY = adata.obsm["spatial"]
IS_T = LIN == "T"
LAYER, MASKS = epidermis_masks()

print(f"cells {N:,} | resolVI calls (p ≥ {P_CALL}):", pd.Series(LIN).value_counts().to_dict())
print(f"epidermis masks for {len(MASKS)} of {len(np.unique(PIECE))} pieces; T cells by layer and slide:")
print(pd.crosstab(SLIDE[IS_T], LAYER[IS_T]).rename_axis(index="slide", columns=None).to_string())
```

```python  # cell 4
# ============================================================================
# Fig aa — two whole slides by cell type
# ============================================================================
# Every imaged cell of the slide. Only the fields of view (FOVs, 0.51 mm) were imaged, and they
# are scattered over the slide, so the tissue pieces are packed side by side at true scale
# (distances *between* pieces are not real). Dashed line = epidermis edge (§0 mask).
SLIDES_WHOLE = ["4", "7"]
BOXES = CS.fov_boxes(adata)
PACK = CS.pack_pieces(BOXES).set_index("piece")
SHIFT = PACK.loc[PIECE, ["dx", "dy"]].to_numpy()
XYP = XY + SHIFT

fig, axes = plt.subplots(1, len(SLIDES_WHOLE), figsize=(7.2, 4.0),
                         gridspec_kw={"width_ratios": [1, 1]})
summ = {}
for ax, s in zip(axes, SLIDES_WHOLE):
    idx = np.flatnonzero(SLIDE == s)
    idx = idx[np.argsort(LIN[idx] != "Unassigned", kind="stable")]       # Unassigned drawn first
    ax.scatter(*XYP[idx].T, s=0.9, linewidths=0, rasterized=True, c=[LIN_COL[l] for l in LIN[idx]])
    for p in np.unique(PIECE[idx]):
        draw_edge(ax, p, shift=tuple(PACK.loc[p, ["dx", "dy"]]), lw=0.45)
        r = PACK.loc[p]
        ax.text(r.px0, r.py1 + 30, p.split("_")[1], fontsize=5, color="#555555", va="bottom")
    x0, y0 = PACK[PACK.slide == s].px0.min(), PACK[PACK.slide == s].py0.min()
    ax.plot([x0, x0 + 500], [y0 - 80] * 2, color="#111111", lw=1.2)
    ax.text(x0 + 250, y0 - 110, "500 µm", ha="center", va="top", fontsize=6)
    comp = pd.Series(LIN[idx]).value_counts(normalize=True)
    t = idx[IS_T[idx]]
    summ[s] = {"ker": comp.get("Keratinocyte", 0), "T": comp.get("T", 0),
               "T_derm": (LAYER[t] == "dermis").mean() if len(t) else np.nan}
    ax.set_title(f"slide {s} · {SLIDE_STAGE[s]}\n{len(idx):,} cells · {summ[s]['ker']:.0%} keratinocyte · "
                 f"{summ[s]['T']:.0%} T", color=STAGE_COL[SLIDE_STAGE[s]], fontsize=8)
    ax.set_aspect("equal"); ax.axis("off")
handles = [plt.Line2D([], [], marker="o", ls="", ms=4, color=LIN_COL[l], label=l) for l in LIN_ORDER]
handles.append(plt.Line2D([], [], color="#111111", ls="--", lw=0.8, label="epidermis edge"))
fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=6.5,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Whole slides, patch (slide 4) and plaque (slide 7): cell types", x=0.02, ha="left", fontsize=9, y=1.06)
save_fig(fig, "aa_whole_slides", "slides 4 (patch) and 7 (plaque): every cell by resolVI cell type, pieces packed")
```

```
<Figure size 720x400 with 2 Axes>
  -> cosmx/cmx28f_aa_whole_slides.png/.pdf
```

```python  # cell 5
# ============================================================================
# Fig a — two windows per stage: cell types and the epidermis / dermis boundary
# ============================================================================
# One window per imaged FOV, cropped to its tissue. Chosen automatically: the FOV must hold both
# layers (keratinocytes 25–70% of its cells) and ≥ 30 T cells; ranked by T cells, at most one per
# piece. Cells are discs of their segmented area; dashed line = epidermis edge (§0 mask).
# Override with your picks from the candidate sheets below: WIN_AT = [fov_key, ...] (patch first).
N_WIN, WIN_AT = 2, None
FOV = adata.obs.fov_key.astype(str).to_numpy()
AREA_UM2 = adata.obs.Area.to_numpy() * abs(np.prod(BOXES.attrs["um_per_px"]))
fv = pd.DataFrame({"fov": FOV, "piece": PIECE, "slide": SLIDE, "ker": LIN == "Keratinocyte", "T": IS_T,
                   "T_epi": IS_T & (LAYER == "epidermis"), "T_derm": IS_T & (LAYER == "dermis")})
fv = fv.groupby(["fov", "piece", "slide"], observed=True).agg(
    n=("ker", "size"), ker=("ker", "mean"), T=("T", "sum"), T_epi=("T_epi", "sum"), T_derm=("T_derm", "sum")).reset_index()
fv["stage"] = fv.slide.map(SLIDE_STAGE)
ok = fv[fv.ker.between(0.25, 0.70) & (fv["T"] >= 30)].sort_values("T", ascending=False)
if WIN_AT is None:
    WIN_AT = [f for st in ["patch", "plaque"]
              for f in ok[ok.stage == st].drop_duplicates("piece").fov.head(N_WIN)]
WIN = fv.set_index("fov").loc[WIN_AT].reset_index()
WIN.to_csv(tab28f("a_windows"), index=False)
print(WIN.to_string(index=False))


def draw_window(ax, w, panel_in, pad=15.0, title=None):
    """One FOV cropped to its tissue: cells as true-area discs by cell type, epidermis edge,
    scale bar. `w` is a row of `fv`."""
    idx = np.flatnonzero(FOV == w.fov)
    lo_, hi_ = XY[idx].min(0), XY[idx].max(0)
    size = (hi_ - lo_).max() + 2 * pad
    c0 = (lo_ + hi_) / 2 - size / 2                                             # square crop around the tissue
    d_pt = np.sqrt(4 * AREA_UM2[idx] / np.pi) / size * panel_in * 72          # true diameter in points
    order = np.argsort(LIN[idx] == "T", kind="stable")                          # T cells on top
    ax.scatter(*XY[idx[order]].T, s=d_pt[order] ** 2, c=[LIN_COL[l] for l in LIN[idx[order]]], linewidths=0)
    draw_edge(ax, w.piece, lw=0.9 * panel_in / 3.1)
    bar = 50 if size < 300 else 100
    ax.plot([c0[0] + 0.05 * size, c0[0] + 0.05 * size + bar], [c0[1] + 0.04 * size] * 2, color="#111111", lw=1.2)
    ax.text(c0[0] + 0.05 * size + bar / 2, c0[1] + 0.055 * size, f"{bar} µm", ha="center", va="bottom",
            fontsize=6 if panel_in > 2.5 else 5)
    ax.set_xlim(c0[0], c0[0] + size); ax.set_ylim(c0[1], c0[1] + size)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("#b0b0b0"); sp.set_linewidth(0.5)
    ax.set_title(title or f"{w.stage} · slide {w.slide} · {w.piece}\nT cells: {w.T_epi} in the epidermis, "
                          f"{w.T_derm} below", color=STAGE_COL[w.stage], fontsize=7.5 if panel_in > 2.5 else 6)


LEGEND = [plt.Line2D([], [], marker="o", ls="", ms=4, color=LIN_COL[l], label=l) for l in LIN_ORDER]
LEGEND.append(plt.Line2D([], [], color="#111111", ls="--", lw=0.9, label="epidermis edge"))
PANEL_IN = 3.1
fig, axes = plt.subplots(2, N_WIN, figsize=(PANEL_IN * N_WIN + 0.6, PANEL_IN * 2 + 0.9), squeeze=False)
for ax in axes.ravel()[len(WIN):]:
    ax.axis("off")
for ax, w in zip(axes.ravel(), WIN.itertuples()):
    draw_window(ax, w, PANEL_IN)
fig.legend(handles=LEGEND, loc="lower center", ncol=len(LEGEND), frameon=False, fontsize=6.5,
           bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Patch and plaque fields of view: cell types and epidermis edge", x=0.02, ha="left", fontsize=9)
save_fig(fig, "a_windows", "2 patch + 2 plaque FOVs: resolVI cell types and the epidermis edge")
```

```
fov piece slide   n      ker   T  T_epi  T_derm  stage
4_36 4_p03     4 469 0.503198 183     32     151  patch
4_25 4_p01     4 276 0.565217 111     66      45  patch
8_22 8_p10     8 808 0.315594 195      3     192 plaque
8_28 8_p09     8 420 0.314286 167      6     161 plaque

<Figure size 680x710 with 4 Axes>
  -> cosmx/cmx28f_a_windows.png/.pdf
```

```python  # cell 6
# ============================================================================
# Fig a — candidate sheets: more windows per stage to choose from
# ============================================================================
# Looser gate than Fig a (keratinocytes 20–80% of the FOV, ≥ 20 T cells) and several FOVs per
# piece allowed; up to N_CAND per stage, ranked by T cells. Each panel title starts with its FOV
# key: copy the ones you want into WIN_AT in the Fig a cell (patch first, then plaque) and re-run.
N_CAND, CAND_COLS, CAND_IN = 16, 4, 2.2
CAND = (fv[fv.ker.between(0.20, 0.80) & (fv["T"] >= 20)]
        .sort_values(["stage", "T"], ascending=[True, False]).groupby("stage").head(N_CAND))
CAND.to_csv(tab28f("a_window_candidates"), index=False)
print(CAND.groupby("stage").size().to_dict(), "candidates (of",
      fv[fv.ker.between(0.20, 0.80) & (fv["T"] >= 20)].groupby("stage").size().to_dict(), "passing)")
for st in ["patch", "plaque"]:
    c = CAND[CAND.stage == st]
    if not len(c):
        continue
    rows_ = int(np.ceil(len(c) / CAND_COLS))
    fig, axes = plt.subplots(rows_, CAND_COLS, figsize=(CAND_IN * CAND_COLS + 0.4, CAND_IN * rows_ + 0.9),
                             squeeze=False)
    for ax in axes.ravel()[len(c):]:
        ax.axis("off")
    for ax, w in zip(axes.ravel(), c.itertuples()):
        draw_window(ax, w, CAND_IN, title=f"{w.fov} · {w.piece}\nT {w.T_epi} in epidermis · {w.T_derm} below · "
                                          f"ker {w.ker:.0%}")
    fig.legend(handles=LEGEND, loc="lower center", ncol=len(LEGEND), frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{st.capitalize()} candidates for Fig a — {len(c)} fields of view, most T cells first "
                 f"(title = FOV key for WIN_AT)", x=0.02, ha="left", fontsize=9, color=STAGE_COL[st])
    save_fig(fig, f"a_candidates_{st}", f"{len(c)} candidate {st} windows (FOV key in each title) for Fig a")
```

```
{'patch': 16, 'plaque': 16} candidates (of {'patch': 19, 'plaque': 24} passing)

<Figure size 920x970 with 16 Axes>
  -> cosmx/cmx28f_a_candidates_patch.png/.pdf

<Figure size 920x970 with 16 Axes>
  -> cosmx/cmx28f_a_candidates_plaque.png/.pdf
```

```python  # cell 7
# ============================================================================
# Fig b — MMP2, MMP9, MMP14 by cell type (all slides)
# ============================================================================
# Per gene: violin per cell type (≤ 20k cells), per-piece means as dots, median tick. Brackets:
# T vs each other cell type, paired Wilcoxon over pieces holding ≥ MIN_CELLS of both; BH over the
# figure. Teal = T higher, rust = T lower.
MMPS = ["MMP2", "MMP9", "MMP14"]
rows, UB = [], {}
for g in MMPS:
    UB[g] = unit_means(expr(g), LIN, LIN4)
    for lev, other in enumerate(LIN4[1:]):
        rows.append({"gene": g, "level": lev, **paired(UB[g], "T", other)})
FB = pd.DataFrame(rows)
FB["q"] = multipletests(FB.p.fillna(1), method="fdr_bh")[1]
FB.loc[FB.p.isna(), "q"] = np.nan
FB.to_csv(tab28f("b_mmp_by_celltype"), index=False)
print(FB[["gene", "b", "n", "diff", "q"]].round(3).to_string(index=False))

fig, axes = plt.subplots(1, len(MMPS), figsize=(8.4, 3.8), gridspec_kw={"wspace": 0.35})
for ax, g in zip(axes, MMPS):
    lo, hi = slot(ax, 0, expr(g), LIN, LIN4, LIN_COL, UB[g], FB[FB.gene == g].to_dict("records"))
    _room(ax, lo, hi, frac=0.72)
    ax.set_xticks(OFFSETS[4]); ax.set_xticklabels(LIN4, rotation=45, ha="right", fontsize=7)
    ax.set_xlim(-0.5, 0.5)
    ax.set_title(g, fontstyle="italic", fontsize=9)
axes[0].set_ylabel(YLAB, fontsize=7)
med = {g: UB[g].median() for g in MMPS}                       # per-piece medians, log1p scale
top = {g: med[g].idxmax() for g in MMPS}
fold = {g: np.expm1(med[g][top[g]]) / max(np.expm1(med[g]["T"]), 1e-9) for g in MMPS}
fig.suptitle("MMP2, MMP9 and MMP14 expression by cell type", x=0.02, ha="left", fontsize=9.5)
save_fig(fig, "b_mmp_by_celltype", "MMP2 / MMP9 / MMP14 per cell type, resolVI expression, all slides")
```

```
gene            b  n   diff   q
 MMP2   Fibroblast 38 -2.025 0.0
 MMP2      Myeloid 38 -0.485 0.0
 MMP2 Keratinocyte 38  1.225 0.0
 MMP9   Fibroblast 38  0.346 0.0
 MMP9      Myeloid 38 -1.139 0.0
 MMP9 Keratinocyte 38  1.071 0.0
MMP14   Fibroblast 38 -1.178 0.0
MMP14      Myeloid 38 -0.620 0.0
MMP14 Keratinocyte 38 -0.377 0.0

<Figure size 840x380 with 3 Axes>
  -> cosmx/cmx28f_b_mmp_by_celltype.png/.pdf
```

```python  # cell 8
# ============================================================================
# Fig c — amoeboid-motion and chemotaxis genes: epidermal vs dermal T cells (all slides)
# ============================================================================
# 28c's amoeboid programs are almost entirely off the 1000-plex panel; these are the panel genes
# closest to each function. Epidermal = inside the §0 epidermis mask; dermal = every other T
# cell. Paired Wilcoxon over pieces holding ≥ MIN_CELLS T cells in both layers; BH over the figure.
# Rows are {row title: {section: genes}}; a section title is drawn when a row has more than one.
FIG_C_ROWS = {
    "amoeboid motion":    {"": ["MYL9", "MYL12A", "RAC1", "RAC2", "ARHGDIB", "PFN1", "LMNA", "AQP3"]},
    "cortex & filaments": {"": ["VIM", "EZR", "S100A4"]},
    "chemotaxis":         {"skin": ["ITGAE", "CD69", "CXCR6", "CXCR3", "CCR10"],
                           "circulation": ["KLF2", "SELL", "CCR7", "CXCR4"]},
}
FIG_C_ROWS = {r: {s: [g for g in gs if g in GI] for s, gs in secs.items()} for r, secs in FIG_C_ROWS.items()}
REACT = "#5e81ac"                                  # 28c reactive arm
LAYERS = ["epidermis", "dermis"]
LAYER_COL_C = {"epidermis": MALIG, "dermis": REACT}
GRP_T = np.where(IS_T, LAYER, "")

UC, rows = {}, []
for r, secs in FIG_C_ROWS.items():
    for s, gs in secs.items():
        for g in gs:
            UC[g] = unit_means(expr(g), GRP_T, LAYERS)
            rows.append({"row": r, "section": s, "gene": g, **paired(UC[g], "epidermis", "dermis")})
FC = pd.DataFrame(rows)
FC["q"] = multipletests(FC.p.fillna(1), method="fdr_bh")[1]
FC.loc[FC.p.isna(), "q"] = np.nan
FC.to_csv(tab28f("c_motility_epi_vs_derm"), index=False)
print(f"pieces with ≥ {MIN_CELLS} T cells in both layers: {FC.n.max()}")
print(FC[["row", "section", "gene", "n", "diff", "q"]].round(3).to_string(index=False))

n_slots = {r: sum(len(gs) for gs in secs.values()) for r, secs in FIG_C_ROWS.items()}
cols = max(n_slots.values())
fig = plt.figure(figsize=(1.0 + 0.75 * cols, 1.7 * len(FIG_C_ROWS) + 0.3))
gs_ = fig.add_gridspec(len(FIG_C_ROWS), cols, hspace=0.9, wspace=0.0, top=0.9, bottom=0.08)
for i, (r, secs) in enumerate(FIG_C_ROWS.items()):
    ax = fig.add_subplot(gs_[i, :n_slots[r]])
    genes = [g for gs in secs.values() for g in gs]
    lo, hi = [], []
    for x, g in enumerate(genes):
        l, h = slot(ax, x, expr(g), GRP_T, LAYERS, LAYER_COL_C, UC[g], FC[FC.gene == g].to_dict("records"))
        lo.append(l); hi.append(h)
    _room(ax, min(lo), max(hi), frac=0.72)
    if len(secs) > 1:
        x0 = 0
        for k, (s, gs) in enumerate(secs.items()):
            if k:
                ax.axvline(x0 - 0.5, color="#9e9e9e", lw=0.6, ls=":")
            ax.text(x0 + (len(gs) - 1) / 2, 1.0, s, transform=blended_transform_factory(ax.transData, ax.transAxes),
                    ha="center", va="bottom", fontsize=7, color="#555555", fontstyle="italic")
            x0 += len(gs)
    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, fontsize=7, fontstyle="italic")
    ax.set_xlim(-0.5, len(genes) - 0.5)
    ax.set_ylabel(YLAB, fontsize=6)
    ax.set_title(r, loc="left", fontsize=8, fontweight="bold", pad=10 if len(secs) > 1 else 4)
handles = [plt.Line2D([], [], marker="o", ls="", ms=5, color=LAYER_COL_C[l], label=f"T cells, {l}") for l in LAYERS]
fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.98, 0.99))
fig.suptitle("Amoeboid-motion and chemotaxis genes in epidermal vs dermal T cells", x=0.02, ha="left",
             fontsize=9.5, y=0.99)
sig_ = FC[FC.q < 0.05]
UP_TXT = ", ".join(sig_[sig_["diff"] > 0].gene) or "none"
DOWN_TXT = ", ".join(sig_[sig_["diff"] < 0].gene) or "none"
save_fig(fig, "c_motility_epi_vs_derm", "amoeboid-motion / cortex / chemotaxis genes, epidermal vs dermal T cells")
```

```
pieces with ≥ 10 T cells in both layers: 19
               row     section    gene  n   diff     q
   amoeboid motion                MYL9 19 -0.203 0.022
   amoeboid motion              MYL12A 19  0.013 0.872
   amoeboid motion                RAC1 19  0.053 0.091
   amoeboid motion                RAC2 19 -0.000 0.677
   amoeboid motion             ARHGDIB 19 -0.099 0.004
   amoeboid motion                PFN1 19 -0.017 0.481
   amoeboid motion                LMNA 19  0.154 0.004
   amoeboid motion                AQP3 19  0.530 0.003
cortex & filaments                 VIM 19 -0.027 0.700
cortex & filaments                 EZR 19  0.058 0.042
cortex & filaments              S100A4 19 -0.019 0.620
        chemotaxis        skin   ITGAE 19  0.147 0.003
        chemotaxis        skin    CD69 19  0.243 0.003
        chemotaxis        skin   CXCR6 19  0.103 0.006
        chemotaxis        skin   CXCR3 19 -0.001 1.000
        chemotaxis        skin   CCR10 19  0.145 0.006
        chemotaxis circulation    KLF2 19 -0.285 0.003
        chemotaxis circulation    SELL 19 -0.165 0.003
        chemotaxis circulation    CCR7 19  0.019 0.620
        chemotaxis circulation   CXCR4 19  0.013 0.820

<Figure size 775x540 with 3 Axes>
  -> cosmx/cmx28f_c_motility_epi_vs_derm.png/.pdf
```

```python  # cell 9
# ============================================================================
# Fig d — Hotspot: which genes each MMP co-localises with, and where T-cell genes fall
# ============================================================================
# From jobs/run_cosmx_harreman.py --resolvi: Hotspot on resolVI expression (log1p, z-scored per
# gene within slide, 10 nearest neighbours within slide). Local-correlation Z of gene a with b:
# how much a in a cell co-varies with b in its spatial neighbours (≈ "found next to each other").
# resolVI output is smooth, so every Z is large; read the ranks. Per MMP, every tested gene
# ranked by Z (grey); the top partners are coloured by the cell type that expresses them most;
# T-cell-specific genes (CD3E/D/G, CD2; CD4 is also myeloid) are purple, with per-slide agreement.
LCZ = pd.read_csv(TAB_DIR / "deg28e_hotspot_lcz_rvi.csv", index_col=0)
PS = pd.read_csv(TAB_DIR / "deg28e_hotspot_focus_per_slide_rvi.csv")
T_GENES = [g for g in ["CD3E", "CD3D", "CD3G", "CD2"] if g in LCZ.index]
N_TOP = 6
rows = []
fig, axes = plt.subplots(1, len(MMPS), figsize=(7.2, 3.1))
for ax, g in zip(axes, MMPS):
    z = LCZ.loc[g].drop(g).sort_values(ascending=False)
    n = len(z)
    ax.scatter(np.arange(1, n + 1), z.to_numpy(), s=3, color="#bdbdbd", linewidths=0, zorder=1)
    ax.axhline(0, color="#111111", lw=0.5)
    tr = ax.transAxes
    ax.text(0.62, 0.97, "top partners", transform=tr, fontsize=5.5, color="#555555", fontweight="bold", va="top")
    for k, (gene, val) in enumerate(z.head(N_TOP).items()):
        lin = MEAN_LIN.loc[LIN4, gene].idxmax()
        ax.scatter(k + 1, val, s=10, color=LIN_COL[lin], zorder=3)
        ax.text(0.62, 0.91 - 0.05 * k, f"{k + 1}. {gene}", transform=tr, fontsize=5.5, fontstyle="italic",
                color=LIN_COL[lin], va="top")
        rows.append({"mmp": g, "gene": gene, "kind": "top", "rank": k + 1, "z": val, "lineage": lin})
    ax.text(0.04, 0.36, "T-cell genes", transform=tr, fontsize=5.5, color=MALIG, fontweight="bold", va="top")
    for j, t in enumerate(T_GENES):
        k = z.index.get_loc(t) + 1
        n_sl = int((PS[(PS.gene_a == g) & (PS.gene_b == t)].z > 1.96).sum())
        ax.scatter(k, z[t], s=14, color=MALIG, edgecolor="white", linewidth=0.3, zorder=4)
        ax.text(0.04, 0.30 - 0.05 * j, f"{t}: rank {k} / {n}", transform=tr, fontsize=5.5, color=MALIG, va="top")
        rows.append({"mmp": g, "gene": t, "kind": "T", "rank": k, "z": z[t], "slides_z_gt_1.96": n_sl})
    ax.set_title(g, fontstyle="italic", fontsize=8)
    ax.set_xlabel("gene rank", fontsize=6.5)
axes[0].set_ylabel("Hotspot local-correlation Z", fontsize=6)
FD = pd.DataFrame(rows)
FD.to_csv(tab28f("d_hotspot_ranks"), index=False)
print(FD.round(2).to_string(index=False))
t_med = FD[FD.kind == "T"].groupby("mmp")["rank"].median()
n_tested = len(LCZ) - 1
handles = [plt.Line2D([], [], marker="o", ls="", ms=4, color=LIN_COL[l], label=f"{l} gene") for l in LIN4]
fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=6.5,
           bbox_to_anchor=(0.5, -0.06))
fig.suptitle("Hotspot spatial co-localisation of MMP2, MMP9 and MMP14 with all tested genes",
             x=0.02, ha="left", fontsize=9)
save_fig(fig, "d_hotspot_mmp_partners", "Hotspot local correlation of MMP2 / MMP9 / MMP14 with every gene; T-cell genes marked")
```

```
mmp   gene kind  rank      z    lineage  slides_z_gt_1.96
 MMP2 COL3A1  top     1 177.53 Fibroblast               NaN
 MMP2 COL6A3  top     2 172.98 Fibroblast               NaN
 MMP2 COL6A2  top     3 172.20 Fibroblast               NaN
 MMP2 CCDC80  top     4 171.31 Fibroblast               NaN
 MMP2 COL1A2  top     5 170.33 Fibroblast               NaN
 MMP2 COL1A1  top     6 169.80 Fibroblast               NaN
 MMP2   CD3E    T    95  72.28        NaN               4.0
 MMP2   CD3D    T    93  74.97        NaN               4.0
 MMP2   CD3G    T   109  68.26        NaN               4.0
 MMP2    CD2    T   115  61.47        NaN               3.0
 MMP9    CD4  top     1 135.22    Myeloid               NaN
 MMP9  ITGB2  top     2 128.18          T               NaN
 MMP9   CD84  top     3 126.40          T               NaN
 MMP9  CXCR4  top     4 123.59          T               NaN
 MMP9 IL10RA  top     5 122.65          T               NaN
 MMP9  PTPRC  top     6 120.62          T               NaN
 MMP9   CD3E    T    58 106.71        NaN               4.0
 MMP9   CD3D    T    54 107.31        NaN               4.0
 MMP9   CD3G    T    72 102.50        NaN               4.0
 MMP9    CD2    T    47 109.12        NaN               4.0
MMP14  CD276  top     1 136.63 Fibroblast               NaN
MMP14   CD81  top     2 133.40 Fibroblast               NaN
MMP14    DCN  top     3 132.88 Fibroblast               NaN
MMP14   APOD  top     4 125.38 Fibroblast               NaN
MMP14 IGFBP5  top     5 121.77 Fibroblast               NaN
MMP14 CXCL14  top     6 112.66 Fibroblast               NaN
MMP14   CD3E    T   192 -21.75        NaN               1.0
MMP14   CD3D    T   196 -22.39        NaN               1.0
MMP14   CD3G    T   200 -22.92        NaN               1.0
MMP14    CD2    T   220 -30.60        NaN               1.0

<Figure size 720x310 with 3 Axes>
  -> cosmx/cmx28f_d_hotspot_mmp_partners.png/.pdf
```

```python  # cell 10
# ============================================================================
# §R  Figure index + summary
# ============================================================================
pd.DataFrame(FIG_INDEX).to_csv(tab28f("figure_index"), index=False)
print(pd.DataFrame(FIG_INDEX).to_string(index=False))
print("\nb  " + "; ".join(f"{g}: {top[g]} {fold[g]:.1f}× T" for g in MMPS))
print(f"c  intraepidermal T ↑ {UP_TXT} | ↓ {DOWN_TXT} | {FC.n.max()} pieces")
print("d  " + "; ".join(f"{g}: partners {', '.join(FD[(FD.mmp == g) & (FD.kind == 'top')].gene)}; "
                        f"CD3/CD2 median rank {t_med[g]:.0f}/{n_tested}" for g in MMPS))
```

```
figure                              file                                                                                  what
       aa_whole_slides        cmx28f_aa_whole_slides.png       slides 4 (patch) and 7 (plaque): every cell by resolVI cell type, pieces packed
             a_windows              cmx28f_a_windows.png                    2 patch + 2 plaque FOVs: resolVI cell types and the epidermis edge
    a_candidates_patch     cmx28f_a_candidates_patch.png                          16 candidate patch windows (FOV key in each title) for Fig a
   a_candidates_plaque    cmx28f_a_candidates_plaque.png                         16 candidate plaque windows (FOV key in each title) for Fig a
     b_mmp_by_celltype      cmx28f_b_mmp_by_celltype.png                     MMP2 / MMP9 / MMP14 per cell type, resolVI expression, all slides
c_motility_epi_vs_derm cmx28f_c_motility_epi_vs_derm.png              amoeboid-motion / cortex / chemotaxis genes, epidermal vs dermal T cells
d_hotspot_mmp_partners cmx28f_d_hotspot_mmp_partners.png Hotspot local correlation of MMP2 / MMP9 / MMP14 with every gene; T-cell genes marked

b  MMP2: Fibroblast 12.2× T; MMP9: Myeloid 3.7× T; MMP14: Fibroblast 4.3× T
c  intraepidermal T ↑ LMNA, AQP3, EZR, ITGAE, CD69, CXCR6, CCR10 | ↓ MYL9, ARHGDIB, KLF2, SELL | 19 pieces
d  MMP2: partners COL3A1, COL6A3, COL6A2, CCDC80, COL1A2, COL1A1; CD3/CD2 median rank 102/276; MMP9: partners CD4, ITGB2, CD84, CXCR4, IL10RA, PTPRC; CD3/CD2 median rank 56/276; MMP14: partners CD276, CD81, DCN, APOD, IGFBP5, CXCL14; CD3/CD2 median rank 198/276
```
