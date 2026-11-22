# old/28b_tme_degradome_figures.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/28b_tme_degradome_figures.ipynb` and is never run._

# 28b — figures for the cutaneous degradome (atlas **v2**, `mal_tcr`)

`28_tme_degradome.ipynb` is three comparisons, one results cell and **no figures**. This
notebook is the figure layer on top of the tables it wrote. F1–F2 recompute **no statistic that
nb28 already ran**: every q-value they draw is read from `tables/deg28_*.csv`, and the places
where a figure needs an arm nb28 did not test are labelled *recomputed* on the figure itself.
FA–FF recompute every statistic they draw (see *FA–FF statistics* below).

| fig | question | source |
|---|---|---|
| F1 | degrading in malignant vs non-malignant vs healthy CD4 | object + `deg28_B_arm_tests` |
| F2 | proteases in fibroblasts (a) and myeloid cells (b) beside the CD4 arms, MF skin | object |
| F3 | which cell type actually carries each degradome family | object (abundance × expression) |
| FB | the clone does not instruct the microenvironment (T-cell → stroma ligands) vs what it does carry (inhibitors, lysosomal cathepsins, granzymes) | object, recomputed |
| FC | LIGHT → LTβR/HVEM → RELB, read as a chain | object, recomputed |
| FD | Th2 axis: TF on the clone, cytokines below detection, receptor and output on the stroma | object, recomputed |
| FE | hyaluronan: synthesis, fragmentation, CD44 — stroma (MF vs healthy) and CD4 | object, recomputed |
| FF | the TGF-β loop runs inside the fibroblast compartment: latent ligand, docking, activation, receptor, output (MF vs healthy) beside the same genes in CD4 | object, recomputed |

## Arms

The malignancy call is `mal_tcr`, as in nb28 — clonal vs non-clonal CD4 within the same biopsy,
CDR3-less cells excluded rather than counted benign. Healthy CD4 from our own atlas is carried
as a **fourth arm** (`CD4_HC`); nb28's `cd4` block has only `clonal / non-clonal / external HC`,
so any star involving `HC CD4` is recomputed here by the same donor-level test
(`degradome_cells.arm_tests`), BH-corrected within the figure.

Direction is carried by `pct_up`, never by `median_delta` — these panels are zero-inflated and
the median paired delta is exactly 0 for most genes. A row with no nonzero paired delta
**abstains**; abstentions are drawn grey, not as zeros.

## FA–FF statistics

- **Cohort.** Every MF arm is restricted to the TCR cohort: the MF samples whose CD4 cells carry
  a CDR3 (38 samples in nb28 §2), so the clone and the stroma come from the same biopsies.
  Healthy = atlas HC (`disease == HC`); the reynolds2021 band is not drawn.
- **Replicate value.** Donor **mean** of log1p CP10K over ≥ `cfg.MIN_CELLS` cells. A per-cell
  median is exactly 0 in most donors for these genes, so it cannot carry the comparison.
- **Tests.** Malignant vs reactive CD4: the §A design, paired Wilcoxon signed-rank on the
  per-biopsy difference (`DC.paired_within`), run on per-biopsy means. Every other comparison:
  donor-level Mann-Whitney. A gene below `cfg.MIN_DETECTION` in both arms, or with fewer than
  `cfg.MIN_DONORS_PER_ARM` donors per arm, is *not testable* (hollow / "n.t.").
- **Correction.** BH over the tests drawn in each figure. These q-values do not match
  `deg28_*` and must not be quoted as nb28's.
- **Colour.** Teal = higher in MF (or in the clone), rust = lower, grey = not significant. The
  colour comes from the test, so a claimed direction the data does not reproduce is drawn grey.

## Outputs

`figures/degradome/deg28b_*.png` and `tables/deg28b_*.csv`. Nothing under `deg28_*` is written
or overwritten — those 188 PNGs are v1 artifacts from `notebooks/old/28_tme_degradome_v1.ipynb`
and are not redrawable from the current tables.

```python  # cell 1
# ============================================================================
# §0  Setup — the same preamble and the same build gate as nb28 §0
# ============================================================================
# Figures only. The gate still runs: if the size factor was not summed over all 42,347 genes
# before the panel subset, every value drawn below is scaled by a cell-type-dependent factor.
import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


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

import degradome_data as cfg          # config only; frozen
import degradome_helpers as D         # build + gate
import degradome_cells as DC          # scores, tests, the plotting layer
import malignancy as MAL              # the three calls
import external_ref as X              # reynolds2021

# 28b owns its own prefix. cfg.fig()/cfg.tab() are deg28_ and would land on nb28's v1 artifacts.
PREFIX = "deg28b_"


def fig28b(name, ext="png"):
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    return cfg.FIG_DIR / f"{PREFIX}{name}.{ext}"


def tab28b(name):
    cfg.TAB_DIR.mkdir(parents=True, exist_ok=True)
    return cfg.TAB_DIR / f"{PREFIX}{name}.csv"


ARM_CALL = "mal_tcr"
CLONAL, NONCLONAL = cfg.CD4_MALIGNANT, cfg.CD4_REACTIVE
HC_CD4 = "CD4_HC"                     # healthy CD4 as an arm of its own; not an nb28 level

adata = D.load_degradome_adata()
eq = D.cached("equivalence",
              lambda: {"max_abs_diff": float(D.assert_build_equivalence(adata))})
assert eq["max_abs_diff"] < 1e-6, f"normalization gate failed: {eq}"
PANEL = [g for g in cfg.panel_genes() if g in set(map(str, adata.var_names))]

# nb28's tables. Nothing here recomputes them.
A = pd.read_csv(cfg.tab("A_paired_genes"))        # paired within biopsy, clonal vs non-clonal
B = pd.read_csv(cfg.tab("B_arm_tests"))           # donor-level, per block x row x gene x arm pair
CC = pd.read_csv(cfg.tab("C_ccc_edges"))          # curated ligand-receptor edges
COV = pd.read_csv(cfg.tab("coverage"))            # which levels may be spoken about

print(f"\nnormalization gate ok: max |lognorm(panel) - lognorm(full)| = {eq['max_abs_diff']:.3g}")
print(f"{adata.n_obs:,} cells | panel {len(PANEL)} genes | arms on {ARM_CALL}")
print(f"tables: A {len(A):,} genes | B {len(B):,} rows | C {len(CC):,} edge x level | "
      f"coverage {len(COV)} levels")
```

```
nb22 sidecar reconciled: 20 levels, 328,461 cells, zero drift
skin_degradome.h5ad: 985,273 cells x 503 genes, 112 donors, 123 samples, 12 studies
  scope ['MF', 'HC']: kept 1,002,481 of 1,345,527 skin cells; dropped 17,208 with no identity
  DEGRADOME-ONLY object: var is a hand-curated 503-gene panel. Not for HVG selection, PCA, UMAP, clustering or genome-wide DE. Use skin_degradome_pseudobulk_full.parquet for anything needing all genes.
[cache] equivalence <- equivalence.json

normalization gate ok: max |lognorm(panel) - lognorm(full)| = 0
985,273 cells | panel 503 genes | arms on mal_tcr
tables: A 503 genes | B 33,972 rows | C 394 edge x level | coverage 29 levels
```

```python  # cell 2
# ============================================================================
# §1  The arms — nb28 §2/§3, condensed, plus a healthy CD4 arm
# ============================================================================
# The object on disk was built with mal_knn as cfg.MALIG_HEADLINE. nb28 derives its own CD4
# split on mal_tcr and keeps the vocabulary identical, so writing that split straight into
# cfg.GROUPBY makes every plotting helper in degradome_cells -- all of which hardcode
# cfg.GROUPBY -- draw the mal_tcr arms with no helper edit.
mal = MAL.load_malignancy(adata.obs["cell_id"] if "cell_id" in adata.obs else adata.obs_names,
                          has_tcr=adata.obs.get("has_tcr"), verbose=False)
mal.index = adata.obs_names

IS_CD4 = (adata.obs[cfg.CELLTYPE_SRC].astype(str) == "CD4").to_numpy()
IS_MF = (adata.obs["disease"].astype(str) == "MF").to_numpy()
IS_HC = (adata.obs["disease"].astype(str) == cfg.HC_DISEASE).to_numpy()

LVL = pd.Series(adata.obs[cfg.GROUPBY].astype(str).to_numpy(), index=adata.obs_names)
LVL = LVL.mask(IS_CD4, pd.Series(MAL.cd4_level(adata.obs[cfg.CELLTYPE_SRC], mal,
                                               call=ARM_CALL).astype(str),
                                 index=adata.obs_names))
# Healthy CD4 is its own arm, not "reactive": clonal/non-clonal is a within-MF contrast and
# folding healthy cells into the non-clonal arm would make it mean two different things.
LVL = LVL.mask(IS_CD4 & IS_HC, HC_CD4)
adata.obs[cfg.GROUPBY] = pd.Categorical(LVL)
adata.obs["arm_level"] = adata.obs[cfg.GROUPBY]
LV = LVL.to_numpy()

CLAIM = [lv for lv in cfg.CLAIM_LEVELS if lv in set(COV.loc[COV["claimable"], cfg.GROUPBY])]
FIBRO = [lv for lv in cfg.FIBRO_LEVELS if lv in CLAIM]
MYELOID = [lv for lv in cfg.MYELOID_LEVELS if lv in CLAIM]

ext = X.load_external()
EXT_T = ext.obs[X.TARGET_COL].astype(str).to_numpy()
ON_OBJ = set(LV)


def tme_rows(groups):
    """{row: {MF, HC, external HC}} for a {row_label: [fine levels]} map, plus per-row notes."""
    rows, notes = {}, {}
    for row, levels in groups.items():
        levels = [lv for lv in levels if lv in ON_OBJ]
        if not levels:
            continue
        m = np.isin(LV, levels)
        tg = cfg.EXT_ARM.get(row)
        rows[row] = {"MF": (adata, m & IS_MF), "HC": (adata, m & IS_HC),
                     "external HC": (ext, np.isin(EXT_T, tg)) if tg else None}
        note = []
        if row in cfg.EXT_POOLED:
            note.append(f"ext pooled: {'+'.join(tg)}")
        if not set(levels) & set(CLAIM):
            note.append("below donor gate")
        if note:
            notes[row] = "; ".join(note)
    return rows, notes


TME_ARMS = ["MF", "HC", "external HC"]
CD4_ARMS4 = ["clonal", "non-clonal", "HC CD4", "external HC"]
# DC.ARM_COLORS gives non-clonal and HC the same green; the healthy CD4 arm needs its own hue.
ARM_COLORS4 = {**DC.ARM_COLORS, "HC CD4": "#1f77b4"}

cd4_rows4 = {"CD4": {
    "clonal": (adata, (LV == CLONAL) & IS_MF),
    "non-clonal": (adata, (LV == NONCLONAL) & IS_MF),
    "HC CD4": (adata, LV == HC_CD4),
    "external HC": (ext, np.isin(EXT_T, cfg.EXT_ARM["CD4"])),
}}
mye_low, mye_low_notes = tme_rows(cfg.MYELOID_COARSE)
mye_high, mye_high_notes = tme_rows({lv: [lv] for lv in cfg.MYELOID_LEVELS})
fib_low, fib_low_notes = tme_rows(cfg.FIBRO_COARSE)
fib_high, fib_high_notes = tme_rows({lv: [lv] for lv in cfg.FIBRO_LEVELS})

FIB_FINE = [lv for lv in cfg.FIBRO_LEVELS if lv in ON_OBJ]
MYE_FINE = [lv for lv in cfg.MYELOID_LEVELS if lv in ON_OBJ]
STROMA_FINE = FIB_FINE + MYE_FINE

for lab, n in [("clonal", (LV == CLONAL) & IS_MF), ("non-clonal", (LV == NONCLONAL) & IS_MF),
               ("HC CD4", LV == HC_CD4), ("excluded, no CDR3", (LV == cfg.CD4_UNASSESSED))]:
    print(f"  {lab}: {int(n.sum()):,} cells / "
          f"{adata.obs.loc[n, cfg.DONOR_KEY].nunique()} donors")
print(f"claimable {len(CLAIM)} levels | fibroblast {FIBRO} | myeloid {MYELOID}")
print(f"blocks: cd4 (1 row, 4 arms) | myeloid_low ({len(mye_low)}) | "
      f"myeloid_high ({len(mye_high)}) | fibro_low ({len(fib_low)}) | fibro_high ({len(fib_high)})")
```

```
clonal: 68,843 cells / 33 donors
  non-clonal: 59,220 cells / 38 donors
  HC CD4: 4,742 cells / 22 donors
  excluded, no CDR3: 195,412 cells / 90 donors
claimable 18 levels | fibroblast ['F_papillary', 'F_reticular', 'F_mesenchymal', 'F_myofibroblast', 'F_inflammatory', 'F_apCAF'] | myeloid ['LC', 'cDC1', 'cDC2', 'DC_LAMP3', 'pDC', 'Mono_CD14', 'moDC', 'Mac_FOLR2', 'Mac_SPP1_TREM2', 'Mac_infl']
blocks: cd4 (1 row, 4 arms) | myeloid_low (4) | myeloid_high (12) | fibro_low (3) | fibro_high (6)
```

```python  # cell 3
# ============================================================================
# §2  Scores, computed once
# ============================================================================
# Family score = plain mean of the family's genes on X (see the degradome_cells docstring for
# why not score_genes). Families below cfg.MIN_FAMILY_SIZE_FOR_SCORE get no mean and are read
# per gene instead -- ha_synthase and glycosidase are the two that matter here.
FAMS = DC.family_score_cells(adata)
BALS = DC.balance_cells(adata)
FAM_COLS = [f"fam_{k}" for k in FAMS]
BAL_COLS = [f"bal_{k}" for k in BALS]

# The replicate unit: one median per (level, donor), gated at cfg.MIN_CELLS.
DMED = DC.donor_median(adata, FAM_COLS + BAL_COLS)

# The external band. nb28 reads the medians the external build wrote rather than re-scoring
# 195,739 cells; a family whose gene set is not the same on both sides abstains entirely.
EXT_DMED = X.load_external_donor_median()
EXT_ABSTAIN, EXT_DROP = X.external_abstentions()


def ext_band(scores, target):
    """{score: one value per external donor} for plot_split_violin / plot_external_band."""
    d = EXT_DMED[EXT_DMED["target"].astype(str) == target]
    return {s: d[s].dropna().to_numpy() for s in scores
            if s in EXT_DMED.columns and s not in set(EXT_DROP) and d[s].notna().any()}


# The paired slice: MF CD4 only, both arms, one row per (sample, arm). This is what §A tests
# and what plot_paired_slopes draws -- the sample is the replicate, so donor, study, chemistry,
# site and treatment are the same tube.
AXIS_CD4_GENES = ["TNFSF14", "LTA", "LTB", "CD40LG", "CXCR4", "CD44", "HYAL3", "HYAL2",
                  "IL7R", "SELL", "S1PR1", "ITGA4", "ITGA5", "ITGA6", "CCR7", "BSG"]
TOP_A = (A[A["q"].notna()].sort_values("q")["gene"].head(6).tolist())
SLICE_GENES = [g for g in dict.fromkeys(TOP_A + AXIS_CD4_GENES) if g in set(PANEL)]

cd4_mf = adata[IS_MF & np.isin(LV, [CLONAL, NONCLONAL])].copy()
SLICE_COLS = DC.gene_score_cells(cd4_mf, SLICE_GENES, verbose=False)
SM = DC.donor_median(cd4_mf, SLICE_COLS, groupby="arm_level", donor_key=cfg.SAMPLE_KEY)
N_PAIRS = int(SM[SM["eligible"]].pivot_table(index=cfg.SAMPLE_KEY, columns="arm_level",
                                             values=SLICE_COLS[0], aggfunc="median")
              .dropna().shape[0])

print(f"family scores: {len(FAM_COLS)} | balances: {len(BAL_COLS)} | "
      f"donor medians {DMED.shape} ({int(DMED['eligible'].sum()):,} eligible)")
print(f"external abstains: {EXT_ABSTAIN or 'none'}")
print(f"paired slice: {cd4_mf.n_obs:,} MF CD4 cells, {N_PAIRS} biopsies with both arms "
      f"above {cfg.MIN_CELLS} cells (nb28 §A reports {int(A['n_pairs'].max())})")
```

```
scored 19 families per cell
  too small to score (<5 genes), plot per gene instead: {'ha_synthase': 3, 'glycosidase': 3, 'mast': 4, 'caspase': 1}
balance ratios: ['mmp_timp', 'cathepsin_cystatin', 'serine_serpin', 'collagenase_timp', 'ha_turnover', 'remodel_tone']
  ABSTAIN fam_ha_catabolic: 4/5 genes present externally (missing TMEM2) -- the two sides would be different gene sets, which is not a comparison
family scores: 19 | balances: 6 | donor medians (2618, 32) (1,450 eligible)
external abstains: ['ha_catabolic']
paired slice: 128,063 MF CD4 cells, 33 biopsies with both arms above 25 cells (nb28 §A reports 33)
```

```python  # cell 4
# ============================================================================
# §3  Figure utilities — read nb28's tables, draw, never re-test
# ============================================================================
# B stores each pair as (arm_a, arm_b) with delta = donor_median_b - donor_median_a, so
# "MF vs HC" carries HC - MF. b_grid flips it to the direction a reader expects (arm_a minus
# arm_b) and keeps nb28's q untouched.
STAR = DC._stars
GREY = "#bdbdbd"


def b_grid(block, rows, genes, arm_pair="MF vs HC"):
    """(row x gene) donor medians, delta and q for one arm pair, straight out of deg28_B.

    `block` may be one block or several -- an axis read across fibroblasts and myeloid cells
    needs fibro_high and myeloid_high together, and their row names do not overlap.
    """
    blocks = [block] if isinstance(block, str) else list(block)
    d = B[B["block"].isin(blocks) & B["row"].isin(rows) & B["gene"].isin(genes)
          & (B["arm_pair"] == arm_pair)].copy()
    d["delta_ab"] = d["donor_median_a"] - d["donor_median_b"]     # arm_a minus arm_b
    return d


def heat(ax, d, rows, cols, value="delta_ab", row_col="row", col_col="gene", cmap="RdBu_r",
         vmax=None, title=None, cbar_label=None, star_col="q", annot=None, scale="shared"):
    """rows x cols heatmap with a BH star in each cell that has one. NaN cells are hatched.

    `scale="column"` gives every gene its own colour scale. Genes on this panel differ by an
    order of magnitude in absolute expression (CXCL12 moves by 1.8, CXCR4 by 0.2), and on one
    shared scale the smaller gene is a blank column -- which is not what the test says.
    """
    M = d.pivot_table(index=row_col, columns=col_col, values=value, aggfunc="mean")
    Q = d.pivot_table(index=row_col, columns=col_col, values=star_col, aggfunc="min")
    # rows/cols are kept as asked even when deg28_B has no row for them: a hatched cell says
    # "we could not look here", which an absent column does not.
    rows, cols = list(rows), list(cols)
    M, Q = M.reindex(index=rows, columns=cols), Q.reindex(index=rows, columns=cols)
    if scale == "column":
        cmax = np.nanmax(np.abs(M.to_numpy()), axis=0)
        cmax = np.where(np.isfinite(cmax) & (cmax > 0), cmax, 1.0)
        Z, v = M.to_numpy() / cmax, 1.0
    else:
        v = np.nanmax(np.abs(M.to_numpy())) if vmax is None else vmax
        v = 1.0 if not np.isfinite(v) or v == 0 else v
        Z = M.to_numpy()
    im = ax.imshow(Z, cmap=cmap, vmin=-v, vmax=v, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            if not np.isfinite(M.iat[i, j]):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=True, color="#f0f0f0",
                                           hatch="///", edgecolor="white", linewidth=0))
                continue
            q = Q.iat[i, j]
            s = STAR(q) if np.isfinite(q) else ""
            if annot is not None:
                s = f"{M.iat[i, j]:{annot}}\n{s}" if s else f"{M.iat[i, j]:{annot}}"
            if s:
                ax.text(j, i, s, ha="center", va="center", fontsize=5.5,
                        color="#111111" if abs(Z[i, j]) < 0.6 * v else "white")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=6, style="italic")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=6)
    ax.tick_params(length=2, pad=1)
    if title:
        ax.set_title(title, fontsize=8)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.ax.tick_params(labelsize=5)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=6)
    return M, Q


def arm_dots(ax, sl, arms, gene_idx, tests=None, row=None, gene=None, colors=None, ylabel=None):
    """One dot per donor per arm, from DC.arm_test_block's slices. The n a comparison has."""
    colors = colors or ARM_COLORS4
    rng = np.random.default_rng(0)
    tops = []
    for i, a in enumerate(arms):
        s = sl.get(a)
        if s is None:
            ax.text(i, 0, "no ext.\ncounterpart", ha="center", va="bottom", fontsize=5,
                    color="#888888", rotation=90)
            continue
        v = s["dmean"][:, gene_idx]
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        tops.append(np.max(v))
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=8, alpha=0.85,
                   color=colors.get(a, "#999999"), edgecolor="white", linewidth=0.3, zorder=3)
        ax.hlines(np.median(v), i - 0.28, i + 0.28, color="#111111", linewidth=1.2, zorder=4)
    top = max(tops) if tops else 1.0
    if tests is not None and row is not None and gene is not None:
        sig = tests[(tests["row"] == row) & (tests["gene"] == gene) & (tests["q"] < 0.05)]
        for k, t in enumerate(sorted(sig.itertuples(), key=lambda t: arms.index(t.arm_b)
                                     - arms.index(t.arm_a))):
            x0, x1 = arms.index(t.arm_a), arms.index(t.arm_b)
            h = top * (1.06 + 0.13 * k)
            ax.plot([x0, x0, x1, x1], [h - top * .03, h, h, h - top * .03], color="#333333",
                    linewidth=0.6, clip_on=False)
            ax.text((x0 + x1) / 2, h, STAR(t.q), ha="center", va="bottom", fontsize=6)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, rotation=90, fontsize=6)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_ylabel(ylabel or "", fontsize=6)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    return ax


def lollipop(ax, d, label_col="gene", value="pct_up", q_col="q", family_col="family",
             centre=0.5, xlabel=None, title=None, n_label=None):
    """One row per gene: dot at `value`, stem to `centre`, colour by family, grey if abstained.

    Direction is pct_up, not median_delta -- these panels are zero-inflated and the median
    paired delta is exactly 0 for most genes (nb28 §A).
    """
    d = d.reset_index(drop=True)
    fams = [f for f in dict.fromkeys(d[family_col].astype(str))]
    pal = DC._level_colors(fams)
    y = np.arange(len(d))[::-1]
    for yi, (_, r) in zip(y, d.iterrows()):
        v, q = r[value], r[q_col]
        ok = np.isfinite(v) and np.isfinite(q)
        c = pal[str(r[family_col])] if ok else GREY
        ax.plot([centre, v if ok else centre], [yi, yi], color=c, linewidth=1.0,
                alpha=0.8, zorder=2)
        ax.scatter([v if ok else centre], [yi], s=22 if ok else 10, color=c, zorder=3,
                   edgecolor="white", linewidth=0.4)
        lab = f"{r[label_col]}{' ' + STAR(q) if ok else ' (abstained)'}"
        ax.text(centre, yi, "  " + lab if (not ok or v >= centre) else lab + "  ",
                ha="left" if (not ok or v >= centre) else "right", va="center", fontsize=5.5,
                style="italic", zorder=4)
    ax.axvline(centre, color="#333333", linewidth=0.8, zorder=1)
    ax.set_yticks([])
    ax.set_ylim(-1, len(d))
    ax.tick_params(axis="x", labelsize=6)
    if xlabel or n_label:
        ax.set_xlabel("\n".join([x for x in (xlabel, n_label) if x]), fontsize=6)
    if title:
        ax.set_title(title, fontsize=8)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=pal[f], label=f,
                          markersize=4) for f in fams]
    return handles


def fam_arm_test(dmed, levels, score_cols, group="disease", arm_a="MF", arm_b="HC"):
    """Donor-level MWU on the FAMILY score, arm_a vs arm_b. BH over this grid only.

    nb28's tested unit is the gene, so a family-level star does not exist in deg28_B and has to
    be computed here. Same replicate (the donor), same test, same gate as DC.arm_tests.
    """
    out = []
    for lv in levels:
        d = dmed[(dmed[cfg.GROUPBY].astype(str) == lv) & dmed["eligible"]]
        for s in score_cols:
            x = d.loc[d[group].astype(str) == arm_a, s].dropna().to_numpy()
            y = d.loc[d[group].astype(str) == arm_b, s].dropna().to_numpy()
            p = np.nan
            if (len(x) >= cfg.MIN_DONORS_PER_ARM and len(y) >= cfg.MIN_DONORS_PER_ARM
                    and np.ptp(np.r_[x, y]) > 0):
                p = float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
            out.append({"row": lv, "gene": s, "n_donors_a": len(x), "n_donors_b": len(y),
                        "donor_median_a": np.median(x) if len(x) else np.nan,
                        "donor_median_b": np.median(y) if len(y) else np.nan, "p": p})
    r = pd.DataFrame(out)
    r["delta_ab"] = r["donor_median_a"] - r["donor_median_b"]
    r["q"] = np.nan
    m = r["p"].notna()
    if m.any():
        r.loc[m, "q"] = multipletests(r.loc[m, "p"], method="fdr_bh")[1]
    return r


FIG_INDEX = []


def logfig(fig_id, path, question, source, note=""):
    FIG_INDEX.append({"figure": fig_id, "file": Path(path).name, "question": question,
                      "source": source, "note": note})
    print(f"  {fig_id} -> {Path(path).name}")


print("figure utilities ready")
```

```
figure utilities ready
```

```python  # cell 5
# ============================================================================
# F1  Degrading in malignant vs non-malignant vs healthy CD4
# ============================================================================
# Two panels because they answer the question at two grains: the family score says how much
# degradative machinery a CD4 cell carries at all, the gene grid says which enzyme.
F1_SCORES = [s for s in ["fam_mmp", "fam_cathepsin", "fam_serine_other", "fam_metallo_other",
                         "fam_granzyme", "fam_ha_catabolic", "fam_inhibitor",
                         "fam_matrix_receptor"] if s in DMED.columns]
CD4_LEVELS3 = [CLONAL, NONCLONAL, HC_CD4]

DC.plot_split_violin(
    adata, DMED, F1_SCORES, CD4_LEVELS3,
    title=("The clone carries no protease surplus: cathepsins, serpins and granzymes separate "
           "MF CD4 from healthy CD4, not clone from reactive\n"
           "violin = cells, tick = median, dots = donors (≥ 25 cells); "
           "band = reynolds2021 healthy CD4, 5 donors"),
    external=ext_band(F1_SCORES, "CD4"), ncols=4,
    save=fig28b("F1a_cd4_family_scores"))
logfig("F1a", fig28b("F1a_cd4_family_scores"),
       "degradome family load, clone vs reactive vs healthy CD4",
       "object (donor medians) + deg28_external_donor_median")

# The headline proteases only: the four MMPs CD4 cells express at all (MMP1/3/13 and most of the
# family sit below cfg.MIN_DETECTION in every arm), the canonical sheddases ADAM10/17 plus the
# leukocyte ADAM8/19, uPAR and heparanase. The full secreted-protease panel is in nb28's tables.
F1_GENES = [g for g in ["MMP2", "MMP9", "MMP14", "MMP25",
                        "ADAM8", "ADAM10", "ADAM17", "ADAM19",
                        "PLAUR", "HPSE"] if g in set(PANEL)]
f1b = DC.plot_family_arms(
    cd4_rows4, F1_GENES, CD4_ARMS4, arm_colors=ARM_COLORS4,
    title=("No major MMP, ADAM, uPAR or heparanase is clone-enriched; "
           "HPSE is higher in the reactive arm"),
    wrap=5, col_w=2.2, row_h=2.6, expressing_only=True, clean=True,
    save=fig28b("F1b_cd4_protease_genes"))
f1b.to_csv(tab28b("F1_source"), index=False)
logfig("F1b", fig28b("F1b_cd4_protease_genes"),
       "which protease, if any, the clone carries",
       "object; donor-level MWU recomputed over the drawn genes (HC CD4 is not an nb28 arm)",
       note="q recomputed figure-wide: the 4-arm set includes atlas HC CD4")
```

```
<Figure size 800x400 with 8 Axes>
  F1a -> deg28b_F1a_cd4_family_scores.png
No major MMP, ADAM, uPAR or heparanase is clone-enriched; HPSE is higher in the reactive arm: 10/10 genes drawn; 11/60 donor-level tests at q < 0.05

<Figure size 1220x520 with 10 Axes>
  F1b -> deg28b_F1b_cd4_protease_genes.png
```

```python  # cell 6
# ============================================================================
# F2  Where the proteases are: fibroblasts and myeloid cells next to the CD4 arms, MF skin
# ============================================================================
# The two CD4 arms stay in both panels -- the contrast is the claim: near-empty CD4 violins for
# every protease beside stromal violins that are not. MF samples only; dots are donor medians
# (>= cfg.MIN_CELLS cells), the donor being the replicate. No test is drawn.
F2_CD4 = {"CD4_clonal": (LV == CLONAL) & IS_MF, "CD4_non_clonal": (LV == NONCLONAL) & IS_MF}
F2_PANELS = {
    "F2a": ("fibro",
            {"F_structural": np.isin(LV, cfg.FIBRO_COARSE["F_structural"]) & IS_MF,
             "F_inflammatory": (LV == "F_inflammatory") & IS_MF,
             "F_apCAF": (LV == "F_apCAF") & IS_MF},
            ["MMP2", "MMP14", "ADAMTS5", "CTSB", "CTSK", "PRSS23", "TIMP1", "TIMP3"],
            "Fibroblasts, not CD4 T cells, carry the proteases in MF skin"),
    "F2b": ("myeloid",
            {lv: (LV == lv) & IS_MF for lv in ["Mac_FOLR2", "Mono_CD14", "cDC2"]},
            ["MMP12", "MMP9", "MMP14", "CTSB", "CTSC", "CTSS", "PLAUR", "SERPINA1"],
            "Myeloid cells, not CD4 T cells, carry the proteases in MF skin"),
}
F2_COLORS = {"CD4_clonal": DC.ARM_COLORS["clonal"], "CD4_non_clonal": DC.ARM_COLORS["non-clonal"],
             "F_structural": "#08519c", "F_inflammatory": "#3182bd", "F_apCAF": "#9ecae1",
             "Mac_FOLR2": "#a63603", "Mono_CD14": "#e6550d", "cDC2": "#fdae6b"}
# Display names on the figure only; the source tables keep the atlas labels (methods).
F2_LABELS = {"CD4_clonal": "Malignant CD4 (TCR clone)",
             "CD4_non_clonal": "Reactive CD4 (same biopsy)",
             "F_structural": "Structural fibroblast",
             "F_inflammatory": "Inflammatory fibroblast",
             "F_apCAF": "MHC-II+ fibroblast (apCAF)",
             "Mac_FOLR2": "Resident macrophage (FOLR2+)",
             "Mono_CD14": "Classical monocyte (CD14+)",
             "cDC2": "Conventional DC2"}

for tag, (lineage, stroma, genes, title) in F2_PANELS.items():
    d = DC.plot_group_violins(adata, {**F2_CD4, **stroma}, genes, title=title,
                              colors=F2_COLORS, labels=F2_LABELS,
                              save=fig28b(f"{tag}_{lineage}_vs_cd4"))
    d.to_csv(tab28b(f"{tag}_source_{lineage}"), index=False)
    logfig(tag, fig28b(f"{tag}_{lineage}_vs_cd4"),
           f"protease genes per cell type, CD4 arms vs {lineage}, MF skin",
           "object (per-cell log1p CP10K; donor medians)")
```

```
CD4_clonal: 68,843 cells / 33 donors >= 25 cells
CD4_non_clonal: 59,220 cells / 38 donors >= 25 cells
F_structural: 34,788 cells / 59 donors >= 25 cells
F_inflammatory: 35,381 cells / 51 donors >= 25 cells
F_apCAF: 16,707 cells / 51 donors >= 25 cells

<Figure size 880x440 with 8 Axes>
  F2a -> deg28b_F2a_fibro_vs_cd4.png
CD4_clonal: 68,843 cells / 33 donors >= 25 cells
CD4_non_clonal: 59,220 cells / 38 donors >= 25 cells
Mac_FOLR2: 16,312 cells / 67 donors >= 25 cells
Mono_CD14: 3,589 cells / 32 donors >= 25 cells
cDC2: 12,961 cells / 63 donors >= 25 cells

<Figure size 880x440 with 8 Axes>
  F2b -> deg28b_F2b_myeloid_vs_cd4.png
```

```python  # cell 7
# ============================================================================
# §4  FA–FE layer — TCR cohort, donor means, every test recomputed here
# ============================================================================
# Every MF arm below is restricted to the TCR cohort -- the MF samples whose CD4 cells carry a
# CDR3, the sample set nb28 §A and §C rest on -- so clone and stroma come from the same biopsies.
# The replicate value is the donor MEAN of log1p CP10K: these genes are zero-inflated and a
# per-cell median is exactly 0 in most donors. Both choices move the numbers, so no star here is
# nb28's q: each is recomputed on donor means and BH-corrected within its own figure.
import textwrap

import scipy.sparse as sp
from matplotlib.patches import FancyArrowPatch
from matplotlib.transforms import offset_copy

_samp = adata.obs[cfg.SAMPLE_KEY].astype(str).to_numpy()
TCR_SAMPLES = sorted(set(_samp[IS_MF & IS_CD4 & mal["mal_tcr_defined"].to_numpy()]))
IS_MF_TCR = IS_MF & np.isin(_samp, TCR_SAMPLES)
VAR = pd.Index(map(str, adata.var_names))

# Display names on the figures only; source tables keep the atlas labels (methods).
LABELS = {CLONAL: "Malignant CD4 (TCR clone)", NONCLONAL: "Reactive CD4 (same biopsy)",
          HC_CD4: "Healthy-donor CD4",
          "F_structural": "Structural fibroblast", "F_inflammatory": "Inflammatory fibroblast",
          "F_apCAF": "MHC-II+ fibroblast (apCAF)", "F_reticular": "Reticular fibroblast",
          "F_mesenchymal": "Mesenchymal fibroblast",
          "Mac_FOLR2": "Resident macrophage (FOLR2+)", "Mono_CD14": "Classical monocyte (CD14+)",
          "cDC2": "Conventional DC2"}
UP, DOWN, NS = "#2a9d8f", "#b5542a", "#bdbdbd"
DIR_COLOR = {"up": UP, "down": DOWN, "ns": NS, "not tested": NS}
MALIG, REACT, HC_COL = "#7b3294", "#5e81ac", "#444444"
ARM_COL = {CLONAL: MALIG, NONCLONAL: REACT, HC_CD4: HC_COL}
XLAB = "donor mean, log1p CP10K"


def mask(ct, arm):
    """Cells of one cell type in one arm: 'MF' = TCR cohort, 'HC' = atlas healthy skin.

    The healthy arm of either CD4 arm is healthy-donor CD4. F_structural is the coarse pool
    (papillary + reticular + mesenchymal + myofibroblast), as in F2.
    """
    if ct == HC_CD4 or (arm == "HC" and ct in (CLONAL, NONCLONAL)):
        return LV == HC_CD4
    return np.isin(LV, cfg.FIBRO_COARSE.get(ct, [ct])) & (IS_MF_TCR if arm == "MF" else IS_HC)


_DM = {}


def donor_means(ct, arm, genes, key=cfg.DONOR_KEY):
    """(donor x gene) mean log1p CP10K, donors with >= cfg.MIN_CELLS cells; plus detection.

    Cached per (ct, arm, key); genes not yet cached are added on demand. `key=cfg.SAMPLE_KEY`
    gives the per-biopsy means the paired test runs on.
    """
    k = (ct, arm, key)
    genes = list(dict.fromkeys(genes))
    need = [g for g in genes if k not in _DM or g not in _DM[k][0].columns]
    if need:
        idx = np.flatnonzero(mask(ct, arm))
        Xs = adata.X[idx][:, VAR.get_indexer(need)]
        Xs = Xs.tocsr() if sp.issparse(Xs) else sp.csr_matrix(Xs)
        codes, uniq = pd.factorize(adata.obs[key].astype(str).to_numpy()[idx])
        cnt = np.bincount(codes, minlength=len(uniq))
        G = sp.csr_matrix((np.ones(len(idx)), (codes, np.arange(len(idx)))),
                          shape=(len(uniq), len(idx)))
        M = pd.DataFrame(np.asarray((G @ Xs).todense()) / np.maximum(cnt, 1)[:, None],
                         index=pd.Index(uniq, name=key), columns=need)
        det = pd.Series(np.asarray((Xs > 0).mean(axis=0)).ravel() if len(idx) else np.nan,
                        index=need)
        n = pd.Series(cnt, index=M.index)
        if k in _DM:
            M, det = _DM[k][0].join(M), pd.concat([_DM[k][1], det])
        _DM[k] = (M, det, n.reindex(M.index))
    M, det, cnt = _DM[k]
    return M.loc[cnt >= cfg.MIN_CELLS, genes], det[genes]


def prefetch(cts, genes, arms=("MF", "HC")):
    for ct in cts:
        for arm in arms:
            donor_means(ct, arm, genes)


def cell_values(ct, genes, max_cells=20_000):
    """Per-cell log1p CP10K in the MF arm, subsampled for the violin KDE only (seed fixed)."""
    idx = np.flatnonzero(mask(ct, "MF"))
    if len(idx) > max_cells:
        idx = np.sort(np.random.default_rng(0).choice(idx, max_cells, replace=False))
    Xs = adata.X[idx][:, VAR.get_indexer(genes)]
    return pd.DataFrame(Xs.toarray() if sp.issparse(Xs) else np.asarray(Xs), columns=genes)


# ---------------------------------------------------------------- tests
def t_mwu(ct, gene, a="MF", b="HC", ct_b=None, label=None):
    """Donor-level Mann-Whitney on donor means: ct in arm a vs (ct_b or ct) in arm b."""
    ct_b = ct_b or ct
    ma, da = donor_means(ct, a, [gene])
    mb, db = donor_means(ct_b, b, [gene])
    x, y = ma[gene].to_numpy(), mb[gene].to_numpy()
    rec = {"label": label or LABELS.get(ct, ct), "gene": gene, "ct": ct, "arm_a": a,
           "ct_b": HC_CD4 if (b == "HC" and ct_b in (CLONAL, NONCLONAL)) else ct_b, "arm_b": b,
           "test": "MWU (donor)", "n_a": len(x), "n_b": len(y),
           "med_a": np.median(x) if len(x) else np.nan,
           "med_b": np.median(y) if len(y) else np.nan,
           "det_a": da[gene], "det_b": db[gene], "p": np.nan, "reason": ""}
    rec["delta"] = rec["med_a"] - rec["med_b"]
    rec["sign"] = np.sign(rec["delta"])
    if not np.nanmax([rec["det_a"], rec["det_b"]]) >= cfg.MIN_DETECTION:
        rec["reason"] = f"below {cfg.MIN_DETECTION:.0%} detection in both arms"
    elif min(len(x), len(y)) < cfg.MIN_DONORS_PER_ARM:
        rec["reason"] = f"too few donors ({len(x)}, {len(y)}; need {cfg.MIN_DONORS_PER_ARM})"
    elif np.ptp(np.r_[x, y]) == 0:
        rec["reason"] = "no variance"
    else:
        rec["p"] = float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
    rec["vals_a"], rec["vals_b"] = x, y
    return rec


def t_paired(genes, label="vs Reactive CD4 (same biopsy)"):
    """Clone vs reactive, paired within biopsy: nb28 §A's signed-rank, run on per-biopsy means.

    DC.paired_within is the §A function itself; only the per-biopsy summary changes (mean, not
    median), and the cohort is the same by construction (mal_tcr needs a CDR3).
    """
    genes = list(dict.fromkeys(genes))
    a, da = donor_means(CLONAL, "MF", genes, key=cfg.SAMPLE_KEY)
    b, db = donor_means(NONCLONAL, "MF", genes, key=cfg.SAMPLE_KEY)
    long = (pd.concat([a.assign(arm=CLONAL), b.assign(arm=NONCLONAL)])
            .reset_index().assign(eligible=True))
    res = DC.paired_within(long, genes, pair_key=cfg.SAMPLE_KEY, arm_col="arm",
                           ref=NONCLONAL, alt=CLONAL, fdr=False).set_index("score")
    both = a.index.intersection(b.index)
    out = []
    for g in genes:
        r = res.loc[g]
        x, y = a.loc[both, g].to_numpy(), b.loc[both, g].to_numpy()
        det_ok = np.nanmax([da[g], db[g]]) >= cfg.MIN_DETECTION
        out.append({"label": label, "gene": g, "ct": CLONAL, "arm_a": "MF", "ct_b": NONCLONAL,
                    "arm_b": "MF", "test": "paired signed-rank (biopsy)",
                    "n_a": len(x), "n_b": len(y),
                    "med_a": np.median(x) if len(x) else np.nan,
                    "med_b": np.median(y) if len(y) else np.nan,
                    "det_a": da[g], "det_b": db[g],
                    "p": r["p"] if det_ok else np.nan,
                    "reason": r["reason"] if det_ok
                    else f"below {cfg.MIN_DETECTION:.0%} detection in both arms",
                    "delta": r["median_delta"], "sign": np.sign(r["n_pos"] - r["n_neg"]),
                    "n_pairs": r["n_pairs"], "n_pos": r["n_pos"], "n_neg": r["n_neg"],
                    "vals_a": x, "vals_b": y})
    return out


def bh(recs):
    """BH over every test drawn in one figure; direction = sign of a vs b where q < 0.05."""
    df = pd.DataFrame(recs).reset_index(drop=True)
    m = df["p"].notna()
    df["q"] = np.nan
    if m.any():
        df.loc[m, "q"] = multipletests(df.loc[m, "p"], method="fdr_bh")[1]
    df["direction"] = np.select([~m, df["q"] >= 0.05, df["sign"] > 0],
                                ["not tested", "ns", "up"], "down")
    return df


def save_source(df, name):
    df.drop(columns=["vals_a", "vals_b"], errors="ignore").to_csv(tab28b(name), index=False)


def show_tests(df, cols=("label", "gene", "n_a", "n_b", "med_a", "med_b", "q", "direction")):
    print(df[[c for c in cols if c in df.columns]].to_string(index=False, float_format="%.3g"))


# ---------------------------------------------------------------- drawing
def it(g):
    return rf"$\mathit{{{g}}}$"


def pct(x):
    if not np.isfinite(x):
        return "–"
    return f"{x:.0%}" if x >= 0.01 else f"{x:.1%}" if x >= 0.001 else "<0.1%"


def sig(q):
    return "n.t." if not np.isfinite(q) else (STAR(q) or "n.s.")


def _clean(ax):
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(labelsize=6, length=2)


def _violin(ax, v, pos, color, width=0.36, horizontal=False):
    v = np.asarray(v)[np.isfinite(v)]
    if len(v) and np.ptp(v) > 0:
        b = ax.violinplot([v], positions=[pos], widths=width, showextrema=False,
                          orientation="horizontal" if horizontal else "vertical")["bodies"][0]
        b.set_facecolor(color)
        b.set_alpha(0.3)
        b.set_edgecolor("none")


def _layout(df, group_col=None):
    """y per row (first row on top) plus one bold header line per group."""
    ys, heads, k, prev = [], [], 0, object()
    for g in (df[group_col] if group_col else [None] * len(df)):
        if group_col and g != prev:
            heads.append((k, g))
            k, prev = k + 1, g
        ys.append(k)
        k += 1
    return np.array([k - 1 - v for v in ys]), [(k - 1 - v, g) for v, g in heads], k


def _gapped(groups, gap=0.6):
    """y per row (first row on top) with `gap` rows between groups and no header rows, plus
    {group: (y_lo, y_hi)} for row_brackets."""
    ys, k, prev = [], 0.0, object()
    for g in groups:
        if ys and g != prev:
            k += gap
        ys.append(k)
        k, prev = k + 1, g
    top = ys[-1] if ys else 0.0
    y = np.array([top - v for v in ys])
    spans = {}
    for g, yi in zip(groups, y):
        lo, hi = spans.get(g, (yi, yi))
        spans[g] = (min(lo, yi), max(hi, yi))
    return y, spans, top + 1


def _heads(ax, heads):
    for yh, g in heads:
        ax.text(-0.02, yh, g, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=6.5, fontweight="bold")


def dumbbell(ax, df, group_col=None, note_col=None, brackets=False):
    """One row per test, df order top to bottom. Open dot = comparator (median of donor means),
    filled dot = MF / malignant arm coloured by direction, faint dots = donors, stars at right.
    `brackets=True` spaces groups apart instead of giving each a header row and returns their
    y spans for row_brackets, which the caller draws once the figure is final."""
    rng = np.random.default_rng(0)
    if brackets:
        y, spans, n = _gapped(list(df[group_col]))
        heads = []
    else:
        y, heads, n = _layout(df, group_col)
    for yi, (_, r) in zip(y, df.iterrows()):
        c = DIR_COLOR[r["direction"]]
        for v, col in ((r["vals_b"], HC_COL), (r["vals_a"], c)):
            v = np.asarray(v)
            ax.scatter(v, yi + rng.uniform(-0.16, 0.16, len(v)), s=3, color=col, alpha=0.3,
                       linewidth=0, zorder=2)
        ax.plot([r["med_b"], r["med_a"]], [yi, yi], color=c, linewidth=1.6, zorder=3)
        ax.scatter([r["med_b"]], [yi], s=28, facecolor="white", edgecolor=HC_COL,
                   linewidth=1.0, zorder=4)
        tested = r["direction"] != "not tested"
        ax.scatter([r["med_a"]], [yi], s=32, facecolor=c if tested else "white", edgecolor=c,
                   linewidth=1.0, zorder=5)
        note = r[note_col] if note_col and isinstance(r.get(note_col), str) else ""
        ax.text(1.01, yi, f"{sig(r['q'])}  {note}".rstrip(), transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=6)
    _heads(ax, heads)
    ax.set_yticks(y)
    ax.set_yticklabels(df["ylabel"], fontsize=6)
    ax.set_ylim(-0.7, n - 0.3)
    ax.axvline(0, color="#e0e0e0", linewidth=0.6, zorder=0)
    _clean(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return spans if brackets else None


def _x_annot(ax, i, r):
    ax.text(i, 0.97, sig(r["q"]), transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=7)
    if r["test"].startswith("paired") and np.isfinite(r["q"]):
        ax.text(i, 0.89, f"{r['n_pos']:.0f}↑/{r['n_neg']:.0f}↓", ha="center", va="top",
                transform=ax.get_xaxis_transform(), fontsize=5.5, color="#555555")


def _gene_ticks(ax, df):
    """Gene names as ticks; under each, % of cells detected (purple malignant, blue reactive)."""
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["gene"], fontsize=6.5, style="italic")
    ax.set_xlim(-0.6, len(df) - 0.4)
    for i, (_, r) in enumerate(df.iterrows()):
        for k, (v, col) in enumerate(((r["det_a"], MALIG), (r["det_b"], REACT))):
            ax.text(i, 0, pct(v), ha="center", va="top", fontsize=5.5, color=col,
                    transform=offset_copy(ax.get_xaxis_transform(), fig=ax.figure,
                                          y=-13 - 7 * k, units="points"))
    _clean(ax)


def det_label(ax):
    ax.text(0, 0, "% cells\ndetected", ha="right", va="top", fontsize=5.5, color="#555555",
            transform=offset_copy(ax.transAxes, fig=ax.figure, x=-4, y=-13, units="points"))


def paired(ax, df, violin=False):
    """Clone vs reactive per gene: one line per biopsy (its two donor means), bar = median.
    Per-cell violins behind when `violin`. Returns the y top the caller should scale to."""
    genes = df["gene"].tolist()
    C = {ct: cell_values(ct, genes) for ct in (NONCLONAL, CLONAL)} if violin else {}
    top = 0.0
    for i, (_, r) in enumerate(df.iterrows()):
        xr, xm = i - 0.2, i + 0.2
        a, b = np.asarray(r["vals_a"]), np.asarray(r["vals_b"])
        if violin:
            for x, ct, col in ((xr, NONCLONAL, REACT), (xm, CLONAL, MALIG)):
                v = C[ct][r["gene"]].to_numpy()
                _violin(ax, v, x, col)
                if (v > 0).sum() >= 20:
                    top = max(top, float(np.quantile(v[v > 0], 0.98)))
        for va, vb in zip(a, b):
            ax.plot([xr, xm], [vb, va], color="#9e9e9e", linewidth=0.4, alpha=0.5, zorder=2)
        for x, v, col in ((xr, b, REACT), (xm, a, MALIG)):
            ax.scatter(np.full(len(v), x), v, s=6, color=col, linewidth=0, zorder=3)
            if len(v):
                ax.hlines(np.median(v), x - 0.12, x + 0.12, color="#111111", linewidth=1.2,
                          zorder=4)
        top = max(top, float(np.nanmax(np.r_[a, b, 0.0])))
        _x_annot(ax, i, r)
    _gene_ticks(ax, df)
    return top


def cell_violins(ax, df):
    """Per-cell violins, reactive | malignant, donor means as dots. For genes the clone does not
    express the empty panel is the result, so it is drawn rather than dropped."""
    rng = np.random.default_rng(0)
    C = {ct: cell_values(ct, df["gene"].tolist()) for ct in (NONCLONAL, CLONAL)}
    for i, (_, r) in enumerate(df.iterrows()):
        for x, ct, col, v in ((i - 0.2, NONCLONAL, REACT, r["vals_b"]),
                              (i + 0.2, CLONAL, MALIG, r["vals_a"])):
            _violin(ax, C[ct][r["gene"]].to_numpy(), x, col)
            v = np.asarray(v)
            ax.scatter(x + rng.uniform(-0.06, 0.06, len(v)), v, s=6, color=col,
                       edgecolor="white", linewidth=0.2, zorder=3)
            if len(v):
                ax.hlines(np.median(v), x - 0.12, x + 0.12, color="#111111", linewidth=1.2,
                          zorder=4)
        _x_annot(ax, i, r)
    _gene_ticks(ax, df)


def group_brackets(ax, groups, y=1.03):
    """Label runs of genes above a panel: {name: [genes]} in plotted order."""
    i = 0
    for name, gs in groups.items():
        x0, x1 = i - 0.35, i + len(gs) - 1 + 0.35
        ax.plot([x0, x1], [y, y], transform=ax.get_xaxis_transform(), color="#333333",
                linewidth=0.7, clip_on=False)
        ax.text((x0 + x1) / 2, y + 0.015, name, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=6.5)
        i += len(gs)


def bracket(ax, x0, x1, y, text, h):
    ax.plot([x0, x0, x1, x1], [y - h, y, y, y - h], color="#333333", linewidth=0.6)
    ax.text((x0 + x1) / 2, y, text, ha="center", va="bottom", fontsize=6)


def row_brackets(ax, spans, sub=None, pad=6):
    """Bracket + bold label per row group ({name: (y_lo, y_hi)}) left of the y tick labels;
    `sub` = {name: grey second line}. The x is measured from the drawn tick labels, so call it
    once everything else is on the figure."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    x_px = min(t.get_window_extent(r).x0 for t in ax.get_yticklabels() if t.get_text())
    x = ax.transAxes.inverted().transform((x_px, 0))[0]
    w = 3 * fig.dpi / 72 / ax.get_window_extent(r).width
    tr = offset_copy(ax.get_yaxis_transform(), fig=fig, x=-pad, units="points")
    for name, (lo, hi) in spans.items():
        ax.plot([x, x - w, x - w, x], [lo - 0.35, lo - 0.35, hi + 0.35, hi + 0.35],
                transform=tr, color="#333333", linewidth=0.7, clip_on=False)
        mid, s = (lo + hi) / 2, (sub or {}).get(name)
        ax.text(x - w, mid, name, ha="right", va="center", fontsize=6.5, fontweight="bold",
                transform=offset_copy(tr, fig=fig, x=-3, y=3.5 if s else 0, units="points"))
        if s:
            ax.text(x - w, mid, s, ha="right", va="center", fontsize=5.5, color="#555555",
                    transform=offset_copy(tr, fig=fig, x=-3, y=-4, units="points"))


def _dot(**k):
    return plt.Line2D([], [], linestyle="", marker="o", markersize=5, **k)


def dir_handles(a="MF", b="healthy"):
    h = [_dot(color=UP, label=f"higher in {a} (q < 0.05)"),
         _dot(color=DOWN, label=f"lower in {a} (q < 0.05)"),
         _dot(color=NS, label="not significant"),
         _dot(markerfacecolor="white", markeredgecolor=NS, label="not testable")]
    return h + ([_dot(markerfacecolor="white", markeredgecolor=HC_COL, label=b)] if b else [])


ARM_HANDLES = [_dot(color=REACT, label=LABELS[NONCLONAL]), _dot(color=MALIG, label=LABELS[CLONAL])]


def caption(fig, text):
    """Wrapped caption under everything already drawn (tick labels, legends included)."""
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    fig.text(bb.x0 / fig.get_figwidth(), bb.y0 / fig.get_figheight() - 0.015,
             textwrap.fill(text, int(fig.get_figwidth() * 21)), fontsize=5.8, color="#555555",
             ha="left", va="top")


STATS_NOTE = (f"MF = TCR cohort ({len(TCR_SAMPLES)} samples); healthy = atlas HC donors. "
              f"Replicate = donor mean log1p CP10K over >= {cfg.MIN_CELLS} cells. Malignant vs "
              f"reactive CD4: paired Wilcoxon signed-rank on the per-biopsy difference; otherwise "
              f"donor-level Mann-Whitney. BH over the tests in this figure (* < 0.05, ** < 0.01, "
              f"*** < 0.001); n.t. = not testable (< {cfg.MIN_DETECTION:.0%} of cells detected in "
              f"both arms, or < {cfg.MIN_DONORS_PER_ARM} donors per arm). Recomputed here: these "
              f"are not nb28's q-values.")

print(f"TCR cohort: {len(TCR_SAMPLES)} MF samples, "
      f"{adata.obs.loc[IS_MF_TCR, cfg.DONOR_KEY].nunique()} donors | "
      f"MF cells kept {int(IS_MF_TCR.sum()):,} of {int(IS_MF.sum()):,}")
```

```
TCR cohort: 38 MF samples, 38 donors | MF cells kept 470,660 of 902,779
```

```python  # cell 8
# ============================================================================
# FB  The clone does not instruct the microenvironment; what it does carry
# ============================================================================
# B1 is the T-cell -> stroma ligands the clone would need to instruct fibroblasts; B2, on the
# same y scale, is what separates clone from reactive in the same biopsy. B1 is drawn, not
# dropped: empty panels are the result.
FB1 = {"T-cell -> stroma ligands": ["CD40LG", "LTA", "IL17A", "IL22", "IL11", "TGFB1"]}
FB2 = {"protease inhibitors": ["TIMP1", "CSTB", "SERPINB1", "SERPINB9"],
       "lysosomal cathepsins": ["CTSA", "CTSD", "CTSH"],
       "granzymes": ["GZMB", "GZMM"]}
_g1 = [g for gs in FB1.values() for g in gs]
_g2 = [g for gs in FB2.values() for g in gs]

FB = bh(t_paired(_g1) + t_paired(_g2))
FB["panel"] = ["B1"] * len(_g1) + ["B2"] * len(_g2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.4), sharey=True,
                               gridspec_kw={"width_ratios": [len(_g1), len(_g2)],
                                            "wspace": 0.08})
cell_violins(ax1, FB[FB["panel"] == "B1"])
top = paired(ax2, FB[FB["panel"] == "B2"])
top = max(top, *(float(np.nanmax(np.r_[a, b, 0.0])) for a, b in
                 FB.loc[FB["panel"] == "B1", ["vals_a", "vals_b"]].itertuples(index=False)))
ax1.set_ylim(-0.04 * top, top / 0.82)
ax1.set_ylabel(XLAB, fontsize=7)
ax2.tick_params(labelleft=False)
det_label(ax1)
group_brackets(ax1, FB1)
group_brackets(ax2, FB2)
fig.legend(handles=ARM_HANDLES, fontsize=6.5, frameon=False, ncol=2, loc="lower right",
           bbox_to_anchor=(0.99, 0.99))
DC._finish(fig, fig28b("FB_what_the_clone_carries"), True)
save_source(FB, "FB_source")
logfig("FB", fig28b("FB_what_the_clone_carries"),
       "T-cell -> stroma ligands the clone lacks vs its inhibitor / lysosomal / granzyme cargo",
       "object, TCR cohort; paired signed-rank on per-biopsy means recomputed, BH within figure")
show_tests(FB, ("panel", "gene", "n_pairs", "med_a", "med_b", "n_pos", "n_neg", "det_a",
                "det_b", "q", "direction"))
```

```
<Figure size 1020x340 with 2 Axes>
  FB -> deg28b_FB_what_the_clone_carries.png
panel     gene  n_pairs    med_a  med_b  n_pos  n_neg    det_a   det_b       q  direction
   B1   CD40LG       33    0.268  0.234     20     13    0.234   0.156   0.255         ns
   B1      LTA       33    0.164  0.145     18     15    0.172  0.0717    0.38         ns
   B1    IL17A       33        0      0      5     14  0.00331  0.0023     NaN not tested
   B1     IL22       33 0.000933 0.0134     14     16   0.0853 0.00735    0.38         ns
   B1     IL11       33 0.000112      0     14      5 0.000741 0.00027     NaN not tested
   B1    TGFB1       33    0.875  0.832     20     13    0.533     0.5   0.255         ns
   B2    TIMP1       33    0.902  0.493     24      9    0.417   0.155 0.00444         up
   B2     CSTB       33     1.23  0.936     26      7    0.666   0.434 0.00089         up
   B2 SERPINB1       33    0.351  0.338     20     13    0.271   0.145   0.255         ns
   B2 SERPINB9       33    0.254   0.29     18     15      0.2   0.154   0.569         ns
   B2     CTSA       33    0.403  0.378     18     15     0.33   0.155   0.255         ns
   B2     CTSD       33    0.557  0.536     17     16    0.462   0.239   0.255         ns
   B2     CTSH       33    0.248  0.123     23     10    0.288   0.068 0.00423         up
   B2     GZMB       33   0.0443  0.193     12     21     0.22  0.0775    0.93         ns
   B2     GZMM       33     0.36  0.352     17     16    0.321   0.132   0.255         ns
```

```python  # cell 9
# ============================================================================
# FC  LIGHT -> LTbetaR / HVEM -> RELB
# ============================================================================
# Read top to bottom as a chain: the ligand on the clone (1), its receptors on fibroblasts (2),
# the transcription factor they drive (3). One x axis, so the three rows are on one scale.
FC_FIBRO = ["F_apCAF", "F_inflammatory", "F_structural"]
prefetch(FC_FIBRO + ["F_mesenchymal"], ["TNFSF14", "TNFRSF14", "LTBR", "CD40", "RELB"])
FC_ROWS = {
    1: t_paired(["TNFSF14"])
       + [t_mwu(CLONAL, "TNFSF14", "MF", "MF", ct_b=NONCLONAL,
                label="vs Reactive CD4 (across donors)"),
          t_mwu(CLONAL, "TNFSF14", "MF", "HC", label="vs healthy-donor CD4")],
    2: [t_mwu(ct, "TNFRSF14") for ct in FC_FIBRO]
       + [t_mwu("F_structural", "LTBR")]
       + [t_mwu(ct, "CD40") for ct in FC_FIBRO],
    3: [t_mwu(ct, "RELB") for ct in ["F_mesenchymal", "F_inflammatory", "F_structural"]],
}
FC = bh([dict(r, row=k) for k, rs in FC_ROWS.items() for r in rs])
FC["ylabel"] = [r["label"] if r["row"] == 1 else f"{it(r['gene'])} · {LABELS[r['ct']]}"
                for _, r in FC.iterrows()]
# TNFSF14 is detected in few cells: say how few, and how many biopsies voted each way
FC["note"] = [("" if r["row"] != 1 else
               (f"{r['n_pos']:.0f}↑/{r['n_neg']:.0f}↓ · " if r["test"].startswith("paired")
                else "") + f"det {pct(r['det_a'])} | {pct(r['det_b'])}")
              for _, r in FC.iterrows()]
FC_TITLES = ["LIGAND  TNFSF14 (LIGHT) in malignant CD4 (filled) vs comparator (open)",
             "RECEPTORS  fibroblasts, MF vs healthy skin",
             "EFFECTOR  RELB in fibroblasts, MF vs healthy skin"]

nr = FC["row"].value_counts().reindex([1, 2, 3]).to_numpy()
fig = plt.figure(figsize=(7.0, 0.26 * nr.sum() + 2.4))
gs = fig.add_gridspec(3, 1, height_ratios=nr + 0.6, hspace=0.5,
                      left=0.40, right=0.76, top=0.9, bottom=0.12)
axes = []
for k in range(3):
    ax = fig.add_subplot(gs[k], sharex=axes[0] if axes else None)
    dumbbell(ax, FC[FC["row"] == k + 1], note_col="note")
    ax.set_title(FC_TITLES[k], fontsize=7, loc="left")
    if k < 2:
        ax.tick_params(labelbottom=False)
    axes.append(ax)
xmax = max(np.nanmax(np.r_[v, 0.0]) for col in ("vals_a", "vals_b") for v in FC[col])
axes[0].set_xlim(-0.02 * xmax, 1.05 * xmax)
axes[-1].set_xlabel(XLAB, fontsize=7)

# the chain: numbered rows joined by downward arrows in the left margin
cy = [(ax.get_position().y0 + ax.get_position().y1) / 2 for ax in axes]
for k, yc in enumerate(cy):
    fig.text(0.03, yc, str(k + 1), ha="center", va="center", fontsize=10, fontweight="bold",
             color="white", bbox=dict(boxstyle="circle,pad=0.35", facecolor="#333333",
                                      edgecolor="none"))
for y0, y1 in zip(cy[:-1], cy[1:]):
    fig.add_artist(FancyArrowPatch((0.03, y0 - 0.04), (0.03, y1 + 0.04),
                                   transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=10, color="#333333", linewidth=1.0))
fig.legend(handles=dir_handles("malignant / MF", "comparator (reactive or healthy)"),
           fontsize=6, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.55, 0.05))
fig.suptitle("LIGHT → LTβR / HVEM → RELB", fontsize=9.5, x=0.0, ha="left",
             y=0.99)
DC._finish(fig, fig28b("FC_light_ltbr_relb"), True)
save_source(FC, "FC_source")
logfig("FC", fig28b("FC_light_ltbr_relb"), "the LIGHT -> LTbetaR/HVEM -> RELB chain",
       "object, TCR cohort; paired signed-rank + donor-level MWU recomputed, BH within figure")
show_tests(FC, ("row", "label", "gene", "n_a", "n_b", "med_a", "med_b", "det_a", "q",
                "direction"))
```

```
<Figure size 700x578 with 3 Axes>
  FC -> deg28b_FC_light_ltbr_relb.png
 row                           label     gene  n_a  n_b  med_a  med_b  det_a        q direction
   1   vs Reactive CD4 (same biopsy)  TNFSF14   33   33  0.272  0.149   0.22 4.49e-06        up
   1 vs Reactive CD4 (across donors)  TNFSF14   33   38  0.272  0.155   0.22  0.00238        up
   1            vs healthy-donor CD4  TNFSF14   33   21  0.272  0.239   0.22    0.249        ns
   2      MHC-II+ fibroblast (apCAF) TNFRSF14   20   14    0.5  0.245  0.329 0.000309        up
   2         Inflammatory fibroblast TNFRSF14   22   19  0.402  0.195  0.307 2.26e-05        up
   2           Structural fibroblast TNFRSF14   21   20  0.344   0.21  0.238 0.000891        up
   2           Structural fibroblast     LTBR   21   20  0.515  0.343  0.434  0.00238        up
   2      MHC-II+ fibroblast (apCAF)     CD40   20   14  0.562   0.33  0.387   0.0216        up
   2         Inflammatory fibroblast     CD40   22   19  0.356    0.2  0.362  0.00138        up
   2           Structural fibroblast     CD40   21   20  0.165 0.0882  0.124  0.00238        up
   3          Mesenchymal fibroblast     RELB   13   18  0.392 0.0774  0.361  0.00495        up
   3         Inflammatory fibroblast     RELB   22   19  0.401 0.0901  0.361  0.00145        up
   3           Structural fibroblast     RELB   21   20  0.282 0.0674  0.191  0.00144        up
```

```python  # cell 10
# ============================================================================
# FD  Th2 axis
# ============================================================================
# Left to right: the transcription factor is on the clone (D1), the cytokines are below
# detection in both CD4 arms (D2, drawn on D1's scale), the receptor and its output are on the
# stroma (D3). IL4R is shown in all five fibroblast subsets: the reproducibility is the point.
FD1 = ["GATA3", "BATF", "STAT3", "TOX"]
FD2 = ["IL4", "IL5", "IL13", "IL9", "IL31", "IL22"]
FD3 = ([("IL4R", ct, "receptor") for ct in ["F_structural", "F_inflammatory", "F_apCAF",
                                            "F_mesenchymal", "F_reticular"]]
       + [(g, "F_structural", "output") for g in ["HAS2", "POSTN", "TNC"]]
       + [("TSLP", "Mac_FOLR2", "alarmin")])
prefetch(["F_structural", "F_inflammatory", "F_apCAF", "F_mesenchymal", "F_reticular",
          "Mac_FOLR2"], ["IL4R", "HAS2", "POSTN", "TNC", "TSLP"])

FD = bh(t_paired(FD1) + t_paired(FD2) + [dict(t_mwu(ct, g), group=grp) for g, ct, grp in FD3])
FD["panel"] = ["D1"] * len(FD1) + ["D2"] * len(FD2) + ["D3"] * len(FD3)
FD["ylabel"] = [f"{it(r['gene'])} · {LABELS[r['ct']]}" for _, r in FD.iterrows()]

fig = plt.figure(figsize=(12.5, 3.6))
gs = fig.add_gridspec(1, 4, width_ratios=[len(FD1), len(FD2), 2.3, 4.2], wspace=0.12,
                      left=0.05, right=0.93, top=0.8, bottom=0.2)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharey=ax1)
ax3 = fig.add_subplot(gs[3])
top = paired(ax1, FD[FD["panel"] == "D1"], violin=True)
cell_violins(ax2, FD[FD["panel"] == "D2"])
ax1.set_ylim(-0.04 * top, top / 0.82)
ax1.set_ylabel("log1p CP10K (violin = cells, dots = donor means)", fontsize=6.5)
ax2.tick_params(labelleft=False)
det_label(ax1)
dumbbell(ax3, FD[FD["panel"] == "D3"], group_col="group")
ax3.set_xlabel(XLAB, fontsize=7)
ax1.set_title("D1  clone: transcription factors (paired)", fontsize=7.5, loc="left")
ax2.set_title("D2  Th2 cytokines in CD4", fontsize=7.5, loc="left")
ax3.set_title("D3  stroma: receptor and output, MF vs healthy", fontsize=7.5, loc="left")
fig.legend(handles=ARM_HANDLES, fontsize=6.5, frameon=False, ncol=2, loc="upper left",
           bbox_to_anchor=(0.05, 0.02))
fig.legend(handles=dir_handles("MF", "healthy"), fontsize=6, frameon=False, ncol=3,
           loc="upper right", bbox_to_anchor=(0.97, 0.02))
fig.suptitle("Th2 axis", fontsize=9.5, x=0.0, ha="left", y=0.99)
DC._finish(fig, fig28b("FD_th2_axis"), True)
save_source(FD, "FD_source")
logfig("FD", fig28b("FD_th2_axis"), "the Th2 axis: TF on the clone, cytokines, stromal answer",
       "object, TCR cohort; paired signed-rank + donor-level MWU recomputed, BH within figure")
show_tests(FD, ("panel", "gene", "ct", "n_a", "n_b", "med_a", "med_b", "det_a", "det_b", "q",
                "direction"))
```

```
<Figure size 1250x360 with 3 Axes>
  FD -> deg28b_FD_th2_axis.png
panel  gene             ct  n_a  n_b    med_a   med_b    det_a    det_b        q  direction
   D1 GATA3  CD4_malignant   33   33    0.936   0.635    0.529    0.318  0.00422         up
   D1  BATF  CD4_malignant   33   33     1.29   0.872     0.64    0.376  0.00468         up
   D1 STAT3  CD4_malignant   33   33     1.08   0.868    0.604    0.431 0.000899         up
   D1   TOX  CD4_malignant   33   33     0.64    0.28    0.427    0.112 7.58e-06         up
   D2   IL4  CD4_malignant   33   33  0.00354 0.00274   0.0283  0.00218   0.0168         up
   D2   IL5  CD4_malignant   33   33        0       0 0.000973 0.000557      NaN not tested
   D2  IL13  CD4_malignant   33   33    0.112  0.0299    0.103    0.013  0.00468         up
   D2   IL9  CD4_malignant   33   33        0       0 0.000421 0.000388      NaN not tested
   D2  IL31  CD4_malignant   33   33        0       0 0.000784 0.000203      NaN not tested
   D2  IL22  CD4_malignant   33   33 0.000933  0.0134   0.0853  0.00735    0.318         ns
   D3  IL4R   F_structural   21   20    0.131  0.0625    0.136   0.0697 0.000151         up
   D3  IL4R F_inflammatory   22   19    0.169  0.0607    0.158    0.144  0.00154         up
   D3  IL4R        F_apCAF   20   14    0.155   0.101    0.103    0.101   0.0168         up
   D3  IL4R  F_mesenchymal   13   18   0.0669   0.027   0.0996   0.0237   0.0182         up
   D3  IL4R    F_reticular   16   17    0.148  0.0762    0.171   0.0977  0.00468         up
   D3  HAS2   F_structural   21   20    0.177   0.161    0.207    0.153    0.285         ns
   D3 POSTN   F_structural   21   20     1.59   0.787    0.476    0.303   0.0111         up
   D3   TNC   F_structural   21   20     0.95   0.173    0.413    0.117 4.83e-05         up
   D3  TSLP      Mac_FOLR2   25   15   0.0052  0.0208   0.0149   0.0228   0.0157       down
```

```python  # cell 11
# ============================================================================
# FE  Hyaluronan — synthesis, fragmentation, receptor
# ============================================================================
# No curated ligand-receptor edge exists for HA, so the claim rests on who makes it (synthesis),
# who cuts it (fragmentation) and who can bind the fragments (CD44) -- on the stroma in E1 (MF vs
# healthy), on the CD4 arms in E2.
FE1 = ([("HAS2", ct, "synthesis") for ct in ["F_structural", "F_mesenchymal"]]
       + [("HAS1", "F_inflammatory", "synthesis")]
       + [("CEMIP", ct, "fragmentation") for ct in ["F_apCAF", "F_inflammatory", "F_reticular",
                                                    "F_structural"]]
       + [("HYAL2", ct, "fragmentation") for ct in ["F_mesenchymal", "Mac_FOLR2", "Mono_CD14",
                                                    "cDC2"]]
       + [("CD44", ct, "receptor") for ct in ["F_reticular", "F_inflammatory", "F_structural"]])
prefetch(sorted({ct for _, ct, _ in FE1}), ["HAS1", "HAS2", "CEMIP", "HYAL2", "CD44"])
prefetch([CLONAL, NONCLONAL], ["CD44", "HYAL3"])

_e1 = [dict(t_mwu(ct, g), group=grp) for g, ct, grp in FE1]

_e2 = (t_paired(["CD44", "HYAL3"])
       + [t_mwu(CLONAL, "CD44", "MF", "HC", label="Malignant vs healthy-donor CD4"),
          t_mwu(NONCLONAL, "CD44", "MF", "HC", label="Reactive vs healthy-donor CD4")])
FE = bh(_e1 + _e2)
FE["panel"] = ["E1"] * len(_e1) + ["E2"] * len(_e2)
FE["ylabel"] = [f"{it(r['gene'])} · {LABELS[r['ct']]}" for _, r in FE.iterrows()]
E2 = FE[FE["panel"] == "E2"]


def _q(gene, test, ct=CLONAL):
    r = E2[(E2["gene"] == gene) & E2["test"].str.startswith(test) & (E2["ct"] == ct)]
    return float(r["q"].iloc[0]) if len(r) else np.nan


def _bars(ax, gene, arms):
    """Bar = median of donor means, dots = donors. `arms` = [(cell type, 'MF' | 'HC')]."""
    rng = np.random.default_rng(0)
    top = 0.0
    for i, (ct, arm) in enumerate(arms):
        v = donor_means(ct, arm, [gene])[0][gene].to_numpy()
        if not len(v):
            continue
        ax.bar(i, np.median(v), width=0.62, color=ARM_COL[ct], alpha=0.3,
               edgecolor=ARM_COL[ct], linewidth=0.8)
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=6, color=ARM_COL[ct],
                   edgecolor="white", linewidth=0.2, zorder=3)
        top = max(top, float(v.max()))
    short = {CLONAL: "Malignant\nCD4", NONCLONAL: "Reactive\nCD4", HC_CD4: "Healthy-\ndonor CD4"}
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([short[ct] for ct, _ in arms], fontsize=6)
    ax.set_title(it(gene), fontsize=8)
    _clean(ax)
    return top


def _stroma_bars(ax, df):
    """One slot per test: healthy bar (hollow) | MF bar (coloured by direction), dots = donors,
    bar = median of donor means, star = MF vs healthy."""
    rng = np.random.default_rng(0)
    top = 0.0
    for i, (_, r) in enumerate(df.iterrows()):
        c = DIR_COLOR[r["direction"]]
        for x, v, fc, ec in ((i - 0.2, r["vals_b"], "white", HC_COL), (i + 0.2, r["vals_a"], c, c)):
            v = np.asarray(v)
            if not len(v):
                continue
            ax.bar(x, np.median(v), width=0.36, color=fc, alpha=0.3 if fc != "white" else 1,
                   edgecolor=ec, linewidth=0.8)
            ax.scatter(x + rng.uniform(-0.08, 0.08, len(v)), v, s=4, color=ec,
                       edgecolor="white", linewidth=0.2, zorder=3)
            top = max(top, float(v.max()))
        ax.text(i, 0.97, sig(r["q"]), transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=7)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["ylabel"], rotation=45, ha="right", fontsize=6)
    ax.set_xlim(-0.6, len(df) - 0.4)
    _clean(ax)
    return top


E1 = FE[FE["panel"] == "E1"]
fig = plt.figure(figsize=(10.5, 3.8))
gs = fig.add_gridspec(1, 3, width_ratios=[len(E1), 2.4, 1.6], wspace=0.3,
                      left=0.06, right=0.98, top=0.82, bottom=0.34)
ax1, ax2, ax3 = (fig.add_subplot(gs[k]) for k in range(3))
t = _stroma_bars(ax1, E1)
ax1.set_ylim(0, t / 0.85)
ax1.set_ylabel(XLAB, fontsize=7)
group_brackets(ax1, {g: d for g, d in E1.groupby("group", sort=False)["gene"]})
ax1.set_title("E1  stroma and myeloid, MF vs healthy", fontsize=7.5, loc="left", pad=16)

t = _bars(ax2, "CD44", [(CLONAL, "MF"), (NONCLONAL, "MF"), (HC_CD4, "HC")])
for k, (x0, x1, q) in enumerate([(0, 1, _q("CD44", "paired")),
                                 (1, 2, _q("CD44", "MWU", NONCLONAL)),
                                 (0, 2, _q("CD44", "MWU", CLONAL))]):
    bracket(ax2, x0, x1, t * (1.08 + 0.13 * k), sig(q), h=0.03 * t)
ax2.set_ylim(0, t * 1.5)
ax2.set_ylabel(XLAB, fontsize=7)
t = _bars(ax3, "HYAL3", [(CLONAL, "MF"), (NONCLONAL, "MF")])
t = t if t > 0 else 1.0
bracket(ax3, 0, 1, t * 1.08, sig(_q("HYAL3", "paired")), h=0.03 * t)
ax3.set_ylim(0, t * 1.3)
ax2.text(0.0, 1.13, "E2  CD4", fontsize=7.5, transform=ax2.transAxes, ha="left")
fig.legend(handles=dir_handles("MF", "healthy"), fontsize=6, frameon=False, ncol=5,
           loc="upper left", bbox_to_anchor=(0.02, 0.04))
fig.suptitle("Hyaluronan: synthesis, fragmentation and CD44 in the stroma and CD4",
             fontsize=9.5, x=0.0, ha="left", y=0.99)
DC._finish(fig, fig28b("FE_hyaluronan"), True)
save_source(FE, "FE_source")
logfig("FE", fig28b("FE_hyaluronan"), "the hyaluronan axis: synthesis, fragmentation, CD44 in stroma and CD4",
       "object, TCR cohort; paired signed-rank + donor-level MWU recomputed, BH within figure")
show_tests(FE, ("panel", "label", "gene", "n_a", "n_b", "med_a", "med_b", "q", "direction"))
```

```
<Figure size 1050x380 with 3 Axes>
  FE -> deg28b_FE_hyaluronan.png
panel                          label  gene  n_a  n_b  med_a   med_b        q  direction
   E1          Structural fibroblast  HAS2   21   20  0.177   0.161    0.284         ns
   E1         Mesenchymal fibroblast  HAS2   13   18  0.201  0.0917   0.0191         up
   E1        Inflammatory fibroblast  HAS1   22   19 0.0535  0.0118   0.0051         up
   E1     MHC-II+ fibroblast (apCAF) CEMIP   20   14 0.0728  0.0263  0.00329         up
   E1        Inflammatory fibroblast CEMIP   22   19  0.231 0.00509 7.27e-05         up
   E1           Reticular fibroblast CEMIP   16   17 0.0472       0 0.000132         up
   E1          Structural fibroblast CEMIP   21   20 0.0407 0.00319 2.18e-05         up
   E1         Mesenchymal fibroblast HYAL2   13   18 0.0915  0.0568    0.219         ns
   E1   Resident macrophage (FOLR2+) HYAL2   25   15 0.0564  0.0374    0.135         ns
   E1     Classical monocyte (CD14+) HYAL2   12    0 0.0224     NaN      NaN not tested
   E1               Conventional DC2 HYAL2   26   17 0.0364  0.0105  0.00884         up
   E1           Reticular fibroblast  CD44   16   17   1.77    1.28  0.00244         up
   E1        Inflammatory fibroblast  CD44   22   19   2.07    1.16 0.000663         up
   E1          Structural fibroblast  CD44   21   20   1.62    1.29   0.0149         up
   E2  vs Reactive CD4 (same biopsy)  CD44   33   33   1.58    1.77   0.0252       down
   E2  vs Reactive CD4 (same biopsy) HYAL3   33   33 0.0248  0.0105 7.27e-05         up
   E2 Malignant vs healthy-donor CD4  CD44   33   21   1.58    1.59    0.763         ns
   E2  Reactive vs healthy-donor CD4  CD44   38   21   1.73    1.59    0.109         ns
```

```python  # cell 12
# ============================================================================
# FF  The TGF-β loop runs inside the fibroblast compartment
# ============================================================================
# Left: every stage of the loop -- latent ligand, docking, activation, receptor, output -- in
# fibroblasts, MF vs healthy. Right: the ligands and activators in CD4 on the same x scale,
# clone vs reactive in the same biopsy. Myeloid TGFB1 is up too; it belongs with FB (the clone
# does not instruct), not here.
FF_STAGES = {
    "LATENT LIGAND": [("TGFB1", ["F_inflammatory", "F_apCAF"]),
                      ("TGFB3", ["F_inflammatory", "F_apCAF"])],
    "DOCKING": [("LRRC32", ["F_structural", "F_inflammatory", "F_apCAF"])],
    "ACTIVATION": [("ITGAV", ["F_structural", "F_apCAF"]), ("ITGB8", ["F_structural"]),
                   ("THBS1", ["F_structural", "F_inflammatory"]), ("TGM2", ["F_inflammatory"])],
    "RECEPTOR": [("TGFBR2", ["F_reticular", "F_structural", "F_mesenchymal"])],
    "OUTPUT": [(g, ["F_structural"]) for g in ["TNC", "POSTN", "PRSS23", "ADAMTS2", "FAP",
                                               "LOXL2", "SERPINE1", "LTBP2"]],
}
FF_CD4 = ["TGFB1", "TGFB3", "LRRC32", "ITGB8", "THBS1", "TGM2"]
_ff = [(st, g, ct) for st, gs in FF_STAGES.items() for g, cts in gs for ct in cts]
prefetch(sorted({ct for *_, ct in _ff}), sorted({g for _, g, _ in _ff}))

FF = bh([dict(t_mwu(ct, g), group=st, panel="F1") for st, g, ct in _ff]
        + [dict(r, panel="F2") for r in t_paired(FF_CD4)])
SHORT = {ct: LABELS[ct].replace(" fibroblast", "") for *_, ct in _ff}
FF["ylabel"] = [f"{it(r['gene'])} · {SHORT.get(r['ct'], '')}" for _, r in FF.iterrows()]
FF["note"] = [f"Δ {d:+.2f}" if np.isfinite(d) else "" for d in FF["delta"]]

# stages in the order above; within OUTPUT, largest MF - healthy shift on top
F1 = FF[FF["panel"] == "F1"]
F1 = (F1.assign(_s=F1["group"].map({s: i for i, s in enumerate(FF_STAGES)}),
                _e=np.where(F1["group"] == "OUTPUT", -F1["delta"], 0.0))
      .sort_values(["_s", "_e"], kind="stable").drop(columns=["_s", "_e"]))
F2 = FF[FF["panel"] == "F2"]
FF = pd.concat([F1, F2])

# The caption states these; print whether the data reproduces them rather than assume it.
print("claim checks:")
for st, d in F1.groupby("group", sort=False):
    print(f"  {st:<13} {'ok  ' if (d['direction'] == 'up').any() else 'FAIL'} "
          f"{int((d['direction'] == 'up').sum())}/{len(d)} rows up in MF | "
          f"Δ {d['delta'].min():+.2f} to {d['delta'].max():+.2f}")
_up, _dn = (F2.loc[F2["direction"] == k, "gene"].tolist() for k in ("up", "down"))
print(f"  CD4 flat    {'ok  ' if not _up else 'FAIL'} higher in clone: {_up or 'none'} | "
      f"lower in clone: {_dn or 'none'}")
_lr = F2.set_index("gene").loc["LRRC32"]
print(f"  LRRC32      {'ok  ' if _lr['direction'] == 'down' else 'FAIL'} reactive > clone in "
      f"{_lr['n_neg']:.0f}/{_lr['n_pairs']:.0f} biopsies, {sig(_lr['q'])}")


def cd4_hviolins(ax, df):
    """Horizontal twin of cell_violins: one row per gene (first on top), malignant above
    reactive. Violin = cells, zeros included; dots = per-biopsy means; % of cells detected under
    each gene. Returns the 98th percentile of expressing cells, for the shared x limit."""
    rng = np.random.default_rng(0)
    C = {ct: cell_values(ct, df["gene"].tolist()) for ct in (NONCLONAL, CLONAL)}
    n, top = len(df), 0.0
    y = np.arange(n)[::-1]
    for yi, (_, r) in zip(y, df.iterrows()):
        for dy, ct, col, v in ((0.2, CLONAL, MALIG, r["vals_a"]),
                               (-0.2, NONCLONAL, REACT, r["vals_b"])):
            c = C[ct][r["gene"]].to_numpy()
            _violin(ax, c, yi + dy, col, horizontal=True)
            if (c > 0).sum() >= 20:
                top = max(top, float(np.quantile(c[c > 0], 0.98)))
            v = np.asarray(v)
            ax.scatter(v, yi + dy + rng.uniform(-0.06, 0.06, len(v)), s=6, color=col,
                       edgecolor="white", linewidth=0.2, zorder=3)
            if len(v):
                ax.vlines(np.median(v), yi + dy - 0.12, yi + dy + 0.12, color="#111111",
                          linewidth=1.2, zorder=4)
        votes = f"  {r['n_pos']:.0f}↑/{r['n_neg']:.0f}↓" if np.isfinite(r["q"]) else ""
        flip = "  reactive > clone" if r["direction"] == "down" else ""
        ax.text(1.01, yi, f"{sig(r['q'])}{votes}{flip}", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=6,
                color={"up": UP, "down": DOWN}.get(r["direction"], "#333333"))
        for k, (d, col) in enumerate(((r["det_a"], MALIG), (r["det_b"], REACT))):
            ax.text(0, yi, pct(d), ha="right", va="top", fontsize=5.5, color=col,
                    transform=offset_copy(ax.get_yaxis_transform(), fig=ax.figure, x=-3,
                                          y=-5 - 6.5 * k, units="points"))
    ax.set_yticks(y)
    ax.set_yticklabels([it(g) for g in df["gene"]], fontsize=6.5)
    ax.set_ylim(-0.7, n - 0.3)
    _clean(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return top


units = len(F1) + 0.6 * (F1["group"].nunique() - 1)
fig = plt.figure(figsize=(9.0, 0.2 * units + 1.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.85], wspace=0.55,
                      left=0.27, right=0.88, top=0.9, bottom=0.12)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)
spans = dumbbell(ax1, F1, group_col="group", note_col="note", brackets=True)
top = cd4_hviolins(ax2, F2)
xmax = max([top] + [float(np.nanmax(np.r_[v, 0.0])) for col in ("vals_a", "vals_b")
                    for v in FF[col]])
ax1.set_xlim(-0.02 * xmax, 1.05 * xmax)
ax1.set_xlabel(XLAB, fontsize=7)
ax2.set_xlabel("log1p CP10K (violin = cells, zeros included; dots = biopsy means)", fontsize=7)
ax1.set_title("Fibroblasts, MF vs healthy skin", fontsize=7.5, loc="left")
ax2.set_title("CD4, malignant vs reactive (same biopsy)\n"
              "under each gene: % cells detected, malignant | reactive", fontsize=7, loc="left")
fig.legend(handles=dir_handles("MF", "healthy"), fontsize=6, frameon=False, ncol=5,
           loc="upper left", bbox_to_anchor=(0.02, 0.05))
fig.legend(handles=ARM_HANDLES, fontsize=6, frameon=False, ncol=2, loc="upper right",
           bbox_to_anchor=(0.98, 0.05))
fig.suptitle("The TGF-β loop runs inside the fibroblast compartment", fontsize=9.5, x=0.0,
             ha="left", y=0.99)
_span = F1.groupby("group", sort=False)["delta"].agg(["min", "max"])
row_brackets(ax1, spans, sub={st: f"Δ {r['min']:+.2f} to {r['max']:+.2f}"
                              for st, r in _span.iterrows()})
caption(fig,
        "The loop is self-contained; no T-cell input is required. Left: each stage of the loop "
        "in fibroblasts, MF vs healthy skin; output genes ordered by effect, largest on top. "
        "Right: the same ligands and activators in CD4 on the same x scale. The latent ligands "
        "and activators move by small amounts and the receptor and output by large ones (Δ = MF "
        "minus healthy, median of donor means, per row and per stage): the claim is a complete "
        "parts list at the detection limit, not a set of large effects. "
        f"LRRC32 is higher in reactive than malignant CD4 ({sig(_lr['q'])}; detected in "
        f"{pct(_lr['det_b'])} vs {pct(_lr['det_a'])} of cells) -- an internal positive control "
        "that CD4 cells are detected for this gene, so a flat clone is not a sensitivity failure. "
        + STATS_NOTE)
DC._finish(fig, fig28b("FF_tgfb_loop"), True)
save_source(FF, "FF_source")
logfig("FF", fig28b("FF_tgfb_loop"),
       "the TGF-beta loop (latent ligand, docking, activation, receptor, output) in fibroblasts "
       "vs the same genes in CD4",
       "object, TCR cohort; donor-level MWU + paired signed-rank recomputed, BH within figure")
show_tests(FF, ("panel", "group", "gene", "ct", "n_a", "n_b", "med_a", "med_b", "det_a",
                "det_b", "q", "direction"))
```

```
claim checks:
  LATENT LIGAND ok   4/4 rows up in MF | Δ +0.04 to +0.28
  DOCKING       ok   3/3 rows up in MF | Δ +0.08 to +0.27
  ACTIVATION    ok   5/6 rows up in MF | Δ +0.00 to +0.37
  RECEPTOR      ok   1/3 rows up in MF | Δ +0.11 to +0.15
  OUTPUT        ok   8/8 rows up in MF | Δ +0.11 to +0.81
  CD4 flat    ok   higher in clone: none | lower in clone: ['LRRC32']
  LRRC32      ok   reactive > clone in 26/33 biopsies, ***

<Figure size 900x718 with 2 Axes>
  FF -> deg28b_FF_tgfb_loop.png
panel         group     gene             ct  n_a  n_b   med_a   med_b    det_a    det_b        q  direction
   F1 LATENT LIGAND    TGFB1 F_inflammatory   22   19   0.434   0.152     0.32    0.247 0.000164         up
   F1 LATENT LIGAND    TGFB1        F_apCAF   20   14   0.346   0.165    0.266    0.141 0.000322         up
   F1 LATENT LIGAND    TGFB3 F_inflammatory   22   19   0.185  0.0776    0.252    0.069   0.0023         up
   F1 LATENT LIGAND    TGFB3        F_apCAF   20   14   0.153   0.111    0.125   0.0933   0.0194         up
   F1       DOCKING   LRRC32   F_structural   21   20   0.299   0.119    0.272    0.138 0.000201         up
   F1       DOCKING   LRRC32 F_inflammatory   22   19    0.38   0.113    0.273    0.218 0.000164         up
   F1       DOCKING   LRRC32        F_apCAF   20   14   0.112  0.0278   0.0734   0.0375 0.000164         up
   F1    ACTIVATION    ITGAV   F_structural   21   20    0.44   0.216    0.339     0.21 0.000549         up
   F1    ACTIVATION    ITGAV        F_apCAF   20   14   0.473   0.273    0.322    0.241  0.00204         up
   F1    ACTIVATION    ITGB8   F_structural   21   20  0.0225  0.0175    0.029   0.0142   0.0411         up
   F1    ACTIVATION    THBS1   F_structural   21   20   0.675    0.48    0.382    0.334   0.0404         up
   F1    ACTIVATION    THBS1 F_inflammatory   22   19   0.512   0.141    0.317    0.317  0.00204         up
   F1    ACTIVATION     TGM2 F_inflammatory   22   19  0.0069       0  0.00897  0.00477      NaN not tested
   F1      RECEPTOR   TGFBR2    F_reticular   16   17   0.544   0.389    0.445    0.338   0.0715         ns
   F1      RECEPTOR   TGFBR2   F_structural   21   20    0.41   0.304    0.378    0.292   0.0404         up
   F1      RECEPTOR   TGFBR2  F_mesenchymal   13   18   0.337   0.191    0.361    0.194   0.0613         ns
   F1        OUTPUT    POSTN   F_structural   21   20    1.59   0.787    0.476    0.303   0.0104         up
   F1        OUTPUT      TNC   F_structural   21   20    0.95   0.173    0.413    0.117 0.000163         up
   F1        OUTPUT   PRSS23   F_structural   21   20    1.09   0.354    0.585    0.215 0.000323         up
   F1        OUTPUT    LOXL2   F_structural   21   20    0.46   0.141    0.306    0.114 0.000164         up
   F1        OUTPUT SERPINE1   F_structural   21   20   0.353  0.0667    0.213    0.084 0.000323         up
   F1        OUTPUT      FAP   F_structural   21   20   0.533    0.31    0.389    0.261   0.0029         up
   F1        OUTPUT  ADAMTS2   F_structural   21   20   0.424   0.275    0.373    0.251  0.00167         up
   F1        OUTPUT    LTBP2   F_structural   21   20   0.216   0.109    0.191    0.076  0.00165         up
   F2           NaN    TGFB1  CD4_malignant   33   33   0.875   0.832    0.533      0.5    0.101         ns
   F2           NaN    TGFB3  CD4_malignant   33   33 0.00177       0  0.00225  0.00162      NaN not tested
   F2           NaN   LRRC32  CD4_malignant   33   33 0.00161  0.0343  0.00553   0.0248 0.000291       down
   F2           NaN    ITGB8  CD4_malignant   33   33 0.00541 0.00608   0.0154  0.00405    0.264         ns
   F2           NaN    THBS1  CD4_malignant   33   33  0.0105  0.0224   0.0187   0.0484   0.0785         ns
   F2           NaN     TGM2  CD4_malignant   33   33       0       0 0.000567 0.000338      NaN not tested
```

```python  # cell 13
# ============================================================================
# §R  What was written
# ============================================================================
IDX = pd.DataFrame(FIG_INDEX)
IDX.to_csv(tab28b("figure_index"), index=False)
print(IDX.to_string(index=False))
print(f"\n{len(IDX)} figures -> {cfg.FIG_DIR}/{PREFIX}*.png")
print(f"source tables -> {cfg.TAB_DIR}/{PREFIX}*.csv")
print("nothing under deg28_* was written; nb28's tables are the input, not the output")
```

```
figure                                 file                                                                                                         question                                                                                  source                                                          note
   F1a     deg28b_F1a_cd4_family_scores.png                                                          degradome family load, clone vs reactive vs healthy CD4                                    object (donor medians) + deg28_external_donor_median                                                              
   F1b    deg28b_F1b_cd4_protease_genes.png                                                                        which protease, if any, the clone carries     object; donor-level MWU recomputed over the drawn genes (HC CD4 is not an nb28 arm) q recomputed figure-wide: the 4-arm set includes atlas HC CD4
   F2a          deg28b_F2a_fibro_vs_cd4.png                                                         protease genes per cell type, CD4 arms vs fibro, MF skin                                            object (per-cell log1p CP10K; donor medians)                                                              
   F2b        deg28b_F2b_myeloid_vs_cd4.png                                                       protease genes per cell type, CD4 arms vs myeloid, MF skin                                            object (per-cell log1p CP10K; donor medians)                                                              
    FB deg28b_FB_what_the_clone_carries.png                           T-cell -> stroma ligands the clone lacks vs its inhibitor / lysosomal / granzyme cargo object, TCR cohort; paired signed-rank on per-biopsy means recomputed, BH within figure                                                              
    FC        deg28b_FC_light_ltbr_relb.png                                                                          the LIGHT -> LTbetaR/HVEM -> RELB chain   object, TCR cohort; paired signed-rank + donor-level MWU recomputed, BH within figure                                                              
    FD               deg28b_FD_th2_axis.png                                                         the Th2 axis: TF on the clone, cytokines, stromal answer   object, TCR cohort; paired signed-rank + donor-level MWU recomputed, BH within figure                                                              
    FE             deg28b_FE_hyaluronan.png                                            the hyaluronan axis: synthesis, fragmentation, CD44 in stroma and CD4   object, TCR cohort; paired signed-rank + donor-level MWU recomputed, BH within figure                                                              
    FF              deg28b_FF_tgfb_loop.png the TGF-beta loop (latent ligand, docking, activation, receptor, output) in fibroblasts vs the same genes in CD4   object, TCR cohort; donor-level MWU + paired signed-rank recomputed, BH within figure                                                              

9 figures -> /home/projects/nyosef/zvise/MF/figures/degradome/deg28b_*.png
source tables -> /home/projects/nyosef/zvise/MF/tables/deg28b_*.csv
nothing under deg28_* was written; nb28's tables are the input, not the output
```
