# old/28d_cosmx_degradome_spatial.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/28d_cosmx_degradome_spatial.ipynb` and is never run._

# 28d — degradome in situ: CosMx early-MF skin (GSE337336)

Spatial follow-up to 28b_mc. 28b places proteases in cell types from scRNA (MMP2/MMP14 in fibroblasts, MMP9/MMP12 in myeloid) and finds T→stroma axes (TNFSF14→LTβR/HVEM, TGF-β, Th2→IL4R). Here: where those proteases sit in tissue, and whether they organise around the T-cell infiltrate.

**Data.** GSE337336 (Choi, Lee et al., Asan): CosMx 1000-plex, FFPE, 16 MF biopsies (8 patch / 8 plaque) on 4 slides, 65,717 cells, no HC. GEO ships per-cell tables only (counts + centroid + stain means; no images, transcripts or cell outlines) → `cosmx_spatial.load_s1()` → `data/S1_cosmx_earlymf_spatial/processed/cosmx_s1.h5ad` (Patient16 is read from inside `RAW.tar`). Only the 33–37 FOVs per slide (0.512 mm squares, ~9 mm² per slide) were imaged — F2a shows where.

**Units — read before any number below.**
- The GEO `patient` label is not a biopsy label: cells of different "patients" are interleaved at nearest-neighbour scale (§0 check). Unit = `piece` (spatially contiguous FOV group) nested in `slide`.
- Stage = slide (3/4 patch, 7/8 plaque): patch vs plaque is 2 vs 2 slides → described, never tested. Both are early (T1–T2) lesions, so the stage colours are not 28b's early/advanced ones.
- No clone markers (KIR3DL2, PLS3, CCR4 off panel; TOX at floor) and no TCR → T cells pooled; no malignant/reactive split.
- Every gene has a ~7–10% detection floor (ambient + segmentation); §2 gates genes against it.
- Lineages are re-called (§1): InSituType's "B.cell" is mostly low-count cells and its "Melanocyte" is PanCK/KRT14-high. Melanocyte markers are off the panel.

| fig | question |
|---|---|
| F0 | lineage markers per called lineage |
| F1 | which lineage carries each protease in situ |
| F2a | what was imaged: every FOV and cell on each slide, at true position |
| F2b | maps of every piece per slide (packed): lineage, MMP2, MMP9, T density |
| F2c | zoom: lineage, MMP9 and MMP12 counts per cell in a 300 µm window per slide |
| F3 | T cells and protease⁺ stroma vs depth from the epidermis |
| F4 | does stromal protease expression rise with local T density (beyond spillover) |
| F5 | neighbourhood enrichment: MMP-high vs MMP-low stroma next to T cells |
| F6 | 28b axes: fibroblast receptor next to ligand⁺ T cells |

**Stats.** Poisson GLM, log-library offset, piece fixed effects, SEs clustered on FOV; pooled over slides + per-slide fits. Robust = pooled BH q < 0.05 within the figure **and** ≥3/4 slides share the sign **and**, for genes T cells also carry, the effect exceeds the spillover reference (CD3E/CD2 read in the same cells with the same model).

Outputs: `figures/degradome/deg28d_*.png`, `tables/deg28d_*.csv`, per-cell calls in `data/S1_cosmx_earlymf_spatial/processed/cosmx_s1_cells_28d.parquet`.

```python  # cell 1
# ============================================================================
# §0  Setup + ingest
# ============================================================================
# 65k cells × 1000 genes: everything here is light. The first run builds the h5ad cache from the
# GEO CSVs (Patient16 is read from inside RAW.tar); later runs read the cache.
import warnings; warnings.filterwarnings("ignore")
import os
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"]:
    os.environ.setdefault(_v, "8")
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
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

import degradome_data as cfg          # panel families + output dirs
import degradome_helpers as D         # _finish
import degradome_cells as DC          # _stars
import cosmx_spatial as CS            # ingest, lineages, graphs, GLM

PREFIX = "deg28d_"
fig28d = lambda name: cfg.FIG_DIR / f"{PREFIX}{name}.png"
tab28d = lambda name: cfg.TAB_DIR / f"{PREFIX}{name}.csv"
FIG_INDEX = []


def logfig(name, what):
    FIG_INDEX.append({"figure": name, "file": fig28d(name).name, "what": what})


MIN_COUNTS = 20     # CosMx cells below this are mostly segmentation debris
R_NICHE = 50        # µm, F4 niche radius (sensitivity: 30, 100)
R_CONTACT = 30      # µm, F5/F6 neighbourhood
R_SMOOTH = 20       # µm, F2 display smoothing only
HI_COUNT = 2        # "MMP-high" = ≥2 counts; a single count is within reach of the ~10% floor
MIN_PIECE_CELLS = 20

adata = CS.load_s1(cache=True)
assert adata.n_obs == CS.N_CELLS_EXPECTED, adata.n_obs
print(adata)
print(adata.obs.groupby(["slide", "stage"], observed=True).agg(
    cells=("fov", "size"), fovs=("fov_key", "nunique"), pieces=("piece", "nunique")))
print("nCount_RNA quantiles:", adata.obs.nCount_RNA.quantile([.01, .05, .25, .5, .75]).round(0).to_dict())

MIX = CS.patient_mixing(adata)
print("\nGEO patient label vs space (per slide × FOV, median):")
print(MIX.groupby("slide")[["n_patients", "nn_same", "random"]].median().round(2))
print("nn_same ≈ random → the GEO patient label carries no spatial structure; it is not used below.")

adata = adata[adata.obs.nCount_RNA >= MIN_COUNTS].copy()
print(f"\nafter nCount_RNA ≥ {MIN_COUNTS}: {adata.n_obs:,} cells")
```

```
AnnData object with n_obs × n_vars = 65717 × 1000
    obs: 'fov', 'x_FOV_px', 'y_FOV_px', 'x_slide_mm', 'y_slide_mm', 'Area', 'AspectRatio', 'nCount_RNA', 'nFeature_RNA', 'Mean.PanCK', 'Mean.CD68', 'Mean.CD45', 'Mean.DAPI', 'nCount_negprobes', 'nCount_falsecode', 'propNegative', 'complexity', 'stage', 'patient_geo', 'insitutype', 'gsm', 'slide', 'fov_key', 'piece'
    uns: 's1_note'
    obsm: 'spatial'
    layers: 'counts'

              cells  fovs  pieces
slide stage                      
3     patch   11932    37      13
4     patch    6422    33       7
7     plaque  21035    36      10
8     plaque  26328    33      11
nCount_RNA quantiles: {0.01: 28.0, 0.05: 50.0, 0.25: 132.0, 0.5: 233.0, 0.75: 404.0}


GEO patient label vs space (per slide × FOV, median):
       n_patients  nn_same  random
slide                             
3             5.0     0.26    0.22
4             4.0     0.32    0.28
7             4.0     0.33    0.30
8             4.0     0.33    0.31
nn_same ≈ random → the GEO patient label carries no spatial structure; it is not used below.


after nCount_RNA ≥ 20: 65,717 cells
```

```python  # cell 2
# ============================================================================
# §1  Coarse lineages — Leiden on the panel, called by marker score, checked against the stains
# ============================================================================
# InSituType's labels are not used: its "B.cell" class is low-count cells with weak MS4A1/CD79A
# and its "Melanocyte" class is PanCK-high/KRT14-high. Melanocytes cannot be called at all here
# (PMEL, MLANA, TYRP1, DCT, MITF, SOX10 are off the panel) and fall into Keratinocyte/Unassigned.
adata.X = CS.lognorm(adata)
SCORES, USED = CS.lineage_scores(adata)
print("markers on panel:", USED)
_off = {k: sorted(set(CS.LINEAGE_MARKERS[k]) - set(v)) for k, v in USED.items()}
print("off panel:", {k: v for k, v in _off.items() if v})

sc.pp.pca(adata, n_comps=30)
sc.pp.neighbors(adata, n_neighbors=15)
sc.tl.leiden(adata, resolution=1.0, key_added="leiden", flavor="igraph", n_iterations=2, random_state=0)

CALLS = CS.call_clusters(adata, SCORES, "leiden")
MANUAL = {}         # {"<leiden id>": "<lineage>"}: overrides, set after reviewing the table below
CALLS["final"] = [MANUAL.get(c, v) for c, v in zip(CALLS.index, CALLS.call)]
with pd.option_context("display.width", 220, "display.max_columns", 30):
    print(CALLS[["n", "top", "margin", "call", "final", "PanCK_rel", "CD45_rel", "CD68_rel", "nCount",
                 "top_slide_frac", "flag_epi_immune", "insitutype_mode"]].round(2))
CALLS.to_csv(tab28d("cluster_calls"))
# Review before trusting the lineages:
#  - flag_epi_immune: keratinocyte by RNA but T-level CD45 → likely intraepidermal T cells with
#    keratinocyte spillover (epidermotropism). Kept as Keratinocyte unless overridden in MANUAL.
#  - top_slide_frac > 0.9: a cluster that is one slide is batch until shown otherwise.
print("\nflag_epi_immune:", CALLS.index[CALLS.flag_epi_immune].tolist(),
      "| single-slide clusters:", CALLS.index[CALLS.top_slide_frac > 0.9].tolist())

adata.obs["lineage"] = pd.Categorical(adata.obs.leiden.map(CALLS.final).astype(str),
                                      categories=[l for l in CS.LINEAGE_COLORS if l in set(CALLS.final)])
adata.obs["t_sub"] = CS.t_subset(adata)

print("\nlineage × InSituType:")
print(pd.crosstab(adata.obs.lineage, adata.obs.insitutype))
print("\nlineage × slide:")
print(pd.crosstab(adata.obs.lineage, adata.obs.slide))
print("\nT subsets (count-based, descriptive only):",
      adata.obs.t_sub[adata.obs.lineage == "T"].value_counts().to_dict())
```

```
markers on panel: {'Keratinocyte': ['KRT5', 'KRT14', 'KRT10', 'KRT1', 'KRT15'], 'Fibroblast': ['COL1A1', 'COL1A2', 'DCN', 'LUM', 'PDGFRA'], 'Endothelial': ['PECAM1', 'VWF', 'CDH5', 'LYVE1', 'PROX1'], 'Pericyte_SMC': ['ACTA2', 'RGS5', 'MYH11', 'TAGLN'], 'T': ['CD3E', 'CD3D', 'CD3G', 'CD2'], 'Myeloid': ['CD68', 'CD163', 'LYZ', 'C1QA', 'C1QB', 'CD14', 'HLA-DRA'], 'B': ['MS4A1', 'CD79A', 'CD19'], 'Plasma': ['MZB1', 'JCHAIN', 'IGKC', 'IGHG1'], 'Mast': ['TPSAB1/B2', 'CPA3', 'KIT']}
off panel: {}

       n           top  margin          call         final  PanCK_rel  CD45_rel  CD68_rel  nCount  top_slide_frac  flag_epi_immune   insitutype_mode
8   8884    Fibroblast    0.73    Fibroblast    Fibroblast       1.03      1.07      1.03   182.0            0.68            False        Fibroblast
2   6864  Keratinocyte    1.58  Keratinocyte  Keratinocyte      13.84      1.02      1.12   361.0            0.36            False      Keratinocyte
1   5199  Keratinocyte    0.94  Keratinocyte  Keratinocyte      11.56      1.89      1.25   394.0            0.48            False            T.cell
4   4528  Keratinocyte    1.50  Keratinocyte  Keratinocyte       9.58      1.19      1.19   489.0            0.51            False      Keratinocyte
0   4052  Keratinocyte    2.19  Keratinocyte  Keratinocyte       9.51      1.05      1.09   436.0            0.47            False      Keratinocyte
3   3140  Keratinocyte    1.27  Keratinocyte  Keratinocyte      56.19      1.14      1.34   586.0            0.45            False      Keratinocyte
9   9574       Myeloid    0.33       Myeloid       Myeloid       1.10      1.38      1.68   212.0            0.50            False     Macrophage.DC
7   8895             T    0.15             T             T       0.92      2.36      0.98   165.0            0.82            False            T.cell
12   387             T    0.15             T             T       0.78      2.45      0.52   398.0            0.93            False            T.cell
6   9025        Plasma    0.01    Unassigned    Unassigned       0.67      1.24      0.97   205.0            0.69            False            B.cell
5   3434    Fibroblast    0.04    Unassigned    Unassigned       0.79      0.89      0.87    53.0            0.63            False            B.cell
10  1248  Pericyte_SMC    0.01    Unassigned    Unassigned       1.22      0.85      1.13   205.0            0.54            False  Endothelial.cell
11   487        Plasma    1.36    Unassigned    Unassigned       0.99      0.99      1.27   189.0            0.67            False            B.cell

flag_epi_immune: [] | single-slide clusters: ['12']

lineage × InSituType:
insitutype    B.cell  Endothelial.cell  Fibroblast  Keratinocyte  \
lineage                                                            
Keratinocyte      44               352          13         18024   
Fibroblast       248                89        6335             0   
T                589               188         290             3   
Myeloid         1348               227         213            27   
Unassigned      7872              1736         573           105   

insitutype    Macrophage.DC  Melanocyte  Pericyte  T.cell  
lineage                                                    
Keratinocyte           1308        2020        18    2004  
Fibroblast              410          66      1167     569  
T                       822          90       243    7057  
Myeloid                7321          34       105     299  
Unassigned              792         290      1491    1335  

lineage × slide:
slide            3     4     7     8
lineage                             
Keratinocyte  9950  4616  6852  2365
Fibroblast     536   263  2078  6007
T              485   857   659  7281
Myeloid        710   554  3501  4809
Unassigned     251   132  7945  5866

T subsets (count-based, descriptive only): {'T_other': 6547, 'T_CD4': 1813, 'T_CD8': 922}
```

```python  # cell 3
# ============================================================================
# §U  Shared arrays + plot utilities
# ============================================================================
LIN = adata.obs.lineage.astype(str).to_numpy()
LIN_ORDER = [l for l in adata.obs.lineage.cat.categories if l != "Unassigned"]
SLIDES = list(adata.obs.slide.cat.categories)
SLIDE_STAGE = adata.obs.groupby("slide", observed=True).stage.first().astype(str).to_dict()
FOV = adata.obs.fov_key.astype(str).to_numpy()
PIECE = adata.obs.piece.astype(str).to_numpy()
SLIDE = adata.obs.slide.astype(str).to_numpy()
LIBOFF = np.log(adata.obs.nCount_RNA.to_numpy())
COUNTS = adata.layers["counts"].tocsc()
# Per-slide marker / line style, so the two slides of one stage stay distinguishable.
SLIDE_MARK = {s: m for s, m in zip(SLIDES, ["o", "s", "^", "D", "v", "P"])}
SLIDE_LS = {s: ["-", "--", ":"][[t for t in SLIDES if SLIDE_STAGE[t] == SLIDE_STAGE[s]].index(s) % 3]
            for s in SLIDES}


def gcounts(g):
    return np.asarray(COUNTS[:, adata.var_names.get_loc(g)].todense()).ravel()


def glog(g):
    return np.asarray(adata.X[:, adata.var_names.get_loc(g)].todense()).ravel()


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)


def dotplot(genes, rows, ax, gate=None, title=""):
    """rows × genes. Size = detection; colour = mean lognorm scaled to the gene's max.
    Genes failing the §2 gate are drawn grey."""
    sub = adata[adata.obs.lineage.isin(rows).to_numpy()]
    Xg = pd.DataFrame(sub[:, genes].X.toarray(), columns=genes)
    lab = sub.obs.lineage.astype(str).to_numpy()
    det = (Xg > 0).groupby(lab).mean().reindex(rows)
    mu = Xg.groupby(lab).mean().reindex(rows)
    mu_s = mu / mu.max().replace(0, 1)
    yy = np.arange(len(rows))
    for j, g in enumerate(genes):
        s = det[g].to_numpy() * 110 + 1
        if gate is None or bool(gate.loc[g, "pass"]):
            ax.scatter(np.full(len(rows), j), yy, s=s, c=mu_s[g].to_numpy(), cmap="Reds", vmin=0, vmax=1,
                       edgecolor="k", linewidth=0.2)
        else:
            ax.scatter(np.full(len(rows), j), yy, s=s, c="#d0d0d0", edgecolor="#9a9a9a", linewidth=0.2)
    for d in [0.1, 0.3, 0.6]:
        ax.scatter([], [], s=d * 110 + 1, c="w", edgecolor="k", linewidth=0.3, label=f"{d:.0%}")
    ax.legend(title="detected", fontsize=5.5, title_fontsize=6, frameon=False,
              loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, rotation=90, fontstyle="italic", fontsize=6.5)
    ax.set_yticks(yy); ax.set_yticklabels(rows, fontsize=7)
    ax.set_xlim(-0.6, len(genes) - 0.4); ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_title(title, fontsize=8.5); _clean(ax)
    return det, mu


def coef_panel(ax, df, labels, spill=None, title=""):
    """Pooled coefficient ± 95% CI (black) with per-slide fits as stage-coloured dots.
    `spill`: (lo, hi) of the spillover-reference point estimates, drawn as a grey band."""
    if spill is not None and np.isfinite(spill).all():
        ax.axhspan(*spill, color="#bdbdbd", alpha=0.35, lw=0, label="T-gene spillover")
    ax.axhline(0, color="k", lw=0.6)
    for j, lab in enumerate(labels):
        d = df[df.label == lab]
        if not (d.slide == "pooled").any():
            ax.text(j, 0, "n/a", ha="center", va="bottom", fontsize=6, color="#9a9a9a")
            continue
        p = d[d.slide == "pooled"].iloc[0]
        ax.errorbar(j, p.coef, yerr=1.96 * p.se, fmt="o", color="k", ms=3.5, lw=1, capsize=2, zorder=3)
        star = DC._stars(p.q) if "q" in d and np.isfinite(p.get("q", np.nan)) else ""
        if star:
            ax.text(j, p.coef + 1.96 * p.se, star, ha="center", va="bottom", fontsize=7)
        for k, s in enumerate(SLIDES):
            r = d[d.slide == s]
            if len(r) and np.isfinite(r.coef.iloc[0]):
                ax.scatter(j - 0.24 + 0.16 * k, r.coef.iloc[0], s=11, color=CS.STAGE_COLORS[SLIDE_STAGE[s]],
                           marker=SLIDE_MARK[s], edgecolor="none", zorder=2)
    for s in SLIDES:
        ax.scatter([], [], s=11, color=CS.STAGE_COLORS[SLIDE_STAGE[s]], marker=SLIDE_MARK[s],
                   label=f"slide {s} ({SLIDE_STAGE[s]})")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6.5)
    ax.set_ylabel("log rate ratio", fontsize=7); ax.tick_params(labelsize=6.5)
    ax.set_title(title, fontsize=8.5); _clean(ax)


def robust_flags(df, key):
    """Pooled BH q within the figure; robust = q < 0.05 and ≥3/4 slides share the pooled sign."""
    pooled = df[df.slide == "pooled"].copy()
    test = ~pooled.control
    pooled.loc[test, "q"] = multipletests(pooled.loc[test, "p"].fillna(1), method="fdr_bh")[1]
    per = df[df.slide != "pooled"].merge(pooled[key + ["coef"]].rename(columns={"coef": "pooled_coef"}), on=key)
    agree = (np.sign(per.coef) == np.sign(per.pooled_coef)).groupby([per[k] for k in key]).sum()
    pooled = pooled.merge(agree.rename("n_slides_agree").reset_index(), on=key, how="left")
    pooled["robust"] = (pooled.q < 0.05) & (pooled.n_slides_agree >= 3)
    return pooled
```

```python  # cell 4
# ============================================================================
# F0 — lineage markers  +  §2 gene gate
# ============================================================================
genes0 = [g for lin in LIN_ORDER for g in USED.get(lin, [])[:3]]
fig, ax = plt.subplots(figsize=(7, 3))
dotplot(genes0, LIN_ORDER, ax, title="Lineage markers by called lineage (CosMx)")
D._finish(fig, fig28d("F0_lineage_markers"), True)
logfig("F0_lineage_markers", "marker dotplot per called lineage")

# Gate: is the gene distinguishable from the ~7-10% floor in its top lineage? enrichment =
# top-lineage mean / median of lineage means (≥2) and top-lineage detection ≥ 0.15. Genes that
# fail are drawn grey and carry no claims below.
PROTEASE_FAMS = ["mmp", "cathepsin", "granzyme", "serine_other", "metallo_other", "inhibitor"]
PROT_GENES = [g for f in PROTEASE_FAMS for g in cfg.DEGRADOME_FAMILIES[f] if g in adata.var_names]
AXIS_GENES = [g for g in ["TNFSF14", "LTBR", "TNFRSF14", "CD40LG", "CD40", "TGFB1", "ITGB8",
                          "TGFBR1", "TGFBR2", "FN1", "GATA3", "IL4R", "CD44"] if g in adata.var_names]
CTRL_GENES = [g for g in ["CD3E", "CD2", "COL1A1", "DCN", "CD68", "KRT14"] if g in adata.var_names]
GATE, MEAN_L, DET_L = CS.gene_gate(adata, PROT_GENES + AXIS_GENES + CTRL_GENES)
GATE.to_csv(tab28d("gene_gate"))
print(f"protease/inhibitor genes of the 28 panel on CosMx: {len(PROT_GENES)}")
print(GATE[["top_lineage", "top_det", "floor_det", "enrichment", "pass"]].round(2)
      .sort_values("enrichment", ascending=False).to_string())
print("\nat floor (no claims):", GATE.index[~GATE["pass"]].tolist())
```

```
<Figure size 700x300 with 1 Axes>
protease/inhibitor genes of the 28 panel on CosMx: 25
           top_lineage  top_det  floor_det  enrichment   pass
KRT14     Keratinocyte     0.87       0.12       38.38   True
COL1A1      Fibroblast     0.89       0.20        7.97   True
DCN         Fibroblast     0.78       0.22        6.22   True
MMP2        Fibroblast     0.42       0.08        5.22   True
FN1         Fibroblast     0.27       0.07        3.59   True
CD3E                 T     0.37       0.14        3.31   True
MMP9           Myeloid     0.27       0.11        3.14   True
MMP12          Myeloid     0.10       0.03        2.84  False
SERPINA1       Myeloid     0.15       0.04        2.81  False
CD2                  T     0.29       0.14        2.57   True
CD68           Myeloid     0.26       0.11        2.44   True
CD40LG               T     0.12       0.06        2.22  False
MMP14       Fibroblast     0.23       0.08        2.15   True
FAP         Fibroblast     0.09       0.04        1.96  False
MMP19       Fibroblast     0.11       0.05        1.85  False
CD44                 T     0.55       0.40        1.69  False
CD40           Myeloid     0.14       0.06        1.67  False
TNFSF14              T     0.14       0.10        1.62  False
MMP1        Fibroblast     0.07       0.04        1.60  False
TIMP1       Fibroblast     0.22       0.14        1.59  False
GZMA                 T     0.06       0.05        1.52  False
CSTB      Keratinocyte     0.55       0.17        1.51  False
TGFB1                T     0.19       0.14        1.47  False
GATA3                T     0.18       0.09        1.45  False
SLPI      Keratinocyte     0.17       0.04        1.44  False
IL4R                 T     0.12       0.09        1.35  False
PRTN3       Fibroblast     0.09       0.06        1.35  False
DPP4                 T     0.06       0.05        1.35  False
CST7           Myeloid     0.16       0.09        1.28  False
MMP7        Fibroblast     0.06       0.04        1.26  False
CTSW                 T     0.10       0.10        1.26  False
SERPINA3    Fibroblast     0.06       0.04        1.25  False
TGFBR2               T     0.13       0.11        1.24  False
TNFRSF14             T     0.19       0.17        1.23  False
GZMK                 T     0.07       0.07        1.20  False
ELANE       Fibroblast     0.13       0.09        1.19  False
LTBR        Fibroblast     0.09       0.05        1.19  False
GZMH                 T     0.06       0.05        1.19  False
CTSG        Fibroblast     0.04       0.04        1.16  False
GZMB      Keratinocyte     0.11       0.05        1.16  False
ITGB8     Keratinocyte     0.12       0.05        1.10  False
CTSD      Keratinocyte     0.50       0.21        1.08  False
PRF1           Myeloid     0.09       0.07        1.08  False
TGFBR1               T     0.09       0.09        1.07  False

at floor (no claims): ['MMP1', 'MMP7', 'MMP12', 'MMP19', 'CTSD', 'CTSG', 'CTSW', 'GZMA', 'GZMB', 'GZMH', 'GZMK', 'PRF1', 'ELANE', 'PRTN3', 'FAP', 'DPP4', 'TIMP1', 'CST7', 'CSTB', 'SERPINA1', 'SERPINA3', 'SLPI', 'TNFSF14', 'LTBR', 'TNFRSF14', 'CD40LG', 'CD40', 'TGFB1', 'ITGB8', 'TGFBR1', 'TGFBR2', 'GATA3', 'IL4R', 'CD44']
```

```python  # cell 5
# ============================================================================
# F1 — protease and inhibitor genes by lineage (CosMx)
# ============================================================================
# In-situ counterpart of 28b F2a/F2b. Expected top lineage from 28b's scRNA reading of MF skin.
EXPECT_28B = {"MMP2": "Fibroblast", "MMP14": "Fibroblast", "FAP": "Fibroblast", "TIMP1": "Fibroblast",
              "MMP9": "Myeloid", "MMP12": "Myeloid", "SERPINA1": "Myeloid", "GZMB": "T"}
f1_genes = GATE.loc[PROT_GENES].sort_values(["pass", "enrichment"], ascending=[False, False]).index.tolist()
fig, ax = plt.subplots(figsize=(max(5, 0.26 * len(f1_genes) + 1.8), 3))
dotplot(f1_genes, LIN_ORDER, ax, gate=GATE, title="Protease and inhibitor genes by lineage (CosMx)")
D._finish(fig, fig28d("F1_protease_by_lineage"), True)
logfig("F1_protease_by_lineage", "protease/inhibitor dotplot per lineage; grey = at floor")

F1 = GATE.loc[f1_genes, ["top_lineage", "top_det", "enrichment", "pass"]].copy()
F1["expected_28b"] = F1.index.map(EXPECT_28B)
F1["agrees"] = np.where(F1.expected_28b.isna(), "", np.where(F1.top_lineage == F1.expected_28b, "yes", "no"))
F1.to_csv(tab28d("F1_top_lineage"))
print(F1.round(2).to_string())
_e = F1[F1.expected_28b.notna() & F1["pass"]]
print(f"\ngated genes with a 28b expectation: {(_e.agrees == 'yes').sum()}/{len(_e)} have the expected top lineage")
```

```
<Figure size 830x300 with 1 Axes>
           top_lineage  top_det  enrichment   pass expected_28b agrees
MMP2        Fibroblast     0.42        5.22   True   Fibroblast    yes
MMP9           Myeloid     0.27        3.14   True      Myeloid    yes
MMP14       Fibroblast     0.23        2.15   True   Fibroblast    yes
MMP12          Myeloid     0.10        2.84  False      Myeloid    yes
SERPINA1       Myeloid     0.15        2.81  False      Myeloid    yes
FAP         Fibroblast     0.09        1.96  False   Fibroblast    yes
MMP19       Fibroblast     0.11        1.85  False          NaN       
MMP1        Fibroblast     0.07        1.60  False          NaN       
TIMP1       Fibroblast     0.22        1.59  False   Fibroblast    yes
GZMA                 T     0.06        1.52  False          NaN       
CSTB      Keratinocyte     0.55        1.51  False          NaN       
SLPI      Keratinocyte     0.17        1.44  False          NaN       
PRTN3       Fibroblast     0.09        1.35  False          NaN       
DPP4                 T     0.06        1.35  False          NaN       
CST7           Myeloid     0.16        1.28  False          NaN       
CTSW                 T     0.10        1.26  False          NaN       
MMP7        Fibroblast     0.06        1.26  False          NaN       
SERPINA3    Fibroblast     0.06        1.25  False          NaN       
GZMK                 T     0.07        1.20  False          NaN       
ELANE       Fibroblast     0.13        1.19  False          NaN       
GZMH                 T     0.06        1.19  False          NaN       
GZMB      Keratinocyte     0.11        1.16  False            T     no
CTSG        Fibroblast     0.04        1.16  False          NaN       
PRF1           Myeloid     0.09        1.08  False          NaN       
CTSD      Keratinocyte     0.50        1.08  False          NaN       

gated genes with a 28b expectation: 3/3 have the expected top lineage
```

```python  # cell 6
# ============================================================================
# F2 — maps of everything that was imaged
# ============================================================================
# CosMx images only the fields of view (FOVs, 0.512 × 0.512 mm each) the operator placed on the
# section, not the whole slide. F2a: each slide at its true coordinates — every cell, every FOV
# outline, piece labels; the blank area was never imaged. F2b (one figure per slide): the same
# pieces packed side by side at true scale, gaps between them removed (so distances *between*
# pieces are not real), coloured by lineage, MMP2, MMP9 and T density. Labels match F2a.
# Graphs are per slide on slide-global µm, so a neighbourhood may span adjacent FOVs.
G_S = CS.neighbors(adata, R_SMOOTH)
G_C = CS.neighbors(adata, R_CONTACT)
G_N = CS.neighbors(adata, R_NICHE)
NB_C, TOT_C = CS.nbr_composition(G_C, LIN)
NB_N, TOT_N = CS.nbr_composition(G_N, LIN)
print(f"mean neighbours: {R_SMOOTH} µm {G_S.nnz / adata.n_obs:.1f} | {R_CONTACT} µm {G_C.nnz / adata.n_obs:.1f}"
      f" | {R_NICHE} µm {G_N.nnz / adata.n_obs:.1f}")

SM = {g: CS.smooth(G_S, glog(g)) for g in ["MMP2", "MMP9"]}
T_DENS = NB_N["T"].to_numpy() / (np.pi * (R_NICHE / 1000) ** 2)       # T cells per mm²
XY = adata.obsm["spatial"]
BOXES = CS.fov_boxes(adata)
PACK = CS.pack_pieces(BOXES)
_fov_mm2 = ((BOXES.x1 - BOXES.x0) * (BOXES.y1 - BOXES.y0) / 1e6)
print(f"µm/px {np.round(BOXES.attrs['um_per_px'], 4)} | FOV {np.sqrt(_fov_mm2.median()):.3f} mm square | "
      f"imaged area per slide (mm²): {_fov_mm2.groupby(BOXES.slide).sum().round(1).to_dict()}")
LIN_HANDLES = [plt.Line2D([], [], marker="o", ls="", ms=4, color=CS.LINEAGE_COLORS[l], label=l)
               for l in adata.obs.lineage.cat.categories]

# F2a — whole slides
fig, axes = plt.subplots(1, len(SLIDES), figsize=(3.1 * len(SLIDES), 4.3), squeeze=False)
lo, hi = BOXES[["x0", "y0"]].min().to_numpy() / 1000, BOXES[["x1", "y1"]].max().to_numpy() / 1000
for ax, s in zip(axes[0], SLIDES):
    idx = np.flatnonzero(SLIDE == s)
    bs = BOXES[BOXES.slide == s]
    ax.scatter(*(XY[idx] / 1000).T, s=0.25, c=[CS.LINEAGE_COLORS[l] for l in LIN[idx]], linewidths=0,
               rasterized=True)
    for b in bs.itertuples():
        ax.add_patch(plt.Rectangle((b.x0 / 1000, b.y0 / 1000), (b.x1 - b.x0) / 1000, (b.y1 - b.y0) / 1000,
                                   fill=False, lw=0.3, ec="#7f7f7f"))
    for p, g in bs.groupby("piece"):
        ax.text(g.x1.max() / 1000 + 0.12, g.y1.max() / 1000, p.split("_")[1], fontsize=5, va="top")
    ax.set_xlim(lo[0] - 0.3, hi[0] + 1.0); ax.set_ylim(lo[1] - 0.3, hi[1] + 0.3)
    ax.set_aspect("equal"); ax.set_xlabel("mm", fontsize=7); ax.tick_params(labelsize=6)
    ax.set_title(f"slide {s} ({SLIDE_STAGE[s]})\n{len(idx):,} cells · {len(bs)} FOVs · {bs.piece.nunique()} pieces",
                 fontsize=7.5)
    _clean(ax)
fig.legend(handles=LIN_HANDLES, loc="lower center", ncol=len(LIN_HANDLES), fontsize=6.5, frameon=False,
           bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Imaged fields of view per slide (grey squares), cells coloured by lineage", fontsize=9.5)
D._finish(fig, fig28d("F2a_slide_overview"), True)
logfig("F2a_slide_overview", "whole slides at true coordinates: every cell, FOV outlines, piece labels")

# F2b — all pieces of a slide, packed
cols = [("lineage", None), ("MMP2", SM["MMP2"]), ("MMP9", SM["MMP9"]), ("T cells / mm²", T_DENS)]
vmax = {name: np.percentile(v, 99) or 1 for name, v in cols if v is not None}
SHIFT = PACK.set_index("piece")[["dx", "dy"]]
XYP = XY + SHIFT.loc[PIECE].to_numpy()
for s in SLIDES:
    idx = np.flatnonzero(SLIDE == s)
    pk, bs = PACK[PACK.slide == s], BOXES[BOXES.slide == s]
    w, h = pk.px1.max() - pk.px0.min(), pk.py1.max() - pk.py0.min()
    pw = 4.6
    fig, axes = plt.subplots(2, 2, figsize=(2 * pw + 1.4, 2 * pw * h / w + 1.2))
    for ax, (name, v) in zip(axes.ravel(), cols):
        if v is None:
            ax.scatter(*XYP[idx].T, s=0.8, c=[CS.LINEAGE_COLORS[l] for l in LIN[idx]], linewidths=0, rasterized=True)
            x0, y0 = pk.px0.min(), pk.py0.min()
            ax.plot([x0, x0 + 500], [y0 - 60, y0 - 60], color="k", lw=1.5)
            ax.text(x0 + 250, y0 - 90, "500 µm", ha="center", va="top", fontsize=6)
        else:
            o = idx[np.argsort(v[idx])]
            im = ax.scatter(*XYP[o].T, s=0.8, c=v[o], cmap="magma_r" if name != "T cells / mm²" else "Blues",
                            vmin=0, vmax=vmax[name], linewidths=0, rasterized=True)
            plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01).ax.tick_params(labelsize=5.5)
        for b in bs.itertuples():
            dx, dy = SHIFT.loc[b.piece]
            ax.add_patch(plt.Rectangle((b.x0 + dx, b.y0 + dy), b.x1 - b.x0, b.y1 - b.y0,
                                       fill=False, lw=0.25, ec="#b0b0b0"))
        for b in pk.itertuples():
            ax.text(b.px0, b.py1 + 40, b.piece.split("_")[1], fontsize=5.5, va="bottom")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(name if name in ("lineage", "T cells / mm²") else f"$\\it{{{name}}}$ (lognorm, {R_SMOOTH} µm mean)",
                     fontsize=8)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.legend(handles=LIN_HANDLES, loc="lower center", ncol=len(LIN_HANDLES), fontsize=6.5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Slide {s} ({SLIDE_STAGE[s]}): all {len(pk)} pieces at true scale, packed "
                 f"(grey = FOV outlines)", fontsize=9.5)
    D._finish(fig, fig28d(f"F2b_maps_slide{s}"), True)
    logfig(f"F2b_maps_slide{s}", f"slide {s}: every piece packed at true scale; lineage / MMP2 / MMP9 / T density")
```

```
mean neighbours: 20 µm 12.3 | 30 µm 27.1 | 50 µm 69.8
µm/px [ 0.12 -0.12] | FOV 0.511 mm square | imaged area per slide (mm²): {'3': 9.7, '4': 8.6, '7': 9.4, '8': 8.6}

<Figure size 1240x430 with 4 Axes>
<Figure size 1060x1195.05 with 7 Axes>
<Figure size 1060x1300.55 with 7 Axes>
<Figure size 1060x1108.27 with 7 Axes>
<Figure size 1060x1186.93 with 7 Axes>
```

```python  # cell 7
# ============================================================================
# F2c — zoom: lineage, MMP9 and MMP12 per cell in small windows
# ============================================================================
# One ZOOM_UM-square window per slide, centred where myeloid cells carry the most MMP9 + MMP12
# counts. Chosen for maximum signal, so illustrative, not representative. Each cell is drawn as a
# disc of its segmented area, so at this scale the map reads like tissue. MMP panels show raw
# counts per cell; myeloid cells are outlined, because single counts in other lineages are the
# ~10% floor. To look elsewhere: ZOOM_AT = [(slide, x_um, y_um), ...] in slide µm (F2a mm × 1000).
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.spatial import cKDTree

ZOOM_UM = 300
ZOOM_AT = None

MYE = LIN == "Myeloid"
M9, M12 = gcounts("MMP9"), gcounts("MMP12")
if ZOOM_AT is None:
    ZOOM_AT = []
    for s in SLIDES:
        cand = np.flatnonzero((SLIDE == s) & MYE)
        w = (M9 + M12)[cand]
        nb = cKDTree(XY[cand]).query_ball_point(XY[cand], r=ZOOM_UM / 2, p=np.inf)   # square window
        ZOOM_AT.append((s, *XY[cand[np.argmax([w[j].sum() for j in nb])]]))

AREA_UM2 = adata.obs.Area.to_numpy() * abs(np.prod(BOXES.attrs["um_per_px"]))
ZCMAP = ListedColormap(["#e6e6e6", "#fdd49e", "#fc8d59", "#d7301f", "#7f0000"])
ZNORM = BoundaryNorm(np.arange(-0.5, 5), ZCMAP.N)
PANEL_IN = 2.6

fig, axes = plt.subplots(len(ZOOM_AT), 3, figsize=(3 * PANEL_IN + 1.4, len(ZOOM_AT) * PANEL_IN + 0.9),
                         squeeze=False)
rows = []
for i, (s, cx, cy) in enumerate(ZOOM_AT):
    half = ZOOM_UM / 2
    idx = np.flatnonzero((SLIDE == s) & (np.abs(XY[:, 0] - cx) <= half) & (np.abs(XY[:, 1] - cy) <= half))
    size = (np.sqrt(4 * AREA_UM2[idx] / np.pi) / ZOOM_UM * PANEL_IN * 72) ** 2      # pt², true cell size
    piece = pd.Series(PIECE[idx]).mode()[0]
    for j, (name, v) in enumerate([("lineage", None), ("MMP9", M9), ("MMP12", M12)]):
        ax = axes[i, j]
        if v is None:
            ax.scatter(*XY[idx].T, s=size, c=[CS.LINEAGE_COLORS[l] for l in LIN[idx]], linewidths=0)
            ax.plot([cx - half + 15, cx - half + 65], [cy - half + 12] * 2, color="k", lw=1.5)
            ax.text(cx - half + 40, cy - half + 18, "50 µm", ha="center", va="bottom", fontsize=5.5)
        else:
            o = idx[np.argsort(v[idx], kind="stable")]
            so = size[np.argsort(v[idx], kind="stable")]
            im = ax.scatter(*XY[o].T, s=so, c=np.minimum(v[o], 4), cmap=ZCMAP, norm=ZNORM,
                            edgecolors=np.where(MYE[o], "k", "none"), linewidths=0.35)
        ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_title(name if v is None else f"$\\it{{{name}}}$ counts per cell", fontsize=8)
        if j == 0:
            ax.set_ylabel(f"slide {s} ({SLIDE_STAGE[s]}) · {piece}\nx {cx / 1000:.2f}, y {cy / 1000:.2f} mm",
                          fontsize=6.5)
    my = MYE[idx]
    rows.append({"slide": s, "stage": SLIDE_STAGE[s], "piece": piece, "x_um": round(cx), "y_um": round(cy),
                 "n_cells": len(idx), "n_myeloid": int(my.sum()), "n_T": int((LIN[idx] == "T").sum()),
                 "n_fibro": int((LIN[idx] == "Fibroblast").sum()), "n_kerat": int((LIN[idx] == "Keratinocyte").sum()),
                 "MMP9+_myeloid": (M9[idx][my] > 0).mean(), "MMP9+_other": (M9[idx][~my] > 0).mean(),
                 "MMP12+_myeloid": (M12[idx][my] > 0).mean(), "MMP12+_other": (M12[idx][~my] > 0).mean()})
cb = fig.colorbar(im, ax=axes[:, 1:], fraction=0.025, pad=0.02, ticks=range(5))
cb.ax.set_yticklabels(["0", "1", "2", "3", "4+"]); cb.ax.tick_params(labelsize=6)
cb.set_label("counts per cell (outlined = myeloid)", fontsize=6.5)
fig.legend(handles=LIN_HANDLES, loc="lower center", ncol=len(LIN_HANDLES), fontsize=6.5, frameon=False,
           bbox_to_anchor=(0.45, -0.01))
fig.suptitle(f"Lineage, MMP9 and MMP12 per cell in {ZOOM_UM} µm windows", fontsize=9.5)
D._finish(fig, fig28d("F2c_zoom_mmp9_mmp12"), True)
logfig("F2c_zoom_mmp9_mmp12", f"{ZOOM_UM} µm windows (max myeloid MMP9+MMP12 per slide): lineage / MMP9 / MMP12 counts")

F2C = pd.DataFrame(rows)
F2C.to_csv(tab28d("F2c_zoom_windows"), index=False)
print(F2C.round(2).to_string(index=False))
```

```
<Figure size 920x1130 with 13 Axes>
slide  stage piece  x_um  y_um  n_cells  n_myeloid  n_T  n_fibro  n_kerat  MMP9+_myeloid  MMP9+_other  MMP12+_myeloid  MMP12+_other
    3  patch 3_p07 12592  6149      381         35   73        6      263           0.31         0.16            0.74          0.31
    4  patch 4_p03  4777  1661      378         23  127        2      206           0.43         0.08            0.35          0.05
    7 plaque 7_p01 13484 16627      533         98   15       93      209           0.60         0.20            0.16          0.13
    8 plaque 8_p06 13260 13086      642        285   13      101       58           0.41         0.13            0.07          0.04
```

```python  # cell 8
# ============================================================================
# F3 — T cells and protease-positive stroma vs depth from the epidermis
# ============================================================================
# Depth = µm to the nearest keratinocyte in the same piece; NaN where the piece holds < 50
# keratinocytes (no epidermis captured). Keratinocytes themselves are excluded from every panel.
DEPTH = CS.dist_to_lineage(adata, "Keratinocyte")
adata.obs["depth_um"] = DEPTH
HAS_EPI = ~np.isnan(DEPTH)
print(f"cells with a depth: {HAS_EPI.mean():.0%} | pieces with epidermis: "
      f"{pd.unique(PIECE[HAS_EPI]).size}/{pd.unique(PIECE).size}")

BINS = [0, 50, 100, 200, 400, np.inf]
BIN_LAB = ["0–50", "50–100", "100–200", "200–400", ">400"]
DBIN = pd.cut(DEPTH, BINS, labels=BIN_LAB, right=False)
NONKER = LIN != "Keratinocyte"
PANELS = [("T cells\n(fraction of non-keratinocytes)", NONKER, LIN == "T", None),
          ("MMP2+ fibroblasts\n(fraction of fibroblasts)", LIN == "Fibroblast", gcounts("MMP2") >= 1, "MMP2"),
          ("MMP9+ myeloid\n(fraction of myeloid)", LIN == "Myeloid", gcounts("MMP9") >= 1, "MMP9"),
          ("MMP12+ myeloid\n(fraction of myeloid)", LIN == "Myeloid", gcounts("MMP12") >= 1, "MMP12")]

rows = []
for s in SLIDES:
    ins = (SLIDE == s) & HAS_EPI & NONKER
    for name, pop, hit, _ in PANELS:
        m = ins & pop
        g = pd.DataFrame({"bin": DBIN[m], "hit": hit[m]}).groupby("bin", observed=False).hit.agg(["mean", "size"])
        g.loc[g["size"] < 30, "mean"] = np.nan
        rows += [{"slide": s, "stage": SLIDE_STAGE[s], "panel": name.split("\n")[0], "bin": b,
                  "frac": r["mean"], "n": int(r["size"])} for b, r in g.iterrows()]
F3 = pd.DataFrame(rows)
F3.to_csv(tab28d("F3_depth_profiles"), index=False)

fig, axes = plt.subplots(1, 4, figsize=(11, 2.8))
for ax, (name, _, _, floor_gene) in zip(axes, PANELS):
    for s in SLIDES:
        d = F3[(F3.panel == name.split("\n")[0]) & (F3.slide == s)]
        ax.plot(range(len(BIN_LAB)), d.frac.to_numpy(), ls=SLIDE_LS[s], marker=SLIDE_MARK[s], ms=3, lw=1,
                color=CS.STAGE_COLORS[SLIDE_STAGE[s]], label=f"slide {s} ({SLIDE_STAGE[s]})")
    if floor_gene:
        ax.axhline(DET_L[floor_gene].median(), color="#9a9a9a", ls="--", lw=0.8, label="median detection\nacross lineages")
    ax.set_xticks(range(len(BIN_LAB))); ax.set_xticklabels(BIN_LAB, rotation=45, fontsize=6.5)
    ax.set_xlabel("µm from epidermis", fontsize=7); ax.set_title(name, fontsize=7.5)
    ax.tick_params(labelsize=6.5); ax.set_ylim(bottom=0); _clean(ax)
axes[-1].legend(fontsize=5.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
fig.suptitle("T cells and protease-positive stroma by depth from the epidermis", fontsize=9.5, y=1.04)
D._finish(fig, fig28d("F3_depth"), True)
logfig("F3_depth", "fraction T / MMP2+ fibro / MMP9+ and MMP12+ myeloid per depth bin, per slide")

print(F3.pivot_table(index=["panel", "slide"], columns="bin", values="frac").reindex(columns=BIN_LAB).round(3).to_string())
```

```
cells with a depth: 92% | pieces with epidermis: 37/41

<Figure size 1100x280 with 4 Axes>
bin                       0–50  50–100  100–200  200–400   >400
panel             slide                                        
MMP12+ myeloid    3      0.160     NaN      NaN      NaN    NaN
                  4      0.182   0.135      NaN      NaN    NaN
                  7      0.175   0.111    0.097    0.121    NaN
                  8      0.103   0.055    0.052    0.041  0.034
MMP2+ fibroblasts 3      0.493     NaN      NaN      NaN    NaN
                  4      0.529     NaN      NaN      NaN    NaN
                  7      0.445   0.439    0.405    0.329    NaN
                  8      0.441   0.382    0.372    0.349  0.377
MMP9+ myeloid     3      0.217     NaN      NaN      NaN    NaN
                  4      0.264   0.486      NaN      NaN    NaN
                  7      0.324   0.338    0.339    0.330    NaN
                  8      0.181   0.208    0.262    0.258  0.328
T cells           3      0.237   0.432      NaN    0.463    NaN
                  4      0.481   0.200      NaN      NaN    NaN
                  7      0.038   0.032    0.061    0.084  0.058
                  8      0.273   0.319    0.340    0.367  0.431
```

```python  # cell 9
# ============================================================================
# F4 — stromal protease expression vs local T-cell density
# ============================================================================
# Per target cell: count ~ log1p(T within R) + log1p(all cells within R) + C(piece),
#                  offset log(nCount_RNA); Poisson, SEs clustered on FOV.
# log_all separates "near T cells" from "in dense tissue". Pieces with < MIN_PIECE_CELLS target
# cells are dropped. Depth is not in the main model: most plaque FOVs hold no epidermis (§F3), so
# requiring it would keep a small minority of stromal cells. The depth-adjusted fit on the
# subset that has a depth is in the sensitivity table: if near-T and near-epidermis were the same
# thing, the coefficient would collapse there.
#
# Spillover: a fibroblast touching T cells picks up T-cell transcripts through segmentation.
# CD3E/CD2 fitted in the same cells with the same model measure how large that artefact is. It
# can only inflate genes T cells themselves carry, so the band is binding only where `t_leak`
# (T-cell detection ≥ half the target lineage's) is set.
F4_SETS = {"Fibroblast": ["MMP2", "MMP14", "TIMP1", "COL1A1"],
           "Myeloid": ["MMP9", "MMP12", "MMP14", "TIMP1"]}
SPILL = [g for g in ["CD3E", "CD2"] if g in adata.var_names]


def _targets(lin, need_depth=True):
    m = (LIN == lin) & (HAS_EPI if need_depth else True)
    idx = np.flatnonzero(m)
    ok = pd.Series(PIECE[idx]).map(pd.Series(PIECE[idx]).value_counts()) >= MIN_PIECE_CELLS
    return idx[ok.to_numpy()]


def niche_fits(lin, genes, nb, tot, use_depth=False, tag=""):
    idx0 = _targets(lin, use_depth)
    X = pd.DataFrame({"log_T": np.log1p(nb["T"].to_numpy()), "log_all": np.log1p(tot.to_numpy()),
                      "log_depth": np.log(np.nan_to_num(DEPTH, nan=0.0) + 10)})
    cov = ["log_T", "log_all"] + (["log_depth"] if use_depth else [])
    rows = []
    for g in [g for g in genes if g in adata.var_names] + SPILL:
        y = gcounts(g)
        for s in ["pooled"] + SLIDES:
            idx = idx0 if s == "pooled" else idx0[SLIDE[idx0] == s]
            if len(idx) < 50:
                continue
            r = CS.niche_glm(y[idx], X.iloc[idx][cov].reset_index(drop=True), LIBOFF[idx], FOV[idx], fe=PIECE[idx])
            rows.append({"lineage": lin, "gene": g, "slide": s, "control": g in SPILL, "tag": tag, **r})
    return pd.DataFrame(rows)


F4_RAW = pd.concat([niche_fits(l, gs, NB_N, TOT_N, tag=f"{R_NICHE}um") for l, gs in F4_SETS.items()],
                   ignore_index=True)
F4_RAW["label"] = F4_RAW.lineage.str[:5] + " " + F4_RAW.gene
F4 = robust_flags(F4_RAW, ["lineage", "gene"])
F4["t_leak"] = [DET_L.loc["T", g] >= 0.5 * DET_L.loc[l, g] for l, g in zip(F4.lineage, F4.gene)]
spill_hi = F4[F4.control].groupby("lineage").coef.max()
F4["beyond_spill"] = [(not lk) or (c < 0) or (c - 1.96 * se > spill_hi.get(l, np.nan))
                      for lk, c, se, l in zip(F4.t_leak, F4.coef, F4.se, F4.lineage)]
F4["robust"] = F4.robust & F4.beyond_spill & ~F4.control
F4.to_csv(tab28d("F4_niche_glm"), index=False)
F4_RAW.to_csv(tab28d("F4_niche_glm_per_slide"), index=False)

fig, axes = plt.subplots(1, 2, figsize=(8, 3), gridspec_kw={"width_ratios": [1, 1]})
for ax, lin in zip(axes, F4_SETS):
    d = pd.concat([F4_RAW[(F4_RAW.lineage == lin) & (F4_RAW.slide != "pooled")], F4[F4.lineage == lin]])
    sp_ = F4[(F4.lineage == lin) & F4.control]
    band = (sp_.coef.min(), sp_.coef.max()) if len(sp_) else None
    labs = [f"{lin[:5]} {g}" for g in F4_SETS[lin] + SPILL if g in adata.var_names]
    coef_panel(ax, d, labs, spill=band, title=f"{lin}: gene vs T cells within {R_NICHE} µm")
axes[-1].legend(fontsize=5.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
D._finish(fig, fig28d("F4_niche_T_density"), True)
logfig("F4_niche_T_density", "Poisson GLM coefficient of log1p(T within 50 µm), pooled + per slide, spillover band")

with pd.option_context("display.width", 220):
    print(F4[["lineage", "gene", "control", "n", "n_pos", "coef", "se", "q", "n_slides_agree",
              "t_leak", "beyond_spill", "robust"]].round(3).to_string(index=False))

# Sensitivity: radius 30 / 100 µm, and + log(depth + 10) on the cells that have a depth.
SENS = [F4_RAW]
for r in [30, 100]:
    nb_r, tot_r = CS.nbr_composition(CS.neighbors(adata, r), LIN)
    SENS += [niche_fits(l, gs, nb_r, tot_r, tag=f"{r}um") for l, gs in F4_SETS.items()]
SENS += [niche_fits(l, gs, NB_N, TOT_N, use_depth=True, tag=f"{R_NICHE}um_depth") for l, gs in F4_SETS.items()]
F4_SENS = pd.concat(SENS, ignore_index=True)
F4_SENS.to_csv(tab28d("F4_sensitivity"), index=False)
print("\npooled coefficient by model (n in the depth model is the epidermis subset):")
print(F4_SENS[F4_SENS.slide == "pooled"].pivot_table(index=["lineage", "gene"], columns="tag",
                                                     values=["coef", "n"]).round(3).to_string())
```

```
<Figure size 800x300 with 2 Axes>
   lineage   gene  control    n  n_pos   coef    se     q  n_slides_agree  t_leak  beyond_spill  robust
Fibroblast   MMP2    False 8844   3733 -0.113 0.059 0.087               2   False          True   False
Fibroblast  MMP14    False 8844   2048 -0.079 0.037 0.068               3   False          True   False
Fibroblast  TIMP1    False 8844   1971  0.181 0.067 0.019               3    True         False   False
Fibroblast COL1A1    False 8844   7824 -0.165 0.025 0.000               4   False          True    True
Fibroblast   CD3E     True 8844   1334  0.238 0.055   NaN               4    True         False   False
Fibroblast    CD2     True 8844   1220  0.095 0.040   NaN               4    True         False   False
   Myeloid   MMP9    False 9539   2559  0.001 0.054 0.987               1   False          True   False
   Myeloid  MMP12    False 9539    996  0.016 0.092 0.980               2   False          True   False
   Myeloid  MMP14    False 9539   1467 -0.148 0.037 0.000               4    True          True    True
   Myeloid  TIMP1    False 9539   1465  0.035 0.051 0.653               3    True         False   False
   Myeloid   CD3E     True 9539   1518  0.131 0.044   NaN               4    True         False   False
   Myeloid    CD2     True 9539   1616  0.111 0.039   NaN               3    True         False   False


pooled coefficient by model (n in the depth model is the epidermis subset):
                    coef                                n                           
tag                100um   30um   50um 50um_depth   100um    30um    50um 50um_depth
lineage    gene                                                                     
Fibroblast CD2     0.150  0.146  0.095      0.081  8844.0  8844.0  8844.0     7352.0
           CD3E    0.160  0.208  0.238      0.219  8844.0  8844.0  8844.0     7352.0
           COL1A1 -0.160 -0.189 -0.165     -0.088  8844.0  8844.0  8844.0     7352.0
           MMP14  -0.068 -0.063 -0.079      0.004  8844.0  8844.0  8844.0     7352.0
           MMP2   -0.116 -0.094 -0.113     -0.054  8844.0  8844.0  8844.0     7352.0
           TIMP1   0.177  0.236  0.181      0.129  8844.0  8844.0  8844.0     7352.0
Myeloid    CD2     0.093  0.124  0.111      0.129  9539.0  9539.0  9539.0     8778.0
           CD3E    0.150  0.161  0.131      0.118  9539.0  9539.0  9539.0     8778.0
           MMP12   0.045  0.053  0.016      0.215  9539.0  9539.0  9539.0     8778.0
           MMP14  -0.102 -0.139 -0.148     -0.132  9539.0  9539.0  9539.0     8778.0
           MMP9    0.029 -0.005  0.001     -0.057  9539.0  9539.0  9539.0     8778.0
           TIMP1   0.035  0.026  0.035      0.020  9539.0  9539.0  9539.0     8778.0
```

```python  # cell 10
# ============================================================================
# F5 — neighbourhood enrichment: MMP-high vs MMP-low stroma next to T cells
# ============================================================================
# Edges within R_CONTACT between labels vs label permutations within each piece (composition
# kept, arrangement shuffled). Shown as log2(observed / expected); z is larger for bigger groups,
# so hi vs lo is compared on the ratio, not on z.
LAB5 = LIN.astype(object).copy()
_fib, _mye = LIN == "Fibroblast", LIN == "Myeloid"
LAB5[_fib] = np.where(gcounts("MMP2")[_fib] >= HI_COUNT, "Fibro MMP2-hi", "Fibro MMP2-lo")
_m912 = gcounts("MMP9") + gcounts("MMP12")
LAB5[_mye] = np.where(_m912[_mye] >= HI_COUNT, "Myeloid MMP9/12-hi", "Myeloid MMP9/12-lo")
_keep5 = LAB5 != "Unassigned"
ORDER5 = ["Keratinocyte", "Fibro MMP2-hi", "Fibro MMP2-lo", "Myeloid MMP9/12-hi", "Myeloid MMP9/12-lo",
          "T", "B", "Plasma", "Mast", "Endothelial", "Pericyte_SMC"]
ORDER5 = [l for l in ORDER5 if l in set(LAB5[_keep5])]
print(pd.Series(LAB5).value_counts().to_dict())


def _nhood(mask, n_perm):
    idx = np.flatnonzero(mask & _keep5)
    z, o, e = CS.nhood_enrichment(G_C[idx][:, idx], LAB5[idx], PIECE[idx], n_perm=n_perm)
    return z.loc[ORDER5, ORDER5], np.log2((o + 1) / (e + 1)).loc[ORDER5, ORDER5]


Z5, L5 = _nhood(np.ones(adata.n_obs, bool), 500)
PER5 = {s: _nhood(SLIDE == s, 200) for s in SLIDES}
L5.to_csv(tab28d("F5_log2_obs_exp")); Z5.to_csv(tab28d("F5_z"))

fig, ax = plt.subplots(figsize=(5.2, 4.2))
lim = max(0.5, np.nanpercentile(np.abs(L5.to_numpy()), 90))       # clipped: rare pairs reach ±4
im = ax.imshow(L5.to_numpy().clip(-lim, lim), cmap="RdBu_r", vmin=-lim, vmax=lim)
for i in range(len(ORDER5)):
    for j in range(len(ORDER5)):
        if abs(Z5.iat[i, j]) > 3:
            ax.text(j, i, "·", ha="center", va="center", fontsize=8, color="k")
ax.set_xticks(range(len(ORDER5))); ax.set_xticklabels(ORDER5, rotation=90, fontsize=6.5)
ax.set_yticks(range(len(ORDER5))); ax.set_yticklabels(ORDER5, fontsize=6.5)
plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="log2(observed / expected)").ax.tick_params(labelsize=6)
ax.set_title(f"Neighbourhood enrichment within {R_CONTACT} µm (· |z| > 3)", fontsize=8.5)
D._finish(fig, fig28d("F5_nhood_enrichment"), True)
logfig("F5_nhood_enrichment", "log2 obs/exp contact frequency, labels permuted within pieces")

rows = []
for s, (z, l) in [("pooled", (Z5, L5))] + [(s, PER5[s]) for s in SLIDES]:
    for hi, lo in [("Fibro MMP2-hi", "Fibro MMP2-lo"), ("Myeloid MMP9/12-hi", "Myeloid MMP9/12-lo")]:
        if {hi, lo, "T"} <= set(l.index):
            rows.append({"slide": s, "stage": SLIDE_STAGE.get(s, ""), "contrast": f"{hi} vs lo, next to T",
                         "log2_hi": l.loc[hi, "T"], "log2_lo": l.loc[lo, "T"],
                         "delta": l.loc[hi, "T"] - l.loc[lo, "T"]})
F5 = pd.DataFrame(rows)
F5.to_csv(tab28d("F5_hi_vs_lo_contrast"), index=False)
print(F5.round(3).to_string(index=False))
for c, d in F5[F5.slide != "pooled"].groupby("contrast"):
    print(f"{c}: hi > lo in {(d.delta > 0).sum()}/{len(d)} slides")
```

```
{'Keratinocyte': 23783, 'Unassigned': 14194, 'T': 9282, 'Myeloid MMP9/12-lo': 7879, 'Fibro MMP2-lo': 7113, 'Fibro MMP2-hi': 1771, 'Myeloid MMP9/12-hi': 1695}

<Figure size 520x420 with 2 Axes>
 slide  stage                            contrast  log2_hi  log2_lo  delta
pooled             Fibro MMP2-hi vs lo, next to T   -0.396   -0.193 -0.204
pooled        Myeloid MMP9/12-hi vs lo, next to T    0.346    0.093  0.254
     3  patch      Fibro MMP2-hi vs lo, next to T    1.240    0.757  0.483
     3  patch Myeloid MMP9/12-hi vs lo, next to T    1.022    0.900  0.122
     4  patch      Fibro MMP2-hi vs lo, next to T    0.590    0.820 -0.231
     4  patch Myeloid MMP9/12-hi vs lo, next to T    0.529    0.886 -0.357
     7 plaque      Fibro MMP2-hi vs lo, next to T    0.451    0.118  0.333
     7 plaque Myeloid MMP9/12-hi vs lo, next to T    0.263    0.239  0.024
     8 plaque      Fibro MMP2-hi vs lo, next to T   -0.529   -0.229 -0.300
     8 plaque Myeloid MMP9/12-hi vs lo, next to T    0.301    0.025  0.276
Fibro MMP2-hi vs lo, next to T: hi > lo in 2/4 slides
Myeloid MMP9/12-hi vs lo, next to T: hi > lo in 3/4 slides
```

```python  # cell 11
# ============================================================================
# F6 — 28b axes in proximity: fibroblast receptor next to ligand⁺ T cells
# ============================================================================
# Per fibroblast: receptor ~ [any ligand⁺ T cell within R_CONTACT] + log1p(T within R_CONTACT)
#                           + log1p(all within R_CONTACT) + C(piece),
# offset log(nCount_RNA), SEs clustered on FOV. The first term is the ligand-specific effect over
# and above T proximity. Ligand⁺ = ≥1 count in a T cell; with ligand genes near the floor in T
# cells (see §2) many ligand⁺ calls are ambient, which biases toward zero.
#
# Spillover reference: CD3E fitted with the same design. Ligand⁺ T cells are the higher-count T
# cells, so they leak more of everything; CD3E's coefficient on the same term measures that.
PAIRS = [("TNFSF14", "LTBR"), ("TNFSF14", "TNFRSF14"), ("CD40LG", "CD40"), ("TGFB1", "TGFBR2"),
         ("ITGB8", "TGFBR2"), ("TGFB1", "FN1"), ("GATA3", "IL4R")]
PAIRS = [(l, r) for l, r in PAIRS if l in adata.var_names and r in adata.var_names]
print("pairs on panel:", PAIRS)
IS_T = LIN == "T"
F6_IDX = _targets("Fibroblast", need_depth=False)

rows = []
for lig, rec in PAIRS:
    n_lig = G_C @ (IS_T & (gcounts(lig) >= 1)).astype(np.float32)
    X = pd.DataFrame({"ligT_near": (n_lig > 0).astype(float), "log_T": np.log1p(NB_C["T"].to_numpy()),
                      "log_all": np.log1p(TOT_C.to_numpy())})
    for g, ctrl in [(rec, False)] + [(c, True) for c in SPILL[:1]]:
        y = gcounts(g)
        for s in ["pooled"] + SLIDES:
            idx = F6_IDX if s == "pooled" else F6_IDX[SLIDE[F6_IDX] == s]
            if len(idx) < 50:
                continue
            r = CS.niche_glm(y[idx], X.iloc[idx].reset_index(drop=True), LIBOFF[idx], FOV[idx], fe=PIECE[idx])
            rows.append({"pair": f"{lig}→{rec}", "ligand": lig, "gene": g, "slide": s, "control": ctrl,
                         "frac_fibro_ligT_near": X.ligT_near.to_numpy()[idx].mean(), **r})
F6_RAW = pd.DataFrame(rows)
F6_RAW["label"] = np.where(F6_RAW.control, F6_RAW.pair + " [" + F6_RAW.gene + "]", F6_RAW.pair)
F6 = robust_flags(F6_RAW, ["pair", "gene"])
F6["ligand_gate"] = F6.ligand.map(lambda l: bool(GATE.loc[l, "pass"]) and GATE.loc[l, "top_lineage"] == "T"
                                  if l in GATE.index else False)
F6["t_leak"] = [DET_L.loc["T", g] >= 0.5 * DET_L.loc["Fibroblast", g] for g in F6.gene]
_sp = F6[F6.control].set_index("pair")
F6["beyond_spill"] = [(not lk) or (c < 0) or (c - 1.96 * se > _sp.coef.get(p, np.nan))
                      for lk, c, se, p in zip(F6.t_leak, F6.coef, F6.se, F6.pair)]
F6["robust"] = F6.robust & F6.beyond_spill & ~F6.control
F6.to_csv(tab28d("F6_axes_proximity"), index=False)
F6_RAW.to_csv(tab28d("F6_axes_proximity_per_slide"), index=False)

fig, ax = plt.subplots(figsize=(5.5, 3))
d = pd.concat([F6_RAW[(F6_RAW.slide != "pooled") & ~F6_RAW.control], F6[~F6.control]])
sp_ = F6[F6.control]
band = (sp_.coef.min(), sp_.coef.max()) if len(sp_) else None
coef_panel(ax, d, [f"{l}→{r}" for l, r in PAIRS], spill=band,
           title=f"Fibroblast receptor vs ligand⁺ T cell within {R_CONTACT} µm")
ax.legend(fontsize=5.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
D._finish(fig, fig28d("F6_axes_proximity"), True)
logfig("F6_axes_proximity", "GLM coefficient of ligand⁺ T within 30 µm on fibroblast receptor, beyond T proximity")

with pd.option_context("display.width", 220):
    print(F6[["pair", "gene", "control", "n", "n_pos", "frac_fibro_ligT_near", "coef", "se", "q",
              "n_slides_agree", "ligand_gate", "t_leak", "beyond_spill", "robust"]].round(3).to_string(index=False))
```

```
pairs on panel: [('TNFSF14', 'LTBR'), ('TNFSF14', 'TNFRSF14'), ('CD40LG', 'CD40'), ('TGFB1', 'TGFBR2'), ('ITGB8', 'TGFBR2'), ('TGFB1', 'FN1'), ('GATA3', 'IL4R')]

<Figure size 550x300 with 1 Axes>
            pair     gene  control    n  n_pos  frac_fibro_ligT_near   coef    se     q  n_slides_agree  ligand_gate  t_leak  beyond_spill  robust
    TNFSF14→LTBR     LTBR    False 8844    755                 0.348 -0.202 0.107 0.137               3        False    True          True   False
    TNFSF14→LTBR     CD3E     True 8844   1334                 0.348  0.054 0.086   NaN               2        False    True         False   False
TNFSF14→TNFRSF14 TNFRSF14    False 8844   1530                 0.348 -0.150 0.067 0.093               3        False    True          True   False
TNFSF14→TNFRSF14     CD3E     True 8844   1334                 0.348  0.054 0.086   NaN               2        False    True         False   False
     CD40LG→CD40     CD40    False 8844    739                 0.300 -0.090 0.108 0.699               1        False    True          True   False
     CD40LG→CD40     CD3E     True 8844   1334                 0.300  0.207 0.072   NaN               3        False    True         False   False
    TGFB1→TGFBR2   TGFBR2    False 8844   1065                 0.391 -0.159 0.072 0.093               2        False    True          True   False
    TGFB1→TGFBR2     CD3E     True 8844   1334                 0.391 -0.011 0.089   NaN               2        False    True          True   False
    ITGB8→TGFBR2   TGFBR2    False 8844   1065                 0.194  0.054 0.086 0.699               2        False    True         False   False
    ITGB8→TGFBR2     CD3E     True 8844   1334                 0.194  0.017 0.053   NaN               2        False    True         False   False
       TGFB1→FN1      FN1    False 8844   2359                 0.391 -0.028 0.053 0.699               2        False   False          True   False
       TGFB1→FN1     CD3E     True 8844   1334                 0.391 -0.011 0.089   NaN               2        False    True          True   False
      GATA3→IL4R     IL4R    False 8844    832                 0.393  0.038 0.117 0.745               1        False    True         False   False
      GATA3→IL4R     CD3E     True 8844   1334                 0.393  0.003 0.079   NaN               3        False    True         False   False
```

```python  # cell 12
# ============================================================================
# §R  Outputs + summary
# ============================================================================
CELLS_OUT = CS.S1_DIR / "processed" / "cosmx_s1_cells_28d.parquet"
adata.obs[["slide", "stage", "fov", "fov_key", "piece", "leiden", "lineage", "t_sub", "insitutype",
           "depth_um", "nCount_RNA"]].assign(x_um=adata.obsm["spatial"][:, 0], y_um=adata.obsm["spatial"][:, 1]
                                             ).to_parquet(CELLS_OUT)
pd.DataFrame(FIG_INDEX).to_csv(tab28d("figure_index"), index=False)
print("cell table →", CELLS_OUT)
print(pd.DataFrame(FIG_INDEX).to_string(index=False))

print("\n" + "=" * 90 + "\nCAVEATS (read first)\n" + "=" * 90)
print(f"- stage = slide: patch {[s for s in SLIDES if SLIDE_STAGE[s] == 'patch']} vs plaque "
      f"{[s for s in SLIDES if SLIDE_STAGE[s] == 'plaque']} → described, never tested; both are early lesions")
print(f"- GEO patient label unusable (nn same-patient {MIX.nn_same.median():.2f} vs random {MIX.random.median():.2f});"
      f" unit = piece ({pd.unique(PIECE).size}) within slide ({len(SLIDES)})")
print(f"- genes at the floor, no claims: {GATE.index[~GATE['pass']].tolist()}")
print("- no malignant/reactive split: KIR3DL2, PLS3, CCR4 off panel, TOX at floor, no TCR")

print("\n" + "=" * 90 + "\nRESULTS\n" + "=" * 90)
_e = F1[F1.expected_28b.notna() & F1["pass"]]
print(f"F1  top lineage matches 28b for {(_e.agrees == 'yes').sum()}/{len(_e)} gated genes:",
      {g: f"{r.top_lineage} (exp {r.expected_28b})" for g, r in _e.iterrows()})
_r4 = F4[~F4.control]
print("F4  robust niche effects (q<0.05, ≥3/4 slides, beyond spillover where binding):",
      {f"{l} {g}": round(c, 3) for l, g, c in _r4[_r4.robust][["lineage", "gene", "coef"]].itertuples(index=False)} or "none")
print("    all:", {f"{l} {g}": f"{c:+.3f} (q {q:.2g}, {int(a)}/4)" for l, g, c, q, a in
                   _r4[["lineage", "gene", "coef", "q", "n_slides_agree"]].itertuples(index=False)})
for r in F5[F5.slide == "pooled"].itertuples():
    _n = F5[(F5.slide != "pooled") & (F5.contrast == r.contrast)]
    print(f"F5  {r.contrast}: Δlog2 {r.delta:+.2f} pooled; hi > lo in {(_n.delta > 0).sum()}/{len(_n)} slides")
_r6 = F6[~F6.control]
print("F6  robust ligand-specific proximity effects:",
      {p: round(c, 3) for p, c in _r6[_r6.robust][["pair", "coef"]].itertuples(index=False)} or "none")
print("    ligands passing the gate in T cells:", sorted(_r6[_r6.ligand_gate].ligand.unique()) or "none")
```

```
cell table → /home/projects/nyosef/zvise/MF/data/S1_cosmx_earlymf_spatial/processed/cosmx_s1_cells_28d.parquet
                figure                              file                                                                                 what
    F0_lineage_markers     deg28d_F0_lineage_markers.png                                                    marker dotplot per called lineage
F1_protease_by_lineage deg28d_F1_protease_by_lineage.png                              protease/inhibitor dotplot per lineage; grey = at floor
    F2a_slide_overview     deg28d_F2a_slide_overview.png             whole slides at true coordinates: every cell, FOV outlines, piece labels
       F2b_maps_slide3        deg28d_F2b_maps_slide3.png         slide 3: every piece packed at true scale; lineage / MMP2 / MMP9 / T density
       F2b_maps_slide4        deg28d_F2b_maps_slide4.png         slide 4: every piece packed at true scale; lineage / MMP2 / MMP9 / T density
       F2b_maps_slide7        deg28d_F2b_maps_slide7.png         slide 7: every piece packed at true scale; lineage / MMP2 / MMP9 / T density
       F2b_maps_slide8        deg28d_F2b_maps_slide8.png         slide 8: every piece packed at true scale; lineage / MMP2 / MMP9 / T density
   F2c_zoom_mmp9_mmp12    deg28d_F2c_zoom_mmp9_mmp12.png     300 µm windows (max myeloid MMP9+MMP12 per slide): lineage / MMP9 / MMP12 counts
              F3_depth               deg28d_F3_depth.png         fraction T / MMP2+ fibro / MMP9+ and MMP12+ myeloid per depth bin, per slide
    F4_niche_T_density     deg28d_F4_niche_T_density.png Poisson GLM coefficient of log1p(T within 50 µm), pooled + per slide, spillover band
   F5_nhood_enrichment    deg28d_F5_nhood_enrichment.png                        log2 obs/exp contact frequency, labels permuted within pieces
     F6_axes_proximity      deg28d_F6_axes_proximity.png GLM coefficient of ligand⁺ T within 30 µm on fibroblast receptor, beyond T proximity

==========================================================================================
CAVEATS (read first)
==========================================================================================
- stage = slide: patch ['3', '4'] vs plaque ['7', '8'] → described, never tested; both are early lesions
- GEO patient label unusable (nn same-patient 0.32 vs random 0.29); unit = piece (41) within slide (4)
- genes at the floor, no claims: ['MMP1', 'MMP7', 'MMP12', 'MMP19', 'CTSD', 'CTSG', 'CTSW', 'GZMA', 'GZMB', 'GZMH', 'GZMK', 'PRF1', 'ELANE', 'PRTN3', 'FAP', 'DPP4', 'TIMP1', 'CST7', 'CSTB', 'SERPINA1', 'SERPINA3', 'SLPI', 'TNFSF14', 'LTBR', 'TNFRSF14', 'CD40LG', 'CD40', 'TGFB1', 'ITGB8', 'TGFBR1', 'TGFBR2', 'GATA3', 'IL4R', 'CD44']
- no malignant/reactive split: KIR3DL2, PLS3, CCR4 off panel, TOX at floor, no TCR

==========================================================================================
RESULTS
==========================================================================================
F1  top lineage matches 28b for 3/3 gated genes: {'MMP2': 'Fibroblast (exp Fibroblast)', 'MMP9': 'Myeloid (exp Myeloid)', 'MMP14': 'Fibroblast (exp Fibroblast)'}
F4  robust niche effects (q<0.05, ≥3/4 slides, beyond spillover where binding): {'Fibroblast COL1A1': -0.165, 'Myeloid MMP14': -0.148}
    all: {'Fibroblast MMP2': '-0.113 (q 0.087, 2/4)', 'Fibroblast MMP14': '-0.079 (q 0.068, 3/4)', 'Fibroblast TIMP1': '+0.181 (q 0.019, 3/4)', 'Fibroblast COL1A1': '-0.165 (q 4.9e-10, 4/4)', 'Myeloid MMP9': '+0.001 (q 0.99, 1/4)', 'Myeloid MMP12': '+0.016 (q 0.98, 2/4)', 'Myeloid MMP14': '-0.148 (q 0.0003, 4/4)', 'Myeloid TIMP1': '+0.035 (q 0.65, 3/4)'}
F5  Fibro MMP2-hi vs lo, next to T: Δlog2 -0.20 pooled; hi > lo in 2/4 slides
F5  Myeloid MMP9/12-hi vs lo, next to T: Δlog2 +0.25 pooled; hi > lo in 3/4 slides
F6  robust ligand-specific proximity effects: none
    ligands passing the gate in T cells: none
```
