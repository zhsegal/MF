# old/28e_cosmx_tcell_compartments.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/28e_cosmx_tcell_compartments.ipynb` and is never run._

# 28e — T cells in situ: compartments, malignancy, motility proxies, and the degradome around them (CosMx)

Follow-up to 28d on the same CosMx early-MF layer (GSE337336; 65,717 cells, 4 slides, 41 pieces). 28d placed the proteases in cell types (F1: MMP2/MMP14 fibroblast, MMP9 myeloid, as in 28b) and found that stromal MMP does **not** rise with local T density per cell (F4), myeloid MMP9/12-hi cells sit next to T in 3/4 slides (F5), and no 28b axis shows in proximity (F6). T cells there were one pool. This notebook reads the T cells against 28b and 28c:

| fig | question | 28b / 28c link |
|---|---|---|
| C | where the T cells are: epidermis (≥50% keratinocyte neighbours), DEJ (≤50 µm), dermis | 28c row C (epidermis vs dermis) |
| F7 | how much of the MMP transcript pool T cells carry, vs fibroblasts and myeloid | 28b F1/F2: the clone carries no headline protease |
| F8 | which T cells look malignant (scRNA-trained transfer score), and are they epidermotropic | 26b `mal_cluster` |
| F9 | motility / residence proxies in epidermal vs dermal T cells, vs keratinocyte contact | 28c amoeboid programs (off-panel) |
| F10 | is there more protease around T cells than around matched cells, and around malignant-like T cells | 28c: protease-independent motility |
| F11 | Hotspot: spatial gene modules and protease × T-gene local correlation | model-free F4/F5 |
| F12 | Harreman: cell-type-aware T ↔ stroma / keratinocyte LR communication | 28b axes (F6) |

**Run order.** (1) 28d §R wrote `cosmx_s1_cells_28d.parquet`. (2) `jobs/run_cosmx_resolvi.sh` (GPU) and `jobs/run_cosmx_transfer.sh` (CPU) — both optional: panels that need them print "skipped". (3) This notebook through §R (writes `cosmx_s1_cells_28e.parquet`). (4) `jobs/run_cosmx_harreman.sh` (`pxlgn` env). (5) Re-run F11, F12, §R.

**Read before any number below.**
- **The 28c amoeboid programs cannot be measured here.** Of their 37 genes only AQP3 (keratinocyte-high) and LMNA are on the 1000-plex panel. F9 reads proxies — cell shape (AspectRatio, Area), LMNA, residence (ITGAE, CD69, CXCR3, CCR10, CXCR6) vs egress (KLF2, SELL, CCR7, CXCR4) genes — and compares signs with 28c row C only qualitatively.
- **Malignancy is a transfer score, not a call.** No TCR; KIR3DL2, PLS3, CCR4, CD7, TNFRSF8 off panel. The score is a logistic model trained on 26b calls restricted to spillover-safe panel genes at CosMx depth; it is used only as tertiles (mal-hi / mid / lo) for group contrasts, and only if held-out-donor AUC ≥ 0.75 **and** it behaves like malignancy in situ (CD8 T cells score below CD4; high scorers are biopsy-specific). F8 reports both gates.
- **Spillover.** A T cell inside the epidermis reads keratin; a fibroblast touching T cells reads CD3E. Raw T-cell MMP counts are an upper bound. resolVI-corrected expression (own-cell term) is shown beside raw, but it does not remove keratinocyte transcripts from intraepidermal T cells (corrected KRT14 rises as much as raw), so F9 instead adjusts for each cell's own keratin read and relies on decreases, which spillover cannot create.
- Units as in 28d: piece within slide; stage = slide (2 vs 2) → described, never tested; robust = pooled BH q < 0.05 **and** ≥3/4 slides share the sign.

Outputs: `figures/degradome/deg28e_*.png`, `tables/deg28e_*.csv`, per-cell `data/S1_cosmx_earlymf_spatial/processed/cosmx_s1_cells_28e.parquet`.

```python  # cell 1
# ============================================================================
# §0  Setup + load: CosMx counts, 28d per-cell table, optional resolVI / transfer outputs
# ============================================================================
# 65k cells × 1000 genes: light. resolVI and the transfer model are trained by jobs; without
# their outputs the panels that need them print "skipped" and the rest runs.
import warnings; warnings.filterwarnings("ignore")
import os
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"]:
    os.environ.setdefault(_v, "8")
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, wilcoxon
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

import degradome_data as cfg          # protease families + output dirs
import degradome_helpers as D         # _finish
import cosmx_spatial as CS            # ingest, graphs, GLM/OLS, compartments, reporting

PREFIX = "deg28e_"
fig28e = lambda name: cfg.FIG_DIR / f"{PREFIX}{name}.png"
tab28e = lambda name: cfg.TAB_DIR / f"{PREFIX}{name}.csv"
FIG_INDEX = []


def logfig(name, what):
    FIG_INDEX.append({"figure": name, "file": fig28e(name).name, "what": what})


R_TIGHT, R_CONTACT, R_NICHE = 20, 30, 50   # µm: Pautrier cluster / contact / niche (as 28d)
EPI_FRAC, DEJ_UM = 0.5, 50.0               # epidermis = ≥50% keratinocyte neighbours within R_CONTACT
P_REASSIGN = 0.7                           # resolVI probability needed to re-call an unlabeled cell
MIN_PIECE_CELLS = 20
TX_MODEL = CS.S1_DIR / "processed" / "malig_transfer_model.json"

adata = CS.load_s1(cache=True, verbose=False)
assert adata.n_obs == CS.N_CELLS_EXPECTED, adata.n_obs
C28 = pd.read_parquet(CS.CELLS_28D)
assert C28.index.equals(adata.obs_names), "28d cell table and cosmx_s1.h5ad differ: re-run 28d §R"
for c in ["leiden", "lineage", "t_sub"]:
    adata.obs[c] = C28[c].astype(str).to_numpy()
adata.X = CS.lognorm(adata)
LIN28 = adata.obs.lineage.to_numpy()
KC_T = (LIN28 == "Keratinocyte") & (adata.obs.insitutype.astype(str).to_numpy() == "T.cell")

HAS_RVI = CS.RESOLVI_H5AD.exists()
RVI = None
if HAS_RVI:
    RVI = ad.read_h5ad(CS.RESOLVI_H5AD)
    assert adata.obs_names.isin(RVI.obs_names).all(), "resolVI output misses cells: re-run the job"
    RVI = RVI[adata.obs_names].copy()
    PROB = RVI.obs.filter(regex="^p_")
    best = PROB.idxmax(1).str[2:].to_numpy()
    unl = RVI.obs.lab_resolvi.astype(str).to_numpy() == "unknown"
    lin_final = np.where(unl & (PROB.max(1).to_numpy() >= P_REASSIGN), best, LIN28)
    print(f"resolVI: {RVI.shape}; {RVI.uns.get('note', '')}")
    print("unlabeled cells re-called (p ≥ %.1f):" % P_REASSIGN)
    print(pd.crosstab(pd.Series(LIN28[unl], name="28d"), pd.Series(lin_final[unl], name="final")))
    print(f"keratinocyte-called / InSituType T.cell ({KC_T.sum():,}): final",
          pd.Series(lin_final[KC_T]).value_counts().to_dict())
else:
    lin_final = LIN28
    print(f"resolVI: skipped (no {CS.RESOLVI_H5AD.name}; jobs/run_cosmx_resolvi.sh). "
          f"{KC_T.sum():,} keratinocyte-called cells that InSituType calls T stay Keratinocyte.")
adata.obs["lineage_final"] = lin_final

HAS_TX = TX_MODEL.exists()
print(f"malignancy transfer model: {'found' if HAS_TX else 'skipped (jobs/run_cosmx_transfer.sh)'}")
print("\nfinal lineage × slide:")
print(pd.crosstab(adata.obs.lineage_final, adata.obs.slide))
```

```
resolVI: (65717, 104); resolVI corrected expression at library 233; semisupervised on 28d lineage; 100 epochs; n_latent 10; nb28e input
unlabeled cells re-called (p ≥ 0.7):
final         Fibroblast  Keratinocyte  Myeloid     T  Unassigned
28d                                                              
Keratinocyte           0          1778        0   229           0
Unassigned          2379           183     1728  5149        2883
keratinocyte-called / InSituType T.cell (2,007): final {'Keratinocyte': 1778, 'T': 229}
malignancy transfer model: found

final lineage × slide:
slide             3     4     7     8
lineage_final                        
Fibroblast      662   342  4280  8848
Keratinocyte   9817  4574  7000  2414
Myeloid         660   520  4083  4743
T               767   977  3816  9331
Unassigned       26     9  1856   992
```

```python  # cell 2
# ============================================================================
# §U  Shared arrays, graphs, fitting utilities
# ============================================================================
LIN = adata.obs.lineage_final.astype(str).to_numpy()
SLIDES = list(adata.obs.slide.cat.categories)
SLIDE_STAGE = adata.obs.groupby("slide", observed=True).stage.first().astype(str).to_dict()
FOV = adata.obs.fov_key.astype(str).to_numpy()
PIECE = adata.obs.piece.astype(str).to_numpy()
SLIDE = adata.obs.slide.astype(str).to_numpy()
NCOUNT = adata.obs.nCount_RNA.to_numpy().astype(float)
LIBOFF = np.log(NCOUNT)
COUNTS = adata.layers["counts"].tocsc()
IS_T = LIN == "T"
LIN4 = ["T", "Fibroblast", "Myeloid", "Keratinocyte"]
N = adata.n_obs


def gcounts(g):
    return np.asarray(COUNTS[:, adata.var_names.get_loc(g)].todense()).ravel().astype(float)


def gnorm(g):
    """Counts scaled to the median library (the 28d gene-gate scale)."""
    return gcounts(g) * (np.median(NCOUNT) / NCOUNT)


def rvi(g):
    """resolVI-corrected expression (own-cell term, median library), or None."""
    if not HAS_RVI or g not in RVI.var_names:
        return None
    return np.asarray(RVI[:, g].X).ravel().astype(float)


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)


def min_piece(idx, n=MIN_PIECE_CELLS):
    """Drop cells of pieces holding fewer than `n` of the selected cells."""
    cnt = pd.Series(PIECE[idx]).map(pd.Series(PIECE[idx]).value_counts()).to_numpy()
    return idx[cnt >= n]


def fits(y, X, idx, kind="glm", **meta):
    """Pooled + per-slide fit of X's first column: piece fixed effects, SEs clustered on FOV.
    kind='glm': Poisson with log-library offset (counts); 'ols': continuous outcome."""
    rows = []
    for s in ["pooled"] + SLIDES:
        ii = idx if s == "pooled" else idx[SLIDE[idx] == s]
        if len(ii) < 50:
            continue
        Xi = X.iloc[ii].reset_index(drop=True)
        r = (CS.niche_glm(y[ii], Xi, LIBOFF[ii], FOV[ii], fe=PIECE[ii]) if kind == "glm"
             else CS.ols_fe(y[ii], Xi, FOV[ii], fe=PIECE[ii]))
        rows.append({**meta, "slide": s, **r})
    return rows


# CD4/CD8 split on the final lineage (28d's t_sub misses re-called T cells); descriptive only.
_cd4, _cd8 = gcounts("CD4"), gcounts("CD8A") + gcounts("CD8B")
TSUB = np.where(IS_T, np.select([_cd8 > _cd4, _cd4 > _cd8], ["T_CD8", "T_CD4"], "T_other"), "")
print("T subsets:", pd.Series(TSUB[IS_T]).value_counts().to_dict())

# Graphs (per slide, µm) and the neighbour composition every section uses.
G20, G30, G50 = (CS.neighbors(adata, r) for r in (R_TIGHT, R_CONTACT, R_NICHE))
NB30, TOT30 = CS.nbr_composition(G30, LIN)
NB50, TOT50 = CS.nbr_composition(G50, LIN)
nb_T = lambda NB: NB["T"].to_numpy() if "T" in NB else np.zeros(N)
print(f"mean neighbours: {R_TIGHT} µm {G20.nnz / N:.1f} | {R_CONTACT} µm {G30.nnz / N:.1f} | "
      f"{R_NICHE} µm {G50.nnz / N:.1f}")

ALL_GENES = sorted({g for f in ["mmp", "cathepsin", "granzyme", "serine_other", "metallo_other", "inhibitor"]
                    for g in cfg.DEGRADOME_FAMILIES[f]}
                   | set(CS.RETENTION + CS.EGRESS + CS.LAMINA + CS.KER_SPILL + CS.T_SPILL + ["AQP3"]))
GATE, MEAN_L, DET_L = CS.gene_gate(adata, sorted(ALL_GENES), lineage_key="lineage_final")
GATE.to_csv(tab28e("gene_gate"))
print(f"gene gate on the final lineages: {int(GATE['pass'].sum())}/{len(GATE)} pass")
```

```
T subsets: {'T_other': 10270, 'T_CD4': 2591, 'T_CD8': 2030}

mean neighbours: 20 µm 12.3 | 30 µm 27.1 | 50 µm 69.8
gene gate on the final lineages: 9/40 pass
```

```python  # cell 3
# ============================================================================
# §1 / C — compartments: epidermis, DEJ, dermis
# ============================================================================
# 28d's depth (µm to the nearest keratinocyte) cannot tell an intraepidermal T cell from a
# papillary-dermis T cell touching the basal layer: both are ~0-15 µm away. The keratinocyte
# fraction among a cell's 30 µm neighbours can. epidermis = every keratinocyte and every cell with
# ≥ EPI_FRAC keratinocyte neighbours; DEJ = otherwise ≤ DEJ_UM from a keratinocyte; dermis beyond;
# no_epidermis = pieces with < 50 keratinocytes. Pautrier-like = intraepidermal T cell with ≥ 2 T
# neighbours within 20 µm (position only, no histology).
DEPTH = CS.dist_to_lineage(adata, "Keratinocyte", lineage_key="lineage_final")
COMP, KFRAC = CS.compartments(G30, LIN, DEPTH, EPI_FRAC, DEJ_UM)
PAUT = CS.pautrier(G20, IS_T, COMP)
N_KER20 = G20 @ (LIN == "Keratinocyte").astype(np.float32)
HAS_EPI = COMP != "no_epidermis"
COMP_ORDER = ["epidermis", "DEJ", "dermis", "no_epidermis"]
COMP_COLORS = {"epidermis": "#c9a66b", "DEJ": "#8fb3d9", "dermis": "#3a5f8a", "no_epidermis": "#d9d9d9"}

TABC = pd.crosstab(SLIDE[IS_T], COMP[IS_T]).reindex(columns=COMP_ORDER, fill_value=0)
TABC["pautrier"] = pd.Series(PAUT[IS_T]).groupby(SLIDE[IS_T]).sum()
TABC["stage"] = [SLIDE_STAGE[s] for s in TABC.index]
TABC.to_csv(tab28e("C_T_by_compartment"))
print("T cells by compartment (rows = slide):")
print(TABC.to_string())
if HAS_RVI:
    print("\nre-called keratinocyte/InSituType-T cells now T:", int((KC_T & IS_T).sum()),
          "| their compartments:", pd.Series(COMP[KC_T & IS_T]).value_counts().to_dict())

_te = IS_T & HAS_EPI
EPI_SHARE = (COMP[_te] == "epidermis").mean()
fig, axes = plt.subplots(1, 2, figsize=(8, 3), gridspec_kw={"width_ratios": [1.1, 1]})
ax = axes[0]
frac = TABC[COMP_ORDER].div(TABC[COMP_ORDER].sum(1), axis=0)
left = np.zeros(len(frac))
for c in COMP_ORDER:
    ax.barh(range(len(frac)), frac[c], left=left, color=COMP_COLORS[c], label=c, height=0.7)
    left += frac[c].to_numpy()
ax.set_yticks(range(len(frac)))
ax.set_yticklabels([f"slide {s} ({SLIDE_STAGE[s]})\nn={int(TABC.loc[s, COMP_ORDER].sum()):,}" for s in frac.index],
                   fontsize=6.5)
ax.set_xlabel("fraction of T cells", fontsize=7); ax.tick_params(labelsize=6.5)
ax.legend(fontsize=6, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
_clean(ax)
ax = axes[1]
for s in SLIDES:
    m = _te & (SLIDE == s)
    ax.hist(KFRAC[m], bins=np.linspace(0, 1, 21), histtype="step", lw=1, density=True,
            color=CS.STAGE_COLORS[SLIDE_STAGE[s]], ls=["-", "--"][SLIDES.index(s) % 2], label=f"slide {s}")
ax.axvline(EPI_FRAC, color="k", ls=":", lw=0.8)
ax.set_xlabel(f"keratinocyte fraction of {R_CONTACT} µm neighbours (T cells)", fontsize=7)
ax.set_ylabel("density", fontsize=7); ax.tick_params(labelsize=6.5)
ax.legend(fontsize=5.5, frameon=False)
_clean(ax)
fig.suptitle(f"{EPI_SHARE:.0%} of T cells in pieces with epidermis sit inside it "
             f"(≥{EPI_FRAC:.0%} keratinocyte neighbours); {int(PAUT.sum())} in Pautrier-like clusters",
             fontsize=9)
D._finish(fig, fig28e("C_compartments"), True)
logfig("C_compartments", "T cells per compartment per slide; keratinocyte-neighbour fraction of T cells")
```

```
T cells by compartment (rows = slide):
col_0  epidermis   DEJ  dermis  no_epidermis  pautrier   stage
row_0                                                         
3            384   344      39             0       154   patch
4            210   724      43             0       115   patch
7            137  1148    2531             0        33  plaque
8             58  1714    6176          1383        31  plaque

re-called keratinocyte/InSituType-T cells now T: 229 | their compartments: {'epidermis': 200, 'DEJ': 28, 'dermis': 1}

<Figure size 800x300 with 2 Axes>
```

```python  # cell 4
# ============================================================================
# F7 — how much of the MMP transcript pool do T cells carry?
# ============================================================================
# (a) Share of all MMP counts per lineage, beside the lineage's share of cells and of total
#     library. A gene at the ambient floor is spread in proportion to library size, so the
#     library share is the "no expression" reference, not the cell share.
# (b) Per piece (unit), mean library-normalised expression in T cells vs fibroblasts / myeloid;
#     paired signed-rank over pieces, BH within source; robust = q < 0.05, T lower, ≥3/4 slides.
#     Raw T-cell values include neighbour spillover (upper bound); resolVI rows use the corrected
#     own-cell term.
# (c) The same contrast in 28b scRNA (tables/deg28bmc_F2{a,b}_source_*: per-donor medians, log1p
#     CP10K), each platform scaled to its highest lineage.
F7_GENES = [g for g in ["MMP2", "MMP9", "MMP14", "MMP12", "MMP1", "TIMP1"] if g in adata.var_names]
F7_HEAD = [g for g in ["MMP2", "MMP9", "MMP14"] if g in F7_GENES]
PARTNER = {"MMP2": "Fibroblast", "MMP14": "Fibroblast", "MMP1": "Fibroblast", "TIMP1": "Fibroblast",
           "MMP9": "Myeloid", "MMP12": "Myeloid"}

rows = []
for s in ["pooled"] + SLIDES:
    m = (LIN != "Unassigned") & ((SLIDE == s) if s != "pooled" else True)
    cell_sh = pd.Series(LIN[m]).value_counts(normalize=True)
    lib = pd.Series(NCOUNT[m]).groupby(LIN[m]).sum()
    for g in ["cells", "library"] + F7_GENES:
        sh = cell_sh if g == "cells" else (lib / lib.sum() if g == "library" else
                                           (lambda c: c / c.sum())(pd.Series(gcounts(g)[m]).groupby(LIN[m]).sum()))
        rows += [{"slide": s, "what": g, "lineage": l, "share": float(sh.get(l, 0))} for l in LIN4]
F7A = pd.DataFrame(rows)
F7A.to_csv(tab28e("F7a_share"), index=False)
_p = F7A[F7A.slide == "pooled"].pivot(index="what", columns="lineage", values="share")
print("pooled share by lineage:")
print(_p.reindex(["cells", "library"] + F7_GENES)[LIN4].round(3).to_string())


def piece_means(v):
    df = pd.DataFrame({"slide": SLIDE, "piece": PIECE, "lin": LIN, "v": v})
    g = df[df.lin.isin(LIN4)].groupby(["slide", "piece", "lin"]).v.agg(["mean", "size"]).reset_index()
    return g[g["size"] >= MIN_PIECE_CELLS].pivot_table(index=["slide", "piece"], columns="lin", values="mean")


EPS = 0.01
rows, PM = [], {}
for src, get in [("raw", gnorm)] + ([("resolVI", rvi)] if HAS_RVI else []):
    for g in F7_GENES:
        v = get(g)
        if v is None:
            continue
        P = piece_means(v)
        PM[(src, g)] = P
        for other in ["Fibroblast", "Myeloid"]:
            d = P[["T", other]].dropna() if {"T", other} <= set(P.columns) else pd.DataFrame()
            if len(d) < 6:
                continue
            lr = np.log2((d["T"] + EPS) / (d[other] + EPS))
            per = lr.groupby(level="slide").median()
            rows.append({"source": src, "gene": g, "vs": other, "n_pieces": len(d),
                         "mean_T": d["T"].median(), "mean_other": d[other].median(),
                         "log2_T_over_other": lr.median(), "p": wilcoxon(d["T"], d[other]).pvalue,
                         "n_slides_T_lower": int((per < 0).sum()), "n_slides": len(per)})
F7B = pd.DataFrame(rows)
F7B["q"] = F7B.groupby("source")["p"].transform(lambda p: multipletests(p.fillna(1), method="fdr_bh")[1])
F7B["robust_T_lower"] = (F7B.q < 0.05) & (F7B.log2_T_over_other < 0) & (F7B.n_slides_T_lower >= 3)
F7B["gate_pass"] = F7B.gene.map(GATE["pass"])
F7B.to_csv(tab28e("F7b_piece_T_vs_stroma"), index=False)
with pd.option_context("display.width", 200):
    print("\nper piece, T vs stroma (medians over pieces of library-normalised means):")
    print(F7B.round(3).to_string(index=False))

SC_ROWS = []
for f, lin_other in [("deg28bmc_F2a_source_fibro.csv", "Fibroblast"), ("deg28bmc_F2b_source_myeloid.csv", "Myeloid")]:
    p = cfg.TAB_DIR / f
    if not p.exists():
        print(f"(c) skipped: {f} missing")
        continue
    t = pd.read_csv(p)
    grp = np.where(t.group.str.startswith("CD4"), "T", lin_other)
    for g in [g for g in F7_GENES if g in t.columns]:
        lv = t.groupby(grp)[g].mean()
        SC_ROWS += [{"gene": g, "lineage": l, "platform": "scRNA (28b)", "value": lv[l]} for l in lv.index]
for g in F7_GENES:
    lv = pd.Series(gnorm(g)).groupby(LIN).mean()
    SC_ROWS += [{"gene": g, "lineage": l, "platform": "CosMx", "value": lv.get(l, np.nan)}
                for l in ["T", "Fibroblast", "Myeloid"]]
F7C = pd.DataFrame(SC_ROWS)
F7C["rel"] = F7C["value"] / F7C.groupby(["gene", "platform"])["value"].transform("max")
F7C.to_csv(tab28e("F7c_platforms"), index=False)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.1), gridspec_kw={"width_ratios": [1.1, 1.3, 1.2]})
ax = axes[0]
cols = ["cells", "library"] + F7_HEAD
bottom = np.zeros(len(cols))
for l in LIN4:
    v = _p.reindex(cols)[l].to_numpy()
    ax.bar(range(len(cols)), v, bottom=bottom, color=CS.LINEAGE_COLORS[l], label=l, width=0.7)
    bottom += v
ax.set_xticks(range(len(cols)))
ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=6.5)
ax.set_ylabel("share (pooled, excl. Unassigned)", fontsize=7); ax.tick_params(labelsize=6.5)
ax.legend(fontsize=5.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0)); _clean(ax)
ax.set_title("transcript share vs cell / library share", fontsize=8)
ax = axes[1]
xt = []
for j, g in enumerate(F7_HEAD):
    for k, src in enumerate(["raw", "resolVI"]):
        if (src, g) not in PM or {"T", PARTNER[g]} - set(PM[(src, g)].columns):
            continue
        d = PM[(src, g)][["T", PARTNER[g]]].dropna()
        lr = np.log2((d["T"] + EPS) / (d[PARTNER[g]] + EPS))
        x = j + (k - 0.5) * 0.35
        for s in SLIDES:
            v = lr[lr.index.get_level_values("slide") == s]
            ax.scatter(np.full(len(v), x) + np.random.default_rng(0).uniform(-0.06, 0.06, len(v)), v, s=7,
                       color=CS.STAGE_COLORS[SLIDE_STAGE[s]], marker=CS.SLIDE_MARKERS[SLIDES.index(s)],
                       edgecolor="none", alpha=0.8 if src == "raw" else 0.5)
        ax.hlines(lr.median(), x - 0.13, x + 0.13, color="k", lw=1.2)
    xt.append(f"{g}\nT / {PARTNER[g][:5]}")
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(range(len(F7_HEAD))); ax.set_xticklabels(xt, fontsize=6.5)
ax.set_ylabel("log2 T / stroma, per piece", fontsize=7); ax.tick_params(labelsize=6.5)
ax.set_title("per piece: left raw, right resolVI" if HAS_RVI else "per piece (raw counts)", fontsize=8)
_clean(ax)
ax = axes[2]
plat = ["CosMx", "scRNA (28b)"]
w = 0.13
for j, g in enumerate(F7_HEAD):
    for k, pl in enumerate(plat):
        for i, l in enumerate(["T", "Fibroblast", "Myeloid"]):
            r = F7C[(F7C.gene == g) & (F7C.platform == pl) & (F7C.lineage == l)]
            if len(r):
                ax.bar(j + (k * 3 + i - 2.5) * w, r.rel.iloc[0], width=w * 0.9, color=CS.LINEAGE_COLORS[l],
                       hatch="" if pl == "CosMx" else "////", edgecolor="w", lw=0)
ax.bar([], [], color="#bbbbbb", label="CosMx"); ax.bar([], [], color="#bbbbbb", hatch="////", label="scRNA (28b)")
ax.set_xticks(range(len(F7_HEAD))); ax.set_xticklabels(F7_HEAD, fontsize=6.5, fontstyle="italic")
ax.set_ylabel("level / top lineage", fontsize=7); ax.tick_params(labelsize=6.5)
ax.legend(fontsize=5.5, frameon=False); ax.set_title("T vs stroma on both platforms", fontsize=8); _clean(ax)
t_cells, t_lib = _p.loc["cells", "T"], _p.loc["library", "T"]
t_mmp = _p.loc[F7_HEAD, "T"]
fig.suptitle(f"T cells are {t_cells:.0%} of cells ({t_lib:.0%} of library) but carry "
             + ", ".join(f"{t_mmp[g]:.0%} of {g}" for g in F7_HEAD), fontsize=9, y=1.03)
D._finish(fig, fig28e("F7_T_share_of_degradome"), True)
logfig("F7_T_share_of_degradome", "MMP transcript share by lineage; per-piece T vs stroma (raw/resolVI); CosMx vs 28b")
```

```
pooled share by lineage:
lineage      T  Fibroblast  Myeloid  Keratinocyte
what                                             
cells    0.237       0.225    0.159         0.379
library  0.159       0.132    0.122         0.588
MMP2     0.124       0.560    0.155         0.160
MMP9     0.192       0.137    0.409         0.262
MMP14    0.121       0.248    0.146         0.485
MMP12    0.123       0.087    0.272         0.519
MMP1     0.207       0.258    0.166         0.369
TIMP1    0.228       0.276    0.152         0.344


per piece, T vs stroma (medians over pieces of library-normalised means):
 source  gene         vs  n_pieces  mean_T  mean_other  log2_T_over_other     p  n_slides_T_lower  n_slides     q  robust_T_lower  gate_pass
    raw  MMP2 Fibroblast        36   0.132       0.682             -2.255 0.000                 4         4 0.000            True       True
    raw  MMP2    Myeloid        37   0.131       0.235             -0.673 0.000                 4         4 0.000            True       True
    raw  MMP9 Fibroblast        36   0.192       0.132              0.511 0.000                 0         4 0.000           False       True
    raw  MMP9    Myeloid        37   0.200       0.488             -1.459 0.000                 4         4 0.000            True       True
    raw MMP14 Fibroblast        36   0.129       0.294             -1.252 0.000                 4         4 0.000            True      False
    raw MMP14    Myeloid        37   0.130       0.191             -0.525 0.000                 4         4 0.000            True      False
    raw MMP12 Fibroblast        36   0.054       0.045              0.169 0.049                 2         4 0.059           False      False
    raw MMP12    Myeloid        37   0.053       0.123             -1.085 0.000                 4         4 0.000            True      False
    raw  MMP1 Fibroblast        36   0.066       0.109             -0.431 0.001                 4         4 0.001            True      False
    raw  MMP1    Myeloid        37   0.073       0.056              0.028 0.694                 2         4 0.694           False      False
    raw TIMP1 Fibroblast        36   0.209       0.329             -0.629 0.000                 4         4 0.000            True      False
    raw TIMP1    Myeloid        37   0.211       0.185              0.107 0.316                 2         4 0.345           False      False
resolVI  MMP2 Fibroblast        36   0.131       1.230             -2.997 0.000                 4         4 0.000            True       True
resolVI  MMP2    Myeloid        37   0.130       0.265             -0.934 0.000                 4         4 0.000            True       True
resolVI  MMP9 Fibroblast        36   0.170       0.115              0.490 0.000                 0         4 0.000           False       True
resolVI  MMP9    Myeloid        37   0.163       0.667             -1.914 0.000                 4         4 0.000            True       True
resolVI MMP14 Fibroblast        36   0.090       0.386             -1.874 0.000                 4         4 0.000            True      False
resolVI MMP14    Myeloid        37   0.090       0.187             -1.015 0.000                 4         4 0.000            True      False
resolVI MMP12 Fibroblast        36   0.036       0.009              1.029 0.000                 0         4 0.000           False      False
resolVI MMP12    Myeloid        37   0.037       0.217             -2.307 0.000                 4         4 0.000            True      False
resolVI  MMP1 Fibroblast        36   0.021       0.088             -1.376 0.000                 4         4 0.000            True      False
resolVI  MMP1    Myeloid        37   0.021       0.026             -0.185 0.032                 4         4 0.032            True      False
resolVI TIMP1 Fibroblast        36   0.215       0.463             -0.888 0.000                 4         4 0.000            True      False
resolVI TIMP1  
... [161 more chars truncated]
```

```python  # cell 5
# ============================================================================
# F8 — malignancy transfer score in situ
# ============================================================================
# jobs/run_cosmx_transfer.py fits a logistic model on 26b `mal_cluster` (skin CD4) using only
# spillover-safe panel genes at CosMx depth and reports held-out-donor AUC. Here: features as in
# training -- log1p(count × lib_scale / nCount), z-scored within CosMx non-CD8 T cells, then the
# training scaler -- give a logit per T cell. The score is relative (within-platform z), so it
# is cut into tertiles over all T cells (mal-hi / mid / lo) and used only for group contrasts.
# Two gates: (1) held-out-donor AUC ≥ 0.75 in scRNA at CosMx depth; (2) in situ, where the
# platform shift is untested: (a) CD8 T cells must score below CD4, and (b) mal-hi cells must be
# more piece-specific in expression space (malignant clones are biopsy-private) -- median
# per-slide Spearman(score, excess same-piece neighbours) > 0. If either fails, the groups are
# reset to 'na' and F9 / F10 / F12 drop their mal-hi / mal-lo splits.
# (c) epidermotropism: score ~ compartment + log nCount + piece (OLS, FOV-clustered), and + the
# cell's own keratin read (log1p KRT14+KRT5) as a spillover sensitivity.
MAL_SCORE = np.full(N, np.nan)
MAL_GROUP = np.full(N, "na", dtype=object)
TX, TX_GATE, TX_OK = None, False, False
if HAS_TX:
    TX = json.loads(TX_MODEL.read_text())
    cv = TX["cv"]
    TX_GATE = cv.get("auc_overall") is not None and cv["auc_overall"] >= TX["auc_gate"]
    print("transfer CV (held-out donors, CosMx depth):", {k: (round(v, 3) if isinstance(v, float) else v)
                                                        for k, v in cv.items()})
    gi = adata.var_names.get_indexer(TX["genes"])
    assert (gi >= 0).all(), "model genes missing from the CosMx panel"
    ti = np.flatnonzero(IS_T)
    Fm = np.log1p(COUNTS[:, gi][ti].toarray() * TX["lib_scale"] / NCOUNT[ti, None])
    ref = TSUB[ti] != "T_CD8"
    Zm = (Fm - Fm[ref].mean(0)) / np.where(Fm[ref].std(0) > 0, Fm[ref].std(0), 1)
    Zm = (Zm - np.array(TX["scaler_mean"])) / np.array(TX["scaler_scale"])
    MAL_SCORE[ti] = Zm @ np.array(TX["coef"]) + TX["intercept"]
    MAL_GROUP[ti] = pd.qcut(MAL_SCORE[ti], 3, labels=["mallo", "malmid", "malhi"]).astype(str)
    print(f"gate 1, held-out AUC ≥ {TX['auc_gate']}: {'passed' if TX_GATE else 'FAILED'}")
    if tab28e("malig_transfer_coef").exists():
        print("top model genes:", pd.read_csv(tab28e("malig_transfer_coef")).head(12)[["gene", "coef"]]
              .round(2).to_dict("records"))
else:
    print("F8 skipped: no transfer model (jobs/run_cosmx_transfer.sh)")

if HAS_TX:
    med_sub = pd.Series(MAL_SCORE[IS_T]).groupby(TSUB[IS_T]).median()
    print("\n(a) score by T subset (median):", med_sub.round(2).to_dict())
    _c4, _c8 = MAL_SCORE[IS_T & (TSUB == "T_CD4")], MAL_SCORE[IS_T & (TSUB == "T_CD8")]
    CD8_OK = bool(med_sub.get("T_CD8", np.inf) < med_sub.get("T_CD4", -np.inf))
    if len(_c4) > 20 and len(_c8) > 20:
        print(f"    CD8 < CD4: MWU p = {mannwhitneyu(_c8, _c4, alternative='less').pvalue:.2g} -> "
              f"{'ok' if CD8_OK else 'FAILED (CD8 score higher)'}")
    print("median score by slide:", {f"{s} ({SLIDE_STAGE[s]})": round(float(np.nanmedian(MAL_SCORE[IS_T & (SLIDE == s)])), 2)
                                     for s in SLIDES})

    # (b) piece specificity in expression space, within slide
    from sklearn.neighbors import NearestNeighbors
    if HAS_RVI:
        LAT, lat_name = RVI.obsm["X_resolvi"], "resolVI latent"
    else:
        from sklearn.decomposition import PCA
        LAT = np.zeros((N, 20), np.float32)
        LAT[IS_T] = PCA(20, random_state=0).fit_transform(adata.X[IS_T].toarray())
        lat_name = "PCA of T cells"
    rows = []
    for s in SLIDES:
        ii = np.flatnonzero(IS_T & (SLIDE == s))
        if len(ii) < 100:
            continue
        nn = NearestNeighbors(n_neighbors=16).fit(LAT[ii]).kneighbors(LAT[ii], return_distance=False)[:, 1:]
        pc = pd.Series(PIECE[ii]).value_counts()
        exp = (pd.Series(PIECE[ii]).map(pc).to_numpy() - 1) / (len(ii) - 1)
        obs = (PIECE[ii][nn] == PIECE[ii][:, None]).mean(1)
        rows.append(pd.DataFrame({"slide": s, "group": MAL_GROUP[ii], "score": MAL_SCORE[ii],
                                  "excess_same_piece": obs - exp}))
    SAME = pd.concat(rows, ignore_index=True)
    SAME.to_csv(tab28e("F8_piece_specificity"), index=False)
    rho = SAME.groupby("slide").apply(lambda d: d[["score", "excess_same_piece"]].corr(method="spearman").iat[0, 1])
    PIECE_OK = bool(np.nanmedian(rho) > 0)
    print(f"(b) same-piece neighbours in {lat_name} beyond composition, Spearman with score per slide:",
          rho.round(3).to_dict(), "->", "ok" if PIECE_OK else "FAILED (no piece specificity)")
    TX_OK = TX_GATE and CD8_OK and PIECE_OK
    print(f"\ntransfer score usable for mal-hi / mal-lo splits: {TX_OK} "
          f"(AUC gate {TX_GATE}, CD8 < CD4 {CD8_OK}, piece specificity {PIECE_OK})")

    # (c) epidermotropism (+ keratin-spillover sensitivity)
    ti = min_piece(np.flatnonzero(IS_T & HAS_EPI))
    lg = pd.Series(np.log(NCOUNT))
    ker = pd.Series(np.log1p(sum(gcounts(g) for g in CS.KER_SPILL if g in adata.var_names)))
    F8_LABS = ["epidermis vs dermis", "epidermis vs dermis | keratin", "DEJ vs dermis"]
    rows = []
    for lab, a, b in zip(F8_LABS, ["epidermis", "epidermis", "DEJ"], ["dermis"] * 3):
        idx = ti[np.isin(COMP[ti], [a, b])]
        X = pd.DataFrame({lab: (COMP == a).astype(float), "log_n": lg})
        if "keratin" in lab:
            X["log_ker"] = ker
        rows += fits(MAL_SCORE, X, idx, kind="ols", contrast=lab, control=False, label=lab)
    F8_RAW = pd.DataFrame(rows)
    F8 = CS.robust_flags(F8_RAW, ["contrast"])
    F8.to_csv(tab28e("F8_score_by_compartment"), index=False)
    print("\n(c) transfer score (logit) by compartment:")
    print(F8[["contrast", "n", "coef", "se", "q", "n_slides_agree", "robust"]].round(3).to_string(index=False))
    print("mal-hi share by compartment:", pd.crosstab(COMP[IS_T], MAL_GROUP[IS_T], normalize="index")
          .get("malhi", pd.Series(dtype=float)).round(3).to_dict())

    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    ax = axes[0]
    for t, c in [("T_CD4", "#d62728"), ("T_other", "#9a9a9a"), ("T_CD8", "#1f77b4")]:
        v = MAL_SCORE[IS_T & (TSUB == t)]
        if len(v):
            ax.hist(v, bins=40, histtype="step", density=True, color=c, lw=1, label=f"{t} (n={len(v):,})")
    ax.set_xlabel("transfer score (logit)", fontsize=7); ax.set_ylabel("density", fontsize=7)
    ax.legend(fontsize=5.5, frameon=False); ax.tick_params(labelsize=6.5); _clean(ax)
    ax.set_title(f"held-out AUC {cv['auc_overall']:.2f} (early {cv.get('auc_early') or float('nan'):.2f}); "
                 f"CD8<CD4 {'ok' if CD8_OK else 'fails'}", fontsize=8)
    ax = axes[1]
    grp_order = ["mallo", "malmid", "malhi"]
    for k, s in enumerate(SLIDES):
        d = SAME[SAME.slide == s]
        if not len(d):
            continue
        m = d.groupby("group").excess_same_piece.mean().reindex(grp_order)
        ax.plot(range(len(grp_order)), m.to_numpy(), marker=CS.SLIDE_MARKERS[k], ms=4, lw=1,
                color=CS.STAGE_COLORS[SLIDE_STAGE[s]], label=f"slide {s} ({SLIDE_STAGE[s]})")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(grp_order))); ax.set_xticklabels(grp_order, fontsize=6.5)
    ax.set_ylabel("same-piece neighbours − expected", fontsize=7); ax.tick_params(labelsize=6.5)
    ax.set_title(f"piece specificity ({lat_name}): {'ok' if PIECE_OK else 'fails'}", fontsize=8)
    ax.legend(fontsize=5.5, frameon=False); _clean(ax)
    CS.coef_panel(axes[2], pd.concat([F8_RAW[F8_RAW.slide != "pooled"], F8]), F8_LABS,
                  SLIDES, SLIDE_STAGE, title="score by compartment (T cells)", ylabel="Δ logit")
    _e = F8[F8.contrast == "epidermis vs dermis"].iloc[0]
    fig.suptitle(("The transfer score separates held-out scRNA donors but fails in situ "
                  f"(CD8 {'<' if CD8_OK else '≥'} CD4, piece specificity {'yes' if PIECE_OK else 'no'}); "
                  if not TX_OK else "Transfer score passes both gates; ")
                 + f"epidermis vs dermis {_e.coef:+.2f} logit (q {_e.q:.2g})", fontsize=9, y=1.03)
    D._finish(fig, fig28e("F8_malignancy_transfer"), True)
    logfig("F8_malignancy_transfer", "transfer score by T subset; piece specificity by tertile; score by compartment")
    if not TX_OK:
        MAL_GROUP[:] = "na"
```

```
transfer CV (held-out donors, CosMx depth): {'auc_overall': 0.917, 'auc_fold_mean': 0.918, 'auc_within_donor_median': 0.908, 'n_donors_within': 82, 'auc_early': 0.934, 'n_early': 59656, 'n_genes': 536, 'n_cells': 247531, 'passes_gate': True}

gate 1, held-out AUC ≥ 0.75: passed
top model genes: [{'gene': 'CCL5', 'coef': -0.52}, {'gene': 'CXCL13', 'coef': 0.5}, {'gene': 'KRT7', 'coef': 0.39}, {'gene': 'B2M', 'coef': -0.39}, {'gene': 'TOX', 'coef': 0.34}, {'gene': 'IL7R', 'coef': -0.3}, {'gene': 'S100A6', 'coef': 0.29}, {'gene': 'KRT86', 'coef': 0.28}, {'gene': 'PPIA', 'coef': 0.27}, {'gene': 'MRC2', 'coef': 0.27}, {'gene': 'SOX4', 'coef': 0.26}, {'gene': 'LGALS3', 'coef': 0.26}]

(a) score by T subset (median): {'T_CD4': 0.52, 'T_CD8': 0.88, 'T_other': 0.6}
    CD8 < CD4: MWU p = 1 -> FAILED (CD8 score higher)
median score by slide: {'3 (patch)': -0.52, '4 (patch)': 1.97, '7 (plaque)': 1.3, '8 (plaque)': 0.4}

(b) same-piece neighbours in resolVI latent beyond composition, Spearman with score per slide: {'3': -0.008, '4': 0.112, '7': 0.028, '8': -0.034} -> ok

transfer score usable for mal-hi / mal-lo splits: False (AUC gate True, CD8 < CD4 False, piece specificity True)


(c) transfer score (logit) by compartment:
                     contrast     n   coef    se     q  n_slides_agree  robust
          epidermis vs dermis  9540 -0.091 0.216 0.673               3   False
epidermis vs dermis | keratin  9540 -0.153 0.222 0.673               3   False
                DEJ vs dermis 12710  0.079 0.067 0.673               2   False
mal-hi share by compartment: {'DEJ': 0.359, 'dermis': 0.322, 'epidermis': 0.362, 'no_epidermis': 0.317}

<Figure size 1100x300 with 3 Axes>
```

```python  # cell 6
# ============================================================================
# F9 — motility / residence proxies: epidermal vs dermal T cells
# ============================================================================
# The 28c programs (osmotic engine, actomyosin, Arp2/3 piston, ESCRT repair, mechanosensors,
# LMNB1/2) are off this panel. What CosMx can read about a T cell in the epidermis:
#   shape  : AspectRatio (elongation) and log Area of the segmented cell, ~ epi + log nCount;
#   lamina : LMNA (28c: lamina ratio higher in epidermal malignant cells);
#   residence (ITGAE/CD103 binds keratinocyte E-cadherin; CD69; CXCR3; CCR10; CXCR6) vs
#   egress (KLF2, SELL, CCR7, CXCR4 -- lower in the clone in 28b/NB28 context).
# Per T cell: gene ~ [epidermis] + log1p(all within 30 µm) + piece, Poisson, offset log nCount;
# epidermis vs dermis and DEJ vs dermis.
# Spillover. An intraepidermal T cell reads keratinocyte transcripts; resolVI does NOT remove
# them here (its corrected KRT14/KRT5 rise as much as raw), so resolVI rows are descriptive only.
# Rules: (i) `leak` = keratinocytes' library-normalised mean ≥ half the T-cell mean; (ii) leak adds
# counts, so a *decrease* cannot come from it -- CD3E (should not move) checks that the offset does
# not dilute T-cell genes in the epidermis; (iii) an *increase* in a leak gene is claimed only if
# it survives adding the cell's own keratin read, log1p(KRT14 + KRT5), as a covariate (`raw|ker`).
F9_GENES = [g for g in CS.LAMINA + CS.RETENTION + CS.EGRESS + ["AQP3"] if g in adata.var_names]
F9_CTRL = [g for g in CS.KER_SPILL + ["CD3E"] if g in adata.var_names]
LEAK = {g: bool(MEAN_L.loc["Keratinocyte", g] >= 0.5 * MEAN_L.loc["T", g]) for g in F9_GENES + F9_CTRL}
print("T mean / keratinocyte mean (library-normalised), leak:",
      {g: f"{MEAN_L.loc['T', g]:.2f}/{MEAN_L.loc['Keratinocyte', g]:.2f}{' leak' if LEAK[g] else ''}"
       for g in F9_GENES + F9_CTRL})

EXPECT_28C = {"LMNA": +1}      # 28c row C, epidermal vs dermal malignant: lamina ratio up
T9 = min_piece(np.flatnonzero(IS_T & HAS_EPI))
lg_all = pd.Series(np.log1p(TOT30.to_numpy()))
lg_n = pd.Series(np.log(NCOUNT))
lg_ker = pd.Series(np.log1p(sum(gcounts(g) for g in CS.KER_SPILL if g in adata.var_names)))
AR = adata.obs.AspectRatio.to_numpy().astype(float)
LAREA = np.log(adata.obs.Area.to_numpy().astype(float))
rows = []
for lab, a, b in [("epi", "epidermis", "dermis"), ("DEJ", "DEJ", "dermis")]:
    idx = T9[np.isin(COMP[T9], [a, b])]
    term = (COMP == a).astype(float)
    for g in F9_GENES + F9_CTRL:
        ctrl = g in F9_CTRL
        rows += fits(gcounts(g), pd.DataFrame({"term": term, "log_all": lg_all}), idx, kind="glm",
                     contrast=lab, source="raw", gene=g, control=ctrl)
        if g not in CS.KER_SPILL:
            rows += fits(gcounts(g), pd.DataFrame({"term": term, "log_all": lg_all, "log_ker": lg_ker}), idx,
                         kind="glm", contrast=lab, source="raw|ker", gene=g, control=ctrl)
        v = rvi(g)
        if v is not None:
            rows += fits(np.log(v + 1e-3), pd.DataFrame({"term": term}), idx, kind="ols", contrast=lab,
                         source="resolVI", gene=g, control=ctrl)
    for name, y in [("AspectRatio", AR), ("log Area", LAREA)]:
        rows += fits(y, pd.DataFrame({"term": term, "log_n": lg_n}), idx, kind="ols", contrast=lab,
                     source="shape", gene=name, control=False)
F9_RAW = pd.DataFrame(rows)
F9_RAW["label"] = F9_RAW.gene
F9 = pd.concat([CS.robust_flags(d, ["gene"]).assign(contrast=c, source=s)
                for (c, s), d in F9_RAW.groupby(["contrast", "source"])], ignore_index=True)
F9["leak"] = F9.gene.map(LEAK).fillna(False).astype(bool)
F9["control"] = F9.control.astype(bool)
_ka = F9[F9.source == "raw|ker"].set_index(["contrast", "gene"])
F9["ker_adj_robust"] = [bool(_ka.robust.get((r.contrast, r.gene), False)
                             and np.sign(_ka.coef.get((r.contrast, r.gene), np.nan)) == np.sign(r.coef))
                        for r in F9.itertuples()]
F9["claim"] = (F9.robust & ~F9.control & F9.source.isin(["raw", "shape"])
               & ((F9.coef < 0) | ~F9.leak | F9.ker_adj_robust))
F9["expect_28c"] = F9.gene.map(EXPECT_28C)
F9.to_csv(tab28e("F9_epidermis_vs_dermis"), index=False)
F9_RAW.to_csv(tab28e("F9_per_slide"), index=False)
with pd.option_context("display.width", 220):
    print(F9[F9.contrast == "epi"][["source", "gene", "n_pos", "coef", "se", "q", "n_slides_agree", "robust", "leak",
                                    "ker_adj_robust", "claim"]].round(3).to_string(index=False))
    print("\nDEJ vs dermis, raw:")
    print(F9[(F9.contrast == "DEJ") & (F9.source == "raw")][["gene", "coef", "q", "n_slides_agree", "claim"]]
          .round(3).to_string(index=False))

# dose-response on keratinocyte contact
KBINS = pd.cut(N_KER20, [-0.5, 0.5, 2.5, 5.5, np.inf], labels=["0", "1-2", "3-5", "6+"])
DR = (pd.DataFrame({"slide": SLIDE[T9], "kbin": KBINS[T9], "AspectRatio": AR[T9],
                    **{g: gcounts(g)[T9] > 0 for g in F9_GENES + CS.KER_SPILL}})
      .groupby(["slide", "kbin"], observed=True).agg(["mean", "size"]))
DR.to_csv(tab28e("F9_keratinocyte_contact"))

# mal-hi vs mal-lo separately (pooled only)
if TX_OK:
    rows = []
    for grp in ["malhi", "mallo"]:
        idx = T9[np.isin(COMP[T9], ["epidermis", "dermis"]) & (MAL_GROUP[T9] == grp)]
        term = (COMP == "epidermis").astype(float)
        for g in F9_GENES:
            r = CS.niche_glm(gcounts(g)[idx], pd.DataFrame({"term": term[idx], "log_all": lg_all.iloc[idx].to_numpy()}),
                             LIBOFF[idx], FOV[idx], fe=PIECE[idx])
            rows.append({"group": grp, "gene": g, **r})
        r = CS.ols_fe(AR[idx], pd.DataFrame({"term": term[idx], "log_n": lg_n.iloc[idx].to_numpy()}), FOV[idx], fe=PIECE[idx])
        rows.append({"group": grp, "gene": "AspectRatio", **r})
    F9_MAL = pd.DataFrame(rows)
    F9_MAL.to_csv(tab28e("F9_by_malignancy"), index=False)
    print("\nepidermis vs dermis within mal-hi / mal-lo (pooled):")
    print(F9_MAL.pivot_table(index="gene", columns="group", values="coef").round(3).to_string())

fig, axes = plt.subplots(1, 3, figsize=(12, 3.1), gridspec_kw={"width_ratios": [2.2, 0.8, 1.1]})
d = F9_RAW[(F9_RAW.contrast == "epi") & (F9_RAW.source == "raw")]
labs = F9_GENES + F9_CTRL
pooled = CS.robust_flags(d, ["gene"])
CS.coef_panel(axes[0], pd.concat([d[d.slide != "pooled"], pooled]), labs, SLIDES, SLIDE_STAGE,
              title="T cells, epidermis vs dermis (raw; † keratinocyte leak, × keratin-adjusted)")
axes[0].set_xticklabels([f"{g}{' †' if LEAK[g] else ''}" for g in labs], rotation=60, ha="right", fontsize=6.5)
ka = F9[(F9.contrast == "epi") & (F9.source == "raw|ker")].set_index("gene")
for j, g in enumerate(labs):
    if g in ka.index and np.isfinite(ka.loc[g, "coef"]):
        axes[0].scatter(j + 0.3, ka.loc[g, "coef"], marker="x", s=14, color="#6a3d9a", zorder=4)
axes[0].scatter([], [], marker="x", s=14, color="#6a3d9a", label="+ own keratin read")
axes[0].legend(fontsize=5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
d = F9_RAW[F9_RAW.source == "shape"].assign(label=lambda x: x.gene + " (" + x.contrast + ")")
pooled = pd.concat([CS.robust_flags(dd, ["gene"]).assign(label=lambda x: x.gene + " (" + c + ")")
                    for c, dd in d.groupby("contrast")])
CS.coef_panel(axes[1], pd.concat([d[d.slide != "pooled"], pooled]),
              ["AspectRatio (epi)", "log Area (epi)", "AspectRatio (DEJ)", "log Area (DEJ)"], SLIDES, SLIDE_STAGE,
              title="cell shape vs dermis", ylabel="Δ (OLS)")
ax = axes[2]
for k, s in enumerate(SLIDES):
    if s in DR.index.get_level_values(0):
        sub = DR.loc[s]
        ax.plot(range(len(sub)), sub[("KLF2", "mean")].to_numpy(), marker=CS.SLIDE_MARKERS[k], ms=4, lw=1,
                color=CS.STAGE_COLORS[SLIDE_STAGE[s]], label=f"slide {s} KLF2")
        ax.plot(range(len(sub)), sub[("KRT14", "mean")].to_numpy(), marker=CS.SLIDE_MARKERS[k], ms=3, lw=0.8,
                ls=":", color=CS.STAGE_COLORS[SLIDE_STAGE[s]])
        ax.set_xticks(range(len(sub))); ax.set_xticklabels(sub.index.astype(str), fontsize=6.5)
ax.plot([], [], ls=":", color="#7f7f7f", label="KRT14 (spillover)")
ax.set_xlabel(f"keratinocyte neighbours within {R_TIGHT} µm", fontsize=7)
ax.set_ylabel("detection in T cells", fontsize=7); ax.tick_params(labelsize=6.5)
ax.legend(fontsize=5, frameon=False); _clean(ax)
_claims = F9[F9.claim & (F9.contrast == "epi") & (F9.source == "raw")]
_ar = F9[(F9.gene == "AspectRatio") & (F9.contrast == "epi")].iloc[0]
fig.suptitle("Intraepidermal vs dermal T cells: "
             + (", ".join(f"{g} {c:+.2f}" for g, c in zip(_claims.gene, _claims.coef)) or "no robust gene")
             + f" (log rate); shape unchanged (AspectRatio {_ar.coef:+.2f}, q {_ar.q:.2g})", fontsize=9, y=1.03)
D._finish(fig, fig28e("F9_motility_proxies"), True)
logfig("F9_motility_proxies", "epidermis vs dermis in T cells: residence/egress/LMNA genes (+ keratin-adjusted), shape, keratinocyte contact")
```

```
T mean / keratinocyte mean (library-normalised), leak: {'LMNA': '0.42/0.47 leak', 'ITGAE': '0.10/0.06 leak', 'CD69': '0.30/0.08', 'CXCR3': '0.18/0.07', 'CCR10': '0.22/0.07', 'CXCR6': '0.19/0.08', 'KLF2': '0.39/0.09', 'SELL': '0.09/0.03', 'CCR7': '0.37/0.08', 'CXCR4': '0.32/0.07', 'AQP3': '0.27/1.40 leak', 'KRT14': '0.28/11.02 leak', 'KRT5': '0.19/9.47 leak', 'CD3E': '0.68/0.11'}

 source        gene  n_pos   coef    se     q  n_slides_agree  robust  leak  ker_adj_robust  claim
    raw        LMNA   2222  0.310 0.155 0.126               2   False  True           False  False
    raw       ITGAE    670  0.497 0.215 0.077               2   False  True           False  False
    raw        CD69   1711  0.213 0.201 0.396               2   False False           False  False
    raw       CXCR3   1178  0.125 0.170 0.507               3   False False           False  False
    raw       CCR10   1456 -0.110 0.182 0.545               3   False False           False  False
    raw       CXCR6   1223 -0.300 0.267 0.396               2   False False           False  False
    raw        KLF2   2120 -0.810 0.189 0.000               4    True False            True   True
    raw        SELL    649 -0.270 0.277 0.403               3   False False           False  False
    raw        CCR7   2242 -0.505 0.167 0.013               3    True False            True   True
    raw       CXCR4   1900  0.161 0.138 0.396               3   False False           False  False
    raw        AQP3   1650  0.279 0.166 0.205               3   False  True           False  False
    raw       KRT14   1395  1.509 0.183   NaN               4   False  True           False  False
    raw        KRT5    970  1.675 0.197   NaN               4   False  True           False  False
    raw        CD3E   3410  0.033 0.069   NaN               2   False False           False  False
raw|ker        LMNA   2222  0.261 0.161 0.291               2   False  True           False  False
raw|ker       ITGAE    670  0.529 0.219 0.058               3   False  True           False  False
raw|ker        CD69   1711  0.230 0.206 0.362               2   False False           False  False
raw|ker       CXCR3   1178  0.115 0.181 0.578               4   False False           False  False
raw|ker       CCR10   1456 -0.154 0.191 0.511               3   False False           False  False
raw|ker       CXCR6   1223 -0.328 0.271 0.356               2   False False           False  False
raw|ker        KLF2   2120 -0.694 0.182 0.002               4    True False            True  False
raw|ker        SELL    649 -0.092 0.265 0.727               2   False False           False  False
raw|ker        CCR7   2242 -0.493 0.171 0.021               3    True False            True  False
raw|ker       CXCR4   1900  0.174 0.140 0.356               3   False False           False  False
raw|ker        AQP3   1650  0.236 0.167 0.347               3   False  True           False  False
raw|ker        CD3E   3410  0.025 0.074   NaN               2   False False           False  False
resolVI        LMNA   9540  0.141 0.048 0.006               2   False  True           False  False
resolVI       ITGAE   9540  0.340 0.082 0.000               4    True  True           False  False
resolVI        CD69   9540  0.138 0.052 0.012               4    True False           False  False
resolVI       CXCR3   9540 -0.024 0.054 0.721               2   False False           False  False
resolVI       CCR10   9540  0.080 0.045 0.088               4   False False           False  False
resolVI       CXCR6   9540  0.166 0.056 0.006               4    True False           False  False
resolVI        KLF2   9540 -0.380 0.069 0.000               4    True False            True  False
resolVI        SELL   9540 -0.200 0.032 0.000               4    True False           False  False
resolVI        CCR7   9540 -0.228 0.106 0.043               3    True False            True  False
resolVI       CXCR4   9540  0.011 0.044 0.793        
... [1326 more chars truncated]
```

```python  # cell 7
# ============================================================================
# F10 — is there more protease around T cells?
# ============================================================================
# 28d F4 asked whether each stromal cell expresses more MMP with more T cells nearby (no). A
# larger degradome next to T cells can also come from *which* cells are there. Three readouts:
# (a) composition: MMP counts carried by non-T cells within R of each T cell ("field"), against
#     anchor cells in the same piece × depth bin × local-density tertile (CS.field_vs_null);
#     log2(Σ observed / Σ expected) per slide and pooled (FOV bootstrap CI). Null anchors are
#     neither T nor the gene's producer lineage (F7 PARTNER): producers cluster with producers,
#     so a fibroblast-anchored null would inflate the MMP2 expectation. Anchors: all T, then
#     mal-hi and mal-lo T separately (same null).
# (b) per cell, corrected: 28d F4's design on resolVI-corrected stromal MMP (OLS on log), next to
#     28d's raw pooled coefficient.
# (c) by T state: stromal count ~ [≥1 mal-hi T within 30 µm] + log1p(T within 30) + log1p(all
#     within 30) + piece (Poisson, offset); the same with mal-lo. 28c predicts protease-
#     independent motility for the clone: no, or a negative, mal-hi term.
F10_GENES = [g for g in ["MMP2", "MMP9", "MMP14", "MMP12"] if g in adata.var_names]
F10_SETS = {"Fibroblast": ["MMP2", "MMP14"], "Myeloid": ["MMP9", "MMP12", "MMP14"]}
DBIN = np.select([COMP == "epidermis", DEPTH <= 50, DEPTH <= 100, DEPTH <= 200, DEPTH <= 400],
                 ["epi", "0-50", "50-100", "100-200", "200-400"], ">400").astype(object)
rng = np.random.default_rng(0)


def dens_bin(tot):
    """Local-density tertile within piece (rank-based: ties and small pieces cannot break it)."""
    r = pd.Series(np.asarray(tot, float)).groupby(PIECE).rank(method="first", pct=True)
    return np.minimum((r * 3).astype(int), 2).astype(str).to_numpy()


def log2_oe(F, boot=200):
    out = {"pooled": np.log2(F.obs.sum() / F.exp.sum())}
    for s in SLIDES:
        f = F[SLIDE[F.cell] == s]
        out[s] = np.log2(f.obs.sum() / f.exp.sum()) if len(f) >= 30 and f.exp.sum() > 0 else np.nan
    g = F.assign(fov=FOV[F.cell]).groupby("fov")[["obs", "exp"]].sum()
    bs = [np.log2(b.obs.sum() / b.exp.sum())
          for b in (g.sample(len(g), replace=True, random_state=int(rng.integers(1e9))) for _ in range(boot))]
    out["lo"], out["hi"] = np.percentile(bs, [2.5, 97.5])
    return out


rows = []
POOL = ~IS_T & HAS_EPI & (LIN != "Unassigned")
anchors = {"all T": IS_T & HAS_EPI}
if TX_OK:
    anchors |= {"mal-hi T": IS_T & HAS_EPI & (MAL_GROUP == "malhi"), "mal-lo T": IS_T & HAS_EPI & (MAL_GROUP == "mallo")}
for R, G, TOT in [(R_CONTACT, G30, TOT30), (R_NICHE, G50, TOT50)]:
    strata = pd.Series(PIECE) + "|" + pd.Series(DBIN) + "|" + pd.Series(dens_bin(TOT))
    for g in F10_GENES:
        vals = gcounts(g) * (~IS_T)
        pool = POOL & (LIN != PARTNER.get(g, ""))
        for an, am in anchors.items():
            F = CS.field_vs_null(G, vals, am, pool, strata)
            rows.append({"radius": R, "gene": g, "anchor": an, "null_excludes": PARTNER.get(g, ""),
                         "n_anchor": len(F), **log2_oe(F)})
F10A = pd.DataFrame(rows)
F10A["n_slides_up"] = (F10A[SLIDES] > 0).sum(1)
F10A.to_csv(tab28e("F10a_protease_field"), index=False)
print("(a) log2 obs/exp non-T protease field around T cells (depth- and density-matched null):")
print(F10A.round(3).to_string(index=False))

# (b) resolVI-corrected per-cell F4
F10B = pd.DataFrame()
if HAS_RVI:
    X_all = pd.DataFrame({"log_T": np.log1p(nb_T(NB50)), "log_all": np.log1p(TOT50.to_numpy())})
    rows = []
    for lin, genes in F10_SETS.items():
        idx = min_piece(np.flatnonzero(LIN == lin))
        for g in [g for g in genes if rvi(g) is not None]:
            rows += fits(np.log(rvi(g) + 1e-3), X_all, idx, kind="ols", lineage=lin, gene=g, control=False)
    F10B = CS.robust_flags(pd.DataFrame(rows), ["lineage", "gene"])
    p28d = cfg.TAB_DIR / "deg28d_F4_niche_glm.csv"
    if p28d.exists():
        raw = pd.read_csv(p28d)[["lineage", "gene", "coef", "q"]].rename(columns={"coef": "coef_28d_raw", "q": "q_28d_raw"})
        F10B = F10B.merge(raw, on=["lineage", "gene"], how="left")
    F10B.to_csv(tab28e("F10b_resolvi_F4"), index=False)
    print("\n(b) 28d F4 on resolVI-corrected stromal MMP (OLS on log), 28d raw Poisson beside:")
    print(F10B[[c for c in ["lineage", "gene", "n", "coef", "se", "q", "n_slides_agree", "robust", "coef_28d_raw",
                            "q_28d_raw"] if c in F10B]].round(3).to_string(index=False))
else:
    print("\n(b) skipped: no resolVI output")

# (c) by T state
if TX_OK:
    rows = []
    for grp in ["malhi", "mallo"]:
        n_g = G30 @ (IS_T & (MAL_GROUP == grp)).astype(np.float32)
        X = pd.DataFrame({"near": (n_g > 0).astype(float), "log_T": np.log1p(nb_T(NB30)),
                          "log_all": np.log1p(TOT30.to_numpy())})
        for lin, genes in F10_SETS.items():
            idx = min_piece(np.flatnonzero(LIN == lin))
            for g in [g for g in genes if g in adata.var_names]:
                rows += fits(gcounts(g), X, idx, kind="glm", group=grp, lineage=lin, gene=g, control=False,
                             label=f"{lin[:5]} {g} · {grp}")
    F10C_RAW = pd.DataFrame(rows)
    F10C = CS.robust_flags(F10C_RAW, ["group", "lineage", "gene"])
    F10C.to_csv(tab28e("F10c_stroma_near_malhi"), index=False)
    print("\n(c) stromal MMP next to mal-hi vs mal-lo T cells (beyond T density):")
    print(F10C[["group", "lineage", "gene", "n", "n_pos", "coef", "se", "q", "n_slides_agree", "robust"]]
          .round(3).to_string(index=False))
else:
    print("\n(c) skipped: no transfer groups")

fig, axes = plt.subplots(1, 2 if TX_OK else 1, figsize=(10 if TX_OK else 5.5, 3.1), squeeze=False)
ax = axes[0, 0]
labs = []
for j, (key, d) in enumerate(F10A.groupby(["gene", "anchor", "radius"], sort=False)):
    r = d.iloc[0]
    ax.errorbar(j, r.pooled, yerr=[[r.pooled - r.lo], [r.hi - r.pooled]], fmt="o", color="k", ms=3.5, lw=1, capsize=2)
    for k, s in enumerate(SLIDES):
        if np.isfinite(r[s]):
            ax.scatter(j - 0.24 + 0.16 * k, r[s], s=11, color=CS.STAGE_COLORS[SLIDE_STAGE[s]],
                       marker=CS.SLIDE_MARKERS[k], edgecolor="none")
    labs.append(f"{key[0]} · {key[1]} · {key[2]}µm")
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=70, ha="right", fontsize=5.5)
ax.set_ylabel("log2 obs / exp protease field", fontsize=7); ax.tick_params(labelsize=6.5); _clean(ax)
ax.set_title("non-T protease around T cells vs matched cells", fontsize=8)
if TX_OK:
    labs_c = [f"{l[:5]} {g} · {grp}" for grp in ["malhi", "mallo"] for l, gs in F10_SETS.items() for g in gs]
    CS.coef_panel(axes[0, 1], pd.concat([F10C_RAW[F10C_RAW.slide != "pooled"], F10C]), labs_c, SLIDES, SLIDE_STAGE,
                  title="stromal MMP next to mal-hi / mal-lo T (30 µm)")
_a = F10A[(F10A.anchor == "all T") & (F10A.radius == R_CONTACT)].set_index("gene")
fig.suptitle("Protease field around T cells vs depth/density-matched cells (30 µm): "
             + ", ".join(f"{g} {_a.loc[g, 'pooled']:+.2f} ({int(_a.loc[g, 'n_slides_up'])}/4 up)" for g in _a.index),
             fontsize=9, y=1.03)
D._finish(fig, fig28e("F10_degradome_near_T"), True)
logfig("F10_degradome_near_T", "log2 obs/exp non-T MMP field around T (all / mal-hi / mal-lo); stromal MMP next to mal-hi vs mal-lo T")
```

```
(a) log2 obs/exp non-T protease field around T cells (depth- and density-matched null):
 radius  gene anchor null_excludes  n_anchor  pooled      3      4      7      8     lo     hi  n_slides_up
     30  MMP2  all T    Fibroblast     12969  -0.276  0.038 -0.848 -0.290 -0.271 -0.394 -0.131            1
     30  MMP9  all T       Myeloid     12749  -0.150 -0.382 -0.016 -0.198 -0.101 -0.280 -0.012            0
     30 MMP14  all T    Fibroblast     12969  -0.222 -0.210 -0.307 -0.260 -0.190 -0.331 -0.131            0
     30 MMP12  all T       Myeloid     12749   0.205  0.446  0.057 -0.052  0.392 -0.044  0.450            3
     50  MMP2  all T    Fibroblast     12902  -0.241 -0.027 -0.614 -0.238 -0.244 -0.348 -0.114            0
     50  MMP9  all T       Myeloid     12735  -0.138 -0.178 -0.125 -0.208 -0.080 -0.295  0.019            0
     50 MMP14  all T    Fibroblast     12902  -0.186 -0.101 -0.298 -0.231 -0.158 -0.282 -0.091            0
     50 MMP12  all T       Myeloid     12735   0.140  0.410 -0.525 -0.003  0.294 -0.100  0.382            2


(b) 28d F4 on resolVI-corrected stromal MMP (OLS on log), 28d raw Poisson beside:
   lineage  gene     n   coef    se     q  n_slides_agree  robust  coef_28d_raw  q_28d_raw
Fibroblast  MMP2 14123 -0.217 0.029 0.000               4    True        -0.121      0.057
Fibroblast MMP14 14123 -0.192 0.019 0.000               4    True        -0.064      0.111
   Myeloid  MMP9  9973 -0.031 0.038 0.421               1   False         0.031      0.638
   Myeloid MMP12  9973 -0.070 0.037 0.075               3   False        -0.013      0.904
   Myeloid MMP14  9973 -0.121 0.017 0.000               4    True        -0.137      0.003

(c) skipped: no transfer groups

<Figure size 550x310 with 1 Axes>
```

F11 and F12 read the Hotspot / Harreman job's tables. The job needs the §R parquet, so on a first pass they print "skipped": run §R, submit `jobs/run_cosmx_harreman.sh`, then re-run F11, F12 and §R.

```python  # cell 9
# ============================================================================
# F11 — Hotspot: do proteases share spatial modules / local correlation with T-cell genes?
# ============================================================================
# From jobs/run_cosmx_harreman.py (pxlgn env; needs this notebook's §R parquet): DANB local
# autocorrelation of every panel gene on a 10-NN graph within slide, pairwise local-correlation
# Z for the top spatial genes + focus genes, modules, and the focus pairs re-run per slide.
# Local correlation of gene a with gene b = a in a cell co-varies with b in its graph neighbours,
# so it is a composition-level readout (like F10a), not a per-cell one (like 28d F4).
HS_LCZ, HS_MOD, HS_SLIDE = tab28e("hotspot_lcz"), tab28e("hotspot_modules"), tab28e("hotspot_focus_per_slide")
MOD_PQ = CS.S1_DIR / "processed" / "cosmx_s1_hotspot_module_scores.parquet"
TOP11 = pd.Series(dtype=float)
if HS_LCZ.exists():
    LCZ = pd.read_csv(HS_LCZ, index_col=0)
    MODS = pd.read_csv(HS_MOD, index_col=0).iloc[:, 0] if HS_MOD.exists() else pd.Series(dtype=int)
    PS = pd.read_csv(HS_SLIDE) if HS_SLIDE.exists() else pd.DataFrame(columns=["slide", "gene_a", "gene_b", "z"])
    rows_g = [g for g in ["MMP2", "MMP9", "MMP14", "MMP12", "MMP1", "TIMP1", "CTSB", "CTSD", "GZMB"] if g in LCZ.index]
    cols_g = [g for g in ["CD3E", "CD3D", "CD2", "CD3G"] + sorted({g for p in CS.AXIS_PAIRS for g in p})
              if g in LCZ.columns]
    Zs = LCZ.loc[rows_g, cols_g]
    n_up = (PS.assign(up=PS.z > 1.96).pivot_table(index="gene_a", columns="gene_b", values="up", aggfunc="sum")
            .reindex(index=rows_g, columns=cols_g))
    Zs.to_csv(tab28e("F11_protease_T_lcz"))
    print("pooled local-correlation Z, proteases × T/axis genes:")
    print(Zs.round(1).to_string())
    if len(MODS):
        print("\nmodules:", MODS[MODS != -1].value_counts().sort_index().to_dict())
        for g in rows_g + ["CD3E", "CD2"]:
            if g in MODS.index:
                m = MODS[g]
                print(f"  {g}: module {m}" + ("" if m == -1 else f" with {sorted(MODS.index[MODS == m])[:20]}"))
    if MOD_PQ.exists():
        MS = pd.read_parquet(MOD_PQ).reindex(adata.obs_names)
        key = np.where(IS_T, np.char.add("T · ", COMP.astype(str)), LIN)
        MSL = MS.groupby(key).mean()
        MSL.to_csv(tab28e("F11_module_score_by_group"))
        print("\nmean module score by lineage (T split by compartment):")
        print(MSL.round(2).to_string())

    TOP11 = Zs[[c for c in cols_g if c in ["CD3E", "CD3D", "CD2", "CD3G"]]].stack().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(max(5, 0.42 * len(cols_g) + 2), 0.4 * len(rows_g) + 1.4))
    lim = max(3, np.nanpercentile(np.abs(Zs.to_numpy()), 95))
    im = ax.imshow(Zs.to_numpy().clip(-lim, lim), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(rows_g)):
        for j in range(len(cols_g)):
            n = n_up.iat[i, j]
            if np.isfinite(n):
                ax.text(j, i, int(n), ha="center", va="center", fontsize=6)
    ax.set_xticks(range(len(cols_g))); ax.set_xticklabels(cols_g, rotation=90, fontsize=6.5, fontstyle="italic")
    ax.set_yticks(range(len(rows_g))); ax.set_yticklabels(rows_g, fontsize=6.5, fontstyle="italic")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="local-correlation Z (pooled)").ax.tick_params(labelsize=6)
    ax.set_title(f"Strongest protease–T-gene local correlation: {TOP11.index[0][0]}–{TOP11.index[0][1]} "
                 f"Z {TOP11.iloc[0]:.1f}\n(number = slides with Z > 1.96)", fontsize=8.5)
    D._finish(fig, fig28e("F11_hotspot_protease_T"), True)
    logfig("F11_hotspot_protease_T", "Hotspot local-correlation Z, proteases × T / axis genes; per-slide count Z>1.96")
else:
    print(f"F11 skipped: no {HS_LCZ.name} (run jobs/run_cosmx_harreman.sh after §R)")
```

```
pooled local-correlation Z, proteases × T/axis genes:
       CD3E  CD3D   CD2  CD3G  CD40  CD40LG  CDH1  CXCL10  CXCL9  CXCR3  ITGAE  LTBR  TGFB1  TGFBR2  TNFRSF14  TNFSF14
MMP2    3.5   2.5   6.8   1.2   2.9     0.9 -18.3   -12.1   -1.7    3.4    2.1  -0.6    7.3    11.2      10.6      3.3
MMP9    9.4   9.2  11.9   7.5   6.1     5.7 -13.1    -3.7    9.4    8.2    2.9  -1.9   10.9     7.4       9.2      6.4
MMP14  -1.9  -1.4  -0.3  -1.8  -3.0    -2.4  -7.0    -6.3   -3.9    1.2    1.5   1.7    1.2     4.2       5.3      1.4
MMP12   5.7   3.4   6.4   3.2  13.8     2.4  -0.4     0.8    1.0    3.6    2.5  -2.9    6.1    -0.6       4.0      3.1
MMP1    4.7   3.9   8.8   3.7   3.8     2.3  -9.3    -5.9    1.0    5.1    3.4  -1.5    3.9     5.9       9.0      4.5
TIMP1  15.7  16.9  14.5  10.8   5.3     7.9 -17.0    -5.9    6.7    7.5    2.5  -1.5   12.1     9.1       9.7      8.9
CTSD   -9.8  -9.7  -6.5  -6.9  -5.6    -8.4  10.6   -11.0  -10.6   -4.2   -1.8   2.4   -5.3    -3.7      -4.8     -6.8
GZMB    4.8   2.9   2.6   4.0   4.2     0.7  -4.2    10.6    8.4    7.5    3.3  -1.1    4.8     2.2       1.6      3.1

modules: {1: 42, 2: 47, 3: 64, 4: 27, 5: 29, 6: 28, 7: 10, 8: 10, 9: 13}
  MMP2: module 1 with ['ACKR1', 'ACTA2', 'APOD', 'BGN', 'CCDC80', 'CCL21', 'CD63', 'CFD', 'COL15A1', 'COL18A1', 'COL1A1', 'COL1A2', 'COL3A1', 'COL4A1', 'COL4A2', 'COL6A1', 'COL6A2', 'COL6A3', 'CTSG', 'DCN']
  MMP9: module 7 with ['CCL13', 'CCL18', 'CD14', 'CXCL12', 'IGHA1', 'IGHG1', 'IGHG2', 'IGKC', 'JCHAIN', 'MMP9']
  MMP14: module 1 with ['ACKR1', 'ACTA2', 'APOD', 'BGN', 'CCDC80', 'CCL21', 'CD63', 'CFD', 'COL15A1', 'COL18A1', 'COL1A1', 'COL1A2', 'COL3A1', 'COL4A1', 'COL4A2', 'COL6A1', 'COL6A2', 'COL6A3', 'CTSG', 'DCN']
  MMP12: module 5 with ['B2M', 'C1QA', 'C1QB', 'C1QC', 'CCL2', 'CCL5', 'CD1C', 'CD40', 'CD74', 'CDKN1A', 'CST7', 'CXCL10', 'CXCL9', 'GSN', 'HLA-DPA1', 'HLA-DPB1', 'HLA-DQA1', 'HLA-DQB1/2', 'HLA-DRA', 'HLA-DRB']
  MMP1: module 1 with ['ACKR1', 'ACTA2', 'APOD', 'BGN', 'CCDC80', 'CCL21', 'CD63', 'CFD', 'COL15A1', 'COL18A1', 'COL1A1', 'COL1A2', 'COL3A1', 'COL4A1', 'COL4A2', 'COL6A1', 'COL6A2', 'COL6A3', 'CTSG', 'DCN']
  TIMP1: module 1 with ['ACKR1', 'ACTA2', 'APOD', 'BGN', 'CCDC80', 'CCL21', 'CD63', 'CFD', 'COL15A1', 'COL18A1', 'COL1A1', 'COL1A2', 'COL3A1', 'COL4A1', 'COL4A2', 'COL6A1', 'COL6A2', 'COL6A3', 'CTSG', 'DCN']
  CTSD: module 9 with ['ARG1', 'AZGP1', 'CD24', 'CHI3L1', 'CLDN4', 'CTSD', 'IL1R2', 'KRT23', 'KRT80', 'RORA', 'SERPINA3', 'SLPI', 'SPP1']
  GZMB: module 6 with ['ANXA1', 'BTG1', 'CCL17', 'CXCL13', 'ELANE', 'FAU', 'GZMB', 'H2AZ1', 'H4C3', 'HMGB2', 'HMGN2', 'ITGAE', 'LMNA', 'MMP19', 'MMP7', 'MS4A1', 'PPIA', 'PRTN3', 'RPL21', 'RPL22']
  CD3E: module 3 with ['ALOX5AP', 'ARHGDIB', 'BASP1', 'CCL19', 'CCL22', 'CCR7', 'CD2', 'CD27', 'CD37', 'CD3D', 'CD3E', 'CD3G', 'CD40LG', 'CD48', 'CD52', 'CD53', 'CD69', 'CD83', 'CLU', 'COTL1']
  CD2: module 3 with ['ALOX5AP', 'ARHGDIB', 'BASP1', 'CCL19', 'CCL22', 'CCR7', 'CD2', 'CD27', 'CD37', 'CD3D', 'CD3E', 'CD3G', 'CD40LG', 'CD48', 'CD52', 'CD53', 'CD69', 'CD83', 'CLU', 'COTL1']

mean module score by lineage (T split by compartment):
                  module_Module 1  module_Module 2  module_Module 3  module_Module 4  module_Module 5  module_Module 6  module_Module 7  module_Module 8  module_Module 9
Fibroblast                   1.26            -0.50             0.03            -0.26            -0.13            -0.16             0.13             0.15             0.00
Keratinocyte                -0.81             1.03            -1.08             0.78            -0.38            -0.21            -0.23            -0.26             0.07
Myeloid                      0.41            -0.66             0.40            -0.46             1.02            -0.05             0.22             0.15            -0.06
T · DEJ                      0.41            -0.82             1.98            -0.69             0.28             0.56             0.06             0.13            -0.10
T · dermis             
... [691 more chars truncated]
```

```python  # cell 10
# ============================================================================
# F12 — Harreman: cell-type-aware T ↔ stroma / keratinocyte LR communication
# ============================================================================
# Harreman's parametric cell-type-aware test (CellChat human LR DB restricted to the panel) on the
# same within-slide 10-NN graph: for gene pair (a, b) and cell types (t, u), does a in t co-vary
# with b in neighbouring u beyond expression alone. T cells enter as T_epidermis / T_DEJ /
# T_dermis (`comp`) and as T_malhi / mid / lo (`mal`); partners Fibroblast / Myeloid /
# Keratinocyte, both directions. The expression filter is ≥5% detection per cell type (Harreman's
# built-in 20% would drop nearly every T-cell ligand on CosMx). 28b axes are marked.
AXIS_SET = {tuple(p) for p in CS.AXIS_PAIRS}
F12 = {}
for tag in ["comp", "mal"]:
    p = tab28e(f"harreman_{tag}_gp")
    if tag == "mal" and not TX_OK:
        print("F12 [mal] skipped: transfer score failed the in-situ gates (F8)")
        continue
    if not p.exists():
        print(f"F12 [{tag}] skipped: no {p.name} (jobs/run_cosmx_harreman.sh)")
        continue
    d = pd.read_csv(p)
    d = d[d["Cell Type 1"].str.startswith("T_") | d["Cell Type 2"].str.startswith("T_")].copy()
    d["pair"] = d["Gene 1"].astype(str) + "→" + d["Gene 2"].astype(str)
    d["cts"] = d["Cell Type 1"] + "→" + d["Cell Type 2"]
    d["axis_28b"] = [(a, b) in AXIS_SET for a, b in zip(d["Gene 1"].astype(str), d["Gene 2"].astype(str))]
    F12[tag] = d
    sig = d[d.Z_FDR < 0.05]
    print(f"[{tag}] T-involving tests: {len(d):,}; FDR < 0.05: {len(sig):,}")
    print(sig.groupby("cts").size().sort_values(ascending=False).head(12).to_dict())
    ax_rows = d[d.axis_28b][["cts", "pair", "Z", "Z_FDR"]]
    print(f"[{tag}] 28b axis pairs:\n" + (ax_rows.round(3).to_string(index=False) if len(ax_rows) else "  none tested"))
    d.to_csv(tab28e(f"F12_harreman_{tag}_T"), index=False)

if "comp" in F12:
    d = F12["comp"]
    top = (d[d.Z_FDR < 0.05].groupby("pair").Z.max().sort_values(ascending=False).head(25).index.tolist()
           or d.groupby("pair").Z.max().sort_values(ascending=False).head(15).index.tolist())
    top = list(dict.fromkeys(top + d[d.axis_28b].pair.unique().tolist()))
    M = d[d.pair.isin(top)].pivot_table(index="pair", columns="cts", values="Z", aggfunc="max").reindex(top)
    order = [c for t in ["T_epidermis", "T_DEJ", "T_dermis"] for c in M.columns if c.startswith(t + "→")] + \
            [c for t in ["T_epidermis", "T_DEJ", "T_dermis"] for c in M.columns if c.endswith("→" + t)]
    M = M[order]
    fig, ax = plt.subplots(figsize=(0.32 * M.shape[1] + 3, 0.22 * M.shape[0] + 1.6))
    lim = max(3, np.nanpercentile(np.abs(M.to_numpy()), 95))
    im = ax.imshow(M.to_numpy().clip(-lim, lim), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    S = d.pivot_table(index="pair", columns="cts", values="Z_FDR", aggfunc="min").reindex(index=top, columns=order)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if S.iat[i, j] < 0.05:
                ax.text(j, i, "·", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(order, rotation=90, fontsize=6)
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels([f"{p}{' (28b)' if p in set(d[d.axis_28b].pair) else ''}" for p in top], fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Harreman Z").ax.tick_params(labelsize=6)
    n_epi = int(((d.Z_FDR < 0.05) & d.cts.str.contains("T_epidermis")).sum())
    n_der = int(((d.Z_FDR < 0.05) & d.cts.str.contains("T_dermis")).sum())
    ax.set_title(f"Significant LR tests involving T cells: epidermal {n_epi}, dermal {n_der} (· FDR < 0.05)",
                 fontsize=8.5)
    D._finish(fig, fig28e("F12_harreman_T"), True)
    logfig("F12_harreman_T", "Harreman cell-type-aware LR Z, T (by compartment) ↔ fibro / myeloid / keratinocyte")
```

```
[comp] T-involving tests: 12,482; FDR < 0.05: 287
{'T_dermis→Fibroblast': 50, 'Fibroblast→T_dermis': 46, 'Keratinocyte→T_dermis': 21, 'T_epidermis→Keratinocyte': 18, 'T_dermis→Keratinocyte': 17, 'T_dermis→Myeloid': 16, 'Myeloid→T_dermis': 16, 'T_DEJ→Fibroblast': 15, 'Keratinocyte→T_epidermis': 15, 'Fibroblast→T_DEJ': 14, 'Keratinocyte→T_DEJ': 12, 'T_epidermis→Fibroblast': 11}
[comp] 28b axis pairs:
                     cts             pair      Z  Z_FDR
        T_DEJ→Fibroblast       ITGAE→CDH1 -0.644  0.979
        T_DEJ→Fibroblast      CD40LG→CD40 -1.908  0.999
        T_DEJ→Fibroblast     TGFB1→TGFBR2  0.352  0.979
        T_DEJ→Fibroblast     TNFSF14→LTBR -0.946  0.981
        T_DEJ→Fibroblast TNFSF14→TNFRSF14 -0.945  0.981
        T_DEJ→Fibroblast      CXCL9→CXCR3  3.336  0.024
        T_DEJ→Fibroblast     CXCL10→CXCR3  1.953  0.381
        Fibroblast→T_DEJ      CD40LG→CD40  0.587  0.974
        Fibroblast→T_DEJ     TGFB1→TGFBR2  2.639  0.120
        Fibroblast→T_DEJ     TNFSF14→LTBR  0.198  0.979
        Fibroblast→T_DEJ TNFSF14→TNFRSF14  1.424  0.661
        Fibroblast→T_DEJ      CXCL9→CXCR3  4.344  0.001
        Fibroblast→T_DEJ     CXCL10→CXCR3  2.903  0.070
           T_DEJ→Myeloid       ITGAE→CDH1 -0.489  0.979
           T_DEJ→Myeloid      CD40LG→CD40  0.967  0.858
           T_DEJ→Myeloid     TGFB1→TGFBR2  1.793  0.459
           T_DEJ→Myeloid     TNFSF14→LTBR  0.245  0.979
           T_DEJ→Myeloid TNFSF14→TNFRSF14 -0.078  0.979
           T_DEJ→Myeloid      CXCL9→CXCR3  1.617  0.566
           T_DEJ→Myeloid     CXCL10→CXCR3  0.574  0.976
           Myeloid→T_DEJ       ITGAE→CDH1 -1.655  0.997
           Myeloid→T_DEJ      CD40LG→CD40 -0.171  0.979
           Myeloid→T_DEJ     TGFB1→TGFBR2  1.364  0.698
           Myeloid→T_DEJ     TNFSF14→LTBR -0.474  0.979
           Myeloid→T_DEJ TNFSF14→TNFRSF14 -0.996  0.983
           Myeloid→T_DEJ      CXCL9→CXCR3  3.528  0.014
           Myeloid→T_DEJ     CXCL10→CXCR3  3.673  0.009
      T_DEJ→Keratinocyte       ITGAE→CDH1  1.177  0.777
      T_DEJ→Keratinocyte      CD40LG→CD40 -0.130  0.979
      T_DEJ→Keratinocyte     TGFB1→TGFBR2 -0.209  0.979
      T_DEJ→Keratinocyte     TNFSF14→LTBR  0.002  0.979
      T_DEJ→Keratinocyte TNFSF14→TNFRSF14  1.871  0.423
      T_DEJ→Keratinocyte      CXCL9→CXCR3  0.023  0.979
      T_DEJ→Keratinocyte     CXCL10→CXCR3  0.577  0.975
      Keratinocyte→T_DEJ       ITGAE→CDH1  0.074  0.979
      Keratinocyte→T_DEJ      CD40LG→CD40  0.194  0.979
      Keratinocyte→T_DEJ     TGFB1→TGFBR2 -2.485  1.000
      Keratinocyte→T_DEJ     TNFSF14→LTBR  0.377  0.979
      Keratinocyte→T_DEJ TNFSF14→TNFRSF14 -1.151  0.985
      Keratinocyte→T_DEJ      CXCL9→CXCR3  2.314  0.217
      Keratinocyte→T_DEJ     CXCL10→CXCR3  1.055  0.834
     T_dermis→Fibroblast       ITGAE→CDH1 -0.232  0.979
     T_dermis→Fibroblast      CD40LG→CD40  0.268  0.979
     T_dermis→Fibroblast     TGFB1→TGFBR2  0.212  0.979
     T_dermis→Fibroblast     TNFSF14→LTBR  0.254  0.979
     T_dermis→Fibroblast TNFSF14→TNFRSF14  1.333  0.708
     T_dermis→Fibroblast      CXCL9→CXCR3  0.357  0.979
     T_dermis→Fibroblast     CXCL10→CXCR3 -1.356  0.990
     Fibroblast→T_dermis      CD40LG→CD40 -3.097  1.000
     Fibroblast→T_dermis     TGFB1→TGFBR2  1.886  0.416
     Fibroblast→T_dermis     TNFSF14→LTBR -1.340  0.989
     Fibroblast→T_dermis TNFSF14→TNFRSF14 -1.505  0.994
     Fibroblast→T_dermis      CXCL9→CXCR3  4.305  0.001
     Fibroblast→T_dermis     CXCL10→CXCR3 -1.202  0.986
        T_dermis→Myeloid       ITGAE→CDH1 -0.445  0.979
        T_dermis→Myeloid      CD40LG→CD40  3.063  0.048
        T_dermis→Myeloid     TGFB1→TGFBR2  1.870  0.423
        T_dermis→Myeloid     TNFSF14→LTBR -0.109  0.979
        T_dermis→Myeloid TNFSF14→TNFRSF14 -0.774  0.979
        T_dermis→Myeloid      CXCL9→CXCR3  3.965  0.004
        T_dermis→Myeloid     CXCL10→CXCR3  1.104  0.812
        Myeloid→T_dermis       ITGAE→CDH1 -2.302  1.000
        Myeloid→T_dermis      CD40LG→CD40 -1.397  0.992
        Myeloi
... [3444 more chars truncated]
```

```python  # cell 11
# ============================================================================
# §R  Outputs + summary
# ============================================================================
pd.DataFrame({"slide": SLIDE, "piece": PIECE, "fov_key": FOV, "lineage_28d": LIN28, "lineage_final": LIN,
              "t_sub": TSUB, "compartment": COMP, "ker_frac_30um": KFRAC, "pautrier": PAUT, "depth_um": DEPTH,
              "n_ker_20um": N_KER20, "mal_score": MAL_SCORE, "mal_group": MAL_GROUP.astype(str),
              "nCount_RNA": NCOUNT}, index=adata.obs_names).to_parquet(CS.CELLS_28E)
pd.DataFrame(FIG_INDEX).to_csv(tab28e("figure_index"), index=False)
print("cell table →", CS.CELLS_28E)
print(pd.DataFrame(FIG_INDEX).to_string(index=False))

print("\n" + "=" * 90 + "\nCAVEATS (read first)\n" + "=" * 90)
print("- 28c amoeboid programs are off-panel (AQP3, LMNA of 37 genes): F9 reads shape / LMNA / residence-egress proxies")
print(f"- malignancy = transfer score, group-level only: held-out AUC gate {'passed' if TX_GATE else 'not passed / no model'}"
      + (f" ({TX['cv']['auc_overall']:.2f})" if TX else "")
      + ("" if not TX else f"; in situ CD8<CD4 {CD8_OK}, piece specificity {PIECE_OK} -> mal splits "
                          f"{'used' if TX_OK else 'NOT used'}"))
print(f"- resolVI {'used' if HAS_RVI else 'not run'}: raw T-cell MMP includes neighbour spillover (upper bound)")
print(f"- stage = slide: patch {[s for s in SLIDES if SLIDE_STAGE[s] == 'patch']} vs plaque "
      f"{[s for s in SLIDES if SLIDE_STAGE[s] == 'plaque']} → described, never tested")

print("\n" + "=" * 90 + "\nRESULTS\n" + "=" * 90)
print(f"C   {EPI_SHARE:.0%} of T cells (pieces with epidermis) are intraepidermal; {int(PAUT.sum())} Pautrier-like")
print(f"F7  T cells: {t_cells:.1%} of cells, {t_lib:.1%} of library; share of " +
      ", ".join(f"{g} {_p.loc[g, 'T']:.1%}" for g in F7_HEAD))
_r7 = F7B[F7B.robust_T_lower]
print("    robust T < stroma (per piece):", {f"{r.source} {r.gene} vs {r.vs[:5]}": round(r.log2_T_over_other, 2)
                                             for r in _r7.itertuples()} or "none")
if HAS_TX:
    print(f"F8  score epidermis vs dermis: {_e.coef:+.2f} logit (q {_e.q:.2g}, {int(_e.n_slides_agree)}/4)")
_c9 = F9[F9.claim & (F9.contrast == "epi")]
print("F9  robust epidermis-vs-dermis proxies in T cells:",
      {f"{r.source} {r.gene}": round(r.coef, 3) for r in _c9.itertuples()} or "none")
print("    all (raw / shape):", {f"{r.gene}": f"{r.coef:+.2f} (q {r.q:.2g}, {int(r.n_slides_agree)}/4)"
                                 for r in F9[(F9.contrast == "epi") & F9.source.isin(["raw", "shape"]) & ~F9.control].itertuples()})
_a10 = F10A[F10A.radius == R_CONTACT]
print("F10 protease field around T (30 µm, log2 o/e):",
      {f"{r.gene} {r.anchor}": f"{r.pooled:+.2f} [{r.lo:+.2f},{r.hi:+.2f}] {int(r.n_slides_up)}/4 up" for r in _a10.itertuples()})
if TX_OK:
    print("    stromal MMP next to mal-hi T (robust):",
          {f"{r.lineage} {r.gene} {r.group}": round(r.coef, 3) for r in F10C[F10C.robust].itertuples()} or "none")
if len(TOP11):
    print("F11 top protease–T-gene local correlation:", {f"{a}-{b}": round(z, 1) for (a, b), z in TOP11.head(5).items()})
for tag, d in F12.items():
    print(f"F12 [{tag}] significant T-involving LR tests: {int((d.Z_FDR < 0.05).sum())}; 28b axes significant:",
          (d.cts + " " + d.pair)[d.axis_28b & (d.Z_FDR < 0.05)].tolist() or "none")
```

```
cell table → /home/projects/nyosef/zvise/MF/data/S1_cosmx_earlymf_spatial/processed/cosmx_s1_cells_28e.parquet
                 figure                               file                                                                                                          what
         C_compartments          deg28e_C_compartments.png                                 T cells per compartment per slide; keratinocyte-neighbour fraction of T cells
F7_T_share_of_degradome deg28e_F7_T_share_of_degradome.png                            MMP transcript share by lineage; per-piece T vs stroma (raw/resolVI); CosMx vs 28b
 F8_malignancy_transfer  deg28e_F8_malignancy_transfer.png                                transfer score by T subset; piece specificity by tertile; score by compartment
    F9_motility_proxies     deg28e_F9_motility_proxies.png epidermis vs dermis in T cells: residence/egress/LMNA genes (+ keratin-adjusted), shape, keratinocyte contact
   F10_degradome_near_T    deg28e_F10_degradome_near_T.png         log2 obs/exp non-T MMP field around T (all / mal-hi / mal-lo); stromal MMP next to mal-hi vs mal-lo T
 F11_hotspot_protease_T  deg28e_F11_hotspot_protease_T.png                               Hotspot local-correlation Z, proteases × T / axis genes; per-slide count Z>1.96
         F12_harreman_T          deg28e_F12_harreman_T.png                            Harreman cell-type-aware LR Z, T (by compartment) ↔ fibro / myeloid / keratinocyte

==========================================================================================
CAVEATS (read first)
==========================================================================================
- 28c amoeboid programs are off-panel (AQP3, LMNA of 37 genes): F9 reads shape / LMNA / residence-egress proxies
- malignancy = transfer score, group-level only: held-out AUC gate passed (0.92); in situ CD8<CD4 False, piece specificity True -> mal splits NOT used
- resolVI used: raw T-cell MMP includes neighbour spillover (upper bound)
- stage = slide: patch ['3', '4'] vs plaque ['7', '8'] → described, never tested

==========================================================================================
RESULTS
==========================================================================================
C   6% of T cells (pieces with epidermis) are intraepidermal; 333 Pautrier-like
F7  T cells: 23.7% of cells, 15.9% of library; share of MMP2 12.4%, MMP9 19.2%, MMP14 12.1%
    robust T < stroma (per piece): {'raw MMP2 vs Fibro': -2.26, 'raw MMP2 vs Myelo': -0.67, 'raw MMP9 vs Myelo': -1.46, 'raw MMP14 vs Fibro': -1.25, 'raw MMP14 vs Myelo': -0.53, 'raw MMP12 vs Myelo': -1.08, 'raw MMP1 vs Fibro': -0.43, 'raw TIMP1 vs Fibro': -0.63, 'resolVI MMP2 vs Fibro': -3.0, 'resolVI MMP2 vs Myelo': -0.93, 'resolVI MMP9 vs Myelo': -1.91, 'resolVI MMP14 vs Fibro': -1.87, 'resolVI MMP14 vs Myelo': -1.02, 'resolVI MMP12 vs Myelo': -2.31, 'resolVI MMP1 vs Fibro': -1.38, 'resolVI MMP1 vs Myelo': -0.18, 'resolVI TIMP1 vs Fibro': -0.89}
F8  score epidermis vs dermis: -0.09 logit (q 0.67, 3/4)
F9  robust epidermis-vs-dermis proxies in T cells: {'raw KLF2': -0.81, 'raw CCR7': -0.505}
    all (raw / shape): {'LMNA': '+0.31 (q 0.13, 2/4)', 'ITGAE': '+0.50 (q 0.077, 2/4)', 'CD69': '+0.21 (q 0.4, 2/4)', 'CXCR3': '+0.13 (q 0.51, 3/4)', 'CCR10': '-0.11 (q 0.55, 3/4)', 'CXCR6': '-0.30 (q 0.4, 2/4)', 'KLF2': '-0.81 (q 0.00021, 4/4)', 'SELL': '-0.27 (q 0.4, 3/4)', 'CCR7': '-0.51 (q 0.013, 3/4)', 'CXCR4': '+0.16 (q 0.4, 3/4)', 'AQP3': '+0.28 (q 0.21, 3/4)', 'AspectRatio': '+0.01 (q 0.83, 1/4)', 'log Area': '-0.01 (q 0.83, 2/4)'}
F10 protease field around T (30 µm, log2 o/e): {'MMP2 all T': '-0.28 [-0.39,-0.13] 1/4 up', 'MMP9 all T': '-0.15 [-0.28,-0.01] 0/4 up', 'MMP14 all T': '-0.22 [-0.33,-0.13] 0/4 up', 'MMP12 all T': '+0.21 [-0.04,+0.45] 3/4 up'}
F11 top protease–T-gene local correlation: {'TIMP1-CD3D': 16.9, 'TIMP1-CD3E': 15.7, 'TIMP1-CD2': 14.5, 'MMP9-CD2': 11.9, 'TIMP1-CD3G': 10.8}
F12 [comp] significant T-involving LR
... [360 more chars truncated]
```
