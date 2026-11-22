# 20_skin/30_perturbseq_per_donor.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/20_skin/30_perturbseq_per_donor.ipynb` and is never run._

# 30 — Per-donor pert2state: patient-specific regulators of the MF malignant-CD4 program

**Extends nb29.** nb29 fitted one cohort signature (malignant vs reactive CD4, 66 donors) on the GWCD4i
Stim48hr knockdown matrix and got held-out r = 0.61. Here one signature is fitted **per donor** (30 donors
with ≥ 600 malignant and ≥ 600 reactive CD4 cells) and we ask: which regulators recur across patients, how
patients group by their regulator weights, and what the top recurrent regulators control downstream.

**Key simplification — X is identical for every donor.** The Perturb-seq matrix (genes × 2,582 regulators,
on-target entry masked), its column standardization and the TruncatedSVD (D = 60) are computed **once**;
only `y` changes. Each donor is one elastic net with **frozen** `alpha` / `l1_ratio` (chosen once on the
30-donor cohort signature with the authors' inner 4-fold CV) on the same 60-dim projection and the **same**
gene folds (5-fold × 3 repeats), so all weights live in one coordinate system and are directly comparable.

**Per-donor signature.** Each donor's malignant and reactive cells are split into 3 seeded, balanced
pseudo-replicates each → 6 pseudobulks → pydeseq2 `~ arm` → Wald `stat`. The pseudo-replicate variance is
sampling-only so |z| is inflated; `prepare_y` standardizes each signature, so only the per-gene relative
weighting matters.

**Sign convention (nb29).** $w_k > 0$: knocking down $k$ pushes a normal CD4 cell toward the malignant
program; $w_k < 0$: the program depends on $k$.

**Deviations from the authors' code path, all deliberate:** scaler + SVD fitted once on all genes instead of
inside every fold (unsupervised, so held-out r is only marginally optimistic); hyper-parameters frozen
across donors instead of re-tuned per fit (weights are only comparable if shrinkage is identical);
LIC x-axis = average linkage on correlation distance (nb29: Ward).

**Run order.** nb29 (§0, §2 caches) → `bash jobs/run_pseq30.sh` (signatures, design, 61 fits; ~1 h CPU) →
this notebook (light: reads the caches, every cell runs on the login kernel in minutes).

```python  # cell 1
# ============================================================
# §0  Parameters, imports, style
# ============================================================
import os, sys, importlib, warnings
from pathlib import Path
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")


def _resolve_nb_dir() -> Path:
    start = Path.cwd()
    for base in [start, *start.parents]:
        for sub in [Path("."), Path("MF")]:
            cand = base / sub
            if cand.name == "MF" and (cand / "data").exists():
                return cand.resolve()
    raise FileNotFoundError(f"could not locate MF/data from {start}")


NB_DIR = _resolve_nb_dir(); print("NB_DIR =", NB_DIR)
sys.path.insert(0, str(NB_DIR / "helpers"))
import numpy as np, pandas as pd, scipy.stats as ss
import matplotlib.pyplot as plt, matplotlib as mpl
import scipy.cluster.hierarchy as sch
import pseq30_helpers as H
import perturbseq_helpers as P
import tcr_signaling_helpers as TH
for _m in (H, P, TH):
    importlib.reload(_m)
warnings.filterwarnings("ignore", category=FutureWarning)
H.set_style()
TEAL, DARK, MUTED, GREY, SCR = (H.DESIGN[k] for k in ("teal", "dark", "muted", "grey", "scr"))
SEED = H.SEED
np.random.seed(SEED)

missing = [p.name for p in (H.DONOR_TAB, H.GENE_SET, H.DONOR_Z, H.COHORT_DE, H.SVD_NPZ, H.LIC_ORDER,
                            H.HYPER, H.REACT_COMP, H.MODEL_FIT, H.WEIGHTS, H.W_SEM) if not p.exists()]
assert not missing, f"missing caches {missing} -- run `bash jobs/run_pseq30.sh` first"
print("knobs:", {k: getattr(H, k) for k in ("MIN_CELLS_ARM", "N_PSEUDO", "GENE_FRAC_DONORS", "MIN_CELLS_GENE",
                                             "N_PCS", "N_SPLITS", "N_REPEATS", "TOP_K", "MIN_RECUR", "FDR_SIG", "FDR_KD")})
```

```
NB_DIR = /home/projects/nyosef/zvise/MF

knobs: {'MIN_CELLS_ARM': 600, 'N_PSEUDO': 3, 'GENE_FRAC_DONORS': 0.9, 'MIN_CELLS_GENE': 10, 'N_PCS': 60, 'N_SPLITS': 5, 'N_REPEATS': 3, 'TOP_K': 25, 'MIN_RECUR': 15, 'FDR_SIG': 0.01, 'FDR_KD': 0.1}
```

## §1 — Inputs: donors, gene set, frozen hyper-parameters, model fit

Everything below was computed by `jobs/run_pseq30.py`. Assertions check the invariants the design relies
on: 30 donors, one gene list for every fit, no NaN in the weight matrix, weight matrix = regulators × 30.

```python  # cell 3
tab = pd.read_csv(H.DONOR_TAB, index_col=0)
donors = tab.index[tab["retained"]].tolist()
genes = [l.strip() for l in H.GENE_SET.read_text().splitlines() if l.strip()]
hp = H.read_json(H.HYPER)
design = H.load_design()
lic_ord = [l.strip() for l in H.LIC_ORDER.read_text().splitlines() if l.strip()]
react = pd.read_csv(H.REACT_COMP, index_col=0)
annot = H.donor_annotation(tab, react)
cohort = pd.read_csv(H.COHORT_DE, index_col=0)
Zd = pd.read_parquet(H.DONOR_Z)
mf = pd.read_csv(H.MODEL_FIT, index_col=0)
wt = pd.read_csv(H.WEIGHTS, index_col=0)
W = wt[donors]                       # regulators x donors (mean over the 15 folds)
Wsem = pd.read_parquet(H.W_SEM)      # the SEM over folds, same shape
w_cohort = wt["cohort_w"]
w_mean, w_sd = wt["mean_w"], wt["sd_w"]
regs = W.index.tolist()

assert len(donors) == 30 and list(Zd.columns) == donors and list(mf.index[~mf["is_cohort"]]) == donors
assert list(design["genes"]) == genes and Zd.reindex(genes).notna().all().all(), "one gene list for every fit"
assert W.shape == (len(regs), 30) and not W.isna().any().any()
assert set(lic_ord) == set(regs)
sig_genes = cohort.index[cohort["padj"] < H.FDR_SIG].intersection(genes)

print(f"donors: {len(donors)}  (li2024: {(tab.loc[donors, 'study'] == 'li2024').sum()})")
print(tab.loc[donors, ["n_malignant", "n_reactive", "study", "stage", "split"]].to_string())
print(f"\ngene set (Perturb-seq measured x testable in >= {H.GENE_FRAC_DONORS:.0%} of donors): {len(genes):,}")
print(f"cohort30 signature genes at FDR < {H.FDR_SIG}: {len(sig_genes):,}")
print(f"design: {design['Z'].shape[0]:,} genes x {len(regs):,} regulators -> {design['Z'].shape[1]} SVs "
      f"({design['explained'].sum():.1%} of variance)")
print(f"frozen hyper-parameters: alpha = {hp['alpha']:.4g}, l1_ratio = {hp['l1_ratio']:.4g}")

# ---- model fit, sorted by held-out r
fit = mf.loc[donors].sort_values("test_r", ascending=False)
r_cohort = mf.loc["COHORT30", "test_r"]
print(f"\ncohort30 fit: held-out r = {r_cohort:.3f} +- {mf.loc['COHORT30', 'test_r_se']:.3f}  (nb29 cohort: 0.61)")
print(fit[["study", "n_malignant", "n_reactive", "test_r", "test_r_se", "scrambled_r", "train_r",
           "r_vs_consensus_loo", "converged"]].round(3).to_string())
flags = []
flags += [f"{d}: a fold hit max_iter" for d in fit.index[~fit["converged"]]]
flags += [f"{d}: weights correlate {fit.loc[d, 'r_vs_consensus_loo']:.2f} with the leave-one-out consensus (< 0.1)"
          for d in fit.index[fit["r_vs_consensus_loo"] < 0.1]]
print("\nFLAGS:", *flags, sep="\n  ") if flags else print("\nno convergence / consensus flags")
```

```
donors: 30  (li2024: 14)
                      n_malignant  n_reactive            study     stage        split
donor                                                                                
Li2024_atlas__CTCL5          7532        4073           li2024  advanced    discovery
Li2024_atlas__CTCL6          3937       10298           li2024     early    discovery
Li2024_atlas__CTCL7          3658       38432           li2024     early    discovery
D1__P204                     3444        5741  chennareddy2025  advanced    discovery
Li2024_atlas__CTCL13         5962        2884           li2024  advanced    discovery
Li2024_atlas__CTCL8          7379        2447           li2024     early  replication
Li2024_atlas__MF30           2342        1923           li2024  advanced  replication
D7__MF311                    5693        1919   rindler2021_mc  advanced    discovery
Li2024_atlas__MF28          10395        1904           li2024  advanced    discovery
D7__P84                      1892        3889   rindler2021_mc  advanced  replication
Li2024_atlas__CTCL3          8823        1778           li2024     early    discovery
D7__MF309                    5395        1748   rindler2021_mc  advanced    discovery
D11__P3                      7596        1589  brentuximab2026  advanced    discovery
Li2024_atlas__PT50           1464        1858           li2024     early  replication
D11__P6                      4575        1376  brentuximab2026  advanced    discovery
D11__P5                      6550        1370  brentuximab2026  advanced    discovery
Li2024_atlas__CTCL2          1357        2200           li2024     early    discovery
Li2024_atlas__PT56           1292        2478           li2024  advanced    discovery
Li2024_atlas__CTCL4          1241        7906           li2024     early    discovery
D11__P2                      9473        1063  brentuximab2026  advanced    discovery
D8__P150                     2393         912        alkon2024     early    discovery
D13__MF18                     826         935     gaydosik2022  advanced  replication
Li2024_atlas__PT53           1417         735           li2024  advanced    discovery
D15__MF48                    3650         709        il4ra2026  advanced  replication
Li2024_atlas__MF21           1761         708           li2024  advanced  replication
D6__MF2                       677         708     gaydosik2019  advanced  replication
D15__MF24                     643        1958        il4ra2026  advanced  replication
D15__MF36                     620        2182        il4ra2026  advanced    discovery
D8__P65                       615         626        alkon2024     early    discovery
D8__P138                      627         614        alkon2024     early    discovery

gene set (Perturb-seq measured x testable in >= 90% of donors): 6,048
cohort30 signature genes at FDR < 0.01: 2,536
design: 6,048 genes x 2,582 regulators -> 60 SVs (41.2% of variance)
frozen hyper-parameters: alpha = 0.1887, l1_ratio = 0.01

cohort30 fit: held-out r = 0.600 +- 0.006  (nb29 cohort: 0.61)
                                study  n_malignant  n_reactive  test_r  test_r_se  scrambled_r  train_r  r_vs_consensus_loo  converged
donor                                                                                                                                 
D15__MF36                   il4ra2026          620        2182   0.602      0.005        0.242    0.619               0.806       True
Li2024_atlas__PT50             li2024         1464        1858   0.599      0.005        0.321    0.617               0.615       True
Li2024_atlas__CTCL7            li2024         3658       38432   0.582      0.007        0.012    0.603               0.385       True
D1__P204              chennareddy2025         3444        5741   0.549      0.010        0.203    0.574               0.622       True
D8__P138                    alkon2024          627         614   0.509      0.009        0.050    0.534   
... [3498 more chars truncated]
```

## §2 — F4 Model fit per donor

Held-out Pearson r (mean ± SE over the 15 gene folds) per donor, with the same donor's fit on the shared
scrambled design (each gene's z permuted across regulators) and the 30-donor cohort fit as the ceiling.
Per-donor r is expected well below the cohort's (0.2–0.45); the scrambled points make it interpretable.
Right: r vs min(n cells) — a diagnostic for residual power dependence, not a filter.

```python  # cell 5
fig, axs = plt.subplots(1, 2, figsize=(7.5, 3), gridspec_kw={"width_ratios": [2.2, 1]})
ax = axs[0]
x = np.arange(len(fit))
ax.errorbar(x, fit["scrambled_r"], yerr=fit["scrambled_r_se"], fmt="o", ms=3, color=SCR, ecolor=SCR,
            elinewidth=0.6, label="scrambled X")
ax.errorbar(x, fit["test_r"], yerr=fit["test_r_se"], fmt="o", ms=3.5, color=TEAL, ecolor=TEAL, elinewidth=0.6,
            label="held-out genes")
ax.axhline(r_cohort, ls="--", lw=0.8, color=DARK)
ax.text(len(fit) - 0.5, r_cohort, f"cohort30 fit r = {r_cohort:.2f}", ha="right", va="bottom", fontsize=6, color=DARK)
ax.axhline(0, lw=0.5, color=MUTED)
ax.set_xticks(x); ax.set_xticklabels([d.replace("Li2024_atlas__", "Li_") for d in fit.index], rotation=90, fontsize=5)
ax.set_ylabel("Pearson r, held-out genes"); ax.legend(frameon=False, loc="center right", fontsize=6)
ax.set_title("Held-out r per donor (mean ± SE over 15 gene folds)", fontsize=7)
ax = axs[1]
rho, p = ss.spearmanr(fit["n_min"], fit["test_r"])
ax.scatter(fit["n_min"], fit["test_r"], s=14, color=TEAL)
ax.set_xscale("log"); ax.set_xlabel("min(n malignant, n reactive)"); ax.set_ylabel("held-out r")
ax.set_title(f"Held-out r vs cell number (Spearman rho = {rho:.2f}, p = {p:.2g})", fontsize=7)
med, scr_med = fit["test_r"].median(), fit["scrambled_r"].median()
fig.suptitle(f"Per-donor model fit: held-out r (teal) vs scrambled-X r (grey); median {med:.2f} vs {scr_med:.2f}, cohort30 {r_cohort:.2f}",
             fontsize=8)
H.savefig(fig, "model_fit")
print(f"Spearman rho(r, min n cells) = {rho:.3f}, p = {p:.3g}")
```

```
<Figure size 825x330 with 2 Axes>
Spearman rho(r, min n cells) = 0.044, p = 0.816
```

## §3 — Regulator weights per donor: the Long-Island-City plots and recurrence

Every LIC panel uses the **same x ordering** (regulators clustered on their knockdown effects over the
cohort30 signature genes; average linkage, correlation distance), so the 30 panels and the consensus are
aligned. **3.1 Recurrence:** per donor the top 25 and bottom 25 regulators by $w$; per regulator, in how
many donors it sits in the top set (consistent positive sign) or the bottom set (consistent negative sign).
Chance expectation ≈ 30 × 50 / 2,582 ≈ 0.6 donors per regulator; a binomial p is given for completeness.

```python  # cell 7
rec = H.recurrence(W, top_k=H.TOP_K, min_recur=H.MIN_RECUR)
rec["cluster"] = P.regulator_cluster_map(P.load_clusters()).reindex(rec.index)
rec.to_csv(H.RECURRENCE)
recurrent = rec.index[rec["recurrent"]].tolist()
print(f"expected recurrences per regulator under chance: {rec.attrs['expected_chance']:.2f}")
print(f"recurrent regulators (top/bottom {H.TOP_K} in >= {H.MIN_RECUR}/30 donors): {len(recurrent)}  "
      f"(positive {int((rec.loc[recurrent, 'direction'] == 'positive').sum())}, "
      f"negative {int((rec.loc[recurrent, 'direction'] == 'negative').sum())})")
print(rec.head(30)[["n_top", "n_bottom", "n_recur", "direction", "mean_w", "sd_w", "frac_same_sign", "q_binom", "cluster"]]
      .round(3).to_string())
```

```
expected recurrences per regulator under chance: 0.58
recurrent regulators (top/bottom 25 in >= 15/30 donors): 2  (positive 0, negative 2)
           n_top  n_bottom  n_recur direction  mean_w   sd_w  frac_same_sign  q_binom                    cluster
regulator                                                                                                       
LCP2           0        16       16  negative  -0.006  0.005           0.933      0.0          c22 TCR signaling
INTS12         0        16       16  negative  -0.007  0.004           0.933      0.0                c45 unknown
ZBED3          0        14       14  negative  -0.005  0.003           0.933      0.0          c22 TCR signaling
UFM1          13         0       13  positive   0.005  0.003           0.967      0.0  c41 protein glycosylation
TMEM131L      13         0       13  positive   0.004  0.004           0.800      0.0  c41 protein glycosylation
KDM1A          0        12       12  negative  -0.005  0.003           0.967      0.0      c47 CtBP core complex
RCOR1          0        12       12  negative  -0.005  0.002           0.967      0.0      c47 CtBP core complex
ZAP70          0        12       12  negative  -0.005  0.004           0.933      0.0          c22 TCR signaling
FITM2          0        12       12  negative  -0.005  0.004           0.900      0.0          c21 TCR signaling
TYK2          11         0       11  positive   0.005  0.003           0.900      0.0                        NaN
CD3E           0        11       11  negative  -0.004  0.004           0.900      0.0          c22 TCR signaling
GSTCD          0        11       11  negative  -0.005  0.004           0.900      0.0                c45 unknown
PSIP1         11         0       11  positive   0.005  0.004           0.867      0.0   c38 MLL2 complex related
MTA2          11         0       11  positive   0.005  0.004           0.833      0.0      c13 CHD4/MTA2 complex
PDCD10        11         3       11  positive   0.005  0.007           0.833      0.0                c77 unknown
MED24         11         1       11  positive   0.004  0.004           0.833      0.0          c10 mediator-SAGA
HDAC1         10         0       10  positive   0.005  0.003           0.933      0.0                c60 unknown
ZNF706        10         0       10  positive   0.004  0.003           0.933      0.0                        NaN
BCAT2          0        10       10  negative  -0.004  0.003           0.933      0.0          c22 TCR signaling
SYNCRIP       10         0       10  positive   0.005  0.003           0.900      0.0                        NaN
IRF1          10         0       10  positive   0.005  0.004           0.900      0.0                        NaN
IRF9          10         0       10  positive   0.005  0.004           0.867      0.0                        NaN
C11orf68      10         0       10  positive   0.004  0.004           0.867      0.0                        NaN
TSC2          10         0       10  positive   0.004  0.003           0.867      0.0          c21 TCR signaling
DDRGK1         9         0        9  positive   0.005  0.002           0.967      0.0  c41 protein glycosylation
RHOA           0         9        9  negative  -0.005  0.002           0.967      0.0                        NaN
CSNK2A1        9         0        9  positive   0.004  0.003           0.867      0.0                        NaN
PUM1           0         9        9  negative  -0.004  0.003           0.867      0.0                c78 unknown
EHMT2          0         9        9  negative  -0.004  0.003           0.867      0.0    c65 EHMT1/EHMT2 complex
PLCG1          0         9        9  negative  -0.004  0.004           0.833      0.0          c22 TCR signaling
```

```python  # cell 8
# F1 -- 30 LIC panels, 5 x 6, shared x ordering and y limits; teal = that donor's top/bottom 25; 3 + 3 labels
ylim = (float(W.min().min()) * 1.05, float(W.max().max()) * 1.05)
fig, axs = plt.subplots(5, 6, figsize=(12, 8.5), sharex=True, sharey=True)
for ax, d in zip(axs.ravel(), fit.index):
    w = W[d]
    hi = w.nlargest(H.TOP_K).index.tolist() + w.nsmallest(H.TOP_K).index.tolist()
    lab = w.nlargest(3).index.tolist() + w.nsmallest(3).index.tolist()
    H.lic_panel(ax, w, lic_ord, highlight=hi, labels=lab, ylim=ylim, s=1.5, label_fs=4.5)
    ax.set_title(f"{d.replace('Li2024_atlas__', 'Li_')} · {tab.loc[d, 'n_malignant']:,}/{tab.loc[d, 'n_reactive']:,} cells · "
                 f"r = {fit.loc[d, 'test_r']:.2f}", fontsize=6, color=DARK)
    ax.tick_params(labelsize=5)
for ax in axs[-1]:
    ax.set_xlabel("regulators (shared LIC order)", fontsize=6)
for ax in axs[:, 0]:
    ax.set_ylabel("$w$", fontsize=7)
fig.suptitle(f"Regulator weights per donor (donors sorted by held-out r; teal = that donor's top/bottom {H.TOP_K} regulators)",
             fontsize=9, y=1.0)
fig.tight_layout()
H.savefig(fig, "lic_per_donor")
```

```
<Figure size 1320x935 with 30 Axes>
```

```python  # cell 9
# F2 -- consensus LIC: mean w across the 30 donors (± SD); colour = recurrence tier, i.e. the fraction of donors in
# whose top/bottom 25 the regulator sits (> 50 %, > 40 %, > 30 %); every coloured regulator is named
TIERS = [("> 50 %", 0.5, TEAL), ("> 40 %", 0.4, "#D97706"), ("> 30 %", 0.3, "#7C3AED")]
frac_recur = rec["n_recur"] / len(donors)
tier_of = {}
for name, thr, col in TIERS[::-1]:              # lowest tier first so a higher tier overwrites
    for r in rec.index[frac_recur > thr]:
        tier_of[r] = (name, col)
tiered = [r for r in rec.sort_values("n_recur", ascending=False).index if r in tier_of]
tier_colors = {r: tier_of[r][1] for r in tiered}
fig, ax = plt.subplots(figsize=(9.5, 4.2))
H.lic_panel(ax, w_mean, lic_ord, highlight=tiered, labels=tiered, colors=tier_colors, yerr=w_sd, s=2.5, label_fs=5)
for name, thr, col in TIERS:
    n = sum(1 for r in tiered if tier_of[r][0] == name)
    ax.scatter([], [], color=col, s=14, label=f"in top/bottom {H.TOP_K} of {name} of donors (> {int(thr * len(donors))}/30; n = {n})")
ax.legend(frameon=False, fontsize=6, loc="upper left")
ax.set_ylabel("mean $w$ across 30 donors (± SD)")
ax.set_xlabel(f"{len(regs):,} regulators, clustered on their knockdown effects on {len(sig_genes):,} cohort signature genes")
ax.set_title("Mean regulator weight across donors; colour = share of donors with the regulator in their top/bottom 25", fontsize=8)
H.savefig(fig, "lic_consensus_recurrent")
```

```
<Figure size 1045x462 with 1 Axes>
```

```python  # cell 10
# the recurrence tiers coloured in F2, printed (each regulator listed once, in its highest tier)
cols = ["n_top", "n_bottom", "n_recur", "direction", "mean_w", "sd_w", "frac_same_sign", "q_binom", "cluster"]
for name, thr, _ in TIERS:
    members = [r for r in tiered if tier_of[r][0] == name]
    n_pos = sum(rec.loc[r, "direction"] == "positive" for r in members)
    print(f"\n=== recurrent in {name} of donors (> {int(thr * len(donors))}/30): {len(members)} regulators "
          f"({n_pos} w > 0 'knockdown mimics MF', {len(members) - n_pos} w < 0 'MF depends on it') ===")
    print(rec.loc[members, cols].round(3).to_string() if members else "none")
```

```
=== recurrent in > 50 % of donors (> 15/30): 2 regulators (0 w > 0 'knockdown mimics MF', 2 w < 0 'MF depends on it') ===
           n_top  n_bottom  n_recur direction  mean_w   sd_w  frac_same_sign  q_binom            cluster
regulator                                                                                               
LCP2           0        16       16  negative  -0.006  0.005           0.933      0.0  c22 TCR signaling
INTS12         0        16       16  negative  -0.007  0.004           0.933      0.0        c45 unknown

=== recurrent in > 40 % of donors (> 12/30): 3 regulators (2 w > 0 'knockdown mimics MF', 1 w < 0 'MF depends on it') ===
           n_top  n_bottom  n_recur direction  mean_w   sd_w  frac_same_sign  q_binom                    cluster
regulator                                                                                                       
ZBED3          0        14       14  negative  -0.005  0.003           0.933      0.0          c22 TCR signaling
UFM1          13         0       13  positive   0.005  0.003           0.967      0.0  c41 protein glycosylation
TMEM131L      13         0       13  positive   0.004  0.004           0.800      0.0  c41 protein glycosylation

=== recurrent in > 30 % of donors (> 9/30): 19 regulators (12 w > 0 'knockdown mimics MF', 7 w < 0 'MF depends on it') ===
           n_top  n_bottom  n_recur direction  mean_w   sd_w  frac_same_sign  q_binom                   cluster
regulator                                                                                                      
KDM1A          0        12       12  negative  -0.005  0.003           0.967      0.0     c47 CtBP core complex
RCOR1          0        12       12  negative  -0.005  0.002           0.967      0.0     c47 CtBP core complex
ZAP70          0        12       12  negative  -0.005  0.004           0.933      0.0         c22 TCR signaling
FITM2          0        12       12  negative  -0.005  0.004           0.900      0.0         c21 TCR signaling
TYK2          11         0       11  positive   0.005  0.003           0.900      0.0                       NaN
CD3E           0        11       11  negative  -0.004  0.004           0.900      0.0         c22 TCR signaling
GSTCD          0        11       11  negative  -0.005  0.004           0.900      0.0               c45 unknown
PSIP1         11         0       11  positive   0.005  0.004           0.867      0.0  c38 MLL2 complex related
MTA2          11         0       11  positive   0.005  0.004           0.833      0.0     c13 CHD4/MTA2 complex
PDCD10        11         3       11  positive   0.005  0.007           0.833      0.0               c77 unknown
MED24         11         1       11  positive   0.004  0.004           0.833      0.0         c10 mediator-SAGA
HDAC1         10         0       10  positive   0.005  0.003           0.933      0.0               c60 unknown
ZNF706        10         0       10  positive   0.004  0.003           0.933      0.0                       NaN
BCAT2          0        10       10  negative  -0.004  0.003           0.933      0.0         c22 TCR signaling
SYNCRIP       10         0       10  positive   0.005  0.003           0.900      0.0                       NaN
IRF1          10         0       10  positive   0.005  0.004           0.900      0.0                       NaN
IRF9          10         0       10  positive   0.005  0.004           0.867      0.0                       NaN
C11orf68      10         0       10  positive   0.004  0.004           0.867      0.0                       NaN
TSC2          10         0       10  positive   0.004  0.003           0.867      0.0         c21 TCR signaling
```

```python  # cell 11
# F2b -- the same consensus LIC with only the > 50 % tier highlighted (the most recurrent regulators)
top50 = [r for r in tiered if tier_of[r][0] == TIERS[0][0]]
fig, ax = plt.subplots(figsize=(9.5, 3.8))
H.lic_panel(ax, w_mean, lic_ord, highlight=top50, labels=[], yerr=w_sd, s=2.5)
pos = {r: i for i, r in enumerate(lic_ord)}
for r in top50:
    ax.annotate(f"{r}\n{rec.loc[r, 'direction']} in {int(rec.loc[r, 'n_recur'])}/30 donors",
                (pos[r], w_mean[r]), fontsize=7, fontweight="bold", color=DARK, xytext=(6, -4), textcoords="offset points")
ax.set_ylabel("mean $w$ across 30 donors (± SD)")
ax.set_xlabel(f"{len(regs):,} regulators, clustered on their knockdown effects on {len(sig_genes):,} cohort signature genes")
ax.set_title(f"Mean regulator weight across donors; teal = regulators in the top/bottom {H.TOP_K} of > 50 % of donors "
             f"({', '.join(top50)})", fontsize=8)
H.savefig(fig, "lic_consensus_top50")

# per-donor view of the same regulators: weight and rank in each donor, counted from the regulator's own direction
ranks_neg = W.rank(axis=0, ascending=True).astype(int)
ranks_pos = W.rank(axis=0, ascending=False).astype(int)
view = pd.DataFrame({f"{r} w": W.loc[r].round(4) for r in top50})
for r in top50:
    view[f"{r} rank"] = (ranks_neg if rec.loc[r, "direction"] == "negative" else ranks_pos).loc[r]
view["stage"] = annot["stage"].reindex(view.index)
print(f"rank = position of the regulator in that donor's weights, counted from the regulator's own direction (1 = strongest of {len(regs):,})")
print(view.sort_values(f"{top50[0]} rank").to_string())
```

```
<Figure size 1045x418 with 1 Axes>
rank = position of the regulator in that donor's weights, counted from the regulator's own direction (1 = strongest of 2,582)
                      LCP2 w  INTS12 w  LCP2 rank  INTS12 rank     stage
D1__P204             -0.0144   -0.0091          1           15  advanced
Li2024_atlas__MF30   -0.0093   -0.0078          1            3  advanced
D7__P84              -0.0110   -0.0042          1           79  advanced
Li2024_atlas__MF28   -0.0101   -0.0044          1           83  advanced
D11__P6              -0.0096   -0.0095          1            2  advanced
D15__MF36            -0.0109   -0.0093          1            5  advanced
D6__MF2              -0.0079   -0.0056          1           33  advanced
D11__P3              -0.0085   -0.0098          2            1  advanced
D8__P138             -0.0118    0.0025          2         2100     early
D11__P2              -0.0091   -0.0097          2            1  advanced
Li2024_atlas__PT50   -0.0090   -0.0085          3            6     early
D8__P65              -0.0074   -0.0060          3           27     early
D11__P5              -0.0079   -0.0106         10            2  advanced
Li2024_atlas__CTCL13 -0.0062   -0.0045         12           71  advanced
Li2024_atlas__MF21   -0.0072   -0.0113         17            1  advanced
Li2024_atlas__CTCL2  -0.0060   -0.0014         19          753     early
D15__MF48            -0.0055   -0.0115         31            1  advanced
Li2024_atlas__CTCL5  -0.0046   -0.0027         55          278  advanced
D15__MF24            -0.0046   -0.0125         82            1  advanced
Li2024_atlas__PT53   -0.0034   -0.0142        153            1  advanced
D7__MF311            -0.0032   -0.0094        245            5  advanced
Li2024_atlas__CTCL4  -0.0017   -0.0035        584          198     early
D13__MF18            -0.0012   -0.0064        699           17  advanced
Li2024_atlas__CTCL8  -0.0012   -0.0051        747           24     early
Li2024_atlas__CTCL6  -0.0011   -0.0039        870          175     early
D7__MF309            -0.0007   -0.0040        934          119  advanced
Li2024_atlas__PT56   -0.0004   -0.0023        976          412  advanced
Li2024_atlas__CTCL3  -0.0002   -0.0096       1177            1     early
Li2024_atlas__CTCL7   0.0038   -0.0039       2263          319     early
D8__P150              0.0044    0.0018       2482         1977     early
```

## §4 — 3.2 Do patients group by their regulator weights? (F5)

Spearman correlation between donors across all regulator weights → distance 1 − ρ → Ward linkage; k over
2–5 by mean silhouette (profile printed so the choice is visible). Cluster membership is tested against study,
stage, cell number, CNV burden and the **reactive-pool composition control** (if donors group by the
CD8A / FOXP3 / cytotoxic content of their reactive CD4 pool, the clustering reads the reference, not the
tumour). li2024 contributes 14/30 donors and uses Flex chemistry — if clusters align with study, that is
stated plainly.

```python  # cell 13
K_FIXED = 4  # clustering resolution: fixed at 4 clusters (silhouette optimum is printed for reference)
cl = H.donor_clustering(W, k_range=H.K_RANGE)
print("silhouette by k:", cl["silhouette"].round(3).to_dict(), f"-> silhouette optimum k = {cl['k']}; using k = {K_FIXED}")
cl["k"] = K_FIXED
cl["labels"] = pd.Series(sch.fcluster(cl["link"], K_FIXED, criterion="maxclust"), index=cl["rho"].index, name="cluster")
labels = cl["labels"]
print("cluster sizes:", labels.value_counts().sort_index().to_dict())
for k in sorted(labels.unique()):
    print(f"  cluster {k}: " + ", ".join(d.replace("Li2024_atlas__", "Li_") for d in labels.index[labels == k]))
cov = H.cluster_covariate_tests(labels, annot)
print("\ncluster vs covariates:")
print(cov.round(4).to_string(index=False))
ct_study = pd.crosstab(annot.loc[labels.index, "study"], labels)
print("\nstudy x cluster:"); print(ct_study.to_string())
print("\nstage x cluster:"); print(pd.crosstab(annot.loc[labels.index, "stage"], labels).to_string())
p_study = float(cov.loc[cov["covariate"] == "study", "p"].iloc[0])
p_react = cov.loc[cov["covariate"].str.startswith("react_"), "p"].min()
print("\nVERDICT:",
      "clusters ALIGN WITH STUDY (p < 0.05) -- treat them as batch, not biology." if p_study < 0.05 else
      "clusters do not track study (p >= 0.05).",
      "Reactive-pool composition differs between clusters (p < 0.05) -- the reference may drive them." if p_react < 0.05 else
      "Reactive-pool composition does not differ between clusters.")
out = annot.loc[labels.index].copy(); out.insert(0, "cluster", labels); out["test_r"] = fit["test_r"].reindex(out.index)
out.to_csv(H.CLUSTERS)

# ---- F5: dendrogram + donor x donor Spearman rho + covariate bars
leaves = cl["leaves"]
rho = cl["rho"].loc[leaves, leaves]
bars = [("cluster", labels, "cat"), ("study", annot["study"], "cat"), ("stage", annot["stage"], "cat"),
        ("log10 n cells", np.log10(annot["n_min"]), "num")]
fig = plt.figure(figsize=(6.2, 6.6))
gs = fig.add_gridspec(2 + len(bars), 2, height_ratios=[0.9, 5] + [0.22] * len(bars), width_ratios=[5, 0.25],
                      hspace=0.06, wspace=0.04)
# cluster colours shared by the dendrogram branches, the heatmap block labels, the tick labels and the cluster bar
ccol = {k: plt.get_cmap("Set2")(j) for j, k in enumerate(sorted(labels.unique()))}
n_d = len(labels)
node_cl = {i: {int(labels.iloc[i])} for i in range(n_d)}
for j, (a, b, *_rest) in enumerate(cl["link"]):
    node_cl[n_d + j] = node_cl[int(a)] | node_cl[int(b)]
axd = fig.add_subplot(gs[0, 0])
with plt.rc_context({"lines.linewidth": 1.0}):
    sch.dendrogram(cl["link"], ax=axd, no_labels=True,
                   link_color_func=lambda i: mpl.colors.to_hex(ccol[next(iter(node_cl[i]))]) if len(node_cl[i]) == 1 else MUTED)
axd.axis("off")
axh = fig.add_subplot(gs[1, 0])
im = axh.imshow(rho.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
axh.set_xticks([]); axh.set_yticks(range(len(leaves)))
axh.set_yticklabels([d.replace("Li2024_atlas__", "Li_") for d in leaves], fontsize=5)
lv = labels.reindex(leaves).to_numpy()
for t, k in zip(axh.get_yticklabels(), lv):
    t.set_color(ccol[int(k)]); t.set_fontweight("bold")
# white separators between clusters + a label in the middle of each diagonal block
bounds = [0] + [i for i in range(1, len(lv)) if lv[i] != lv[i - 1]] + [len(lv)]
for b in bounds[1:-1]:
    axh.axhline(b - 0.5, color="white", lw=1.5); axh.axvline(b - 0.5, color="white", lw=1.5)
for s0, s1 in zip(bounds[:-1], bounds[1:]):
    k = int(lv[s0]); c = (s0 + s1 - 1) / 2
    axh.text(c, c, f"cluster {k}\n(n = {s1 - s0})", ha="center", va="center", fontsize=6.5, fontweight="bold", color=DARK,
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=ccol[k], lw=1.2, alpha=0.95), zorder=5)
cax = fig.add_subplot(gs[1, 1]); fig.colorbar(im, cax=cax).set_label("Spearman rho of $w$", fontsize=6); cax.tick_params(labelsize=5)
cat_cmaps = {"cluster": plt.get_cmap("Set2"), "study": plt.get_cmap("tab10"), "stage": plt.get_cmap("Pastel1")}
for i, (name, v, kind) in enumerate(bars):
    axb = fig.add_subplot(gs[2 + i, 0])
    v = v.reindex(leaves)
    if kind == "cat":
        cats = sorted(v.dropna().astype(str).unique())
        cmap = cat_cmaps[name]
        col = {c: cmap(j % cmap.N) for j, c in enumerate(cats)}
        axb.imshow([[cats.index(str(x)) if pd.notna(x) else -1 for x in v]], cmap=mpl.colors.ListedColormap([col[c] for c in cats]),
                   aspect="auto", vmin=0, vmax=max(len(cats) - 1, 1), interpolation="nearest")
        for j, c in enumerate(cats):
            axb.scatter([], [], marker="s", color=col[c], label=f"{name}={c}", s=12)
    else:
        axb.imshow([v.to_numpy(dtype=float)], cmap="Greys", aspect="auto", interpolation="nearest")
    axb.set_xticks([]); axb.set_yticks([0]); axb.set_yticklabels([name], fontsize=5.5)
    for s in axb.spines.values():
        s.set_visible(False)
handles = [h for ax in fig.axes for h in ax.get_legend_handles_labels()[0]]
lbls = [l for ax in fig.axes for l in ax.get_legend_handles_labels()[1]]
fig.legend(handles, lbls, loc="center left", bbox_to_anchor=(1.0, 0.3), frameon=False, fontsize=5, ncol=1)
fig.suptitle(f"Donor × donor Spearman correlation of regulator weights; Ward dendrogram cut at k = {K_FIXED} clusters "
             f"(silhouette {cl['silhouette'][K_FIXED]:.2f})", fontsize=8, y=0.92)
H.savefig(fig, "donor_clustering")
```

```
silhouette by k: {2: 0.266, 3: 0.24, 4: 0.222, 5: 0.231} -> silhouette optimum k = 2; using k = 4
cluster sizes: {1: 9, 2: 7, 3: 11, 4: 3}
  cluster 1: Li_CTCL5, Li_CTCL8, Li_CTCL3, Li_PT50, Li_CTCL2, D11__P2, Li_PT53, D15__MF48, D8__P65
  cluster 2: Li_CTCL6, Li_CTCL7, D7__MF309, Li_CTCL4, D8__P150, D13__MF18, D8__P138
  cluster 3: D1__P204, Li_CTCL13, Li_MF30, Li_MF28, D7__P84, D11__P3, D11__P6, D11__P5, Li_MF21, D15__MF24, D15__MF36
  cluster 4: D7__MF311, Li_PT56, D6__MF2

cluster vs covariates:
           covariate    test      p                                                                                                                                                                             detail
               study    chi2 0.1922 alkon2024: 1/2/0/0; brentuximab2026: 1/0/3/0; chennareddy2025: 0/0/1/0; gaydosik2019: 0/0/0/1; gaydosik2022: 0/1/0/0; il4ra2026: 1/0/2/0; li2024: 6/3/4/1; rindler2021_mc: 0/1/1/1
               stage    chi2 0.0036                                                                                                                                                 advanced: 4/2/11/3; early: 5/5/0/0
               n_min kruskal 0.9545                                                                        c1: median 1.36e+03 (n=9); c2: median 1.24e+03 (n=7); c3: median 1.59e+03 (n=11); c4: median 1.29e+03 (n=3)
      malignant_frac kruskal 0.2966                                                                                    c1: median 0.658 (n=9); c2: median 0.469 (n=7); c3: median 0.674 (n=11); c4: median 0.489 (n=3)
        n_event_arms kruskal 0.0166                                                                                                 c1: median 15 (n=9); c2: median 4 (n=7); c3: median 15 (n=11); c4: median 15 (n=3)
     react_CD8A_frac kruskal 0.4318                                                                                   c1: median 0.129 (n=9); c2: median 0.0831 (n=7); c3: median 0.183 (n=11); c4: median 0.106 (n=3)
    react_FOXP3_frac kruskal 0.5141                                                                                c1: median 0.0828 (n=9); c2: median 0.0939 (n=7); c3: median 0.0807 (n=11); c4: median 0.0485 (n=3)
react_cytotoxic_frac kruskal 0.0021                                                                                    c1: median 0.166 (n=9); c2: median 0.126 (n=7); c3: median 0.389 (n=11); c4: median 0.268 (n=3)

study x cluster:
cluster          1  2  3  4
study                      
alkon2024        1  2  0  0
brentuximab2026  1  0  3  0
chennareddy2025  0  0  1  0
gaydosik2019     0  0  0  1
gaydosik2022     0  1  0  0
il4ra2026        1  0  2  0
li2024           6  3  4  1
rindler2021_mc   0  1  1  1

stage x cluster:
cluster   1  2   3  4
stage                
advanced  4  2  11  3
early     5  5   0  0

VERDICT: clusters do not track study (p >= 0.05). Reactive-pool composition differs between clusters (p < 0.05) -- the reference may drive them.

<Figure size 682x726 with 7 Axes>
```

```python  # cell 14
# per cluster: mean w over its donors; top 10 most negative (MF depends on the regulator in these donors) and
# top 10 most positive (knockdown mimics MF in these donors). q = one-vs-rest Mann-Whitney across donors, BH over regulators.
from statsmodels.stats.multitest import multipletests
reg_cluster = P.regulator_cluster_map(P.load_clusters())
N_SHOW = 10
rows = []
for k in sorted(labels.unique()):
    inside, outside = labels.index[labels == k], labels.index[labels != k]
    d = pd.DataFrame({"regulator": W.index, "cluster_id": k,
                      "mean_w_in": W[inside].mean(axis=1).to_numpy(), "mean_w_out": W[outside].mean(axis=1).to_numpy()})
    d["diff_in_minus_out"] = d["mean_w_in"] - d["mean_w_out"]
    d["p"] = [ss.mannwhitneyu(W.loc[r, inside], W.loc[r, outside], alternative="two-sided").pvalue for r in W.index]
    d["q"] = multipletests(d["p"], method="fdr_bh")[1]
    d["authors_cluster"] = reg_cluster.reindex(d["regulator"]).to_numpy()
    d["n_recur"] = rec["n_recur"].reindex(d["regulator"]).to_numpy()
    rows.append(d)
cm = pd.concat(rows, ignore_index=True).sort_values(["cluster_id", "mean_w_in"])
cm.to_csv(H.table("cluster_markers"), index=False)
cols = ["regulator", "mean_w_in", "mean_w_out", "q", "n_recur", "authors_cluster"]
for k, sub in cm.groupby("cluster_id"):
    members = [d.replace("Li2024_atlas__", "Li_") for d in labels.index[labels == k]]
    stage_in = annot.loc[labels.index[labels == k], "stage"].value_counts().to_dict()
    print(f"\n=== cluster {k}: n = {len(members)} donors, stage {stage_in}, {int((sub['q'] < 0.1).sum())} regulators differ from the rest at q < 0.1 ===")
    print("    " + ", ".join(members))
    print(f"\n-- top {N_SHOW} NEGATIVE regulators in cluster {k} (MF program depends on them) --")
    print(sub.sort_values("mean_w_in").head(N_SHOW)[cols].round(4).to_string(index=False))
    print(f"\n-- top {N_SHOW} POSITIVE regulators in cluster {k} (knockdown mimics the MF program) --")
    print(sub.sort_values("mean_w_in", ascending=False).head(N_SHOW)[cols].round(4).to_string(index=False))
```

```
=== cluster 1: n = 9 donors, stage {'early': 5, 'advanced': 4}, 0 regulators differ from the rest at q < 0.1 ===
    Li_CTCL5, Li_CTCL8, Li_CTCL3, Li_PT50, Li_CTCL2, D11__P2, Li_PT53, D15__MF48, D8__P65

-- top 10 NEGATIVE regulators in cluster 1 (MF program depends on them) --
regulator  mean_w_in  mean_w_out      q  n_recur       authors_cluster
   INTS12    -0.0076     -0.0061 0.6966       16           c45 unknown
    KDM1A    -0.0066     -0.0045 0.3148       12 c47 CtBP core complex
    RCOR1    -0.0063     -0.0050 0.4961       12 c47 CtBP core complex
    GSTCD    -0.0058     -0.0043 0.6180       11           c45 unknown
     RHOA    -0.0058     -0.0040 0.3543        9                   NaN
     LCP2    -0.0052     -0.0057 0.8101       16     c22 TCR signaling
     AVL9    -0.0052     -0.0033 0.3403        6                   NaN
     XPO6    -0.0051     -0.0032 0.4007        6                   NaN
    ZBED3    -0.0050     -0.0052 0.8934       14     c22 TCR signaling
    ZAP70    -0.0050     -0.0052 0.9738       12     c22 TCR signaling

-- top 10 POSITIVE regulators in cluster 1 (knockdown mimics the MF program) --
regulator  mean_w_in  mean_w_out      q  n_recur           authors_cluster
   ZNF706     0.0072      0.0029 0.1297       10                       NaN
     MTA2     0.0070      0.0038 0.2471       11     c13 CHD4/MTA2 complex
     UFM1     0.0069      0.0046 0.3050       13 c41 protein glycosylation
  SYNCRIP     0.0065      0.0038 0.3743       10                       NaN
    PSIP1     0.0065      0.0037 0.3148       11  c38 MLL2 complex related
  CSNK2A1     0.0061      0.0031 0.2874        9                       NaN
   DDRGK1     0.0058      0.0041 0.3543        9 c41 protein glycosylation
   ACAD11     0.0058      0.0039 0.3050        8 c41 protein glycosylation
    HDAC1     0.0057      0.0050 0.7329       10               c60 unknown
     TSC2     0.0057      0.0028 0.2932       10         c21 TCR signaling

=== cluster 2: n = 7 donors, stage {'early': 5, 'advanced': 2}, 993 regulators differ from the rest at q < 0.1 ===
    Li_CTCL6, Li_CTCL7, D7__MF309, Li_CTCL4, D8__P150, D13__MF18, D8__P138

-- top 10 NEGATIVE regulators in cluster 2 (MF program depends on them) --
regulator  mean_w_in  mean_w_out      q  n_recur          authors_cluster
    STAT3    -0.0072      0.0024 0.0002        3 c79 Th17 differentiation
   PPHLN1    -0.0070     -0.0020 0.0050        8             c101 unknown
  SMARCE1    -0.0069      0.0008 0.0002        4    c81 core BAF subunits
   SUCLG1    -0.0064     -0.0002 0.0042        3              c86 unknown
     EMX1    -0.0058     -0.0009 0.0011        2                      NaN
   FBXO11    -0.0057     -0.0009 0.0021        3              c80 unknown
   NDUFS7    -0.0057     -0.0006 0.0004        1                      NaN
    KDM1A    -0.0056     -0.0050 0.6997       12    c47 CtBP core complex
    ZNF12    -0.0056     -0.0009 0.0009        2                      NaN
  BHLHE40    -0.0056     -0.0007 0.0024        3              c80 unknown

-- top 10 POSITIVE regulators in cluster 2 (knockdown mimics the MF program) --
regulator  mean_w_in  mean_w_out      q  n_recur           authors_cluster
   PDCD10     0.0102      0.0030 0.0195       11               c77 unknown
 TMEM131L     0.0087      0.0031 0.0109       13 c41 protein glycosylation
     MTA2     0.0077      0.0039 0.0884       11     c13 CHD4/MTA2 complex
    PSIP1     0.0074      0.0037 0.1059       11  c38 MLL2 complex related
   ARRDC4     0.0069      0.0023 0.0082        8  c38 MLL2 complex related
    CEP43     0.0067      0.0007 0.0031        2               c27 unknown
    MEMO1     0.0064      0.0016 0.0060        5         c21 TCR signaling
    ZMYM2     0.0063      0.0004 0.0007        3               c27 unknown
    HDAC1     0.0062      0.0049 0.3447       10               c60 unknown
  SYNCRIP     0.0061      0.0042 0.2412       10                       NaN

=== cluster 3: n = 11 donors, stage {'advanced
... [4126 more chars truncated]
```

## §5 — 3.3 Programs: which regulator sets carry the consensus and the cluster-specific weights? (F3)

nb29's signed pathway tests over the authors' 111 functional clusters and the project's TCR/costim modules
(≥ 5 members): one-sided Mann–Whitney for members' $w$ > rest ("knockdown mimics the state") and members'
$w$ < rest ("the state depends on the set"), BH across sets. Run for the consensus (mean $w$) and for each
donor cluster's mean $w$. Programs significant in only one cluster are flagged, not interpreted.

```python  # cell 16
clusters_df = P.load_clusters()
sets = {**P.cluster_sets(clusters_df), **{f"TCRmod:{k}": v for k, v in TH.TCR_MODULES.items()}}
w_groups = {"consensus": w_mean, **{f"cluster{k}": W[labels.index[labels == k]].mean(axis=1) for k in sorted(labels.unique())}}
prog = H.program_tests(w_groups, sets, min_n=5)
prog.to_csv(H.PROGRAMS, index=False)
sig_prog = prog[prog["q_signed"] < 0.1]
print(f"significant programs (q < 0.1): {len(sig_prog)} rows over {sig_prog['set'].nunique()} sets")
for g, sub in sig_prog.groupby("group"):
    print(f"\n[{g}]")
    print(sub[["set", "n", "mean_w", "delta_signed", "direction", "q_signed"]].round(3).to_string(index=False))
non = sig_prog[sig_prog["group"] != "consensus"]
cons_sets = set(sig_prog.loc[sig_prog["group"] == "consensus", "set"])
only_one = [s for s, n in non.groupby("set")["group"].nunique().items() if n == 1 and s not in cons_sets]
print("\nFLAG -- significant in exactly one cluster and not in the consensus:", only_one if only_one else "none")

# ---- F3: consensus LIC coloured by the significant consensus programs (top 8 by q)
cons_sig = sig_prog[sig_prog["group"] == "consensus"].sort_values("q_signed").head(8)
pal = plt.get_cmap("Dark2")
colors, members_all = {}, []
for j, (_, row) in enumerate(cons_sig.iterrows()):
    mem = [m for m in sets[row["set"]] if m in W.index]
    for m in mem:
        colors.setdefault(m, pal(j))
    members_all += mem
fig, ax = plt.subplots(figsize=(7, 3.2))
H.lic_panel(ax, w_mean, lic_ord, highlight=list(dict.fromkeys(members_all)), colors=colors, s=2.5)
for j, (_, row) in enumerate(cons_sig.iterrows()):
    ax.scatter([], [], color=pal(j), s=12, label=f"{row['set'][:38]} (n={row['n']}, q={row['q_signed']:.2g}, {row['direction'].split(' (')[0]})")
ax.legend(frameon=False, fontsize=5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
ax.set_ylabel("mean $w$ across 30 donors"); ax.set_xlabel("regulators (shared LIC order)")
ax.set_title(f"Mean regulator weight across donors; colour = members of the {len(cons_sig)} programs significant for the consensus (q < 0.1)"
             if len(cons_sig) else "Mean regulator weight across donors; no program significant for the consensus (q < 0.1)", fontsize=8)
H.savefig(fig, "lic_consensus_programs")
```

```
significant programs (q < 0.1): 112 rows over 50 sets

[cluster1]
                                      set   n  mean_w  delta_signed                    direction  q_signed
                        c22 TCR signaling  11  -0.004        -0.923 MF depends on set (w < rest)     0.000
                c32 Ragulator-Rag complex  11  -0.003        -0.904 MF depends on set (w < rest)     0.000
                c31 Ragulator-Rag complex  12  -0.003        -0.850 MF depends on set (w < rest)     0.000
                              c27 unknown  89   0.001         0.314      KD mimics MF (w > rest)     0.000
                         c9 mediator-SAGA  34   0.001         0.455      KD mimics MF (w > rest)     0.000
                              c93 unknown   8  -0.003        -0.771 MF depends on set (w < rest)     0.000
                        c99 TCR signaling 195  -0.001        -0.179 MF depends on set (w < rest)     0.000
                      c48 tRNA processing   9  -0.003        -0.787 MF depends on set (w < rest)     0.000
                      c68 tRNA processing  17  -0.001        -0.528 MF depends on set (w < rest)     0.001
                              c64 unknown  17  -0.001        -0.515 MF depends on set (w < rest)     0.001
                    c0 Integrator complex   7  -0.002        -0.739 MF depends on set (w < rest)     0.001
                    c33 elongator complex  12  -0.002        -0.614 MF depends on set (w < rest)     0.001
            c51 Oxidative phosphorylation  22  -0.001        -0.443 MF depends on set (w < rest)     0.001
                              c74 unknown  11   0.002         0.639      KD mimics MF (w > rest)     0.003
                              c61 unknown  17   0.001         0.506      KD mimics MF (w > rest)     0.003
                             c102 unknown  10  -0.002        -0.592 MF depends on set (w < rest)     0.004
                 c38 MLL2 complex related   6   0.003         0.711      KD mimics MF (w > rest)     0.009
                        c10 mediator-SAGA  40   0.001         0.302      KD mimics MF (w > rest)     0.009
               c7 nonsense-mediated decay  10  -0.002        -0.539 MF depends on set (w < rest)     0.011
                              c26 unknown  15   0.001         0.429      KD mimics MF (w > rest)     0.025
                c41 protein glycosylation  40   0.001         0.239      KD mimics MF (w > rest)     0.041
                             c100 unknown  13   0.001         0.417      KD mimics MF (w > rest)     0.041
                              c54 unknown  33   0.001         0.267      KD mimics MF (w > rest)     0.041
                              c35 unknown  34   0.001         0.255      KD mimics MF (w > rest)     0.041
                        c21 TCR signaling 100  -0.000        -0.142 MF depends on set (w < rest)     0.043
               TCRmod:L1_proximal_kinases   6  -0.002        -0.557 MF depends on set (w < rest)     0.043
                       c50 Arp2/3 complex   8  -0.001        -0.487 MF depends on set (w < rest)     0.043
c15 ribosomal and mitochrondrial function  82   0.000         0.151      KD mimics MF (w > rest)     0.071

[cluster2]
                                      set   n  mean_w  delta_signed                    direction  q_signed
                c32 Ragulator-Rag complex  11  -0.002        -0.730 MF depends on set (w < rest)     0.001
                c31 Ragulator-Rag complex  12  -0.002        -0.701 MF depends on set (w < rest)     0.001
                              c27 unknown  89   0.001         0.256      KD mimics MF (w > rest)     0.001
                   c36 citrate metabolism   8   0.002         0.757      KD mimics MF (w > rest)     0.001
                 c38 MLL2 complex related   6   0.004         0.811      KD mimics MF (w > rest)     0.001
                          c76 EMC complex  78  -0.001        -0.251 MF depends on set (w < rest)     0.002
            c51 Oxidative phosphorylation  22  -0.001        -0.426 MF
... [8563 more chars truncated]
```

## §6 — 3.4 Downstream of the top recurrent regulators (F6)

Top 2–3 regulators by recurrence (both signs if both recur). The authors' rule, verbatim: a cohort30
signature gene (FDR < 0.01) is **controlled** by regulator $k$ if (a) it is DE at 10 % FDR on knockdown of
$k$ and (b) the sign of its knockdown z is consistent with both the sign of $w_k$ and the sign of the
signature ($w > 0$: knockdown-up genes are up in malignant; $w < 0$: knockdown-up genes are down in
malignant). Reads two layers of the Perturb-seq h5ad (~1 GB).

```python  # cell 18
pos = rec[rec["recurrent"] & (rec["direction"] == "positive")].head(2).index.tolist()
neg = rec[rec["recurrent"] & (rec["direction"] == "negative")].head(2).index.tolist()
top_regs = (pos[:2] + neg[:1]) if (pos and neg) else (pos + neg)[:3]
if not top_regs:
    top_regs = rec.head(3).index.tolist(); print("no regulator passes the recurrence cutoff; using the top 3 by n_recur")
print("top regulators:", {r: f"{rec.loc[r, 'direction']} n_recur={int(rec.loc[r, 'n_recur'])} mean_w={rec.loc[r, 'mean_w']:+.3f}" for r in top_regs})

pseq = P.load_pseq(H.CONDITION, H.PSEQ_H5, layers=("zscore", "adj_p_value"))
ds_sets = H.downstream_sets(H.CONDITION)
universe = set(sig_genes)
ctrl, enr, top_side = [], [], {}
N_LABEL = 8
for r in top_regs:
    df = H.controlled_genes(pseq, r, float(w_mean[r]), cohort, fdr_sig=H.FDR_SIG, fdr_kd=H.FDR_KD)
    df["n_recur"] = int(rec.loc[r, "n_recur"]); ctrl.append(df)
    c = df[df["controlled"]]
    up = c[c["lfc_cohort"] > 0].sort_values("padj_cohort").head(N_LABEL)
    dn = c[c["lfc_cohort"] < 0].sort_values("padj_cohort").head(N_LABEL)
    top_side[r] = (up, dn)
    print(f"\n{r} (w = {w_mean[r]:+.3f}, recurs in {int(rec.loc[r, 'n_recur'])}/30): {len(df)} signature genes DE on knockdown, "
          f"{len(c)} controlled ({len(c) / max(len(df), 1):.0%} sign-consistent); "
          f"{int((c['lfc_cohort'] > 0).sum())} up in malignant, {int((c['lfc_cohort'] < 0).sum())} down in malignant")
    print(f"  most significant controlled genes UP in malignant:   " + ", ".join(up.index))
    print(f"  most significant controlled genes DOWN in malignant: " + ", ".join(dn.index))
    print("  top 15 controlled genes by |knockdown z|:")
    print(c.head(15)[["z_kd", "lfc_cohort", "padj_cohort"]].round(3).to_string())
    e = H.hypergeom_enrichment(c.index, universe, ds_sets)
    if len(e):
        e.insert(0, "regulator", r); enr.append(e)
        print(f"  enrichment over the authors' downstream programs (q < 0.1): "
              f"{e[e['q'] < 0.1][['set', 'overlap', 'expected', 'q']].round(3).to_string(index=False) if (e['q'] < 0.1).any() else 'none'}")
ctrl = pd.concat(ctrl)
ctrl.to_csv(H.DOWNSTREAM)
enr = pd.concat(enr, ignore_index=True) if enr else pd.DataFrame()

# ---- F6: one volcano per regulator on the cohort30 signature; teal = genes controlled by the regulator;
#          bold labels = the N_LABEL most significant controlled genes on each side (up / down in malignant)
fig, axs = plt.subplots(1, len(top_regs), figsize=(4.8 * len(top_regs) + 0.6, 4.8), sharey=True, squeeze=False)
vol = cohort.reindex(genes).dropna(subset=["padj"])
for ax, r in zip(axs.ravel(), top_regs):
    df = ctrl[ctrl["regulator"] == r]
    c = df[df["controlled"]]
    up, dn = top_side[r]
    ax.scatter(vol["log2FoldChange"], -np.log10(vol["padj"].clip(1e-300)), s=2, color=GREY, lw=0, rasterized=True,
               label="cohort30 signature gene")
    ax.scatter(c["lfc_cohort"], -np.log10(c["padj_cohort"].clip(1e-300)), s=9, color=TEAL, lw=0, zorder=3,
               label=f"controlled by {r} ({len(c)})")
    for g, row in pd.concat([up, dn]).iterrows():
        ax.annotate(g, (row["lfc_cohort"], -np.log10(max(row["padj_cohort"], 1e-300))), fontsize=6.5, fontweight="bold",
                    xytext=(3, 2), textcoords="offset points", color=DARK, zorder=4)
    ax.axvline(0, lw=0.5, color=MUTED, ls="--")
    ax.set_title(f"{r}: w = {w_mean[r]:+.3f}, in the bottom/top {H.TOP_K} of {int(rec.loc[r, 'n_recur'])}/30 donors\n"
                 f"{len(c)} of {len(df)} knockdown-DE signature genes are sign-consistent (controlled)", fontsize=7.5)
    ax.set_xlabel("cohort30 log2FC (malignant / reactive)")
    ax.legend(frameon=False, fontsize=6, loc="upper right")
axs[0, 0].set_ylabel("-log10 FDR (cohort30 malignant vs reactive)")
fig.suptitle("Cohort30 signature volcano per recurrent regulator; teal = genes the regulator controls; bold = most significant per side",
             fontsize=8.5)
fig.tight_layout()
H.savefig(fig, "downstream")
del pseq
```

```
top regulators: {'LCP2': 'negative n_recur=16 mean_w=-0.006', 'INTS12': 'negative n_recur=16 mean_w=-0.007'}


LCP2 (w = -0.006, recurs in 16/30): 589 signature genes DE on knockdown, 514 controlled (87% sign-consistent); 132 up in malignant, 382 down in malignant
  most significant controlled genes UP in malignant:   VDAC1, CCT6A, ANAPC11, GGCT, TBRG4, SNRPE, UQCRQ, RAD50
  most significant controlled genes DOWN in malignant: RASSF3, CD7, STK38, PXN, SARAF, RASA3, R3HDM4, GLIPR1
  top 15 controlled genes by |knockdown z|:
            z_kd  lfc_cohort  padj_cohort
gene                                     
EVL       14.961      -0.636        0.000
LPAR6     13.893      -1.640        0.000
STK17B    11.406      -0.465        0.002
XAF1      11.354      -0.475        0.007
SYNE2     11.351      -0.758        0.000
TXNIP     10.581      -1.253        0.000
SOCS3     10.528      -0.819        0.001
HAPLN3    10.384      -0.600        0.000
ARHGAP25  10.273      -0.852        0.000
GBP5      10.206      -2.800        0.000
IFI6       9.546      -0.867        0.000
GIMAP7     9.524      -1.434        0.000
LTB        9.270      -1.534        0.000
SLFN5      9.118      -0.776        0.000
PARP8      8.937      -1.301        0.000
  enrichment over the authors' downstream programs (q < 0.1):                                       set  overlap  expected     q
                        c22 TCR signaling      438   177.736 0.000
                             c106 unknown      298   149.468 0.000
                        c21 TCR signaling      424   296.614 0.000
                              c93 unknown       62    12.585 0.000
                             c102 unknown      135    54.018 0.000
            c51 Oxidative phosphorylation      223   120.427 0.000
                        c99 TCR signaling      458   369.799 0.000
                      c68 tRNA processing      127    52.856 0.000
                              c60 unknown      111    44.724 0.000
                              c88 unknown      222   132.237 0.000
                      c48 tRNA processing      120    54.211 0.000
                              c80 unknown      177    96.612 0.000
                              c96 unknown      279   185.674 0.000
                              c30 unknown      169    94.289 0.000
                c31 Ragulator-Rag complex      168    94.483 0.000
                c32 Ragulator-Rag complex      161    90.804 0.000
                              c77 unknown      326   247.630 0.000
                             c101 unknown      111    58.277 0.000
c28 ribosomal and mitochrondrial function      309   234.077 0.000
               c87 Fanconi anemia complex      113    60.407 0.000
                              c95 unknown      122    68.345 0.000
                       c50 Arp2/3 complex       59    24.201 0.000
c15 ribosomal and mitochrondrial function      337   269.314 0.000
                c41 protein glycosylation      153    99.517 0.000
                    c47 CtBP core complex       47    20.329 0.000
                              c64 unknown       29     9.874 0.000
                              c54 unknown      126    82.479 0.000
                              c91 unknown       32    12.391 0.000
              c111 Arp2/3 and CCC complex       61    32.720 0.000
                          c76 EMC complex      245   196.516 0.000
                              c98 unknown       50    26.331 0.000
                 c38 MLL2 complex related       42    21.104 0.000
c58 ribosomal and mitochrondrial function      401   360.893 0.000
                             c105 unknown       18     6.583 0.000
                              c90 unknown       78    51.501 0.000
                              c35 unknown       93    64.279 0.000
                             c109 unknown       72    47.048 0.000
                             c107 unknown       33    17.619 0.000
                              c62 unknown       18     7.551 0.000
               
... [5703 more chars truncated]
```

## §7 — Full numerical report

Every number above, re-printed in one block and saved to `tables/pseq30_full_report_v1.txt`. Each section is
guarded, so a partial run prints the error for that block and continues.

```python  # cell 20
import io as _io
_buf = _io.StringIO()
def _p(*a, **k):
    print(*a, **k); print(*a, **k, file=_buf)
def _hdr(t): _p("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)
def _sub(t): _p("\n" + "-" * 100 + f"\n{t}\n" + "-" * 100)
def _sec(title, fn):
    _sub(title)
    try: fn()
    except Exception as e: _p(f"[not available in this run: {type(e).__name__}: {e}]")

_hdr("30_perturbseq_per_donor -- FULL REPORT")
_p(f"""
WHAT WAS DONE
Per-donor pert2state on the GWCD4i {H.CONDITION} knockdown resource. Cohort: nb29's MF CD4-clone skin donors with
>= {H.MIN_CELLS_ARM} malignant and >= {H.MIN_CELLS_ARM} reactive CD4 cells (30 donors). Per donor: {H.N_PSEUDO} + {H.N_PSEUDO} seeded pseudo-replicate
pseudobulks -> pydeseq2 ~ arm -> Wald stat = y_donor. Cohort reference: pydeseq2 ~ donor + arm on the same 30 donors.
Design built ONCE: {len(genes):,} genes (measured in Perturb-seq x testable in >= {H.GENE_FRAC_DONORS:.0%} donors) x {len(regs):,}
regulators (zscore_masked), column-standardized, TruncatedSVD D = {H.N_PCS}. Hyper-parameters frozen once on the cohort
signature (ElasticNetCV, 4-fold): alpha = {hp['alpha']:.4g}, l1_ratio = {hp['l1_ratio']:.4g}. Every fit = ElasticNet on the
same 60-dim projection with the same {H.N_SPLITS}-fold x {H.N_REPEATS} gene folds; w = V^T coef averaged over folds. Null =
the same donor fitted on one shared scrambled design (each gene's z permuted across regulators).
Sign: w > 0 knockdown mimics the malignant program; w < 0 the program depends on the regulator.
""")
_sec("1. Retained donors", lambda: _p(tab.loc[donors, ["n_malignant", "n_reactive", "study", "stage", "split"]].to_string()))
_sec("2. Model fit (sorted by held-out r)", lambda: (_p(f"cohort30 r = {r_cohort:.3f}"),
     _p(fit[["study", "n_min", "test_r", "test_r_se", "scrambled_r", "train_r", "r_vs_consensus_loo", "converged"]].round(3).to_string()),
     _p(f"Spearman rho(r, min n cells) = {rho:.3f} p = {p:.3g}"), _p("flags: " + ("; ".join(flags) if flags else "none"))))
_sec("3. Recurrence (top 30)", lambda: (_p(f"chance expectation {rec.attrs['expected_chance']:.2f}; recurrent (>= {H.MIN_RECUR}/30): {len(recurrent)}"),
     _p(rec.head(30)[["n_top", "n_bottom", "n_recur", "direction", "mean_w", "sd_w", "frac_same_sign", "q_binom", "cluster"]].round(3).to_string())))
_sec("4. Donor clustering", lambda: (_p("silhouette by k: " + str(cl["silhouette"].round(3).to_dict()) + f"; k = {cl['k']}"),
     _p(out[["cluster", "study", "stage", "n_min", "n_event_arms", "react_CD8A_frac", "react_FOXP3_frac", "test_r"]].round(3).to_string()),
     _p(cov.round(4).to_string(index=False)), _p(ct_study.to_string())))
_sec("5. Programs (q < 0.1)", lambda: (_p(sig_prog[["group", "set", "n", "mean_w", "delta_signed", "direction", "q_signed"]].round(3).to_string(index=False)),
     _p("significant in exactly one cluster: " + (", ".join(only_one) if only_one else "none"))))
_sec("6. Downstream of the top recurrent regulators", lambda: (
     [_p(f"{r}: {int((ctrl['regulator'] == r).sum())} KD-DE signature genes, {int(ctrl.loc[ctrl['regulator'] == r, 'controlled'].sum())} controlled; "
         f"top 15: " + ", ".join(ctrl[(ctrl['regulator'] == r) & ctrl['controlled']].head(15).index)) for r in top_regs],
     _p(enr[enr["q"] < 0.1][["regulator", "set", "overlap", "expected", "q"]].round(3).to_string(index=False) if len(enr) else "no enrichment rows")))
_sec("7. Outputs", lambda: _p("\n".join(str(p_) for p_ in (H.MODEL_FIT, H.WEIGHTS, H.RECURRENCE, H.CLUSTERS, H.PROGRAMS, H.DOWNSTREAM, H.REPORT))
     + "\nfigures: " + ", ".join(f"pseq30_{n}" for n in ("lic_per_donor", "lic_consensus_recurrent", "lic_consensus_programs", "model_fit", "donor_clustering", "downstream"))))
H.REPORT.write_text(_buf.getvalue())
print("\nsaved", H.REPORT)
```

```
====================================================================================================
30_perturbseq_per_donor -- FULL REPORT
====================================================================================================

WHAT WAS DONE
Per-donor pert2state on the GWCD4i Stim48hr knockdown resource. Cohort: nb29's MF CD4-clone skin donors with
>= 600 malignant and >= 600 reactive CD4 cells (30 donors). Per donor: 3 + 3 seeded pseudo-replicate
pseudobulks -> pydeseq2 ~ arm -> Wald stat = y_donor. Cohort reference: pydeseq2 ~ donor + arm on the same 30 donors.
Design built ONCE: 6,048 genes (measured in Perturb-seq x testable in >= 90% donors) x 2,582
regulators (zscore_masked), column-standardized, TruncatedSVD D = 60. Hyper-parameters frozen once on the cohort
signature (ElasticNetCV, 4-fold): alpha = 0.1887, l1_ratio = 0.01. Every fit = ElasticNet on the
same 60-dim projection with the same 5-fold x 3 gene folds; w = V^T coef averaged over folds. Null =
the same donor fitted on one shared scrambled design (each gene's z permuted across regulators).
Sign: w > 0 knockdown mimics the malignant program; w < 0 the program depends on the regulator.


----------------------------------------------------------------------------------------------------
1. Retained donors
----------------------------------------------------------------------------------------------------
                      n_malignant  n_reactive            study     stage        split
donor                                                                                
Li2024_atlas__CTCL5          7532        4073           li2024  advanced    discovery
Li2024_atlas__CTCL6          3937       10298           li2024     early    discovery
Li2024_atlas__CTCL7          3658       38432           li2024     early    discovery
D1__P204                     3444        5741  chennareddy2025  advanced    discovery
Li2024_atlas__CTCL13         5962        2884           li2024  advanced    discovery
Li2024_atlas__CTCL8          7379        2447           li2024     early  replication
Li2024_atlas__MF30           2342        1923           li2024  advanced  replication
D7__MF311                    5693        1919   rindler2021_mc  advanced    discovery
Li2024_atlas__MF28          10395        1904           li2024  advanced    discovery
D7__P84                      1892        3889   rindler2021_mc  advanced  replication
Li2024_atlas__CTCL3          8823        1778           li2024     early    discovery
D7__MF309                    5395        1748   rindler2021_mc  advanced    discovery
D11__P3                      7596        1589  brentuximab2026  advanced    discovery
Li2024_atlas__PT50           1464        1858           li2024     early  replication
D11__P6                      4575        1376  brentuximab2026  advanced    discovery
D11__P5                      6550        1370  brentuximab2026  advanced    discovery
Li2024_atlas__CTCL2          1357        2200           li2024     early    discovery
Li2024_atlas__PT56           1292        2478           li2024  advanced    discovery
Li2024_atlas__CTCL4          1241        7906           li2024     early    discovery
D11__P2                      9473        1063  brentuximab2026  advanced    discovery
D8__P150                     2393         912        alkon2024     early    discovery
D13__MF18                     826         935     gaydosik2022  advanced  replication
Li2024_atlas__PT53           1417         735           li2024  advanced    discovery
D15__MF48                    3650         709        il4ra2026  advanced  replication
Li2024_atlas__MF21           1761         708           li2024  advanced  replication
D6__MF2                       677         708     gaydosik2019  advanced  replication
D15__MF24                     643        1958        il4ra2026  advanced  replication
D15__MF36                     620        2182        il4ra2026  advanced    discovery
D8__P65         
... [37951 more chars truncated]
```
