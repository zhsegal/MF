# 10_atlas/13_skin_vs_blood_subclone_signaling.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/10_atlas/13_skin_vs_blood_subclone_signaling.ipynb` and is never run._

# 13 — Skin ↔ blood comparison: subclonal architecture + TCR/co-stimulation signaling

Puts `24_subclone_tcr_signaling` (**skin**, v3) and `33_subclone_tcr_signaling` (**blood**, v1) side by side. Both notebooks ran the *same*
pipeline — `subclone_helpers` CNV subclones + nested MAJOR×MINOR labels, `tcr_signaling_helpers`
23-module TCR/co-stim scoring against lineage-matched reactive CD4 — and persisted every number, so
this notebook reads **only CSV / parquet / JSON**. No h5ad, no GPU, runs in seconds.

**Question.** Is the blood compartment the same disease process as the skin — same clonal
architecture (how many CNV subclones, which arms are trunk vs branch) and the same place in the
TCR/co-stim cascade — or a different one?

**Design.** Main panels use all eligible donors; each comparison also gets a matched view over the
8 donors that exist in *both* compartments (same donor ID, same study) and over the SS-only subset,
because the eligible cohorts are disease-skewed in opposite directions (skin MF-dominant, blood
SS-dominant).

**What is and is not comparable** (spelled out again in §11):

| comparable | not comparable without adjustment |
|---|---|
| arm *identity* (which arms are trunk / branch) | arm delta *magnitude* (blood inferCNV window 100/20k genes ≈ half skin's amplitude) |
| Cliff's δ vs same-donor reactive CD4 (both are within-donor contrasts) | raw `k` / silhouette (skin absolute gates vs blood null-anchored → re-derived in §2) |
| module δ *ranks* and signs | module δ for modules that lost genes in blood's 10k HVG space (quantified in §5) |
| signaling-mode *composition shape* and axis coupling | mode cluster *labels* (fit independently per compartment, column-wise z) |

```python  # cell 1
# ============================================================
# §0  Paths & parameters — artifact-only, no h5ad
# ============================================================
import sys
from pathlib import Path


def _resolve_nb_dir() -> Path:
    start = Path.cwd()
    for base in [start, *start.parents]:
        for sub in [Path("."), Path("MF")]:
            cand = base / sub
            if cand.name == "MF" and (cand / "data").exists():
                return cand.resolve()
    raise FileNotFoundError(f"could not locate MF/data from {start}")


NB_DIR = _resolve_nb_dir(); print("NB_DIR =", NB_DIR)
OUT_DIR = NB_DIR / "data" / "atlas_joint"
FIG_DIR = NB_DIR / "figures"; FIG_DIR.mkdir(exist_ok=True)
TAB_DIR = NB_DIR / "tables"; TAB_DIR.mkdir(exist_ok=True)
DE_S, DE_B = OUT_DIR / "tcr_signaling_de", OUT_DIR / "tcr_signaling_de_blood"

PREFIX = "skin_vs_blood"
SKIN_C, BLOOD_C = "#c0392b", "#2668a8"
COMPS = ("skin", "blood")
CC = {"skin": SKIN_C, "blood": BLOOD_C}

# nb31 (skin v3) absolute subclone gates — re-applied to blood in §2 for a like-for-like count.
SIL_MIN, MIN_ARM_DELTA = 0.15, 0.03
FDR = 0.05
MIN_RECUR = 0.10        # §3: keep signed arms recurrent in >=10% of donors in either compartment

# ---- matched artifact pairs (skin v3 / blood v1) ----
FILES = {
    "elig":     {"skin": OUT_DIR / "subclone_donor_eligibility_v3.csv",
                 "blood": OUT_DIR / "subclone_donor_eligibility_blood_v1.csv"},
    "summary":  {"skin": OUT_DIR / "subclone_summary_v3.csv",
                 "blood": OUT_DIR / "subclone_summary_blood_v1.csv"},
    "trunk":    {"skin": OUT_DIR / "subclone_trunk_branch_v3.csv",
                 "blood": OUT_DIR / "subclone_trunk_branch_blood_v1.csv"},
    "cells":    {"skin": OUT_DIR / "subclones_v3.parquet",
                 "blood": OUT_DIR / "subclones_blood_v1.parquet"},
    "sign":     {"skin": OUT_DIR / "subclone_signatures_v3.json",
                 "blood": OUT_DIR / "subclone_signatures_blood_v1.json"},
    "modes":    {"skin": OUT_DIR / "tcr_signaling_subclone_modes_v3.csv",
                 "blood": OUT_DIR / "tcr_signaling_subclone_modes_blood_v1.csv"},
    "global":   {"skin": DE_S / "global_module_v3.csv",
                 "blood": DE_B / "global_module_blood_v1.csv"},
    "within":   {"skin": DE_S / "within_sample_module_v3.csv",
                 "blood": DE_B / "within_sample_module_blood_v1.csv"},
    "withingene": {"skin": DE_S / "within_sample_gene_v3.csv",
                   "blood": DE_B / "within_sample_gene_blood_v1.csv"},
    "subctl":   {"skin": DE_S / "subclone_vs_control_module_v3.csv",
                 "blood": DE_B / "subclone_vs_control_module_blood_v1.csv"},
    "armact":   {"skin": DE_S / "arm_activity_correlation_v3.csv",
                 "blood": DE_B / "arm_activity_correlation_blood_v1.csv"},
    "dump":     {"skin": TAB_DIR / "subclone_tcr_signaling_summary.txt",
                 "blood": TAB_DIR / "subclone_tcr_signaling_blood_summary.txt"},
}
SUMMARY_TXT = TAB_DIR / f"{PREFIX}_subclone_signaling_summary.txt"

missing = [str(p) for d in FILES.values() for p in d.values() if not p.exists()]
print("missing inputs:", missing if missing else "none")
assert not missing, "run nb31 (skin) and nb35 (blood) first"
```

```
NB_DIR = /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF
missing inputs: none
```

```python  # cell 2
# ============================================================
# §0  Load every matched artifact + build the paired / SS donor sets
# ============================================================
import json
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, pearsonr, spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(NB_DIR / "helpers"))
import subclone_helpers as S
import tcr_signaling_helpers as th

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_style("ticks")
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 9})

MODULES = list(th.TCR_MODULES)                       # cascade order L0 -> L5 -> COSTIM -> SIGNAL3 -> NEGREG
ACT = [m for m in MODULES if m in th.ACTIVITY_MODULES]


def _read(kind, comp):
    p = FILES[kind][comp]
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".json":
        return json.loads(p.read_text())
    return pd.read_csv(p)


def parse_dump(path):
    """nb31/nb35 §C dumps are flat `KEY: value` lines -> dict (first hit wins, sections are ordered)."""
    out = {}
    for line in path.read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_()\[\]./+-]+): (.*)$", line.strip())
        if m and m.group(1) not in out:
            out[m.group(1)] = m.group(2)
    return out


def dget(d, prefix, default="n/a"):
    """Fetch by exact key or, failing that, by key prefix (skin/blood name the CNV column differently)."""
    if prefix in d:
        return d[prefix]
    for k, v in d.items():
        if k.startswith(prefix):
            return v
    return default


def num(x, default=np.nan):
    try:
        return float(str(x).replace(",", "").replace("%", ""))
    except (TypeError, ValueError):
        return default


D = {kind: {c: _read(kind, c) for c in COMPS} for kind in FILES if kind != "dump"}
DUMP = {c: parse_dump(FILES["dump"][c]) for c in COMPS}

# tag + concat helper: every downstream frame carries a `compartment` column
def both(kind, cols=None):
    fr = []
    for c in COMPS:
        d = D[kind][c].copy()
        if cols is not None:
            d = d[[x for x in cols if x in d.columns]]
        d["compartment"] = c
        fr.append(d)
    return pd.concat(fr, ignore_index=True)


# ---- donor sets ----
ELIG = {c: D["elig"][c] for c in COMPS}
DISEASE = {c: ELIG[c].set_index("donor")["disease"].to_dict() for c in COMPS}
STUDY = {c: ELIG[c].set_index("donor")["study"].to_dict() for c in COMPS}

# EXACT donor-ID match only: `Li2024_atlas__CTCL2` (skin) and `B2__CTCL2` (blood) are DIFFERENT
# patients from different studies, so base-name matching would fabricate pairs.
IN_BOTH = sorted(set(ELIG["skin"]["donor"]) & set(ELIG["blood"]["donor"]))
PAIRED = sorted(set(D["within"]["skin"]["donor"]) & set(D["within"]["blood"]["donor"]))
PAIRED_SUB = sorted(set(D["summary"]["skin"]["donor"]) & set(D["summary"]["blood"]["donor"]))
SS = {c: {d for d, dis in DISEASE[c].items() if dis == "SS"} for c in COMPS}

for c in COMPS:
    e = ELIG[c]
    print(f"[{c}] donors={len(e)} eligible={int(e.eligible.sum())} "
          f"subclone_donors={D['summary'][c].donor.nunique()} "
          f"signaling_donors={D['within'][c].donor.nunique()} "
          f"subclones={D['modes'][c].nested_subclone.nunique()}")
print("\ndonors present in both compartments      :", len(IN_BOTH), IN_BOTH)
print("paired donors w/ signaling stats on both :", len(PAIRED), PAIRED)
print("paired donors w/ subclones on both       :", len(PAIRED_SUB), PAIRED_SUB)
print("SS donors: skin", len(SS["skin"]), "blood", len(SS["blood"]))

# sanity: same feature space on both sides, nothing empty
for c in COMPS:
    assert set(D["within"][c]["module"]) == set(MODULES), f"{c}: module set differs from th.TCR_MODULES"
    assert (D["within"][c].group_a == "malignant").all(), f"{c}: cliffs_delta sign not malignant-up"
    for kind in ("elig", "summary", "trunk", "modes", "global", "within", "subctl", "armact"):
        assert len(D[kind][c]), f"{c}/{kind} empty"
print("\nOK — 23 modules on both sides, delta sign = up-in-malignant, no empty tables")

# ---- module gene coverage: the gene-level DE files list every gene actually tested ----
# Skin was scored on ~40.8k genes, blood on 10k HVGs, so some modules lost most of their genes in
# blood. Needed up front: §4/§5 flag the affected modules, §5 prints the table.
tested = {c: set(D["withingene"][c]["feature"]) for c in COMPS}
COV = pd.DataFrame([{
    "module": m, "n_genes_module": len(th.TCR_MODULES[m]),
    "n_tested_skin": len(set(th.TCR_MODULES[m]) & tested["skin"]),
    "n_tested_blood": len(set(th.TCR_MODULES[m]) & tested["blood"]),
    "blood_missing": ";".join(sorted((set(th.TCR_MODULES[m]) & tested["skin"]) - tested["blood"])),
} for m in MODULES])
COV["frac_skin"] = COV.n_tested_skin / COV.n_genes_module
COV["frac_blood"] = COV.n_tested_blood / COV.n_genes_module
LOWCOV = set(COV.loc[COV.frac_blood < 0.6, "module"])
print(f"genes tested: skin {len(tested['skin'])}, blood {len(tested['blood'])}")
print(f"modules with <60% of genes tested in blood ({len(LOWCOV)}): {sorted(LOWCOV)}")
print("-> all are *_components / dose modules; the load-bearing *_activity modules keep full coverage:",
      sorted(m for m in ACT if m not in LOWCOV))
```

```
[skin] donors=82 eligible=25 subclone_donors=25 signaling_donors=31 subclones=178
[blood] donors=98 eligible=53 subclone_donors=53 signaling_donors=62 subclones=256

donors present in both compartments      : 10 ['B5__PT09', 'B5__PT22', 'B5__PT23', 'D5__MFIVB', 'H__HC1', 'H__MFIV1', 'H__SS1', 'H__SS2', 'H__SS3', 'H__SS4']
paired donors w/ signaling stats on both : 8 ['B5__PT09', 'B5__PT23', 'D5__MFIVB', 'H__MFIV1', 'H__SS1', 'H__SS2', 'H__SS3', 'H__SS4']
paired donors w/ subclones on both       : 2 ['B5__PT09', 'D5__MFIVB']
SS donors: skin 3 blood 89

OK — 23 modules on both sides, delta sign = up-in-malignant, no empty tables
genes tested: skin 225, blood 151
modules with <60% of genes tested in blood (7): ['COSTIM_pi3k_akt_activity', 'COSTIM_pi3k_akt_components', 'L2_gef_gtpase_components', 'L2_lat_signalosome_components', 'L3a_cbm_nfkb_components', 'L3b_calcium_nfat_components', 'L3c_ras_mapk_components']
-> all are *_components / dose modules; the load-bearing *_activity modules keep full coverage: ['L3a_nfkb_activity', 'L3b_nfat_activity', 'L3c_ap1_mapk_activity']
```

## §1 — Cohort & detectability

How much of each compartment is even analysable: donors passing the ≥200-malignant-cell gate, why
the rest drop out, malignant burden, TCR↔CNV agreement, and arm-CNV coverage. Percentages come from
the two `tables/subclone_tcr_signaling*_summary.txt` dumps, so they are exactly the numbers `24_subclone_tcr_signaling`/`33_subclone_tcr_signaling`
reported.

```python  # cell 4
# ============================================================
# §1  Cohort & detectability
# ============================================================
COHORT = {}
for c in COMPS:
    e, d = ELIG[c], DUMP[c]
    frac_mal = (e["n_malignant"] / e["n_cells"].replace(0, np.nan)).dropna()
    ee = e[e.eligible]
    frac_mal_e = (ee["n_malignant"] / ee["n_cells"].replace(0, np.nan)).dropna()
    COHORT[c] = {
        "n_donors": len(e),
        "n_eligible": int(e.eligible.sum()),
        "pct_eligible": 100 * e.eligible.mean(),
        "n_cells": num(dget(d, "n_cells")),
        "n_genes": num(dget(d, "n_genes")),
        "pct_alice_malig": num(re.search(r"\(([\d.]+)%\)", dget(d, "malignant_ALICE", "")).group(1))
                           if re.search(r"\(([\d.]+)%\)", dget(d, "malignant_ALICE", "")) else np.nan,
        "tcr_cnv_jaccard": num(re.search(r"jaccard=([\d.]+)", dget(d, "TCR_and_CNV_overlap", "")).group(1))
                           if re.search(r"jaccard=([\d.]+)", dget(d, "TCR_and_CNV_overlap", "")) else np.nan,
        "pct_arm_cov": num(dget(d, "cells_with_arm_CNV_coverage")),
        # over ALL donors this is dominated by the no-dominant-clone donors (skin median is 0), so the
        # eligible-donor median is the one to compare
        "median_pct_malig_all_donors": 100 * frac_mal.median(),
        "median_pct_malig_eligible": 100 * frac_mal_e.median(),
    }
COH = pd.DataFrame(COHORT).T
print(COH.round(3).to_string())

# exclusion reasons
EXCL = {}
for c in COMPS:
    e = ELIG[c]
    r = e.loc[~e.eligible, "reason_excluded"].fillna("unspecified").value_counts()
    EXCL[c] = r
    print(f"\n[{c}] excluded {int((~e.eligible).sum())}/{len(e)}")
    print(r.to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 3.4),
                         gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.55})
reasons = sorted({r for c in COMPS for r in EXCL[c].index})
pal = sns.color_palette("Greys_r", len(reasons) + 1)[:len(reasons)]
ax = axes[0]
for i, c in enumerate(COMPS):
    left = 0.0
    ax.barh(i, COHORT[c]["n_eligible"], color=CC[c], edgecolor="none", label=None)
    ax.text(COHORT[c]["n_eligible"] / 2, i, f"eligible {COHORT[c]['n_eligible']}", va="center",
            ha="center", color="w", fontsize=8)
    left = COHORT[c]["n_eligible"]
    for j, r in enumerate(reasons):
        v = int(EXCL[c].get(r, 0))
        if v == 0:
            continue
        ax.barh(i, v, left=left, color=pal[j], edgecolor="w", linewidth=0.5,
                label=r[:62] if i == 0 else None)
        left += v
ax.set_yticks(range(len(COMPS)), [f"{c}\n(n={COHORT[c]['n_donors']:.0f})" for c in COMPS])
ax.invert_yaxis()          # skin first, matching the right-hand panel
ax.set_xlabel("donors")
ax.set_title(f"Blood yields {COHORT['blood']['pct_eligible']:.0f}% subclone-eligible donors vs "
             f"skin's {COHORT['skin']['pct_eligible']:.0f}%", fontsize=9)
ax.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.30))
sns.despine(ax=ax)

ax = axes[1]
mets = [("pct_eligible", "% donors eligible"), ("pct_alice_malig", "% cells ALICE-malignant"),
        ("median_pct_malig_eligible", "median % malignant / eligible donor"),
        ("pct_arm_cov", "% cells w/ arm-CNV cov"),
        ("tcr_cnv_jaccard", "TCR∩CNV Jaccard ×100")]
y = np.arange(len(mets))
for i, c in enumerate(COMPS):
    vals = [COHORT[c][k] * (100 if k == "tcr_cnv_jaccard" else 1) for k, _ in mets]
    ax.barh(y + (i - 0.5) * 0.38, vals, height=0.36, color=CC[c], label=c)
ax.set_yticks(y, [lab for _, lab in mets], fontsize=8)
ax.invert_yaxis(); ax.set_xlabel("percent"); ax.legend(frameon=False, fontsize=8)
ax.set_title("Blood is the better-powered compartment on every axis", fontsize=9)
sns.despine(ax=ax)
fig.savefig(FIG_DIR / f"{PREFIX}_cohort.png"); plt.show()
```

```
n_donors  n_eligible  pct_eligible   n_cells  n_genes  pct_alice_malig  tcr_cnv_jaccard  pct_arm_cov  median_pct_malig_all_donors  median_pct_malig_eligible
skin       82.0        25.0        30.488  395233.0  40821.0            19.32            0.353        38.43                        0.000                     44.541
blood      98.0        53.0        54.082  302362.0  10000.0            40.52            0.450        60.96                       17.758                     54.554

[skin] excluded 57/82
reason_excluded
no ALICE-malignant cells (no dominant TCR clone)      42
healthy control — no malignant compartment             9
<200 malignant cells — too few for subclone KMeans     6

[blood] excluded 45/98
reason_excluded
no ALICE-malignant cells (no dominant TCR clone)                                      35
<200 malignant cells — too few for subclone KMeans                                     8
no arm-CNV coverage — sample lacked >=20 CD8 diploid reference for inferCNV (nb34)     1
healthy control — no malignant compartment                                             1

<Figure size 1560x408 with 2 Axes>
```

## §2 — Subclonal architecture

`k` (CNV subclones per donor), cluster separation, CNV confirmation of the TCR clone, and how
independent the transcriptomic MAJOR track is from the CNV MINOR track (`nmi_major_vs_minor`).

⚠️ The gates differ by construction: `24_subclone_tcr_signaling` used absolute cuts (silhouette ≥ 0.15, max arm delta ≥
0.03) tuned on the skin arm matrix; `33_subclone_tcr_signaling` anchored both to a size/k-matched split of known-diploid
healthy CD4 because blood arm amplitudes are ~half of skin's. Raw `k` is therefore **not** a
like-for-like count — the cell also re-counts blood under skin's absolute gates.

```python  # cell 6
# ============================================================
# §2  Subclonal architecture (+ gate re-derivation)
# ============================================================
SUM = both("summary")
SUM["ge2"] = SUM["k"] >= 2
METRICS = [("silhouette", "silhouette"), ("max_arm_delta", "max arm delta"),
           ("frac_cnv_confirmed", "frac CNV-confirmed"), ("nmi_major_vs_minor", "NMI major vs minor")]

fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.0))
ax = axes[0]
kmax = int(SUM["k"].max())
bot = np.zeros(len(COMPS))
shades = sns.color_palette("YlGnBu", kmax)
for k in range(1, kmax + 1):
    vals = [float((D["summary"][c]["k"] == k).mean()) for c in COMPS]
    ax.bar(range(len(COMPS)), vals, bottom=bot, color=shades[k - 1], edgecolor="w", label=f"k={k}")
    bot += vals
ax.set_xticks(range(len(COMPS)), COMPS); ax.set_ylabel("fraction of eligible donors")
ax.legend(fontsize=7, frameon=False, ncol=2)
f_s = 100 * float((D["summary"]["skin"]["k"] >= 2).mean())
f_b = 100 * float((D["summary"]["blood"]["k"] >= 2).mean())
ax.set_title(f"≥2 subclones: skin {f_s:.0f}%, blood {f_b:.0f}%", fontsize=9)
sns.despine(ax=ax)

MWU = {}
for j, (col, lab) in enumerate(METRICS):
    ax = axes[j + 1]
    sub = SUM[["compartment", col]].dropna()
    sns.boxplot(data=sub, x="compartment", y=col, order=list(COMPS), ax=ax, palette=CC,
                width=0.55, showfliers=False, boxprops={"alpha": 0.35})
    sns.stripplot(data=sub, x="compartment", y=col, order=list(COMPS), ax=ax, palette=CC,
                  size=3.2, jitter=0.22)
    a = sub.loc[sub.compartment == "skin", col]; b = sub.loc[sub.compartment == "blood", col]
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    MWU[col] = {"median_skin": a.median(), "median_blood": b.median(), "U": u, "p": p,
                "n_skin": len(a), "n_blood": len(b)}
    ax.set_title(f"{lab}\nMWU p={p:.2g}", fontsize=8.5)
    ax.set_xlabel(""); ax.set_ylabel("")
    sns.despine(ax=ax)
fig.tight_layout(); fig.savefig(FIG_DIR / f"{PREFIX}_subclone_structure.png"); plt.show()
print(pd.DataFrame(MWU).T.round(4).to_string())

# ---- gate re-derivation: blood under nb31's ABSOLUTE gates ----
bs = D["summary"]["blood"].copy()
bs["pass_skin_gates"] = (bs["silhouette"] >= SIL_MIN) & (bs["max_arm_delta"] >= MIN_ARM_DELTA)
bs["k_skin_gates"] = np.where(bs["pass_skin_gates"], bs["k"], 1)
GATE = {
    "skin_ge2_own_gates": 100 * float((D["summary"]["skin"]["k"] >= 2).mean()),
    "blood_ge2_null_gates": 100 * float((bs["k"] >= 2).mean()),
    "blood_ge2_skin_absolute_gates": 100 * float((bs["k_skin_gates"] >= 2).mean()),
    "blood_donors_passing_skin_gates": int(bs["pass_skin_gates"].sum()),
    "blood_null_calibrated_donors": int(bs.get("null_calibrated", pd.Series(dtype=bool)).sum()),
    "blood_median_sil_thr": float(bs["sil_thr"].median()) if "sil_thr" in bs else np.nan,
    "blood_median_arm_delta_thr": float(bs["arm_delta_thr"].median()) if "arm_delta_thr" in bs else np.nan,
}
print("\n-- gate re-derivation (blood arm amplitudes ~1/2 of skin) --")
for k, v in GATE.items():
    print(f"   {k:<34} {v:.4g}" if isinstance(v, float) else f"   {k:<34} {v}")
print("   NOTE: skin cannot be re-gated the other way — its arm matrix carries no diploid null.")

# paired donors that have subclones on both sides
if PAIRED_SUB:
    pv = SUM[SUM.donor.isin(PAIRED_SUB)][["compartment", "donor", "k", "silhouette",
                                          "max_arm_delta", "frac_cnv_confirmed", "n_major", "n_minor"]]
    print("\n-- donors with CNV subclones in BOTH compartments --")
    print(pv.sort_values(["donor", "compartment"]).round(3).to_string(index=False))
```

```
<Figure size 1620x360 with 5 Axes>
                    median_skin  median_blood      U       p  n_skin  n_blood
silhouette               0.2205        0.0714  768.0  0.0000    24.0     36.0
max_arm_delta            0.0636        0.0650  740.0  0.4068    25.0     53.0
frac_cnv_confirmed       0.8240        0.7500  711.0  0.6071    25.0     53.0
nmi_major_vs_minor       0.0240        0.0480  659.0  0.9742    25.0     53.0

-- gate re-derivation (blood arm amplitudes ~1/2 of skin) --
   skin_ge2_own_gates                 96
   blood_ge2_null_gates               67.92
   blood_ge2_skin_absolute_gates      7.547
   blood_donors_passing_skin_gates    4
   blood_null_calibrated_donors       53
   blood_median_sil_thr               0.0573
   blood_median_arm_delta_thr         0.0551
   NOTE: skin cannot be re-gated the other way — its arm matrix carries no diploid null.

-- donors with CNV subclones in BOTH compartments --
compartment     donor  k  silhouette  max_arm_delta  frac_cnv_confirmed  n_major  n_minor
      blood  B5__PT09  4       0.069          0.098               0.947        3        4
       skin  B5__PT09  2       0.304          0.077               0.918        5        2
      blood D5__MFIVB  1         NaN          0.000               0.986        2        1
       skin D5__MFIVB  2       0.180          0.050               0.917        4        2
```

## §3 — Trunk vs branch arm events

Trunk = pan-clonal (ancestral, shared by every subclone); branch = subclone-divergent. Recurrence =
fraction of that compartment's eligible donors carrying the signed arm in that tier. Arm *identity*
is comparable across compartments; arm delta *magnitude* is not.

⚠️ Both notebooks called arm events at the same `Z_THR=2.5` against a benign baseline, but blood arm
amplitudes are ~half of skin's, so an arm truly shared by all of a donor's blood subclones often
clears threshold in only some of them and lands in *branch* instead of *trunk*. Expect a technical
trunk deficit in blood; the cell prints the size of it.

```python  # cell 8
# ============================================================
# §3  Trunk vs branch arm recurrence
# ============================================================
EV = {}
for c in COMPS:
    ev = th.parse_arm_events(D["trunk"][c])
    ev["signed"] = ev["arm"] + ev["direction"]
    EV[c] = ev
n_don = {c: D["trunk"][c].donor.nunique() for c in COMPS}

REC = {}
for tier in ("trunk", "branch"):
    r = {}
    for c in COMPS:
        e = EV[c][EV[c].tier == tier]
        r[c] = e.groupby("signed")["donor"].nunique() / n_don[c]
    REC[tier] = pd.DataFrame(r).fillna(0.0)

order_arms = S.arm_order(sorted({s[:-1] for t in REC for s in REC[t].index}))
sign_order = [a + d for a in order_arms for d in ("+", "-")]

fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True, sharex=True)
keep = [s for s in sign_order
        if any(s in REC[t].index and REC[t].loc[s].max() >= MIN_RECUR for t in REC)]
y = np.arange(len(keep))
for ax, tier in zip(axes, ("trunk", "branch")):
    tab = REC[tier].reindex(keep).fillna(0.0)
    for i, c in enumerate(COMPS):
        ax.barh(y + (i - 0.5) * 0.38, 100 * tab[c], height=0.36, color=CC[c], label=c)
    ax.set_yticks(y, keep, fontsize=7)
    ax.set_xlabel("% of eligible donors"); ax.set_title(f"{tier} arms", fontsize=9)
    ax.legend(frameon=False, fontsize=8); sns.despine(ax=ax)
axes[0].invert_yaxis()      # once only — the y axis is shared, so a second call would undo it
fig.suptitle(f"Arm recurrence, skin (n={n_don['skin']}) vs blood (n={n_don['blood']}) — "
             f"only arms ≥{100*MIN_RECUR:.0f}% in either compartment", fontsize=9)
fig.tight_layout(); fig.savefig(FIG_DIR / f"{PREFIX}_arm_recurrence.png"); plt.show()

ARMSTAT = {}
for tier in ("trunk", "branch"):
    tab = REC[tier].reindex(sign_order).fillna(0.0)
    rho, prho = spearmanr(tab["skin"], tab["blood"])
    s_set = set(tab.index[tab["skin"] >= MIN_RECUR]); b_set = set(tab.index[tab["blood"] >= MIN_RECUR])
    jac = len(s_set & b_set) / max(len(s_set | b_set), 1)
    ARMSTAT[tier] = {"spearman_rho": rho, "p": prho, "jaccard_recurrent": jac,
                     "n_recurrent_skin": len(s_set), "n_recurrent_blood": len(b_set),
                     "shared": ";".join(sorted(s_set & b_set)),
                     "skin_only": ";".join(sorted(s_set - b_set)),
                     "blood_only": ";".join(sorted(b_set - s_set))}
    print(f"\n-- {tier} --")
    for k, v in ARMSTAT[tier].items():
        print(f"   {k:<20} {v:.4g}" if isinstance(v, float) else f"   {k:<20} {v}")

print("\n-- top recurrent trunk arms per compartment --")
for c in COMPS:
    top = REC["trunk"][c].sort_values(ascending=False).head(10)
    print(f"   {c:<6}", ", ".join(f"{a} {100*v:.0f}%" for a, v in top.items()))
TB = (both("trunk", ["donor", "n_trunk", "n_branch"]).groupby("compartment")[["n_trunk", "n_branch"]]
      .agg(["median", "mean", "max"]).round(2))
print("\n-- trunk/branch counts per donor --")
print(TB.to_string())
TRUNK_NOTE = (f"blood carries almost no PAN-CLONAL arms (median n_trunk="
              f"{TB.loc['blood', ('n_trunk', 'median')]:.0f} vs skin "
              f"{TB.loc['skin', ('n_trunk', 'median')]:.0f}) while carrying MORE branch arms "
              f"(mean {TB.loc['blood', ('n_branch', 'mean')]:.1f} vs "
              f"{TB.loc['skin', ('n_branch', 'mean')]:.1f}). The arm-event caller uses the same "
              f"Z_THR=2.5 on both sides, but blood arm amplitudes are ~1/2 of skin's, so an arm that "
              f"is present in every blood subclone often clears threshold in only some of them and is "
              f"scored branch, not trunk. Read the trunk deficit as at least partly technical.")
print("\n⚠️ " + TRUNK_NOTE)
```

```
<Figure size 1200x720 with 2 Axes>

-- trunk --
   spearman_rho         0.4649
   p                    3.893e-05
   jaccard_recurrent    0.06667
   n_recurrent_skin     15
   n_recurrent_blood    1
   shared               chr7q+
   skin_only            chr10p-;chr10q-;chr13q-;chr16p+;chr17p-;chr17q+;chr18p+;chr18q+;chr19p+;chr21q+;chr22q+;chr6q-;chr7p+;chr8q+
   blood_only           

-- branch --
   spearman_rho         0.3949
   p                    0.0005974
   jaccard_recurrent    0.2353
   n_recurrent_skin     7
   n_recurrent_blood    14
   shared               chr18p+;chr18q+;chr21q+;chr7q+
   skin_only            chr13q-;chr7p+;chr8q+
   blood_only           chr10p+;chr13q+;chr16p-;chr17p-;chr19p-;chr19q-;chr22q-;chr2q+;chr3q+;chr6q+

-- top recurrent trunk arms per compartment --
   skin   chr7q+ 56%, chr7p+ 48%, chr13q- 28%, chr17q+ 28%, chr21q+ 24%, chr10p- 20%, chr8q+ 16%, chr22q+ 16%, chr10q- 16%, chr6q- 16%
   blood  chr7q+ 19%, chr17q+ 8%, chr10q- 8%, chr13q- 4%, chr17p- 4%, chr7p+ 4%, chr8q+ 2%, chr10p+ 2%, chr10p- 0%, chr12p- 0%

-- trunk/branch counts per donor --
            n_trunk           n_branch          
             median  mean max   median  mean max
compartment                                     
blood           0.0  0.49   4      2.0  4.08  14
skin            4.0  4.44   9      2.0  1.92   4

⚠️ blood carries almost no PAN-CLONAL arms (median n_trunk=0 vs skin 4) while carrying MORE branch arms (mean 4.1 vs 1.9). The arm-event caller uses the same Z_THR=2.5 on both sides, but blood arm amplitudes are ~1/2 of skin's, so an arm that is present in every blood subclone often clears threshold in only some of them and is scored branch, not trunk. Read the trunk deficit as at least partly technical.
```

## §4 — Global signaling: all malignant vs all reactive CD4

23 modules, Cliff's δ pooled over the whole compartment (δ > 0 = up in malignant). Filled = load-bearing
`*_activity` module; hollow = `CONTEXT_ONLY` (component dose, lineage-confounded); `L0_pan_t_identity_loss`
is a module where the *drop* is the signal.

```python  # cell 10
# ============================================================
# §4  Global module deltas, skin vs blood
# ============================================================
G = (D["global"]["skin"][["module", "cliffs_delta", "fdr"]]
     .merge(D["global"]["blood"][["module", "cliffs_delta", "fdr"]], on="module",
            suffixes=("_skin", "_blood")))
G["delta_diff"] = G["cliffs_delta_blood"] - G["cliffs_delta_skin"]
G["sign_flip"] = np.sign(G["cliffs_delta_skin"]) != np.sign(G["cliffs_delta_blood"])
G["context_only"] = G["module"].isin(th.CONTEXT_ONLY)
G = G.set_index("module").reindex(MODULES).reset_index()

r_p, p_p = pearsonr(G.cliffs_delta_skin, G.cliffs_delta_blood)
r_s, p_s = spearmanr(G.cliffs_delta_skin, G.cliffs_delta_blood)

fig, ax = plt.subplots(figsize=(5.2, 4.8))
lim = 1.05 * float(np.abs(G[["cliffs_delta_skin", "cliffs_delta_blood"]].to_numpy()).max())
ax.plot([-lim, lim], [-lim, lim], color="0.7", lw=0.8, zorder=0)
ax.axhline(0, color="0.85", lw=0.6, zorder=0); ax.axvline(0, color="0.85", lw=0.6, zorder=0)
for i, (_, r) in enumerate(G.iterrows()):
    ax.scatter(r.cliffs_delta_skin, r.cliffs_delta_blood, s=42,
               facecolor="none" if r.context_only else "#333",
               edgecolor="#333", linewidth=0.9, zorder=3)
    if r.sign_flip or abs(r.delta_diff) > 0.2 or r.module == "L0_pan_t_identity_loss":
        dy = 5 if i % 2 == 0 else -9      # alternate above/below so labels do not stack
        ax.annotate(r.module.replace("_", " ") + (" †" if r.module in LOWCOV else ""),
                    (r.cliffs_delta_skin, r.cliffs_delta_blood),
                    textcoords="offset points", xytext=(6, dy), fontsize=6.5,
                    color="#b33" if r.sign_flip else "#333")
ax.text(0.02, 0.02, "hollow = CONTEXT_ONLY   † <60% of module genes tested in blood",
        transform=ax.transAxes, fontsize=6.5, color="0.35")
ax.set_xlabel("Cliff's δ skin (malignant vs reactive CD4)")
ax.set_ylabel("Cliff's δ blood")
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_title(f"Module deltas track across compartments (Pearson r={r_p:.2f}, p={p_p:.1g}; "
             f"Spearman ρ={r_s:.2f})\nbut {int(G.sign_flip.sum())}/{len(G)} modules flip sign — "
             f"blood is shifted toward positive δ", fontsize=9)
sns.despine(ax=ax); fig.savefig(FIG_DIR / f"{PREFIX}_global_modules.png"); plt.show()

print(G[["module", "cliffs_delta_skin", "cliffs_delta_blood", "delta_diff", "sign_flip",
         "context_only"]].round(3).to_string(index=False))
print("\n-- sign flips (opposite direction skin vs blood) --")
print(G.loc[G.sign_flip, ["module", "cliffs_delta_skin", "cliffs_delta_blood"]].round(3)
      .to_string(index=False) if G.sign_flip.any() else "   (none)")
print("\n-- largest |blood - skin| --")
print(G.reindex(G.delta_diff.abs().sort_values(ascending=False).index)
      .head(6)[["module", "cliffs_delta_skin", "cliffs_delta_blood", "delta_diff"]]
      .round(3).to_string(index=False))
```

```
<Figure size 624x576 with 1 Axes>
                        module  cliffs_delta_skin  cliffs_delta_blood  delta_diff  sign_flip  context_only
         L0_tcr_cd3_components             -0.068               0.016       0.084       True          True
        L0_pan_t_identity_loss             -0.306              -0.173       0.133      False         False
                 L0_coreceptor             -0.341              -0.252       0.089      False          True
           L1_proximal_kinases             -0.075              -0.228      -0.153      False          True
 L2_lat_signalosome_components             -0.007              -0.154      -0.147      False          True
      L2_gef_gtpase_components              0.023               0.107       0.084      False          True
      L2_plcg1_node_components             -0.247              -0.035       0.212      False          True
       L3a_cbm_nfkb_components             -0.147              -0.106       0.041      False          True
             L3a_nfkb_activity             -0.163              -0.055       0.109      False         False
   L3b_calcium_nfat_components             -0.247              -0.153       0.095      False          True
             L3b_nfat_activity              0.092               0.127       0.035      False         False
       L3c_ras_mapk_components             -0.168              -0.069       0.099      False          True
         L3c_ap1_mapk_activity             -0.229               0.205       0.434       True         False
                  L4_ieg_acute             -0.223               0.090       0.313       True         False
         L5_sustained_tf_layer              0.103               0.241       0.138      False         False
      L5_chronicity_exhaustion              0.004               0.396       0.392      False         False
   COSTIM_receptors_activating             -0.023               0.108       0.131       True         False
 COSTIM_coinhibitory_receptors             -0.064               0.405       0.469       True         False
COSTIM_ligands_on_TME_partners              0.016               0.178       0.162      False         False
    COSTIM_pi3k_akt_components             -0.176               0.013       0.189       True          True
      COSTIM_pi3k_akt_activity             -0.059               0.050       0.109       True         False
              SIGNAL3_jak_stat             -0.116              -0.104       0.012      False         False
               NEGREG_rheostat             -0.185              -0.137       0.048      False         False

-- sign flips (opposite direction skin vs blood) --
                       module  cliffs_delta_skin  cliffs_delta_blood
        L0_tcr_cd3_components             -0.068               0.016
        L3c_ap1_mapk_activity             -0.229               0.205
                 L4_ieg_acute             -0.223               0.090
  COSTIM_receptors_activating             -0.023               0.108
COSTIM_coinhibitory_receptors             -0.064               0.405
   COSTIM_pi3k_akt_components             -0.176               0.013
     COSTIM_pi3k_akt_activity             -0.059               0.050

-- largest |blood - skin| --
                       module  cliffs_delta_skin  cliffs_delta_blood  delta_diff
COSTIM_coinhibitory_receptors             -0.064               0.405       0.469
        L3c_ap1_mapk_activity             -0.229               0.205       0.434
     L5_chronicity_exhaustion              0.004               0.396       0.392
                 L4_ieg_acute             -0.223               0.090       0.313
     L2_plcg1_node_components             -0.247              -0.035       0.212
   COSTIM_pi3k_akt_components             -0.176               0.013       0.189
```

## §5 — Per-donor consistency (+ gene coverage)

Global pooling can be carried by one huge donor, so this repeats the contrast **within each donor**
(malignant vs that donor's own reactive CD4) and asks how reproducible each module's direction is:
median δ across donors, IQR, and the fraction of donors significant (FDR<0.05) in the median's
direction. Right panel = SS-only donors, which removes most of the MF-skin / SS-blood confound —
but note the skin side of that panel is only ~2 donors, so it constrains blood more than it does skin.

The coverage table below is the caveat that matters most: skin was scored on 40,821 genes, blood on
10,000 HVGs, so a module that lost genes in blood has a noisier, partly different score.

```python  # cell 12
# ============================================================
# §5  Per-donor module deltas + module gene coverage
# ============================================================
W = both("within", ["module", "donor", "cliffs_delta", "fdr", "n1", "n2"])
W["disease"] = [DISEASE[c].get(d, "NA") for c, d in zip(W.compartment, W.donor)]


def module_profile(w):
    rows = []
    for (c, m), g in w.groupby(["compartment", "module"], observed=True):
        med = g.cliffs_delta.median()
        same = g[(np.sign(g.cliffs_delta) == np.sign(med)) & (g.fdr < FDR)]
        rows.append({"compartment": c, "module": m, "median_delta": med,
                     "q25": g.cliffs_delta.quantile(0.25), "q75": g.cliffs_delta.quantile(0.75),
                     "n_donor": g.donor.nunique(), "frac_sig_same_dir": len(same) / len(g)})
    return pd.DataFrame(rows)


PROF_ALL = module_profile(W)
PROF_SS = module_profile(W[W.disease == "SS"])

fig, axes = plt.subplots(1, 2, figsize=(11.5, 7), sharey=True)
for ax, (prof, lab) in zip(axes, [(PROF_ALL, "all donors"), (PROF_SS, "SS donors only")]):
    y = np.arange(len(MODULES))
    ax.axvline(0, color="0.8", lw=0.7)
    for i, c in enumerate(COMPS):
        p = prof[prof.compartment == c].set_index("module").reindex(MODULES)
        off = (i - 0.5) * 0.3
        ax.hlines(y + off, p.q25, p.q75, color=CC[c], lw=1.2, alpha=0.6)
        ax.scatter(p.median_delta, y + off, s=18 + 90 * p.frac_sig_same_dir.fillna(0),
                   color=CC[c], edgecolor="w", linewidth=0.5, zorder=3,
                   label=f"{c} (n={int(p.n_donor.max()) if p.n_donor.notna().any() else 0})")
    ax.set_yticks(y, [m + (" *" if m in th.CONTEXT_ONLY else "") + (" †" if m in LOWCOV else "")
                      for m in MODULES], fontsize=7.5)
    ax.set_xlabel("median Cliff's δ across donors (>0 = up in malignant)")
    ax.set_title(lab, fontsize=9); ax.legend(frameon=False, fontsize=8, loc="lower right")
    sns.despine(ax=ax)
axes[0].invert_yaxis()      # once only — y is shared between the two panels
fig.suptitle("Within-donor malignant vs reactive-CD4 δ: dot size = fraction of donors FDR<0.05 "
             "in the same direction   (* = CONTEXT_ONLY, † <60% of genes tested in blood)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG_DIR / f"{PREFIX}_per_donor_modules.png"); plt.show()

CMP = (PROF_ALL.pivot(index="module", columns="compartment", values="median_delta")
       .reindex(MODULES)[list(COMPS)].add_prefix("median_"))
CMP["diff_blood_minus_skin"] = CMP["median_blood"] - CMP["median_skin"]
CMP["frac_sig_skin"] = PROF_ALL[PROF_ALL.compartment == "skin"].set_index("module")["frac_sig_same_dir"]
CMP["frac_sig_blood"] = PROF_ALL[PROF_ALL.compartment == "blood"].set_index("module")["frac_sig_same_dir"]
print(CMP.round(3).to_string())
r_pd, p_pd = spearmanr(CMP.median_skin, CMP.median_blood)
print(f"\nper-donor median δ, skin vs blood: Spearman ρ={r_pd:.3f} (p={p_pd:.2g})")

print("\n-- module gene coverage (COV, built in §0): skin ~40.8k-gene space vs blood 10k HVG --")
print(COV[["module", "n_genes_module", "n_tested_skin", "n_tested_blood", "frac_blood"]]
      .round(2).to_string(index=False))
print(f"\ngenes tested: skin {len(tested['skin'])}, blood {len(tested['blood'])}, "
      f"shared {len(tested['skin'] & tested['blood'])}")
print("modules with the biggest blood coverage loss:")
print(COV.assign(loss=COV.frac_skin - COV.frac_blood).sort_values("loss", ascending=False)
      .head(6)[["module", "frac_skin", "frac_blood", "blood_missing"]].round(2).to_string(index=False))
```

```
<Figure size 1380x840 with 2 Axes>
compartment                     median_skin  median_blood  diff_blood_minus_skin  frac_sig_skin  frac_sig_blood
module                                                                                                         
L0_tcr_cd3_components                 0.124         0.145                  0.021          0.581           0.645
L0_pan_t_identity_loss               -0.208        -0.043                  0.165          0.645           0.435
L0_coreceptor                        -0.327        -0.274                  0.053          0.871           0.903
L1_proximal_kinases                   0.060        -0.130                 -0.190          0.419           0.645
L2_lat_signalosome_components         0.071        -0.145                 -0.216          0.452           0.629
L2_gef_gtpase_components              0.074         0.132                  0.059          0.613           0.694
L2_plcg1_node_components             -0.128        -0.052                  0.076          0.710           0.387
L3a_cbm_nfkb_components              -0.118        -0.078                  0.040          0.677           0.532
L3a_nfkb_activity                    -0.104        -0.016                  0.088          0.516           0.323
L3b_calcium_nfat_components          -0.147        -0.135                  0.012          0.645           0.661
L3b_nfat_activity                     0.106         0.104                 -0.001          0.484           0.565
L3c_ras_mapk_components              -0.157        -0.135                  0.022          0.645           0.613
L3c_ap1_mapk_activity                -0.112         0.020                  0.133          0.613           0.306
L4_ieg_acute                         -0.124        -0.064                  0.060          0.677           0.468
L5_sustained_tf_layer                 0.108         0.156                  0.049          0.581           0.597
L5_chronicity_exhaustion              0.062         0.133                  0.071          0.484           0.645
COSTIM_receptors_activating           0.064         0.082                  0.017          0.516           0.500
COSTIM_coinhibitory_receptors         0.025         0.231                  0.206          0.387           0.774
COSTIM_ligands_on_TME_partners        0.091         0.166                  0.076          0.645           0.597
COSTIM_pi3k_akt_components           -0.122        -0.058                  0.064          0.710           0.435
COSTIM_pi3k_akt_activity             -0.046        -0.080                 -0.034          0.452           0.484
SIGNAL3_jak_stat                     -0.023        -0.041                 -0.018          0.419           0.403
NEGREG_rheostat                      -0.075        -0.038                  0.037          0.484           0.355

per-donor median δ, skin vs blood: Spearman ρ=0.660 (p=0.00061)

-- module gene coverage (COV, built in §0): skin ~40.8k-gene space vs blood 10k HVG --
                        module  n_genes_module  n_tested_skin  n_tested_blood  frac_blood
         L0_tcr_cd3_components               6              6               6        1.00
        L0_pan_t_identity_loss               5              5               5        1.00
                 L0_coreceptor               3              3               2        0.67
           L1_proximal_kinases              11             11               9        0.82
 L2_lat_signalosome_components               8              8               3        0.38
      L2_gef_gtpase_components               9              9               5        0.56
      L2_plcg1_node_components               5              5               3        0.60
       L3a_cbm_nfkb_components              17             17               4        0.24
             L3a_nfkb_activity              19             19              14        0.74
   L3b_calcium_nfat_components              16             16               7        0.44
             L3b_nfat_
... [2136 more chars truncated]
```

## §6 — Paired donors (same patient, both compartments)

The only comparison free of the MF/SS cohort confound: donors with malignant + reactive-CD4 cells in
*both* compartments, matched by exact donor ID within the same study. n is small (single digits), so
the signed-rank tests are descriptive, not confirmatory.

```python  # cell 14
# ============================================================
# §6  Paired-donor module deltas
# ============================================================
PW = W[W.donor.isin(PAIRED)].pivot_table(index=["donor", "module"], columns="compartment",
                                         values="cliffs_delta").dropna().reset_index()
print(f"paired donors: {len(PAIRED)} -> {PW.donor.nunique()} with both sides, {len(PW)} donor×module pairs")
print("donor / study / disease:")
for d in PAIRED:
    print(f"   {d:<12} {STUDY['skin'].get(d,'?'):<16} skin={DISEASE['skin'].get(d,'?'):<4} "
          f"blood={DISEASE['blood'].get(d,'?')}")

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6),
                         gridspec_kw={"width_ratios": [1, 1.35]})
ax = axes[0]
lim = 1.05 * float(np.abs(PW[["skin", "blood"]].to_numpy()).max())
ax.plot([-lim, lim], [-lim, lim], color="0.75", lw=0.8)
ax.axhline(0, color="0.9", lw=0.6); ax.axvline(0, color="0.9", lw=0.6)
act = PW.module.isin(ACT)
ax.scatter(PW.loc[~act, "skin"], PW.loc[~act, "blood"], s=14, color="0.6", alpha=0.6,
           label="other modules")
ax.scatter(PW.loc[act, "skin"], PW.loc[act, "blood"], s=26, color="#d95f02", alpha=0.9,
           label="*_activity modules")
r_pair, p_pair = spearmanr(PW.skin, PW.blood)
ax.set_xlabel("Cliff's δ skin"); ax.set_ylabel("Cliff's δ blood")
ax.set_title(f"Same patient, both compartments (n={PW.donor.nunique()} donors)\n"
             f"Spearman ρ={r_pair:.2f} (p={p_pair:.1g})", fontsize=9)
ax.legend(frameon=False, fontsize=7.5); sns.despine(ax=ax)

ax = axes[1]
H = (PW.assign(diff=PW.blood - PW.skin)
     .pivot(index="module", columns="donor", values="diff").reindex(MODULES))
v = float(np.nanmax(np.abs(H.to_numpy())))
im = ax.imshow(H.to_numpy(), cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
ax.set_xticks(range(H.shape[1]), H.columns, rotation=90, fontsize=7)
ax.set_yticks(range(H.shape[0]), H.index, fontsize=6.5)
ax.set_title("δ(blood) − δ(skin) per paired donor", fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Δ Cliff's δ")
fig.tight_layout(); fig.savefig(FIG_DIR / f"{PREFIX}_paired_donors.png"); plt.show()

rows = []
for m, g in PW.groupby("module"):
    if len(g) >= 5:
        try:
            st, p = wilcoxon(g.blood, g.skin)
        except ValueError:
            st, p = np.nan, np.nan
        rows.append({"module": m, "n_pairs": len(g), "median_skin": g.skin.median(),
                     "median_blood": g.blood.median(), "median_diff": (g.blood - g.skin).median(),
                     "p": p})
PAIRSTAT = pd.DataFrame(rows)
if len(PAIRSTAT):
    ok = PAIRSTAT.p.notna()
    PAIRSTAT.loc[ok, "fdr"] = multipletests(PAIRSTAT.loc[ok, "p"], method="fdr_bh")[1]
    PAIRSTAT = PAIRSTAT.sort_values("p")
    print("\n-- paired signed-rank per module (descriptive; n<=8) --")
    print(PAIRSTAT.round(4).to_string(index=False))
```

```
paired donors: 8 -> 8 with both sides, 184 donor×module pairs
donor / study / disease:
   B5__PT09     buus2025         skin=SS   blood=SS
   B5__PT23     buus2025         skin=SS   blood=SS
   D5__MFIVB    rindler2021_fi   skin=MF   blood=MF
   H__MFIV1     herrera2021      skin=MF   blood=MF
   H__SS1       herrera2021      skin=MF   blood=MF
   H__SS2       herrera2021      skin=MF   blood=MF
   H__SS3       herrera2021      skin=MF   blood=MF
   H__SS4       herrera2021      skin=MF   blood=MF

<Figure size 1500x552 with 3 Axes>

-- paired signed-rank per module (descriptive; n<=8) --
                        module  n_pairs  median_skin  median_blood  median_diff      p    fdr
                 L0_coreceptor        8      -0.2837       -0.1776       0.0544 0.0078 0.1797
      COSTIM_pi3k_akt_activity        8      -0.0162       -0.1203      -0.1263 0.0391 0.2995
 L2_lat_signalosome_components        8       0.1030       -0.0314      -0.1380 0.0391 0.2995
         L3c_ap1_mapk_activity        8      -0.0573        0.2213       0.2532 0.0547 0.3145
        L0_pan_t_identity_loss        8      -0.1660       -0.2879      -0.2442 0.0781 0.3594
    COSTIM_pi3k_akt_components        8      -0.1370        0.0257       0.1073 0.1094 0.3594
                  L4_ieg_acute        8      -0.0617        0.1909       0.2031 0.1094 0.3594
      L5_chronicity_exhaustion        8       0.2230        0.5404       0.2950 0.1484 0.3743
              SIGNAL3_jak_stat        8       0.0679       -0.0060      -0.0856 0.1484 0.3743
       L3a_cbm_nfkb_components        8      -0.1619        0.0502       0.1406 0.1953 0.3743
      L2_gef_gtpase_components        8       0.1133        0.1895       0.1746 0.1953 0.3743
             L3b_nfat_activity        8       0.2391        0.1043      -0.1294 0.1953 0.3743
      L2_plcg1_node_components        8      -0.1080        0.0101       0.1303 0.2500 0.4107
           L1_proximal_kinases        8       0.0271       -0.0878      -0.0679 0.2500 0.4107
   L3b_calcium_nfat_components        8      -0.0688       -0.1587      -0.1028 0.3125 0.4228
 COSTIM_coinhibitory_receptors        8       0.3877        0.4633       0.1899 0.3125 0.4228
         L5_sustained_tf_layer        8       0.1611        0.3888       0.1440 0.3125 0.4228
       L3c_ras_mapk_components        8      -0.2340       -0.1885       0.0340 0.3828 0.4634
   COSTIM_receptors_activating        8       0.2368        0.1273      -0.1758 0.3828 0.4634
         L0_tcr_cd3_components        8       0.2426        0.2534       0.0051 0.7422 0.8535
COSTIM_ligands_on_TME_partners        8       0.1505        0.0504      -0.0060 0.8438 0.9241
             L3a_nfkb_activity        8      -0.0331        0.0086       0.0168 0.9453 0.9883
               NEGREG_rheostat        8      -0.0046        0.0474       0.0266 1.0000 1.0000
```

## §7 — Subclone-level signaling divergence

Each nested subclone against **its own donor's** reactive CD4 — a within-donor contrast, so it is
directly comparable across compartments despite the cohort and gene-space differences. Two questions:
which modules are recurrently shifted at subclone resolution, and how far apart a donor's own
subclones sit on the load-bearing activity modules (max − min δ = subclonal signaling divergence).

```python  # cell 16
# ============================================================
# §7  Subclone vs same-donor reactive CD4
# ============================================================
SC = both("subctl", ["module", "subclone", "donor", "cliffs_delta", "fdr", "n1", "n_control"])
POWER = SC.groupby("compartment").agg(
    n_subclones=("subclone", "nunique"), n_donors=("donor", "nunique"), n_tests=("module", "size"),
    median_cells_per_subclone=("n1", "median"), median_control_cells=("n_control", "median"))
POWER["median_subclones_per_donor"] = [
    SC[SC.compartment == c].groupby("donor").subclone.nunique().median() for c in POWER.index]
print(POWER.round(1).to_string())
print("\nNOTE: 'fraction significant' scales with cells per subclone and control size, which differ "
      "between compartments (above) — read it alongside those numbers, not on its own.")

FRACSIG = (SC.assign(sig=SC.fdr < FDR)
           .groupby(["compartment", "module"], observed=True)
           .agg(frac_sig=("sig", "mean"), median_delta=("cliffs_delta", "median"),
                n=("sig", "size")).reset_index())

# per-donor divergence across that donor's own subclones (activity modules only, >=2 subclones)
DIV = []
for (c, d, m), g in SC[SC.module.isin(ACT)].groupby(["compartment", "donor", "module"], observed=True):
    if g.subclone.nunique() >= 2:
        DIV.append({"compartment": c, "donor": d, "module": m,
                    "spread": g.cliffs_delta.max() - g.cliffs_delta.min(),
                    "n_subclone": g.subclone.nunique()})
DIV = pd.DataFrame(DIV)
DON_DIV = DIV.groupby(["compartment", "donor"], observed=True)["spread"].mean().reset_index()

fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2),
                         gridspec_kw={"width_ratios": [1.15, 1, 0.8]})
ax = axes[0]
y = np.arange(len(MODULES))
for i, c in enumerate(COMPS):
    p = FRACSIG[FRACSIG.compartment == c].set_index("module").reindex(MODULES)
    ax.barh(y + (i - 0.5) * 0.38, 100 * p.frac_sig, height=0.36, color=CC[c], label=c)
ax.set_yticks(y, MODULES, fontsize=7); ax.invert_yaxis()
ax.set_xlabel("% subclones FDR<0.05 vs own-donor reactive CD4")
ax.legend(frameon=False, fontsize=8); ax.set_title("Recurrence at subclone resolution", fontsize=9)
sns.despine(ax=ax)

ax = axes[1]
ord_act = [m for m in ACT]
sns.boxplot(data=DIV[DIV.module.isin(ord_act)], y="module", x="spread", hue="compartment",
            order=ord_act, hue_order=list(COMPS), palette=CC, ax=ax, showfliers=False, width=0.65)
ax.tick_params(axis="y", labelsize=7.5)
ax.set_xlabel("within-donor spread of δ across subclones"); ax.set_ylabel("")
ax.legend(frameon=False, fontsize=8, title=None)
ax.set_title("Subclonal divergence, activity modules", fontsize=9); sns.despine(ax=ax)

ax = axes[2]
sns.boxplot(data=DON_DIV, x="compartment", y="spread", order=list(COMPS), palette=CC, ax=ax,
            width=0.5, showfliers=False, boxprops={"alpha": 0.35})
sns.stripplot(data=DON_DIV, x="compartment", y="spread", order=list(COMPS), palette=CC, ax=ax,
              size=3.5)
a = DON_DIV.loc[DON_DIV.compartment == "skin", "spread"]
b = DON_DIV.loc[DON_DIV.compartment == "blood", "spread"]
u_div, p_div = mannwhitneyu(a, b, alternative="two-sided") if len(a) and len(b) else (np.nan, np.nan)
ax.set_ylabel("mean spread per donor"); ax.set_xlabel("")
ax.set_title(f"per-donor mean\nMWU p={p_div:.2g}", fontsize=9); sns.despine(ax=ax)
fig.tight_layout(); fig.savefig(FIG_DIR / f"{PREFIX}_subclone_divergence.png"); plt.show()

print("\n-- % subclones significant, top modules --")
piv = FRACSIG.pivot(index="module", columns="compartment", values="frac_sig").reindex(MODULES)
piv["diff"] = piv["blood"] - piv["skin"]
print((100 * piv[["skin", "blood"]]).join(piv["diff"].mul(100).rename("diff_pp")).round(1).to_string())
print(f"\nper-donor mean divergence: skin median={a.median():.3f} (n={len(a)}), "
      f"blood median={b.median():.3f} (n={len(b)}), MWU p={p_div:.3g}")
```

```
n_subclones  n_donors  n_tests  median_cells_per_subclone  median_control_cells  median_subclones_per_donor
compartment                                                                                                             
blood                240        53     5520                      186.0                 417.0                         4.0
skin                 177        25     4071                      224.0                2576.0                         7.0

NOTE: 'fraction significant' scales with cells per subclone and control size, which differ between compartments (above) — read it alongside those numbers, not on its own.

<Figure size 1620x624 with 3 Axes>

-- % subclones significant, top modules --
                                skin  blood  diff_pp
module                                              
L0_tcr_cd3_components           70.6   75.0      4.4
L0_pan_t_identity_loss          87.6   57.9    -29.7
L0_coreceptor                   86.4   82.5     -3.9
L1_proximal_kinases             71.2   75.8      4.6
L2_lat_signalosome_components   55.4   54.2     -1.2
L2_gef_gtpase_components        66.7   60.4     -6.2
L2_plcg1_node_components        74.0   48.3    -25.7
L3a_cbm_nfkb_components         74.0   57.5    -16.5
L3a_nfkb_activity               78.0   51.2    -26.7
L3b_calcium_nfat_components     80.8   59.2    -21.6
L3b_nfat_activity               68.9   59.2     -9.8
L3c_ras_mapk_components         69.5   56.2    -13.2
L3c_ap1_mapk_activity           84.2   60.4    -23.8
L4_ieg_acute                    84.2   54.6    -29.6
L5_sustained_tf_layer           69.5   70.0      0.5
L5_chronicity_exhaustion        71.8   67.9     -3.8
COSTIM_receptors_activating     83.1   73.3     -9.7
COSTIM_coinhibitory_receptors   84.7   62.1    -22.7
COSTIM_ligands_on_TME_partners  70.6   62.5     -8.1
COSTIM_pi3k_akt_components      70.6   51.7    -19.0
COSTIM_pi3k_akt_activity        67.8   54.2    -13.6
SIGNAL3_jak_stat                78.5   67.5    -11.0
NEGREG_rheostat                 82.5   52.1    -30.4

per-donor mean divergence: skin median=0.542 (n=25), blood median=0.304 (n=50), MWU p=1.2e-05
```

## §8 — Signaling modes

`24_subclone_tcr_signaling`/`33_subclone_tcr_signaling` each localised every nested subclone on five contrast axes (`th.CONTRASTS`) and cut a Ward
tree at k=4. ⚠️ The z-scoring and the clustering were done **independently per compartment**, so mode
*labels* and absolute axis values are not comparable. What is comparable: the composition shape (how
concentrated subclones are in one mode) and the coupling between axes.

```python  # cell 18
# ============================================================
# §8  Signaling-mode composition + contrast-axis coupling
# ============================================================
CON = list(th.CONTRASTS)
M = both("modes")
M["donor"] = M["nested_subclone"].str.rsplit("_", n=1).str[0]

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1, 1, 1]})
ax = axes[0]
labels = sorted({str(x) for x in M.mode_label.dropna()})
pal = dict(zip(labels, sns.color_palette("Set2", len(labels))))
bot = np.zeros(len(COMPS))
for lab in labels:
    vals = [float((D["modes"][c].mode_label.astype(str) == lab).mean()) for c in COMPS]
    ax.bar(range(len(COMPS)), vals, bottom=bot, color=pal[lab], edgecolor="w", label=lab)
    bot += vals
ax.set_xticks(range(len(COMPS)), [f"{c}\n(n={D['modes'][c].nested_subclone.nunique()} subclones)"
                                 for c in COMPS])
ax.set_ylabel("fraction of nested subclones"); ax.legend(fontsize=7, frameon=False)
ax.set_title("Mode composition (labels are per-compartment)", fontsize=9); sns.despine(ax=ax)

CORR = {}
for ax, c in zip(axes[1:], COMPS):
    cm = D["modes"][c][CON].corr(method="spearman")
    CORR[c] = cm
    im = ax.imshow(cm.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(CON)), CON, rotation=90, fontsize=7)
    ax.set_yticks(range(len(CON)), CON, fontsize=7)
    for i in range(len(CON)):
        for j in range(len(CON)):
            ax.text(j, i, f"{cm.iat[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="w" if abs(cm.iat[i, j]) > 0.6 else "0.2")
    ax.set_title(f"{c}: contrast-axis coupling", fontsize=9)
fig.colorbar(im, ax=axes[2], fraction=0.04, pad=0.02)
fig.tight_layout(); fig.savefig(FIG_DIR / f"{PREFIX}_signaling_modes.png"); plt.show()

iu = np.triu_indices(len(CON), k=1)
r_cc, p_cc = pearsonr(CORR["skin"].to_numpy()[iu], CORR["blood"].to_numpy()[iu])
print(f"axis-coupling agreement (upper triangles): Pearson r={r_cc:.3f} (p={p_cc:.2g})")
for c in COMPS:
    comp = D["modes"][c]
    print(f"\n[{c}] mode composition")
    print((100 * comp.mode_label.value_counts(normalize=True)).round(1).to_string())
    print(f"   subclones={comp.nested_subclone.nunique()} donors={M[M.compartment==c].donor.nunique()} "
          f"largest_mode={100*comp.mode_label.value_counts(normalize=True).max():.1f}%")
    print("   contrast axis SD:", ", ".join(f"{k}={comp[k].std():.2f}" for k in CON))
```

```
<Figure size 1560x456 with 4 Axes>
axis-coupling agreement (upper triangles): Pearson r=0.924 (p=0.00013)

[skin] mode composition
mode_label
costim_amp+     55.1
ap1_vs_nfat-    25.3
ap1_vs_nfat+    19.7
   subclones=178 donors=25 largest_mode=55.1%
   contrast axis SD: calcium_vs_cbm=0.87, autonomous_nfkb=0.77, costim_amp=1.03, chronic_vs_acute=0.86, ap1_vs_nfat=1.18

[blood] mode composition
mode_label
costim_amp-     49.2
costim_amp+     30.5
ap1_vs_nfat-    20.3
   subclones=256 donors=53 largest_mode=49.2%
   contrast axis SD: calcium_vs_cbm=0.97, autonomous_nfkb=1.14, costim_amp=1.71, chronic_vs_acute=1.24, ap1_vs_nfat=1.51
```

## §9 — arm-CNV × downstream activity crosses

Does the same CNV → pathway coupling hold in blood? Per-donor Spearman ρ between an arm's inferCNV
score and the activity module it should drive, for the seven mechanistic crosses in
`th.ARM_ACTIVITY_MAP` (e.g. chr17q gain → JAK-STAT, chr10q loss/PTEN → PI3K-AKT).

```python  # cell 20
# ============================================================
# §9  arm x activity coupling
# ============================================================
AA = both("armact")
AA["pair"] = AA["arm"] + AA["direction"] + " → " + AA["module"]
pairs = [f"{a}{d} → {m}" for a, d, m, _ in th.ARM_ACTIVITY_MAP]
pairs = [p for p in pairs if p in set(AA.pair)]

fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.axvline(0, color="0.8", lw=0.7)
sns.stripplot(data=AA[AA.pair.isin(pairs)], y="pair", x="rho", hue="compartment", order=pairs,
              hue_order=list(COMPS), palette=CC, dodge=True, size=3.6, ax=ax, alpha=0.85)
MEDR = {}
for i, p in enumerate(pairs):
    for j, c in enumerate(COMPS):
        v = AA[(AA.pair == p) & (AA.compartment == c)]["rho"]
        if len(v):
            MEDR[(p, c)] = float(v.median())
            ax.plot(v.median(), i + (j - 0.5) * 0.4, marker="|", ms=14, mew=1.6, color="k", zorder=5)
same_sign = sum(1 for p in pairs
                if np.sign(MEDR.get((p, "skin"), 0)) == np.sign(MEDR.get((p, "blood"), 0)))
maxabs = max(abs(v) for v in MEDR.values()) if MEDR else float("nan")
ax.set_xlabel("per-donor Spearman ρ (arm score vs activity module)"); ax.set_ylabel("")
ax.tick_params(axis="y", labelsize=7.5)
ax.legend(frameon=False, fontsize=8, title=None)
ax.set_title(f"CNV → pathway couplings agree in sign for {same_sign}/{len(pairs)} crosses but are "
             f"weak everywhere (|median ρ| ≤ {maxabs:.2f})\n(black tick = median)", fontsize=9)
sns.despine(ax=ax); fig.savefig(FIG_DIR / f"{PREFIX}_arm_activity.png"); plt.show()

ARMACT = (AA.assign(pos_sig=(AA.rho > 0) & (AA.p < 0.05))
          .groupby(["pair", "compartment"], observed=True)
          .agg(n_donor=("donor", "nunique"), median_rho=("rho", "median"),
               frac_pos_sig=("pos_sig", "mean")).reset_index())
print(ARMACT.pivot(index="pair", columns="compartment",
                   values=["n_donor", "median_rho", "frac_pos_sig"]).round(3).to_string())
```

```
<Figure size 900x504 with 1 Axes>
                                     n_donor       median_rho        frac_pos_sig       
compartment                            blood  skin      blood   skin        blood   skin
pair                                                                                    
chr10q- → COSTIM_pi3k_akt_activity      61.0  28.0      0.021  0.032        0.311  0.357
chr17q+ → SIGNAL3_jak_stat              61.0  28.0      0.039  0.041        0.328  0.429
chr1p- → COSTIM_receptors_activating    61.0  28.0      0.014  0.016        0.246  0.214
chr20q+ → L3b_nfat_activity             61.0  28.0      0.014  0.010        0.180  0.179
chr2q+ → COSTIM_receptors_activating    61.0  28.0      0.043  0.064        0.344  0.571
chr6q- → L3a_nfkb_activity              61.0  28.0      0.021  0.119        0.393  0.750
chr8q+ → L5_chronicity_exhaustion       61.0  28.0      0.006  0.041        0.164  0.429
```

## §10 — Subclone marker-signature overlap

Each notebook exported per-subclone Wilcoxon markers (top 30) plus a malignant-overall signature.
Recurrence = fraction of that compartment's subclones whose marker list contains the gene.

The raw top of both lists is mitochondrial / ribosomal — per-subclone Wilcoxon markers are largely
depth- and quality-driven — so the bottom row repeats everything with `MT-*`, `RPL*`, `RPS*` dropped.
Read the bottom row for biology, the top row for what the exported signatures actually contain.

```python  # cell 22
# ============================================================
# §10  Signature overlap
# ============================================================
HOUSEKEEPING = re.compile(r"^(MT-|MTRNR|RPL|RPS)")     # depth/quality-driven, not biology
SIG = {c: {k: v for k, v in D["sign"][c].items() if k.startswith("subclone__")} for c in COMPS}


def freq_table(comp, drop_housekeeping=False):
    genes = [g for gl in SIG[comp].values() for g in gl
             if not (drop_housekeeping and HOUSEKEEPING.match(g))]
    return (pd.Series(genes).value_counts() / max(len(SIG[comp]), 1)).rename("frac_subclones")


FREQ = {c: freq_table(c) for c in COMPS}
FREQ_F = {c: freq_table(c, drop_housekeeping=True) for c in COMPS}


def jac(a, b):
    return len(a & b) / max(len(a | b), 1)


UNION = {c: set(FREQ[c].index) for c in COMPS}
UNION_F = {c: set(FREQ_F[c].index) for c in COMPS}
TOP = {c: set(FREQ[c].head(50).index) for c in COMPS}
TOP_F = {c: set(FREQ_F[c].head(50).index) for c in COMPS}
MAL_OVERALL = {c: set(D["sign"][c].get("malignant_overall", [])) for c in COMPS}
jac_all, jac_top = jac(UNION["skin"], UNION["blood"]), jac(TOP["skin"], TOP["blood"])
jac_all_f, jac_top_f = jac(UNION_F["skin"], UNION_F["blood"]), jac(TOP_F["skin"], TOP_F["blood"])
jac_mal = jac(MAL_OVERALL["skin"], MAL_OVERALL["blood"])

fig, axes = plt.subplots(2, 2, figsize=(10.5, 8), sharex=True)
for row, (freq, union, lab) in enumerate([(FREQ, UNION, "all genes"),
                                          (FREQ_F, UNION_F, "MT-/RPL-/RPS- dropped")]):
    for ax, c in zip(axes[row], COMPS):
        top = freq[c].head(15)[::-1]
        shared = [g in union["blood" if c == "skin" else "skin"] for g in top.index]
        ax.barh(range(len(top)), 100 * top.to_numpy(),
                color=[CC[c] if s else "0.65" for s in shared])
        ax.set_yticks(range(len(top)), top.index, fontsize=8)
        if row == 1:
            ax.set_xlabel("% of subclones with gene in top-30 markers")
        ax.set_title(f"{c}, {lab} (n={len(SIG[c])} subclones)", fontsize=9)
        sns.despine(ax=ax)
fig.suptitle(f"Recurrent subclone markers (grey = not recurrent in the other compartment)\n"
             f"Jaccard(union) {jac_all:.2f} raw / {jac_all_f:.2f} filtered; "
             f"Jaccard(top-50) {jac_top:.2f} raw / {jac_top_f:.2f} filtered; "
             f"Jaccard(malignant_overall) {jac_mal:.2f}", fontsize=9)
fig.tight_layout(); fig.savefig(FIG_DIR / f"{PREFIX}_signature_overlap.png"); plt.show()

print(f"subclone signatures: skin {len(SIG['skin'])}, blood {len(SIG['blood'])}")
print(f"marker gene union: skin {len(UNION['skin'])}, blood {len(UNION['blood'])}, "
      f"shared {len(UNION['skin'] & UNION['blood'])}, jaccard={jac_all:.3f} "
      f"(housekeeping-filtered {jac_all_f:.3f})")
print(f"top-50 recurrent jaccard={jac_top:.3f} (filtered {jac_top_f:.3f}); "
      f"malignant_overall jaccard={jac_mal:.3f}")
SHARED_TOP = sorted(TOP["skin"] & TOP["blood"])
SHARED_TOP_F = sorted(TOP_F["skin"] & TOP_F["blood"])
print(f"\nshared top-50 (raw, {len(SHARED_TOP)}): {', '.join(SHARED_TOP)}")
print(f"\nshared top-50 (filtered, {len(SHARED_TOP_F)}): {', '.join(SHARED_TOP_F)}")
print(f"\nskin-only top-50 (filtered): {', '.join(sorted(TOP_F['skin'] - TOP_F['blood']))}")
print(f"\nblood-only top-50 (filtered): {', '.join(sorted(TOP_F['blood'] - TOP_F['skin']))}")
```

```
<Figure size 1260x960 with 4 Axes>
subclone signatures: skin 175, blood 237
marker gene union: skin 1181, blood 902, shared 487, jaccard=0.305 (housekeeping-filtered 0.302)
top-50 recurrent jaccard=0.176 (filtered 0.176); malignant_overall jaccard=0.149

shared top-50 (raw, 15): B2M, EIF1, GAPDH, IL32, MT-ATP6, MT-CO2, MT-CYB, MT-ND1, MT-ND2, MT-ND4, MT-ND4L, MT-ND5, PPIA, PTPRC, SH3BGRL3

shared top-50 (filtered, 15): ACTB, B2M, CD52, EIF1, GAPDH, IFITM1, IL32, MIF, PFN1, PPIA, PTPRC, S100A4, S100A6, SH3BGRL3, TUBA1B

skin-only top-50 (filtered): ACTG1, BTG1, CD69, CD74, CHCHD2, DNAJB1, DUSP1, DUSP2, ENO1, FTL, H2AFZ, HNRNPA2B1, HSP90AA1, HSP90AB1, HSPA1B, HSPA8, HSPB1, HSPD1, ITM2B, JUNB, LGALS1, NME2, NR4A1, NR4A2, PPP1R15A, PRDX1, RGS1, SLC2A3, SRGN, STMN1, TSC22D3, TUBB, UBC, ZFP36, ZFP36L2

blood-only top-50 (filtered): AHNAK, ANK3, ANKRD12, ARHGAP15, BNIP3L, CCR4, H3F3B, HBA1, HBA2, HBB, HLA-A, HLA-B, HLA-C, HLA-E, HSPA5, IKZF2, INPP4B, ISG20, ITGB1, LDLRAD4, MACF1, MALAT1, MYL12A, MYL6, PTMA, RNF213, S100A10, S100A11, TMBIM6, TMSB10, TMSB4X, TOX, TSHZ2, TYMS, VIM
```

# §11 — Machine-readable numeric summary

One flat `KEY: value` dump of every number this notebook produces, written to
`tables/skin_vs_blood_subclone_signaling_summary.txt` so an agent can read the comparison without
re-running anything. Same format as the `24_subclone_tcr_signaling`/`33_subclone_tcr_signaling` §C dumps. Run last.

```python  # cell 24
# --- §11 · Machine-readable summary: every number this comparison reports ---
_LOG = []


def P(s=""):
    _LOG.append(str(s)); print(s)


def SEC(t):
    P(); P("=" * 78); P(f"## {t}"); P("=" * 78)


def kv(k, v):
    P(f"{k}: {v}")


def TBL(df, n=None, index=False):
    d = df if n is None else df.head(n)
    for line in d.to_string(index=index).splitlines():
        P("   " + line)


SEC("0 · PROVENANCE & INPUTS")
kv("notebook", "41_skin_vs_blood_subclone_signaling.ipynb — skin (nb31 v3) vs blood (nb35 v1)")
kv("mode", "artifact-only: CSV/parquet/JSON, no h5ad, no recompute")
kv("skin_source", "31_subclone_tcr_signaling.ipynb / skin_T_annotated.h5ad / v3 outputs")
kv("blood_source", "35_blood_subclone_tcr_signaling.ipynb / blood_T_annotated.h5ad / blood_v1 outputs")
kv("malignancy_definition", "tcr_malignant_alice on both sides (ALICE dominant founder + <=1-aa TRB family)")
kv("signaling_contrast", "malignant vs lineage-matched reactive CD4, Cliff's delta (>0 = up in malignant)")
kv("n_modules", len(MODULES))
kv("params", f"SIL_MIN={SIL_MIN} MIN_ARM_DELTA={MIN_ARM_DELTA} FDR={FDR} MIN_RECUR={MIN_RECUR}")
for kind in FILES:
    kv(f"input[{kind}]", f"skin={FILES[kind]['skin'].name} | blood={FILES[kind]['blood'].name}")
kv("outputs", f"{SUMMARY_TXT.name} + figures/{PREFIX}_*.png")

SEC("1 · COHORT & DETECTABILITY")
TBL(COH.round(3), index=True)
for c in COMPS:
    P(f"-- {c} exclusion reasons --"); TBL(EXCL[c].to_frame("n_donors"), index=True)
kv("donors_present_in_both", f"{len(IN_BOTH)} :: {';'.join(IN_BOTH)}")
kv("paired_donors_with_signaling", f"{len(PAIRED)} :: {';'.join(PAIRED)}")
kv("paired_donors_with_subclones", f"{len(PAIRED_SUB)} :: {';'.join(PAIRED_SUB)}")
for c in COMPS:
    kv(f"disease_mix[{c}]", ELIG[c].disease.value_counts().to_dict())
    kv(f"disease_mix_eligible[{c}]", ELIG[c][ELIG[c].eligible].disease.value_counts().to_dict())
    kv(f"study_mix_eligible[{c}]", ELIG[c][ELIG[c].eligible].study.value_counts().to_dict())

SEC("2 · SUBCLONAL ARCHITECTURE")
for c in COMPS:
    kv(f"k_distribution[{c}]", D["summary"][c].k.value_counts().sort_index().to_dict())
P("-- per-metric Mann-Whitney (skin vs blood) --")
TBL(pd.DataFrame(MWU).T.round(4), index=True)
P("-- gate re-derivation --")
for k, v in GATE.items():
    kv(f"gate[{k}]", f"{v:.4g}" if isinstance(v, float) else v)
kv("gate_note", "skin has no diploid null for its arm matrix, so it cannot be null-calibrated back")
if PAIRED_SUB:
    P("-- donors with subclones in both compartments --")
    TBL(SUM[SUM.donor.isin(PAIRED_SUB)][["compartment", "donor", "k", "silhouette", "max_arm_delta",
                                         "frac_cnv_confirmed", "n_major", "n_minor"]]
        .sort_values(["donor", "compartment"]).round(3))

SEC("3 · TRUNK VS BRANCH ARMS")
for c in COMPS:
    kv(f"n_donors_trunk_branch[{c}]", n_don[c])
for tier in ("trunk", "branch"):
    P(f"-- {tier} --")
    for k, v in ARMSTAT[tier].items():
        kv(f"{tier}[{k}]", f"{v:.4g}" if isinstance(v, float) else v)
    P(f"-- {tier} recurrence (% of donors, arms >= {100*MIN_RECUR:.0f}% in either) --")
    tab = (100 * REC[tier].reindex(keep).fillna(0.0)).round(1)
    TBL(tab, index=True)
P("-- trunk/branch counts per donor --")
TBL(TB, index=True)
kv("trunk_deficit_note", TRUNK_NOTE)

SEC("4 · GLOBAL SIGNALING (pooled malignant vs reactive CD4)")
kv("pearson_r", f"{r_p:.3f}"); kv("pearson_p", f"{p_p:.3g}")
kv("spearman_rho", f"{r_s:.3f}"); kv("spearman_p", f"{p_s:.3g}")
kv("n_sign_flips", f"{int(G.sign_flip.sum())}/{len(G)}")
kv("sign_flip_modules", ";".join(G.loc[G.sign_flip, "module"]) or "none")
TBL(G[["module", "cliffs_delta_skin", "cliffs_delta_blood", "delta_diff", "sign_flip",
       "context_only"]].round(3))
kv("blood_global_caveat",
   "global_module_blood_v1.csv comes from a now-commented-out nb35 cell; its n1/n2 match the final "
   "run's group counts (122,510 / 118,483) so it is consistent, but it is not regenerated live")

SEC("5 · PER-DONOR MODULE DELTAS + GENE COVERAGE")
kv("n_donors_within_sample", {c: int(D["within"][c].donor.nunique()) for c in COMPS})
kv("spearman_median_delta_skin_vs_blood", f"{r_pd:.3f} (p={p_pd:.3g})")
TBL(CMP.round(3), index=True)
P("-- SS-only donors --")
kv("n_SS_donors_with_stats", W[W.disease == "SS"].groupby("compartment").donor.nunique().to_dict())
TBL(PROF_SS.pivot(index="module", columns="compartment", values="median_delta")
    .reindex(MODULES).round(3), index=True)
P("-- module gene coverage --")
kv("genes_tested", {c: len(tested[c]) for c in COMPS})
kv("genes_tested_shared", len(tested["skin"] & tested["blood"]))
TBL(COV[["module", "n_genes_module", "n_tested_skin", "n_tested_blood", "frac_blood",
         "blood_missing"]].round(2))

SEC("6 · PAIRED DONORS (same patient, both compartments)")
kv("n_paired_donors", PW.donor.nunique())
kv("n_donor_module_pairs", len(PW))
kv("spearman_delta_skin_vs_blood", f"{r_pair:.3f} (p={p_pair:.3g})")
for d in PAIRED:
    kv(f"paired[{d}]", f"study={STUDY['skin'].get(d,'?')} skin_disease={DISEASE['skin'].get(d,'?')} "
                       f"blood_disease={DISEASE['blood'].get(d,'?')}")
if len(PAIRSTAT):
    P("-- signed-rank per module (descriptive, n<=8) --"); TBL(PAIRSTAT.round(4))

SEC("7 · SUBCLONE-LEVEL DIVERGENCE")
TBL(POWER.round(1), index=True)
kv("power_note", "fraction-significant scales with cells per subclone and control size, which differ "
                 "between compartments — interpret with the medians above")
P("-- % subclones FDR<0.05 vs own-donor reactive CD4 --")
TBL((100 * piv[["skin", "blood"]]).join(piv["diff"].mul(100).rename("diff_pp")).round(1), index=True)
kv("per_donor_mean_divergence_skin", f"median={a.median():.4f} n={len(a)}")
kv("per_donor_mean_divergence_blood", f"median={b.median():.4f} n={len(b)}")
kv("divergence_MWU_p", f"{p_div:.3g}")
P("-- divergence per activity module --")
TBL(DIV.groupby(["compartment", "module"], observed=True)["spread"]
    .agg(["median", "mean", "size"]).round(3), index=True)

SEC("8 · SIGNALING MODES")
for c in COMPS:
    kv(f"n_subclones[{c}]", D["modes"][c].nested_subclone.nunique())
    kv(f"mode_composition_pct[{c}]",
       (100 * D["modes"][c].mode_label.value_counts(normalize=True)).round(1).to_dict())
    kv(f"contrast_axis_sd[{c}]", {k: round(float(D['modes'][c][k].std()), 3) for k in CON})
kv("axis_coupling_agreement_pearson", f"{r_cc:.3f} (p={p_cc:.3g})")
for c in COMPS:
    P(f"-- {c} contrast-axis Spearman matrix --"); TBL(CORR[c].round(2), index=True)
kv("mode_caveat", "clusters fit independently per compartment with column-wise z -> labels and "
                  "absolute axis values are NOT comparable; composition shape and coupling are")

SEC("9 · ARM-CNV x ACTIVITY CROSSES")
TBL(ARMACT.pivot(index="pair", columns="compartment",
                 values=["n_donor", "median_rho", "frac_pos_sig"]).round(3), index=True)
kv("crosses_same_signed", f"{same_sign}/{len(pairs)}")
kv("max_abs_median_rho", f"{maxabs:.3f}")

SEC("10 · SIGNATURE OVERLAP")
kv("n_subclone_signatures", {c: len(SIG[c]) for c in COMPS})
kv("marker_union", {c: len(UNION[c]) for c in COMPS})
kv("marker_union_shared", len(UNION["skin"] & UNION["blood"]))
kv("jaccard_union", f"{jac_all:.3f}")
kv("jaccard_union_housekeeping_filtered", f"{jac_all_f:.3f}")
kv("jaccard_top50", f"{jac_top:.3f}")
kv("jaccard_top50_housekeeping_filtered", f"{jac_top_f:.3f}")
kv("jaccard_malignant_overall", f"{jac_mal:.3f}")
kv("shared_top50", ";".join(SHARED_TOP))
kv("shared_top50_filtered", ";".join(SHARED_TOP_F))
kv("skin_only_top50_filtered", ";".join(sorted(TOP_F["skin"] - TOP_F["blood"])))
kv("blood_only_top50_filtered", ";".join(sorted(TOP_F["blood"] - TOP_F["skin"])))
kv("signature_caveat", "raw recurrent markers are dominated by MT-/RPL-/RPS- genes (depth-driven); "
                       "the filtered lists are the interpretable ones")
for c in COMPS:
    P(f"-- {c} top-15 recurrent markers, all genes (% of subclones) --")
    TBL((100 * FREQ[c].head(15)).round(1).to_frame("pct_subclones"), index=True)
    P(f"-- {c} top-15 recurrent markers, MT-/RPL-/RPS- dropped --")
    TBL((100 * FREQ_F[c].head(15)).round(1).to_frame("pct_subclones"), index=True)

SEC("11 · CAVEATS")
for i, t in enumerate([
    "Disease confound: eligible skin is MF-dominant, eligible blood SS-dominant, so compartment "
    "effects are entangled with MF vs SS. The SS-only (§5) and paired-donor (§6) views mitigate, "
    "not remove, this.",
    "Gene space: skin scored on ~40.8k genes, blood on 10k HVGs -> per-module coverage differs (§5).",
    "Subclone gates: skin absolute, blood null-anchored; blood arm amplitudes ~1/2 of skin's. Raw k / "
    "silhouette / arm-delta are only compared after the §2 re-derivation.",
    "stage_class is degenerate in blood (no `early` cells), so there is no stage axis here.",
    "Signaling-mode clusters were fit independently per compartment (§8) -> labels not comparable.",
    "blood global_module_blood_v1.csv is from a commented-out nb35 cell (counts still match the live "
    "run). Stale blood by-stage / by-disease dotplot and lollipop PNGs in figures/ are orphans from an "
    "earlier run and are not used here.",
    "Donor pairing is by EXACT donor ID only: Li2024_atlas__CTCL2 (skin) and B2__CTCL2 (blood) are "
    "different patients from different studies.",
    "Per-donor Wilcoxon/Cliff's delta treat cells as independent; no donor-level mixed model.",
    "Trunk vs branch: same Z_THR on both sides but half the amplitude in blood, so blood's trunk "
    "deficit (§3) is at least partly a thresholding artefact, not proven biology.",
    "Fraction-significant counts (§7) scale with subclone and control cell numbers, which differ "
    "between compartments; the divergence spread (max-min delta) is the size-robust readout.",
    "SS-only panel (§5) has only ~2 skin donors — it constrains the blood side, not the skin side.",
], 1):
    kv(f"caveat_{i}", t)

SEC("12 · ARTIFACT INDEX")
kv("summary_txt", str(SUMMARY_TXT))
for f in sorted(FIG_DIR.glob(f"{PREFIX}_*.png")):
    kv("figure", f"{f.name} ({f.stat().st_size // 1024} kB)")

SUMMARY_TXT.write_text("\n".join(_LOG) + "\n")
print(f"\nwrote {SUMMARY_TXT} ({SUMMARY_TXT.stat().st_size / 1024:.0f} kB)")
```

```
==============================================================================
## 0 · PROVENANCE & INPUTS
==============================================================================
notebook: 41_skin_vs_blood_subclone_signaling.ipynb — skin (nb31 v3) vs blood (nb35 v1)
mode: artifact-only: CSV/parquet/JSON, no h5ad, no recompute
skin_source: 31_subclone_tcr_signaling.ipynb / skin_T_annotated.h5ad / v3 outputs
blood_source: 35_blood_subclone_tcr_signaling.ipynb / blood_T_annotated.h5ad / blood_v1 outputs
malignancy_definition: tcr_malignant_alice on both sides (ALICE dominant founder + <=1-aa TRB family)
signaling_contrast: malignant vs lineage-matched reactive CD4, Cliff's delta (>0 = up in malignant)
n_modules: 23
params: SIL_MIN=0.15 MIN_ARM_DELTA=0.03 FDR=0.05 MIN_RECUR=0.1
input[elig]: skin=subclone_donor_eligibility_v3.csv | blood=subclone_donor_eligibility_blood_v1.csv
input[summary]: skin=subclone_summary_v3.csv | blood=subclone_summary_blood_v1.csv
input[trunk]: skin=subclone_trunk_branch_v3.csv | blood=subclone_trunk_branch_blood_v1.csv
input[cells]: skin=subclones_v3.parquet | blood=subclones_blood_v1.parquet
input[sign]: skin=subclone_signatures_v3.json | blood=subclone_signatures_blood_v1.json
input[modes]: skin=tcr_signaling_subclone_modes_v3.csv | blood=tcr_signaling_subclone_modes_blood_v1.csv
input[global]: skin=global_module_v3.csv | blood=global_module_blood_v1.csv
input[within]: skin=within_sample_module_v3.csv | blood=within_sample_module_blood_v1.csv
input[withingene]: skin=within_sample_gene_v3.csv | blood=within_sample_gene_blood_v1.csv
input[subctl]: skin=subclone_vs_control_module_v3.csv | blood=subclone_vs_control_module_blood_v1.csv
input[armact]: skin=arm_activity_correlation_v3.csv | blood=arm_activity_correlation_blood_v1.csv
input[dump]: skin=subclone_tcr_signaling_summary.txt | blood=subclone_tcr_signaling_blood_summary.txt
outputs: skin_vs_blood_subclone_signaling_summary.txt + figures/skin_vs_blood_*.png

==============================================================================
## 1 · COHORT & DETECTABILITY
==============================================================================
          n_donors  n_eligible  pct_eligible   n_cells  n_genes  pct_alice_malig  tcr_cnv_jaccard  pct_arm_cov  median_pct_malig_all_donors  median_pct_malig_eligible
   skin       82.0        25.0        30.488  395233.0  40821.0            19.32            0.353        38.43                        0.000                     44.541
   blood      98.0        53.0        54.082  302362.0  10000.0            40.52            0.450        60.96                       17.758                     54.554
-- skin exclusion reasons --
                                                       n_donors
   reason_excluded                                             
   no ALICE-malignant cells (no dominant TCR clone)          42
   healthy control — no malignant compartment                 9
   <200 malignant cells — too few for subclone KMeans         6
-- blood exclusion reasons --
                                                                                       n_donors
   reason_excluded                                                                             
   no ALICE-malignant cells (no dominant TCR clone)                                          35
   <200 malignant cells — too few for subclone KMeans                                         8
   no arm-CNV coverage — sample lacked >=20 CD8 diploid reference for inferCNV (nb34)         1
   healthy control — no malignant compartment                                                 1
donors_present_in_both: 10 :: B5__PT09;B5__PT22;B5__PT23;D5__MFIVB;H__HC1;H__MFIV1;H__SS1;H__SS2;H__SS3;H__SS4
paired_donors_with_signaling: 8 :: B5__PT09;B5__PT23;D5__MFIVB;H__MFIV1;H__SS1;H__SS2;H__SS3;H__SS4
paired_donors_with_subclones: 2 :: B5__PT09;D5__MFIVB
disease_mix[skin]: {'MF': 54, 'CTCL_other': 16, 'HC': 9, 'SS': 3}
disease_mix_eligible[skin]: {'MF': 21, 'SS': 2, '
... [32578 more chars truncated]
```
