# 20_skin/23_malignancy_tcr_cnv.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/20_skin/23_malignancy_tcr_cnv.ipynb` and is never run._

# 23 — Malignant annotation on the v2 atlas: ALICE TCR clonality + inferCNV

Two independent lines of evidence for the same question, on the `21_reannotation` **v2** re-annotation
`data/atlas_joint/skin_T_annotated.h5ad` (539,916 T cells / 146 donors / 174 samples / 14 studies,
against v1's 395,233 / ~60 / 7 studies):

| | steps | unit | output |
|---|---|---|---|
| **TCR clonality** | 1–4 | ALICE per donor, reported per sample | `tables/skin_tcr_clonality_per_sample.csv` |
| **CNV** | 5–9 | inferCNV per donor | `skin_T_malignancy_v5.parquet` |

The TCR half answers one question — **which samples carry a clonal CD4 population, how large, and
what kind of sample is it** — and ends in a single table. (It used to run three ALICE analyses:
the cross-patient convergence test returned 0 hits out of 30 pooled founders and the reactive-CD8
pass answers a different question about the TIL infiltrate; both were cut and are recoverable from
git history.)

The object already carries TCR obs (`tra_cdr3`/`trb_cdr3`/`has_tcr`), MrVI latents (`X_mrvi_u`), and
raw counts, so it *replaces* the old hand-assembled `skin_T_tcr_malig_v2.h5ad` — no separate curated
object is written. Helpers are reused verbatim (`alice_helpers`, `skin_T_cnv_helpers`).

Seven of the 14 v2 skin cohorts carry no V(D)J at all, so Step 1 gates the TCR and CNV arms
**separately**: `obs["tcr_cohort"]` marks which donors have a repertoire, and no downstream step is
allowed to read missing TCR data as a negative TCR call. Caches are versioned by filename only —
TCR/ALICE are `_v4`, CNV is `_v5`; the v3 and v4 CNV caches stay on disk (`24_subclone_tcr_signaling` reads v4, `old/37_semantic_clonality_transfer` reads v3).

> **HEAVY** cells (TCR-object build, OLGA Pgen sweep, per-donor inferCNV, per-arm inferCNV) run on the
> GPU/compute kernel — not the login node.

```python  # cell 1
# ============================================================
# Parameters — inputs, cohort gates, ALICE + inferCNV knobs.
# ============================================================
import hashlib, json
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

# ---- inputs (the nb10b re-annotation on the **v2** atlas) ----
# v2 is a different cohort, not a refresh: 539,916 T cells / 146 donors / 174 samples / 14 studies,
# against v1's 395,233 / ~60 / 7 studies. Every cache below is versioned by filename only, so each
# one that survived from the v1 run would be loaded silently and the notebook would report v1
# numbers on a v2 input. Hence the bump: TCR object + ALICE -> v4, CNV -> v5. Delete nothing —
# nb31 / nb37 still read skin_T_malignancy_v3.
OBJ     = OUT_DIR / "skin_T_annotated.h5ad"              # nb10b v2 re-annotation (cell_type_T)
TCR_OBJ = OUT_DIR / "skin_T_tcr_annotated_v4.h5ad"       # fresh TCR-complete cache (built on first run)
LI_TCR  = NB_DIR / "data" / "Li2024_atlas" / "li2024_tcr_malignancy.parquet"
GTF           = NB_DIR / "data" / "cache" / "Homo_sapiens.GRCh38.110.chr.gtf.gz"
INTEGRATED_H5 = NB_DIR / "data" / "Integrated_CTCL_skincellatlas_final_portal_tags.h5ad"

# ---- clonality rule + cohort gates (see Step 1 / Step 4) ----
FRAC_THRESH, RATIO_THRESH, EXPANDED_MIN = 0.05, 1.33, 2
TCR_MIN_CELLS = 300   # ALICE arm: a repertoire this thin cannot calibrate an OLGA null
CNV_MIN_T     = 200   # CNV arm: fewer T than this cannot anchor a CNV cluster (MIN_CLUSTER_FRAC)
# v2 vocabulary: cell_type_T = {CD4, CD8, CD4_Treg} (v1's UNK is gone — nb10b §11.5 resolved it).
# CD4_Treg stays OUT of the clonal cohort, as v1's "Tregs" did: reactive Tregs expand clonally too,
# so a Treg-labelled dominant clone is not evidence of tumour. They keep a CNV score, so a FOXP3+
# tumour clone can still surface through the CNV arm.
CD4_T_TYPES = ["CD4"]
DROP_ENTITIES = {"MF_gamma_delta", "CD8_aggressive_epidermotropic_CTCL"}

# ---- ALICE params ----
ALPHA = 0.05
Q = None              # None -> calibrate per repertoire from the null bulk
SEED = 0

# ---- TCR outputs (v4 = built on the nb10b v2 re-annotation) ----
ALICE_CD4_PARQUET = OUT_DIR / "alice_cd4_per_donor_v4.parquet"   # per-clonotype ALICE result
ALICE_MAL_PARQUET = OUT_DIR / "alice_malignancy_v4.parquet"      # per-cell tcr_clonal
CLONALITY_CSV     = TAB_DIR / "skin_tcr_clonality_per_sample.csv"  # <- the conclusive table

# ---- CNV outputs (v5 = the v4 null-anchored arm-consensus caller, unchanged, rebuilt on v2).
#      v3 (nb31/nb37) and v4 (the v1-cohort run) both stay on disk; neither is overwritten. ----
OUT_PARQUET = OUT_DIR / "skin_T_malignancy_v5.parquet"           # per-cell malignancy (v5)
SUMMARY_CSV = TAB_DIR / "skin_cnv_v5_summary.csv"                # <- the conclusive CNV table
# CNV_CACHE / ARM_CACHE (v5) are defined alongside the estimator in Step 5.
```

```
NB_DIR = /home/projects/nyosef/zvise/MF
```

```python  # cell 2
import sys, gc, importlib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc

sys.path.insert(0, str(NB_DIR / "helpers"))
import atlas_join_helpers as H
import skin_T_cnv_helpers as C
import alice_helpers as A
for _m in (H, C, A):
    importlib.reload(_m)
np.random.seed(SEED)
sc.settings.verbosity = 1
```

```
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
```

## Step 1 — load the TCR-complete object, find the dominant CD4 clone, gate the cohort · HEAVY

Loads `skin_T_annotated.h5ad` and folds in the Li2024 V(D)J parquet (~12 GB cache, built once).
Dominance is judged **on CD4 only** (`dom_mask`) so a co-expanded reactive CD8 clone cannot win the
"top clone" contest in a donor whose infiltrate is larger than its tumour.

Two gates, because the two halves of this notebook need different things from a donor:

| gate | who it selects | rule |
|---|---|---|
| `keep_donors` | everything downstream (incl. inferCNV, Steps 9–15) | ≥ `CNV_MIN_T` T cells |
| `tcr_donors`  | the ALICE clonality call (Steps 2–4) | ≥ `TCR_MIN_CELLS` TCR+ CD4 cells |

inferCNV needs no repertoire, so seven v2 cohorts with no V(D)J at all (brentuximab26, lyp26,
rindler21mc, alkon24, gaydosik19, gaydosik23, jonak21) stay in the object and get a CNV score;
`obs["tcr_cohort"]` marks them so nothing downstream reads "no TCR data" as "TCR-negative".

```python  # cell 4
adata = C.build_or_load_tcr_object(OBJ, TCR_OBJ, LI_TCR, H)   # ~12 GB cache built on first run
adata.obs["cell_type_T2"] = adata.obs["cell_type_T"].astype(str)   # shared-helper alias
is_cd4 = adata.obs["cell_type_T2"].isin(CD4_T_TYPES).to_numpy()
is_li  = adata.obs["study"].astype(str).eq("li2024").to_numpy()
print("cell_type_T:\n", adata.obs["cell_type_T"].value_counts())

# Unified TRB clone key + per-donor dominant clone, decided within CD4 only. Sets obs:
# tcr_clone_id / tcr_clone_size / tcr_is_expanded / tcr_is_dominant_clone / tcr_is_malignant.
dom_tbl = C.recompute_dominant_clone(adata, H, is_li, FRAC_THRESH, RATIO_THRESH, EXPANDED_MIN,
                                     dom_mask=is_cd4)
# One row per (donor, TRB CDR3) over CD4 cells — the repertoire ALICE runs on. Built before the
# gates so the TCR gate can read its cell counts straight off it.
clono_cd4 = A.clonotype_table(adata.obs[is_cd4], group="donor")

META_COLS = ["study", "dataset", "disease", "disease_stage", "stage_class", "entity", "tissue",
             "tissue_detail", "lesion_type", "cohort_country", "cohort_region",
             "treatment_context", "blood_involvement", "sex", "tech"]
meta_cols = [c for c in META_COLS if c in adata.obs.columns]
donor_meta = (adata.obs[["donor", *meta_cols]].astype(str)
              .groupby("donor", observed=True)
              .agg(lambda s: ", ".join(sorted(s[s != "nan"].unique()))))

# ---- gate 1: keep_donors (everything, incl. CNV) ----
n_T = adata.obs["donor"].astype(str).value_counts()
drop = {}
if {"D5__MFIVB", "D1__P303"} <= set(donor_meta.index):
    drop["D1__P303"] = "duplicate of D5__MFIVB"
for d in donor_meta.index[donor_meta["entity"].isin(DROP_ENTITIES)]:
    drop.setdefault(d, "non-ab-CD4 entity")
for d in n_T.index[n_T < CNV_MIN_T]:
    drop.setdefault(str(d), f"n_T < {CNV_MIN_T}")
keep_donors = [d for d in donor_meta.index if d not in drop]

# ---- gate 2: tcr_donors (ALICE only). herrera2021 is exempt (small but deeply sequenced); HC
# donors are exempt because they are never query patients — they are the diploid CNV reference plus
# the held-out negative control (Step 9), so repertoire depth is irrelevant to their role. ----
_meta = donor_meta.loc[keep_donors]
n_tcr_cd4 = (clono_cd4.groupby("donor", observed=True)["n_cells"].sum()
             .reindex(keep_donors).fillna(0).astype(int))
_exempt = _meta["study"].eq("herrera2021") | _meta["disease"].eq("HC")
tcr_donors = sorted(n_tcr_cd4.index[(n_tcr_cd4 >= TCR_MIN_CELLS) | _exempt])

adata = adata[adata.obs["donor"].isin(keep_donors)].copy()
adata.obs["tcr_cohort"] = adata.obs["donor"].astype(str).isin(tcr_donors)
clono_cd4 = clono_cd4[clono_cd4["donor"].isin(tcr_donors)].reset_index(drop=True)

print("dropped donors:", drop)
print(f"kept: {len(keep_donors)} donors / {adata.n_obs:,} T cells / "
      f"{adata.obs['sample_id'].nunique()} samples | HC donors: {int(_meta['disease'].eq('HC').sum())}")
print(f"TCR (ALICE) cohort: {len(tcr_donors)} donors / {int(adata.obs['tcr_cohort'].sum()):,} cells"
      f" | CNV-only: {len(keep_donors) - len(tcr_donors)} donors /"
      f" {int((~adata.obs['tcr_cohort']).sum()):,} cells")
print("CD4 clonotypes:", len(clono_cd4), "| donors:", clono_cd4["donor"].nunique(),
      "| donors with a dominant clone:", int(clono_cd4.groupby("donor")["is_founder"].any().sum()))
```

```
loaded cached TCR-complete object -> skin_T_tcr_annotated_v4.h5ad (539916, 42347)

study:
 study
li2024             233430
chennareddy2025     83679
brentuximab2026     40964
lyp2026             36899
buus2025            36576
rindler2021_mc      28555
il4ra2026           26049
alkon2024           18927
gaydosik2019        10925
gaydosik2022        10010
brunner2024          9132
jonak2021            2799
gaydosik2023         1147
herrera2021           824
Name: count, dtype: int64

has_tcr by study:
                    size     sum
study                          
alkon2024         18927       0
brentuximab2026   40964       0
brunner2024        9132    7568
buus2025          36576   20742
chennareddy2025   83679   52977
gaydosik2019      10925       0
gaydosik2022      10010    6379
gaydosik2023       1147       0
herrera2021         824     632
il4ra2026         26049   13890
jonak2021          2799       0
li2024           233430  114700
lyp2026           36899       0
rindler2021_mc    28555       0
cached tumor_cell: 100490 / 539916

cell_type_T:
 cell_type_T
CD4         442982
CD8          55923
CD4_Treg     41011
Name: count, dtype: int64

dropped donors: {'D1__P124': 'non-ab-CD4 entity', 'D1__P127': 'non-ab-CD4 entity', 'D1__P195': 'non-ab-CD4 entity', 'D1__P202': 'non-ab-CD4 entity', 'D1__P218': 'non-ab-CD4 entity', 'D1__P230': 'non-ab-CD4 entity', 'D1__P318_thick': 'non-ab-CD4 entity', 'D1__P318_thin': 'non-ab-CD4 entity', 'D1__P71': 'non-ab-CD4 entity', 'D1__P95': 'non-ab-CD4 entity', 'D1__P97': 'non-ab-CD4 entity', 'H__SS2': 'n_T < 200', 'D14__SC405': 'n_T < 200', 'D14__SC320': 'n_T < 200', 'H__SS1': 'n_T < 200', 'D7__P90': 'n_T < 200', 'D14__SC354': 'n_T < 200', 'D8__P144': 'n_T < 200', 'Li2024_atlas__CTCL15': 'n_T < 200', 'Li2024_atlas__CTCL11': 'n_T < 200', 'H__MFIV1': 'n_T < 200', 'D6__NOR3': 'n_T < 200', 'D15__NS16': 'n_T < 200', 'H__SS3': 'n_T < 200', 'D1__P121': 'n_T < 200', 'D6__NOR4': 'n_T < 200', 'D6__NOR2': 'n_T < 200', 'D6__NOR1': 'n_T < 200', 'D15__MF31': 'n_T < 200', 'D15__MF22': 'n_T < 200', 'D13__SC327_HTO1': 'n_T < 200', 'H__HC1': 'n_T < 200'}
kept: 114 donors / 492,653 T cells / 143 samples | HC donors: 11
TCR (ALICE) cohort: 52 donors / 281,229 cells | CNV-only: 62 donors / 211,424 cells
CD4 clonotypes: 41207 | donors: 44 | donors with a dominant clone: 36
```

## Step 2 — OLGA generative null

`A.load_olga_trb()` loads the default human-TRB IGoR model. Expected ≤1-aa neighbors of a clonotype =
`N_unique × Q × Σ Pgen(single-mismatch variants)`; observed vs expected is a Poisson survival test,
BH-corrected per repertoire. **Requires OLGA** — run `%pip install olga` once in this kernel.

```python  # cell 6
# %pip install olga    # uncomment + run once if olga is not importable
pgen_model = A.load_olga_trb()
pgen = A.make_pgen(pgen_model)

# sanity: the Rindler IVB single-aa-variant pair (a textbook ALICE edge) must form a graph edge
_a, _b = "CASSQDRALENTIYF", "CASSQDRTLENTIYF"
assert _b in set(A.one_mismatch_variants(_a)) and A.neighbor_graph([_a, _b]).has_edge(_a, _b)
print(f"OLGA ready | pgen({_a}) = {pgen(_a):.2e}")
```

```
OLGA ready | pgen(CASSQDRALENTIYF) = 7.86e-11
```

## Step 3 — ALICE on the CD4 repertoire → the per-cell clonality call · HEAVY (OLGA sweep)

One ALICE run, per donor, on the pooled CD4 repertoire. For each donor:

1. **seed** = the dominant CD4 clone(s); if the donor has none, its largest clonotype
   (`founder_source` records which — the second case is what lets ALICE *recover* a tumour whose
   cells are split across β-variants and so never looked dominant as an exact clone).
2. **family** = the ≤1-aa connected component of the seed, kept to clonotypes ALICE called
   significant against the OLGA null, plus the seed itself.
3. **donor call** = family ≥ `FRAC_THRESH` of the donor's TCR+ CD4 **and** ≥ `RATIO_THRESH` × its
   largest non-family clone.

`obs["tcr_clonal"]` is then the headline per-cell column: this CD4 cell carries the founder family
of a donor that passed. `tcr_is_malignant` / `tcr_is_dominant_clone` / `tcr_malignant_alice` are set
from it, which is the contract Steps 9–15 read.

*(Removed in this rewrite: the cross-patient convergence test — 0 significant hits out of 30 pooled
founders, an explicit negative control — and the reactive-CD8 ALICE pass, which answers a different
question about the TIL infiltrate. Both are in git history before this commit if they are needed.)*

```python  # cell 8
alice_cd4 = pd.read_parquet(ALICE_CD4_PARQUET) if ALICE_CD4_PARQUET.exists() else None
if alice_cd4 is None or set(alice_cd4["donor"]) != set(clono_cd4["donor"]):
    if alice_cd4 is not None:
        print("ALICE cache stale (donor set changed) -> recompute")
    alice_cd4 = A.run_alice_by_group(clono_cd4, pgen, group="donor", Q=Q, alpha=ALPHA)
    alice_cd4.to_parquet(ALICE_CD4_PARQUET, index=False)
    print("wrote", ALICE_CD4_PARQUET)
else:
    print("loaded cached", ALICE_CD4_PARQUET)
print("ALICE donors:", alice_cd4["donor"].nunique(),
      "| significant clonotypes:", int(alice_cd4["significant"].sum()))

# PT35's tumour is split across two co-dominant clones, neither of which wins the dominance test on
# its own; seeding from both is a documented, donor-specific exception carried from v1.
PT35_EXCEPTION = "Li2024_atlas__PT35"

# Herrera's SS donors are annotated MF upstream; relabel before anything strata by disease, and
# before the donor call below reads `disease` to force HC benign.
donor_of = adata.obs["donor"].astype(str)
_ss = donor_of.isin([f"H__SS{i}" for i in range(1, 7)]).to_numpy()
if _ss.any():
    adata.obs["disease"] = adata.obs["disease"].astype(str)
    adata.obs.loc[_ss, "disease"] = "SS"
    adata.obs["disease"] = adata.obs["disease"].astype("category")
    print("relabeled Herrera SS donors -> disease='SS':", int(_ss.sum()), "cells")
HC_DONORS = set(donor_of[adata.obs["disease"].astype(str).eq("HC").to_numpy()])

fam_by_donor, rows = {}, []
for donor, sub in clono_cd4.groupby("donor", observed=True):
    sig = set(alice_cd4.loc[(alice_cd4["donor"] == donor) & alice_cd4["significant"], "cdr3"])
    if donor == PT35_EXCEPTION:
        seeds, src = set(sub.nlargest(2, "n_cells")["cdr3"]), "pt35_top2"
    elif sub["is_founder"].any():
        seeds, src = set(sub.loc[sub["is_founder"], "cdr3"]), "dominant"
    else:
        seeds, src = {sub.nlargest(1, "n_cells")["cdr3"].iloc[0]}, "largest_clonotype"
    family = seeds | (A.founder_family(sub, seeds=seeds) & sig)
    fam_by_donor[donor] = family

    in_fam = sub["cdr3"].isin(family)
    n_fam, n_tot = int(sub.loc[in_fam, "n_cells"].sum()), int(sub["n_cells"].sum())
    rest = sub.loc[~in_fam, "n_cells"]
    n_next = int(rest.max()) if len(rest) else 0
    frac = n_fam / n_tot if n_tot else 0.0
    fold = (n_fam / n_next) if n_next else np.inf
    rows.append({"donor": donor, "founder_cdr3": "; ".join(sorted(seeds)), "founder_source": src,
                 "n_family_clones": len(family), "n_family_significant": len(family & sig),
                 "donor_family_frac": frac, "donor_family_fold": fold,
                 # HC is benign by definition — a healthy donor's largest clone is not a tumour
                 # clone however dominant it looks, and this is the one place to say so, so that
                 # obs["tcr_clonal"] and the per-sample table cannot disagree about it.
                 "donor_is_clonal": (donor not in HC_DONORS)
                                    and (frac >= FRAC_THRESH) and (fold >= RATIO_THRESH)})
DONOR_FAM = pd.DataFrame(rows).set_index("donor")

# ---- map the family back to cells ----
# TRB comes from the unified clone key (Li2024 cells carry theirs only there), as in clonotype_table.
is_cd4 = adata.obs["cell_type_T2"].isin(CD4_T_TYPES).to_numpy()
trb = (adata.obs["tcr_clone_id"].astype(str)
       .str.extract(r"^TRB:([A-Z]+)$", expand=False).fillna(""))
trb = trb.mask(trb == "", adata.obs["trb_cdr3"].astype(str))

clonal = np.zeros(adata.n_obs, dtype=bool)
for dn, family in fam_by_donor.items():
    if not DONOR_FAM.loc[dn, "donor_is_clonal"]:
        continue                      # a polyclonal donor's largest clone is not a tumour clone
    clonal |= is_cd4 & (donor_of == dn).to_numpy() & trb.isin(family).to_numpy()

# the headline column, plus the three aliases Steps 9-15 read
adata.obs["tcr_clonal"] = clonal
adata.obs["tcr_malignant_alice"]   = clonal
adata.obs["tcr_is_malignant"]      = clonal
adata.obs["tcr_is_dominant_clone"] = clonal
pd.DataFrame({"cell_id": adata.obs_names, "tcr_clonal": clonal}).to_parquet(ALICE_MAL_PARQUET,
                                                                            index=False)
print(f"\nclonal donors: {int(DONOR_FAM['donor_is_clonal'].sum())}/{len(DONOR_FAM)}"
      f" | founder source: {DONOR_FAM['founder_source'].value_counts().to_dict()}")
print(f"clonal CD4 cells: {int(clonal.sum()):,} / {int(is_cd4.sum()):,}")
print("wrote", ALICE_MAL_PARQUET)
```

```
loaded cached /home/projects/nyosef/zvise/MF/data/atlas_joint/alice_cd4_per_donor_v4.parquet
ALICE donors: 44 | significant clonotypes: 66
relabeled Herrera SS donors -> disease='SS': 288 cells


clonal donors: 38/44 | founder source: {'dominant': 36, 'largest_clonotype': 7, 'pt35_top2': 1}
clonal CD4 cells: 81,963 / 401,851
wrote /home/projects/nyosef/zvise/MF/data/atlas_joint/alice_malignancy_v4.parquet
```

## Step 4 — the conclusive per-sample clonality table

One row per sample: **is it clonal, how many CD4 cells, what fraction of them are clonal**, plus the
descriptive fields needed to stratify (study, disease, stage, site, lesion type, origin, treatment).

`family_frac` and `family_fold` are recomputed **within each sample** — against that sample's own
TCR+ CD4 cells and its own largest non-family clone — so `is_clonal` is a statement about the
sample, not the donor's call copied down every row. A sample is clonal when its donor is clonal
**and** the family clears both thresholds locally; a biopsy where the tumour clone is absent
therefore reads False even in a clonal patient.

Samples whose donor has no repertoire at all (`tcr_cohort=False`) get `is_clonal = <NA>`, never
False — no TCR data is absence of evidence, not evidence of absence.

```python  # cell 10
cd4 = pd.DataFrame({
    "sample_id": adata.obs["sample_id"].astype(str).to_numpy(),
    "donor": donor_of.to_numpy(),
    "cdr3": trb.to_numpy(),
    "has_tcr": adata.obs["has_tcr"].astype(bool).to_numpy(),
    "clonal": clonal,
})[is_cd4]
cd4["tcr_ok"] = cd4["has_tcr"] & cd4["cdr3"].map(A._valid)     # same validity rule as clonotype_table

rows = []
for sid, sub in cd4.groupby("sample_id", observed=True):
    dn = sub["donor"].iat[0]
    fam = fam_by_donor.get(dn, set())
    vc = sub[sub["tcr_ok"]].groupby("cdr3", observed=True).size()
    in_fam = vc.index.isin(fam)
    n_fam, n_rep, n_clonal = int(vc[in_fam].sum()), int(vc.sum()), int(sub["clonal"].sum())
    n_next = int(vc[~in_fam].max()) if (~in_fam).any() else 0
    rows.append({
        "sample_id": sid, "donor": dn,
        "n_cd4": len(sub), "n_cd4_tcr": n_rep, "n_clonal": n_clonal,
        "frac_clonal": (n_clonal / n_rep) if n_rep else np.nan,
        "frac_clonal_of_cd4": n_clonal / len(sub),
        "family_frac": (n_fam / n_rep) if n_rep else np.nan,
        "family_fold": (n_fam / n_next) if n_next else (np.inf if n_fam else np.nan)})
smp = pd.DataFrame(rows)

meta_by_sample = (adata.obs[["sample_id", *meta_cols]].astype(str)
                  .groupby("sample_id", observed=True)
                  .agg(lambda s: ", ".join(sorted(s[s != "nan"].unique())))
                  .reset_index())
n_smp_donor = adata.obs.groupby("donor", observed=True)["sample_id"].nunique()

clon = (smp.merge(DONOR_FAM.reset_index(), on="donor", how="left")
        .merge(meta_by_sample, on="sample_id", how="left")
        .assign(n_samples_in_donor=lambda d: d["donor"].map(n_smp_donor).astype(int),
                tcr_cohort=lambda d: d["donor"].isin(tcr_donors)))

# clonal = the donor is clonal AND the family clears both thresholds in THIS sample. A sample with
# no repertoire of its own is <NA>, never False.
clon["is_clonal"] = ((clon["donor_is_clonal"].eq(True)
                      & clon["family_frac"].ge(FRAC_THRESH)
                      & clon["family_fold"].ge(RATIO_THRESH))
                     .astype("boolean")
                     .mask(~clon["tcr_cohort"] | clon["n_cd4_tcr"].eq(0)))

COLS = (["sample_id", "donor", "n_samples_in_donor", "tcr_cohort",
         "is_clonal", "n_cd4", "n_cd4_tcr", "n_clonal", "frac_clonal", "frac_clonal_of_cd4",
         "family_frac", "family_fold", "founder_cdr3", "founder_source",
         "n_family_clones", "n_family_significant", "donor_is_clonal", "donor_family_frac"]
        + meta_cols)
clon = clon[[c for c in COLS if c in clon.columns]].sort_values(
    ["is_clonal", "frac_clonal"], ascending=False, na_position="last").reset_index(drop=True)
clon.to_csv(CLONALITY_CSV, index=False)

_cl = clon["is_clonal"]
print("wrote", CLONALITY_CSV, clon.shape)
print(f"samples: {len(clon)} | clonal {int(_cl.eq(True).fillna(False).sum())}"
      f" | not clonal {int(_cl.eq(False).fillna(False).sum())}"
      f" | no TCR data {int(_cl.isna().sum())}")
print(f"CD4 cells: {int(clon['n_cd4'].sum()):,} | TCR+ {int(clon['n_cd4_tcr'].sum()):,}"
      f" | clonal {int(clon['n_clonal'].sum()):,}")
print("\nclonal rate by disease x stage:")
print(clon.assign(_c=_cl.astype("float"))
      .pivot_table(index="disease", columns="stage_class", values="_c", aggfunc="mean")
      .round(2).to_string())
clon
```

```
wrote /home/projects/nyosef/zvise/MF/tables/skin_tcr_clonality_per_sample.csv (143, 33)
samples: 143 | clonal 46 | not clonal 15 | no TCR data 82
CD4 cells: 401,851 | TCR+ 146,072 | clonal 81,963

clonal rate by disease x stage:
stage_class   HC  advanced  early  unknown
disease                                   
CTCL_other   NaN      0.75    NaN      NaN
HC           0.0       NaN    NaN      NaN
MF           NaN      1.00    1.0      0.9
SS           NaN      0.52    NaN      NaN

               sample_id                donor  n_samples_in_donor  tcr_cohort  \
0     Li2024_atlas__PT35   Li2024_atlas__PT35                   1        True   
1     Li2024_atlas__PT11   Li2024_atlas__PT11                   1        True   
2              D13__MF17            D13__MF17                   1        True   
3     Li2024_atlas__PT50   Li2024_atlas__PT50                   1        True   
4    Li2024_atlas__CTCL1  Li2024_atlas__CTCL1                   1        True   
..                   ...                  ...                 ...         ...   
138   Li2024_atlas__MF22   Li2024_atlas__MF22                   1       False   
139   Li2024_atlas__MF26   Li2024_atlas__MF26                   1       False   
140   Li2024_atlas__MF27   Li2024_atlas__MF27                   1       False   
141   Li2024_atlas__MF28   Li2024_atlas__MF28                   1       False   
142   Li2024_atlas__MF30   Li2024_atlas__MF30                   1       False   

     is_clonal  n_cd4  n_cd4_tcr  n_clonal  frac_clonal  frac_clonal_of_cd4  \
0         True   3810       2535      3343     1.318738            0.877428   
1         True   2230       1726      1832     1.061414            0.821525   
2         True   3752       3266      3226     0.987753            0.859808   
3         True   3322        852       824     0.967136            0.248043   
4         True   8028       6961      6728     0.966528            0.838067   
..         ...    ...        ...       ...          ...                 ...   
138       <NA>    875          0         0          NaN            0.000000   
139       <NA>   1999          0         0          NaN            0.000000   
140       <NA>    832          0         0          NaN            0.000000   
141       <NA>  12299          0         0          NaN            0.000000   
142       <NA>   4265          0         0          NaN            0.000000   

     ...         entity             tissue   tissue_detail lesion_type  \
0    ...     MF_classic               Skin            skin     unknown   
1    ...  MF_unresolved               Skin            skin     unknown   
2    ...             MF               Skin      Skin tumor     unknown   
3    ...     MF_classic               Skin            skin     unknown   
4    ...     MF_classic  Dermis, Epidermis  skin_epidermis     unknown   
..   ...            ...                ...             ...         ...   
138  ...     MF_classic               Skin            skin     unknown   
139  ...     MF_classic               Skin            skin     unknown   
140  ...     MF_classic               Skin            skin     unknown   
141  ...     MF_classic               Skin            skin     unknown   
142  ...     MF_classic               Skin            skin     unknown   

     cohort_country   cohort_region  \
0               USA  North_American   
1               USA  North_American   
2               USA  North American   
3               USA  North_American   
4                UK        European   
..              ...             ...   
138           China      East_Asian   
139           China      East_Asian   
140           China      East_Asian   
141           China      East_Asian   
142           China      East_Asian   

                               treatment_context  blood_involvement sex  \
0                               none/unspecified            unknown   M   
1                               none/unspecified            unknown   M   
2    Be
... [861 more chars truncated]
```

## Step 4b — what "clonal" looks like: the TRB distribution of one sample

Illustration, not an input to any call: one clonal sample's CD4 repertoire, top clonotypes vs the rest.

```python  # cell 12
# Top TRB clonotypes of one clonal sample (illustration only — reads in-memory objects).
SHOWCASE_SAMPLE = None     # set a sample_id to override the automatic pick
TOP_N = 10

_cl = clon[clon["is_clonal"].eq(True) & clon["n_cd4_tcr"].ge(200)]
SID = SHOWCASE_SAMPLE or _cl.sort_values("family_frac", ascending=False)["sample_id"].iat[0]
row = clon[clon["sample_id"].eq(SID)].iloc[0]
seeds = {s for s in str(DONOR_FAM.loc[row["donor"], "founder_cdr3"]).split("; ") if s}

vc = (cd4[cd4["sample_id"].eq(SID) & cd4["tcr_ok"]]
      .groupby("cdr3", observed=True).size().sort_values(ascending=False))
pct = vc / vc.sum() * 100
top = pct.head(TOP_N)
bars = pd.concat([top, pd.Series({f"all other clonotypes (n={len(pct) - TOP_N:,})":
                                  pct.iloc[TOP_N:].sum()})])[::-1]
colors = ["#C1443C" if c in seeds else "#B9BDC2" if c.startswith("all other") else "#3B6FB6"
          for c in bars.index]

fig, ax = plt.subplots(figsize=(6, 3.6), dpi=130)
ax.barh(np.arange(len(bars)), bars.to_numpy(), height=0.7, color=colors)
for y, v in enumerate(bars.to_numpy()):
    ax.text(v + 1, y, f"{v:.1f}%", va="center", fontsize=8, color="#33373B")
ax.set(yticks=np.arange(len(bars)), xlabel="% of sample CD4 TRB repertoire",
       xlim=(0, bars.max() * 1.18))
ax.set_yticklabels(bars.index, fontsize=7.5, family="monospace")
ax.set_title(f"{SID}: one CD4 clone is {top.iloc[0]:.0f}% of the repertoire\n"
             f"({int(vc.sum()):,} TCR+ cells, {len(pct):,} clonotypes)", fontsize=10, loc="left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / f"skin_tcr_clone_distribution_{SID}.png", dpi=200, bbox_inches="tight",
            facecolor="white")
plt.show()
```

```
<Figure size 780x468 with 1 Axes>
```

## Step 5 — inferCNV inputs: query + pooled diploid reference · HEAVY (GPU kernel)

The CNV arm is the **null-anchored, de novo signed arm-consensus caller** — developed in `32_malignancy_tcr_cnv` on
blood (v3.1) and validated on skin as v4 (held-out healthy control 3.1 % false positive, median
per-donor AUROC 0.807 vs the TCR call). It is the method, not a choice among methods, so nothing
below is tunable; the earlier knob panel, the focal-cluster / union / legacy-GMM channels and the
method-selection QC tables are gone (recoverable from git history).

`C.prepare_skin_cnv_inputs` builds the query (annotated T of disease donors) and a **pooled**
diploid reference of three categories whose biases partly cancel:

| `cnv_ref` | cells | bias | role |
|---|---|---|---|
| `nonclonal` | within-donor non-dominant T | right batch + lineage, **tumour-contaminated** | scored like any query cell, **not** a baseline |
| `cd8_ref` | same-donor non-dominant CD8 (≤2000/donor) | right batch, wrong lineage | baseline; **half held out as the batch-matched null** |
| `hc_atlas` | atlas HC T | right tissue + lineage, few donors | baseline; ⅔ used, ⅓ held out |
| `healthy` | external healthy/AD/pso benign-T | right lineage, many donors, wrong batch | baseline |

`cd8_ref` lives inside `acnv` per donor; `hc_atlas` / `healthy` come back in `SHARED_REF`. The
held-out HC control (`HC_CONTROL`, ⅓ of the atlas HC T) stays in the query and is the run's
acceptance test — a working caller must leave it near zero. Self-normalises from `raw_counts`,
adds genomic positions from the GTF, restricts to chr1-22/X/Y.

```python  # cell 14
import os

# ---- the estimator, fixed. Splatted into BOTH inferCNV passes below: they must hold out exactly
# the same null cells, or the arm null describes different diploid cells than the score null.
# The helper *defaults* are the superseded v2/v3 estimator (ref_mode="bounded",
# dynamic_threshold=1.5, shuffle=False, null_frac=0, window=250), so these overrides have to stay
# explicit — dropping one silently reproduces the failing estimator.
#   window=100          skin's ~15.4k positioned genes make v3's 250 too coarse (~1.6% of genome)
#   ref_mode="mean"     one pooled reference vector; infercnvpy's >=2-category path silently
#                       switches to a bounded dead-band that suppresses real signal
#   dynamic_threshold   None: the 1.5*std(chunk) zeroing depended on chunk composition, which made
#                       scores incomparable across donors
#   shuffle=True        permute before chunking so every chunk has the same query/ref mix
#   null_frac=0.5       half the baseline is held out and scored as query -> the empirical null
#   null_cats           the in-object half of the null is batch-matched same-donor CD8
CNV_EST  = dict(window=100, ref_mode="mean", dynamic_threshold=None, shuffle=True,
                null_frac=0.5, null_cats=("cd8_ref",), seed=SEED)
REF_CATS = ("hc_atlas", "cd8_ref", "healthy")   # nonclonal deliberately absent (tumour-contaminated)

CNV_CACHE  = OUT_DIR / "skin_T_cnv_per_cell_v5_res0.5.parquet"
ARM_CACHE  = OUT_DIR / "skin_T_arm_cnv_v5.parquet"
CNV_N_JOBS = 8                              # score pass: 15.4k genes densify, keep memory bounded
ARM_JOBS   = len(os.sched_getaffinity(0))   # arm pass is chunked at 100; respects the LSF cpuset

acnv, SHARED_REF, CNV_DONORS, HC_CONTROL = C.prepare_skin_cnv_inputs(
    adata, GTF, INTEGRATED_H5, SEED,
    n_healthy_ref=10000, hc_ref_frac=2 / 3, cd8_ref_cap=2000)
print("query donors:", len(CNV_DONORS), "| acnv:", acnv.shape,
      "| query:", int((acnv.obs["cnv_ref"] == "query").sum()),
      "| held-out HC control cells:", len(HC_CONTROL))
```

```
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/legacy_api_wrap/__init__.py:88: UserWarning: Some cells have zero counts
  return fn(*args_all, **kw)

atlas HC T: 3195 -> hc_atlas reference | 1597 -> held-out control (stays in query)

/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/legacy_api_wrap/__init__.py:88: UserWarning: Some cells have zero counts
  return fn(*args_all, **kw)

external healthy reference: 10000 cells

INFO:root:Extracted GTF attributes: ['gene_id', 'gene_version', 'gene_name', 'gene_source', 'gene_biotype', 'transcript_id', 'transcript_version', 'transcript_name', 'transcript_source', 'transcript_biotype', 'tag', 'transcript_support_level', 'exon_number', 'exon_id', 'exon_version', 'protein_id', 'protein_version', 'ccds_id']

WARNING: GTF file misses annotation for 301 genes in adata.

WARNING: Skipped 28 genes because of duplicate identifiers in GTF file.

cnv_ref: {'query': np.int64(354143), 'nonclonal': np.int64(92652), 'cd8_ref': np.int64(42663)} | shared ref: 13195 {'healthy': np.int64(10000), 'hc_atlas': np.int64(3195)} | genes on chr1-22/X/Y: 15438 | positioned: 15438
query donors: 114 | held-out HC control cells: 1597
query donors: 114 | acnv: (489458, 15438) | query: 354143 | held-out HC control cells: 1597
```

```python  # cell 15
# ---- 5a. per-donor inferCNV vs the pooled reference -> per-cell scores + CNV Leiden clusters.
# Returns the full per-cell frame INCLUDING the held-out `ref_null` rows; Step 6 needs them.
PER_CELL = C.run_per_donor_infercnv(
    acnv, SHARED_REF, CNV_DONORS, CNV_CACHE, **CNV_EST, ref_cats=REF_CATS,
    topk_frac=0.10, leiden_res=0.5, depth_col="n_genes_cnv",
    n_jobs=CNV_N_JOBS, chunk=1000, force=False)

print("\nper-cell rows:", PER_CELL.shape)
print(PER_CELL["cnv_ref"].value_counts().to_string())
print("\nnull cells per donor (need >= 50 for a usable threshold):")
print(PER_CELL[PER_CELL.cnv_ref.eq("ref_null")].groupby("donor", observed=True)
      .size().describe()[["min", "25%", "50%", "max"]].round(0).to_string())
```

```
loaded cached inferCNV (skin_T_cnv_per_cell_v5_res0.5.parquet) for 114 donors

per-cell rows: (1241971, 7)
cnv_ref
ref_null     773766
query        354143
nonclonal     92652
cd8_ref       21410

null cells per donor (need >= 50 for a usable threshold):
min    6446.0
25%    6620.0
50%    6708.0
max    7604.0
```

```python  # cell 16
# ---- 5b. second pass with `calculate_gene_values=True`, averaged per chromosome arm -> the
# cells x ~41 arm matrix the caller is built on. Same `**CNV_EST`, so `_split_null` holds out
# exactly the same cells as the score pass. Checkpoints per donor in `<cache>_parts/`.
arm = C.compute_arm_cnv_per_cell(
    acnv, CNV_DONORS, ARM_CACHE, shared_ref=SHARED_REF, **CNV_EST, reference_cat=REF_CATS,
    keep_cats=("query", "nonclonal", "ref_null"), n_jobs=ARM_JOBS, chunk=100, force=False)
ARM_COLS = [c for c in arm.columns if c != "donor"]
print("wrote", ARM_CACHE, "| arm matrix:", arm[ARM_COLS].shape, "| arms:", ARM_COLS)

_null_arm = set(arm.index[arm.index.astype(str).str.contains(r"\|NULL\|")])
_null_pc = set(PER_CELL.index[PER_CELL.cnv_ref.eq("ref_null")])
print(f"shared-reference null cells — arm pass {len(_null_arm)}, score pass "
      f"{len(_null_pc & set(arm.index))}: {_null_arm <= _null_pc}")
assert _null_arm <= _null_pc, "the two inferCNV passes held out different reference cells"
```

```
loaded cached arm-CNV (skin_T_arm_cnv_v5.parquet) for 114 samples (1220561, 43)
wrote /home/projects/nyosef/zvise/MF/data/atlas_joint/skin_T_arm_cnv_v5.parquet | arm matrix: (1220561, 41) | arms: ['chr10p', 'chr10q', 'chr11p', 'chr11q', 'chr12p', 'chr12q', 'chr13q', 'chr14q', 'chr15q', 'chr16p', 'chr16q', 'chr17p', 'chr17q', 'chr18p', 'chr18q', 'chr19p', 'chr19q', 'chr1p', 'chr1q', 'chr20p', 'chr20q', 'chr21q', 'chr22q', 'chr2p', 'chr2q', 'chr3p', 'chr3q', 'chr4p', 'chr4q', 'chr5p', 'chr5q', 'chr6p', 'chr6q', 'chr7p', 'chr7q', 'chr8p', 'chr8q', 'chr9p', 'chr9q', 'chrXp', 'chrXq']

shared-reference null cells — arm pass 752513, score pass 773766: True
```

## Step 6 — the malignancy call

`cnv_focal_score` is `mean|X_cnv|` over the top windows, so scattered dropout noise scores like a
clean chr7q gain. A real event is **signed**, contiguous and **shared across the clone**, so
`C.arm_consensus_score` tests each donor's own CNV clusters arm-by-arm against that donor's
held-out diploid null:

1. per arm, the null's mean and sd over the donor's `ref_null` cells;
2. a cluster carries an event on an arm when it is ≥5 SE from the null mean **and** shifted by
   ≥0.01 in absolute terms;
3. a candidate cluster must hold ≥ `max(30, 5 % of the donor's T)` cells — a signature fitted to
   ~40 cells is fitted to their noise;
4. a cluster shifted on >70 % of arms is a global offset, not a karyotype, and is rejected; one
   shifted on >15 arms keeps its 15 strongest (Sézary karyotypes really do carry 13–15);
5. the largest surviving cluster defines the donor's signed signature;
6. every cell scores as its projection onto that unit signature, cut at the depth-conditioned
   98th percentile of the same projection over the null cells.

**No TCR input anywhere**, so the CNV call stays independent evidence. A donor with no cluster
clearing the null gets no signature and no calls — `indeterminate`, not silently negative.

```python  # cell 18
CD4 = acnv.obs["cnv_ref"].isin(["query", "nonclonal"]).to_numpy()   # every scored T cell
PT_DONORS = sorted(acnv.obs.loc[CD4, "donor"].astype(str).unique())

arm_proj, arm_call, arm_state, SIG = C.arm_consensus_score(
    arm, PER_CELL, PT_DONORS, arm_cols=ARM_COLS, cluster_col="cnv_leiden",
    depth_col="n_genes_cnv", q=0.98, min_cluster_frac=0.05,
    max_event_arms=15, global_arm_frac=0.7)

acnv.obs["cnv_arm_proj"] = (pd.Series(arm_proj, index=arm.index)
                            .reindex(acnv.obs_names).to_numpy(dtype=float))
acnv.obs["cnv_arm_malignant"] = (pd.Series(arm_call, index=arm.index)
                                 .reindex(acnv.obs_names).eq(True).to_numpy() & CD4)
acnv.obs["cnv_state"] = (pd.Series(arm_state, index=arm.index).reindex(acnv.obs_names)
                         .fillna("indeterminate").astype(str))

# ---- propagate to `adata`. A donor is callable iff the arm pass found a signature for it. ----
CALLABLE = set(SIG.loc[SIG["callable"], "donor"].astype(str))
for col in ["cnv_score", "cnv_cell_score", "cnv_focal_score", "cnv_arm_proj"]:
    adata.obs[col] = acnv.obs[col].where(CD4).reindex(adata.obs_names)
adata.obs["cnv_leiden"] = acnv.obs["cnv_leiden"].reindex(adata.obs_names).fillna("")
adata.obs["cnv_state"] = (acnv.obs["cnv_state"].reindex(adata.obs_names)
                          .fillna("indeterminate").astype(str))
adata.obs["cnv_arm_malignant"] = (acnv.obs["cnv_arm_malignant"]
                                  .reindex(adata.obs_names).eq(True).to_numpy())
adata.obs["cnv_callable"] = adata.obs["donor"].astype(str).isin(CALLABLE).to_numpy()

_hc = adata.obs_names.isin(HC_CONTROL)
print(f"\ncallable donors: {len(CALLABLE)}/{len(PT_DONORS)}  |  "
      f"malignant {int(adata.obs['cnv_arm_malignant'].sum())}/{adata.n_obs} cells")
print(f"held-out HC control: {int(adata.obs.loc[_hc, 'cnv_arm_malignant'].sum())}/{int(_hc.sum())}"
      f" called malignant ({adata.obs.loc[_hc, 'cnv_arm_malignant'].mean():.1%})"
      f"   [gate <= 5%; v4/v1-cohort was 3.1%]")
SIG
```

```
[B5__PT09] anchor=B5__PT09_2 (n=3330) arms=10 [chr8p-, chr5p-, chr19p+, chr4q-, chr16p+, chr12p-, chr13q+, chr3q-, chr10p+, chr7p+]  called=423/16590 (2.5%)  null FPR=1.0%
[B5__PT22] anchor=B5__PT22_0 (n=2262) arms=6 [chr10p+, chr22q-, chr6q+, chr9q+, chr2q+, chr9p+]  called=753/8442 (8.9%)  null FPR=1.0%

[B5__PT23] anchor=B5__PT23_1 (n=5682) arms=15 [chr10p+, chr8q+, chr17p-, chr17q+, chr5p-, chr6q+, chr11q-, chr8p-, chr22q-, chr20p+, chr18q+, chr4q+, chr7p+, chr2p-, chr9q+]  called=5321/7896 (67.4%)  null FPR=1.0%
[D10__JONAK1] anchor=D10__JONAK1_0 (n=2318) arms=8 [chr5p-, chr21q-, chr20q-, chr20p+, chr11p+, chr6q-, chr10q-, chr18p-]  called=116/2643 (4.4%)  null FPR=1.1%

[D11__P1] anchor=D11__P1_3 (n=688) arms=15 [chr10p+, chr7p+, chr7q+, chr18p+, chr18q+, chr13q-, chr10q-, chr8q+, chr22q+, chr17q+, chr20p-, chr19p-, chr14q-, chr2p-, chr19q-]  called=674/868 (77.6%)  null FPR=1.0%
[D11__P2] anchor=D11__P2_1 (n=4223) arms=15 [chr7p+, chr7q+, chr6q-, chr12p-, chr8p-, chr9p+, chr19p+, chr4q-, chr16p+, chr9q+, chr10q-, chr22q+, chr3q-, chr20q+, chr5q+]  called=8043/11333 (71.0%)  null FPR=1.0%

[D11__P3] anchor=D11__P3_1 (n=7944) arms=15 [chr7p+, chr7q+, chr8q+, chr17q+, chr4p+, chr13q-, chr10p+, chr4q+, chr20p-, chr10q-, chr8p-, chr11p-, chr3p-, chr2p-, chr15q-]  called=7922/9930 (79.8%)  null FPR=1.0%
[D11__P4] anchor=D11__P4_1 (n=503) arms=7 [chr10p+, chr5p-, chr18p+, chr9p+, chr7p+, chr18q+, chr20q-]  called=91/828 (11.0%)  null FPR=1.0%

[D11__P5] anchor=D11__P5_1 (n=5988) arms=15 [chr7p+, chr7q+, chr21q+, chr8p-, chr13q-, chr6q+, chr18q+, chr10p+, chr9q-, chr3q-, chr2p+, chr11p-, chr2q+, chr16q-, chr1q-]  called=7233/8689 (83.2%)  null FPR=1.0%
[D11__P6] anchor=D11__P6_3 (n=2325) arms=15 [chr7p+, chr7q+, chr18q+, chr16p+, chr12p-, chr9q+, chr10q-, chr9p-, chr5q-, chr2p-, chr17q+, chr4q-, chr16q-, chr6p+, chr3p-]  called=5388/6817 (79.0%)  null FPR=1.1%

[D13__MF17] anchor=D13__MF17_2 (n=3473) arms=14 [chr7q+, chr5p-, chr3q-, chr19p+, chr20p+, chr5q-, chr13q-, chr9q-, chr7p+, chr18p-, chr17p+, chr8p-, chr16p+, chr11p-]  called=1112/3881 (28.7%)  null FPR=1.1%
[D13__MF18] anchor=D13__MF18_1 (n=1045) arms=12 [chr5p-, chr18p-, chr6p+, chr19p+, chr17q+, chr10p-, chr16p+, chr17p+, chr3q-, chr13q-, chr5q-, chr20q-]  called=368/1968 (18.7%)  null FPR=1.1%

[D13__MF19] anchor=D13__MF19_4 (n=399) arms=12 [chr17q+, chr10q-, chr5p-, chr10p+, chr17p-, chr3q-, chr6p+, chr3p-, chr18q+, chr9p+, chr19p+, chr19q+]  called=293/499 (58.7%)  null FPR=1.1%
[D13__MF21] anchor=D13__MF21_3 (n=445) arms=15 [chr5p-, chr17q+, chr10p-, chr19q+, chr16p+, chr12p-, chr4p+, chr2p-, chr18q+, chr20p+, chr20q-, chr16q+, chr14q-, chr5q-, chr10q-]  called=67/691 (9.7%)  null FPR=1.0%

[D13__MF25] anchor=D13__MF25_0 (n=275) arms=7 [chr5p-, chr17q+, chr19p+, chr3q-, chr19q+, chr6p+, chr10q-]  called=27/306 (8.8%)  null FPR=1.1%
[D13__SC327_HTO2] anchor=D13__SC327_HTO2_0 (n=70) arms=3 [chr20q-, chr19q+, chr3q-]  called=2/71 (2.8%)  null FPR=1.1%
[D13__SC374_HTO10] anchor=D13__SC374_HTO10_1 (n=294) arms=11 [chr6p+, chr20q-, chr9p+, chr18p-, chr5p-, chr19q+, chr21q-, chr3q-, chr15q-, chr19p+, chr17p+]  called=38/306 (12.4%)  null FPR=1.1%

[D13__SC374_HTO9] anchor=D13__SC374_HTO9_0 (n=76) arms=3 [chr20q-, chr6p+, chr19q+]  called=4/84 (4.8%)  null FPR=1.1%
[D14__SC348] anchor=D14__SC348_3 (n=150) arms=15 [chr10p-, chr4p+, chr14q-, chr12p-, chr4q+, chr18q+, chr20p+, chr2p-, chr7p-, chr12q-, chr19p-, chr9q+, chr11q-, chr19q+, chr16q+]  called=0/150 (0.0%)  null FPR=0.7%
[D14__SC353] anchor=D14__SC353_3 (n=70) arms=15 [chr10p-, chr4p+, chr14q-, chr12p-, chr4q+, chr18q+, chr2p-, chr7p-, chr12q-, chr19p-, chr9q+, chr11q-, chr19q+, chr16q+, chr5q-]  called=0/70 (0.0%)  null FPR=0.7%

[D15__MF24] anchor=D15__MF24_3 (n=1939) arms=10 [chr10p+, chr22q-, chr6p+, chr5p-, chr6q+, chr9p+, chr2q+, chr11q-, chr4q+, chr20q-]  called=437/2870 (15.2%)  null FPR=1.0%
[D15__MF26] anchor=D15__MF26_2 (n=3079) arms=15 [chr10p+, chr7p+, chr18p+, chr8q+, chr7q+, chr18q+, chr8p+, chr4p+, 
... [20505 more chars truncated]
```

## Step 7 — inferCNV heatmaps for a few donors · HEAVY (GPU kernel)

A visual check on the call: three high-burden callable donors, one `indeterminate` donor (should
read flat, or flat-plus-noise), and one held-out HC-control donor (must read flat). Picked from
the result rather than hardcoded, since the v2 re-annotation changed the donor set.

Drawn with the **same estimator** that produced the call (`window=100`, pooled-mean reference),
so the picture is the matrix the caller actually saw. Payloads cached under
`cnv_heatmap_cache_v5`, so re-plots are free.

```python  # cell 20
_d = adata.obs[adata.obs["cnv_focal_score"].notna()].copy()
_d["pt"] = _d["donor"].astype(str)
_g = _d.groupby("pt", observed=True).agg(frac=("cnv_arm_malignant", "mean"),
                                         n=("cnv_arm_malignant", "size"),
                                         ok=("cnv_callable", "first"))
_hc_d = sorted(_d.loc[_d.index.isin(HC_CONTROL), "pt"].unique())
_pool = _g[(_g["n"] >= 500) & ~_g.index.isin(_hc_d)]
HEATMAP_DONORS = (_pool[_pool["ok"]].sort_values("frac", ascending=False).head(3).index.tolist()
                  + _pool[~_pool["ok"]].sort_values("n", ascending=False).head(1).index.tolist()
                  + _hc_d[:1])
print("heatmap donors:", HEATMAP_DONORS)

CNV_HM_CACHE = OUT_DIR / "cnv_heatmap_cache_v5"
sub = acnv[acnv.obs["donor"].astype(str).isin(HEATMAP_DONORS).to_numpy()].copy()
if sub.n_obs:
    C.run_cnv_heatmaps(
        sub, SHARED_REF, call_col="cnv_arm_malignant", reference_cat=REF_CATS,
        ref_mode=CNV_EST["ref_mode"], dynamic_threshold=CNV_EST["dynamic_threshold"],
        window=CNV_EST["window"], fig_prefix="skin_T_v5_cnv_heatmap", fig_dir=FIG_DIR,
        n_per_study=len(HEATMAP_DONORS), vlim_scale=3.0, cmap="RdBu_r",
        order_by="cnv_leiden", n_jobs=CNV_N_JOBS, chunk=1000,
        cnv_cache_dir=CNV_HM_CACHE, force_cnv=False)
del sub
gc.collect()
```

```
heatmap donors: ['D1__P171', 'Li2024_atlas__PT35', 'Li2024_atlas__CTCL12', 'Li2024_atlas__CTCL18', 'D13__SC327_HTO2']

heatmap donors (5/study, by TCR burden): ['D1__P171', 'Li2024_atlas__PT35', 'D13__SC327_HTO2', 'Li2024_atlas__CTCL12', 'Li2024_atlas__CTCL18'] | call_col = cnv_arm_malignant | ORDER_BY = cnv_leiden

  0%|          | 0/20 [00:00<?, ?it/s]
  5%|▌         | 1/20 [00:01<00:31,  1.64s/it]
 30%|███       | 6/20 [00:02<00:05,  2.35it/s]
 40%|████      | 8/20 [00:03<00:03,  3.28it/s]
 60%|██████    | 12/20 [00:03<00:01,  5.12it/s]
 75%|███████▌  | 15/20 [00:03<00:00,  6.22it/s]
 85%|████████▌ | 17/20 [00:03<00:00,  6.18it/s]
100%|██████████| 20/20 [00:04<00:00,  4.96it/s]


[D1__P171] n=19483  ref=['hc_atlas', 'cd8_ref', 'healthy']

[D1__P171] cached heatmap inputs -> skin_T_v5_cnv_heatmap_D1__P171.npz
[chennareddy2025/D1__P171] color vlim=±0.4329

<Figure size 1000x500 with 6 Axes>
  0%|          | 0/19 [00:00<?, ?it/s]
  5%|▌         | 1/19 [00:00<00:15,  1.19it/s]
 16%|█▌        | 3/19 [00:01<00:05,  3.13it/s]
 21%|██        | 4/19 [00:02<00:11,  1.30it/s]
 37%|███▋      | 7/19 [00:03<00:04,  2.71it/s]
 47%|████▋     | 9/19 [00:03<00:02,  3.81it/s]
 74%|███████▎  | 14/19 [00:03<00:00,  6.67it/s]
 84%|████████▍ | 16/19 [00:03<00:00,  6.26it/s]
 89%|████████▉ | 17/19 [00:04<00:00,  6.14it/s]
100%|██████████| 19/19 [00:04<00:00,  4.68it/s]


[Li2024_atlas__PT35] n=18275  ref=['hc_atlas', 'cd8_ref', 'healthy']

[Li2024_atlas__PT35] cached heatmap inputs -> skin_T_v5_cnv_heatmap_Li2024_atlas__PT35.npz
[li2024/Li2024_atlas__PT35] color vlim=±0.4908

<Figure size 1000x500 with 6 Axes>
  0%|          | 0/14 [00:00<?, ?it/s]
  7%|▋         | 1/14 [00:00<00:11,  1.17it/s]
 21%|██▏       | 3/14 [00:00<00:02,  3.85it/s]
 36%|███▌      | 5/14 [00:02<00:04,  2.03it/s]
 57%|█████▋    | 8/14 [00:02<00:01,  3.75it/s]
 79%|███████▊  | 11/14 [00:02<00:00,  5.57it/s]
 93%|█████████▎| 13/14 [00:03<00:00,  5.27it/s]
100%|██████████| 14/14 [00:03<00:00,  4.39it/s]


[D13__SC327_HTO2] n=13266  ref=['hc_atlas', 'healthy']
[D13__SC327_HTO2] cached heatmap inputs -> skin_T_v5_cnv_heatmap_D13__SC327_HTO2.npz

[gaydosik2022/D13__SC327_HTO2] color vlim=±0.3282

<Figure size 1000x500 with 6 Axes>
  0%|          | 0/17 [00:00<?, ?it/s]
  6%|▌         | 1/17 [00:00<00:14,  1.10it/s]
 12%|█▏        | 2/17 [00:01<00:06,  2.30it/s]
 18%|█▊        | 3/17 [00:01<00:03,  3.54it/s]
 24%|██▎       | 4/17 [00:02<00:07,  1.64it/s]
 29%|██▉       | 5/17 [00:03<00:10,  1.14it/s]
 47%|████▋     | 8/17 [00:03<00:03,  2.71it/s]
 94%|█████████▍| 16/17 [00:03<00:00,  8.15it/s]
100%|██████████| 17/17 [00:03<00:00,  4.44it/s]


[Li2024_atlas__CTCL12] n=16686  ref=['hc_atlas', 'cd8_ref', 'healthy']

[Li2024_atlas__CTCL12] cached heatmap inputs -> skin_T_v5_cnv_heatmap_Li2024_atlas__CTCL12.npz
[li2024/Li2024_atlas__CTCL12] color vlim=±0.6292

<Figure size 1000x500 with 6 Axes>
  0%|          | 0/21 [00:00<?, ?it/s]
  5%|▍         | 1/21 [00:01<00:27,  1.39s/it]
 24%|██▍       | 5/21 [00:02<00:06,  2.34it/s]
 29%|██▊       | 6/21 [00:03<00:07,  1.94it/s]
 52%|█████▏    | 11/21 [00:03<00:02,  4.49it/s]
 62%|██████▏   | 13/21 [00:03<00:01,  5.19it/s]
 71%|███████▏  | 15/21 [00:03<00:01,  5.91it/s]
 76%|███████▌  | 16/21 [00:04<00:00,  6.24it/s]
 81%|████████  | 17/21 [00:04<00:00,  4.23it/s]
100%|██████████| 21/21 [00:04<00:00,  4.56it/s]


[Li2024_atlas__CTCL18] n=20740  ref=['hc_atlas', 'cd8_ref', 'healthy']

[Li2024_atlas__CTCL18] cached heatmap inputs -> skin_T_v5_cnv_heatmap_Li2024_atlas__CTCL18.npz
[li2024/Li2024_atlas__CTCL18] color vlim=±0.6677

<Figure size 1000x500 with 6 Axes>
0
```

## Step 8 — combine TCR + CNV & persist

`combined_malignant = TCR ∨ (CNV ∧ donor is callable)`. In an `indeterminate` donor the CNV caller
found no clone-level event clearing the diploid null, so a negative there is absence of evidence,
not evidence of absence — those cells fall back to the TCR label and are tagged
`*_cnv_indeterminate`. Symmetrically, in the seven v2 cohorts with no V(D)J a TCR negative means
*no repertoire*, so those cells get their own `*_no_tcr_data` levels and no downstream reader can
mistake missing TCR for a negative TCR.

Persisted per-cell to `skin_T_malignancy_v5.parquet`.

```python  # cell 22
tcr_m  = adata.obs["tcr_is_malignant"].to_numpy()
cnv_m  = adata.obs["cnv_arm_malignant"].to_numpy().astype(bool)
cnv_ok = adata.obs["cnv_callable"].to_numpy()
has_tcr_arm = adata.obs["tcr_cohort"].to_numpy()

adata.obs["combined_malignant"] = tcr_m | (cnv_m & cnv_ok)
adata.obs["malignant_evidence"] = np.select(
    [tcr_m & cnv_m & cnv_ok, tcr_m & ~cnv_ok, tcr_m,
     ~tcr_m & cnv_m & cnv_ok & has_tcr_arm, ~tcr_m & cnv_m & cnv_ok & ~has_tcr_arm,
     ~cnv_ok & ~has_tcr_arm, ~cnv_ok, ~has_tcr_arm],
    ["both", "tcr_only_cnv_indeterminate", "tcr_only",
     "cnv_only", "cnv_only_no_tcr_data",
     "none_no_tcr_data_cnv_indeterminate", "none_cnv_indeterminate", "cnv_negative_no_tcr_data"],
    default="none")
print("combined malignant:", int(adata.obs["combined_malignant"].sum()), "/", adata.n_obs)
print("\nevidence:\n", adata.obs["malignant_evidence"].value_counts())

# TPR/FPR are only meaningful where both channels exist: TCR-cohort donors with a callable CNV run.
both_avail = (adata.obs["has_tcr"].to_numpy() & has_tcr_arm & cnv_ok
              & adata.obs["cnv_focal_score"].notna().to_numpy())
ct = pd.crosstab(adata.obs.loc[both_avail, "tcr_is_malignant"],
                 adata.obs.loc[both_avail, "cnv_arm_malignant"], rownames=["tcr"], colnames=["cnv"])
if ct.shape == (2, 2):
    print(f"\nvs the TCR call on {int(both_avail.sum())} cells with both signals:  "
          f"TPR={ct.loc[True, True] / ct.loc[True].sum():.2f}  "
          f"FPR={ct.loc[False, True] / ct.loc[False].sum():.2f}   [v4/v1-cohort was 0.64 / 0.07]")

keep = ["donor", "study", "disease", "cell_type", "cell_type_T2", "cached_malignant",
        "tcr_cohort", "has_tcr", "tcr_clone_id", "tcr_clone_size", "tcr_is_expanded",
        "tcr_is_dominant_clone", "tcr_is_malignant", "tcr_clonal", "tcr_malignant_alice",
        "cnv_score", "cnv_cell_score", "cnv_focal_score", "cnv_arm_proj", "cnv_leiden",
        "cnv_arm_malignant", "cnv_state", "cnv_callable",
        "combined_malignant", "malignant_evidence"]
out = adata.obs[[c for c in keep if c in adata.obs.columns]].copy()
out.index.name = "obs_name"
out.to_parquet(OUT_PARQUET)
print("\nwrote", OUT_PARQUET, out.shape)
```

```
combined malignant: 181948 / 492653

evidence:
 malignant_evidence
none                                  175043
cnv_negative_no_tcr_data              127796
cnv_only_no_tcr_data                   75762
both                                   54039
tcr_only                               27924
cnv_only                               24223
none_no_tcr_data_cnv_indeterminate      7866
Name: count, dtype: int64

vs the TCR call on 176283 cells with both signals:  TPR=0.66  FPR=0.11   [v4/v1-cohort was 0.64 / 0.07]


wrote /home/projects/nyosef/zvise/MF/data/atlas_joint/skin_T_malignancy_v5.parquet (492653, 25)
```

## Step 9 — conclusive per-donor CNV table

One row per donor with a CNV run: cohort size, the TCR-clonal fraction where a repertoire exists,
the donor's called state and signed arm signature, and the fraction of its T cells called
malignant. `auroc_vs_tcr` is the projection's discrimination against the TCR call, reported only
for donors with ≥50 TCR-positive **and** ≥50 TCR-negative cells (AUROC at ~100 % prevalence is
meaningless); CNV-only donors have no TCR truth and are left `NaN`.

The three lines under the table are the run's acceptance check.

```python  # cell 24
from sklearn.metrics import roc_auc_score

qdf = adata.obs[adata.obs["cnv_focal_score"].notna()].copy()
qdf["pt"] = qdf["donor"].astype(str)
rows = []
for pt, x in qdf.groupby("pt", observed=True):
    t = x[x["has_tcr"].to_numpy()]
    yt = t["tcr_is_malignant"].to_numpy().astype(bool)
    balanced = yt.sum() >= 50 and (~yt).sum() >= 50
    s = t["cnv_arm_proj"].to_numpy(dtype=float)
    ok = np.isfinite(s)
    rows.append({
        "donor": pt, "study": str(x["study"].iat[0]), "n_T": len(x), "n_tcr": len(t),
        "tcr_malig_frac": round(float(yt.mean()), 3) if len(t) else np.nan,
        "cnv_state": "callable" if bool(x["cnv_callable"].iat[0]) else "indeterminate",
        "frac_cnv_malignant": round(float(x["cnv_arm_malignant"].mean()), 3),
        "auroc_vs_tcr": (round(float(roc_auc_score(yt[ok], s[ok])), 3)
                         if balanced and ok.sum() > 100 and yt[ok].any() and (~yt[ok]).any()
                         else np.nan),
        "is_hc_control": bool(x.index.isin(HC_CONTROL).any())})

# `anchor_frac` is only written for callable donors, so it is absent from SIG entirely
# if nothing was callable — take whatever columns are there.
_sig_cols = [c for c in ["donor", "n_event_arms", "anchor_frac", "signature", "reason"]
             if c in SIG.columns]
summary = (pd.DataFrame(rows).merge(SIG[_sig_cols], on="donor", how="left")
           .sort_values("frac_cnv_malignant", ascending=False))
summary.to_csv(SUMMARY_CSV, index=False)
print("wrote", SUMMARY_CSV, summary.shape)

_hc = adata.obs_names.isin(HC_CONTROL)
_bal = summary["auroc_vs_tcr"].notna()
print(f"\n1. held-out HC control: {int(adata.obs.loc[_hc, 'cnv_arm_malignant'].sum())}"
      f"/{int(_hc.sum())} called malignant "
      f"({adata.obs.loc[_hc, 'cnv_arm_malignant'].mean():.1%})   [gate <= 5%; v4 was 3.1%]")
print(f"2. median per-donor AUROC vs TCR {summary.loc[_bal, 'auroc_vs_tcr'].median():.3f} "
      f"(n={int(_bal.sum())} balanced)  [v4 was 0.807]  |  callable "
      f"{int(summary.cnv_state.eq('callable').sum())}/{len(summary)}  |  donors called 0%/100%: "
      f"{int(summary.frac_cnv_malignant.eq(0).sum())}/{int(summary.frac_cnv_malignant.eq(1).sum())}")
print("3. recurrent signed arm events (expect chr17p-, chr10q-, chr8q+, chr7p+; a chr19q- at the "
      "top is a suspected gene-density artifact):")
print(SIG.loc[SIG["callable"], "signature"].str.split(", ").explode().value_counts()
      .head(12).to_string())
summary
```

```
wrote /home/projects/nyosef/zvise/MF/tables/skin_cnv_v5_summary.csv (114, 13)

1. held-out HC control: 104/1597 called malignant (6.5%)   [gate <= 5%; v4 was 3.1%]
2. median per-donor AUROC vs TCR 0.719 (n=37 balanced)  [v4 was 0.807]  |  callable 112/114  |  donors called 0%/100%: 4/0
3. recurrent signed arm events (expect chr17p-, chr10q-, chr8q+, chr7p+; a chr19q- at the top is a suspected gene-density artifact):
signature
chr5p-     63
chr7q+     49
chr16p+    48
chr17q+    47
chr10q-    47
chr7p+     45
chr20q-    44
chr6q-     41
chr8p-     39
chr13q-    38
chr3q-     37
chr12p-    36

                    donor            study   n_T  n_tcr  tcr_malig_frac  \
35               D1__P171  chennareddy2025  5951   5395           0.881   
107    Li2024_atlas__PT35           li2024  4394   2935           0.844   
82   Li2024_atlas__CTCL12           li2024  3360      0             NaN   
45                D3__P87      brunner2024   351    132           0.000   
106    Li2024_atlas__PT11           li2024  2300   1778           0.902   
..                    ...              ...   ...    ...             ...   
34               D1__P116  chennareddy2025   164    137           0.000   
19             D14__SC353     gaydosik2023    70      0             NaN   
18             D14__SC348     gaydosik2023   150      0             NaN   
87   Li2024_atlas__CTCL18           li2024  7148      0             NaN   
86   Li2024_atlas__CTCL17           li2024   289      0             NaN   

         cnv_state  frac_cnv_malignant  auroc_vs_tcr  is_hc_control  \
35        callable               0.923         0.649          False   
107       callable               0.922         0.730          False   
82        callable               0.909           NaN          False   
45        callable               0.883           NaN          False   
106       callable               0.882         0.953          False   
..             ...                 ...           ...            ...   
34        callable               0.012           NaN           True   
19        callable               0.000           NaN           True   
18        callable               0.000           NaN           True   
87   indeterminate               0.000           NaN          False   
86   indeterminate               0.000           NaN          False   

     n_event_arms  anchor_frac  \
35             15     0.767098   
107            15     0.927401   
82             15     0.904762   
45             15     0.988604   
106            15     0.846087   
..            ...          ...   
34              1     0.975610   
19             15     1.000000   
18             15     1.000000   
87              0          NaN   
86              0          NaN   

                                             signature                  reason  
35   chr7p+, chr10q-, chr7q+, chr17p-, chr17q+, chr...                          
107  chr7p+, chr13q-, chr10q-, chr5p-, chr7q+, chr1...                          
82   chr21q+, chr13q-, chr6q-, chr17p-, chr5p-, chr...                          
45   chr12p-, chr7q+, chr10p-, chr17p-, chr10q-, ch...                          
106  chr10p-, chr13q-, chr8p-, chr7p+, chr10q-, chr...                          
..                                                 ...                     ...  
34                                             chr20q-                          
19   chr10p-, chr4p+, chr14q-, chr12p-, chr4q+, chr...                          
18   chr10p-, chr4p+, chr14q-, chr12p-, chr4q+, chr...                          
87                                                      no cluster clears null  
86                                                      no cluster clears null  

[114 rows x 13 columns]
```

## Step 10 — concordance with `mal_cluster` (26b), the project headline

Read-only. Nothing here re-runs ALICE or inferCNV, and nothing here is written back into
`skin_T_malignancy_v5.parquet` — 26b consumes this notebook's outputs, so a rerun of Steps 1–8
forces a rerun of both cluster jobs and 26b.

`mal_cluster` is the headline call since 2026-10-22 (`helpers/malignancy.py`). This step asks
how the two independent lines of evidence here — ALICE clonality and inferCNV — agree with it,
per donor and overall. Disagreement is informative in both directions: `mal_cluster` is defined
on cells with no recovered TCR and on donors inferCNV could not call, so it covers cells neither
arm here can see; conversely a cluster call that contradicts a confident founder family in a
TCR-rich sample is a review candidate for 26b, not a result.

```python  # cell 26
# ============================================================
# Step 10 — concordance of the TCR / CNV arms with mal_cluster (26b).  READ-ONLY.
# ============================================================
import malignancy as MAL
importlib.reload(MAL)

CONCORDANCE_CSV = TAB_DIR / "skin_malignancy_concordance_v1.csv"

_m = MAL.load_malignancy(adata.obs_names, has_tcr=adata.obs["has_tcr"].to_numpy())
print(MAL.coverage_report(_m).to_string(index=False))

# Pairwise agreement over the cells where both calls of a pair are defined, CD4 only: the other
# T types have no TCR/CNV arm here, so including them would just dilute with undefined rows.
_cd4 = (adata.obs["cell_type_T2"].astype(str) == "CD4").to_numpy()
conc = MAL.concordance(_m[_cd4])
print("\n-- pairwise concordance, CD4 --")
print(conc.to_string(index=False))

# Per donor, so a single deep-sequenced patient cannot carry the summary.
_by_donor = MAL.concordance(_m[_cd4], by=adata.obs["donor"].astype(str).to_numpy()[_cd4])
_by_donor.to_csv(CONCORDANCE_CSV, index=False)
print("\nwrote", CONCORDANCE_CSV, _by_donor.shape)

# Where the headline and the sequence fact disagree while BOTH are defined -> 26b review candidates.
_tcr_def = _m["mal_tcr_defined"].to_numpy() & _cd4
_clu_def = _m["mal_cluster_defined"].to_numpy() & _cd4
_both = _tcr_def & _clu_def
_disagree = _both & (_m["mal_tcr"].to_numpy() != _m["mal_cluster"].to_numpy())
print(f"\nCD4 with both mal_tcr and mal_cluster defined: {int(_both.sum()):,}"
      f" | disagree: {int(_disagree.sum()):,} ({_disagree.sum() / max(int(_both.sum()), 1):.1%})")
if _disagree.any():
    _d = (pd.DataFrame({"sample_id": adata.obs["sample_id"].astype(str).to_numpy()[_disagree],
                        "tcr": _m["mal_tcr"].to_numpy()[_disagree]})
          .groupby("sample_id")["tcr"].agg(["size", "mean"])
          .rename(columns={"size": "n_disagree", "mean": "frac_tcr_malignant"})
          .sort_values("n_disagree", ascending=False))
    print("\ntop samples by disagreement (frac_tcr_malignant = 1 -> ALICE says malignant, "
          "cluster says benign):")
    print(_d.head(10).to_string())
del _m
```

```
/home/projects/nyosef/zvise/MF/helpers/malignancy.py:164: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  n_rescued = int((good.fillna(False).astype(bool) & ~tcr_mask).sum())
/home/projects/nyosef/zvise/MF/helpers/malignancy.py:165: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  tcr_mask = good.where(good.notna(), tcr_mask).fillna(False).astype(bool)

/home/projects/nyosef/zvise/MF/helpers/malignancy.py:222: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["mal_knn"] = call.fillna(False).to_numpy().astype(bool) & defined

             n_defined  frac_defined  n_malignant  frac_malignant_where_defined
mal_tcr       193830.0        0.3934      80328.0                        0.4144
mal_cnv       484787.0        0.9840     154024.0                        0.3177
mal_knn       401217.0        0.8144     259612.0                        0.6471
mal_cluster   491797.0        0.9983     235587.0                        0.4790
 n_defined  frac_defined  n_malignant  frac_malignant_where_defined
  193830.0        0.3934      80328.0                        0.4144
  484787.0        0.9840     154024.0                        0.3177
  401217.0        0.8144     259612.0                        0.6471
  491797.0        0.9983     235587.0                        0.4790

-- pairwise concordance, CD4 --
group       a           b  n_both  agree  jaccard  a_pos  b_pos
  all mal_tcr     mal_cnv  152606 0.7807   0.6113 0.5264 0.3826
  all mal_tcr     mal_knn  152606 0.8958   0.8348 0.5264 0.6305
  all mal_tcr mal_cluster  152606 0.9606   0.9299 0.5264 0.5573
  all mal_cnv     mal_knn  393801 0.6642   0.4981 0.3609 0.6414
  all mal_cnv mal_cluster  393704 0.7520   0.5695 0.3610 0.5430
  all mal_knn mal_cluster  401120 0.8573   0.7871 0.6471 0.5509


wrote /home/projects/nyosef/zvise/MF/tables/skin_malignancy_concordance_v1.csv (684, 8)

CD4 with both mal_tcr and mal_cluster defined: 152,606 | disagree: 6,006 (3.9%)

top samples by disagreement (frac_tcr_malignant = 1 -> ALICE says malignant, cluster says benign):
                                       n_disagree  frac_tcr_malignant
sample_id                                                            
D1__P311_thick                                736            0.175272
Li2024_atlas__CTCL8                           588            0.020408
D1__P204                                      508            0.159449
Li2024_atlas__CTCL3                           373            0.061662
D15__MF30                                     258            0.116279
Li2024_atlas__CTCL5                           241            0.029046
D15__MF24                                     206            0.000000
B5__PT23_Skin_Erythroderma_Dermis_B.1         196            0.035714
Li2024_atlas__CTCL6                           172            0.000000
Li2024_atlas__CTCL7                           162            0.166667
```
