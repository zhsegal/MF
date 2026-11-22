# old/21_tcr_alice_neighborhood.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/old/21_tcr_alice_neighborhood.ipynb` and is never run._

# 21 — ALICE-style TCR-neighborhood enrichment (CD4 malignant subclones · cross-patient · reactive CD8)

ALICE (Pogorelyy et al., PNAS 2019): clonotypes are vertices, edges join CDR3s differing by
**≤1 amino acid**; flag clonotypes with **more neighbors than a generative VDJ-recombination
null predicts** (OLGA `humanTRB` Pgen × thymic-selection factor *Q*; Poisson test, BH-FDR).
Control-free single-snapshot variant — **no healthy background, no TCRNET, TRB only**, Pgen
marginalized over V/J (the atlas clone table has no V gene).

Three uses (see `CLONE_CNV_REVIEW.md` §4/§6):
1. **Subclone-family recovery** (CD4) — the dominant clone's ≤1-aa neighborhood = the malignant
   β-variant family the exact-match caller misses (Srinivas single-aa variants; branched evolution).
   *Caveat:* enrichment here is **somatic divergence, not antigen** — a sequence-family detector;
   disambiguate vs antigen with CNV context.
2. **Cross-patient convergence** (CD4 founders pooled) — shared antigen/superantigen? Expected
   clean negative (one clone/donor is sparse — underpowered).
3. **Reactive clusters** in the **CD8 infiltrate** of the same donors — antigen-driven TIL responses.

Load chain (cells 1–4) mirrors **nb18** and is CPU-light. The OLGA Pgen sweep (Use 1 / Use 3) is the
only heavy step — cached to parquet; run on the kernel or via `jobs/run_alice_pgen.sh`.

```python  # cell 1
# ============================================================
# Parameters — nb18 load chain + cohort filter + ALICE knobs.
# ============================================================
import hashlib, json
from pathlib import Path


def _resolve_nb_dir() -> Path:
    start = Path.cwd()
    for base in [start, *start.parents]:
        for sub in [Path("."), Path("notebooks/MF"), Path("scvi-tools-neural-nmf/notebooks/MF")]:
            cand = base / sub
            if cand.name == "MF" and (cand / "data").exists():
                return cand.resolve()
    raise FileNotFoundError(f"could not locate MF/data from {start}")


NB_DIR = _resolve_nb_dir(); print("NB_DIR =", NB_DIR)

# ---- inputs (same as nb18) ----
OBJ     = NB_DIR / "data" / "atlas_joint" / "skin_T_annotated_v2.h5ad"   # v2 CD4/CD8 re-annotation (cell_type_T2)
TCR_OBJ = NB_DIR / "data" / "atlas_joint" / "skin_T_tcr_annotated.h5ad"
LI_TCR  = NB_DIR / "data" / "Li2024_atlas" / "li2024_tcr_malignancy.parquet"

# ---- dominant-clone rule + cohort filter (identical to nb18) ----
FRAC_THRESH, RATIO_THRESH, EXPANDED_MIN = 0.05, 1.33, 2
# v2 annotation lives in obs["cell_type_T2"] = {CD4, CD4_Treg, CD8, NK}; CD4 part on CD4, CD8 part on CD8.
CD4_T_TYPES = ["CD4"]
CD8_T_TYPES = ["CD8"]
DROP_ENTITIES = {"MF_gamma_delta", "CD8_aggressive_epidermotropic_CTCL"}

# ---- ALICE params ----
ALPHA = 0.05
Q = None              # None -> calibrate per repertoire from the null bulk
SEED = 0

# ---- outputs (v2 = re-annotated cohort; kept separate from the cell_type_T run) ----
OUT_DIR = NB_DIR / "data" / "atlas_joint"
FIG_DIR = NB_DIR / "figures"; FIG_DIR.mkdir(exist_ok=True)
ALICE_CD4_PARQUET = OUT_DIR / "alice_cd4_per_donor_v2.parquet"
ALICE_CD8_PARQUET = OUT_DIR / "alice_cd8_v2.parquet"
USE1_CSV = OUT_DIR / "alice_use1_subclone_families_v2.csv"
USE2_CSV = OUT_DIR / "alice_use2_crosspatient_v2.csv"
USE3_CSV = OUT_DIR / "alice_use3_reactive_cd8_v2.csv"
```

```
NB_DIR = /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF
```

```python  # cell 2
import sys, gc, importlib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(NB_DIR))
import atlas_join_helpers as H
import skin_T_cnv_helpers as C
import alice_helpers as A
for _m in (H, C, A):
    importlib.reload(_m)
np.random.seed(SEED)
```

```python  # cell 3
# Load cached TCR object, recompute dominant clone, apply nb18 sample selection.
adata = C.build_or_load_tcr_object(OBJ, TCR_OBJ, LI_TCR, H)   # ~12 GB cached

# v2 re-annotation: the cached TCR object predates cell_type_T2 -> merge it in by barcode.
import scanpy as sc
_v2 = sc.read_h5ad(OBJ, backed="r")
assert _v2.obs_names.equals(adata.obs_names), "v2 obs order mismatch"
adata.obs["cell_type_T2"] = _v2.obs["cell_type_T2"].astype(str).values
_v2.file.close(); del _v2
print("cell_type_T2:\n", adata.obs["cell_type_T2"].value_counts())

is_li = adata.obs["study"].astype(str).eq("li2024").to_numpy()
dom_tbl = C.recompute_dominant_clone(adata, H, is_li, FRAC_THRESH, RATIO_THRESH, EXPANDED_MIN)
clone_summary = C.clone_summary_table(adata, FRAC_THRESH, RATIO_THRESH)

META_COLS = ["study", "dataset", "disease", "disease_stage", "tissue", "entity", "sex", "tech"]
meta_cols = [c for c in META_COLS if c in adata.obs.columns]
donor_meta = (adata.obs[["donor", *meta_cols]].astype(str)
              .groupby("donor", observed=True)
              .agg(lambda s: ", ".join(sorted(s[s != "nan"].unique()))))
clone_summary = clone_summary.merge(donor_meta, on="donor", how="left")

# nb18 sample-selection: de-dup, drop non-ab-CD4 entities, size gate (>=300 except herrera).
drop = {}
if {"D5__MFIVB", "D1__P303"} <= set(clone_summary["donor"]):
    drop["D1__P303"] = "duplicate of D5__MFIVB"
for d in clone_summary.loc[clone_summary["entity"].isin(DROP_ENTITIES), "donor"]:
    drop.setdefault(d, "non-ab-CD4 entity")
small = (clone_summary["n_tcr_cells"] < 300) & (clone_summary["study"] != "herrera2021")
for d in clone_summary.loc[small, "donor"]:
    drop.setdefault(d, "n_tcr_cells < 300")
keep_donors = clone_summary.loc[~clone_summary["donor"].isin(drop), "donor"].tolist()
clone_summary = clone_summary[~clone_summary["donor"].isin(drop)].reset_index(drop=True)
adata = adata[adata.obs["donor"].isin(keep_donors)].copy()

# HC samples are benign by definition.
hc = adata.obs["disease"].astype(str).eq("HC").to_numpy()
adata.obs.loc[hc, ["tcr_is_malignant", "tcr_is_dominant_clone"]] = False
clone_summary.loc[clone_summary["disease"].eq("HC"), "malignant"] = False
print("kept donors:", len(keep_donors), "| cells:", adata.n_obs)
```

```
loaded cached TCR-complete object -> skin_T_tcr_annotated.h5ad (401566, 49683)

study:
 study
li2024             232365
chennareddy2025    128927
brunner2024         25778
gaydosik2019        11443
rindler2021_fi       2134
herrera2021           919
Name: count, dtype: int64

has_tcr by study:
                    size     sum
study                          
brunner2024       25778   12392
chennareddy2025  128927   69086
gaydosik2019      11443       0
herrera2021         919     659
li2024           232365  114640
rindler2021_fi     2134    1371
cached tumor_cell: 99802 / 401566
cell_type_T2:
 cell_type_T2
CD4         280011
CD8          59297
NK           40665
CD4_Treg     21593
Name: count, dtype: int64
kept donors: 34 | cells: 242959
```

```python  # cell 4
# Per-clonotype tables: CD4 MF cohort (Use 1/2) and CD8 infiltrate of same donors (Use 3).
obs = adata.obs
cd4 = obs[obs["cell_type_T2"].isin(CD4_T_TYPES)]
cd8 = obs[obs["cell_type_T2"].isin(CD8_T_TYPES)]
clono_cd4 = A.clonotype_table(cd4, group="donor")
clono_cd8 = A.clonotype_table(cd8, group="donor")
print("CD4 clonotypes:", len(clono_cd4), "| donors:", clono_cd4["donor"].nunique(),
      "| founders:", int(clono_cd4["is_founder"].sum()))
print("CD8 clonotypes:", len(clono_cd8), "| donors:", clono_cd8["donor"].nunique())
clono_cd4.head()
```

```
CD4 clonotypes: 30550 | donors: 34 | founders: 29
CD8 clonotypes: 11122 | donors: 34

     donor               cdr3  n_cells  is_founder
0  D1__P76  CASSLALSGGAYNEQFF     5815        True
1  D1__P76      CASRGQGVSPLHF        3       False
2  D1__P76      CASSLEGNQPQHF        3       False
3  D1__P76       CAWSGGGAGQFF        2       False
4  D1__P76    CSASYRTTTNEKLFF        2       False
```

## OLGA generative null + sanity check

`A.load_olga_trb()` loads the default human-TRB IGoR model shipped with OLGA. The expected number
of ≤1-aa neighbors of a clonotype is `N_unique × Q × Σ Pgen(single-mismatch variants)`; observed vs
expected is a Poisson survival test, BH-corrected per repertoire. *Q* is calibrated from the
low-degree null bulk (`A.calibrate_Q`); set `Q=...` above to fix it and sweep for sensitivity.

**Requires OLGA** — run once in this kernel: `%pip install olga`

```python  # cell 6
# %pip install olga    # uncomment + run once if olga is not importable
pgen_model = A.load_olga_trb()
pgen = A.make_pgen(pgen_model)

# sanity: the Rindler IVB single-aa-variant pair (a textbook ALICE edge / Srinivas subclone)
a, b = "CASSQDRALENTIYF", "CASSQDRTLENTIYF"
print("pgen(a) =", pgen(a), " pgen(b) =", pgen(b))
print("b is a single-mismatch variant of a:", b in set(A.one_mismatch_variants(a)))
print("graph edge a-b:", A.neighbor_graph([a, b]).has_edge(a, b))
```

```
pgen(a) = 7.856194078998701e-11  pgen(b) = 4.702072793866696e-11
b is a single-mismatch variant of a: True
graph edge a-b: True
```

## Use 1 — per-patient malignant subclone-family recovery

For each CD4 donor: run ALICE on the full clonotype table, then take the ≤1-aa connected component
containing the dominant founder = the **malignant subclone family**. Compare the tumor fraction it
captures against the exact-match founder fraction. **Heavy** (OLGA Pgen sweep) — cached.

```python  # cell 8
# Heavy: OLGA Pgen sweep per CD4 donor (only clonotypes with >=1 observed neighbor). Cached.
# Stale-cache guard: recompute when the cached donor set != the current clono_cd4 donor set.
alice_cd4 = pd.read_parquet(ALICE_CD4_PARQUET) if ALICE_CD4_PARQUET.exists() else None
if alice_cd4 is None or set(alice_cd4["donor"]) != set(clono_cd4["donor"]):
    if alice_cd4 is not None:
        print("CD4 cache stale (donor set changed) -> recompute")
    alice_cd4 = A.run_alice_by_group(clono_cd4, pgen, group="donor", Q=Q, alpha=ALPHA)
    alice_cd4.to_parquet(ALICE_CD4_PARQUET, index=False)
    print("wrote", ALICE_CD4_PARQUET)
else:
    print("loaded cached", ALICE_CD4_PARQUET)
print("CD4 donors:", alice_cd4["donor"].nunique(),
      "| significant clonotypes:", int(alice_cd4["significant"].sum()))
```

```
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/alice_cd4_per_donor_v2.parquet
CD4 donors: 34 | significant clonotypes: 46
```

```python  # cell 9
# Founder subclone-family per donor + the tumor fraction / dominance it recovers.
# Malignant donors: founder = dominant clone. Non-malignant donors: founder = LARGEST clonotype.
# The malignant family = founder + its ALICE-SIGNIFICANT <=1-aa variants (non-significant neighbors
# are coincidental sequence similarity -> dropped). family_frac/fold ask whether collapsing that
# significant family crosses the nb18 threshold (dom_frac >= FRAC_THRESH AND fold >= RATIO_THRESH).
def _dominance(sub, members):
    """Collapse `members` into one clone; return (n, total, dom_frac, fold_change) vs the rest."""
    total = sub["n_cells"].sum()
    n = sub.loc[sub["cdr3"].isin(members), "n_cells"].sum()
    rest = sub.loc[~sub["cdr3"].isin(members), "n_cells"]
    second = int(rest.max()) if len(rest) else 0
    dom_frac = n / total if total else 0.0
    fold = (n / second) if second else np.inf
    return int(n), total, dom_frac, fold


rows = []
for donor, sub in clono_cd4.groupby("donor", observed=True):
    is_mal = bool(sub["is_founder"].any())
    if is_mal:
        founders = set(sub.loc[sub["is_founder"], "cdr3"]); src = "dominant"
    else:  # non-malignant: seed on the largest clonotype
        founders = {sub.sort_values("n_cells", ascending=False)["cdr3"].iloc[0]}; src = "largest_clonotype"
    sig = set(alice_cd4.loc[(alice_cd4["donor"] == donor) & alice_cd4["significant"], "cdr3"])
    fam = A.founder_family(sub, seeds=founders)
    fam_sig = set(founders) | (fam & sig)        # founder + significant <=1-aa variants

    ex_n, total, ex_frac, ex_fold = _dominance(sub, founders)
    fam_n, _, fam_frac, fam_fold = _dominance(sub, fam_sig)
    exact_call = (ex_frac >= FRAC_THRESH) and (ex_fold >= RATIO_THRESH)
    family_call = (fam_frac >= FRAC_THRESH) and (fam_fold >= RATIO_THRESH)
    rows.append({
        "donor": donor, "founder_source": src, "founder": "; ".join(sorted(founders)),
        "n_variant_clones": max(len(fam) - len(founders), 0),
        "n_family_significant": len(fam & sig),
        "exact_frac": ex_frac, "family_frac": fam_frac, "delta_frac": (fam_n - ex_n) / total,
        "exact_fold": ex_fold, "family_fold": fam_fold,
        "exact_call": exact_call, "family_call": family_call,
        "flipped_by_family": (not is_mal) and family_call})

use1 = (pd.DataFrame(rows)
        .merge(clone_summary[["donor", "study", "disease", "disease_stage", "entity", "malignant"]],
               on="donor", how="left")
        .sort_values(["flipped_by_family", "delta_frac"], ascending=False))
use1.to_csv(USE1_CSV, index=False)
print("wrote", USE1_CSV, "| donors:", len(use1),
      "| non-malignant flipped by family:", int(use1["flipped_by_family"].sum()))
use1
```

```
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/alice_use1_subclone_families_v2.csv | donors: 34 | non-malignant flipped by family: 0

                  donor     founder_source               founder  \
13            D5__MFIVB           dominant       CASSQDRALENTIYF   
31   Li2024_atlas__PT50           dominant  CASNLRRTSGGPSSYNEQFF   
10             D3__P196           dominant     CASSEARPGLGTDTQYF   
30   Li2024_atlas__PT47           dominant        CASSQSRTVYGYTF   
32   Li2024_atlas__PT53           dominant      CASSEEDRGPYNEQFF   
28   Li2024_atlas__PT11           dominant      CASSQDSWPYHSPLHF   
27  Li2024_atlas__CTCL8           dominant      CASSLVMGVANTEAFF   
0              D1__P115  largest_clonotype    CASSHLRGLFRKNTEAFF   
1              D1__P116  largest_clonotype         CAIRHTGRYGYTF   
2              D1__P171           dominant      CAISEAPSGLYNEQFF   
3              D1__P204           dominant         CASSIRGNEKLFF   
4              D1__P311           dominant         CASSFGGVSPLHF   
5              D1__P312           dominant       CASSVANRGTYEQYF   
6               D1__P76           dominant     CASSLALSGGAYNEQFF   
7               D1__PGS           dominant     CASSFPRGASGANVLTF   
8              D3__P105           dominant         CASSLTGGETQYF   
9              D3__P192           dominant            CSVEVTEAFF   
11              D3__P87           dominant      CASSYAGTGVGNTIYF   
12              D3__P94  largest_clonotype         CASSTGVFEKLFF   
14               H__HC1  largest_clonotype       CASIRDTANTGELFF   
15             H__MFIV1           dominant       CSARFLRGGYNEQFF   
16               H__SS1           dominant       CASSRQGGSGNTIYF   
17               H__SS2           dominant      CASKLFMDREKGEAFF   
18               H__SS3           dominant         CASSTTGNTEAFF   
19               H__SS4           dominant      CASSRLGGAYNSPLHF   
20  Li2024_atlas__CTCL1           dominant         CSARTGSNQPQHF   
21  Li2024_atlas__CTCL2           dominant    CASRASSNSGRASYEQYF   
22  Li2024_atlas__CTCL3           dominant          CSARTGGYGYTF   
23  Li2024_atlas__CTCL4           dominant        CASSVSGSNTEAFF   
24  Li2024_atlas__CTCL5           dominant       CASSPSTQSNEKLFF   
25  Li2024_atlas__CTCL6           dominant      CASSQAFGPPHGTIYF   
26  Li2024_atlas__CTCL7           dominant        CASSYRGPYNEQFF   
29   Li2024_atlas__PT35  largest_clonotype       CASSRLGGSGNTIYF   
33   Li2024_atlas__PT56           dominant        CASRGGLRGTEAFF   

    n_variant_clones  n_family_significant  exact_frac  family_frac  \
13                 1                     2    0.793778     0.960889   
31                13                    13    0.511527     0.531700   
10                 2                     3    0.768760     0.788668   
30                 4                     2    0.765306     0.772109   
32                12                    10    0.674125     0.679862   
28                 5                     2    0.906550     0.908147   
27                 1                     1    0.759514     0.759659   
0                  0                     0    0.021084     0.021084   
1                  0                     0    0.021858     0.021858   
2                  0                     0    0.975948     0.975948   
3                  1                     0    0.370499     0.370499   
4                  1                     0    0.907256     0.907256   
5                  0                     0    0.515502     0.515502   
6                  0                     0    0.876942     0.876942   
7                  0                     0    0.935835     0.935835   
8                  0                     0    0.176471     0.176471   
9                  0                     0    0.918601     0.918601   
11                 0                     0    0.998259     0.998259   
12                 0                     0    0.037322     0.037322   
14                 0      
... [7374 more chars truncated]
```

## Use 2 — cross-patient convergence (expected negative)

Pool one founder per malignant donor into a single repertoire and test for ≤1-aa neighborhood
enrichment **across patients** (shared antigen / superantigen). Per Gniadecki the expectation is a
clean negative; one clone per donor is sparse, so this is underpowered — a companion to GLIPH2, not a
substitute. A positive here would be the surprising, report-worthy result.

```python  # cell 12
# Cross-patient: pool founders, test neighborhood enrichment across patients.
founders = (clono_cd4[clono_cd4["is_founder"]]
            .drop_duplicates(["donor", "cdr3"])[["donor", "cdr3"]].copy())
pool = founders[["cdr3"]].copy(); pool["is_founder"] = False  # no within-pool founder; test degree only
use2 = A.alice_test(pool, pgen, Q=Q, alpha=ALPHA)
use2 = use2.merge(founders[["donor", "cdr3"]], on="cdr3", how="left")
use2.to_csv(USE2_CSV, index=False)
print("pooled founders:", len(founders),
      "| cross-patient significant:", int(use2["significant"].sum()))
use2.head(10)
```

```
pooled founders: 29 | cross-patient significant: 0

                cdr3  is_founder  obs_deg    Q  exp_deg  pois_p  q_bh  \
0  CASSLALSGGAYNEQFF       False        0  1.0      0.0     1.0   1.0   
1   CAISEAPSGLYNEQFF       False        0  1.0      0.0     1.0   1.0   
2      CASSIRGNEKLFF       False        0  1.0      0.0     1.0   1.0   
3      CASSFGGVSPLHF       False        0  1.0      0.0     1.0   1.0   
4    CASSVANRGTYEQYF       False        0  1.0      0.0     1.0   1.0   
5  CASSFPRGASGANVLTF       False        0  1.0      0.0     1.0   1.0   
6   CASSYAGTGVGNTIYF       False        0  1.0      0.0     1.0   1.0   
7      CASSLTGGETQYF       False        0  1.0      0.0     1.0   1.0   
8         CSVEVTEAFF       False        0  1.0      0.0     1.0   1.0   
9  CASSEARPGLGTDTQYF       False        0  1.0      0.0     1.0   1.0   

   significant     donor  
0        False   D1__P76  
1        False  D1__P171  
2        False  D1__P204  
3        False  D1__P311  
4        False  D1__P312  
5        False   D1__PGS  
6        False   D3__P87  
7        False  D3__P105  
8        False  D3__P192  
9        False  D3__P196
```

## Use 3 — reactive antigen-driven clusters in the CD8 infiltrate

ALICE on the **CD8** T cells of the same donors — per donor and pooled per disease stage — to surface
convergent, antigen-driven reactive TIL clusters (anti-tumor / viral / *S. aureus*). Optional: annotate
significant hits against VDJdb / McPAS specificity (needs the DB file). **Heavy** — cached.

```python  # cell 14
# Reactive CD8 infiltrate: ALICE per donor and pooled per disease stage. Heavy -> cached.
# Stale-cache guard: recompute when the cached donor set != the current clono_cd8 donor set.
stage = obs[["donor", "disease_stage"]].astype(str).drop_duplicates().set_index("donor")["disease_stage"]
clono_cd8 = clono_cd8.assign(disease_stage=clono_cd8["donor"].map(stage).astype(str))
alice_cd8 = pd.read_parquet(ALICE_CD8_PARQUET) if ALICE_CD8_PARQUET.exists() else None
cached_donors = set(alice_cd8.loc[alice_cd8["scope"] == "donor", "donor"]) if alice_cd8 is not None else None
if alice_cd8 is None or cached_donors != set(clono_cd8["donor"]):
    if alice_cd8 is not None:
        print("CD8 cache stale (donor set changed) -> recompute")
    per_donor = A.run_alice_by_group(clono_cd8, pgen, group="donor", Q=Q, alpha=ALPHA)
    per_donor["scope"] = "donor"
    by_stage = A.run_alice_by_group(clono_cd8, pgen, group="disease_stage", Q=Q, alpha=ALPHA)
    by_stage["scope"] = "stage"
    alice_cd8 = pd.concat([per_donor, by_stage], ignore_index=True)
    alice_cd8.to_parquet(ALICE_CD8_PARQUET, index=False)
    print("wrote", ALICE_CD8_PARQUET)
else:
    print("loaded cached", ALICE_CD8_PARQUET)
use3 = alice_cd8[alice_cd8["significant"]].sort_values("obs_deg", ascending=False)
use3.to_csv(USE3_CSV, index=False)
print("CD8 donors:", alice_cd8.loc[alice_cd8['scope'] == 'donor', 'donor'].nunique(),
      "| reactive CD8 significant clonotypes:", len(use3))
use3
```

```
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/alice_cd8_v2.parquet
CD8 donors: 34 | reactive CD8 significant clonotypes: 46

                     donor                cdr3  n_cells  is_founder  \
19470   Li2024_atlas__PT11    CGSSQDSWPYHSPLHF        1       False   
10375   Li2024_atlas__PT11    CGSSQDSWPYHSPLHF        1       False   
10376   Li2024_atlas__PT11    CASSQDSWPYHSPLHF      340        True   
10377   Li2024_atlas__PT11    CTSSQDSWPYHSPLHF        1       False   
19471   Li2024_atlas__PT11    CASSQDSWPYHSPLHF      340        True   
19472   Li2024_atlas__PT11    CTSSQDSWPYHSPLHF        1       False   
10555   Li2024_atlas__PT50   CAIREKTGQGSGNTIYF        1       False   
14944   Li2024_atlas__PT50   CAIREKTGQGSGNTIYF        1       False   
269               D1__P171     CASSVLAGGIYEQYF        1       False   
537               D1__P204       CASLSGIDYEQYY        1       False   
538               D1__P204   CASSPHPTGAGADEQYF        1       False   
539               D1__P204   CASSPRPTGAGADEQYF        1       False   
1627              D1__P311        CASSPYTGELFF        5       False   
1628              D1__P311        CASSSYTGELFF        4       False   
2625               D3__P94      CSARDRGLGNTIYF        1       False   
2624               D3__P94      CSARDRGIGNTIYF        1       False   
2387              D3__P192       CASSQPQGGELFF        1       False   
2386              D3__P192       CASSQVQGGELFF        2       False   
1779              D1__P312  CSARGWLAGRSSTDTQYF        1       False   
1778              D1__P312  CSARGWLAGRSGTDTQYF        2       False   
1630              D1__P311         CASSFMDEQFF        1       False   
1629              D1__P311         CASSFPDEQFF        1       False   
10378   Li2024_atlas__PT11    YGSSQDSWPYHSPLHF        2       False   
10401   Li2024_atlas__PT35     CASSRLGGSGNTIYF        8       False   
10403   Li2024_atlas__PT35        CASSSEQETQYF        1       False   
10402   Li2024_atlas__PT35     CASSRQGGSGNTIYF        7       False   
3417   Li2024_atlas__CTCL2    CSAQQELAGLYNEQFF        1       False   
3416   Li2024_atlas__CTCL2    CSAQGELAGLYNEQFF        1       False   
2787             D5__MFIVB     CASSQDRALENTIYF        2        True   
2786             D5__MFIVB     CASSQDRTLENTIYF        2       False   
10781   Li2024_atlas__PT56     CASSFRQGTIHEQFY        1       False   
10557   Li2024_atlas__PT50   CAIREKTGQGSGNTTYF        1       False   
10556   Li2024_atlas__PT50   CAIREKTGQGSGNTIYC        1       False   
10404   Li2024_atlas__PT35        CASSSERETQYF        1       False   
14946   Li2024_atlas__PT50   CAIREKTGQGSGNTTYF        1       False   
14945   Li2024_atlas__PT50   CAIREKTGQGSGNTIYC        1       False   
11391  Li2024_atlas__CTCL2    CSAQQELAGLYNEQFF        1       False   
18058   Li2024_atlas__PT56     CASSFRQGTIHEQFY        1       False   
19473   Li2024_atlas__PT11    YGSSQDSWPYHSPLHF        2       False   
19496            D5__MFIVB     CASSQDRTLENTIYF        2       False   
19497            D5__MFIVB     CASSQDRALENTIYF        2        True   
20043             D1__P204       CASLSGIDYEQYY        1       False   
20044             D1__P204   CASSPHPTGAGADEQYF        1       False   
20045             D1__P204   CASSPRPTGAGADEQYF        1       False   
20046             D1__P312  CSARGWLAGRSGTDTQYF        2       False   
20047             D1__P312  CSARGWLAGRSSTDTQYF        1       False   

      disease_stage  obs_deg             Q       exp_deg        pois_p  \
19470          IVA2        3      1.000000  1.705613e-11  8.269709e-34   
10375          IVA2        3      1.000000  1.705613e-11  8.269709e-34   
10376          IVA2        2      1.000000  7.968065e-09  3.174503e-17   
10377          IVA2        2      1.000000  1.705156e-11  1.453779e-22   
19471          IVA2        2      1.000000  7.968065e-09  3.174503e-17   
19472          IVA2        2      1.000000  1.705156e
... [4953 more chars truncated]
```

```python  # cell 15
# CD8 coverage: every donor (significant or not), with clonotype count + #significant hits.
cd8_cov = (alice_cd8[alice_cd8["scope"] == "donor"]
           .groupby("donor", observed=True)
           .agg(n_clonotypes=("cdr3", "size"),
                n_with_neighbor=("obs_deg", lambda s: int((s >= 1).sum())),
                n_significant=("significant", "sum"))
           .reset_index()
           .merge(clone_summary[["donor", "study", "disease", "disease_stage", "malignant"]],
                  on="donor", how="left")
           .sort_values("n_significant", ascending=False))
print("CD8 donors:", len(cd8_cov), "| with >=1 reactive hit:", int((cd8_cov["n_significant"] > 0).sum()))
cd8_cov
```

```
CD8 donors: 34 | with >=1 reactive hit: 9

                  donor  n_clonotypes  n_with_neighbor  n_significant  \
32   Li2024_atlas__PT53           207                4              4   
12              D3__P94           462                4              4   
29   Li2024_atlas__PT35           215               11              3   
31   Li2024_atlas__PT50            48                5              3   
30   Li2024_atlas__PT47            42                2              2   
5              D1__P312           137                2              2   
13            D5__MFIVB           133                4              2   
9              D3__P192           274                2              2   
10             D3__P196            57                2              2   
0              D1__P115           151                0              0   
1              D1__P116           108                2              0   
7               D1__PGS           235                2              0   
6               D1__P76           197                0              0   
11              D3__P87             4                0              0   
8              D3__P105            80                0              0   
4              D1__P311           179                6              0   
3              D1__P204           666               21              0   
2              D1__P171           325                4              0   
17               H__SS2            36                0              0   
16               H__SS1             8                0              0   
15             H__MFIV1            18                0              0   
14               H__HC1             6                0              0   
18               H__SS3             3                0              0   
19               H__SS4            15                0              0   
20  Li2024_atlas__CTCL1           492               12              0   
21  Li2024_atlas__CTCL2           438                8              0   
25  Li2024_atlas__CTCL6           950               26              0   
24  Li2024_atlas__CTCL5           669               21              0   
23  Li2024_atlas__CTCL4           991               17              0   
22  Li2024_atlas__CTCL3           774               17              0   
28   Li2024_atlas__PT11            26                0              0   
27  Li2024_atlas__CTCL8           613               14              0   
26  Li2024_atlas__CTCL7          2784              128              0   
33   Li2024_atlas__PT56           424               14              0   

              study     disease disease_stage  malignant  
32           li2024          MF           IIB       True  
12      brunner2024  CTCL_other            NA      False  
29           li2024          MF           IIB      False  
31           li2024          MF            IB       True  
30           li2024          MF           IIB       True  
5   chennareddy2025          MF      advanced       True  
13   rindler2021_fi          MF           IVB       True  
9       brunner2024  CTCL_other            NA       True  
10      brunner2024  CTCL_other            NA       True  
0   chennareddy2025          HC            HC      False  
1   chennareddy2025          HC            HC      False  
7   chennareddy2025          MF      advanced       True  
6   chennareddy2025          MF      advanced       True  
11      brunner2024  CTCL_other            NA       True  
8       brunner2024  CTCL_other            NA       True  
4   chennareddy2025          MF      advanced       True  
3   chennareddy2025          MF      advanced       True  
2   chennareddy2025          MF      advanced       True  
17      herrera2021          MF                     True  
16      herrera2021          MF                     True  
15      herrera2021          MF                     True  
14      herrera2021          HC                    False  
18      herrera2021          MF             
... [661 more chars truncated]
```

## Figures

```python  # cell 17
# Figures (one finding each; figsize 5x3).
plt.rcParams["figure.dpi"] = 120

# 1) Use 1 — tumor fraction recovered by the ALICE family vs exact-match founder.
d = use1.sort_values("family_frac", ascending=False)
fig, ax = plt.subplots(figsize=(5, 3))
ax.scatter(d["exact_frac"], d["family_frac"], s=18)
lim = [0, max(d["family_frac"].max(), d["exact_frac"].max()) * 1.05]
ax.plot(lim, lim, "k--", lw=0.8)
ax.set_xlabel("exact-match founder frac"); ax.set_ylabel("ALICE family frac")
ax.set_title("ALICE recovers β-variant tumor cells")
fig.tight_layout(); fig.savefig(FIG_DIR / "alice_use1_family_vs_exact.png", bbox_inches="tight")

# 2) Use 1 — CD4 observed vs expected degree, founders highlighted.
m = alice_cd4.merge(clono_cd4[["donor", "cdr3", "is_founder"]], on=["donor", "cdr3"], how="left")
fig, ax = plt.subplots(figsize=(5, 3))
ax.scatter(m["exp_deg"] + 1e-9, m["obs_deg"], s=6, alpha=.3, label="all")
f = m[m["is_founder"].fillna(False)]
ax.scatter(f["exp_deg"] + 1e-9, f["obs_deg"], s=20, color="tab:red", label="founder")
ax.set_xscale("log"); ax.set_xlabel("expected degree (null)"); ax.set_ylabel("observed degree")
ax.set_title("CD4 clonotype neighborhood enrichment"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(FIG_DIR / "alice_use1_obs_vs_exp.png", bbox_inches="tight")

# 3) Use 2 — pooled cross-patient degree histogram (expected negative).
fig, ax = plt.subplots(figsize=(5, 3))
ax.hist(use2["obs_deg"], bins=range(0, int(use2["obs_deg"].max()) + 2))
ax.set_xlabel("cross-patient neighbor degree"); ax.set_ylabel("founders")
ax.set_title(f"Cross-patient convergence: {int(use2['significant'].sum())} hits")
fig.tight_layout(); fig.savefig(FIG_DIR / "alice_use2_crosspatient_hist.png", bbox_inches="tight")

# 4) Use 3 — reactive CD8 significant clonotypes by stage.
st = use3[use3.get("scope", "") == "stage"] if "scope" in use3 else use3
if len(st) and "disease_stage" in st.columns:
    c = st.groupby("disease_stage").size().sort_values()
    fig, ax = plt.subplots(figsize=(5, 3))
    c.plot.barh(ax=ax)
    ax.set_xlabel("significant CD8 clonotypes"); ax.set_title("Reactive CD8 clusters by stage")
    fig.tight_layout(); fig.savefig(FIG_DIR / "alice_use3_reactive_by_stage.png", bbox_inches="tight")
plt.show()
```

```
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
File ~/.local/share/mamba/envs/neural_nmf_env/lib/python3.12/site-packages/pandas/core/indexes/base.py:3812, in Index.get_loc(self, key)
   3811 try:
-> 3812     return self._engine.get_loc(casted_key)
   3813 except KeyError as err:

File pandas/_libs/index.pyx:167, in pandas._libs.index.IndexEngine.get_loc()

File pandas/_libs/index.pyx:196, in pandas._libs.index.IndexEngine.get_loc()

File pandas/_libs/hashtable_class_helper.pxi:7088, in pandas._libs.hashtable.PyObjectHashTable.get_item()

File pandas/_libs/hashtable_class_helper.pxi:7096, in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'is_founder'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
Cell In[20], line 18
     16 fig, ax = plt.sub
<Figure size 600x360 with 1 Axes>
<Figure size 600x360 with 1 Axes>
```

```python  # cell 18
# Cache final ALICE malignancy -> adata.obs["tcr_malignant_alice"].
# Per donor: dominant CD4 clone + its ALICE-significant <=1-aa neighbors = malignant.
# Non-dominant donors -> not assigned (stay False), as before.
# Exception: Li2024_atlas__PT35 -> its top-2 largest CD4 clones (+ their significant neighbors).
PT35_EXCEPTION = "Li2024_atlas__PT35"
ALICE_MAL_PARQUET = OUT_DIR / "alice_malignancy_v2.parquet"

mal_sets = {}   # donor -> set of malignant CDR3 (seed clone(s) + significant <=1-aa neighbors)
for donor, sub in clono_cd4.groupby("donor", observed=True):
    sig = set(alice_cd4.loc[(alice_cd4["donor"] == donor) & alice_cd4["significant"], "cdr3"])
    if donor == PT35_EXCEPTION:
        seeds = set(sub.sort_values("n_cells", ascending=False)["cdr3"].iloc[:2])
    elif sub["is_founder"].any():
        seeds = set(sub.loc[sub["is_founder"], "cdr3"])
    else:
        continue   # no dominant clone -> leave benign (no malignancy call)
    fam = A.founder_family(sub, seeds=seeds)
    mal_sets[donor] = seeds | (fam & sig)

# Map malignant CDR3 sets back to CD4 cells (TRB from the unified clone key, as in clonotype_table).
trb = (adata.obs["tcr_clone_id"].astype(str)
       .str.extract(r"^TRB:([A-Z]+)$", expand=False).fillna(""))
trb = trb.mask(trb == "", adata.obs["trb_cdr3"].astype(str))
donor = adata.obs["donor"].astype(str)
is_cd4 = adata.obs["cell_type_T2"].isin(CD4_T_TYPES).to_numpy()

mal = np.zeros(adata.n_obs, dtype=bool)
for d, mset in mal_sets.items():
    mal |= is_cd4 & (donor == d).to_numpy() & trb.isin(mset).to_numpy()
adata.obs["tcr_malignant_alice"] = mal

pd.DataFrame({"cell_id": adata.obs_names, "tcr_malignant_alice": mal}).to_parquet(ALICE_MAL_PARQUET, index=False)
print("malignant donors:", len(mal_sets), "| malignant cells:", int(mal.sum()),
      "| PT35 malignant cells:", int((is_cd4 & (donor == PT35_EXCEPTION).to_numpy() & mal).sum()))
print("wrote", ALICE_MAL_PARQUET)
```

```
malignant donors: 30 | malignant cells: 68620 | PT35 malignant cells: 2474
wrote /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/MF/data/atlas_joint/alice_malignancy_v2.parquet
```

```python  # cell 19
pd.crosstab(adata.obs["tcr_malignant_alice"], adata.obs["cell_type_T2"], margins=True)
```

```
cell_type_T2            CD4  CD4_Treg    CD8     NK     All
tcr_malignant_alice                                        
False                100394     15667  37207  21071  174339
True                  68620         0      0      0   68620
All                  169014     15667  37207  21071  242959
```

```python  # cell 20
# Save the full annotated adata (with tcr_malignant_alice) for reuse elsewhere.
# Herrera SS donors (H__SS1..SS6) are Sezary — correct an upstream disease='MF' mislabel
# (patients.tsv `disease` column) so every downstream consumer of this object reads disease='SS'.
_ss = adata.obs["donor"].astype(str).isin([f"H__SS{i}" for i in range(1, 7)])
if _ss.any():
    adata.obs["disease"] = adata.obs["disease"].astype(str)
    adata.obs.loc[_ss, "disease"] = "SS"
    adata.obs["disease"] = adata.obs["disease"].astype("category")
    print("relabeled Herrera SS donors -> disease='SS':", int(_ss.sum()), "cells")

ADATA_MALIG_H5AD = OUT_DIR / "skin_T_tcr_malig_v2.h5ad"
adata.write_h5ad(ADATA_MALIG_H5AD)
print("wrote", ADATA_MALIG_H5AD, "|", adata.shape)
```
