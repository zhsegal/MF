# 40_targets/skin/42_target_screen_skin.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/40_targets/skin/42_target_screen_skin.ipynb` and is never run._

# 42 — Surface-target screen, **skin**: cohort fix, artifact tests, malignant vs reactive CD4

The skin twin of 43, rebuilt on the same template. The first pass ranked **ENO1, ANXA2, HSP90AA1,
VDAC1, GPI, CALU and QSOX1** at the top of the prevalence axis and put **COL6A3, LGALS3 and GRN**
on the shortlist, while the antigens whose answer is already known came out at **CCR4 rank 386,
KIR3DL2 533 and IL2RA 1,968 of 2,115 with a negative log2FC**. Only TNFRSF8 (CD30) was anywhere
near the top. A screen that cannot recover the drugs that already work in this disease is not yet
measuring selectivity, and its novel hits are not evidence.

Two things are wrong with that pass, and they are not the same thing.

**1. The cohort, which was wrong and is now fixed.** 146 donor keys are not 146 patients. Nineteen
carry a malignant clone that is **not CD4** — CD8, γδ or double-negative by `clin_phenotype`, plus
the CD8 aggressive cytotoxic case and four pcAECTCL — so `mal_cluster` finds no tumour in their CD4
compartment and all of their CD4 cells sit in the **reactive** arm of a tumour-vs-reactive test.
Fifteen more carry a diagnosis outside the MF/SS spectrum (LyP, SPP, LPP). Ten patients were
biopsied twice or more and appear under up to four keys each. §0b removes them: **109 independent
donor keys, 107 people**, diagnosis from `clin_dx`.

**2. The ranking's shape, which is what §2 is for.** ER, mitochondrial and glycolytic transcripts
at the top of a *surface* screen, stromal collagen on the shortlist, and `frac_donors_up` near 0.9
for all of them is the signature of something that scales with the cell rather than with the
antigen. Skin offers two mechanisms, and they need different controls:

- **stromal and epidermal contamination.** Dissociating skin shears keratinocytes, and keratin,
  collagen and melanocyte transcripts are the classic skin ambient and doublet load. A malignant
  cluster that is denser or dirtier picks up more of it.
- **a biosynthetic-activity axis.** A larger, more transcriptionally active malignant cell raises
  every abundant housekeeping transcript at once, including the secretory machinery, without any
  gene being selectively expressed. Depth matching, trimming and the per-cell UMI coupling are the
  tests for this one; the contamination null cannot see it.

§2 puts six tests to both, against an **empirical null** made of transcripts a CD4 T cell cannot
make — the keratins, the collagens, VWF, PMEL, the tryptases, LYZ, HBB. Those genes are what
contamination scores in this cohort, so the thresholds are read off them rather than chosen, and
the section is allowed to conclude that **nothing is excluded**: the calibration verdict prints
before any flag is assigned.

| § | what it does |
|---|---|
| §0b | the cohort fix: 146 donor keys → the independent set, diagnosis from `clin_dx` |
| §1 | input, on the rebuilt QC tables |
| **§2** | **the artifact tests** — six against an empirical contamination null, plus the TCR-call control |
| §3 | malignant vs reactive CD4 on the surviving genes, with per-donor figures |
| §4 | on-target off-tumour: the other T arms, the non-T skin compartments, healthy skin |
| §5–§6 | surface plausibility and escape risk |
| §7 | the ranked table, the control check, and the surfaceome plausibility layer |
| §8 | dossier |
| §9 | the numerical hand-off block |

**Inputs.** `jobs/run_target_screen_qc.sh --tissue skin`, which supersedes `run_target_screen.sh`
for this compartment too: true-library CP10K, robust location statistics, a contamination-filtered
recompute, depth matching, per-cell couplings to the keratinocyte / fibroblast / myeloid modules,
and the lineage-control genes. §2d additionally needs the same job with
`--call mal_tcr --no-cluster --out-dir data/targets/by_call_mal_tcr`. §4 also reads the degradome
pseudobulk cube for the non-T compartments and Reynolds 2021 for healthy skin.

**Outputs** — `tables/tgt42_*`, `figures/targets/tgt42_*`.

> Unlike blood, the skin object carries **all 42,347 genes**, so `CD4` and `TNFRSF8` are present
> and all four positive controls are testable. A gene missing from the screen here is missing
> because the panel ∪ surfaceome selection did not ask for it, not because an HVG filter dropped
> it.

> **This notebook cannot call a target.** It ranks transcripts. CTLA4 and PDCD1 are internalised,
> SELL is shed by ADAM17, and nothing here measures epitope density. There is no CITE-seq in skin,
> so the protein check that 45 runs for blood has no skin equivalent: every candidate below needs
> flow or imaging before anyone says the word target.

```python  # cell 1
# ============================================================================
# §0  Setup
# ============================================================================
import warnings; warnings.filterwarnings("ignore")
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


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

import surface_targets as ST
import target_screen as TS
import target_cohort as TC
import target_qc as TQ
import target_figs as TF
import target_het as TH
import cluster_malig_helpers as CM
import malignancy as MAL
import degradome_data as dcfg          # the non-T pseudobulk cube lives in the degradome build
import external_ref as X               # Reynolds 2021 healthy skin

# ALL THREE, before any path is read. `malignancy` and `target_qc` both default to one
# compartment at import -- `malignancy` to skin, `target_qc` to blood -- so a notebook that sets
# only one of them gets the other's parquets or the other's contaminants, and the first symptom
# is an unrelated KeyError several cells later. The asserts below are the guard, not the comment.
CM.use_tissue("skin")
MAL.use_tissue("skin")
TQ.use_tissue("skin")
assert MAL.CLUSTER_PQ.name.startswith("skin_"), MAL.CLUSTER_PQ
assert "KRT14" in TQ.BAND_GENES, TQ.BAND_GENES

TISSUE = "skin"
TARGETS = NB_DIR / "data" / "targets"
PREFIX = "tgt42_"
MALIG, REACT, HC_CD4 = "CD4_malignant", "CD4_reactive", "CD4_HC"
TREG, CD8, UNASSESSED = "CD4_Treg", "CD8", "CD4_unassessed"

# Gates. Stated once, here, so every table below can be read against them.
MIN_CELLS = 20        # a donor's arm must have this many cells to contribute a point
MIN_DONORS = 5        # a gene needs this many paired donors to be tested
DETECT_FLOOR = 0.10   # "the antigen is present" for prevalence purposes

ST.TAB_PREFIX = ST.FIG_PREFIX = PREFIX
TF.rc()
pd.set_option("display.width", 240)

print("NB_DIR =", NB_DIR)
print("tissue:", CM.TISSUE, "| mal_cluster source:", MAL.CLUSTER_PQ.name,
      "| headline:", MAL.HEADLINE)
print("contamination null:", len(TQ.BAND_GENES), "genes |",
      "coupling modules:", TQ.COUPLING_MODULES[TISSUE])
```

```
NB_DIR = /home/projects/nyosef/zvise/MF
tissue: skin | mal_cluster source: skin_T_malignancy_cluster_v1.parquet | headline: mal_cluster
contamination null: 40 genes | coupling modules: ['keratinocyte', 'fibroblast', 'myeloid']
```

## §0b — The cohort, before any statistic

`donor` in the target tables is not a patient. Three corrections, all read from the curated
`clin_*` columns (`docs/CLINICAL_METADATA.md`), none of them guessed here:

- **the clone is not CD4.** `clin_phenotype` is the lineage of the *malignant* clone, and 19 keys
  carry CD8, γδ or double-negative disease — the four pcAECTCL cases, D6 MF5 (CD8 aggressive
  cytotoxic, `CLINICAL_METADATA.md:145`), the Berti CD8 and γδ-MF patients of D1, and two DN. The
  malignancy call looks for a malignant CD4 cluster and finds none, so **every one of their CD4
  cells enters `CD4_reactive`** — the benign side of the comparison — while their actual tumour
  sits in the CD8 arm. This is the correction that matters most and it has nothing to do with
  pseudo-replication.
- **the diagnosis is outside the MF/SS spectrum.** Seven LyP, six SPP and two LPP keys. LyP is a
  CD30+ lymphoproliferative disorder and parapsoriasis is a precursor state; both are interesting
  and neither belongs inside an MF/SS tumour-vs-reactive contrast.
- **one patient under several keys.** Ten patients were biopsied more than once —
  `vienna:MF318` under four keys, `vienna:MF311` under three, `pittsburgh:MF19` in both this
  atlas and the blood one. The lesional sample is kept (the deepest, among lesions), the rest are
  dropped. Keys that differ only by an HTO index are **not** collapsed: an HTO index demultiplexes
  different donors that share a patient key, which is the `D13__HB1-3` case in blood.

Diagnosis comes from `clin_dx` throughout. The atlas `disease` column disagrees with it for 27
keys — it files every LyP, SPP, LPP and pcAECTCL case under `CTCL_other` and calls six SS patients
MF — and is not read again in this notebook.

Unlike blood, no healthy donor is mislabelled here: `disease == "HC"` and `clin_dx == "HC"` name
the same 22 keys, so the `CD4_HC` arm needed no repair.

`burden` — the malignant share of a donor's CD4 compartment — is attached here. §2 uses it as the
mechanistic test for contamination.

```python  # cell 3
# ============================================================================
# §0b  Cohort: which donor keys are independent MF/SS patients with a CD4 clone
# ============================================================================
don = TC.skin_donor_table()
don.to_csv(ST.tab("skin_cohort"), index=True)

_raw = pd.read_parquet(TARGETS / f"{TISSUE}_target_long.parquet")
long = TC.filter_long(_raw, don)
bur = TC.burden(long)
don = don.join(bur, how="left")

print(TC.cohort_report(don, long))
print("\nwrote", ST.tab("skin_cohort"))

print("\n" + "=" * 78)
print("dropped keys, every one with its reason:")
print(don.loc[~don["cohort_keep"], ["study", "dx", "phenotype", "lesion", "person", "n_cells",
                                    "drop_reason"]]
      .sort_values(["drop_reason", "study"]).to_string())

print("\n" + "=" * 78)
print("the independent cohort, by what it can be split on:")
_k = don[don["cohort_keep"]]
for col, label in (("dx", "diagnosis"), ("lesion", "lesion biopsied"),
                   ("clin_stage_class", "stage class"), ("sex", "sex")):
    print(f"  {label:18s} " + ", ".join(f"{k}:{v}" for k, v in
                                        _k[col].value_counts(dropna=False).items()))
print(f"  {'age':18s} {int(_k['age'].notna().sum())} donors, median "
      f"{_k['age'].median():.0f}")
print(f"  {'treatment known':18s} {int(_k['tx_known'].sum())} donors")
print("\nburden -- the malignant share of a donor's CD4 compartment, from the arm cell counts "
      "themselves. In skin it is a biopsy property as much as a patient property (a tumour "
      "lesion is not a patch), and §2 uses it as the mechanistic test for contamination:")
print(_k["burden"].describe(percentiles=[.25, .5, .75]).round(3).to_string())
```

```
skin donor keys in the object: 146
  -   7 dropped: malignant clone is not CD4 (gd)
  -   7 dropped: diagnosis outside the MF/SS spectrum (LyP)
  -   6 dropped: diagnosis outside the MF/SS spectrum (SPP)
  -   6 dropped: malignant clone is not CD4 (CD8)
  -   2 dropped: malignant clone is not CD4 (DN)
  -   2 dropped: same person as D7__MF311
  -   2 dropped: diagnosis outside the MF/SS spectrum (LPP)
  -   1 dropped: same person as D7__MF309
  -   1 dropped: same person as D8__P90
  -   1 dropped: same person as D8__P65
  -   1 dropped: same person as D1__P312
  -   1 dropped: same person as D13__MF19
  = 109 independent donor keys (107 people)

dx x study, independent keys:
study  alkon2024  brentuximab2026  brunner2024  buus2025  chennareddy2025  gaydosik2019  gaydosik2022  gaydosik2023  herrera2021  il4ra2026  jonak2021  li2024  rindler2021_mc
dx                                                                                                                                                                            
HC             0                0            0         0                4             4             4             5            1          4          0       0               0
MF             5                6            3         0                5             2             3             0            1         10          1      35               4
MF/SS          0                0            0         0                0             0             2             0            0          0          0       0               0
SS             0                0            2         3                0             1             0             0            3          1          0       0               0

lesion class, independent keys: lesion 77, healthy 22, unspecified 10
ex vivo 2-day culture (vehicle arm kept, flagged): 7 keys
shared dominant TRB clone (buus PT05/06/18):       0 keys

donors with BOTH a malignant and a reactive arm: 75 (75 people)
dx
MF       67
SS        6
MF/SS     2

wrote /home/projects/nyosef/zvise/MF/tables/tgt42_skin_cohort.csv

==============================================================================
dropped keys, every one with its reason:
                              study             dx       phenotype        lesion             person  n_cells                                 drop_reason
donor                                                                                                                                                   
D8__P143                  alkon2024            LPP  not_applicable         patch        vienna:P143      859  diagnosis outside the MF/SS spectrum (LPP)
D8__P194                  alkon2024            LPP  not_applicable         patch        vienna:P194     2166  diagnosis outside the MF/SS spectrum (LPP)
D9__P101                    lyp2026            LyP             CD4    LyP_lesion        vienna:P101     3509  diagnosis outside the MF/SS spectrum (LyP)
D9__P118                    lyp2026            LyP             CD4    LyP_lesion        vienna:P118      892  diagnosis outside the MF/SS spectrum (LyP)
D9__P193                    lyp2026            LyP              gd    LyP_lesion        vienna:P193    11896  diagnosis outside the MF/SS spectrum (LyP)
D9__P213                    lyp2026            LyP             CD8    LyP_lesion        vienna:P213     7308  diagnosis outside the MF/SS spectrum (LyP)
D9__P217                    lyp2026            LyP             CD4    LyP_lesion        vienna:P217      625  diagnosis outside the MF/SS spectrum (LyP)
D9__P220                    lyp2026            LyP             CD8    LyP_lesion  vienna:GSM9038965     5259  diagnosis outside the MF/SS spectrum (LyP)
D9__PEN                     lyp2026            LyP             CD8    LyP_lesion         vienna:PEN     4739  diagnosis outside the MF/SS spectrum (LyP)
D8__P102                  alkon2024            SPP  not_applicable         patch        vienna:P102    
... [5030 more chars truncated]
```

## §1 — Input, and the four things that make skin different from blood

1. **The object carries all 42,347 genes.** The screen tests the panel ∪ surfaceome selection
   (~2,100 genes), but absence from that selection is a choice this project made, not an HVG
   filter: `CD4` and `TNFRSF8` are both present, so all four positive controls are testable —
   which is why §7's control check is a sharper instrument here than in blood.
2. **The malignant fraction is a property of the lesion.** Blood in SS is ~92% malignant CD4 and
   the reactive arm barely exists; a skin biopsy ranges from a patch with a small clone to a
   tumour that is almost all clone. 94 of 97 donors with a malignant arm also have a reactive
   one, so the paired test is well powered — and `burden` spans most of its range, which makes
   the burden-scaling test in §2 far more informative than it can be in blood.
3. **The contaminants are the tissue.** Keratinocytes, fibroblasts, endothelium, melanocytes,
   mast cells and macrophages are all present in the same dissociation, and keratin and collagen
   are the classic skin ambient transcripts. `CD4_unassessed` (14 donors, 39,692 cells) is carried
   but never used as a comparator: those are cells the call could not be evaluated on, and
   folding them into "reactive" would make the reactive arm mean "malignant was not detected".
4. **The tables are the QC build.** `mean_all` is log1p CP10K on the **true** library
   (`obs.total_counts`), with `trim1_mean`, `wins90_mean`, `mean_all_clean` (lineage-positive and
   high-doublet cells dropped) and `mean_all_ds` (binomially thinned to a common depth) beside
   it. Those four columns are what separates a selective antigen from a cell that is simply
   making more of everything.

```python  # cell 5
# ============================================================================
# §1  Load the rest of the QC build
# ============================================================================
meta = TC.filter_long(pd.read_parquet(TARGETS / f"{TISSUE}_target_meta.parquet"), don)
man = json.loads((TARGETS / f"{TISSUE}_target_manifest.json").read_text())
cl_path = TARGETS / f"{TISSUE}_target_cluster.parquet"
cluster_long = TC.filter_long(pd.read_parquet(cl_path), don) if cl_path.exists() else None
qc_path = TARGETS / f"{TISSUE}_target_qc_cell.parquet"
qc_cell = pd.read_parquet(qc_path) if qc_path.exists() else None

assert man.get("builder") == "run_target_screen_qc.py", (
    "these tables came from the v1 builder, which has no robust statistics, no clean recompute "
    "and no lineage controls -- run jobs/run_target_screen_qc.sh --tissue skin first. "
    f"manifest says: {man.get('builder', 'run_target_screen.py')}")

print(f"source {Path(man['source']).name} | call {man['malignancy_call']} | "
      f"{man['n_genes_in_object']:,} genes in object -> "
      f"{man['gene_selection'].get('n_total', man['gene_selection'].get('n_screened')):,} "
      f"screened (+{man['gene_selection'].get('n_lineage_added', 0)} lineage controls, "
      f"+{man['gene_selection'].get('n_traffic_added', 0)} trafficking)")
print("library:", man.get("library"))
print("clean rule:", json.dumps(man.get("clean_rule", {}), default=str))
print(f"depth matching: {man.get('ds_depth'):,} UMI on {man.get('frac_cells_ds', 0):.1%} "
      "of cells")
print("coupling modules:", man.get("coupling_modules"))

arms = (meta.groupby("arm").agg(donors=("donor", "nunique"), cells=("n_cells", "sum"),
                                studies=("study", "nunique"))
        .sort_values("cells", ascending=False))
print("\narms, independent cohort:")
print(arms.to_string())

paired = meta[meta["arm"].isin([MALIG, REACT])].groupby("donor")["arm"].nunique().eq(2)
PAIRED_DONORS = sorted(paired.index[paired])
print(f"\ndonors with both arms: {len(PAIRED_DONORS)} of "
      f"{meta.loc[meta['arm'].eq(MALIG), 'donor'].nunique()} with a malignant arm")
DONOR_STUDY = dict(zip(meta["donor"], meta["study"]))

print("\ncontamination load per arm (median over donors) -- the asymmetry an artifact needs:")
_cl = [c for c in ("frac_lineage_pos", "median_total_counts", "median_doublet_score",
                   "score_keratinocyte", "score_fibroblast", "score_myeloid",
                   "score_endothelial", "score_melanocyte", "score_mast")
       if c in meta.columns]
print(meta.groupby("arm")[_cl].median().round(4).to_string())
print("\nA malignant arm that is dirtier than its reactive arm on these rows is the precondition "
      "for the whole of §2. One that is not says the contamination story cannot be the "
      "explanation, whatever the gene names look like.")
```

```
source skin_T_tcr_annotated_v4.h5ad | call mal_cluster | 42,347 genes in object -> 2,355 screened (+30 lineage controls, +210 trafficking)
library: obs.total_counts (mean_all); HVG column sum kept as mean_all_hvg
clean rule: {"strict_lineage_markers": ["C1QA", "C1QB", "CD14", "CD163", "CD68", "CD79A", "CDH5", "CLDN5", "COL1A1", "COL1A2", "COL3A1", "CPA3", "DCN", "DCT", "DMKN", "HBA2", "HBB", "JCHAIN", "KRT1", "KRT10", "KRT14", "KRT5", "KRTDAP", "LUM", "LYZ", "MLANA", "MS4A1", "MZB1", "PMEL", "SFN", "TPSAB1", "TPSB2", "TYRP1", "VWF"], "strict_min_counts": 2, "strict_min_genes": 2, "frac_lineage_pos": 0.391025641025641, "doublet_quantile_per_study": 0.9, "frac_cells_clean": 0.58430385467369}
depth matching: 2,000 UMI on 54.6% of cells
coupling modules: ['keratinocyte', 'fibroblast', 'myeloid']

arms, independent cohort:
                donors   cells  studies
arm                                    
CD4_malignant       78  187748       12
CD4_reactive        82  147233       12
CD8                 86   45562       13
CD4_Treg            77   35560       13
CD4_HC              21    4724        5
CD4_unassessed       2      71        1

donors with both arms: 75 of 78 with a malignant arm

contamination load per arm (median over donors) -- the asymmetry an artifact needs:
                frac_lineage_pos  median_total_counts  median_doublet_score  score_keratinocyte  score_fibroblast  score_myeloid  score_endothelial  score_melanocyte  score_mast
arm                                                                                                                                                                              
CD4_HC                    0.3673              2368.00                0.0183              0.1869            0.0403         0.0426             0.0117            0.0011      0.0010
CD4_Treg                  0.3091              2549.00                0.0153              0.0859            0.0119         0.0198             0.0055            0.0014      0.0000
CD4_malignant             0.4267              3330.75                0.0205              0.1161            0.0255         0.0372             0.0100            0.0020      0.0007
CD4_reactive              0.3318              2038.50                0.0138              0.0962            0.0152         0.0624             0.0163            0.0017      0.0007
CD4_unassessed            0.7022             12662.50                0.0311              0.2137            0.0224         0.0662             0.0264            0.0083      0.0000
CD8                       0.3046              2107.75                0.0146              0.1122            0.0157         0.0390             0.0349            0.0005      0.0000

A malignant arm that is dirtier than its reactive arm on these rows is the precondition for the whole of §2. One that is not says the contamination story cannot be the explanation, whatever the gene names look like.
```

## §2 — Is the ranking an artifact? A test that is allowed to fail

The charge against the first ranking is specific, so the test can be too.

**Two mechanisms, not one.**

*Contamination.* A skin dissociation shears keratinocytes and releases collagen; those transcripts
enter droplets as ambient RNA and as doublets. Ambient RNA is shared by both arms and cannot by
itself make a gene look selective — but doublets are not shared: if the malignant clusters are
denser, dirtier or sit in a different part of the lesion, they take up more of the tissue around
them, and every stromal transcript rises in the malignant arm in every patient at once.

*Biosynthetic activity.* A malignant T cell that is larger and more transcriptionally active than
the small reactive lymphocyte beside it raises **every** abundant transcript — ENO1, HSP90AA1,
VDAC1, CALU, the whole ER and glycolytic programme — without expressing any of them selectively.
This produces exactly the first pass's top list, and the contamination null cannot see it: it is
not contamination. The tests that can are depth matching (`mean_all_ds`), trimming, and the
per-cell coupling to library size.

**Six tests**, each with its own failure mode:

| test | what fires it |
|---|---|
| `fails_trim` | the effect collapses when the top 1% of cells is trimmed — carried by a handful of cells, not by the population |
| `fails_concentration` | more of the arm's counts sit in its top 1% of cells than for other genes *detected at the same rate* |
| `fails_lineage` | per-cell expression tracks the keratinocyte, fibroblast or myeloid module score **inside** the malignant arm |
| `fails_umi` | it tracks per-cell library size, or detection is confined to the top-UMI decile, beyond what its prevalence implies |
| `fails_clean` | it does not survive dropping lineage-positive and high-doublet-score cells |
| `fails_burden` | its per-donor effect grows with that donor's malignant:reactive abundance ratio |

Two failures ⇒ `artifact`, one ⇒ `suspect`. Only up-in-malignant genes can be flagged;
contamination entering the malignant arm cannot push a gene down.

**The null, and the escape hatch.** The keratins, `KRTDAP`, `DMKN`, `SFN`, the collagens, `DCN`,
`LUM`, `VWF`, `CDH5`, `CLDN5`, `PMEL`, `MLANA`, `TYRP1`, `DCT`, `LYZ`, `CD14`, `CD68`, `C1QA/B`,
`CD163`, the tryptases, `CPA3`, `MS4A1`, `CD79A`, `JCHAIN`, `MZB1`, `HBB` and `HBA2` are not CD4
T-cell transcripts, so whatever they score *is* what contamination scores in this cohort, and
every threshold is read off them rather than chosen. That also makes the section refutable: **if
those genes are not up in the malignant arm, the mechanism is not operating here, and §2b excludes
nothing.** The calibration verdict prints before any flag is assigned — read it first, because it
decides whether the rest of the section means anything. `PECAM1` (CD31) and `HLA-DRA` are
deliberately out of the null: T cells carry both.

**Why the statistics are prevalence-matched.** `top1_share`, the top-UMI-decile odds ratio and the
module correlations are all functions of how often a gene is detected at all: a gene seen in two
cells necessarily concentrates, and a gene seen in 80% of cells necessarily rises with depth
because detection saturates. Compared raw, the band would condemn CD7 and acquit a gene seen once.
Each is taken as a residual against genes of the same detection rate.

**§2d adds the two artifacts that are not contamination** — the malignancy call being a cluster
label while the comparison is between those same clusters, and the effect living in one or two
clusters rather than in the clone. The control for the first is the whole screen rebuilt with the
arms taken from the TCR clone, which is measured independently of the transcriptome.

```python  # cell 7
# ============================================================================
# §2a  The six artifact statistics, and the empirical contamination null
# ============================================================================
# The same paired test, five times, on five location estimators. An effect carried by a handful
# of cells survives only the first; an effect that is really a depth difference dies in the last.
sel = TS.paired_selectivity(long, MALIG, REACT, value="mean_all", min_donors=MIN_DONORS)
sel_trim = TS.paired_selectivity(long, MALIG, REACT, value="trim1_mean", min_donors=MIN_DONORS)
sel_wins = TS.paired_selectivity(long, MALIG, REACT, value="wins90_mean", min_donors=MIN_DONORS)
sel_clean = TS.paired_selectivity(long, MALIG, REACT, value="mean_all_clean",
                                  min_donors=MIN_DONORS)
sel_ds = TS.paired_selectivity(long, MALIG, REACT, value="mean_all_ds", min_donors=MIN_DONORS)
blk = TS.blocked_selectivity(long, MALIG, REACT, block=DONOR_STUDY, value="mean_all")
sel = sel.merge(blk, on="gene", how="left")

conc = TQ.concentration(long, MALIG)                       # top1/top5 share, detection
brn = TQ.burden_scaling(long, bur, MALIG, REACT)           # does the effect scale with burden
confine = (TQ.cluster_confinement(cluster_long, MALIG)
           if cluster_long is not None else None)

ev = TQ.assemble(sel, sel_trim, sel_clean, qc_cell, brn, conc, confine)
for name, s in (("wins", sel_wins), ("ds", sel_ds)):
    ev = ev.merge(s[["gene", "log2fc_paired"]].rename(
        columns={"log2fc_paired": f"log2fc_{name}"}), on="gene", how="left")
    ev[f"retained_{name}"] = TQ._ratio(ev[f"log2fc_{name}"], ev["log2fc_paired"])
ev = TQ.prevalence_residual(ev)

print(f"{len(sel):,} genes tested paired on {len(PAIRED_DONORS)} donors | "
      f"{int((sel['q'] < 0.05).sum()):,} at q < 0.05")

print("\n" + "=" * 78)
print("THE NULL: what pure contamination scores in this cohort")
print("=" * 78)
ok, msg = TQ.band_is_calibrated(ev, None)
print(msg)
band = TQ.contamination_band(ev)
print(f"\nlineage controls found: {band.attrs['n_genes']} "
      f"({', '.join(band.attrs['genes_found'])})")
print("\nper-gene, in full:")
_bcols = [c for c in ["gene", "log2fc_paired", "frac_donors_up", "median_pct_expr",
                      "retained_trim", "retained_clean", "retained_ds", "top1_share_excess",
                      *(f"{c}_excess" for c in TQ.coupling_cols()),
                      "or_topdecile_excess", "rho_burden"]
          if c in ev.columns]
print(ev[ev["gene"].isin(TQ.BAND_GENES)][_bcols]
      .sort_values("log2fc_paired", ascending=False)
      .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
print("\nquantiles of the band (thresholds are read from the p90 row):")
print(band.round(4).to_string())

# The activity axis, which the band above cannot test. If the malignant arm is simply deeper,
# `retained_ds` -- the share of the effect that survives thinning every cell to a common depth --
# falls for the whole ranking at once, not for a handful of genes.
print("\n" + "=" * 78)
print("THE OTHER MECHANISM: how much of each effect survives depth matching")
print("=" * 78)
_up = ev[(ev["q"] < 0.05) & (ev["log2fc_paired"] > 0.3)]
print(f"genes up at q < 0.05 with log2FC > 0.3: {len(_up)}")
for nm, c in (("trim top 1%", "retained_trim"), ("winsorise p90", "retained_wins"),
              ("drop dirty cells", "retained_clean"), ("match depth", "retained_ds")):
    if c in _up:
        print(f"  {nm:18s} median retained {_up[c].median():.2f} | "
              f"{int((_up[c] < 0.5).sum())} of {len(_up)} genes keep less than half")
print("\nIf `match depth` alone collapses the ranking, the screen was measuring library size. "
      "If every column collapses together, it was measuring a handful of cells.")
```

```
2,355 genes tested paired on 75 donors | 985 at q < 0.05

==============================================================================
THE NULL: what pure contamination scores in this cohort
==============================================================================
40 lineage controls: median log2FC +0.00, median frac_donors_up 0.35 -- the controls are NOT up in the malignant arm, so differential contamination is not what this cohort shows. Do not exclude any gene on this basis; find the real mechanism first.

lineage controls found: 40 (C1QA, C1QB, CD14, CD163, CD68, CD79A, CDH5, CLDN5, COL1A1, COL1A2, COL3A1, COL6A2, CPA3, DCN, DCT, DMKN, EGFL7, HBA2, HBB, JCHAIN, KRT1, KRT10, KRT14, KRT17, KRT5, KRT6A, KRTDAP, LUM, LYZ, MLANA, MS4A1, MS4A2, MZB1, PMEL, SFN, SFRP2, TPSAB1, TPSB2, TYRP1, VWF)

per-gene, in full:
  gene  log2fc_paired  frac_donors_up  median_pct_expr  retained_trim  retained_clean  retained_ds  top1_share_excess  r_keratinocyte_excess  r_fibroblast_excess  r_myeloid_excess  or_topdecile_excess  rho_burden
   SFN        0.04547            0.56          0.06343              0          0.5459            0            0.01715                 0.3343             0.006717           0.01136              -0.4659      0.2957
  CD68        0.03461          0.5333          0.02084              0               0           -1           -0.04983               0.007509             0.004229            0.3418              -0.1176     -0.2034
 KRT10        0.02607          0.5333           0.4078         0.6464              -1           -1                  0                  0.719             0.002063          0.005635              -0.3544    -0.09417
 EGFL7        0.02159            0.56         0.004552              0               0            0                  0              -0.004297               0.0157          0.001719               0.1441      0.1161
COL1A1        0.01534          0.5333          0.03572              0               0            0           0.002923                 0.0079               0.5899           0.01321              -0.5517    -0.03563
 KRT17       0.006603          0.5333          0.01562              0               0            0           -0.08479                 0.1872            0.0002126           0.00765              -0.3221      0.0918
  C1QB       0.004421          0.5067         0.006523              0               0            0                  0                0.00866             0.007884            0.2005              -0.3854     0.03271
  CD14              0          0.4533         0.002413            NaN             NaN          NaN                  0                0.00158             0.001698            0.1529             -0.07638    -0.02521
 SFRP2              0          0.3467         0.001083            NaN             NaN          NaN                  0                0.01343               0.2771          0.009333              -0.1048      0.3009
   VWF              0          0.4267         0.000995            NaN             NaN          NaN                  0             -0.0009722            -0.001346          0.002557              -0.1907     0.03451
 TYRP1              0          0.2933         0.001169            NaN             NaN          NaN                  0                0.01347             0.001552         0.0004516              -0.3302  -0.0005888
  PMEL              0          0.4933          0.00269            NaN             NaN          NaN                  0               0.000894               -0.005         -0.001256              -0.1974     0.05468
   HBB              0          0.3333         0.005974            NaN             NaN          NaN           -0.05033              -0.004241            -0.008877          0.008231              -0.5981     0.05612
  HBA2              0          0.3333         0.001269            NaN             NaN          NaN                  0               -0.01146             -0.01004         -0.005355        
... [7615 more chars truncated]
```

```python  # cell 8
# ============================================================================
# §2b  The verdict -- which is only allowed to exclude anything if the null is calibrated
# ============================================================================
out = TQ.artifact_evidence(ev, band)
if not ok:
    # The lineage controls are the whole basis for calling a gene contamination. If they are not
    # up in the malignant arm, the mechanism is absent from this cohort and the six tests are
    # measuring something else. Nothing is excluded; the evidence is still printed, because the
    # columns are informative about every gene whether or not contamination is the explanation.
    out["artifact_flag"] = "clean"
    out["fail_reasons"] = "- (null not calibrated; no exclusion applied)"
    print("*" * 78)
    print("NO GENE IS EXCLUDED ON CONTAMINATION GROUNDS.")
    print(msg)
    print("*" * 78)

ev = out
print()
print(TQ.verdict_text(ev))

print("\n" + "=" * 78)
print("the six tests, per gene, for every gene with a positive effect at q < 0.05")
print("=" * 78)
_full = ev[(ev["q"] < 0.05) & (ev["log2fc_paired"] > 0)].sort_values("log2fc_paired",
                                                                    ascending=False)
_fc = [c for c in ["gene", "log2fc_paired", "frac_donors_up", "median_pct_expr",
                   "retained_trim", "retained_wins", "retained_clean", "retained_ds",
                   "top1_share_excess", *(f"{c}_excess" for c in TQ.coupling_cols()),
                   "r_umi_excess", "or_topdecile_excess", "rho_burden", "top_cluster_share",
                   "n_fails", "fail_reasons"] if c in ev.columns]
print(_full[_fc].to_string(index=False, float_format=lambda v: f"{v:.4g}"))

ev.to_csv(ST.tab("artifact_evidence"), index=False)
band.to_csv(ST.tab("contamination_band"))
print("\nwrote", ST.tab("artifact_evidence"), "and", ST.tab("contamination_band"))
print("\nColumn key:  retained_* = effect kept after trimming the top 1% of cells / "
      "winsorising at the 90th percentile / dropping lineage-positive and high-doublet cells / "
      "binomially thinning to a common depth. 1.0 = unchanged, 0 = gone. *_excess = the "
      "statistic minus the cohort median for genes detected at the same rate. rho_burden = "
      "Spearman of the per-donor effect against that donor's malignant:reactive ratio.")
```

```
******************************************************************************
NO GENE IS EXCLUDED ON CONTAMINATION GROUNDS.
40 lineage controls: median log2FC +0.00, median frac_donors_up 0.35 -- the controls are NOT up in the malignant arm, so differential contamination is not what this cohort shows. Do not exclude any gene on this basis; find the real mechanism first.
******************************************************************************

artifact_flag
clean    2355

the genes that prompted this, by name:
  gene  log2fc_paired  frac_donors_up  median_pct_expr  retained_trim  retained_clean  top1_share_excess  r_keratinocyte_excess  r_fibroblast_excess  r_myeloid_excess  or_topdecile_excess  rho_burden  n_fails artifact_flag                                  fail_reasons
COL6A3       0.810520        0.880000         0.112559       1.025831        1.148306           0.022876              -0.013865             0.017665          0.005285             0.132139    0.064609        2         clean - (null not calibrated; no exclusion applied)
   GRN       0.777706        0.906667         0.210751       1.078762        0.885650           0.001729               0.017008             0.004042          0.028189             0.529467    0.070868        1         clean - (null not calibrated; no exclusion applied)
LGALS3       0.663436        0.586667         0.427148       1.073072        0.861482           0.035563               0.026563            -0.003736          0.014822            -0.068130   -0.074367        1         clean - (null not calibrated; no exclusion applied)
 SYPL1       0.598749        0.906667         0.362422       1.053856        0.896051          -0.010097               0.009600            -0.000845          0.013970             0.220476    0.162674        1         clean - (null not calibrated; no exclusion applied)
  CALU       0.586939        0.880000         0.190358       1.146000        1.005408          -0.000410              -0.000574             0.001211          0.001846             0.358469    0.155334        1         clean - (null not calibrated; no exclusion applied)
 QSOX1       0.583663        0.853333         0.133156       1.177422        1.073486          -0.018471              -0.001065             0.020548          0.011848             0.352712   -0.163129        1         clean - (null not calibrated; no exclusion applied)
 VDAC1       0.583615        0.946667         0.481712       1.022390        0.976308           0.014870              -0.001857            -0.004360          0.008609             1.170785   -0.136387        2         clean - (null not calibrated; no exclusion applied)
   GPI       0.506628        0.826667         0.473935       1.032367        0.966124           0.017506              -0.000104             0.002169          0.006070             0.479834    0.019260        2         clean - (null not calibrated; no exclusion applied)

excluded from the headline (0 genes), every one with its reason:
Empty DataFrame
Columns: [gene, log2fc_paired, frac_donors_up, median_pct_expr, retained_trim, retained_clean, top1_share_excess, r_keratinocyte_excess, r_fibroblast_excess, r_myeloid_excess, or_topdecile_excess, rho_burden, n_fails, artifact_flag, fail_reasons]
Index: []

==============================================================================
the six tests, per gene, for every gene with a positive effect at q < 0.05
==============================================================================
     gene  log2fc_paired  frac_donors_up  median_pct_expr  retained_trim  retained_wins  retained_clean  retained_ds  top1_share_excess  r_keratinocyte_excess  r_fibroblast_excess  r_myeloid_excess  r_umi_excess  or_topdecile_excess  rho_burden  top_cluster_share  n_fails                                  fail_reasons
     CD70          1.102            0.92           0.2096          1.102          1.851          0.8991        0.693            0.05777               0.007685       
... [166267 more chars truncated]
```

```python  # cell 9
# ============================================================================
# §2c FX  The null, and where the disputed genes actually sit
# ============================================================================
SETS = {"lineage controls\n(pure contamination)": TQ.BAND_GENES,
        "all screened genes": sorted(set(ev["gene"]) - set(TQ.BAND_GENES)),
        "disputed genes": TQ.SUSPECTS,
        "validated antigens": ST.POSITIVE_CONTROLS}
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4),
                         gridspec_kw={"width_ratios": [1.05, 1.0]})

ax = axes[0]
rng = np.random.default_rng(0)
for i, (lab, gs) in enumerate(SETS.items()):
    v = ev.loc[ev["gene"].isin(gs), "log2fc_paired"].dropna()
    col = [TF.DOWN, "#bdbdbd", "#2e86c1", TF.UP][i]
    ax.scatter(np.full(len(v), i) + rng.normal(0, .07, len(v)), v, s=7 if i == 1 else 18,
               color=col, alpha=.35 if i == 1 else .9, linewidths=0)
    if len(v):
        ax.hlines(np.median(v), i - .25, i + .25, color="#111111", lw=1.4, zorder=5)
ax.axhline(0, lw=.5, c="k")
ax.set_xticks(range(len(SETS)))
ax.set_xticklabels(SETS, fontsize=6)
ax.set_ylabel("log2FC, malignant vs reactive CD4")
ax.set_title("Contamination null vs candidates", loc="left", fontsize=7.5)
TF._clean(ax)

# Right: the three-arm view. Contamination that rides into the malignant arm has no reason to
# spare healthy CD4 -- a healthy skin biopsy carries keratinocytes and fibroblasts too, and more
# of them than a dense infiltrate does. A transcript absent from healthy and reactive CD4 and
# present in the clone is tumour-restricted, whatever one thinks of its gene name.
ax = axes[1]
show = TQ.SUSPECTS + ST.POSITIVE_CONTROLS + ["CD7"]
det = (long[long["gene"].isin(show) & long["arm"].isin([HC_CD4, REACT, MALIG])]
       .pivot_table(index="gene", columns="arm", values="pct_expr", aggfunc="median")
       .reindex([g for g in show if g in set(long["gene"])])
       .sort_values(MALIG))
for j, (arm, col, mk) in enumerate(((HC_CD4, TF.COMPARATOR, "o"), (REACT, TF.REACT, "o"),
                                    (MALIG, TF.MALIG, "o"))):
    if arm in det:
        ax.scatter(det[arm], np.arange(len(det)) + (j - 1) * 0.2, s=22, color=col,
                   marker=mk, linewidths=0, label=arm.replace("CD4_", ""))
for y, (_, r_) in enumerate(det.iterrows()):
    ax.plot([r_.min(), r_.max()], [y, y], color="#dddddd", lw=0.8, zorder=0)
ax.set_yticks(range(len(det)))
ax.set_yticklabels([TF.it(g) for g in det.index], fontsize=6.5)
ax.set_xlabel("median fraction of a donor's cells detected")
ax.set_xscale("symlog", linthresh=1e-3)
ax.set_xlim(left=0)
ax.set_ylim(-0.7, len(det) - 0.3)
ax.legend(fontsize=5.5, frameon=False, loc="upper left", title="CD4 arm",
          title_fontsize=5.5)
ax.set_title("Detection across CD4 arms", loc="left", fontsize=7.5)
TF._clean(ax)

fig.tight_layout()
TF.save(fig, "FX_artifact_null", caption_text=(
    "Left: paired log2FC by gene set. Right: median detection per CD4 arm, symlog x."))
TF.save_source(det.reset_index(), "FX_three_arm_detection_source")
TF.logfig("FX_artifact_null", "contamination null vs candidates; three-arm detection",
          len(ev))

print("median detection by arm, the genes in question (and the controls):")
print(det.round(4).to_string())
```

```
wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt42_FX_artifact_null.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt42_FX_artifact_null.pdf

<Figure size 920x340 with 2 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt42_FX_three_arm_detection_source.csv
median detection by arm, the genes in question (and the controls):
arm      CD4_HC  CD4_malignant  CD4_reactive
gene                                        
KIR3DL2  0.0026         0.0000        0.0037
CD7      0.2113         0.0380        0.5391
TNFRSF8  0.0000         0.0380        0.0111
COL6A3   0.0158         0.1126        0.0247
IL2RA    0.0828         0.1224        0.1261
QSOX1    0.0261         0.1332        0.0677
CALU     0.0392         0.1904        0.0944
GRN      0.0410         0.2108        0.1028
CCR4     0.0656         0.2190        0.1349
SYPL1    0.1786         0.3624        0.2000
LGALS3   0.3607         0.4271        0.2919
GPI      0.1250         0.4739        0.3033
VDAC1    0.1745         0.4817        0.2925
```

```python  # cell 10
# ============================================================================
# §2d  Two more ways the effect could be an artifact, and neither is contamination
# ============================================================================
# (1) CLUSTER CIRCULARITY. The skin malignancy call is `mal_cluster` -- a *cluster* label. The
#     arms are therefore groups of Leiden clusters, and the comparison is between the same
#     clusters whose expression defined them. A gene confined to one or two malignant clusters
#     would be a cluster marker rediscovered as an antigen.
if confine is not None:
    print("share of the malignant arm's signal held by its single strongest cluster "
          f"({int(confine['n_clusters'].median())} clusters):")
    _q = confine["top_cluster_share"].describe(percentiles=[.5, .9, .99]).round(4)
    print("  all screened genes:", _q.to_dict())
    print("\n  the genes in question:")
    print(confine[confine["gene"].isin(TQ.SUSPECTS + ST.POSITIVE_CONTROLS)]
          .sort_values("top_cluster_share", ascending=False)
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n  A gene spread evenly over N clusters scores 1/N. Confinement would show as a "
          "share far above that floor.")

# (2) THE CALL ITSELF. `mal_tcr` assigns malignancy from the expanded TCR clone, which is
#     measured independently of the transcriptome. If an effect holds under that call it cannot
#     be an artifact of how the clusters were drawn. Built by:
#         bash jobs/run_target_screen_qc.sh --tissue skin --call mal_tcr --no-cluster \
#              --out-dir data/targets/by_call_mal_tcr
TCR_DIR = TARGETS / "by_call_mal_tcr"
_keep = ["gene", "log2fc_paired", "frac_donors_up", "n_donors_paired"]
if (TCR_DIR / f"{TISSUE}_target_long.parquet").exists():
    long_tcr = TC.filter_long(pd.read_parquet(TCR_DIR / f"{TISSUE}_target_long.parquet"), don)
    sel_tcr = TS.paired_selectivity(long_tcr, MALIG, REACT, value="mean_all",
                                    min_donors=MIN_DONORS)
    both_calls = sel[_keep].merge(sel_tcr[_keep], on="gene", suffixes=("_cluster", "_tcr"))
    _r = stats.spearmanr(both_calls["log2fc_paired_cluster"], both_calls["log2fc_paired_tcr"])
    print("\n" + "=" * 78)
    print(f"the same screen with the arms built from the TCR clone instead of the cluster: "
          f"rho = {_r.statistic:.2f} over {len(both_calls):,} genes "
          f"({both_calls['n_donors_paired_tcr'].median():.0f} paired donors vs "
          f"{both_calls['n_donors_paired_cluster'].median():.0f})")
    print("=" * 78)
    print(both_calls[both_calls["gene"].isin(TQ.SUSPECTS + ST.POSITIVE_CONTROLS
                                             + ST.LOSS_MARKERS)]
          .sort_values("log2fc_paired_cluster", ascending=False)
          .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    both_calls.to_csv(ST.tab("call_sensitivity"), index=False)
    print("\nAn effect that is an artifact of the clustering cannot survive a call made from "
          "the TCR. One that does survive is a property of the clone. Skin TCR recovery is "
          "patchier than blood's, so the paired-donor counts above are the limit on this "
          "control, not the correlation.")
    print("wrote", ST.tab("call_sensitivity"))
else:
    print(f"\n{TCR_DIR.name} not built -- the cluster-circularity control is MISSING. "
          "Run: bash jobs/run_target_screen_qc.sh --tissue skin --call mal_tcr --no-cluster "
          f"--out-dir {TCR_DIR}")
```

```
share of the malignant arm's signal held by its single strongest cluster (511 clusters):
  all screened genes: {'count': 2249.0, 'mean': 0.118, 'std': 0.1879, 'min': 0.0027, '50%': 0.0494, '90%': 0.2935, '99%': 1.0, 'max': 1.0}

  the genes in question:
   gene  top_cluster_share  top2_cluster_share  n_clusters
TNFRSF8            0.03746             0.07004         511
 COL6A3             0.0208             0.04038         511
  IL2RA            0.01925              0.0354         511
KIR3DL2             0.0121             0.02419         511
   CCR4            0.01204             0.02314         511
  QSOX1           0.008617              0.0172         511
   CALU           0.008524             0.01665         511
    GRN           0.008363             0.01605         511
 LGALS3           0.006962             0.01362         511
    GPI           0.006792             0.01308         511
  SYPL1           0.005815             0.01127         511
  VDAC1           0.005274             0.01049         511

  A gene spread evenly over N clusters scores 1/N. Confinement would show as a share far above that floor.

by_call_mal_tcr not built -- the cluster-circularity control is MISSING. Run: bash jobs/run_target_screen_qc.sh --tissue skin --call mal_tcr --no-cluster --out-dir /home/projects/nyosef/zvise/MF/data/targets/by_call_mal_tcr
```

## §3 — Malignant vs reactive CD4, on what survived

The comparison the whole screen exists for, now on an independent cohort of MF/SS patients with a
CD4 clone, and on genes that passed §2. Paired, because both arms come from the same biopsy and
the same library prep: the donor is the block, and an unpaired test would spend its power on
between-patient variation that pairing removes. The study-blocked test (van Elteren) sits beside
it for the donors carrying only one arm, never instead of it.

Five figures, in the style of `28_tme_degradome_mc`, because a cohort median hides the thing that
decides whether a target is usable: **what it does inside each patient.**

| figure | what it shows |
|---|---|
| **FA** | the leading antigens, one grey line per donor joining its reactive and malignant means |
| **FB** | the antigens whose answer is already known — read this one first |
| **FC** | the ranked candidate list as a dumbbell, with prevalence and off-tumour beside each row |
| **FD** | the donor × gene effect matrix, donors ordered by malignant burden |
| **FE** | the same comparison on *detection rate*, which is what a CAR is dosed against |

The star above each gene is the BH-adjusted paired Wilcoxon; `n↑/n↓` underneath it is the donor
vote, and the two percentages under each gene name are the median share of cells detected in the
malignant and reactive arms. Titles say what the panel is; the finding is in the table printed
under it.

FD is the figure to argue with. Donors run left to right by malignant share of their CD4
compartment, so a row that darkens towards the right is a row whose effect tracks arm composition
— the pattern §2 tests for formally with `fails_burden`, shown here so it can also be seen. In
skin that ordering is close to a patch → plaque → tumour ordering, so a row that darkens to the
right may also be a real lesion-stage effect; 47 separates the two.

```python  # cell 12
# ============================================================================
# §3  Malignant vs reactive CD4, on the genes that survived §2
# ============================================================================
prev = TS.prevalence(long, MALIG, detect_floor=DETECT_FLOOR)
off = TS.off_tumour(long, target_arms=[MALIG], healthy_t_arms=[HC_CD4], value="pct_expr")
off = off.rename(columns={"worst_offtumour": "worst_T_arm_pct",
                          "worst_offtumour_arm": "worst_T_arm"})

core = (sel.merge(prev, on="gene", how="left")
           .merge(off, on="gene", how="left")
           .merge(ev[["gene", "artifact_flag", "n_fails", "fail_reasons",
                      "retained_trim", "retained_clean", "retained_ds", "rho_burden"]],
                  on="gene", how="left"))
core = core.merge(ST.annotate(core["gene"])[["gene", "cd_name", "panel_class", "stage"]],
                  on="gene", how="left")

GATES = ((core["artifact_flag"].eq("clean")) & (core["q"] < 0.05)
         & (core["log2fc_paired"] > 0.5) & (core["frac_donors_positive"] >= 0.5)
         & (~core["gene"].isin(ST.LOSS_MARKERS)) & (~core["gene"].isin(TQ.BAND_GENES)))
up = core[GATES].sort_values("log2fc_paired", ascending=False)
SHORT_GENES = up["gene"].tolist()[:30]
TOP_GENES = SHORT_GENES[:12]

CAPTION = (f"Paired Wilcoxon within donor, BH. {len(PAIRED_DONORS)} donors, "
           f"{meta['study'].nunique()} studies. log1p CP10K.")
LINE_CAPTION = CAPTION + " Lines: red = up in malignant, blue = down."

print(f"{len(up)} genes pass every gate "
      "(clean, q<0.05, log2FC>0.5, detected in >=50% of donors):\n")
_c = ["gene", "log2fc_paired", "frac_donors_up", "frac_donors_positive", "median_pct_expr",
      "q", "z_blocked", "retained_trim", "retained_clean", "retained_ds", "rho_burden",
      "worst_T_arm", "cd_name", "stage"]
print(up[[c for c in _c if c in up]].to_string(index=False,
                                               float_format=lambda v: f"{v:.4g}"))

print("\nthe loss phenotype -- CD7 and DPP4/CD26 loss is the textbook CTCL immunophenotype, and "
      "is the sanity check that the cleaning did not flatten real biology:")
print(core[core["gene"].isin(ST.LOSS_MARKERS)][["gene", "log2fc_paired", "frac_donors_up",
                                                "q", "artifact_flag"]]
      .sort_values("log2fc_paired").to_string(index=False,
                                              float_format=lambda v: f"{v:.4g}"))

print("\npaired log2FC vs the study-blocked z, over every tested gene -- the two tests should "
      "agree, and where they do not the gene is carried by one study:")
_ok = core.dropna(subset=["z_blocked"])
print(f"  r = {np.corrcoef(_ok['log2fc_paired'], _ok['z_blocked'])[0, 1]:.2f} "
      f"over {len(_ok):,} genes")
```

```
28 genes pass every gate (clean, q<0.05, log2FC>0.5, detected in >=50% of donors):

    gene  log2fc_paired  frac_donors_up  frac_donors_positive  median_pct_expr         q  z_blocked  retained_trim  retained_clean  retained_ds  rho_burden    worst_T_arm cd_name   stage
    CD70          1.102            0.92                0.8077           0.2096 3.089e-10       6.47          1.102          0.8991        0.693     -0.0992 CD4_unassessed    CD70 phase 1
     CD9          1.085          0.7333                0.7179           0.2499 7.092e-07      4.799          1.132          0.9871        0.906     -0.1278 CD4_unassessed                
     CPM          1.023          0.7467                0.5769           0.1371 7.081e-08      4.049          1.106          0.8688       0.7924      -0.081 CD4_unassessed                
  COL6A3         0.8105            0.88                0.5641           0.1126 3.281e-09      5.013          1.026           1.148       0.9338     0.06461       CD4_Treg                
   CD151         0.7937            0.76                0.7949           0.1958  9.39e-08      5.107          1.152           0.943       0.7913     0.07249 CD4_unassessed                
     GRN         0.7777          0.9067                0.8462           0.2108 9.806e-10      5.867          1.079          0.8856       0.6025     0.07087 CD4_unassessed                
    TFRC         0.7313          0.8933                0.9487           0.2956 1.675e-09      4.855          1.069           1.064       0.7064     0.07147 CD4_unassessed                
 SLC39A6          0.707          0.9467                0.9103           0.1826 1.911e-10      6.979          1.145           1.077       0.5994    -0.06825 CD4_unassessed                
  SLC4A2         0.6978          0.9333                 0.641           0.1297 1.911e-10      6.443          1.085             1.1       0.8195  -0.0001138 CD4_unassessed                
TNFRSF25         0.6977          0.7467                0.8718           0.3394 9.137e-05      2.922          1.052          0.8551       0.7695      0.2509 CD4_unassessed                
  SLC1A4         0.6901          0.7467                0.6282           0.1437  5.23e-05      3.937          1.085          0.6596        0.534      -0.179 CD4_unassessed                
  LGALS3         0.6634          0.5867                0.8846           0.4271  0.001147      2.725          1.073          0.8615       0.9168    -0.07437 CD4_unassessed                
   PDIA4         0.6373            0.88                 0.859           0.2552 2.497e-09      3.768           1.11          0.9007        0.595    -0.07624 CD4_unassessed                
    FUT8         0.6353          0.8667                0.7436           0.1844 6.877e-09      4.139          1.102           1.031        0.692    -0.08418 CD4_unassessed                
    CD82          0.632          0.7733                     1           0.4766 1.406e-07      4.194          1.045          0.9715       0.6248    -0.08634 CD4_unassessed                
  SLC5A3         0.6169          0.8933                0.7692           0.2169 1.805e-09      4.376          1.109           1.136       0.7467    -0.04879 CD4_unassessed                
    IL6R         0.6145            0.72                0.7692           0.2334 2.308e-05      2.868           1.12          0.7774       0.9334      0.2893 CD4_unassessed                
   SYPL1         0.5987          0.9067                0.9487           0.3624 1.151e-09      4.319          1.054          0.8961       0.6077      0.1627 CD4_unassessed                
    CALU         0.5869            0.88                0.8462           0.1904 6.179e-10      6.298          1.146           1.005       0.5611      0.1553 CD4_unassessed                
   QSOX1         0.5837          0.8533                0.6923           0.1332 7.512e-09       5.22          1.177           1.073       0.5567     -0.1631 CD4_unassessed      
... [2259 more chars truncated]
```

```python  # cell 13
# ============================================================================
# §3 FA  The leading surviving antigens, one line per donor
# ============================================================================
recA = TF.t_paired(long, TOP_GENES, MALIG, REACT, value="mean_all", min_donors=MIN_DONORS)
recA = recA.merge(ST.annotate(recA["gene"])[["gene", "cd_name", "stage"]], on="gene", how="left")

fig, ax = plt.subplots(figsize=(0.82 * len(recA) + 1.8, 3.6))
TF.paired_slopes(ax, recA)
ax.set_title(f"Top antigens, malignant vs reactive CD4 (n={len(PAIRED_DONORS)})",
             loc="left", fontsize=7.5)
TF.save(fig, "FA_top_candidates_paired", caption_text=LINE_CAPTION)
TF.save_source(recA, "FA_top_candidates_paired_source")
TF.logfig("FA_top_candidates_paired", "leading clean antigens, paired per donor", len(recA))

TF.show_tests(recA, cols=TF.SHOW_COLS + ["cd_name", "stage"])
```

```
wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt42_FA_top_candidates_paired.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt42_FA_top_candidates_paired.pdf

<Figure size 1164x360 with 1 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt42_FA_top_candidates_paired_source.csv
    gene  n_pairs  med_a   med_b   delta  log2fc  det_a   det_b  n_pos  n_neg         p         q direction reason cd_name   stage
    CD70       75 0.2763 0.09084  0.1599   1.102 0.2154 0.05066     69      6 1.181e-12 4.722e-12        up           CD70 phase 1
     CD9       75 0.3729 0.09391  0.1755   1.085 0.2616 0.05634     55     19 4.909e-08 6.545e-08        up                       
     CPM       75 0.1615 0.04217  0.1113   1.023 0.1332 0.02542     56     19 3.127e-09 5.361e-09        up                       
  COL6A3       75 0.1432 0.04006 0.08764  0.8105 0.1111 0.02475     66      9 4.888e-11 9.777e-11        up                       
   CD151       75 0.2793  0.1292  0.1306  0.7937 0.1996 0.08639     57     18 4.306e-09 6.459e-09        up                       
     GRN       75 0.2947  0.1487  0.1349  0.7777 0.2119  0.1014     68      7 6.662e-12 1.999e-11        up                       
    TFRC       75 0.4084  0.2287   0.171  0.7313 0.2939  0.1434     67      8 1.493e-11 3.584e-11        up                       
 SLC39A6       75 0.2259  0.1169  0.1007   0.707  0.187 0.07757     71      4  5.68e-13 3.408e-12        up                       
  SLC4A2       75 0.1481 0.05847 0.07295  0.6978 0.1195 0.03933     70      5 5.256e-13 3.408e-12        up                       
TNFRSF25       75 0.4276   0.267  0.2013  0.6977 0.3276  0.1679     56     19 1.354e-05 1.477e-05        up                       
  SLC1A4       75  0.174 0.09946 0.06671  0.6901 0.1429 0.05813     56     19 7.174e-06 8.608e-06        up                       
  LGALS3       75 0.7799  0.6023   0.288  0.6634 0.4798  0.2855     44     31 0.0002528 0.0002528        up
```

```python  # cell 14
# ============================================================================
# §3 FB  Calibration: the antigens whose answer is already known
# ============================================================================
# Read this panel before FA. CCR4 (mogamulizumab, approved in CTCL), TNFRSF8/CD30 (brentuximab,
# approved), KIR3DL2 (lacutamab) and IL2RA are the positive controls; CD7 and DPP4/CD26 loss IS
# the CTCL immunophenotype, so they must come out strongly negative. The v1 skin screen failed
# exactly here -- CCR4 386th, IL2RA negative -- which is why this panel decides whether anything
# in FA is evidence. All four controls are testable in skin: the object is not an HVG subset.
CTRL_GENES = [g for g in (ST.POSITIVE_CONTROLS + ST.LOSS_MARKERS)
              if g in set(long["gene"])]
recB = TF.t_paired(long, CTRL_GENES, MALIG, REACT, value="mean_all", min_donors=MIN_DONORS)
recB = recB.merge(ev[["gene", "artifact_flag"]], on="gene", how="left")

fig, ax = plt.subplots(figsize=(1.0 * len(recB) + 1.6, 3.4))
TF.paired_slopes(ax, recB)
ax.set_title("Validated antigens and CTCL loss markers", loc="left", fontsize=7.5)
TF.save(fig, "FB_controls_paired", caption_text=LINE_CAPTION)
TF.save_source(recB, "FB_controls_paired_source")
TF.logfig("FB_controls_paired", "positive controls and loss markers, paired", len(recB))

TF.show_tests(recB, cols=TF.SHOW_COLS + ["artifact_flag"])
_missing = sorted(set(ST.POSITIVE_CONTROLS + ST.LOSS_MARKERS) - set(long["gene"]))
if _missing:
    print(f"\nnot testable, absent from the screened gene set: {_missing}")
print("\nCD26 is an alias of DPP4, which is present under that symbol. A positive control that "
      "sits in the middle of the ranking is a calibration failure and the arms are the first "
      "suspect -- not a discovery about that drug.")
```

```
wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt42_FB_controls_paired.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt42_FB_controls_paired.pdf

<Figure size 1160x340 with 1 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt42_FB_controls_paired_source.csv
   gene  n_pairs   med_a    med_b    delta   log2fc    det_a   det_b  n_pos  n_neg         p         q direction reason artifact_flag
   CCR4       75  0.2717   0.2369  0.05246    0.324   0.2125   0.134     42     33   0.04205   0.07009        ns                clean
TNFRSF8       75 0.04049  0.01636  0.01709   0.3318   0.0374 0.01089     50     25 3.555e-05 0.0001777        up                clean
KIR3DL2       75       0 0.006845        0        0        0 0.00431     27     16 0.0001999 0.0006665      down                clean
  IL2RA       75  0.1886   0.2393 -0.02839  -0.1718   0.1237  0.1219     29     46    0.3286    0.4108        ns                clean
    CD7       75 0.07043    1.148  -0.9057   -2.848  0.03902  0.5353      7     68 6.045e-11 6.045e-10      down                clean
   DPP4       75 0.00937  0.07732 -0.05162  -0.8411 0.008855 0.04444     19     55  0.001045  0.002612      down                clean
    CD5       75  0.4936   0.4731 -0.01209 -0.05845   0.3194  0.2546     37     38    0.7117    0.7117        ns                clean
    CD6       75  0.7038   0.6408  0.05031  0.09565   0.3947  0.3449     42     33    0.2205    0.3151        ns                clean
   CD27       75  0.4457   0.5007 -0.07352  -0.3589   0.2691  0.2786     35     40    0.6882    0.7117        ns                clean
    CD2       75    1.01    1.168  -0.1312   -0.146   0.6017  0.5818     34     41   0.01987   0.03975      down                clean

not testable, absent from the screened gene set: ['CD26']

CD26 is an alias of DPP4, which is present under that symbol. A positive control that sits in the middle of the ranking is a calibration failure and the arms are the first suspect -- not a discovery about that drug.
```

```python  # cell 15
# ============================================================================
# §3 FC  The surviving candidates as one ranked panel
# ============================================================================
recC = TF.t_paired(long, SHORT_GENES, MALIG, REACT, value="mean_all", min_donors=MIN_DONORS)
recC = recC.merge(ST.annotate(recC["gene"])[["gene", "cd_name", "stage"]], on="gene", how="left")
recC = recC.merge(core[["gene", "frac_donors_positive", "median_pct_expr", "worst_T_arm",
                        "worst_T_arm_pct"]], on="gene", how="left")
recC = recC.merge(prev[["gene", "n_donors"]].rename(columns={"n_donors": "n_donors_malig"}),
                  on="gene", how="left")
recC["ylabel"] = [f"{TF.it(r['gene'])}" + (f" · {r['cd_name']}" if r["cd_name"] else "")
                  for _, r in recC.iterrows()]
recC = recC.sort_values("log2fc", ascending=False)

fig, _ = TF.shortlist_panel(recC, floor=DETECT_FLOOR,
                            title="Shortlist: effect, patients positive, off-tumour burden")
TF.save(fig, "FC_shortlist_dumbbell", caption_text=CAPTION)
TF.save_source(recC, "FC_shortlist_dumbbell_source")
TF.logfig("FC_shortlist_dumbbell", "ranked clean candidates, both arms", len(recC))

TF.show_tests(recC, cols=TF.SHOW_COLS + ["cd_name", "stage", "frac_donors_positive",
                                         "median_pct_expr", "worst_T_arm", "worst_T_arm_pct"])
```

```
wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt42_FC_shortlist_dumbbell.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt42_FC_shortlist_dumbbell.pdf

<Figure size 900x822 with 3 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt42_FC_shortlist_dumbbell_source.csv
    gene  n_pairs  med_a   med_b   delta  log2fc  det_a   det_b  n_pos  n_neg         p         q direction reason cd_name   stage  frac_donors_positive  median_pct_expr    worst_T_arm  worst_T_arm_pct
    CD70       75 0.2763 0.09084  0.1599   1.102 0.2154 0.05066     69      6 1.181e-12 8.264e-12        up           CD70 phase 1                0.8077           0.2096 CD4_unassessed           0.3314
     CD9       75 0.3729 0.09391  0.1755   1.085 0.2616 0.05634     55     19 4.909e-08 6.248e-08        up                                       0.7179           0.2499 CD4_unassessed           0.4547
     CPM       75 0.1615 0.04217  0.1113   1.023 0.1332 0.02542     56     19 3.127e-09  5.15e-09        up                                       0.5769           0.1371 CD4_unassessed           0.2716
  COL6A3       75 0.1432 0.04006 0.08764  0.8105 0.1111 0.02475     66      9 4.888e-11 1.141e-10        up                                       0.5641           0.1126       CD4_Treg          0.03846
   CD151       75 0.2793  0.1292  0.1306  0.7937 0.1996 0.08639     57     18 4.306e-09 6.699e-09        up                                       0.7949           0.1958 CD4_unassessed           0.3198
     GRN       75 0.2947  0.1487  0.1349  0.7777 0.2119  0.1014     68      7 6.662e-12 2.665e-11        up                                       0.8462           0.2108 CD4_unassessed           0.3679
    TFRC       75 0.4084  0.2287   0.171  0.7313 0.2939  0.1434     67      8 1.493e-11 4.646e-11        up                                       0.9487           0.2956 CD4_unassessed            0.537
 SLC39A6       75 0.2259  0.1169  0.1007   0.707  0.187 0.07757     71      4  5.68e-13 5.301e-12        up                                       0.9103           0.1826 CD4_unassessed           0.4904
  SLC4A2       75 0.1481 0.05847 0.07295  0.6978 0.1195 0.03933     70      5 5.256e-13 5.301e-12        up                                        0.641           0.1297 CD4_unassessed            0.402
TNFRSF25       75 0.4276   0.267  0.2013  0.6977 0.3276  0.1679     56     19 1.354e-05 1.458e-05        up                                       0.8718           0.3394 CD4_unassessed           0.4967
  SLC1A4       75  0.174 0.09946 0.06671  0.6901 0.1429 0.05813     56     19 7.174e-06 8.034e-06        up                                       0.6282           0.1437 CD4_unassessed           0.4136
  LGALS3       75 0.7799  0.6023   0.288  0.6634 0.4798  0.2855     44     31 0.0002528 0.0002621        up                                       0.8846           0.4271 CD4_unassessed           0.6354
   PDIA4       75 0.3454  0.2024  0.1458  0.6373 0.2611   0.141     66      9 3.075e-11 7.827e-11        up                                        0.859           0.2552 CD4_unassessed           0.4385
    FUT8       75  0.237  0.1377 0.09629  0.6353 0.1833 0.08451     65     10 1.665e-10 3.329e-10        up                                       0.7436           0.1844 CD4_unassessed           0.4153
    CD82       75  0.805  0.5138  0.2903   0.632 0.4952  0.2767     58     17 6.923e-09  1.02e-08        up                                            1           0.4766 CD4_unassessed           0.5743
  SLC5A3       75 0.2796  0.1461  0.1203  0.6169 0.2093 0.08759     67      8  2.07e-11 5.795e-11        up                                       0.7692           0.2169 CD4_unassessed           0.5129
    IL6R       75  0.277   0.154  0.1231  0.6145 0.2341  0.1045     54     21 2.744e-06 3.201e-06        up                                       0.7692           0.2334 CD4_unassessed           0.3725
   SYPL1       75 0.5011  0.3221  0.1751  0.5987    0.4  0.1981     68      7 
... [2143 more chars truncated]
```

```python  # cell 16
# ============================================================================
# §3 FD  Leading markers inside each sample: the donor x gene effect matrix
# ============================================================================
M = TH.lfc_matrix(long, MALIG, REACT, genes=TOP_GENES, min_donors=MIN_DONORS)
M = M.loc[[g for g in TOP_GENES if g in M.index]]
# donors ordered by their own malignant burden, so the eye can check the one thing §2 warned
# about: an effect that grows left to right is an effect that grows with arm composition.
_b = bur["burden"].reindex(M.columns).dropna().sort_values()
M = M[list(_b.index)]

fig, ax = plt.subplots(figsize=(0.13 * M.shape[1] + 3.2, 0.19 * len(M) + 1.8))
TF.donor_gene_heat(ax, M, stars_by=dict(zip(core["gene"], core["q"])))
ax.set_xlabel(f"donor, ordered by malignant share of CD4 "
              f"({_b.iloc[0]:.0%} → {_b.iloc[-1]:.0%})")
ax.set_title("Per-donor log2FC, malignant vs reactive", loc="left", fontsize=7.5)
TF.save(fig, "FD_donor_gene_heat", caption_text=CAPTION)
TF.save_source(M.reset_index(), "FD_donor_gene_heat_source")
TF.logfig("FD_donor_gene_heat", "per-donor log2FC matrix, donors by burden", M.size)

print("per-donor log2FC, complete matrix (genes x donors):")
print(M.round(3).to_string())
print("\nper-gene summary of that matrix:")
_sum = pd.DataFrame({
    "n_donors": M.notna().sum(1),
    "median": M.median(1), "iqr": M.quantile(.75, axis=1) - M.quantile(.25, axis=1),
    "min": M.min(1), "max": M.max(1),
    "frac_up": (M > 0).sum(1) / M.notna().sum(1),
    "frac_up_1": (M > 1).sum(1) / M.notna().sum(1),
}).sort_values("median", ascending=False)
print(_sum.to_string(float_format=lambda v: f"{v:.3g}"))
print("\n`frac_up_1` -- donors where the antigen is at least doubled -- is the column that "
      "separates a cohort-wide antigen from one that is strong in a subset. 47 takes that "
      "apart formally.")
```

```
wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt42_FD_donor_gene_heat.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt42_FD_donor_gene_heat.pdf

<Figure size 1295x408 with 2 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt42_FD_donor_gene_heat_source.csv
per-donor log2FC, complete matrix (genes x donors):
donor     Li2024_atlas__MF17  Li2024_atlas__CTCL7  Li2024_atlas__MF27  Li2024_atlas__CTCL4  D15__MF28  D10__JONAK1  D13__MF21  D3__P105  D15__MF36  D8__P90  D15__MF24  D11__P4  D15__MF34  Li2024_atlas__CTCL6  Li2024_atlas__CTCL14  D7__P73  D7__P84  H__SS1  Li2024_atlas__PT56  Li2024_atlas__MF22  D13__MF25  D1__P204  Li2024_atlas__CTCL2  Li2024_atlas__PT50  D13__MF18  D6__MF2  D8__P65  D8__P138  D13__MF19  Li2024_atlas__MF30  D1__P312  D6__SC157  D15__MF30  H__MFIV1  Li2024_atlas__PT52  Li2024_atlas__CTCL5  Li2024_atlas__PT47  Li2024_atlas__PT53  B5__PT09  Li2024_atlas__CTCL13  D8__P163  Li2024_atlas__CTCL17  Li2024_atlas__MF21  D8__P150  D7__MF311  Li2024_atlas__CTCL8  D7__MF309  D6__SC158  D11__P6  Li2024_atlas__CTCL16  D15__MF35  D3__P196  D3__P192  H__SS4  D1__PGS  D11__P3  D11__P5  Li2024_atlas__CTCL3  B5__PT23  D15__MF48  Li2024_atlas__MF28  Li2024_atlas__PT11  Li2024_atlas__CTCL10  D11__P1  Li2024_atlas__CTCL12  D11__P2  D1__P303  Li2024_atlas__PT35  D15__MF26  D1__P171  Li2024_atlas__CTCL9  Li2024_atlas__CTCL1  Li2024_atlas__PT55  Li2024_atlas__CTCL18  D13__MF17
gene                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
CD70                   0.958                1.659               3.227                2.637      2.410        0.841      1.157     0.485      0.105    1.177      1.786    1.092      1.180                1.184                 1.447    0.511    0.110   1.102               0.820               2.759      0.372     0.831                3.329               0.504     -0.208    0.046    2.952     3.114      1.254               2.321     2.644      0.342      0.459     1.908              -0.849                0.887               1.394              -0.357     1.916                 1.222     1.935                 0.098               1.093     0.092      0.412               -0.593      2.517     -0.124    0.812                 1.528      1.011     2.393    -0.002   0.892    1.056    1.492    0.618                0.230     2.075      1.138               0.780               0.057                 1.185    0.303                 1.405    2.365     0.105               2.447      1.239     1.986                2.146                0.390               0.211                 1.144      1.116
CD9                    0.000                4.834               0.864                3.298      2.733        1.343     -0.854    -0.163      2.802    1.576      2.187    1.085      1.663                5.308                 0.506    2.396   -0.159  -2.326               0.297               1.518     -1.420     1.203                3.511               1.339      2.820   -1.898    1.311     2.397     -0.487     
... [12659 more chars truncated]
```

```python  # cell 17
# ============================================================================
# §3 FE  Detection, not expression: the axis a CAR is actually dosed against
# ============================================================================
recE = TF.t_paired(long, TOP_GENES, MALIG, REACT, value="pct_expr", min_donors=MIN_DONORS)
fig, ax = plt.subplots(figsize=(10.2, 3.4))
TF.paired_slopes(ax, recE, ylabel="fraction of a donor's cells detected", show_detection=False)
ax.set_title("Fraction of CD4 cells detected, per donor", loc="left", fontsize=7.5)
TF.save(fig, "FE_detection_paired", caption_text=LINE_CAPTION)
TF.save_source(recE, "FE_detection_paired_source")
TF.logfig("FE_detection_paired", "paired detection rate per donor", len(recE))

print("Detection is the quantity that decides whether a CAR sees the tumour; expression is the "
      "quantity the ranking is sorted on. They are not the same ordering, and where they "
      "disagree the detection column wins for a cell-therapy target.\n")
TF.show_tests(recE)

print("\n" + "=" * 78)
print("eligibility: in what fraction of patients would at least X% of the tumour be positive")
print("=" * 78)
elig = TH.eligibility(long, MALIG, genes=TOP_GENES, donors=PAIRED_DONORS)
print(elig.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
elig.to_csv(ST.tab("eligibility_skin"), index=False)
print("\nwrote", ST.tab("eligibility_skin"))
```

```
wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt42_FE_detection_paired.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt42_FE_detection_paired.pdf

<Figure size 1020x340 with 1 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt42_FE_detection_paired_source.csv
Detection is the quantity that decides whether a CAR sees the tumour; expression is the quantity the ranking is sorted on. They are not the same ordering, and where they disagree the detection column wins for a cell-therapy target.

    gene  n_pairs  med_a   med_b   delta  log2fc  det_a   det_b  n_pos  n_neg         p         q direction reason
    CD70       75 0.2154 0.05066   0.149   1.223 0.2154 0.05066     72      3 8.559e-14 5.312e-13        up       
     CD9       75 0.2616 0.05634  0.1316   1.121 0.2616 0.05634     59     15 1.861e-09 2.482e-09        up       
     CPM       75 0.1332 0.02542 0.09239  0.9543 0.1332 0.02542     60     15 6.482e-10 9.722e-10        up       
  COL6A3       75 0.1111 0.02475 0.07437  0.8427 0.1111 0.02475     68      7  4.77e-12 9.541e-12        up       
   CD151       75 0.1996 0.08639  0.1237  0.9458 0.1996 0.08639     66      9 6.662e-12 1.142e-11        up       
     GRN       75 0.2119  0.1014  0.1181  0.8348 0.2119  0.1014     71      4 7.738e-13 1.857e-12        up       
    TFRC       75 0.2939  0.1434   0.149  0.8064 0.2939  0.1434     69      6 2.222e-13 6.667e-13        up       
 SLC39A6       75  0.187 0.07757  0.1092  0.8974  0.187 0.07757     72      3 9.273e-14 5.312e-13        up       
  SLC4A2       75 0.1195 0.03933 0.07902  0.7388 0.1195 0.03933     74      1 1.328e-13 5.312e-13        up       
TNFRSF25       75 0.3276  0.1679  0.1662  0.8206 0.3276  0.1679     60     15 2.947e-08 3.214e-08        up       
  SLC1A4       75 0.1429 0.05813 0.07512  0.6824 0.1429 0.05813     59     16 2.691e-08 3.214e-08        up       
  LGALS3       75 0.4798  0.2855  0.1799  0.6314 0.4798  0.2855     50     25 9.639e-06 9.639e-06        up       

==============================================================================
eligibility: in what fraction of patients would at least X% of the tumour be positive
==============================================================================
    gene  n_donors  median_pct_expr  frac_donors_ge10  frac_donors_ge25  frac_donors_ge50  frac_donors_ge75
    TFRC        75            0.294              0.96               0.6               0.2              0.04
 SLC39A6        75            0.187              0.92             0.293            0.0667                 0
TNFRSF25        75            0.328              0.88             0.613              0.28              0.04
  LGALS3        75             0.48              0.88             0.773              0.48             0.253
     GRN        75            0.212             0.867             0.453             0.107                 0
   CD151        75              0.2             0.827             0.427             0.107            0.0133
    CD70        75            0.215             0.813             0.427            0.0533                 0
     CD9        75            0.262             0.733             0.507             0.227            0.0267
  SLC4A2        75            0.119              0.64               0.2            0.0267                 0
  SLC1A4        75            0.143             0.627             0.307            0.0533            0.0133
     CPM        75            0.133             0.573              0.24            0.0533            0.0133
  COL6A3        75            0.111              0.56             0.173             0.107            0.0267

wrote /home/projects/nyosef/zvise/MF/tables/tgt42_eligibility_skin.csv
```

## §4 — On-target off-tumour in skin

This is the axis where skin is **stronger than blood**, and it is worth saying why: 43 can only
report the T and NK arms, because the non-T blood compartments were never put through the screen
builder, so a myeloid or HSPC hit is invisible there. Here the whole tissue is available.

Three costs, never averaged:

1. **tissue toxicity** — the worst non-T compartment in MF skin (keratinocyte, fibroblast,
   endothelium, myeloid, mast …), from the degradome pseudobulk cube: sample-level summed counts
   over all 42,347 genes, which is enough to answer "is this also on keratinocytes".
2. **healthy skin** — Reynolds 2021, a different lab, chemistry, site and sex. It cannot be
   blocked at all, so it is used to flag a candidate, never to rank one.
3. **T-cell aplasia** — expression on healthy CD4 and on the reactive arm. For CTCL this is often
   accepted: CD7, CD5 and CD4 agents all cause it. It is a cost to state, not a filter to hide.

A skin-directed agent and a systemic one do not pay these in the same currency: a keratinocyte
hit is a blistering risk for an intralesional agent and a non-issue for an antibody that never
reaches the epidermis at dose. The columns stay separate for that reason.

```python  # cell 19
# ============================================================================
# §4a  Off-tumour: the other T arms, then the non-T skin compartments
# ============================================================================
print("worst non-target T arm, for the surviving candidates:")
_o = (core[core["gene"].isin(SHORT_GENES)]
      [["gene", "cd_name", "worst_T_arm", "worst_T_arm_pct", "healthy_t_burden",
        "median_pct_expr", "frac_donors_positive"]]
      .sort_values("worst_T_arm_pct", ascending=False))
print(_o.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
print("\n`healthy_t_burden` is the aplasia cost: the share of HEALTHY CD4 cells that carry the "
      f"antigen, over {meta.loc[meta['arm'].eq(HC_CD4), 'donor'].nunique()} healthy-skin donors. "
      f"`worst_T_arm` ranges over {sorted(set(meta['arm']) - {MALIG})}.")

# non-T skin compartments, from the degradome cube (CP10K means, sample level)
cube = pd.read_parquet(dcfg.NB_DIR / "data" / "degradome" /
                       "skin_degradome_pseudobulk_full.parquet")
cmeta = pd.read_parquet(dcfg.NB_DIR / "data" / "degradome" /
                        "skin_degradome_pseudobulk_meta.parquet")
genes = [g for g in prev["gene"] if g in cube.columns]
lib = cube.sum(axis=1).to_numpy()[:, None]
cp10k = np.log1p(cube[genes].to_numpy() / np.where(lib == 0, 1, lib) * 1e4)
nonT = cmeta["degradome_level"].astype(str)
is_nonT = ~nonT.str.startswith(("CD4", "CD8", "Treg", "NK", "ILC", "Prolif"))
print(f"\nnon-T levels in the cube: {sorted(nonT[is_nonT].unique())}")

cells_ok = cmeta["n_cells"].to_numpy() >= MIN_CELLS
nt = pd.DataFrame(cp10k[is_nonT.to_numpy() & cells_ok], columns=genes)
nt["level"] = nonT[is_nonT.to_numpy() & cells_ok].to_numpy()
_med = nt.groupby("level").median(numeric_only=True)
worst_nonT = (_med.max().rename("worst_nonT_skin").reset_index()
              .rename(columns={"index": "gene"}))
worst_nonT_lvl = (_med.idxmax().rename("worst_nonT_level").reset_index()
                  .rename(columns={"index": "gene"}))
off = off.merge(worst_nonT, on="gene", how="left").merge(worst_nonT_lvl, on="gene", how="left")

print("\nthe surviving candidates against the worst non-T skin compartment:")
_r = (off[off["gene"].isin(SHORT_GENES)]
      .merge(prev[["gene", "median_mean_all"]], on="gene", how="left")
      .assign(nonT_over_tumour=lambda d: d["worst_nonT_skin"] - d["median_mean_all"])
      .sort_values("nonT_over_tumour", ascending=False))
print(_r[["gene", "worst_nonT_skin", "worst_nonT_level", "median_mean_all",
          "nonT_over_tumour"]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
print("\n`nonT_over_tumour` > 0 means the gene is higher on some non-T skin cell than on the "
      "tumour arm itself. That is a tissue-toxicity flag for a skin-directed agent and a "
      "contamination flag for the screen -- the two readings are not exclusive, which is why "
      "§2 tests the second one per cell rather than inferring it from this column.")

print("\nthe worst offenders over ALL screened genes, for scale:")
_all = (off.merge(prev[["gene", "median_mean_all"]], on="gene", how="left")
        .sort_values("worst_nonT_skin", ascending=False).head(15))
print(_all[["gene", "worst_nonT_skin", "worst_nonT_level", "median_mean_all"]]
      .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
```

```
worst non-target T arm, for the surviving candidates:
    gene cd_name    worst_T_arm  worst_T_arm_pct  healthy_t_burden  median_pct_expr  frac_donors_positive
     GPI         CD4_unassessed            0.813             0.125            0.474                 0.974
   SURF4         CD4_unassessed            0.749             0.185            0.432                 0.974
   VDAC1         CD4_unassessed            0.708             0.175            0.482                 0.987
  LGALS3         CD4_unassessed            0.635             0.361            0.427                 0.885
   SYPL1         CD4_unassessed             0.62             0.179            0.362                 0.949
    CD82         CD4_unassessed            0.574             0.176            0.477                     1
 TNFRSF4               CD4_Treg            0.556             0.137            0.349                 0.897
    TFRC         CD4_unassessed            0.537             0.118            0.296                 0.949
  SLC5A3         CD4_unassessed            0.513            0.0499            0.217                 0.769
TNFRSF25         CD4_unassessed            0.497            0.0943            0.339                 0.872
 SLC39A6         CD4_unassessed             0.49            0.0447            0.183                  0.91
    CALU         CD4_unassessed            0.456            0.0392             0.19                 0.846
   AP2S1         CD4_unassessed            0.456            0.0957            0.358                  0.91
     CD9         CD4_unassessed            0.455             0.107             0.25                 0.718
   PDIA4         CD4_unassessed            0.439             0.082            0.255                 0.859
 TMEM30A         CD4_unassessed            0.421            0.0612            0.188                 0.846
    FUT8         CD4_unassessed            0.415            0.0373            0.184                 0.744
  SLC1A4         CD4_unassessed            0.414            0.0239            0.144                 0.628
  SLC4A2         CD4_unassessed            0.402            0.0156             0.13                 0.641
    IL6R         CD4_unassessed            0.373            0.0726            0.233                 0.769
     GRN         CD4_unassessed            0.368             0.041            0.211                 0.846
   QSOX1         CD4_unassessed            0.366            0.0261            0.133                 0.692
    CD70    CD70 CD4_unassessed            0.331             0.024             0.21                 0.808
   CD151         CD4_unassessed             0.32            0.0612            0.196                 0.795
    STX6         CD4_unassessed            0.314            0.0261            0.145                 0.679
     CPM         CD4_unassessed            0.272           0.00669            0.137                 0.577
  TSPAN2           CD4_reactive           0.0534            0.0377            0.135                 0.654
  COL6A3               CD4_Treg           0.0385            0.0158            0.113                 0.564

`healthy_t_burden` is the aplasia cost: the share of HEALTHY CD4 cells that carry the antigen, over 21 healthy-skin donors. `worst_T_arm` ranges over ['CD4_HC', 'CD4_Treg', 'CD4_reactive', 'CD4_unassessed', 'CD8'].


non-T levels in the cube: ['B', 'DC_LAMP3', 'F_apCAF', 'F_inflammatory', 'F_mesenchymal', 'F_myofibroblast', 'F_papillary', 'F_reticular', 'Keratinocyte', 'LC', 'Mac_FOLR2', 'Mac_SPP1_TREM2', 'Mac_infl', 'Mast', 'Melanocyte', 'Mono_CD14', 'Mono_CD16', 'Neutrophils', 'Plasma', 'Vascular', 'cDC1', 'cDC2', 'moDC', 'pDC']

the surviving candidates against the worst non-T skin compartment:
    gene  worst_nonT_skin worst_nonT_level  median_mean_all  nonT_over_tumour
  COL6A3             3.03      F_papillary            0.146              2.89
  LGALS3             3.24             Mast            0.747              2.49
     CD9              2.7      F
... [3341 more chars truncated]
```

```python  # cell 20
# ============================================================================
# §4b  Healthy skin (Reynolds 2021) -- different lab, chemistry, site and sex
# ============================================================================
# The external arm cannot be blocked at all, so a shift in a gene with no biological reason to
# differ is the platform floor, not a result. It is used to flag candidates that are high on
# healthy skin cells, never to rank them.
try:
    ext = X.load_external()
    ext_genes = [g for g in prev["gene"] if g in set(map(str, ext.var_names))]
    lvl = ext.obs[X.TARGET_COL].astype(str)
    Xe = ext[:, ext_genes].X
    Xe = np.asarray(Xe.todense()) if hasattr(Xe, "todense") else np.asarray(Xe)
    ext_med = (pd.DataFrame(Xe, columns=ext_genes).assign(level=lvl.to_numpy())
               .groupby("level").median(numeric_only=True))
    ext_worst = ext_med.max().rename("worst_healthy_skin").reset_index().rename(
        columns={"index": "gene"})
    ext_worst["worst_healthy_skin_level"] = ext_med.idxmax().to_numpy()
    off = off.merge(ext_worst, on="gene", how="left")
    print(f"healthy-skin arm: {ext.n_obs:,} cells, {lvl.nunique()} levels, "
          f"{len(ext_genes)} of the screened genes present")
    print("\nthe surviving candidates on healthy skin:")
    print(off[off["gene"].isin(SHORT_GENES)]
          [["gene", "worst_healthy_skin", "worst_healthy_skin_level", "worst_nonT_skin",
            "worst_nonT_level"]]
          .sort_values("worst_healthy_skin", ascending=False)
          .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("\nA candidate high on healthy skin is a candidate whose on-target toxicity does not "
          "need the disease to be present. Read it as a flag against the lesion-restricted "
          "story, not as a measured toxicity.")
except Exception as e:                      # the reference is optional, the screen is not
    print("healthy-skin arm unavailable:", e)
    print("-> off-tumour is atlas-internal only; say so wherever the table is quoted.")
```

```
healthy-skin arm: 188,698 cells, 15 levels, 274 of the screened genes present

the surviving candidates on healthy skin:
    gene  worst_healthy_skin worst_healthy_skin_level  worst_nonT_skin worst_nonT_level
  LGALS3                3.69                     Mast             3.24             Mast
 TNFRSF4                1.17                     moDC            0.639              pDC
  COL6A3               0.902               Fibroblast             3.03      F_papillary
    CD70                   0                      CD4             0.77                B
    IL6R                   0                      CD4             1.12      Neutrophils
   AP2S1                 NaN                      NaN             1.68        Mono_CD14
    CALU                 NaN                      NaN             1.48    F_mesenchymal
   CD151                 NaN                      NaN             1.74  F_myofibroblast
    CD82                 NaN                      NaN             1.58             Mast
     CD9                 NaN                      NaN              2.7      F_papillary
     CPM                 NaN                      NaN             1.15             Mast
    FUT8                 NaN                      NaN             0.38                B
     GPI                 NaN                      NaN            0.977         Mac_infl
     GRN                 NaN                      NaN             2.59   Mac_SPP1_TREM2
   PDIA4                 NaN                      NaN            0.878           Plasma
   QSOX1                 NaN                      NaN            0.862      Neutrophils
  SLC1A4                 NaN                      NaN             1.23       Melanocyte
 SLC39A6                 NaN                      NaN                1       Melanocyte
  SLC4A2                 NaN                      NaN            0.371       Melanocyte
  SLC5A3                 NaN                      NaN            0.843    F_mesenchymal
    STX6                 NaN                      NaN            0.402      Neutrophils
   SURF4                 NaN                      NaN            0.836    F_mesenchymal
   SYPL1                 NaN                      NaN             1.29                B
    TFRC                 NaN                      NaN             1.09   Mac_SPP1_TREM2
 TMEM30A                 NaN                      NaN            0.625      F_papillary
TNFRSF25                 NaN                      NaN            0.221      Neutrophils
  TSPAN2                 NaN                      NaN            0.259         Vascular
   VDAC1                 NaN                      NaN              1.2       Melanocyte

A candidate high on healthy skin is a candidate whose on-target toxicity does not need the disease to be present. Read it as a flag against the lesion-restricted story, not as a measured toxicity.
```

## §5–§6 — Surface plausibility and escape risk

**§5, trafficking.** A transcript is the first step of a long pipeline: translocation, folding,
glycosylation, export, endocytosis, recycling, shedding. Two things are reported, both context
rather than correction — the **module scores per arm** (is the malignant arm's secretory and
recycling machinery in a different state from the reactive arm, which is when a
transcript-to-surface inference is least safe) and the **per-gene caveat** beside each candidate.
`SELL` is the worked example: its transcript says nothing without `ADAM17` beside it.

This is also where the §2 activity axis reappears as biology rather than as an artifact. If every
trafficking module is up in the malignant arm, that is the same "larger, busier cell" that
inflates the housekeeping ranking — and it means a transcript-to-surface inference is *least*
reliable exactly for the genes the screen ranks highest. 45 asks the sharper version of this
question against measured CITE-seq protein, which exists in blood only; **skin has no protein
layer at all**, so for skin this section is the end of the argument rather than a step in it.

**§6, escape risk.** An antigen whose detection rate swings between the per-sample clusters of
**one** patient is the one that selects an antigen-negative clone under therapy — the failure
mode of every single-antigen CAR in a heterogeneous tumour. The clusters are the same ones the
malignancy call was made on.

```python  # cell 22
# ============================================================================
# §5  Trafficking machinery, per arm
# ============================================================================
tr_genes = [g for g in ST.TRAFFIC_GENES if g in set(long["gene"])]
tr = long[long["gene"].isin(tr_genes)].copy()
tr["module"] = tr["gene"].map(lambda g: ST.GENE_TO_TRAFFIC.get(g, [""])[0])

# z across donors within a module, so a module is compared to itself, not to other modules
piv = tr.groupby(["arm", "donor", "module"])["mean_all"].mean().reset_index()
piv["z"] = piv.groupby("module")["mean_all"].transform(lambda s: (s - s.mean()) / (s.std() or 1))
arm_mod = piv.pivot_table(index="module", columns="arm", values="z")
keep = [c for c in (MALIG, REACT, HC_CD4) if c in arm_mod.columns]
arm_mod = arm_mod[keep]
print("trafficking module z by arm (mean over donors) -- the capacity to put a protein on the "
      "surface and keep it there:")
print(arm_mod.round(2).to_string())

if MALIG in arm_mod and REACT in arm_mod:
    delta = (arm_mod[MALIG] - arm_mod[REACT]).sort_values()
    print("\nmalignant - reactive, by module:")
    print(delta.round(2).to_string())
    print(f"\n{int((delta > 0).sum())} of {len(delta)} modules are up in the malignant arm.")
    print("A large positive delta on `shedding` or `endocytosis` means the arm that looks "
          "antigen-high by transcript may be the arm that keeps least of it on the surface. "
          "Every module up at once is the biosynthetic-activity axis of §2, and it is the "
          "reason a transcript ranking in this compartment should be read with the "
          "`retained_ds` column in hand.")

TRAFFIC_DELTA = (arm_mod[MALIG] - arm_mod[REACT]).to_dict() if (
    MALIG in arm_mod and REACT in arm_mod) else {}
```

```
trafficking module z by arm (mean over donors) -- the capacity to put a protein on the surface and keep it there:
arm                     CD4_malignant  CD4_reactive  CD4_HC
module                                                     
endocytosis                      0.34         -0.05   -1.43
er_folding_qc                    0.42         -0.10   -1.11
er_to_golgi                      0.45         -0.13   -1.26
er_translocation                 0.34         -0.06   -1.17
escrt_exosome                    0.49         -0.17   -1.39
glycosylation                    0.25          0.04   -0.93
recycling                        0.32         -0.04   -1.34
retrograde                       0.41         -0.05   -1.31
shedding                         0.42          0.10   -0.63
snare_fusion                     0.25         -0.11   -1.28
ubiquitin_proteostasis           0.16         -0.02   -1.19

malignant - reactive, by module:
module
ubiquitin_proteostasis    0.18
glycosylation             0.21
shedding                  0.33
recycling                 0.36
snare_fusion              0.36
endocytosis               0.39
er_translocation          0.40
retrograde                0.46
er_folding_qc             0.52
er_to_golgi               0.58
escrt_exosome             0.66

11 of 11 modules are up in the malignant arm.
A large positive delta on `shedding` or `endocytosis` means the arm that looks antigen-high by transcript may be the arm that keeps least of it on the surface. Every module up at once is the biosynthetic-activity axis of §2, and it is the reason a transcript ranking in this compartment should be read with the `retained_ds` column in hand.
```

```python  # cell 23
# ============================================================================
# §5b  Per-gene tier + the machinery each candidate must be read against
# ============================================================================
tier = pd.DataFrame({"gene": prev["gene"]})
tier["surface_tier"] = [
    "caveated" if g in ST.SURFACE_CAVEAT else "unknown" for g in tier["gene"]]
tier["surface_caveat"] = tier["gene"].map(ST.SURFACE_CAVEAT).fillna("")

# SELL is the worked example: its transcript says nothing without ADAM17 beside it.
pair_rows = []
for g, partner in [("SELL", "ADAM17"), ("TNFRSF8", "ADAM17"), ("IL2RA", "ADAM17"),
                   ("CTLA4", "AP2M1"), ("PDCD1", "FBXO38"), ("CCR4", "ARRB2")]:
    if g in set(long["gene"]) and partner in set(long["gene"]):
        a = long[long["gene"].eq(g)].groupby("arm")["mean_all"].median()
        b = long[long["gene"].eq(partner)].groupby("arm")["mean_all"].median()
        pair_rows.append({"gene": g, "partner": partner,
                          "gene_malig": a.get(MALIG, np.nan), "gene_react": a.get(REACT, np.nan),
                          "partner_malig": b.get(MALIG, np.nan),
                          "partner_react": b.get(REACT, np.nan)})
pairs = pd.DataFrame(pair_rows)
print("candidate beside the machine that removes it from the surface:")
print(pairs.round(3).to_string(index=False))
print("\nRead each row as a pair. A higher transcript with a higher sink is not a higher "
      "epitope.")

_cav = tier[tier["gene"].isin(SHORT_GENES) & tier["surface_caveat"].ne("")]
print("\ndocumented surface caveats among the surviving candidates:")
print(_cav[["gene", "surface_caveat"]].to_string(index=False) if len(_cav)
      else "  none of the surviving candidates has a documented internalisation, shedding or "
           "lattice mechanism in `surface_targets.SURFACE_CAVEAT` -- which means nobody has "
           "looked, not that there is none")
```

```
candidate beside the machine that removes it from the surface:
   gene partner  gene_malig  gene_react  partner_malig  partner_react
   SELL  ADAM17       0.209       0.503          0.124          0.105
TNFRSF8  ADAM17       0.041       0.017          0.124          0.105
  IL2RA  ADAM17       0.184       0.246          0.124          0.105
  CTLA4   AP2M1       0.383       0.352          0.560          0.410
  PDCD1  FBXO38       0.242       0.160          0.091          0.069

Read each row as a pair. A higher transcript with a higher sink is not a higher epitope.

documented surface caveats among the surviving candidates:
  none of the surviving candidates has a documented internalisation, shedding or lattice mechanism in `surface_targets.SURFACE_CAVEAT` -- which means nobody has looked, not that there is none
```

```python  # cell 24
# ============================================================================
# §6  Escape risk
# ============================================================================
esc = TS.escape_risk(long, MALIG, cluster_long=cluster_long)
print("escape risk computed on",
      "per-sample clusters" if cluster_long is not None else "donors only")
cols = [c for c in ("gene", "between_donor_sd", "within_donor_sd", "min_donor_pct") if c in esc]

print("\nthe surviving candidates, by within-patient heterogeneity:")
print(esc[esc["gene"].isin(SHORT_GENES)]
      .merge(prev[["gene", "median_pct_expr"]], on="gene", how="left")
      .sort_values("within_donor_sd" if "within_donor_sd" in esc else "between_donor_sd",
                   ascending=False)
      [cols + ["median_pct_expr"]]
      .to_string(index=False, float_format=lambda v: f"{v:.3g}"))

print("\n`between_donor_sd` is the spread of detection across patients and `within_donor_sd` "
      "the spread across one patient's own clusters -- a high second column is the "
      "antigen-negative subclone a single-target agent would select for, and `min_donor_pct` is "
      "the worst patient. 47 takes the between-patient term apart properly.")

hi = esc.merge(prev[["gene", "median_pct_expr", "frac_donors_positive"]], on="gene")
hi = hi[hi["frac_donors_positive"] >= 0.5].sort_values(
    "within_donor_sd" if "within_donor_sd" in hi else "between_donor_sd", ascending=False)
print("\nfor scale, over every prevalent gene -- most within-patient heterogeneity:")
print(hi.head(10)[cols + ["median_pct_expr"]].to_string(index=False,
                                                        float_format=lambda v: f"{v:.3g}"))
print("\nand the most uniform:")
print(hi.tail(10)[cols + ["median_pct_expr"]].to_string(index=False,
                                                        float_format=lambda v: f"{v:.3g}"))
```

```
escape risk computed on per-sample clusters

the surviving candidates, by within-patient heterogeneity:
    gene  between_donor_sd  within_donor_sd  min_donor_pct  median_pct_expr
   VDAC1             0.192            0.204         0.0805            0.482
   AP2S1             0.187            0.203         0.0222            0.358
   SURF4             0.201            0.179         0.0313            0.432
 TNFRSF4             0.228            0.177              0            0.349
     GPI             0.212            0.176         0.0476            0.474
    CD82               0.2            0.175          0.114            0.477
   SYPL1              0.18            0.145         0.0476            0.362
   PDIA4             0.174            0.145         0.0159            0.255
  LGALS3             0.289            0.139         0.0156            0.427
     CD9             0.225            0.134              0             0.25
   CD151             0.164            0.129         0.0171            0.196
     GRN             0.161            0.129         0.0213            0.211
    CALU             0.134            0.129         0.0256             0.19
    CD70             0.147            0.126        0.00494             0.21
  SLC1A4             0.182            0.126        0.00386            0.144
    TFRC               0.2            0.123         0.0293            0.296
TNFRSF25              0.22            0.121        0.00184            0.339
 SLC39A6             0.135            0.113         0.0316            0.183
  SLC5A3             0.172            0.107         0.0209            0.217
 TMEM30A             0.123           0.0925         0.0256            0.188
    FUT8             0.199           0.0868              0            0.184
     CPM             0.175           0.0798              0            0.137
    IL6R              0.15           0.0797              0            0.233
   QSOX1             0.122           0.0772        0.00285            0.133
  COL6A3             0.209           0.0758              0            0.113
  SLC4A2             0.127           0.0753         0.0102             0.13
  TSPAN2              0.12           0.0752        0.00227            0.135
    STX6             0.108           0.0749              0            0.145

`between_donor_sd` is the spread of detection across patients and `within_donor_sd` the spread across one patient's own clusters -- a high second column is the antigen-negative subclone a single-target agent would select for, and `min_donor_pct` is the worst patient. 47 takes the between-patient term apart properly.

for scale, over every prevalent gene -- most within-patient heterogeneity:
  gene  between_donor_sd  within_donor_sd  min_donor_pct  median_pct_expr
 PDIA6              0.18             0.21          0.122            0.419
 VDAC1             0.192            0.204         0.0805            0.482
 AP2S1             0.187            0.203         0.0222            0.358
 C1QBP             0.203            0.197         0.0484            0.353
  SRP9             0.186            0.194          0.113            0.419
  P4HB             0.197            0.193         0.0684            0.478
SLC3A2             0.184            0.191         0.0598            0.466
  CALR             0.185             0.19          0.118            0.593
SEC61B             0.235            0.184         0.0333            0.544
SEC61G             0.231            0.184         0.0635            0.539

and the most uniform:
 gene  between_donor_sd  within_donor_sd  min_donor_pct  median_pct_expr
SPPL3             0.139           0.0473        0.00855            0.101
 GNLY             0.339            0.046              0            0.094
HLA-E             0.418           0.0325              0            0.689
 IST1             0.153           0.0316              0           0.0973
IL6ST             0.227           0.0243              0            0.102
HLA-B             0.46
... [342 more chars truncated]
```

## §7 — The table, the control check that licenses reading it, and the plausibility layer

No composite score and no cutoff: the axes stay in their own columns and the filtering decision
belongs next to the sentence that justifies the threshold.

**Run the control check first.** CCR4, TNFRSF8, KIR3DL2 and IL2RA are approved or registrational
in CTCL. If they are not near the top, the screen is miscalibrated and nothing below them is
evidence — that is what happened to the v1 skin pass, and it is the single number by which this
rebuild should be judged.

Then the layer the statistics cannot supply. `surface_confident`, `n_tm` and `surfy_class` come
from CSPA and Surfy, not from this cohort: a transcript can be reproducibly, tumour-restrictedly
up and still be a hopeless antibody or CAR target because the protein is cytoplasmic, secreted or
mitochondrial. In skin that is not a hypothetical — ENO1, VDAC1, GPI and CALU all passed the v1
statistics.

```python  # cell 26
# ============================================================================
# §7  Assemble, check the controls, shortlist
# ============================================================================
surf = ST.load_surfaceome(strict=False)
rank = TS.rank_table(prev, sel, off, esc, ST.annotate, surfaceome=surf)
rank = rank.merge(tier[["gene", "surface_tier", "surface_caveat"]], on="gene", how="left")
rank = rank.merge(ev[["gene", "artifact_flag", "n_fails", "fail_reasons", "retained_trim",
                      "retained_clean", "retained_ds", "rho_burden", "top1_share"]],
                  on="gene", how="left")
if "top_cluster_share" in ev:
    rank = rank.merge(ev[["gene", "top_cluster_share"]], on="gene", how="left")

hit, missing, n_tested = TS.control_check(rank, ST.POSITIVE_CONTROLS)
print(f"positive controls of {n_tested:,} ranked genes:")
print(hit.to_string(index=False))
if missing:
    print(f"*** not testable: {missing} ***")
_worst = hit["rank"].max() if len(hit) else np.nan
print(f"\nworst control rank: {_worst} of {n_tested:,} "
      f"({_worst / n_tested:.0%} of the list)" if len(hit) else "")
print("A control outside the top ~10% is a calibration failure, not a discovery. The v1 pass "
      "put the worst control at 1,968 of 2,115 -- compare that number with the one above "
      "before reading anything below.")

short = rank[rank["gene"].isin(SHORT_GENES)].copy()
show = ["gene", "cd_name", "frac_donors_positive", "median_pct_expr", "log2fc_paired", "q",
        "worst_T_arm", "healthy_t_burden", "worst_nonT_skin", "worst_nonT_level",
        "worst_healthy_skin", "between_donor_sd", "stage", "n_evidence", "surface_confident",
        "n_tm", "surfy_class", "surface_tier", "artifact_flag"]
print(f"\n{len(short)} candidates pass the gates:")
print(short[[c for c in show if c in short]]
      .to_string(index=False, float_format=lambda v: f"{v:.4g}"))

print("\n" + "=" * 78)
print("PLAUSIBILITY, which the statistics cannot decide")
print("=" * 78)
print("`surface_confident` / `n_tm` / `surfy_class` come from CSPA + Surfy, not from this "
      "cohort. Genes with no surfaceome evidence are the ones to argue about before anyone "
      "orders a reagent:")
_noev = short[short["surface_confident"].isna() | ~short["surface_confident"].astype(bool)]
print(_noev[["gene", "cd_name", "log2fc_paired", "median_pct_expr", "n_evidence",
             "surfy_class", "topology"]].to_string(index=False,
                                                   float_format=lambda v: f"{v:.3g}")
      if len(_noev) else "  every shortlisted gene has confident surfaceome evidence")

print("\nnovel = passes the gates AND has no entry in CLINICAL_PRECEDENT:")
novel = short[short["stage"].fillna("").eq("")]
print(novel[["gene", "cd_name", "frac_donors_positive", "log2fc_paired", "panel_class",
             "surface_confident", "surfy_class"]]
      .to_string(index=False, float_format=lambda v: f"{v:.3g}"))

rank.to_csv(ST.tab("candidates_skin"), index=False)
short.to_csv(ST.tab("shortlist_skin"), index=False)
print("\nwrote", ST.tab("candidates_skin"), rank.shape, "and", ST.tab("shortlist_skin"),
      short.shape)
```

```
positive controls of 2,355 ranked genes:
   gene  rank  log2fc_paired  frac_donors_positive
TNFRSF8   130       0.331786              0.192308
   CCR4   137       0.323966              0.705128
KIR3DL2   690       0.000000              0.320513
  IL2RA  2180      -0.171807              0.628205

worst control rank: 2180 of 2,355 (93% of the list)
A control outside the top ~10% is a calibration failure, not a discovery. The v1 pass put the worst control at 1,968 of 2,115 -- compare that number with the one above before reading anything below.

28 candidates pass the gates:
    gene cd_name  frac_donors_positive  median_pct_expr  log2fc_paired         q    worst_T_arm  healthy_t_burden  worst_nonT_skin worst_nonT_level  worst_healthy_skin  between_donor_sd   stage  n_evidence surface_confident n_tm   surfy_class surface_tier artifact_flag
    CD70    CD70                0.8077           0.2096          1.102 3.089e-10 CD4_unassessed           0.02397           0.7701                B                   0            0.1471 phase 1           2              True  1.0  Unclassified      unknown         clean
     CD9                        0.7179           0.2499          1.085 7.092e-07 CD4_unassessed            0.1071            2.704      F_papillary                 NaN             0.225                   3              True  4.0 Miscellaneous      unknown         clean
     CPM                        0.5769           0.1371          1.023 7.081e-08 CD4_unassessed          0.006689            1.152             Mast                 NaN            0.1753                   3              True  0.0       Enzymes      unknown         clean
  COL6A3                        0.5641           0.1126         0.8105 3.281e-09       CD4_Treg            0.0158            3.034      F_papillary              0.9016            0.2088                   1              True               None      unknown         clean
   CD151                        0.7949           0.1958         0.7937  9.39e-08 CD4_unassessed           0.06122            1.744  F_myofibroblast                 NaN            0.1638                   3              True  4.0 Miscellaneous      unknown         clean
     GRN                        0.8462           0.2108         0.7777 9.806e-10 CD4_unassessed           0.04096            2.594   Mac_SPP1_TREM2                 NaN            0.1608                   1              True               None      unknown         clean
    TFRC                        0.9487           0.2956         0.7313 1.675e-09 CD4_unassessed            0.1176            1.092   Mac_SPP1_TREM2                 NaN            0.1997                   2              True               None      unknown         clean
 SLC39A6                        0.9103           0.1826          0.707 1.911e-10 CD4_unassessed           0.04469            1.001       Melanocyte                 NaN            0.1352                   4              True  6.0  Transporters      unknown         clean
  SLC4A2                         0.641           0.1297         0.6978 1.911e-10 CD4_unassessed           0.01562           0.3712       Melanocyte                 NaN            0.1274                   2              True               None      unknown         clean
TNFRSF25                        0.8718           0.3394         0.6977 9.137e-05 CD4_unassessed           0.09434           0.2211      Neutrophils                 NaN            0.2203                   3              True  1.0     Receptors      unknown         clean
  SLC1A4                        0.6282           0.1437         0.6901  5.23e-05 CD4_unassessed           0.02389            1.234       Melanocyte                 NaN            0.1817                   3              True  9.0  Transporters      unknown         clean
  LGALS3                        0.8846           0.4271         0.6634  0.001147 CD4_unassessed            0.3607            3.239             Mast               3.686            0.
... [8300 more chars truncated]
```

```python  # cell 27
# ============================================================================
# §7b FG  Two views of the whole ranking: calibration, and the trade every target makes
# ============================================================================
d = rank.dropna(subset=["log2fc_paired", "frac_donors_positive"])
is_ctrl = d["gene"].isin(ST.POSITIVE_CONTROLS)
is_short = d["gene"].isin(short["gene"])
is_band = d["gene"].isin(TQ.BAND_GENES)

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))

ax = axes[0]
ax.scatter(d["log2fc_paired"], d["frac_donors_positive"], s=4, c="#d5d8dc", linewidths=0)
ax.scatter(d.loc[is_band, "log2fc_paired"], d.loc[is_band, "frac_donors_positive"],
           s=12, facecolors="none", edgecolors=TF.DOWN, linewidths=.6,
           label="lineage controls")
ax.scatter(d.loc[is_short, "log2fc_paired"], d.loc[is_short, "frac_donors_positive"],
           s=10, c="#2e86c1", linewidths=0, label="shortlist")
ax.scatter(d.loc[is_ctrl, "log2fc_paired"], d.loc[is_ctrl, "frac_donors_positive"],
           s=26, c="#c0392b", linewidths=0, label="approved / registrational")
for _, r_ in d[is_ctrl].iterrows():
    ax.annotate(r_["gene"], (r_["log2fc_paired"], r_["frac_donors_positive"]), fontsize=6,
                xytext=(3, 3), textcoords="offset points")
ax.axvline(0, lw=0.5, c="k")
ax.set_xlabel("log2FC malignant vs same-donor reactive CD4")
ax.set_ylabel("fraction of donors positive")
ax.set_title("Selectivity against prevalence, whole ranking", loc="left", fontsize=7.5)
ax.legend(fontsize=5.5, frameon=False, loc="lower left")
TF._clean(ax)

ax = axes[1]
if "healthy_t_burden" in rank.columns:
    d2 = rank.dropna(subset=["log2fc_paired", "healthy_t_burden"])
    d2 = d2[d2["frac_donors_positive"] >= 0.3]
    k = d2["gene"].isin(ST.POSITIVE_CONTROLS) | d2["gene"].isin(SHORT_GENES[:12])
    ax.scatter(d2["healthy_t_burden"], d2["log2fc_paired"], s=5, c="#d5d8dc", linewidths=0)
    ax.scatter(d2.loc[k, "healthy_t_burden"], d2.loc[k, "log2fc_paired"], s=18, c="#c0392b",
               linewidths=0)
    for _, r_ in d2[k].iterrows():
        ax.annotate(r_["gene"], (r_["healthy_t_burden"], r_["log2fc_paired"]), fontsize=6,
                    xytext=(3, 2), textcoords="offset points")
    ax.axhline(0, lw=0.5, c="k")
    ax.set_xlabel("share of healthy CD4 cells positive  (T-cell aplasia cost)")
    ax.set_ylabel("log2FC malignant vs reactive")
    ax.set_title("Selectivity against healthy-T burden", loc="left", fontsize=7.5)
    TF._clean(ax)
else:
    ax.set_axis_off()
    ax.text(0.5, 0.5, "no CD4_HC arm in this run", ha="center", fontsize=7)

fig.tight_layout()
TF.save(fig, "FG_ranking_overview", caption_text=(
    f"{len(PAIRED_DONORS)} paired donors. Left: every ranked gene; red = approved or "
    "registrational in CTCL, open red = the contamination null. Right: donors positive >= 30%."))
TF.save_source(d[["gene", "log2fc_paired", "frac_donors_positive", "healthy_t_burden"]],
               "FG_ranking_overview_source")
TF.logfig("FG_ranking_overview", "selectivity vs prevalence and vs aplasia cost", len(d))

print("Where the validated antigens sit in the left panel is the calibration statement; where "
      "the lineage controls sit is the contamination statement. Both are read off one figure "
      "on purpose.")
```

```
wrote
  /home/projects/nyosef/zvise/MF/figures/targets/tgt42_FG_ranking_overview.png
  /home/projects/nyosef/zvise/MF/figures/targets/tgt42_FG_ranking_overview.pdf

<Figure size 960x340 with 2 Axes>
wrote /home/projects/nyosef/zvise/MF/tables/tgt42_FG_ranking_overview_source.csv
Where the validated antigens sit in the left panel is the calibration statement; where the lineage controls sit is the contamination statement. Both are read off one figure on purpose.
```

## §8 — Dossier

What was screened, on which cohort and call, with which gates, and what the result is licensed to
say. The compartment comparison — which antigens separate tumour from reactive in **both** skin
and blood — lives in `43 §8`, which reads `tgt42_candidates_skin.csv`; with this rebuild both
sides of that comparison have been through the same cohort fix and the same artifact tests, so
the caveat 43 carried about an uncleaned skin table no longer applies.

```python  # cell 29
# ============================================================================
# §8  Dossier
# ============================================================================
def SEC(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)
def kv(k, v): print(f"  {k:36s} {v}")

SEC("1 - INPUT")
kv("object", Path(man["source"]).name)
kv("builder", man.get("builder"))
kv("cells", f"{man['n_cells']:,}")
kv("genes in object", f"{man['n_genes_in_object']:,}")
kv("genes screened", f"{man['gene_selection'].get('n_total', '?'):,}")
kv("malignancy call", man["malignancy_call"])
kv("library for CP10K", man.get("library"))

SEC("2 - COHORT")
kv("donor keys in the object", len(don))
for reason, n in don.loc[~don["cohort_keep"], "drop_reason"].value_counts().items():
    kv(f"dropped: {reason}", n)
kv("independent keys", int(don["cohort_keep"].sum()))
kv("people", don.loc[don["cohort_keep"], "person"].nunique())
kv("donors with both CD4 arms", len(PAIRED_DONORS))
print()
print(pd.crosstab(don.loc[don["cohort_keep"], "dx"],
                  don.loc[don["cohort_keep"], "study"]).to_string())

SEC("3 - ARMS"); print(arms.to_string())

SEC("4 - GATES")
kv("min cells per donor-arm", MIN_CELLS)
kv("min paired donors", MIN_DONORS)
kv("detection floor", DETECT_FLOOR)
kv("shortlist", "clean, q<0.05, paired log2FC>0.5, >=50% donors positive, "
                "not a loss marker, not a lineage control")

SEC("5 - ARTIFACT TESTS")
kv("lineage controls found", band.attrs["n_genes"])
kv("coupling modules", TQ.COUPLING_MODULES[TISSUE])
kv("null calibrated", ok)
print("  " + msg)
print()
print(ev["artifact_flag"].value_counts().to_string())
kv("\n  genes excluded from the headline",
   int((ev["artifact_flag"].ne("clean") & (ev["log2fc_paired"] > 0)).sum()))

SEC("6 - CALIBRATION"); print(hit.to_string(index=False))
if missing:
    print(f"  not testable: {missing}")

SEC("7 - WHAT THIS IS NOT")
for line in [
    "A target call. It is a transcript ranking.",
    "A measure of epitope density, accessibility, turnover or shedding. Skin carries no ADT, "
    "so the protein check 45 runs for blood has no equivalent here.",
    "Blocked for the healthy-skin arm: Reynolds 2021 is a different lab, chemistry, site and "
    "sex, and that arm flags a candidate rather than ranking it.",
    "Able to see genes the screened set does not carry -- absent is not unexpressed.",
    "Independent of the malignancy call: every arm inherits mal_cluster's errors, and §2d's "
    "TCR rebuild is the only control for that.",
    "A statement about patients whose clone is not CD4, who are now out of the cohort "
    "entirely, or about LyP and parapsoriasis, which were never this screen's question.",
]:
    print("  - " + line)

SEC("8 - OUTPUTS")
TF.write_index("figure_index")
for p in sorted(ST.TAB_DIR.glob(f"{PREFIX}*")) + sorted(ST.FIG_DIR.glob(f"{PREFIX}*")):
    print("  ", p.relative_to(NB_DIR))
```

```
==============================================================================
1 - INPUT
==============================================================================
  object                               skin_T_tcr_annotated_v4.h5ad
  builder                              run_target_screen_qc.py
  cells                                539,916
  genes in object                      42,347
  genes screened                       2,355
  malignancy call                      mal_cluster
  library for CP10K                    obs.total_counts (mean_all); HVG column sum kept as mean_all_hvg

==============================================================================
2 - COHORT
==============================================================================
  donor keys in the object             146
  dropped: malignant clone is not CD4 (gd) 7
  dropped: diagnosis outside the MF/SS spectrum (LyP) 7
  dropped: diagnosis outside the MF/SS spectrum (SPP) 6
  dropped: malignant clone is not CD4 (CD8) 6
  dropped: malignant clone is not CD4 (DN) 2
  dropped: same person as D7__MF311    2
  dropped: diagnosis outside the MF/SS spectrum (LPP) 2
  dropped: same person as D7__MF309    1
  dropped: same person as D8__P90      1
  dropped: same person as D8__P65      1
  dropped: same person as D1__P312     1
  dropped: same person as D13__MF19    1
  independent keys                     109
  people                               107
  donors with both CD4 arms            75

study  alkon2024  brentuximab2026  brunner2024  buus2025  chennareddy2025  gaydosik2019  gaydosik2022  gaydosik2023  herrera2021  il4ra2026  jonak2021  li2024  rindler2021_mc
dx                                                                                                                                                                            
HC             0                0            0         0                4             4             4             5            1          4          0       0               0
MF             5                6            3         0                5             2             3             0            1         10          1      35               4
MF/SS          0                0            0         0                0             0             2             0            0          0          0       0               0
SS             0                0            2         3                0             1             0             0            3          1          0       0               0

==============================================================================
3 - ARMS
==============================================================================
                donors   cells  studies
arm                                    
CD4_malignant       78  187748       12
CD4_reactive        82  147233       12
CD8                 86   45562       13
CD4_Treg            77   35560       13
CD4_HC              21    4724        5
CD4_unassessed       2      71        1

==============================================================================
4 - GATES
==============================================================================
  min cells per donor-arm              20
  min paired donors                    5
  detection floor                      0.1
  shortlist                            clean, q<0.05, paired log2FC>0.5, >=50% donors positive, not a loss marker, not a lineage control

==============================================================================
5 - ARTIFACT TESTS
==============================================================================
  lineage controls found               40
  coupling modules                     ['keratinocyte', 'fibroblast', 'myeloid']
  null calibrated                      False
  40 lineage controls: median log2FC +0.00, median frac_donors_up 0.35 -- the controls are NOT up in the malignant arm, so differential contamination is not what this cohort shows. Do not exclude any 
... [3091 more chars truncated]
```

```python  # cell 30
# ============================================================================
# §9  The complete numerical result, as one block to hand to another model
# ============================================================================
# Everything this notebook computed, with a sentence on what each number is and how it was
# produced. Printed here and written to tables/tgt42_llm_export.md; paste the whole block
# between the banner lines into any other LLM and it needs no other file.
_ctx = f"""
Compartment: MF/CTCL **skin**, {man['n_cells']:,} T/NK cells from
{Path(man['source']).name} ({man['n_genes_in_object']:,} genes in the object; the screen tests
the {man['gene_selection'].get('n_total', 0):,}-gene surface panel union surfaceome selection
plus lineage and trafficking controls).

Cohort: {len(don)} donor keys reduced to {int(don['cohort_keep'].sum())} independent keys
({don.loc[don['cohort_keep'], 'person'].nunique()} people) by dropping non-CD4 clones,
diagnoses outside the MF/SS spectrum, and repeat biopsies of one patient. Diagnosis from the
curated `clin_dx`. {len(PAIRED_DONORS)} donors carry both a malignant and a reactive CD4 arm and
enter the paired test.

Arms: malignancy call `{man['malignancy_call']}`; CD4 cells are split into CD4_malignant,
CD4_reactive, CD4_unassessed (never used as a comparator) and CD4_HC (healthy donors), beside
CD4_Treg and CD8.

Headline test: per gene, the within-donor difference of mean log1p CP10K between the malignant
and reactive arms, paired Wilcoxon over donors, BH-adjusted (`q`). `log2fc_paired` is the median
per-donor log2 ratio; `frac_donors_up` the share of donors with a positive difference.
`z_blocked` is the study-blocked van Elteren statistic, which also uses donors with one arm.

Known answers, for calibration: CCR4 (mogamulizumab) and TNFRSF8/CD30 (brentuximab) are approved
in CTCL, KIR3DL2 is lacutamab's target, IL2RA is registrational. CD7 and DPP4/CD26 loss is the
textbook CTCL immunophenotype and must come out negative. The previous version of this screen
failed that check (worst control 1,968 of 2,115), which is what prompted the rebuild.
"""

_sections = [
    {"name": "§0b Cohort", "what": "which donor keys are independent MF/SS patients with a CD4 "
                                   "clone, and why each dropped key was dropped.",
     "how": "curated clin_* columns only (docs/CLINICAL_METADATA.md); no expression used.",
     "table": don[don["cohort_keep"]][["study", "dx", "dx2", "lesion", "lesion_class",
                                       "clin_stage", "clin_stage_class", "sex", "age",
                                       "is_cultured", "n_cells", "burden"]],
     "file": str(ST.tab("skin_cohort").relative_to(NB_DIR)), "max_rows": 120},
    {"name": "§0b Dropped keys", "what": "the keys removed from the cohort, with the reason.",
     "how": "same source; the three rules are non-CD4 clone, off-spectrum diagnosis, repeat "
            "biopsy of a patient already in the cohort.",
     "table": don.loc[~don["cohort_keep"], ["study", "dx", "phenotype", "lesion", "n_cells",
                                            "drop_reason"]]},
    {"name": "§1 Arms", "what": "donors, cells and studies per arm after the cohort fix.",
     "how": "groups of >= 20 cells.", "table": arms},
    {"name": "§1 Contamination load per arm",
     "what": "how dirty each arm is: share of lineage-positive cells, library size, doublet "
             "score and the tissue module scores.",
     "how": "median over the arm's donors. A malignant arm dirtier than its reactive arm is the "
            "precondition for the contamination story in §2.",
     "table": meta.groupby("arm")[_cl].median()},
    {"name": "§2a Contamination null",
     "what": "what pure contamination scores here: the six statistics computed on transcripts a "
             "CD4 T cell cannot make (keratins, collagens, VWF, PMEL, tryptases, LYZ, HBB).",
     "how": "same paired test and per-cell couplings as every other gene. Thresholds for the "
            "artifact flags are read off the p90 row of this band, not chosen.",
     "table": ev[ev["gene"].isin(TQ.BAND_GENES)][_bcols]},
    {"name": "§2a Calibration verdict",
     "what": "whether the contamination mechanism is operating in this cohort at all.",
     "how": "the null is calibrated only if the lineage controls are themselves up in the "
            "malignant arm (median log2FC > 0.1 and median frac_donors_up > 0.5). If not, no "
            "gene is excluded on contamination grounds.",
     "table": {"null_calibrated": ok, "verdict": msg}},
    {"name": "§2b Artifact verdict, per gene",
     "what": "the six tests and the resulting flag for every gene up at q < 0.05.",
     "how": "fails_trim / fails_clean / fails_ds are the share of the effect retained after "
            "trimming the top 1% of cells, dropping lineage-positive and high-doublet cells, "
            "and depth-matching; fails_concentration / fails_lineage / fails_umi are "
            "prevalence-matched residuals against the band; fails_burden is the Spearman of the "
            "per-donor effect against that donor's malignant:reactive ratio. Two failures = "
            "artifact, one = suspect.",
     "table": _full[_fc], "file": str(ST.tab("artifact_evidence").relative_to(NB_DIR)),
     "max_rows": 80},
    {"name": "§3 Candidates passing every gate",
     "what": "the genes that are selectively up on malignant CD4 and survived §2.",
     "how": "clean, q < 0.05, paired log2FC > 0.5, detected in >= 50% of donors, not a loss "
            "marker, not a lineage control.",
     "table": up[[c for c in _c if c in up]]},
    {"name": "§3 Loss markers",
     "what": "the CTCL loss phenotype, as the sanity check that cleaning did not flatten real "
             "biology.", "how": "same paired test.",
     "table": core[core["gene"].isin(ST.LOSS_MARKERS)][["gene", "log2fc_paired",
                                                        "frac_donors_up", "q",
                                                        "artifact_flag"]]},
    {"name": "§3 FD Per-gene summary of the donor x gene effect matrix",
     "what": "how consistent each leading antigen's effect is across patients.",
     "how": "per-donor log2FC matrix; frac_up_1 is the share of donors where the antigen is at "
            "least doubled.", "table": _sum},
    {"name": "§3 FE Eligibility",
     "what": "in what fraction of patients at least X% of the malignant CD4 cells carry the "
             "antigen.", "how": "transcript detection per donor, thresholded.",
     "table": elig, "file": str(ST.tab("eligibility_skin").relative_to(NB_DIR))},
    {"name": "§4 Off-tumour",
     "what": "what else carries each candidate: the worst non-target T arm, healthy CD4, the "
             "worst non-T skin compartment and healthy skin.",
     "how": "T arms from this screen's long table (share of cells detected); non-T compartments "
            "from the degradome pseudobulk cube (log1p CP10K, sample level, median per level); "
            "healthy skin from Reynolds 2021, which is unblocked and flags rather than ranks.",
     "table": short[[c for c in ("gene", "worst_T_arm", "worst_T_arm_pct", "healthy_t_burden",
                                 "worst_nonT_skin", "worst_nonT_level", "worst_healthy_skin",
                                 "worst_healthy_skin_level") if c in short]]},
    {"name": "§5 Trafficking modules per arm",
     "what": "the state of the machinery that puts a protein on the surface and keeps it there.",
     "how": "per-module z across donors, then the arm mean. Context, not a correction factor.",
     "table": arm_mod},
    {"name": "§6 Escape risk",
     "what": "how much each candidate's detection varies between patients and between one "
             "patient's own clusters.",
     "how": "SD of detection across donors and, within a donor, across its per-sample clusters.",
     "table": esc[esc["gene"].isin(SHORT_GENES)][cols]},
    {"name": "§7 Control check",
     "what": "where the antigens with a known answer sit in the ranking. This decides whether "
             "anything else here is evidence.",
     "how": "rank by paired log2FC over all screened genes.", "table": hit},
    {"name": "§7 Shortlist with the plausibility layer",
     "what": "the final candidate table: effect, prevalence, off-tumour cost, heterogeneity, "
             "clinical precedent and surfaceome evidence.",
     "how": "surface_confident / n_tm / surfy_class come from CSPA and Surfy, not from this "
            "cohort; they are the check that a statistically clean hit is a protein an antibody "
            "could reach.",
     "table": short[[c for c in show if c in short]],
     "file": str(ST.tab("shortlist_skin").relative_to(NB_DIR))},
]
TF.llm_export("42", "Surface-target screen, skin", _ctx, _sections)
```

```
======================== COPY FROM HERE ========================

# 42 — Surface-target screen, skin

This block is a complete, self-contained numerical report from one notebook of the MF/CTCL
surface-target project. Everything another model needs in order to reason about these numbers is
inside it: what was measured, on which cohort, with which test, and the result tables themselves.
No file outside this block has to be read. Numbers are given to 3 significant figures; tables are
CSV with a header row. A column named `q` is a Benjamini-Hochberg adjusted p-value unless the
section says otherwise.

## Context

Compartment: MF/CTCL **skin**, 539,916 T/NK cells from
skin_T_tcr_annotated_v4.h5ad (42,347 genes in the object; the screen tests
the 2,355-gene surface panel union surfaceome selection
plus lineage and trafficking controls).

Cohort: 146 donor keys reduced to 109 independent keys
(107 people) by dropping non-CD4 clones,
diagnoses outside the MF/SS spectrum, and repeat biopsies of one patient. Diagnosis from the
curated `clin_dx`. 75 donors carry both a malignant and a reactive CD4 arm and
enter the paired test.

Arms: malignancy call `mal_cluster`; CD4 cells are split into CD4_malignant,
CD4_reactive, CD4_unassessed (never used as a comparator) and CD4_HC (healthy donors), beside
CD4_Treg and CD8.

Headline test: per gene, the within-donor difference of mean log1p CP10K between the malignant
and reactive arms, paired Wilcoxon over donors, BH-adjusted (`q`). `log2fc_paired` is the median
per-donor log2 ratio; `frac_donors_up` the share of donors with a positive difference.
`z_blocked` is the study-blocked van Elteren statistic, which also uses donors with one arm.

Known answers, for calibration: CCR4 (mogamulizumab) and TNFRSF8/CD30 (brentuximab) are approved
in CTCL, KIR3DL2 is lacutamab's target, IL2RA is registrational. CD7 and DPP4/CD26 loss is the
textbook CTCL immunophenotype and must come out negative. The previous version of this screen
failed that check (worst control 1,968 of 2,115), which is what prompted the rebuild.

## §0b Cohort

**What:** which donor keys are independent MF/SS patients with a CD4 clone, and why each dropped key was dropped.

**How:** curated clin_* columns only (docs/CLINICAL_METADATA.md); no expression used.

**Full table on disk:** `tables/tgt42_skin_cohort.csv`

```csv
donor,study,dx,dx2,lesion,lesion_class,clin_stage,clin_stage_class,sex,age,is_cultured,n_cells,burden
B5__PT09,buus2025,SS,SS,erythroderma,lesion,,,F,71,False,18455,0.664
B5__PT22,buus2025,SS,SS,erythroderma,lesion,,,M,70,False,9892,
B5__PT23,buus2025,SS,SS,erythroderma,lesion,,,F,85,False,8229,0.836
D10__JONAK1,jonak2021,MF,MF,plaque,lesion,IB,early,F,33,False,2799,0.153
D11__P1,brentuximab2026,MF,MF,plaque,lesion,IIIA,advanced,F,52,False,898,0.883
D11__P2,brentuximab2026,MF,MF,tumor,lesion,IIIA,advanced,F,48,False,11850,0.899
D11__P3,brentuximab2026,MF,MF,tumor,lesion,IIB,advanced,F,55,False,10804,0.827
D11__P4,brentuximab2026,MF,MF,plaque,lesion,IIIA,advanced,F,45,False,897,0.254
D11__P5,brentuximab2026,MF,MF,tumor,lesion,IVA2,advanced,F,61,False,9000,0.827
D11__P6,brentuximab2026,MF,MF,plaque,lesion,IVA2,advanced,M,64,False,7515,0.769
D13__MF17,gaydosik2022,MF,MF,tumor,lesion,IIB,advanced,F,76,False,3991,0.985
D13__MF18,gaydosik2022,MF,MF,tumor,lesion,IIB,advanced,M,87,False,2677,0.469
D13__MF19,gaydosik2022,MF/SS,unresolved,tumor,lesion,IIIB,advanced,M,62,False,528,0.518
D13__MF21,gaydosik2022,MF,MF,tumor,lesion,IIB,advanced,M,61,False,966,0.177
D13__MF25,gaydosik2022,MF/SS,unresolved,erythroderma,lesion,IVA1,advanced,M,68,False,430,0.354
D13__SC327_HTO1,gaydosik2022,HC,HC,healthy_skin,healthy,,,,,False,28,
D13__SC327_HTO2,gaydosik2022,HC,HC,healthy_skin,healthy,,,,,False,225,
D13__SC374_HTO10,gaydosik2022,HC,HC,healthy_skin,healthy,,,,,False,935,
D13__SC374_HTO9,gaydosik2022,HC,HC,healthy_skin,healthy,,,,,False,230,
D14__SC320,gaydosik2023,HC,HC,healthy_skin,healthy,,,F,,False,140,
D14__SC348,gaydosik2023,HC,HC,hea
... [41960 more chars truncated]
```
