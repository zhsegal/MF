# 40_targets/skin/49_skin_cd4_surfaceome_numeric.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/40_targets/skin/49_skin_cd4_surfaceome_numeric.ipynb` and is never run._

# 49 — Skin CD4 surfaceome: malignant vs reactive CD4, numbers only

A lean, figure-free re-run of the skin surface-target screen (42), restricted to the **CD4
compartment** of the skin atlas. The question: which surface-protein transcripts are higher on
**malignant** CD4 T cells than on the **reactive (non-malignant) CD4 T cells of the same patient**,
in how many patients, and how much is there on healthy-skin CD4?

Every code cell prints its result as a **complete CSV table** under a `WHAT` / `HOW` header, so
any block can be copied straight into another analysis. The last cell writes every table, with its
explanation, to one file: `tables/tgt49_llm_export.md`. Full tables also go to `tables/tgt49_*.csv`.

**Out of scope here (42 covers both).** 42's six artifact tests (contamination null, trimming,
depth matching, lineage coupling, burden scaling), whose flags are in
`tables/tgt42_artifact_evidence.csv`. Off-tumour expression on non-T skin cells (keratinocyte,
fibroblast, myeloid…). Surface **protein**: skin has no CITE-seq, so every number below is a
transcript.

## Data

| item | value |
|---|---|
| object | `data/atlas_joint/skin_T_tcr_annotated_v4.h5ad`: skin T/NK cells of atlas v2, raw counts, all 42,347 genes |
| input to this notebook | `data/targets/skin_target_long.parquet`, one row per (arm, donor, gene), built by `jobs/run_target_screen_qc.sh --tissue skin` |
| normalisation | per cell, log1p(counts / `obs.total_counts` × 10,000), i.e. log1p CP10K on the **true** library size |
| malignancy call | `mal_cluster` (nb26b, `skin_T_malignancy_cluster_v1.parquet`): a per-cell malignant / non-malignant / undefined call, made at the level of per-sample Leiden clusters |
| arms used | `CD4_malignant`: CD4 cells called malignant. `CD4_reactive`: CD4 cells called non-malignant, in a patient. `CD4_HC`: CD4 cells of healthy-skin donors (`clin_dx == "HC"`). |
| arms not used | `CD8`, `CD4_Treg`, and `CD4_unassessed` (CD4 cells the call could not evaluate). They are counted in §1 and then dropped. |
| minimum group | an (arm, donor) pair enters only with ≥ 20 cells |
| genes | surface genes (a-priori CTCL panel ∪ confident surfaceome reference), plus lineage-control and trafficking-machinery genes the builder added as references. Exact counts are printed in §1–§2. |
| cohort | donor keys reduced to independent MF/SS patients whose malignant clone is CD4 (§1) |

## Column glossary (every table uses these names)

| column | meaning |
|---|---|
| `pct_expr` | fraction (0–1) of a donor's cells in an arm with ≥ 1 UMI of the gene |
| `mean_all` | mean over a donor's cells in an arm of log1p CP10K, zeros included |
| `log2fc_paired` | **median over paired donors** of log2((`mean_all`_malig + 0.05) / (`mean_all`_react + 0.05)). This is a ratio of *log-scale* means, so it is compressed relative to a raw fold change. Read it as a ranked effect size, not as "2^x times more mRNA". |
| `n_donors_paired` | donors with both a malignant and a reactive arm (≥ 20 cells each) for that gene |
| `frac_donors_up` | share of paired donors whose malignant `mean_all` > reactive `mean_all` |
| `p`, `q` | Wilcoxon signed-rank on the per-donor differences (malig − react `mean_all`). `q` is Benjamini–Hochberg over all tested genes. |
| `log2fc_detect`, `frac_donors_up_detect`, `q_detect` | the same paired test on `pct_expr` instead of `mean_all`, with the same 0.05 pseudo-count |
| `malig_median_pct`, `react_median_pct`, `hc_median_pct` | median over that arm's donors of `pct_expr` |
| `malig_frac_donors_pos` (and `react_`, `hc_`) | share of that arm's donors with `pct_expr` ≥ 0.10 ("positive patient") |
| `malig_median_mean` (and `react_`, `hc_`) | median over that arm's donors of `mean_all` |
| `log2fc_vs_hc`, `q_vs_hc` | malignant CD4 vs **healthy-skin** CD4: log2((median `mean_all` over malignant donors + 0.05) / (same over HC donors + 0.05)), with a two-sided Mann–Whitney on donor-level `mean_all` (different people, so unpaired), BH-adjusted |
| `rank_surface` | rank of `log2fc_paired` among tested surface genes (1 = most malignant-selective) |
| `burden` | malignant share of a donor's CD4: n_malignant / (n_malignant + n_reactive) cells |
| `gene_set` | `surface` (panel ∪ confident surfaceome), `lineage_control` (non-T lineage transcripts), `trafficking_control` (secretory / endocytic machinery), `other` |
| `is_lineage_control` | gene is in 42's skin contamination null (keratins, collagens, VWF, PMEL, tryptases, LYZ, HBB…), transcripts a T cell does not make |
| `source` | how a surface gene entered: `panel`, `surfaceome` or `both` |
| `surface_confident`, `n_evidence`, `n_tm`, `topology`, `surfy_class` | external surfaceome annotation (CSPA, SURFY, HPA, GO), not from this cohort |
| `cd_name`, `agent`, `stage` | clinical precedent, if any (`surface_targets.CLINICAL_PRECEDENT`) |
| `positive_control`, `loss_marker` | CCR4 / TNFRSF8 (CD30) / KIR3DL2 / IL2RA (approved or registrational in CTCL), and the CTCL loss markers CD7, DPP4 (CD26), CD5, CD6, CD27, CD2 |

| § | content |
|---|---|
| §1 | input, cohort, arms |
| §2 | gene universe |
| §3 | malignant vs reactive CD4, every gene (+ healthy CD4 reference) |
| §4 | calibration: known antigens and loss markers |
| §5 | shortlist |
| §6 | per-donor detail for the shortlist and controls |
| §7 | stratification: lesion, stage, disease |
| §8 | one-file export |

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
from scipy import stats
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

import surface_targets as ST
import target_screen as TS
import target_cohort as TC
import target_qc as TQ
import target_het as TH
import target_figs as TF
import cluster_malig_helpers as CM
import malignancy as MAL

# All three before any path is read: `malignancy` defaults to skin and `target_qc` to blood at
# import, so setting only one gives the other compartment's parquets or contaminant list.
CM.use_tissue("skin")
MAL.use_tissue("skin")
TQ.use_tissue("skin")
assert MAL.CLUSTER_PQ.name.startswith("skin_"), MAL.CLUSTER_PQ
assert "KRT14" in TQ.BAND_GENES, TQ.BAND_GENES

TISSUE = "skin"
TARGETS = NB_DIR / "data" / "targets"
PREFIX = "tgt49_"
ST.TAB_PREFIX = ST.FIG_PREFIX = PREFIX

MALIG, REACT, HC = "CD4_malignant", "CD4_reactive", "CD4_HC"
CD4_ARMS = [MALIG, REACT, HC]

MIN_CELLS = 20        # applied by the builder; restated so every table can be read against it
MIN_DONORS = 5        # a gene needs this many paired donors (or donors per arm) to be tested
DETECT_FLOOR = 0.10   # a donor is "positive" for a gene if >= 10% of its cells detect it
PSEUDO = 0.05         # added to both arms before every log2 ratio
SHORT_Q, SHORT_LFC, SHORT_PREV = 0.05, 0.5, 0.5   # shortlist gates, §5
N_DETAIL = 30         # shortlist genes carried into the per-donor detail, §6
STRAT_MIN_DONORS = 3  # a stratum needs this many paired donors to carry a p, §7

SECTIONS = []         # every reported table, in order, for the §8 export


def _tidy(table):
    """Make a table print cleanly as 3-significant-digit CSV.

    dict -> key,value with values kept as written (counts stay exact); a leftover row index from
    filtering/sorting is dropped; integer-valued float columns (counts with NaN) -> Int64, so
    12345 is not printed as 1.23e+04.
    """
    if isinstance(table, dict):
        s = pd.Series(table, dtype=object, name="value")
        s.index.name = "key"
        return s
    if not isinstance(table, pd.DataFrame):
        return table
    out = table.copy()
    if out.index.name is None and not isinstance(out.index, pd.RangeIndex):
        out = out.reset_index(drop=True)
    for c in out.columns:
        s = out[c]
        if pd.api.types.is_float_dtype(s) and s.notna().any() and (s.dropna() % 1 == 0).all():
            out[c] = s.astype("Int64")
    return out


def report(name, what, how, table, file=None, note=None, export=True):
    """Print one result block -- header, WHAT, HOW, complete CSV -- and queue it for §8."""
    table = _tidy(table)
    print("=" * 96)
    print(name)
    print("=" * 96)
    print("WHAT:", what)
    print("HOW: ", how)
    if file:
        print("FILE:", file)
    if note:
        print("NOTE:", note)
    n = len(table) if isinstance(table, (pd.DataFrame, pd.Series)) else ""
    print(f"[{n} rows]\n")
    print(TF._as_csv(table))
    print()
    if export:
        SECTIONS.append({"name": name, "what": what, "how": how, "table": table,
                         "file": file, "note": note})


def save(df, name, index=False):
    p = ST.tab(name)
    df.to_csv(p, index=index)
    return str(p.relative_to(NB_DIR))


print("NB_DIR =", NB_DIR)
print("tissue:", CM.TISSUE, "| malignancy call:", MAL.HEADLINE, "| from:", MAL.CLUSTER_PQ.name)
```

```
NB_DIR = /home/projects/nyosef/zvise/MF
tissue: skin | malignancy call: mal_cluster | from: skin_T_malignancy_cluster_v1.parquet
```

## §1 — Input, cohort, arms

`donor` in the target tables is a sample key, not a patient. The cohort is 42's, built by
`target_cohort.skin_donor_table()` from the curated `clin_*` columns (`docs/CLINICAL_METADATA.md`).
No expression is used. Three drop rules, applied in this order:

1. **The malignant clone is not CD4** (`clin_phenotype` = CD8, γδ or double-negative, plus
   pcAECTCL). The call finds no malignant CD4 cluster in these patients, so all of their CD4 cells
   would land in `CD4_reactive`, the benign side of the comparison.
2. **The diagnosis is outside the MF/SS spectrum** (LyP, SPP, LPP).
3. **The patient is already in the cohort under another key** (repeat biopsies). The lesional,
   deepest sample is kept.

Diagnosis is `clin_dx` throughout. `dx2` collapses it to MF / SS / HC / unresolved.

**Paired donors** are those with both a `CD4_malignant` and a `CD4_reactive` arm (≥ 20 cells
each). Only they enter the headline test.

```python  # cell 3
# ============================================================================
# §1  Input, cohort, arms
# ============================================================================
man = json.loads((TARGETS / f"{TISSUE}_target_manifest.json").read_text())
assert man.get("builder") == "run_target_screen_qc.py", (
    f"tables are not the QC build (builder = {man.get('builder')}); "
    "run jobs/run_target_screen_qc.sh --tissue skin")
gsel = man["gene_selection"]

report(
    "§1a Input",
    "the object, call, normalisation and gene counts behind every table below.",
    "read from the builder manifest, data/targets/skin_target_manifest.json.",
    {"object": Path(man["source"]).name,
     "tables_built": man.get("written"),
     "cells_in_object": man["n_cells"],
     "genes_in_object": man["n_genes_in_object"],
     "genes_screened_total": gsel.get("n_total"),
     "genes_surface_panel_or_surfaceome": gsel.get("n_screened"),
     "genes_lineage_controls_added": gsel.get("n_lineage_added"),
     "genes_trafficking_machinery_added": gsel.get("n_traffic_added"),
     "panel_genes_absent_from_object": ", ".join(gsel.get("panel_absent_from_object", []))
                                       or "none",
     "malignancy_call": man["malignancy_call"],
     "library_size_for_cp10k": man.get("library"),
     "min_cells_per_donor_arm": man.get("min_cells")})

# --- cohort ------------------------------------------------------------------------------
don = TC.skin_donor_table()
raw = pd.read_parquet(TARGETS / f"{TISSUE}_target_long.parquet")
long_all = TC.filter_long(raw, don)              # independent keys only
long = long_all[long_all["arm"].isin(CD4_ARMS)].copy()
del raw

don = don.join(TC.burden(long), how="left")
_both = long[long["arm"].isin([MALIG, REACT])].groupby("donor")["arm"].nunique().eq(2)
PAIRED = sorted(_both.index[_both])
don["paired"] = don.index.isin(PAIRED)
don["cd4_arms"] = (long.groupby("donor")["arm"].unique()
                   .map(lambda a: "|".join(sorted(a))).reindex(don.index).fillna(""))
keep = don[don["cohort_keep"]]

_summary = {"donor_keys_in_object": len(don)}
for reason, n in don.loc[~don["cohort_keep"], "drop_reason"].value_counts().items():
    _summary[f"dropped: {reason}"] = int(n)
_summary.update({
    "independent_donor_keys": len(keep),
    "people": keep["person"].nunique(),
    "donors_with_malignant_arm": long.loc[long["arm"].eq(MALIG), "donor"].nunique(),
    "donors_with_reactive_arm": long.loc[long["arm"].eq(REACT), "donor"].nunique(),
    "donors_with_healthy_CD4_arm": long.loc[long["arm"].eq(HC), "donor"].nunique(),
    "paired_donors (malignant AND reactive)": len(PAIRED),
    "paired_people": don.loc[PAIRED, "person"].nunique(),
})
for dx, n in don.loc[PAIRED, "dx"].value_counts().items():
    _summary[f"paired donors, dx = {dx}"] = int(n)
report("§1b Cohort summary",
       "how many donor keys, patients and paired donors the analysis rests on.",
       "skin_donor_table() drop rules (non-CD4 clone, off-spectrum dx, repeat biopsy), then arm "
       "presence in the CD4 tables (>= 20 cells per donor-arm).",
       _summary)

drop_tab = (don.loc[~don["cohort_keep"], ["study", "dx", "phenotype", "lesion", "person",
                                          "n_cells", "drop_reason"]]
            .sort_values(["drop_reason", "study"]).reset_index())
report("§1c Dropped donor keys",
       "every donor key removed from the cohort, with the reason.",
       "curated clin_* columns only; n_cells = all T/NK cells of the key in the object.",
       drop_tab)

COH_COLS = ["study", "dx", "dx2", "phenotype", "lesion", "lesion_class", "clin_stage",
            "clin_stage_class", "sex", "age", "is_cultured", "n_cells", "n_malignant",
            "n_reactive", "burden", "paired", "cd4_arms"]
coh = (keep[COH_COLS].rename(columns={"n_cells": "n_cells_object"})
       .sort_values(["dx2", "study"]).reset_index())
f = save(coh, "cohort")
report("§1d Cohort, one row per independent donor key",
       "the clinical description and CD4 composition of every donor that can enter a table.",
       "clinical fields from clin_*; n_malignant / n_reactive = CD4 cells in each arm (NaN = arm "
       "absent or < 20 cells); burden = n_malignant / (n_malignant + n_reactive); paired = has "
       "both arms; cd4_arms = the CD4 arms present.",
       coh, file=f)

# --- arms ------------------------------------------------------------------------------------
meta = TC.filter_long(pd.read_parquet(TARGETS / f"{TISSUE}_target_meta.parquet"), don)
arms = (meta.groupby("arm")
        .agg(donors=("donor", "nunique"), cells=("n_cells", "sum"),
             studies=("study", "nunique"), median_cells_per_donor=("n_cells", "median"),
             median_umi_per_cell=("median_total_counts", "median"),
             median_genes_per_cell=("median_n_genes", "median"),
             median_doublet_score=("median_doublet_score", "median"))
        .sort_values("cells", ascending=False))
arms["used_here"] = arms.index.isin(CD4_ARMS)
report("§1e Arms",
       "donors, cells and per-cell QC for every arm of the skin T/NK object; only the three "
       "CD4 arms (used_here = True) are analysed below.",
       "(arm, donor) groups with >= 20 cells, independent cohort; median_* = median over the "
       "arm's donors of each donor's per-cell median.",
       arms.reset_index())
```

```
================================================================================================
§1a Input
================================================================================================
WHAT: the object, call, normalisation and gene counts behind every table below.
HOW:  read from the builder manifest, data/targets/skin_target_manifest.json.
[12 rows]

key,value
object,skin_T_tcr_annotated_v4.h5ad
tables_built,2026-10-26 13:59
cells_in_object,539916
genes_in_object,42347
genes_screened_total,2355
genes_surface_panel_or_surfaceome,2115
genes_lineage_controls_added,30
genes_trafficking_machinery_added,210
panel_genes_absent_from_object,KIR2DL2
malignancy_call,mal_cluster
library_size_for_cp10k,obs.total_counts (mean_all); HVG column sum kept as mean_all_hvg
min_cells_per_donor_arm,20


================================================================================================
§1b Cohort summary
================================================================================================
WHAT: how many donor keys, patients and paired donors the analysis rests on.
HOW:  skin_donor_table() drop rules (non-CD4 clone, off-spectrum dx, repeat biopsy), then arm presence in the CD4 tables (>= 20 cells per donor-arm).
[23 rows]

key,value
donor_keys_in_object,146
dropped: malignant clone is not CD4 (gd),7
dropped: diagnosis outside the MF/SS spectrum (LyP),7
dropped: diagnosis outside the MF/SS spectrum (SPP),6
dropped: malignant clone is not CD4 (CD8),6
dropped: malignant clone is not CD4 (DN),2
dropped: same person as D7__MF311,2
dropped: diagnosis outside the MF/SS spectrum (LPP),2
dropped: same person as D7__MF309,1
dropped: same person as D8__P90,1
dropped: same person as D8__P65,1
dropped: same person as D1__P312,1
dropped: same person as D13__MF19,1
independent_donor_keys,109
people,107
donors_with_malignant_arm,78
donors_with_reactive_arm,82
donors_with_healthy_CD4_arm,21
paired_donors (malignant AND reactive),75
paired_people,75
"paired donors, dx = MF",67
"paired donors, dx = SS",6
"paired donors, dx = MF/SS",2

================================================================================================
§1c Dropped donor keys
================================================================================================
WHAT: every donor key removed from the cohort, with the reason.
HOW:  curated clin_* columns only; n_cells = all T/NK cells of the key in the object.
[37 rows]

donor,study,dx,phenotype,lesion,person,n_cells,drop_reason
D8__P143,alkon2024,LPP,not_applicable,patch,vienna:P143,859,diagnosis outside the MF/SS spectrum (LPP)
D8__P194,alkon2024,LPP,not_applicable,patch,vienna:P194,2166,diagnosis outside the MF/SS spectrum (LPP)
D9__P101,lyp2026,LyP,CD4,LyP_lesion,vienna:P101,3509,diagnosis outside the MF/SS spectrum (LyP)
D9__P118,lyp2026,LyP,CD4,LyP_lesion,vienna:P118,892,diagnosis outside the MF/SS spectrum (LyP)
D9__P193,lyp2026,LyP,gd,LyP_lesion,vienna:P193,11896,diagnosis outside the MF/SS spectrum (LyP)
D9__P213,lyp2026,LyP,CD8,LyP_lesion,vienna:P213,7308,diagnosis outside the MF/SS spectrum (LyP)
D9__P217,lyp2026,LyP,CD4,LyP_lesion,vienna:P217,625,diagnosis outside the MF/SS spectrum (LyP)
D9__P220,lyp2026,LyP,CD8,LyP_lesion,vienna:GSM9038965,5259,diagnosis outside the MF/SS spectrum (LyP)
D9__PEN,lyp2026,LyP,CD8,LyP_lesion,vienna:PEN,4739,diagnosis outside the MF/SS spectrum (LyP)
D8__P102,alkon2024,SPP,not_applicable,patch,vienna:P102,1384,diagnosis outside the MF/SS spectrum (SPP)
D8__P144,alkon2024,SPP,not_applicable,patch,vienna:P144,106,diagnosis outside the MF/SS spectrum (SPP)
D8__P149,alkon2024,SPP,not_applicable,patch,vienna:P149,654,diagnosis outside the MF/SS spectrum (SPP)
D8__P78,alkon2024,SPP,not_applicable,patch,vienna:P78,417,diagnosis outside the MF/SS spectrum (SPP)
D8__P79,alkon2024,SPP,not_applicable,patch,vienna:P79,1645,diagnosis outside the MF/SS spectrum (SPP)
D8__P88,alkon2024,SPP,not_applicable,patch,vienna:P88,865,diagnosis outside the MF/SS spectrum (
... [16173 more chars truncated]
```

## §2 — Gene universe

The builder screened the **surface genes**: the a-priori CTCL panel (`surface_targets.SURFACE_PANEL`)
∪ the confident surfaceome reference (`data/external_ref/surfaceome/surfaceome.parquet`). A gene is
`surface_confident` with two independent sources, or with one source that is evidence rather than
prediction (CSPA mass spectrometry on intact cells, SURFY curated/GPI). The union matters:
`n_evidence ≥ 2` alone would drop KIR3DL2. Two reference sets ride along:

- **lineage controls**: keratinocyte, fibroblast, endothelial, melanocyte, myeloid, mast, B and RBC
  transcripts a CD4 T cell does not make. Whatever they score in §3 is roughly what contamination
  scores. `is_lineage_control` marks 42's contamination null; those genes are never shortlisted.
- **trafficking machinery**: secretion, glycosylation, endocytosis and shedding genes. Kept as
  context only.

Absence from this universe means the gene was not asked about, not that it is unexpressed: the
object carries all 42,347 genes.

```python  # cell 5
# ============================================================================
# §2  Gene universe
# ============================================================================
surf_all = ST.load_surfaceome(confident_only=False, strict=True)
surf_conf = surf_all[surf_all["surface_confident"]]
GENES = sorted(long["gene"].unique())

SURFACE, _ = ST.screen_genes(GENES, surfaceome=surf_conf)
SURFACE = set(SURFACE)
PANEL, CONF = set(ST.PANEL_GENES), set(surf_conf["gene"])
LINEAGE = set(TQ.BAND_GENES) | {g for gl in man.get("lineage_modules", {}).values() for g in gl}
TRAFFIC = set(ST.TRAFFIC_GENES)


def _gene_set(g):
    if g in SURFACE:
        return "surface"
    if g in LINEAGE:
        return "lineage_control"
    if g in TRAFFIC:
        return "trafficking_control"
    return "other"


gene_tab = pd.DataFrame({"gene": GENES})
gene_tab["gene_set"] = gene_tab["gene"].map(_gene_set)
gene_tab["source"] = np.select(
    [gene_tab["gene"].isin(PANEL & CONF), gene_tab["gene"].isin(PANEL),
     gene_tab["gene"].isin(CONF)], ["both", "panel", "surfaceome"], "")
gene_tab.loc[gene_tab["gene_set"].ne("surface"), "source"] = ""
gene_tab["is_lineage_control"] = gene_tab["gene"].isin(TQ.BAND_GENES)

_counts = (gene_tab.assign(source=gene_tab["source"].replace("", "-"))
           .groupby(["gene_set", "source"]).size().rename("n_genes").reset_index())
report("§2a Gene universe",
       "how many genes of each kind the screen tests.",
       "surface = a-priori panel UNION confident surfaceome, intersected with the genes in the "
       "tables; lineage/trafficking = reference genes the builder added; a surface gene that is "
       "also a lineage or trafficking gene counts as surface.",
       _counts,
       note=f"{int(gene_tab['is_lineage_control'].sum())} genes form 42's contamination null "
            f"(is_lineage_control) and are excluded from the shortlist.")

_ctrl = pd.DataFrame({"gene": ST.POSITIVE_CONTROLS + ST.LOSS_MARKERS})
_ctrl["role"] = np.where(_ctrl["gene"].isin(ST.POSITIVE_CONTROLS), "positive_control",
                         "loss_marker")
_ctrl["in_tables"] = _ctrl["gene"].isin(GENES)
_ctrl["gene_set"] = _ctrl["gene"].map(dict(zip(gene_tab["gene"], gene_tab["gene_set"])))
report("§2b Control genes present",
       "whether each calibration gene can be tested.",
       "positive controls are approved or registrational CTCL antigens; loss markers are the "
       "textbook CTCL loss phenotype. CD26 is an alias of DPP4, so its absence is expected.",
       _ctrl)
```

```
================================================================================================
§2a Gene universe
================================================================================================
WHAT: how many genes of each kind the screen tests.
HOW:  surface = a-priori panel UNION confident surfaceome, intersected with the genes in the tables; lineage/trafficking = reference genes the builder added; a surface gene that is also a lineage or trafficking gene counts as surface.
NOTE: 40 genes form 42's contamination null (is_lineage_control) and are excluded from the shortlist.
[6 rows]

gene_set,source,n_genes
lineage_control,-,28
other,-,2
surface,both,125
surface,panel,17
surface,surfaceome,1973
trafficking_control,-,210

================================================================================================
§2b Control genes present
================================================================================================
WHAT: whether each calibration gene can be tested.
HOW:  positive controls are approved or registrational CTCL antigens; loss markers are the textbook CTCL loss phenotype. CD26 is an alias of DPP4, so its absence is expected.
[11 rows]

gene,role,in_tables,gene_set
CCR4,positive_control,True,surface
TNFRSF8,positive_control,True,surface
KIR3DL2,positive_control,True,surface
IL2RA,positive_control,True,surface
CD7,loss_marker,True,surface
DPP4,loss_marker,True,surface
CD5,loss_marker,True,surface
CD6,loss_marker,True,surface
CD27,loss_marker,True,surface
CD2,loss_marker,True,surface
CD26,loss_marker,False,
```

## §3 — Malignant vs reactive CD4, every gene

**Headline test (paired, within patient).** For each gene and each paired donor, take the
malignant arm's `mean_all` and the reactive arm's `mean_all`. Both arms come from the same biopsy
and library prep, so the donor is the block.

- effect: `log2fc_paired` = median over donors of log2((malig + 0.05) / (react + 0.05))
- consistency: `frac_donors_up` = share of donors with malig > react
- test: Wilcoxon signed-rank on the per-donor differences (malig − react), BH over all tested
  genes → `q`. A gene needs ≥ 5 paired donors.

The same test is run on **detection** (`pct_expr`), which is closer to what a CAR or antibody is
dosed against: the fraction of tumour cells carrying the transcript.

**Prevalence, per arm.** Median `pct_expr` and the share of donors with `pct_expr` ≥ 0.10, for each
of the malignant, reactive and healthy CD4 arms. Each arm uses all of its donors, not only the
paired ones.

**Healthy CD4 reference (unpaired).** Malignant CD4 donors vs healthy-skin CD4 donors on
donor-level `mean_all`: two-sided Mann–Whitney, BH. These are different people, studies and
chemistries, so treat it as a flag ("is this also on healthy skin CD4?"), not as a ranked effect.

Printed in full: every **surface** gene at q < 0.05 (up, then down), and the lineage controls. The
complete table for all genes, with every column, is in `tables/tgt49_all_genes.csv`.

```python  # cell 7
# ============================================================================
# §3  Malignant vs reactive CD4, every gene
# ============================================================================
sel = TS.paired_selectivity(long, MALIG, REACT, value="mean_all", min_donors=MIN_DONORS,
                            pseudo=PSEUDO)
det = (TS.paired_selectivity(long, MALIG, REACT, value="pct_expr", min_donors=MIN_DONORS,
                             pseudo=PSEUDO)
       .drop(columns="n_donors_paired")
       .rename(columns={"log2fc_paired": "log2fc_detect", "frac_donors_up": "frac_donors_up_detect",
                        "p": "p_detect", "q": "q_detect"}))


def _prev(arm, tag):
    return (TS.prevalence(long, arm, detect_floor=DETECT_FLOOR)
            .rename(columns={"median_pct_expr": f"{tag}_median_pct",
                             "mean_pct_expr": f"{tag}_mean_pct",
                             "median_mean_all": f"{tag}_median_mean",
                             "frac_donors_positive": f"{tag}_frac_donors_pos",
                             "n_donors": f"{tag}_n_donors"}))


def vs_reference(long, arm_pos, arm_ref, value="mean_all", min_donors=MIN_DONORS):
    """Unpaired: donor-level `value` in arm_pos vs arm_ref, Mann-Whitney, BH."""
    d = long[long["arm"].isin([arm_pos, arm_ref])]
    rows = []
    for gene, g in d.groupby("gene"):
        x = g.loc[g["arm"].eq(arm_pos), value].to_numpy()
        y = g.loc[g["arm"].eq(arm_ref), value].to_numpy()
        if len(x) < min_donors or len(y) < min_donors:
            continue
        p = (stats.mannwhitneyu(x, y, alternative="two-sided").pvalue
             if np.ptp(np.r_[x, y]) > 0 else 1.0)
        rows.append({"gene": gene,
                     "log2fc_vs_hc": np.log2((np.median(x) + PSEUDO) / (np.median(y) + PSEUDO)),
                     "p_vs_hc": p})
    out = pd.DataFrame(rows)
    out["q_vs_hc"] = multipletests(out["p_vs_hc"], method="fdr_bh")[1]
    return out


vs_hc = vs_reference(long, MALIG, HC)

ann = ST.annotate(GENES)[["gene", "cd_name", "agent", "stage", "positive_control",
                          "loss_marker", "surface_caveat"]]
surf_cols = ["gene", "surface_confident", "n_evidence", "n_tm", "topology", "surfy_class"]

res = (gene_tab
       .merge(sel, on="gene", how="left")
       .merge(det, on="gene", how="left")
       .merge(_prev(MALIG, "malig"), on="gene", how="left")
       .merge(_prev(REACT, "react"), on="gene", how="left")
       .merge(_prev(HC, "hc"), on="gene", how="left")
       .merge(vs_hc, on="gene", how="left")
       .merge(ann, on="gene", how="left")
       .merge(surf_all[surf_cols], on="gene", how="left"))
res["surface_confident"] = res["surface_confident"].fillna(False).astype(bool)
res["n_evidence"] = res["n_evidence"].fillna(0).astype(int)

_s = res["gene_set"].eq("surface") & res["log2fc_paired"].notna()
res["rank_surface"] = np.nan
res.loc[_s, "rank_surface"] = res.loc[_s, "log2fc_paired"].rank(ascending=False, method="min")
res = res.sort_values("log2fc_paired", ascending=False).reset_index(drop=True)

ALL_COLS = ["gene", "gene_set", "source", "is_lineage_control", "cd_name", "agent", "stage",
            "positive_control", "loss_marker", "rank_surface",
            "n_donors_paired", "log2fc_paired", "frac_donors_up", "p", "q",
            "log2fc_detect", "frac_donors_up_detect", "p_detect", "q_detect",
            "malig_n_donors", "malig_median_pct", "malig_mean_pct", "malig_frac_donors_pos",
            "malig_median_mean",
            "react_n_donors", "react_median_pct", "react_mean_pct", "react_frac_donors_pos",
            "react_median_mean",
            "hc_n_donors", "hc_median_pct", "hc_mean_pct", "hc_frac_donors_pos", "hc_median_mean",
            "log2fc_vs_hc", "p_vs_hc", "q_vs_hc",
            "surface_confident", "n_evidence", "n_tm", "topology", "surfy_class",
            "surface_caveat"]
res = res[ALL_COLS]
F_ALL = save(res, "all_genes")

# the compact column set every printed gene table uses
PRINT_COLS = ["gene", "cd_name", "rank_surface", "n_donors_paired", "log2fc_paired",
              "frac_donors_up", "q", "log2fc_detect", "q_detect", "malig_median_pct",
              "react_median_pct", "hc_median_pct", "malig_frac_donors_pos", "log2fc_vs_hc",
              "q_vs_hc", "surface_confident", "n_tm"]

surf = res[res["gene_set"].eq("surface")]
tested = surf["q"].notna()
report("§3a Counts",
       "how many surface genes are selectively up or down on malignant vs same-patient reactive "
       "CD4.",
       "paired Wilcoxon on mean_all, BH over every tested gene (surface + controls).",
       {"genes_tested_all_sets": int(res["q"].notna().sum()),
        "surface_genes_tested": int(tested.sum()),
        "surface_q<0.05_up": int(((surf["q"] < 0.05) & (surf["log2fc_paired"] > 0)).sum()),
        "surface_q<0.05_down": int(((surf["q"] < 0.05) & (surf["log2fc_paired"] < 0)).sum()),
        "surface_q<0.05_up_and_log2fc>0.5": int(((surf["q"] < 0.05)
                                                  & (surf["log2fc_paired"] > 0.5)).sum()),
        "surface_q<0.05_down_and_log2fc<-0.5": int(((surf["q"] < 0.05)
                                                     & (surf["log2fc_paired"] < -0.5)).sum()),
        "median_paired_donors_per_gene": float(surf["n_donors_paired"].median())},
       file=F_ALL)

up = surf[(surf["q"] < 0.05) & (surf["log2fc_paired"] > 0)].sort_values("log2fc_paired",
                                                                         ascending=False)
report("§3b Surface genes UP on malignant CD4 (q < 0.05)",
       "every surface gene whose expression is higher on malignant than on same-patient reactive "
       "CD4, sorted by effect.",
       "paired Wilcoxon on mean_all, BH; no effect-size or prevalence gate (see §5 for that).",
       up[PRINT_COLS], file=F_ALL)

down = surf[(surf["q"] < 0.05) & (surf["log2fc_paired"] < 0)].sort_values("log2fc_paired")
report("§3c Surface genes DOWN on malignant CD4 (q < 0.05)",
       "every surface gene whose expression is lower on malignant than on same-patient reactive "
       "CD4, sorted by effect (most lost first). These are candidates for loss-of-marker "
       "diagnostics, not for targeting.",
       "paired Wilcoxon on mean_all, BH.",
       down[PRINT_COLS], file=F_ALL)

lin = res[res["is_lineage_control"]].sort_values("log2fc_paired", ascending=False)
report("§3d Lineage controls (contamination reference)",
       "the same statistics for transcripts a CD4 T cell does not make. If these are up on "
       "malignant CD4, part of any malignant-up signal can be contamination of that size.",
       "same tests as §3b. 42 §2 turns this reference into formal per-gene artifact flags; this "
       "notebook only reports it.",
       lin[PRINT_COLS])

# sanity: 42 ran the same function on the same table
_t42 = ST.TAB_DIR / "tgt42_candidates_skin.csv"
if _t42.exists():
    _o = (pd.read_csv(_t42, usecols=["gene", "log2fc_paired"])
          .merge(res[["gene", "log2fc_paired"]], on="gene", suffixes=("_42", "_49")).dropna())
    print(f"sanity vs 42: {len(_o):,} shared genes, max |log2fc_paired(42) - log2fc_paired(49)| "
          f"= {(_o['log2fc_paired_42'] - _o['log2fc_paired_49']).abs().max():.2e}  "
          "(expected ~0: same input, same estimator)")
else:
    print(f"{_t42.name} not found -- sanity comparison against 42 skipped")
```

```
================================================================================================
§3a Counts
================================================================================================
WHAT: how many surface genes are selectively up or down on malignant vs same-patient reactive CD4.
HOW:  paired Wilcoxon on mean_all, BH over every tested gene (surface + controls).
FILE: tables/tgt49_all_genes.csv
[7 rows]

key,value
genes_tested_all_sets,2355
surface_genes_tested,2115
surface_q<0.05_up,388
surface_q<0.05_down,213
surface_q<0.05_up_and_log2fc>0.5,33
surface_q<0.05_down_and_log2fc<-0.5,62
median_paired_donors_per_gene,75.0

================================================================================================
§3b Surface genes UP on malignant CD4 (q < 0.05)
================================================================================================
WHAT: every surface gene whose expression is higher on malignant than on same-patient reactive CD4, sorted by effect.
HOW:  paired Wilcoxon on mean_all, BH; no effect-size or prevalence gate (see §5 for that).
FILE: tables/tgt49_all_genes.csv
[388 rows]

gene,cd_name,rank_surface,n_donors_paired,log2fc_paired,frac_donors_up,q,log2fc_detect,q_detect,malig_median_pct,react_median_pct,hc_median_pct,malig_frac_donors_pos,log2fc_vs_hc,q_vs_hc,surface_confident,n_tm
CD70,CD70,1,75,1.1,0.92,3.09e-10,1.22,4.71e-11,0.21,0.0498,0.024,0.808,1.91,1.05e-07,True,1.0
CD9,,2,75,1.08,0.733,7.09e-07,1.12,1.55e-08,0.25,0.0583,0.107,0.718,0.771,0.0172,True,4.0
CPM,,3,75,1.02,0.747,7.08e-08,0.954,6.44e-09,0.137,0.0248,0.00669,0.577,1.89,4.34e-06,True,0.0
MRC2,,4,75,0.892,0.88,2.51e-09,0.896,1.78e-10,0.0805,0.0206,0.0087,0.449,1.28,8.37e-07,True,1.0
SLC29A1,,5,75,0.841,0.933,1.61e-10,0.899,4.71e-11,0.0825,0.0198,0.0125,0.397,0.987,1.05e-07,True,11.0
COL6A3,,6,75,0.811,0.88,3.28e-09,0.843,1.5e-10,0.113,0.0247,0.0158,0.564,1.47,2.98e-06,True,
CD151,,7,75,0.794,0.76,9.39e-08,0.946,1.77e-10,0.196,0.0889,0.0612,0.795,1.04,4.47e-05,True,4.0
ADGRE1,,8,75,0.792,0.707,7.67e-07,0.704,3.75e-08,0.0487,0.00961,0,0.308,1.15,5.69e-07,True,7.0
GRN,,9,75,0.778,0.907,9.81e-10,0.835,6.51e-11,0.211,0.103,0.041,0.846,1.58,2.29e-07,True,
NRN1,,10,75,0.776,0.773,1.35e-08,0.902,1.76e-09,0.0461,0.00362,0.00472,0.282,0.793,0.00053,True,0.0
TFRC,,11,75,0.731,0.893,1.67e-09,0.806,4.71e-11,0.296,0.147,0.118,0.949,0.806,4.47e-05,True,
SLC39A6,,12,75,0.707,0.947,1.91e-10,0.897,4.71e-11,0.183,0.0789,0.0447,0.91,1.2,2.11e-07,True,6.0
SLC4A2,,13,75,0.698,0.933,1.91e-10,0.739,4.71e-11,0.13,0.0406,0.0156,0.641,1.51,1.05e-07,True,
TNFRSF25,,14,75,0.698,0.747,9.14e-05,0.821,1.79e-07,0.339,0.171,0.0943,0.872,1.31,2.13e-06,True,1.0
SLC1A4,,15,75,0.69,0.747,5.23e-05,0.682,1.64e-07,0.144,0.0615,0.0239,0.628,1.38,2.59e-06,True,9.0
LGALS3,,16,75,0.663,0.587,0.00115,0.631,3.82e-05,0.427,0.292,0.361,0.885,0.21,0.184,True,
CD82,,17,75,0.632,0.773,1.41e-07,0.714,2.06e-09,0.477,0.283,0.176,1,1.04,1.32e-05,True,4.0
EVC2,,18,75,0.624,0.853,1.09e-08,0.657,1.57e-10,0.0536,0.0132,0,0.231,1.1,2.11e-07,True,1.0
SLC5A3,,19,75,0.617,0.893,1.81e-09,0.719,4.71e-11,0.217,0.0912,0.0499,0.769,1.45,7.22e-07,True,12.0
SGCB,,20,75,0.615,0.867,2.3e-08,0.727,2.05e-10,0.0735,0.0285,0.0204,0.372,0.76,2.42e-06,True,1.0
IL6R,,21,75,0.614,0.72,2.31e-05,0.778,1.51e-08,0.233,0.107,0.0726,0.769,0.857,0.000122,True,1.0
SYPL1,,22,75,0.599,0.907,1.15e-09,0.807,5.78e-11,0.362,0.2,0.179,0.949,0.674,0.000435,True,4.0
CALU,,23,75,0.587,0.88,6.18e-10,0.801,4.71e-11,0.19,0.0944,0.0392,0.846,1.26,5.31e-07,True,
QSOX1,,24,75,0.584,0.853,7.51e-09,0.714,1.06e-10,0.133,0.0677,0.0261,0.692,1.23,2.32e-07,True,1.0
VDAC1,,25,75,0.584,0.947,1.91e-10,0.645,4.71e-11,0.482,0.293,0.175,0.987,1.28,5.15e-07,True,
TSPAN3,,26,75,0.574,0.84,3.6e-09,0.717,1.1e-10,0.0966,0.0338,0.00943,0.462,1.28,5.69e-07,True,4.0
SLC39A14,,27,75,0.566,0.853,6.76e-09,0.717,2.59e-10,0.057,0.0178,0.0114,0.282,0.66,0.000261,True,6.0
TNFRSF4,,28,75,0.542,0.667,0.00308,0.538,1.81e-05,0.349,0.223,0.
... [59931 more chars truncated]
```

## §4 — Calibration: the antigens whose answer is known

Read this before any novel hit. **CCR4** (mogamulizumab) and **TNFRSF8/CD30** (brentuximab vedotin)
are approved in CTCL, **KIR3DL2** is lacutamab's target, and **IL2RA/CD25** is registrational. A
screen that measures malignant selectivity should rank them near the top of the surface genes.
**CD7** and **DPP4/CD26** loss is the textbook CTCL immunophenotype, so they should come out
negative. CD5, CD6, CD27 and CD2 are reported beside them.

`rank_surface` is the rank of `log2fc_paired` among tested surface genes, and `rank_pct` is that
rank divided by the number of tested surface genes. The v1 skin screen put its worst control at
1,968 of 2,115, which is what prompted the 42 rebuild.

```python  # cell 9
# ============================================================================
# §4  Calibration
# ============================================================================
N_SURF_TESTED = int(res.loc[res["gene_set"].eq("surface"), "log2fc_paired"].notna().sum())
CAL_COLS = ["gene", "cd_name", "agent", "stage", "rank_surface", "rank_pct", "n_donors_paired",
            "log2fc_paired", "frac_donors_up", "q", "log2fc_detect", "frac_donors_up_detect",
            "q_detect", "malig_median_pct", "react_median_pct", "hc_median_pct",
            "malig_frac_donors_pos", "log2fc_vs_hc", "q_vs_hc"]

cal = res.assign(rank_pct=res["rank_surface"] / N_SURF_TESTED)

pos = cal[cal["gene"].isin(ST.POSITIVE_CONTROLS)].sort_values("rank_surface")
report("§4a Positive controls",
       "where the four clinically validated CTCL antigens land in the ranking.",
       f"rank of log2fc_paired among {N_SURF_TESTED:,} tested surface genes; rank_pct = rank / "
       f"{N_SURF_TESTED:,}. Pass = all four near the top (well inside the top ~10%).",
       pos[CAL_COLS])

loss = cal[cal["gene"].isin(ST.LOSS_MARKERS)].sort_values("log2fc_paired")
report("§4b Loss markers",
       "the CTCL loss phenotype (CD7, DPP4/CD26, CD5, CD6, CD27, CD2) on malignant vs reactive CD4.",
       "same paired test; CD7 and DPP4 should be negative with a small q. A missing gene is "
       "absent from the tables (CD26 is DPP4).",
       loss[CAL_COLS])
```

```
================================================================================================
§4a Positive controls
================================================================================================
WHAT: where the four clinically validated CTCL antigens land in the ranking.
HOW:  rank of log2fc_paired among 2,115 tested surface genes; rank_pct = rank / 2,115. Pass = all four near the top (well inside the top ~10%).
[4 rows]

gene,cd_name,agent,stage,rank_surface,rank_pct,n_donors_paired,log2fc_paired,frac_donors_up,q,log2fc_detect,frac_donors_up_detect,q_detect,malig_median_pct,react_median_pct,hc_median_pct,malig_frac_donors_pos,log2fc_vs_hc,q_vs_hc
TNFRSF8,CD30,brentuximab vedotin; AFM13 (CD30xCD16A),approved,99,0.0468,75,0.332,0.667,0.000215,0.455,0.747,4.36e-07,0.038,0.0111,0,0.192,0.86,1.05e-07
CCR4,CCR4,mogamulizumab,approved,101,0.0478,75,0.324,0.56,0.0909,0.568,0.613,0.00126,0.219,0.135,0.0656,0.705,1.12,0.000897
KIR3DL2,CD158k,lacutamab (IPH4102),phase 2,522,0.247,75,0,0.36,0.000947,0,0.36,0.000438,0,0.00372,0.00256,0.321,-0.108,0.678
IL2RA,CD25,denileukin diftitox-cxdl (Lymphir); camidanlumab tesirine,approved (2024-08),1956,0.925,75,-0.172,0.387,0.454,0.0171,0.507,0.214,0.122,0.126,0.0828,0.628,0.16,0.153

================================================================================================
§4b Loss markers
================================================================================================
WHAT: the CTCL loss phenotype (CD7, DPP4/CD26, CD5, CD6, CD27, CD2) on malignant vs reactive CD4.
HOW:  same paired test; CD7 and DPP4 should be negative with a small q. A missing gene is absent from the tables (CD26 is DPP4).
[6 rows]

gene,cd_name,agent,stage,rank_surface,rank_pct,n_donors_paired,log2fc_paired,frac_donors_up,q,log2fc_detect,frac_donors_up_detect,q_detect,malig_median_pct,react_median_pct,hc_median_pct,malig_frac_donors_pos,log2fc_vs_hc,q_vs_hc
CD7,CD7,WU-CART-007; BEAM-201 (base-edited),phase 1/2,2115,1,75,-2.85,0.0933,3.6e-09,-2.28,0.133,2.59e-10,0.038,0.539,0.211,0.308,-2.06,0.0526
DPP4,,,,2090,0.988,75,-0.841,0.253,0.00391,-0.575,0.307,0.0257,0.0087,0.0457,0.0492,0.179,-1.15,0.267
CD27,,,,2022,0.956,75,-0.359,0.467,0.792,-0.013,0.493,0.315,0.246,0.278,0.0816,0.59,1.15,0.0218
CD2,,,,1931,0.913,75,-0.146,0.453,0.0478,0.0676,0.533,1,0.6,0.583,0.392,0.897,0.278,0.00618
CD5,CD5,CD5 CAR-T; MT-101 (CAR-monocyte),phase 1/2,1854,0.877,75,-0.0584,0.493,0.816,0.173,0.587,0.177,0.31,0.262,0.104,0.744,1.09,0.00268
CD6,,,,327,0.155,75,0.0957,0.56,0.344,0.323,0.667,0.00331,0.395,0.349,0.183,0.885,0.982,0.00164
```

## §5 — Shortlist

A gene is shortlisted if it passes **all** of these, in this order:

1. it is a **surface** gene (panel ∪ confident surfaceome);
2. it is not in the lineage-control null and not a loss marker;
3. `q` < 0.05 on the paired malignant-vs-reactive test;
4. `log2fc_paired` > 0.5;
5. `malig_frac_donors_pos` ≥ 0.5, i.e. ≥ 10 % of malignant CD4 cells detect it in at least half of
   the patients.

The funnel table prints how many genes survive each gate.

**No artifact filter is run here.** 42 adds `artifact_flag == "clean"`, but on the current build
(2026-10-26) 42's contamination null was not calibrated (the lineage controls are not up in the
malignant arm), so 42 excluded nothing. This shortlist therefore matches 42's gates. That does
**not** clear the genes 42 singled out as implausible surface targets (COL6A3, LGALS3, GRN, QSOX1,
CALU, VDAC1, GPI, SYPL1): their objection is protein localisation, which no statistic here can
settle. Surface status is a database annotation, not a measurement.

```python  # cell 11
# ============================================================================
# §5  Shortlist
# ============================================================================
GATES = [
    ("surface gene", res["gene_set"].eq("surface")),
    ("not lineage control, not loss marker",
     ~res["is_lineage_control"] & ~res["loss_marker"].astype(bool)),
    (f"q < {SHORT_Q}", res["q"] < SHORT_Q),
    (f"log2fc_paired > {SHORT_LFC}", res["log2fc_paired"] > SHORT_LFC),
    (f"malig_frac_donors_pos >= {SHORT_PREV}", res["malig_frac_donors_pos"] >= SHORT_PREV),
]
mask = pd.Series(True, index=res.index)
funnel = [{"gate": "all genes in the tables", "n_genes_remaining": len(res)}]
for name, m in GATES:
    mask &= m.fillna(False)
    funnel.append({"gate": name, "n_genes_remaining": int(mask.sum())})
report("§5a Shortlist funnel",
       "how many genes survive each gate, applied cumulatively in the order listed.",
       "gates as stated in the markdown above.",
       pd.DataFrame(funnel))

short = res[mask].sort_values("log2fc_paired", ascending=False).reset_index(drop=True)
short["disputed_in_42"] = short["gene"].isin(TQ.SUSPECTS)
SHORT_COLS = PRINT_COLS + ["agent", "stage", "hc_frac_donors_pos", "react_frac_donors_pos",
                           "n_evidence", "topology", "surfy_class", "source", "surface_caveat",
                           "disputed_in_42"]
F_SHORT = save(short, "shortlist")
report("§5b Shortlist",
       "surface genes selectively up on malignant CD4, present in at least half of patients.",
       "all five gates; sorted by log2fc_paired. hc_* columns = expression on healthy-skin CD4, "
       "i.e. the T-cell on-target off-tumour cost. disputed_in_42 = one of the genes 42 names as "
       "implausible surface targets (cytoplasmic, ER, mitochondrial, secreted or stromal "
       "protein) despite passing the statistics.",
       short[SHORT_COLS], file=F_SHORT)

novel = short[short["stage"].fillna("").eq("")]
report("§5c Shortlist genes with no clinical precedent",
       "the shortlist minus every gene with an entry in surface_targets.CLINICAL_PRECEDENT.",
       "stage column empty = no approved, trial or preclinical record curated in this project; "
       "that is a statement about this project's curation, not a literature search.",
       novel[SHORT_COLS])
```

```
================================================================================================
§5a Shortlist funnel
================================================================================================
WHAT: how many genes survive each gate, applied cumulatively in the order listed.
HOW:  gates as stated in the markdown above.
[6 rows]

gate,n_genes_remaining
all genes in the tables,2355
surface gene,2115
"not lineage control, not loss marker",2093
q < 0.05,817
log2fc_paired > 0.5,33
malig_frac_donors_pos >= 0.5,23

================================================================================================
§5b Shortlist
================================================================================================
WHAT: surface genes selectively up on malignant CD4, present in at least half of patients.
HOW:  all five gates; sorted by log2fc_paired. hc_* columns = expression on healthy-skin CD4, i.e. the T-cell on-target off-tumour cost. disputed_in_42 = one of the genes 42 names as implausible surface targets (cytoplasmic, ER, mitochondrial, secreted or stromal protein) despite passing the statistics.
FILE: tables/tgt49_shortlist.csv
[23 rows]

gene,cd_name,rank_surface,n_donors_paired,log2fc_paired,frac_donors_up,q,log2fc_detect,q_detect,malig_median_pct,react_median_pct,hc_median_pct,malig_frac_donors_pos,log2fc_vs_hc,q_vs_hc,surface_confident,n_tm,agent,stage,hc_frac_donors_pos,react_frac_donors_pos,n_evidence,topology,surfy_class,source,surface_caveat,disputed_in_42
CD70,CD70,1,75,1.1,0.92,3.09e-10,1.22,4.71e-11,0.21,0.0498,0.024,0.808,1.91,1.05e-07,True,1.0,INCA036873 (CD70xCD3); CTX130/CTX131,phase 1,0,0.195,2,CY:1-20;TM:21-42;NC:43-193,Unclassified,both,,False
CD9,,2,75,1.08,0.733,7.09e-07,1.12,1.55e-08,0.25,0.0583,0.107,0.718,0.771,0.0172,True,4.0,,,0.524,0.329,3,CY:1-11;TM:12-35;NC:36-54;TM:55-78;CY:79-88;TM:89-111;NC:112-194;TM:195-220;CY:221-228,Miscellaneous,surfaceome,,False
CPM,,3,75,1.02,0.747,7.08e-08,0.954,6.44e-09,0.137,0.0248,0.00669,0.577,1.89,4.34e-06,True,0.0,,,0,0.0732,3,SP:1-17;NC:18-443,Enzymes,surfaceome,,False
COL6A3,,6,75,0.811,0.88,3.28e-09,0.843,1.5e-10,0.113,0.0247,0.0158,0.564,1.47,2.98e-06,True,,,,0,0.134,1,,,surfaceome,,True
CD151,,7,75,0.794,0.76,9.39e-08,0.946,1.77e-10,0.196,0.0889,0.0612,0.795,1.04,4.47e-05,True,4.0,,,0.19,0.415,3,CY:1-20;TM:21-43;NC:44-57;TM:58-81;CY:82-91;TM:92-115;NC:116-221;TM:222-245;CY:246-253,Miscellaneous,surfaceome,,False
GRN,,9,75,0.778,0.907,9.81e-10,0.835,6.51e-11,0.211,0.103,0.041,0.846,1.58,2.29e-07,True,,,,0.0952,0.524,1,,,surfaceome,,True
TFRC,,11,75,0.731,0.893,1.67e-09,0.806,4.71e-11,0.296,0.147,0.118,0.949,0.806,4.47e-05,True,,,,0.571,0.659,2,,,both,,False
SLC39A6,,12,75,0.707,0.947,1.91e-10,0.897,4.71e-11,0.183,0.0789,0.0447,0.91,1.2,2.11e-07,True,6.0,,,0.0476,0.256,4,SP:1-26;NC:27-322;TM:323-346;CY:347-353;TM:354-377;NC:378-424;TM:425-444;CY:445-657;TM:658-679;NC:680-684;TM:685-703;CY:704-723;TM:724-745;NC:746-755,Transporters,surfaceome,,False
SLC4A2,,13,75,0.698,0.933,1.91e-10,0.739,4.71e-11,0.13,0.0406,0.0156,0.641,1.51,1.05e-07,True,,,,0,0.134,2,,,surfaceome,,False
TNFRSF25,,14,75,0.698,0.747,9.14e-05,0.821,1.79e-07,0.339,0.171,0.0943,0.872,1.31,2.13e-06,True,1.0,,,0.333,0.768,3,SP:1-24;NC:25-200;TM:201-223;CY:224-417,Receptors,surfaceome,,False
SLC1A4,,15,75,0.69,0.747,5.23e-05,0.682,1.64e-07,0.144,0.0615,0.0239,0.628,1.38,2.59e-06,True,9.0,,,0.0476,0.317,3,CY:1-42;TM:43-65;NC:66-84;TM:85-105;CY:106-116;TM:117-139;NC:140-219;TM:220-237;CY:238-257;TM:258-277;NC:278-296;TM:297-320;CY:321-327;TM:328-349;NC:350-368;TM:369-394;CY:395-414;TM:415-438;NC:439-532,Transporters,surfaceome,,False
LGALS3,,16,75,0.663,0.587,0.00115,0.631,3.82e-05,0.427,0.292,0.361,0.885,0.21,0.184,True,,,,0.81,0.963,2,,,surfaceome,,True
CD82,,17,75,0.632,0.773,1.41e-07,0.714,2.06e-09,0.477,0.283,0.176,1,1.04,1.32e-05,True,4.0,,,0.667,0.988,2,CY:1-11;TM:12-36;NC:37-55;TM:56-75;CY:76-84;TM:85-108;NC:109-228;TM:229-253;CY:254-267,Miscellaneous,both,,False
SLC5A3
... [7085 more chars truncated]
```

## §6 — Per-donor detail

A cohort median hides what decides whether an antigen is usable: what it does **inside each
patient**. **Detail genes** = the top 30 shortlist genes by `log2fc_paired`, plus the positive
controls and the loss markers. All tables use the **paired donors**, ordered by `burden` (low →
high), and each row carries the donor's study, diagnosis, lesion, stage and burden.

| table | cell value |
|---|---|
| §6a | per-donor log2((malig `mean_all` + 0.05) / (react `mean_all` + 0.05)). The headline `log2fc_paired` is the median of each column. |
| §6b | malignant-arm `pct_expr`: share of the donor's malignant CD4 cells detecting the gene |
| §6c | reactive-arm `pct_expr`, same donors |
| §6d | healthy-skin CD4 `pct_expr`, one row per healthy donor (a different set of people) |
| §6e | per-gene summary of §6a: n, median, IQR, min, max, `frac_up` (lfc > 0), `frac_up_1` (lfc > 1, roughly doubled) |
| §6f | eligibility: share of paired donors whose malignant CD4 is ≥ 10 / 25 / 50 / 75 % positive |

`frac_up_1` and eligibility separate an antigen that is up in every patient from one that is
strong in a subset. A subset antigen is a biomarker-selected product, not a failed one.

```python  # cell 13
# ============================================================================
# §6  Per-donor detail
# ============================================================================
_present = set(res["gene"])
DETAIL = list(dict.fromkeys(short["gene"].head(N_DETAIL).tolist()
                            + [g for g in ST.POSITIVE_CONTROLS + ST.LOSS_MARKERS
                               if g in _present]))
ROLE = {g: ("positive_control" if g in ST.POSITIVE_CONTROLS
            else "loss_marker" if g in ST.LOSS_MARKERS else "shortlist") for g in DETAIL}

DON_P = don.loc[PAIRED].sort_values("burden")
ANN = ["study", "dx2", "lesion", "clin_stage_class", "burden", "n_malignant", "n_reactive"]


def _donor_table(M, donors, ann_cols):
    """Donor rows x detail-gene columns, donor annotation in front."""
    M = M.reindex(index=donors.index, columns=[g for g in DETAIL if g in M.columns])
    return donors[ann_cols].join(M).reset_index()


def _arm_pct(arm):
    return (long[long["arm"].eq(arm) & long["gene"].isin(DETAIL)]
            .pivot_table(index="donor", columns="gene", values="pct_expr"))


print(f"detail genes ({len(DETAIL)}):", ", ".join(f"{g} [{ROLE[g]}]" for g in DETAIL), "\n")

lfc = TH.lfc_matrix(long, MALIG, REACT, genes=DETAIL, pseudo=PSEUDO, min_donors=MIN_DONORS).T
t = _donor_table(lfc, DON_P, ANN)
report("§6a Per-donor log2FC, malignant vs reactive CD4",
       "the paired effect inside every patient, for each detail gene.",
       "per donor: log2((mean_all_malig + 0.05) / (mean_all_react + 0.05)), mean_all = mean log1p "
       "CP10K over the arm's cells; donors ordered by burden, low to high. NaN = gene not tested "
       "in that donor.",
       t, file=save(t, "donor_lfc"))

t = _donor_table(_arm_pct(MALIG), DON_P, ANN)
report("§6b Per-donor detection, malignant CD4",
       "share of each patient's malignant CD4 cells with >= 1 UMI of the gene.",
       "pct_expr of the CD4_malignant arm; paired donors, burden order.",
       t, file=save(t, "donor_pct_malignant"))

t = _donor_table(_arm_pct(REACT), DON_P, ANN)
report("§6c Per-donor detection, reactive CD4",
       "share of each patient's reactive (non-malignant) CD4 cells with >= 1 UMI of the gene.",
       "pct_expr of the CD4_reactive arm; same donors and order as §6b.",
       t, file=save(t, "donor_pct_reactive"))

_hc_d = long.loc[long["arm"].eq(HC), "donor"].unique()
DON_HC = don.loc[_hc_d].sort_values("study")
t = _donor_table(_arm_pct(HC), DON_HC, ["study", "dx", "lesion", "n_cells"])
report("§6d Per-donor detection, healthy-skin CD4",
       "share of each healthy donor's CD4 cells with >= 1 UMI of the gene: the T-cell on-target "
       "off-tumour cost.",
       "pct_expr of the CD4_HC arm (clin_dx == HC); n_cells = all T/NK cells of the donor in "
       "the object. Different people, studies and chemistries from the patients.",
       t, file=save(t, "donor_pct_healthy"))

L = lfc.reindex(columns=[g for g in DETAIL if g in lfc.columns])
_n = L.notna().sum()
summ = pd.DataFrame({
    "role": pd.Series(ROLE).reindex(L.columns), "n_donors": _n,
    "median": L.median(), "q25": L.quantile(.25), "q75": L.quantile(.75),
    "min": L.min(), "max": L.max(),
    "frac_up": (L > 0).sum() / _n, "frac_up_1": (L > 1).sum() / _n,
    "frac_down_1": (L < -1).sum() / _n}).reindex(L.columns)
summ.index.name = "gene"
summ = summ.join(res.set_index("gene")[["cd_name", "q"]]).reset_index()
report("§6e Per-gene summary of the donor effects",
       "how consistent each detail gene's effect is across patients.",
       "over the columns of §6a: median and quartiles of per-donor log2FC; frac_up = share of "
       "donors with log2FC > 0; frac_up_1 = log2FC > 1; frac_down_1 = log2FC < -1; q = the "
       "headline paired q from §3.",
       summ, file=save(summ, "detail_summary"))

THR = (0.10, 0.25, 0.50, 0.75)
elig = pd.concat([
    TH.eligibility(long, MALIG, thresholds=THR, genes=DETAIL, donors=PAIRED)
      .assign(arm=MALIG, donors="paired patients"),
    TH.eligibility(long, REACT, thresholds=THR, genes=DETAIL, donors=PAIRED)
      .assign(arm=REACT, donors="paired patients"),
    TH.eligibility(long, HC, thresholds=THR, genes=DETAIL)
      .assign(arm=HC, donors="healthy donors"),
])
elig["role"] = elig["gene"].map(ROLE)
elig["gene"] = pd.Categorical(elig["gene"], DETAIL, ordered=True)
elig = elig.sort_values(["gene", "arm"]).reset_index(drop=True)
elig["gene"] = elig["gene"].astype(str)
elig = elig[["gene", "role", "arm", "donors", "n_donors", "median_pct_expr"]
            + [f"frac_donors_ge{int(x * 100)}" for x in THR]]
report("§6f Eligibility",
       "in what share of donors at least X% of the arm's CD4 cells carry the transcript. The "
       "malignant rows are the recruitment curve; the reactive and healthy rows are the "
       "T-cell off-tumour curve.",
       "per donor, pct_expr of the arm >= threshold, averaged over donors; malignant and reactive "
       "rows use the paired donors, healthy rows all CD4_HC donors.",
       elig, file=save(elig, "eligibility"))
```

```
detail genes (33): CD70 [shortlist], CD9 [shortlist], CPM [shortlist], COL6A3 [shortlist], CD151 [shortlist], GRN [shortlist], TFRC [shortlist], SLC39A6 [shortlist], SLC4A2 [shortlist], TNFRSF25 [shortlist], SLC1A4 [shortlist], LGALS3 [shortlist], CD82 [shortlist], SLC5A3 [shortlist], IL6R [shortlist], SYPL1 [shortlist], CALU [shortlist], QSOX1 [shortlist], VDAC1 [shortlist], TNFRSF4 [shortlist], TSPAN2 [shortlist], GPI [shortlist], TMEM30A [shortlist], CCR4 [positive_control], TNFRSF8 [positive_control], KIR3DL2 [positive_control], IL2RA [positive_control], CD7 [loss_marker], DPP4 [loss_marker], CD5 [loss_marker], CD6 [loss_marker], CD27 [loss_marker], CD2 [loss_marker] 

================================================================================================
§6a Per-donor log2FC, malignant vs reactive CD4
================================================================================================
WHAT: the paired effect inside every patient, for each detail gene.
HOW:  per donor: log2((mean_all_malig + 0.05) / (mean_all_react + 0.05)), mean_all = mean log1p CP10K over the arm's cells; donors ordered by burden, low to high. NaN = gene not tested in that donor.
FILE: tables/tgt49_donor_lfc.csv
[75 rows]

donor,study,dx2,lesion,clin_stage_class,burden,n_malignant,n_reactive,CD70,CD9,CPM,COL6A3,CD151,GRN,TFRC,SLC39A6,SLC4A2,TNFRSF25,SLC1A4,LGALS3,CD82,SLC5A3,IL6R,SYPL1,CALU,QSOX1,VDAC1,TNFRSF4,TSPAN2,GPI,TMEM30A,CCR4,TNFRSF8,KIR3DL2,IL2RA,CD7,DPP4,CD5,CD6,CD27,CD2
Li2024_atlas__MF17,li2024,MF,tumor,advanced,0.00837,37,4385,0.958,0,0.842,0.279,0.167,0.32,-0.207,0.397,0.529,-0.309,1.45,0.372,0.0137,-0.603,-0.553,0.141,0.617,0.744,1.59,-1.61,0.228,0.837,0.16,-0.852,1.61,0,-0.157,-2.23,-0.177,-0.206,-0.21,-0.515,-0.277
Li2024_atlas__CTCL7,li2024,MF,mixed,early,0.0869,3658,38432,1.66,4.83,0.45,0.246,1.79,0.75,-0.604,2.66,0.697,-1.75,2.21,1.85,1.62,0.743,-1.29,0.0516,-0.303,1.25,0.282,-0.629,-0.886,0.00171,0.661,-2.78,0.477,0,0.114,-3.12,0.194,-0.958,-0.761,-1.57,0.995
Li2024_atlas__MF27,li2024,MF,tumor,advanced,0.129,107,725,3.23,0.864,0.448,1.91,1.29,1.3,0.704,0.702,1.44,-1.34,1.41,1.2,0.78,0.479,-0.467,0.134,0.619,0.756,1.11,0.0975,0.789,0.825,0.147,-0.165,1.92,0,0.57,-1.01,-1.25,0.0552,-0.0978,0.583,0.143
Li2024_atlas__CTCL4,li2024,MF,mixed,early,0.136,1241,7906,2.64,3.3,0.539,0.194,0.077,0.679,0.824,2.25,0.435,0.0293,2.66,1.65,0.595,1.5,-0.338,0.346,0.376,0.56,0.255,-1.32,0.339,0.29,0.206,-1.2,1.08,0,-0.331,-2.01,-0.484,-0.723,0.763,-1.53,0.378
D15__MF28,il4ra2026,MF,tumor,advanced,0.151,123,691,2.41,2.73,1.68,0.706,-0.0346,-0.502,0.672,-0.293,0.264,-1.01,0.842,1.34,-0.721,1.14,1.08,0.192,0.478,0.761,0.00329,-0.914,0.156,-0.523,0.448,-0.113,0.503,-0.296,0.32,-2.85,-0.232,-2.01,-2.26,-1.11,-0.859
D10__JONAK1,jonak2021,MF,plaque,early,0.153,348,1920,0.841,1.34,1.39,1.58,1.31,0.499,1.07,0.847,2.06,1.47,0.135,-0.693,1.24,0.846,0.843,0.493,0.833,0.307,0.959,1.57,1.66,0.912,0.506,0.434,0.124,2.33,-0.427,-2.76,-0.865,1.53,1.47,0.576,-0.321
D13__MF21,gaydosik2022,MF,tumor,advanced,0.177,118,547,1.16,-0.854,3.14,-0.284,1.31,0.923,1.66,1.07,2.34,0.384,2.25,-0.714,0.619,-0.0928,-0.69,2.6,2.06,1.38,1.6,-0.76,0.152,1.67,0.0151,2.7,-0.245,3.33,1.33,-2.48,-0.755,0.875,-0.794,1.78,-0.146
D3__P105,brunner2024,MF,erythroderma,advanced,0.185,117,516,0.485,-0.163,1.2,0.156,-0.385,1.11,-0.887,-0.216,0.439,0.209,-0.835,-0.631,0.566,0.738,0.413,0.515,-0.537,-0.326,0.185,0.616,2.18,-0.896,0.576,1.52,-1.22,1.32,-0.172,-2.6,-0.337,0.202,-0.312,0.121,-0.436
D15__MF36,il4ra2026,MF,mixed,advanced,0.221,620,2182,0.105,2.8,1.59,1.62,1.17,0.434,0.341,0.938,0.412,0.754,-0.17,-0.393,0.768,0.661,0.329,0.844,0.444,0.584,0.895,-0.385,1.45,0.787,0.536,-0.113,-0.0519,3.2,-1.3,-2.65,-0.826,-0.491,0.308,-0.504,0.0733
D8__P90,alkon2024,MF,patch,early,0.227,68,232,1.18,1.58,-0.253,0.529,-0.535,-1.03,-0.0237,0.387,0.345,0.0701,1.66,1.44,-0.145,0.36,-1.56,0.17,-0.0641,-0.0589,0.485,-3.13,0.53,0.337,0.653,-2.3,0.983,-0.233,-0.0703,-3.15,1.99,-2.6,0.23,-3.4,-3.31

... [73332 more chars truncated]
```

## §7 — Stratification: lesion, stage, disease

The §3 paired test, rerun **inside each stratum** of the paired donors
(`target_screen.strat_selectivity`, the same estimator 47 uses). Three splits:

| split | strata | source |
|---|---|---|
| lesion | `patch_plaque` (patch, plaque) vs `tumour_eryth` (tumour, erythroderma) | `clin_lesion`. Mixed, papule and unspecified are excluded. |
| stage | `early` vs `advanced` | `clin_stage_class`. Missing stage is excluded. |
| disease | `MF` vs `SS` | `dx2`. MF/SS-unresolved is excluded. |

Per stratum: `log2fc_paired`, `n_donors_paired`, `frac_donors_up`, and a Wilcoxon `p` when the
stratum has ≥ 3 paired donors. `q` is BH over every (gene, stratum) row of that split.

**Interaction** (does the effect differ between the two strata?): a Mann–Whitney of the per-donor
log2FC between the two strata, giving `delta_lfc` (median of the first stratum named in `strata`
minus median of the second) and `p_interaction` / `q_interaction`, BH over genes. This is a
two-sample comparison of effect sizes, not a model.

**Read the n columns.** SS has few skin donors, so a non-significant interaction there is
mostly no power, not evidence of sameness. Lesion type and burden are confounded: tumours carry
more clone. The per-stratum burden summary prints first for that reason.

Printed in full: the detail genes, and every surface gene with `q_interaction` < 0.05. The full
tables are in `tables/tgt49_strat_{lesion,stage,disease}.csv`.

```python  # cell 15
# ============================================================================
# §7  Stratification: lesion, stage, disease
# ============================================================================
DONS = don.loc[PAIRED]
_les = {"patch": "patch_plaque", "plaque": "patch_plaque",
        "tumor": "tumour_eryth", "erythroderma": "tumour_eryth"}
SPLITS = {
    "lesion": DONS["lesion"].map(_les),
    "stage": DONS["clin_stage_class"].where(DONS["clin_stage_class"].isin(["early", "advanced"])),
    "disease": DONS["dx2"].where(DONS["dx2"].isin(["MF", "SS"])),
}
STRAT_KEEP = ["log2fc_paired", "n_donors_paired", "frac_donors_up", "p", "q"]
INTER = ["strata", "n_a", "n_b", "delta_lfc", "p_interaction", "q_interaction"]
STRAT = {}

for split, lab in SPLITS.items():
    lab = lab.dropna()
    counts = lab.value_counts()

    # who is in each stratum, and how much clone they carry
    bsum = DONS.loc[lab.index].groupby(lab)["burden"].describe(percentiles=[.25, .5, .75])
    bsum.index.name = "stratum"
    bsum = bsum.rename(columns={"count": "n_paired_donors"}).reset_index()
    _g = [DONS.loc[lab.index[lab.eq(s)], "burden"].dropna() for s in counts.index[:2]]
    p_b = (stats.mannwhitneyu(*_g, alternative="two-sided").pvalue
           if len(_g) == 2 and min(map(len, _g)) >= 3 else np.nan)
    bsum["mwu_p_burden_between_strata"] = p_b
    report(f"§7 {split}: strata",
           f"paired donors per {split} stratum and their malignant burden.",
           "burden = n_malignant / (n_malignant + n_reactive) CD4 cells; mwu_p = two-sided "
           "Mann-Whitney of burden between the two strata. A small p means this split and burden "
           "are partly the same comparison.",
           bsum)

    if len(counts) < 2 or counts.min() < STRAT_MIN_DONORS:
        print(f"{split}: fewer than {STRAT_MIN_DONORS} paired donors in a stratum "
              f"({counts.to_dict()}) -- not tested\n")
        continue

    s = TS.strat_selectivity(long, MALIG, REACT, lab, value="mean_all",
                             min_donors=STRAT_MIN_DONORS, pseudo=PSEUDO)
    w = s.pivot_table(index="gene", columns="stratum",
                      values=[c for c in STRAT_KEEP if c in s.columns], dropna=False)
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w[sorted(w.columns, key=lambda c: (c.rsplit("_", 1)[-1], c))]
    inter = [c for c in INTER if c in s.columns]
    if inter:
        w = w.join(s[["gene"] + inter].drop_duplicates("gene").set_index("gene"))
    w = (w.join(res.set_index("gene")[["gene_set", "log2fc_paired", "q"]]
                .rename(columns={"log2fc_paired": "log2fc_all", "q": "q_all"}))
          .reset_index())
    w["detail_gene"] = w["gene"].isin(DETAIL)
    STRAT[split] = w
    f = save(w, f"strat_{split}")

    det_rows = w[w["detail_gene"]].copy()
    det_rows["_o"] = det_rows["gene"].map({g: i for i, g in enumerate(DETAIL)})
    report(f"§7 {split}: detail genes per stratum",
           f"the malignant-vs-reactive effect of each detail gene inside each {split} stratum.",
           "strat_selectivity: paired Wilcoxon inside the stratum (p only if >= 3 donors; q = BH "
           "over all gene x stratum rows); delta_lfc / q_interaction = Mann-Whitney of per-donor "
           "log2FC between the two strata, BH over genes; log2fc_all / q_all = the §3 result on "
           "all paired donors.",
           det_rows.sort_values("_o").drop(columns="_o"), file=f)

    if "q_interaction" in w:
        sig = w[(w["q_interaction"] < 0.05) & w["gene_set"].eq("surface")].sort_values(
            "delta_lfc")
        report(f"§7 {split}: surface genes whose effect differs between strata "
               "(q_interaction < 0.05)",
               f"every surface gene whose malignant-vs-reactive effect depends on {split}.",
               "Mann-Whitney of per-donor log2FC between the two strata, BH over genes; sorted "
               "by delta_lfc (first stratum in `strata` minus second).",
               sig, file=f)
```

```
================================================================================================
§7 lesion: strata
================================================================================================
WHAT: paired donors per lesion stratum and their malignant burden.
HOW:  burden = n_malignant / (n_malignant + n_reactive) CD4 cells; mwu_p = two-sided Mann-Whitney of burden between the two strata. A small p means this split and burden are partly the same comparison.
[2 rows]

stratum,n_paired_donors,mean,std,min,25%,50%,75%,max,mwu_p_burden_between_strata
patch_plaque,19,0.544,0.234,0.153,0.329,0.566,0.752,0.883,0.394
tumour_eryth,30,0.589,0.291,0.00837,0.359,0.624,0.827,0.985,0.394


================================================================================================
§7 lesion: detail genes per stratum
================================================================================================
WHAT: the malignant-vs-reactive effect of each detail gene inside each lesion stratum.
HOW:  strat_selectivity: paired Wilcoxon inside the stratum (p only if >= 3 donors; q = BH over all gene x stratum rows); delta_lfc / q_interaction = Mann-Whitney of per-donor log2FC between the two strata, BH over genes; log2fc_all / q_all = the §3 result on all paired donors.
FILE: tables/tgt49_strat_lesion.csv
[33 rows]

gene,frac_donors_up_tumour_eryth,log2fc_paired_tumour_eryth,n_donors_paired_tumour_eryth,p_tumour_eryth,q_tumour_eryth,frac_donors_up_patch_plaque,log2fc_paired_patch_plaque,n_donors_paired_patch_plaque,p_patch_plaque,q_patch_plaque,strata,n_a,n_b,delta_lfc,p_interaction,q_interaction,gene_set,log2fc_all,q_all,detail_gene
CD70,0.933,1.1,30,1.86e-08,1.01e-05,0.947,1.09,19,2.67e-05,0.000952,tumour_eryth vs patch_plaque,30,19,0.0132,0.992,1,surface,1.1,3.09e-10,True
CD9,0.7,0.946,30,0.00511,0.034,0.632,1.08,19,0.0258,0.0988,tumour_eryth vs patch_plaque,30,19,-0.139,0.845,1,surface,1.08,7.09e-07,True
CPM,0.833,1.39,30,1.99e-06,0.000248,0.632,1.39,19,0.00945,0.05,tumour_eryth vs patch_plaque,30,19,0.00716,0.829,1,surface,1.02,7.08e-08,True
COL6A3,0.833,0.842,30,3.79e-06,0.000302,1,1.17,19,3.81e-06,0.000302,tumour_eryth vs patch_plaque,30,19,-0.328,0.3,0.898,surface,0.811,3.28e-09,True
CD151,0.533,0.103,30,0.0234,0.092,0.842,0.988,19,0.000164,0.00333,tumour_eryth vs patch_plaque,30,19,-0.885,0.0726,0.74,surface,0.794,9.39e-08,True
GRN,0.867,0.623,30,1.99e-06,0.000248,0.947,0.953,19,0.000164,0.00333,tumour_eryth vs patch_plaque,30,19,-0.33,0.2,0.88,surface,0.778,9.81e-10,True
TFRC,0.867,0.693,30,7.99e-06,0.00042,0.947,1.07,19,7.63e-06,0.00042,tumour_eryth vs patch_plaque,30,19,-0.38,0.0454,0.704,surface,0.731,1.67e-09,True
SLC39A6,0.933,0.512,30,2.05e-07,5.25e-05,0.947,0.84,19,1.14e-05,0.000548,tumour_eryth vs patch_plaque,30,19,-0.328,0.0354,0.678,surface,0.707,1.91e-10,True
SLC4A2,0.833,0.537,30,3.79e-06,0.000302,1,0.763,19,3.81e-06,0.000302,tumour_eryth vs patch_plaque,30,19,-0.226,0.179,0.876,surface,0.698,1.91e-10,True
TNFRSF25,0.8,0.722,30,0.000667,0.00842,0.789,0.947,19,0.000965,0.0106,tumour_eryth vs patch_plaque,30,19,-0.224,0.659,0.999,surface,0.698,9.14e-05,True
SLC1A4,0.733,0.724,30,0.0106,0.0541,0.632,0.307,19,0.169,0.35,tumour_eryth vs patch_plaque,30,19,0.416,0.845,1,surface,0.69,5.23e-05,True
LGALS3,0.5,0.0216,30,0.416,0.584,0.526,1.06,19,0.0955,0.241,tumour_eryth vs patch_plaque,30,19,-1.04,0.813,1,surface,0.663,0.00115,True
CD82,0.8,0.593,30,0.000505,0.00703,0.737,0.609,19,0.00079,0.00935,tumour_eryth vs patch_plaque,30,19,-0.0161,0.813,1,surface,0.632,1.41e-07,True
SLC5A3,0.833,0.514,30,3.05e-05,0.00105,0.947,0.846,19,1.91e-05,0.000735,tumour_eryth vs patch_plaque,30,19,-0.332,0.103,0.785,surface,0.617,1.81e-09,True
IL6R,0.767,0.844,30,0.000123,0.00274,0.789,0.687,19,0.0024,0.0198,tumour_eryth vs patch_plaque,30,19,0.157,0.43,0.933,surface,0.614,2.31e-05,True
SYPL1,0.9,0.604,30,2.55e-07,6.18e-05,0.895,0.378,19,0.0024,0.0198,tumour_eryth vs patch_plaque,30,19,0.226,0.492,0.949,surface,0.599,1.15
... [27355 more chars truncated]
```

## §8 — One-file export

Every table reported above, in order, with its WHAT / HOW text and a short context paragraph, is
written as CSV-in-markdown to **`tables/tgt49_llm_export.md`**. Paste that file, or any section of
it, into another analysis. It needs no other file.

Set `PRINT_EXPORT = True` to also print the whole block here. It is long, and Jupyter may truncate
it, so the file is the reliable copy.

```python  # cell 17
# ============================================================================
# §8  One-file export
# ============================================================================
PRINT_EXPORT = False

CONTEXT = f"""
Compartment: MF/CTCL **skin**, CD4 T cells only. Source object {Path(man['source']).name}
({man['n_cells']:,} T/NK cells, {man['n_genes_in_object']:,} genes, raw counts), aggregated per
(arm, donor, gene) by jobs/run_target_screen_qc.py. Expression = log1p CP10K on the true library
size (obs.total_counts); `mean_all` = mean over a donor-arm's cells, `pct_expr` = fraction of
cells with >= 1 UMI. A donor-arm enters with >= {MIN_CELLS} cells.

Arms: malignancy call `{man['malignancy_call']}` (per-cell, made at the level of per-sample Leiden
clusters). CD4_malignant = CD4 cells called malignant; CD4_reactive = CD4 cells called
non-malignant in a patient; CD4_HC = CD4 cells of healthy-skin donors. CD8, Treg and
CD4_unassessed are not analysed.

Cohort: {len(don)} donor keys -> {int(don['cohort_keep'].sum())} independent keys
({don.loc[don['cohort_keep'], 'person'].nunique()} people) after dropping non-CD4 clones,
off-spectrum diagnoses (LyP/SPP/LPP) and repeat biopsies. {len(PAIRED)} donors have both a
malignant and a reactive CD4 arm ("paired donors"); {len(_hc_d)} healthy donors carry CD4_HC.

Genes: {N_SURF_TESTED:,} tested surface genes (CTCL panel UNION confident surfaceome), plus
lineage-control (non-T) and trafficking-machinery reference genes.

Headline test: per gene, paired over donors. log2fc_paired = median over donors of
log2((mean_all_malig + {PSEUDO}) / (mean_all_react + {PSEUDO})), a ratio of log-scale means and so
compressed relative to a raw fold change; Wilcoxon signed-rank on per-donor differences; BH ->
q. The same on pct_expr -> *_detect. Malignant vs healthy CD4 is unpaired (Mann-Whitney on
donor-level mean_all) -> *_vs_hc.

Shortlist gates: surface gene, not a lineage control or loss marker, q < {SHORT_Q},
log2fc_paired > {SHORT_LFC}, >= {DETECT_FLOOR:.0%} of malignant cells positive in >= {SHORT_PREV:.0%}
of patients. No artifact filter is run here; the companion notebook's six artifact tests
excluded nothing on this build because its contamination null was not calibrated. Genes flagged
`disputed_in_42` passed the statistics but are doubtful surface proteins by localisation.
Transcript only: skin has no surface protein measurement.

Calibration: CCR4, TNFRSF8 (CD30), KIR3DL2 and IL2RA are approved or registrational CTCL
antigens and should rank high; CD7 and DPP4 (CD26) loss is the CTCL phenotype and should be
negative.
"""

path = TF.llm_export("49_skin_cd4_surfaceome_numeric",
                     "Skin CD4 surfaceome: malignant vs reactive CD4",
                     CONTEXT, SECTIONS, prefix=PREFIX, print_block=PRINT_EXPORT)

print("\nfiles written by this notebook:")
for p in sorted(ST.TAB_DIR.glob(f"{PREFIX}*")):
    print(f"  {p.relative_to(NB_DIR)}  ({p.stat().st_size / 1024:,.0f} KB)")
```

```
[204,894 characters. Paste everything between the two banner lines into the other model. If the output above is truncated, copy /home/projects/nyosef/zvise/MF/tables/tgt49_llm_export.md instead.]

files written by this notebook:
  tables/tgt49_all_genes.csv  (1,164 KB)
  tables/tgt49_cohort.csv  (14 KB)
  tables/tgt49_detail_summary.csv  (6 KB)
  tables/tgt49_donor_lfc.csv  (52 KB)
  tables/tgt49_donor_pct_healthy.csv  (11 KB)
  tables/tgt49_donor_pct_malignant.csv  (52 KB)
  tables/tgt49_donor_pct_reactive.csv  (52 KB)
  tables/tgt49_eligibility.csv  (11 KB)
  tables/tgt49_llm_export.md  (200 KB)
  tables/tgt49_shortlist.csv  (14 KB)
  tables/tgt49_strat_disease.csv  (550 KB)
  tables/tgt49_strat_lesion.csv  (625 KB)
  tables/tgt49_strat_stage.csv  (604 KB)
```
