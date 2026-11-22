# 40_targets/41_surface_panel.ipynb

_Rendered from the archived notebook on 2026-11-22. This is the record of what it printed; the notebook itself is at `notebooks/_archive/40_targets/41_surface_panel.ipynb` and is never run._

# 41 — The surface-target panel and the surfaceome reference

The roster for the CAR-T / bispecific screen, and an honest account of what it does and does not
cover. Nothing here touches the atlas counts; 42 (skin) and 43 (blood) do the screening.

- **§0** setup
- **§1** the a-priori panel: 12 functional classes, 143 genes, with the CTCL additions
- **§2** the surfaceome reference: four sources, what each is worth, and why `n_evidence >= 2`
  alone would be the wrong gate
- **§3** the screen gene set = panel ∪ confident surfaceome, and its coverage in both compartments
- **§4** the trafficking machinery: 11 phases between transcript and displayed epitope
- **§5** clinical precedent, and the positive controls every later ranking is checked against

Writes `tables/tgt41_panel.csv`, `tgt41_surfaceome_summary.csv`, `tgt41_coverage.csv`.

> The governing caveat, repeated here because every number downstream inherits it: **mRNA is not
> surface protein.** This notebook builds a hypothesis roster. Only flow or CITE-seq can call a
> target.

```python  # cell 1
# ============================================================================
# §0  Setup
# ============================================================================
import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

import surface_targets as ST          # the curated panel, machinery, precedent
import cluster_malig_helpers as CM    # compartment objects

plt.rcParams["figure.dpi"] = 120
print("NB_DIR =", NB_DIR)
```

```
NB_DIR = /home/projects/nyosef/zvise/MF
```

## §1 — The a-priori panel

Twelve functional classes. The class is half the interpretation: a hit in `pan_t` carries a
T-aplasia cost that a hit in `ctcl_associated` may not, and a hit in `cla_biosynthesis` is not a
target at all — those are Golgi enzymes carried so a `SELPLG` result can be read against the
machinery that builds its epitope.

`LOSS_MARKERS` are in the panel with the opposite sign: CD7, CD26/DPP4 and CD5 are **lost** in
CTCL, which is what makes them diagnostic. Ranking them by "high in malignant" is a sign error.

```python  # cell 3
# ============================================================================
# §1  The panel
# ============================================================================
panel = ST.annotate(ST.PANEL_GENES)
by_class = (pd.Series({k: len(v) for k, v in ST.SURFACE_PANEL.items()}, name="n_genes")
            .sort_values(ascending=False))
print(f"{len(ST.PANEL_GENES)} genes in {len(ST.SURFACE_PANEL)} classes\n")
print(by_class.to_string())

print("\nloss markers (a CTCL phenotype is LOSS -- do not rank these by 'high in malignant'):")
print("  " + ", ".join(ST.LOSS_MARKERS))

print("\ngenes carrying a documented transcript-to-surface caveat:")
for g, note in ST.SURFACE_CAVEAT.items():
    print(f"  {g:10s} {note}")

panel.to_csv(ST.tab("panel"), index=False)
print("\nwrote", ST.tab("panel"))
panel.head(10)
```

```
143 genes in 12 classes

ctcl_associated       23
integrin_adhesion     18
chemokine_receptor    16
cytokine_receptor     15
kir_lectin_nk         14
tcr_core              13
checkpoint            12
tnfrsf_costim         12
coreceptor             8
pan_t                  8
cla_biosynthesis       7
transporter            5

loss markers (a CTCL phenotype is LOSS -- do not rank these by 'high in malignant'):
  CD7, DPP4, CD5, CD6, CD27, CD2, CD26

genes carrying a documented transcript-to-surface caveat:
  CTLA4      constitutively internalised by AP2M1 via its YVKM motif; surface pool is a few % of total. LRBA/DEF6 recycling sets what is displayed
  PDCD1      rapid internalisation and FBXO38-mediated degradation; transcript overstates surface
  SELL       shed by ADAM17 within minutes of activation -- read SELL beside ADAM17
  CD4        down-modulated from the surface by Nef-independent AP-2 routes on activation
  TNFRSF8    the soluble CD30 ectodomain is shed by ADAM17; serum sCD30 is a CTCL biomarker
  KIR3DL2    stable surface receptor, internalises on ligand binding -- the basis of lacutamab ADCC
  CCR4       ligand-induced internalisation (CCL17/CCL22); a high-CCL17 lesion lowers surface CCR4
  IL2RA      shed as sIL-2R; serum sIL-2R tracks CTCL burden, so surface and transcript diverge

wrote /home/projects/nyosef/zvise/MF/tables/tgt41_panel.csv

    gene panel_class traffic_module              cd_name  \
0   TRAC    tcr_core                                       
1  TRBC1    tcr_core                 TCR beta constant 1   
2  TRBC2    tcr_core                                       
3   TRDC    tcr_core                                       
4  TRGC1    tcr_core                                       
5  TRGC2    tcr_core                                       
6   CD3D    tcr_core                                       
7   CD3E    tcr_core                                       
8   CD3G    tcr_core                                       
9  CD247    tcr_core                                       

                                           agent      stage  \
0                                                             
1  AUTO4 (terminated); FG-M131 ADC; TRBC1 CAR-NK  phase 1/2   
2                                                             
3                                                             
4                                                             
5                                                             
6                                                             
7                                                             
8                                                             
9                                                             

          indication     evidence  \
0                                   
1  PTCL, CTCL cohort  NCT07758491   
2                                   
3                                   
4                                   
5                                   
6                                   
7                                   
8                                   
9                                   

                                   off_tumor_concern no_precedent_note  \
0                                                                        
1  spares the TRBC2 repertoire -- only TRBC1+ nor...                     
2                                                                        
3                                                                        
4                                                                        
5                                                                        
6                                                                        
7                                                                        
8                                                                        
9                                                                        

   positive_control  loss_marker surface_cave
... [577 more chars truncated]
```

## §2 — The surfaceome reference

Four sources, built by `jobs/download_surfaceome.sh` → `jobs/build_surfaceome_ref.py`:

| source | what it is | strength |
|---|---|---|
| SURFY | ML classifier over 20,193 UniProt entries (Bausch-Fluck 2018 PNAS) | genome-wide recall; mostly **prediction** |
| CSPA | MS of labelled surface glycoproteins on **intact** cells (2015) | the only experimental evidence — proves reachability |
| HPA | immunofluorescence subcellular location = plasma membrane | independent, antibody-based |
| GO:0009986 | `cell surface`, human, with descendants (QuickGO) | curated annotation |

`GO:0005886` *plasma membrane* is deliberately excluded — it truncates at QuickGO's 50k download
cap, so membership would depend on accession order, and it includes cytoplasm-facing proteins.

**Why `n_evidence >= 2` is the wrong gate on its own.** It drops **KIR3DL2**, the most
Sézary-restricted antigen known and a phase-2 antibody target, because no CSPA, HPA or GO
cell-surface record exists for it — SURFY alone carries it, as a *curated* entry rather than a
prediction. It also drops TRAC/TRBC1/TRBC2, which SURFY labels `nonsurface` as standalone chains
although the TCR complex is the most displayed structure on a T cell. So the gate is
`surface_confident` = two sources **or** one source that is evidence rather than prediction, and
the screen takes the **union** with the curated panel, never the intersection.

```python  # cell 5
# ============================================================================
# §2  The surfaceome reference
# ============================================================================
surf_all = ST.load_surfaceome(confident_only=False)
surf = ST.load_surfaceome()                       # surface_confident only
print(f"{len(surf_all):,} genes with any surface evidence | "
      f"{len(surf):,} surface_confident\n")

ev = pd.DataFrame({
    "in_surfy": surf_all["in_surfy"], "in_cspa": surf_all["in_cspa"],
    "in_hpa_pm": surf_all["in_hpa_pm"], "in_go_cs": surf_all["in_go_cs"],
}).sum().rename("n_genes")
print(ev.to_string())
print("\nby number of independent sources:")
print(surf_all["n_evidence"].value_counts().sort_index(ascending=False).to_string())

# The point of the confidence gate, stated as a number rather than an assertion.
naive = set(surf_all.loc[surf_all["n_evidence"] >= 2, "gene"])
conf = set(surf["gene"])
rescued = sorted(conf - naive)
lost_ctrl = [g for g in ST.POSITIVE_CONTROLS if g not in naive]
print(f"\n`n_evidence >= 2` alone: {len(naive):,} genes")
print(f"`surface_confident`:      {len(conf):,} genes  (+{len(rescued):,} single-source but "
      "curated/MS-confirmed)")
print(f"clinically validated CTCL antigens a naive >=2 filter would DROP: {lost_ctrl}")

summary = surf_all.groupby("n_evidence").agg(
    n_genes=("gene", "size"), n_cspa=("in_cspa", "sum"),
    n_surfy_curated=("surfy_curated", "sum"), n_confident=("surface_confident", "sum")).reset_index()
summary.to_csv(ST.tab("surfaceome_summary"), index=False)
print("\nwrote", ST.tab("surfaceome_summary"))
summary
```

```
5,432 genes with any surface evidence | 2,325 surface_confident

in_surfy     2799
in_cspa      1443
in_hpa_pm    2263
in_go_cs     1164

by number of independent sources:
n_evidence
4     143
3     409
2     990
1    3890

`n_evidence >= 2` alone: 1,542 genes
`surface_confident`:      2,325 genes  (+783 single-source but curated/MS-confirmed)
clinically validated CTCL antigens a naive >=2 filter would DROP: ['KIR3DL2']


wrote /home/projects/nyosef/zvise/MF/tables/tgt41_surfaceome_summary.csv

   n_evidence  n_genes  n_cspa  n_surfy_curated  n_confident
0           1     3890     471              594          783
1           2      990     491              343          990
2           3      409     338              217          409
3           4      143     143              100          143
```

## §3 — The screen gene set, and what each compartment can actually see

`screen_genes()` = curated panel ∪ confident surfaceome, intersected with the genes the object
carries. The intersection is the part to watch: **a gene absent from the object is absent from the
result, which is not the same as being unexpressed**, and the blood object is a 10,000-gene HVG
subset rather than the full 42,347.

```python  # cell 7
# ============================================================================
# §3  Coverage in the two compartments
# ============================================================================
import h5py

rows, cov = [], {}
for tissue in ("skin", "blood"):
    obj = CM.CONFIG[tissue]["TCR_OBJ"]
    if not obj.exists():
        print(f"{tissue:6s} {obj.name}: NOT ON DISK -- skipped")
        continue
    with h5py.File(obj, "r") as h5:
        var = pd.Index(CM._read_elem(h5["var"][h5["var"].attrs["_index"]])).astype(str)
    genes, rep = ST.screen_genes(var)
    cov[tissue] = (genes, rep, var)
    missing_panel = rep["panel_absent_from_object"]
    missing_ctrl = [g for g in ST.POSITIVE_CONTROLS if g not in set(var)]
    print(f"{tissue:6s} {obj.name}: {len(var):,} genes in object -> {len(genes):,} screened "
          f"({rep['by_source']})")
    print(f"       panel present {rep['n_panel_present']}/{rep['n_panel']}"
          + (f" | ABSENT: {missing_panel}" if missing_panel else ""))
    if missing_ctrl:
        print(f"       *** positive controls MISSING from the object: {missing_ctrl} ***")
    rows.append({"tissue": tissue, "genes_in_object": len(var), "n_screened": len(genes),
                 "panel_present": rep["n_panel_present"], "panel_total": rep["n_panel"],
                 "panel_absent": ";".join(missing_panel),
                 "controls_absent": ";".join(missing_ctrl)})

coverage = pd.DataFrame(rows)
coverage.to_csv(ST.tab("coverage"), index=False)
print("\nwrote", ST.tab("coverage"))
coverage
```

```
skin   skin_T_tcr_annotated_v4.h5ad: 42,347 genes in object -> 2,115 screened ({'surfaceome': 1973, 'both': 125, 'panel': 17})
       panel present 142/143 | ABSENT: ['KIR2DL2']

blood  blood_T_annotated.h5ad: 10,000 genes in object -> 1,030 screened ({'surfaceome': 914, 'both': 102, 'panel': 14})
       panel present 116/143 | ABSENT: ['BTLA', 'CD164', 'CD3G', 'CD4', 'CD44', 'CD53', 'FUT4', 'GCNT1', 'ICOSLG', 'IL12RB1', 'IL15RA', 'IL31RA', 'ITGA2', 'ITGA5', 'ITGA6', 'ITGAE', 'ITGB7', 'KIR2DL2', 'LY75', 'PDCD1LG2', 'SLC2A1', 'SLC3A2', 'ST3GAL4', 'ST3GAL6', 'TNFRSF14', 'TNFRSF1A', 'TNFRSF8']
       *** positive controls MISSING from the object: ['TNFRSF8'] ***

wrote /home/projects/nyosef/zvise/MF/tables/tgt41_coverage.csv

  tissue  genes_in_object  n_screened  panel_present  panel_total  \
0   skin            42347        2115            142          143   
1  blood            10000        1030            116          143   

                                        panel_absent controls_absent  
0                                            KIR2DL2                  
1  BTLA;CD164;CD3G;CD4;CD44;CD53;FUT4;GCNT1;ICOSL...         TNFRSF8
```

### The blood HVG problem

If blood shows ~10,000 genes above, the blood object is the MrVI HVG input, not a full-gene
object. A rare-but-specific antigen is exactly the kind of gene HVG selection discards, so a
blood screen run on it has a floor that has nothing to do with biology. Check the controls line:
if any of CCR4 / TNFRSF8 / KIR3DL2 / IL2RA is missing from the object, the blood screen needs a
full-gene source before it can be read.

```python  # cell 9
# ============================================================================
# §3b  Which panel genes the blood HVG subset drops
# ============================================================================
if "blood" in cov and "skin" in cov:
    sk, bl = set(cov["skin"][2]), set(cov["blood"][2])
    dropped = sorted((set(ST.PANEL_GENES) & sk) - bl)
    print(f"panel genes present in skin but NOT in the blood object: {len(dropped)}")
    if dropped:
        ann = ST.annotate(dropped)
        with_precedent = ann[ann["stage"].ne("")][["gene", "cd_name", "agent", "stage"]]
        print(ann[["gene", "panel_class"]].to_string(index=False))
        if len(with_precedent):
            print("\n*** of those, these have a clinical agent and would be invisible in blood:")
            print(with_precedent.to_string(index=False))
else:
    print("need both objects on disk for this comparison")
```

```
panel genes present in skin but NOT in the blood object: 26
    gene                       panel_class
    BTLA                        checkpoint
   CD164                   ctcl_associated
    CD3G                          tcr_core
     CD4                          tcr_core
    CD44                 integrin_adhesion
    CD53                        coreceptor
    FUT4                  cla_biosynthesis
   GCNT1                  cla_biosynthesis
  ICOSLG                        checkpoint
 IL12RB1                 cytokine_receptor
  IL15RA                 cytokine_receptor
  IL31RA                   ctcl_associated
   ITGA2                 integrin_adhesion
   ITGA5                 integrin_adhesion
   ITGA6                 integrin_adhesion
   ITGAE                 integrin_adhesion
   ITGB7 integrin_adhesion|ctcl_associated
    LY75                   ctcl_associated
PDCD1LG2                        checkpoint
  SLC2A1                       transporter
  SLC3A2                       transporter
 ST3GAL4                  cla_biosynthesis
 ST3GAL6                  cla_biosynthesis
TNFRSF14                     tnfrsf_costim
TNFRSF1A                 cytokine_receptor
 TNFRSF8     tnfrsf_costim|ctcl_associated

*** of those, these have a clinical agent and would be invisible in blood:
   gene cd_name                                     agent                stage
    CD4     CD4 zanolimumab (HuMax-CD4); LB1901 CD4 CAR-T phase 3 (terminated)
 IL31RA IL-31RA                               nemolizumab case reports in CTCL
TNFRSF8    CD30   brentuximab vedotin; AFM13 (CD30xCD16A)             approved
```

## §4 — The trafficking machinery

Eleven phases between a transcript and a displayed epitope. This is the answer to "why is
transcript not protein": the Sec61 translocon and TRAP complex put the chain in the membrane, the
calnexin cycle decides whether it folds, COPII exports it, glycosyltransferases build the lattice
that holds it, AP-2 pulls it back in, LRBA/RAB11 recycle it or the retromer rescues it, E3 ligases
destroy it, SNAREs display it, ADAM17 cuts it off, and ESCRT ships it out in an exosome.

Scored per arm in 42/43 as **context**, not as a per-gene correction factor.

```python  # cell 11
# ============================================================================
# §4  Trafficking machinery
# ============================================================================
mod = pd.Series({k: len(v) for k, v in ST.TRAFFIC_MODULES.items()}, name="n_genes")
print(f"{len(ST.TRAFFIC_GENES)} genes in {len(ST.TRAFFIC_MODULES)} phases\n")
print(mod.to_string())

if "skin" in cov:
    present = set(cov["skin"][2])
    miss = {k: [g for g in v if g not in present] for k, v in ST.TRAFFIC_MODULES.items()}
    miss = {k: v for k, v in miss.items() if v}
    print("\ngenes not in the skin object (a module with few genes left is not scoreable):")
    for k, v in miss.items():
        print(f"  {k:24s} {len(v):2d} missing: {v[:8]}{' ...' if len(v) > 8 else ''}")

# The panel genes whose surface density is governed by a specific machine.
print("\ncandidate x machinery pairs the ranking tables must print together:")
for g, note in ST.SURFACE_CAVEAT.items():
    partners = [m for m in ("shedding", "endocytosis", "recycling", "glycosylation")
                if any(w in note for w in ("shed", "internalis", "recycl", "glycan", "lattice"))]
    print(f"  {g:10s} -> {partners}")
```

```
248 genes in 11 phases

er_translocation          17
er_folding_qc             26
er_to_golgi               28
glycosylation             36
endocytosis               18
recycling                 20
retrograde                23
ubiquitin_proteostasis    27
snare_fusion              18
shedding                  15
escrt_exosome             20

genes not in the skin object (a module with few genes left is not scoreable):
  er_folding_qc             1 missing: ['HRD1']
  ubiquitin_proteostasis    3 missing: ['MARCHF1', 'MARCHF5', 'MARCHF8']

candidate x machinery pairs the ranking tables must print together:
  CTLA4      -> ['shedding', 'endocytosis', 'recycling', 'glycosylation']
  PDCD1      -> ['shedding', 'endocytosis', 'recycling', 'glycosylation']
  SELL       -> ['shedding', 'endocytosis', 'recycling', 'glycosylation']
  CD4        -> []
  TNFRSF8    -> ['shedding', 'endocytosis', 'recycling', 'glycosylation']
  KIR3DL2    -> ['shedding', 'endocytosis', 'recycling', 'glycosylation']
  CCR4       -> ['shedding', 'endocytosis', 'recycling', 'glycosylation']
  IL2RA      -> ['shedding', 'endocytosis', 'recycling', 'glycosylation']
```

## §5 — Clinical precedent and the positive controls

Verified against ClinicalTrials.gov and PubMed on 2026-10-22. Four antigens are approved or
registrational in CTCL — **CCR4** (mogamulizumab), **TNFRSF8**/CD30 (brentuximab vedotin),
**KIR3DL2** (lacutamab) and **IL2RA**/CD25 (denileukin diftitox). They are the screen's positive
controls: a ranking that does not put them near the top is miscalibrated, and the novel hits
below them are not evidence of anything.

Newest entrants worth watching, both with CTCL-specific cohorts opened in 2026: **CCR8**
(DT-7012, first-in-human dedicated to R/R MF/SS) and **TNFRSF1B**/TNFR2 (three independent
phase-1 programmes).

```python  # cell 13
# ============================================================================
# §5  Precedent
# ============================================================================
prec = ST.annotate(sorted(ST.CLINICAL_PRECEDENT))
prec = prec[["gene", "cd_name", "agent", "stage", "indication", "evidence",
             "off_tumor_concern", "positive_control"]]
pd.set_option("display.max_colwidth", 54)
print(prec.to_string(index=False))

print("\npositive controls:", ST.POSITIVE_CONTROLS)
print("\nchecked and found to have NO agent in CTCL / mature T-cell neoplasms:")
for g, why in ST.NO_PRECEDENT_CHECKED.items():
    print(f"  {g:8s} {why}")

prec.to_csv(ST.tab("clinical_precedent"), index=False)
print("\nwrote", ST.tab("clinical_precedent"))
```

```
gene             cd_name                                                       agent                         stage                                       indication      evidence                                                       off_tumor_concern  positive_control
     BSG               CD147                                                 CD147 CAR-T               phase 1 (early)                                            T-NHL   NCT05013372                       ubiquitous (RBC, endothelium) -- broad off-tumour             False
    CCR4                CCR4                                               mogamulizumab                      approved                                  MF/SS (MAVORIC)   NCT04128072 Treg depletion -> rash, autoimmunity, steroid-refractory GvHD post-allo              True
    CCR8                CCR8                                                     DT-7012      phase 1 (first-in-human)     R/R MF/SS -- CTCL-dedicated, started 2026-01   NCT07213882      depletes CCR8+ Tregs -> autoimmunity (also the intended mechanism)             False
    CD37                CD37                                      AGS67E (ADC); CAR-37 T                       phase 1 lymphoid malignancies -- no CTCL-specific cohort   NCT02175433                                        B/myeloid depletion, neutropenia             False
    CD38                CD38                                     daratumumab; isatuximab                     phase 2/3                            T-ALL, PTCL -- NOT MF   NCT07072585  NK/plasma-cell loss, hypogammaglobulinaemia, blood-typing interference             False
     CD4                 CD4                   zanolimumab (HuMax-CD4); LB1901 CD4 CAR-T          phase 3 (terminated)                                            MF/SS   NCT00127881                         profound CD4 aplasia -> opportunistic infection             False
    CD47                CD47 TTI-621 (SIRPaFc, intratumoral); magrolimab + mogamulizumab phase 1b/2 -- both terminated                                            MF/SS   NCT02890368                      RBC antigen sink -> anaemia; ubiquitous expression             False
     CD5                 CD5                            CD5 CAR-T; MT-101 (CAR-monocyte)                     phase 1/2                                    PTCL and CTCL   NCT07022964                                          fratricide, normal T depletion             False
    CD52                CD52                                                 alemtuzumab           phase 2 (off-label)                            SS / erythrodermic MF   NCT00057967                   pan-lymphocyte + monocyte depletion, CMV reactivation             False
     CD7                 CD7                         WU-CART-007; BEAM-201 (base-edited)                     phase 1/2                T-ALL; MF-transformed / SS cohort   NCT05377827        fratricide + T/NK aplasia; CD7 is often already LOST on SS cells             False
    CD70                CD70                        INCA036873 (CD70xCD3); CTX130/CTX131                       phase 1                              MF/SS >= IIB cohort   NCT07195916                            activated T/B + Treg loss; renal tubule CD70             False
    CD84       CD84 / SLAMF5                                                      CART84                     phase 1/2                    AML and T-ALL -- not yet CTCL   NCT07471789                                      platelet / B / monocyte expression             False
   IL2RA                CD25   denileukin diftitox-cxdl (Lymphir); camidanlumab tesirine            approved (2024-08)                                         R/R CTCL   NCT00050999                            capillary leak, Treg loss, visual impairment              True
  IL31RA             IL-31RA                                                 nemolizumab          case reports in CTCL               CTCL pruritus -- NOT cytoreductive PMID 42662382               
... [1791 more chars truncated]
```

```python  # cell 14
# ============================================================================
# §5b  One figure: where the precedent sits in the panel
# ============================================================================
cls_counts = pd.Series({k: len(v) for k, v in ST.SURFACE_PANEL.items()}).sort_values()
has_prec = {k: sum(g in ST.CLINICAL_PRECEDENT for g in v) for k, v in ST.SURFACE_PANEL.items()}
prec_counts = pd.Series(has_prec).reindex(cls_counts.index)

fig, ax = plt.subplots(figsize=(5, 3))
ax.barh(cls_counts.index, cls_counts.values, color="#d5d8dc", label="panel genes")
ax.barh(prec_counts.index, prec_counts.values, color="#c0392b", label="has a clinical agent")
ax.set_xlabel("genes")
ax.set_title("Every CTCL agent so far targets a chemokine receptor,\na TNF-family receptor or a "
             "pan-T antigen", fontsize=8)
ax.legend(fontsize=7, loc="lower right")
ax.tick_params(labelsize=7)
fig.tight_layout()
fig.savefig(ST.fig("panel_precedent"), dpi=150, bbox_inches="tight")
plt.show()
```

```
<Figure size 600x360 with 1 Axes>
```

## Next

The screen splits by compartment from here: `skin/42_target_screen_skin` →
`skin/47_skin_stratification_heterogeneity` → `skin/48_skin_pooled_stratification`, and
`blood/43_target_screen_blood` → `blood/44` → `blood/45` → `blood/46`. All of them read
`data/targets/<tissue>_target_long.parquet`, now built by the QC builder in both compartments:

```
jobs/run_target_screen_qc.sh --tissue skin     # MEM_MB=64000 WALL_H=8
jobs/run_target_screen_qc.sh --tissue blood
```

```python  # cell 16
# ============================================================================
# §6  The complete numerical result, as one block to hand to another model
# ============================================================================
import target_figs as TF                      # noqa: E402  (only this cell needs it)

ST.TAB_PREFIX = ST.FIG_PREFIX = "tgt41_"
_ctx = f"""
This notebook builds the ROSTER for the MF/CTCL surface-target screen -- which genes are
eligible to be screened and what is already known about them. It touches no atlas counts; the
screening itself is notebooks 42/47/48 (skin) and 43/44/45/46 (blood).

The screen gene set is the curated a-priori panel UNION the confident surfaceome, intersected
with the genes each compartment's object actually carries. `surface_confident` means two
independent sources, or one source that is experimental evidence rather than prediction -- a
plain `n_evidence >= 2` gate would drop KIR3DL2, the most Sezary-restricted antigen known.

Sources: SURFY (ML over UniProt), CSPA (mass spectrometry of labelled surface glycoproteins on
intact cells -- the only direct evidence of reachability), HPA immunofluorescence, and GO
cell-surface. GO plasma membrane is deliberately excluded.

The governing caveat for everything downstream: mRNA is not surface protein.
"""
_sections = [
    {"name": "§1 Panel by functional class",
     "what": "the a-priori panel: how many genes in each of the 12 classes.",
     "how": "hand-curated in helpers/surface_targets.py; the class carries half the "
            "interpretation (a pan-T hit has an aplasia cost a CTCL-associated one may not).",
     "table": by_class, "file": "tables/tgt41_panel.csv"},
    {"name": "§2 Surfaceome evidence",
     "what": "how many genes each source supports, and how many clear the confidence gate.",
     "how": "counts over the merged reference.", "table": ev},
    {"name": "§2 Genes per number of independent sources",
     "what": "the distribution of evidence depth.", "how": "value counts of n_evidence.",
     "table": summary, "file": "tables/tgt41_surfaceome_summary.csv"},
    {"name": "§3 Coverage per compartment",
     "what": "how many genes each compartment's object carries, how many are screened, and "
             "which panel genes or positive controls are missing from it.",
     "how": "intersection of the screen set with the object's var names. A gene absent here is "
            "absent from that compartment's result, which is not the same as unexpressed.",
     "table": coverage, "file": "tables/tgt41_coverage.csv"},
    {"name": "§4 Trafficking machinery",
     "what": "the 11 phases between a transcript and a displayed epitope, and their size.",
     "how": "curated gene sets; scored per arm in the screens as context, never as a per-gene "
            "correction factor.", "table": mod},
    {"name": "§5 Clinical precedent",
     "what": "every panel gene with an agent in CTCL or mature T-cell neoplasms.",
     "how": "verified against ClinicalTrials.gov and PubMed, 2026-10-22. The four positive "
            f"controls are {ST.POSITIVE_CONTROLS}: a ranking that does not put them near the "
            "top is miscalibrated.",
     "table": prec, "file": "tables/tgt41_clinical_precedent.csv"},
]
TF.llm_export("41", "The surface-target panel and the surfaceome reference", _ctx, _sections)
```

```
======================== COPY FROM HERE ========================

# 41 — The surface-target panel and the surfaceome reference

This block is a complete, self-contained numerical report from one notebook of the MF/CTCL
surface-target project. Everything another model needs in order to reason about these numbers is
inside it: what was measured, on which cohort, with which test, and the result tables themselves.
No file outside this block has to be read. Numbers are given to 3 significant figures; tables are
CSV with a header row. A column named `q` is a Benjamini-Hochberg adjusted p-value unless the
section says otherwise.

## Context

This notebook builds the ROSTER for the MF/CTCL surface-target screen -- which genes are
eligible to be screened and what is already known about them. It touches no atlas counts; the
screening itself is notebooks 42/47/48 (skin) and 43/44/45/46 (blood).

The screen gene set is the curated a-priori panel UNION the confident surfaceome, intersected
with the genes each compartment's object actually carries. `surface_confident` means two
independent sources, or one source that is experimental evidence rather than prediction -- a
plain `n_evidence >= 2` gate would drop KIR3DL2, the most Sezary-restricted antigen known.

Sources: SURFY (ML over UniProt), CSPA (mass spectrometry of labelled surface glycoproteins on
intact cells -- the only direct evidence of reachability), HPA immunofluorescence, and GO
cell-surface. GO plasma membrane is deliberately excluded.

The governing caveat for everything downstream: mRNA is not surface protein.

## §1 Panel by functional class

**What:** the a-priori panel: how many genes in each of the 12 classes.

**How:** hand-curated in helpers/surface_targets.py; the class carries half the interpretation (a pan-T hit has an aplasia cost a CTCL-associated one may not).

**Full table on disk:** `tables/tgt41_panel.csv`

```csv
key,n_genes
ctcl_associated,23
integrin_adhesion,18
chemokine_receptor,16
cytokine_receptor,15
kir_lectin_nk,14
tcr_core,13
checkpoint,12
tnfrsf_costim,12
coreceptor,8
pan_t,8
cla_biosynthesis,7
transporter,5
```

## §2 Surfaceome evidence

**What:** how many genes each source supports, and how many clear the confidence gate.

**How:** counts over the merged reference.

```csv
key,n_genes
in_surfy,2799
in_cspa,1443
in_hpa_pm,2263
in_go_cs,1164
```

## §2 Genes per number of independent sources

**What:** the distribution of evidence depth.

**How:** value counts of n_evidence.

**Full table on disk:** `tables/tgt41_surfaceome_summary.csv`

```csv
n_evidence,n_genes,n_cspa,n_surfy_curated,n_confident
1,3890,471,594,783
2,990,491,343,990
3,409,338,217,409
4,143,143,100,143
```

## §3 Coverage per compartment

**What:** how many genes each compartment's object carries, how many are screened, and which panel genes or positive controls are missing from it.

**How:** intersection of the screen set with the object's var names. A gene absent here is absent from that compartment's result, which is not the same as unexpressed.

**Full table on disk:** `tables/tgt41_coverage.csv`

```csv
tissue,genes_in_object,n_screened,panel_present,panel_total,panel_absent,controls_absent
skin,42347,2115,142,143,KIR2DL2,
blood,10000,1030,116,143,BTLA;CD164;CD3G;CD4;CD44;CD53;FUT4;GCNT1;ICOSLG;IL12RB1;IL15RA;IL31RA;ITGA2;ITGA5;ITGA6;ITGAE;ITGB7;KIR2DL2;LY75;PDCD1LG2;SLC2A1;SLC3A2;ST3GAL4;ST3GAL6;TNFRSF14;TNFRSF1A;TNFRSF8,TNFRSF8
```

## §4 Trafficking machinery

**What:** the 11 phases between a transcript and a displayed epitope, and their size.

**How:** curated gene sets; scored per arm in the screens as context, never as a per-gene correction factor.

```csv
key,n_genes
er_translocation,17
er_folding_qc,26
er_to_golgi,28
glycosylation,36
endocytosis,18
recycling,20
retrograde,23
ubiquitin_proteostasis,27
snare_fusion,18
shedding,15
escrt_exosome,20
```

## §5 Clinical precedent

**What:** every panel gene with an agent in CTCL or mature T-cell neoplasms.

**How:** verified
... [3411 more chars truncated]
```
