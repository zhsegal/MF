# Clinical metadata layer (`clin_*`) — v1, completed 2026-10-22

Per-sample clinical metadata, re-mined from each source paper's tables and supplements plus GEO
and ArrayExpress. Blocked supplements were supplied by hand on 2026-10-22. Coverage: 265 atlas
samples, 194 persons. The person count is after identity merges (§4).

| file | what |
|---|---|
| `data/clinical/raw/{vienna,li_pittsburgh,blood}.tsv` | base extraction, one row per `sample_id`, each with `source` and `confidence` (exact / inferred / geo) |
| `data/clinical/user_supplied/` | the 9 hand-supplied supplements (§2) |
| `data/clinical/parse_user_supplied.py` | supplements → `raw/supplements.tsv`, an overlay that updates base fields one by one; the PDF tables are transcribed in it |
| `data/clinical/raw/b4_hto_arms.tsv` | geskin2026 lane × HTO → ex vivo drug arm (GEO GSE290850) |
| `helpers/clinical_meta.py` | vocabularies, treatment agents, `harmonize`, `attach_clinical(obs)`, obs patchers |
| `jobs/build_clinical_metadata.py` | raw → `data/clinical_metadata_v1.csv` plus the report tables |
| `jobs/apply_clinical_meta.py` | writes the `clin_*` columns into every v2 object, the obs cache and `skin_degradome.h5ad`; skips files locked by a kernel |
| `jobs/run_sex_infer.sh` | per-sample XIST / chrY call → `tables/sample_sex_inferred.csv` |
| `tables/clinical_v1_coverage.csv` | per study: persons with a known value, per field |
| `tables/clinical_v1_contrasts.csv` | persons and samples per level of each comparison axis, by compartment, incl. per-agent prior/current exposure |
| `tables/clinical_v1_conflicts.csv` | every disagreement with the current atlas columns |
| `tables/clinical_v1_tcr_identity.csv` | dominant TRB clones shared across patient_keys |
| `tables/clinical_v1_stage_changes.csv` | persons whose samples carry different stages |

The existing columns are **not** modified: `disease`, `stage_clean`, `entity`, `lineage`, `sex`,
`treatment_context` and `patient_key` still hold their old values. Use the `clin_*` columns, or
`attach_clinical(obs)` on any older object. To fold a reviewed conflict into the atlas columns,
add it to `atlas_v2_fixups`.

```python
import clinical_meta as CM
obs = CM.attach_clinical(adata.obs)        # joins on sample_id; B4 arm comes from real_donor per cell
```

## 1 · Columns

| column | values | note |
|---|---|---|
| `clin_person_key` | `patient_key` + merges | use this for donor-level statistics (§4) |
| `clin_group` | CTCL / HC / parapsoriasis / CD30_LPD | |
| `clin_dx` | MF, SS, MF/SS, MF_or_SS_unresolved, eCTCL_unresolved, pcAECTCL, CD8_aggr_CTCL, LyP, SPP, LPP, HC | erythrodermic MF is `MF` with lesion `erythroderma` |
| `clin_phenotype` | CD4, CD8, gd, DN, unknown, not_applicable | phenotype of the malignant clone |
| `clin_cd30`, `clin_lct`, `clin_fmf` | pos/neg, yes/no, yes | `unknown` means not reported, not negative |
| `clin_stage`, `clin_stage_class`, `clin_stage_timing` | ISCL/EORTC; early (≤IIA) / advanced; sampling / dx / worst | not set for HC, parapsoriasis or LyP |
| `clin_T/N/M/B` | from TNMB | |
| `clin_lesion` | patch, plaque, papule, tumor, erythroderma, mixed, nonlesional, cleared_lesion, LyP_lesion, healthy_skin, skin_unspecified, blood, LN | the lesion that was biopsied |
| `clin_age`, `clin_sex`, `clin_sex_source`, `clin_race`, `clin_duration_y` | sex source is `paper` or `expr` | |
| `clin_blood_tumor_pct` | numeric | % malignant or Sézary-phenotype cells among blood CD4 T cells, where stated |
| **`clin_prior_tx`** | `;`-joined agents | treatment **before** the sample |
| **`clin_current_tx`** | `;`-joined agents | treatment **at** sampling |
| `clin_prior_tx_classes`, `clin_current_tx_classes` | `;`-joined classes | about 20 classes |
| `clin_prior_systemic`, `clin_current_systemic`, `clin_ever_systemic` | yes / no / unknown | ECP counts as systemic; a bare "Steroid" counts as unknown |
| `clin_tx_naive` | yes / no / washout_6mo / unknown | washout_6mo = the PKU cohort |
| `clin_n_prior_lines` | numeric | dorando26 only |
| `clin_timepoint`, `clin_days_from_first` | single, baseline, pre, on_treatment, post_progression, follow_up, serial | |
| `clin_response`, `clin_response_scope`, `clin_overall_response` | R/NR/PD/SD/cleared/mixed; lesion or patient; PR/SD | |
| `clin_drug_sensitivity` | romidepsin_sensitive / romidepsin_resistant | |
| `clin_course`, `clin_outcome` | aggressive/indolent; DOD / died_other / died_unspecified / alive | |
| `clin_ex_vivo`, `clin_arm` | none / culture_ctr / culture_drug; drug arm | per cell for B4 |
| `clin_input` | tissue, PBMC, CD45_sort, CD4_sort, T_sort, malignant_sort, nonmalignant_sort, sorted_other | |
| `clin_qc_flag` | text | dominant TRB clone shared with another patient, unexplained |
| CSV only | `raw_*` (verbatim text), `clin_source`, `clin_stage_note`, `clin_sex_expr`, `raw_tx_during_study` (dorando26 therapy sequence) | |

**Treatment agents** are listed in `AGENTS` in `helpers/clinical_meta.py`, as name, class and pattern:
- **skin-directed:** topical steroid, mechlorethamine, topical bexarotene / immunomodulators, NB-UVB, PUVA, other phototherapy, TSEB, local radiotherapy
- **systemic:** bexarotene, acitretin, interferon, methotrexate, vorinostat, romidepsin, resminostat, ECP, mogamulizumab, brentuximab, alemtuzumab, pralatrexate, gemcitabine, doxorubicin, chlorambucil, other chemotherapy, checkpoint inhibitors, anti-CD47, other targeted/trial agents, dupilumab, alloSCT, rituximab, other immunomodulators, systemic steroid

## 2 · Sources

| study | primary per-patient source | status |
|---|---|---|
| chennareddy2025 (D1) | Chennareddy, BJD 2025, Table 1 (PMC11758594) | FULL |
| brunner2024 (D3) | Chennareddy, JACI 2025 (CIE), Table I — **user-supplied** | FULL |
| rindler2021_fi (D5) | Rindler, Front Immunol 2021 (case) | FULL |
| rindler2021_mc (D7) | Rindler, Mol Cancer 2021, Table 1 | FULL |
| alkon2024 (D8) | Alkon, JACI 2025, Table I — **user-supplied** | FULL |
| lyp2026 (D9) | Meledathu, Blood Adv 2026, Table 1 | FULL |
| jonak2021 (D10) | Jonak, BJD 2021 (case text) | PARTIAL (no race/outcome published) |
| li2024 CTCL1–18 | Li, Nat Immunol 2024, Suppl Table 1 | age/sex/stage/site only; nothing more published |
| li2024 MF14–30 | Liu, Nat Commun 2022, Suppl Data 1 | FULL |
| li2024 PT11–56 | Song, Cancer Discov 2022, Table S1 — **user-supplied** | FULL |
| gaydosik2019 (D6) | Gaydosik, CCR 2019, Suppl Table 1 — **user-supplied**; cross-checked with Gaydosik 2022 and 2023 | FULL |
| gaydosik2022 (D13) | Gaydosik, CCR 2022, Table S1 — **user-supplied** | FULL; GEO→paper ID map inferred (§3) |
| gaydosik2023 (D14) | JCI Insight 2023 (HC only) | FULL |
| il4ra2026 (D15) | Gaydosik, JITC 2026, Table S1 A–D | FULL |
| geskin2026 (B4) | Geskin, Cancer Immunol Res 2026, Suppl Table 1A — **user-supplied** | PARTIAL: rows map to SZ30 and SZ32 only (§3) |
| borcherding2019/2023 (B1/B2) | Borcherding, CCR 2019 + Blood Adv 2023, Suppl Table 1 | FULL |
| buus2025 (B5) | Buus, Cancer Discov 2025, Table S1 | PARTIAL (no stage published) |
| ren2023 (B6) | Ren, Blood Adv 2023, Table 1 | FULL |
| harro2023 (B7) | Harro, Blood Adv 2023, Suppl File 1 | FULL; row map inferred (§3) |
| dorando2026 (B8) | Dorando, Blood 2026, Table S1 + Therapies — **user-supplied**; Andrews 2019 | PARTIAL: MF vs SS and per-sample treatment not resolvable (§3) |
| herrera2021 (H) | Herrera, Blood 2021, Suppl Table 1 — **user-supplied** | FULL |
| brentuximab2026 (D11) | Jiang, Adv Sci 2026, Table S1 | FULL |
| all | Wang/Wilcox, Blood 2026, Table S1 — **user-supplied** | secondary aggregator, used to cross-check only (§3) |

## 3 · Curation decisions

All of these live in `helpers/clinical_meta.py` or `data/clinical/parse_user_supplied.py`.

- **`PERSON_MERGE`** (§4).
- **`STAGE_OVERRIDE`** keeps one stage when the raw text has two:
  - Li MF18 → IVA2 (T3N3M0B1).
  - P72 → IVA1.
  - B1 → IVA1.
- **"RTX"** is rituximab, per the ChennaBJD25 footnote, except for P303. Front Immunol 2021 lists radiotherapy for P303 (`AGENT_SWAP`).
- **Gaydosik 2019 treatment at biopsy** comes from the explicit "at biopsy" column of Gaydosik 2022 / 2023 where the same patient appears (CTCL-5 = MF1, CTCL-8 = MF3, CTCL-12 = MF4). The 2019 table lists treatment as of writing; for example, CTCL-12's romidepsin started after the biopsy.
- **gaydosik2022 GEO→paper IDs:**
  - SZ29 + MF25 = SS2 (68/M, Hodgkin lymphoma A+AVD).
  - SZ30 = SS4 (= Geskin SS30).
  - MF19 + SZ22 = SS1 (shared TRB clone).
  - SZ16 = SS3, by elimination against Wilcox S1. SZ16's current treatment is dated from the timeline (TTI-621), because the table's "at biopsy" lacutamab started after the draw.
- **harro2023 row map, decided by expression sex:**
  - rows 1–4 → SS1–4 (F, M, M, M all match);
  - MF3 = row 5 (68/M);
  - MF1 and MF2 = rows 6 and 7 (both female, order unresolved), so age and TNMB are withheld.
  - Wilcox's in-order map fails 4 of 7.
- **geskin2026:** only SZ30 and SZ32 are verified, by TCR and sex. For SZ5, SZ38 and SZ39 the table's sexes contradict expression, so only what every candidate row shares is kept (SS, IVA).
- **dorando2026:**
  - Per-sample blood Sézary % assumes the scRNA sample numbers are chronological within a patient.
  - Therapy days have no calendar anchor, so the sequence is kept as `raw_tx_during_study` and not assigned to samples.
- **Expression sex** fills sex only where no paper reports it.
  - XIST is the primary signal, because chrY loss is common in male clones.
  - Li2024's object carries no XIST or chrY reads at all (`no_signal`).
- **Unknown ≠ negative.** An LCT, FMF, CD30 or phenotype the paper doesn't report stays `unknown`.

## 4 · Identity: same person under two keys

| merged into | members | evidence |
|---|---|---|
| vienna:MF318 | P97, P124, P318 thin/thick | ChennaBJD25 footnote |
| vienna:P195 | P218 | footnote + shared TRB clone (91% / 54%) |
| B1__P4 | B1__P5 | Borcherding 2019 methods: P5 is the non-malignant sort |
| pittsburgh:MF19 | D13 SZ22 (blood), **Herrera SS2** (skin + blood) | shared dominant TRB `CASKLFMDREKGEAFF`; both 62/M SS, dx 2019. One patient sampled at Pittsburgh/Columbia and at NYU; stage IIIB vs IVA1 at the two timepoints |
| pittsburgh:MF26 | B4 SZ32 | shared TRB clone (88% / 11%); 30–32/M, erythrodermic IVB, 2021 |

**Unexplained:** buus2025 PT05, PT06 and PT18 share PT06's dominant clone (96% in PT06, 22% in PT18, 7% in PT05). That points to contamination or a label swap, not one person. These samples carry `clin_qc_flag`.

## 5 · Findings that change analyses (all rows are in `clinical_v1_conflicts.csv`)

1. **geskin2026 (B4), 260,455 blood cells: these are ex vivo 2-day cultures with 6 drug arms per lane.**
   - PF46–50 are healthy donors labelled SS.
   - 213,181 cells are drug-treated. Filter `clin_ex_vivo != "culture_drug"`.
2. **Disease relabels:**
   - ren23: 7 SS + 4 MF.
   - herrera21: SS1–6 are SS.
   - buus25: PT18 and PT21 are MF.
   - D6: MF2 is SS; MF5 is a CD8 aggressive cytotoxic CTCL with LCT (died 2017).
   - D15: MF31 is SS.
   - D1: Berti CD8 ×4; γδ-MF ×5.
   - D3: 3 erythrodermic MF + 2 SS.
3. **The `skin_degradome` "MF skin" set includes D6 MF5 (CD8 aggressive cytotoxic CTCL) and two SS skin samples.**
4. **Treatment exposure is confounded with study.** All of D11 is on BV. PKU is naive or 6-month washout. Blood cohorts are mostly on ECP and bexarotene. Check `clinical_v1_contrasts.csv` before testing a treatment axis.

## 6 · Still unavailable

- buus2025 stage, and per-timepoint treatment (only in the Fig 2F image).
- dorando2026 MF vs SS.
- li2024 CTCL1–18 treatment: Wilcox S1 says "none", but Li publishes nothing, so it is not used.
- Jonak 2021 race and outcome.
- Healthy-donor demographics for D14, D15, B4 and B6.
