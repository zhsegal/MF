# Atlas handoff — audit conclusions & ingest plan

Date: 2026-09-05 · For the Claude Code session that maintains the MF/CTCL atlas.
Sources: audit of `DATA_PROVENANCE.md`, 5-round literature sweep, head-to-head diff against
Wang/Wilcox *Blood* 2026 (`blood.2025032507`) supplemental Table S1.

Companion files: `WILCOX_COMPARISON_2026-09-05.md`, `wilcox_gap_donors.csv`,
`sweep_candidates_2026-09-05.csv`.

---

## 1 · Fix in `DATA_PROVENANCE.md` (bookkeeping — do first, cheap)

| # | Issue | Action |
|---|---|---|
| F1 | **Disease split over-counts by exactly 1,000.** MF 550,879 + SS 445,211 + CTCL_other 146,589 + HC 32,015 = 1,174,694 vs stated 1,173,694. Only failing sum in the doc. | `atlas_obs_full.parquet.disease.value_counts()`; correct. Quote no disease-stratified number until fixed. |
| F2 | **§2's D4 exclusion rationale is false.** "already integrated in the atlas … Redundant" — none of `GSE173205`'s 10 donors are in our atlas (proven against Wilcox Table S1). | Rewrite §2. Move D4 from *excluded* to *ingest*. See §3 T1-a. |
| F3 | **`phs002933` misattributed.** Doc credits "Gaydosik 2022 OncoImmunology". It is Su T, Duran GE, … Khodadoust MS, *OncoImmunology* 2022;11:2115197 — **Stanford**. | Fix attribution. Wrong PI on a dbGaP application = rejection. |
| F4 | **`E-MTAB-14559` used for two data types** — §1 (Li scRNA, 419,579 cells) and §5 (15 healthy Visium, ERP165340). All three E-MTABs are Li's, so it's a which-is-which error. | Resolve against BioStudies; rewrite §1/§2/§5 so each accession appears once. Also record that the §1 object is a *processed* file, not the ArrayExpress FASTQ. |
| F5 | **§0 stale**: says "8 further standalone GEO cohorts"; §1 lists 9 non-Li members. Predates buus25 (added 2026-07-21). | Fix §0 and `atlas_joint/README.md` in one pass. |
| F6 | **§1b blood gap undocumented.** Skin: 749,510 = Skin compartment exactly. Blood: 302,362 vs 423,042 → 120,680 cells dropped with no stated rule. | One sentence naming the cell-level filter. The asymmetry with skin reads as loss. |
| F7 | **§3 spatial exclusion conflates two technologies.** The "GeoMx DSP-WTA, ROI-level pseudobulk" rule is correct for Danielsen/Choi/Amechi but wrongly swallows CosMx (Jung 2025), which is single-cell-resolution. | Split the rule: *ROI-pseudobulk (exclude)* vs *single-cell imaging spatial (in scope, track)*. |
| F8 | Minor | `SRS9777814–830` is 17 accessions vs "13 of 15 channels"; DestVI ref 415,742 vs Li's 419,579 (3,837 unexplained); brunner24 46.2 % retention — state whether pre- or post-dedup. |
| F9 | Add `E-MTAB-8142` to the Li reference-set ledger (alongside `GSE168508` / `GSE121212`). | Not a cohort; §2's inventory of Li inputs is incomplete without it. |

Confirmed correct, no action: B1⊂B2 nesting; the TRUST4 recovery (Li's own methods confirm
`PRJNA754592` deposited BAMs only — the 13,025 mined cells are genuinely novel); Xue accessions;
the D1↔D3 dedup.

---

## 2 · Integrity checks before any ingest

**C1 — Donor-ID collisions are live. Change the merge key.**

Wilcox had to suffix institution names because bare IDs repeat across deposits:

| Bare ID | Deposit A | Deposit B |
|---|---|---|
| `MF17`, `MF18`, `MF21` | `HRA000166` (PKU) — **in our Li object, stored as bare `real_donor`** | `GSE182861` (Pittsburgh) — different patients |
| `CTCL2`–`CTCL6` | `E-MTAB-12303` (Li) — ours | `GSE146586` (Iowa) — also ours |
| `SS1`–`SS4` | `GSE182861` (Pittsburgh) | `GSE207679` (Moffitt) |

Row 2 is already safe (`B2__` / `Li2024_atlas__` prefixes). **Rows 1 and 3 are not.** Before
ingesting `GSE182861` or `GSE207679`, key on `(accession, donor)` — never bare `real_donor`.
Retro-namespace the Li PKU donors.

**C2 — D1 ↔ Li.** `GSE266862` declares in its own data-availability section that HC *and some
CTCL* samples overlap `GSE173205` and `GSE222840`. Now that `GSE173205` is an ingest target this
becomes live: check D1 barcodes against the new `GSE173205` cells, not just against D3.

**C3 — Vienna family.** `GSE165623` (D5, n=1) / `GSE173205` (new) / `GSE266862` (D1) /
`GSE269981` (D3) / `GSE247047` (new) all come from Brunner/Jonak. Run one barcode-overlap matrix
across the family rather than pairwise. `GSE222840` (Alkon prurigo/AD) is not a CTCL cohort but
**is** the shared healthy-control reference for D1 and `GSE247047` — dedup HCs against it.

**C4 — WashU family.** If Dorando is ingested (§3 T3), overlap-check against B1 (`GSE124899`) and
B2 (`GSE146586`); Borcherding co-authors all three, same clinic. `GSE132053` is a 2019-vintage
accession number, which is what a shared older deposit looks like.

---

## 3 · Datasets to add

### Tier 1 — confirmed by Wilcox Table S1, open, patients we verifiably lack (**+36**)

Full per-donor metadata (sex, age, stage, LCT, treatment, platform) in `wilcox_gap_donors.csv`.

| Accession | + patients | Tissue | Source | Note |
|---|---|---|---|---|
| `GSE173205` | **10** | skin | Vienna (Rindler *Mol Cancer* 2021) | The F2 correction. Donors `MF309 MF311 MF312 MF318 P65 P73 P84 P90 P107 P138`. Stages IA–IVA1, treatment-annotated. |
| `GSE197619` | **11** | blood | Yale (Ren 2023 *Blood Adv*) | `P1`–`P11`. All heavily pre-treated (ECP, bexarotene, IFN-α/γ, moga, vorinostat). |
| `GSE207679` | **7** | blood | Moffitt/USF (Harro 2023 *Blood Adv*) | **All treatment-naive.** The progenitor-origin cohort — read the paper, it bears on the mouse HSPC arm. |
| `GSE182861` | **7** | skin+blood | Pittsburgh (Gaydosik) | `MF17 MF18 MF21 MF24 SS1 SS3 SS4`. **Apply C1 before merging.** |
| `GSE309805`, `GSE309807` | 1 | skin | Michigan in-house (Wilcox) | `P1_UM`, MF IVA + LCT, 10x Flex, **paired DMSO / pacritinib ex vivo** — a perturbation design absent from our corpus. Also check whether their harmonized atlas object is deposited here. |

```bash
# GEO supplementary matrices
for G in GSE173205 GSE197619 GSE207679 GSE182861 GSE309805 GSE309807; do
  wget -r -np -nH --cut-dirs=4 -A '*.tar,*.gz' \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/${G%???}nnn/${G}/suppl/"
done
```

### Tier 2 — verify first, may not be gaps

- **`GSE272005`** (Cornell, 8 skin) — donors `PT11 PT35 PT47 PT50 PT52 PT53 PT55 PT56` are an
  exact 8/8 match to our Li `MDA` sub-cohort. **Not new patients.** But it is a GEO re-deposit of
  the cohort whose SRA record (`PRJNA754592`) held only BAMs. *Check the supplementary file list
  for CellRanger VDJ output* — if present it replaces the TRUST4 route for those 13,025 cells and
  removes a methods caveat. This also closes sweep item A5.
- **`GSE269981`** — Wilcox extracted 8 CTCL donors; we hold 5. `P178` (blood, IVA1) is in the gap
  CSV; identify the other two.
- **`GSE206123`** — Gaydosik's co-accession, *not* used by Wilcox. Triage against D6
  (`GSE128531`) and against `GSE182861` before assuming it adds anything.

### Tier 3 — from the sweep, accession confirmed

- **`GSE247047`** — Alkon/Chennareddy 2024 *JACI*, parapsoriasis → early-stage MF spectrum with
  TCRB/TCRG clonality. Open. Fills the early-stage gap; corpus is advanced/leukemic dominated.
  Declares partial overlap with `GSE173205` + `GSE222840` — fold into the C3 matrix.
- **Dorando 2026** *Blood* 147(21):2503 — `GSE132053`, `GSE290264`, `GSE290557` + Zenodo
  `17537147`, `17517259`. 99 serial skin/blood/LN samples from 34 patients. Serial-through-therapy
  is a design we lack. **Do C4 first.**

### Tier 4 — accession not yet located, chase before ingesting

Ordered by value. Details in `sweep_candidates_2026-09-05.csv`.

1. **Meledathu 2026** *Blood Adv* 10(14):5163 — lymphomatoid papulosis vs advanced CTCL, all
   comparators documented-lethal-outcome. Entity absent from corpus. Same lab as `GSE266862`.
2. **Chennareddy 2025** *JACI* 155(3):892 — 8 erythrodermic CTCL + paired blood/skin. Distinct
   paper from our D1.
3. **Harro 2023** — already have the accession via Tier 1, but read the paper.
4. Costanza 2025 (LyP + pcALCL + CODEX + methylation), Luo 2024 (20 pts, CDK9),
   Cabrera-Perez 2025 (dupilumab mechanism — pairs with B4), Jung 2025 (CosMx, needs F7 first).

An author-level accession sweep for the Vienna group (Brunner/Jonak) resolves items 1, 2 and
`GSE247047` in one pass — that lab declares its reuse consistently.

### Reference sets Wilcox used that we may want

`E-MTAB-8142` (8 healthy skin), `E-MTAB-10026` (28 healthy PBMC), `E-MTAB-11536` (2, LN/blood),
`GSE159929` (2), `GSE229279` (1). Only if we want a matched healthy-blood denominator; our blood
arm currently has no comparable normal reference.

---

## 4 · Positioning (affects the grant text, not the code)

Wilcox is **2,083,007 cells / 116 patients / 126 biopsies**, published 2026-06-30, and claims
"largest available scRNA-seq CTCL atlas" in print. Ours is 1,173,694 cells / 116 donor IDs /
149 samples. Overlap is 81 of their 116 patients — neither atlas contains the other.

Do not compete on cell count. What we hold that they have nothing comparable to:

- **Spatial** — 23 Visium sections + DestVI. They have none.
- **geskin26 (`GSE290850`, 260,455 cells) and buus25 (`GSE284075`, 127,861)** — our two largest
  members, absent from their atlas entirely. Plus borcherding19 (`GSE124899`).
- **Deeper on shared sources** — `GSE266862` 22 vs 13, `GSE128531` 9 vs 5, Li object 36 vs 18.
- **FFPE/Flex at scale** (10 samples vs their 1) and **ECCITE** (14).
- **A malignant call with a measured error rate** — 0.981 precision / 0.834 recall vs Li's labels.
  They use clonotype >100 cells + inferCNV + Numbat with no external validation set.
- **Audited provenance** — supersession ledger, per-cohort QC proxies, documented dedup history.

Tiers 1–3 take us to ~153 CTCL patients and make ours a strict superset on every source except
their in-house biopsy.

**Two of their results bear on the mouse arm** and are worth reading, not just mining: their
myeloid meta-program MP2 is *"Endocytosis (MARCO, **APOE**)"* in human CTCL macrophages — the
APOE/LAM through-line appearing independently — with expanded C1QA/STAB1 macrophages and S100A8
monocytes under alternative polarization driven by IL-10→STAT3. Their CellPhoneDB v5 + NicheNet
edges (`CSF1–CSF1R`, `IL10–IL10R`, `IFNG–IFNGR`, `CCL5–CCR4`, `CD274–PDCD1`) are a useful external
check on our LR blacklists.

---

## 5 · Confidence ledger entry

> **Corpus completeness — probable, not solid.** Five literature sweep rounds (2026-09-05) across
> repository, technology, group, entity, treatment and temporal axes; discovery curve flat by
> round 5. Independently cross-checked against a second integration atlas (Wilcox, *Blood* 2026,
> 116 patients, >2M cells), which surfaced 36 patients across 4 sources the sweep had missed or
> mis-excluded. **No programmatic repository query has been run at any point** — controlled-access
> holdings without publications remain undiscoverable by this method. ~12 sweep candidates still
> have unresolved accessions.

Still open, in value order: (1) scripted GEO/ArrayExpress/ENA/GSA/dbGaP sweep using the
Cancers 2025;17(17):2921 Appendix A Boolean; (2) author-level queries for Brunner/Jonak (Vienna),
Fuschiotti/Akilov (Pittsburgh), Payton/Borcherding (WashU), Bagot (Saint-Louis); (3) EGA browse
from `EGAS00001005229` to resolve the two "EGA (presumed)" entries in §3.
