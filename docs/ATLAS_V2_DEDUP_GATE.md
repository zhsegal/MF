# Atlas v2 — Phase A dedup gate: results

**Date:** 2026-09-05. **Inputs:** 14 newly downloaded GEO accessions + the 8 held cohorts whose
raw deposits are on disk. **Code:** `check_overlap.py` → `tables/atlas_dedup_v2.csv`;
`make_dedup_decisions.py` → `tables/atlas_dedup_v2_decisions.csv`.

---

## 0 · Method

`atlas_join_helpers.detect_duplicate_cells` (`:1032`) claims in its docstring to compare cleaned
barcodes. It does not — it counts cells by bare `real_donor` for D1/D3 against a hardcoded
6-name list, so it can confirm an overlap you already know about but never discover one. This
does the comparison, straight off the raw deposits.

**Metric.** For every pair of donor units, `containment = |A∩B| / min(|A|,|B|)` plus
`enrichment = |A∩B| / (|A|·|B| / 737,280)` — observed over expected from the 10x v3 whitelist.
Containment alone is size-biased (a 400-barcode set is ~10 % contained in a 40k set by pure
chance); enrichment removes that.

**Two things had to be fixed before the comparison meant anything:**

1. **Barcode normalisation.** `_clean_bc` (`:246`) strips only the `-N` lane suffix. Deposits
   also *prefix* the sample: D6 writes `SC50nor_AAACCTGAGCCCGAAA`. Without stripping that, D6
   shares zero barcodes with everything and its genuine re-deposit is invisible. `clean()` now
   reduces any string to its 16-nt core.
2. **Unfiltered `.h5`.** `GSE182861` / `GSE206123` ship `raw_feature_bc_matrix.h5`, which holds
   the *entire* 737,280-barcode whitelist — every comparison against one is trivially 1.000.
   Detected by size and cell-called from `indptr` alone (genes per barcode ≥ 200, matching
   `qc_filter`'s own default).

**All pairs, not a candidate list.** 39,063 pairs with any overlap were scored across all 21
cohorts. Restricting to a hand-written candidate list is precisely how the `GSE173205` patient
count was got wrong; the family map is now a *result*, not an assumption.

**Separation is unambiguous.** 89 DUPLICATE (containment 0.478–1.000, enrichment 20–629×),
38,972 chance (≤0.061, ≤1.5×), **2 in the review band, 0 requiring a judgement call.**
Positive control: `D1/P112 ↔ D3/P112_HC1` reproduces the known n=6,400 / 100 % result.

---

## 1 · The headline correction

**`GSE173205` (D7) contributes 0 of its 4 `MF###` donors — all four are already in the atlas
under Vienna's other naming convention.**

| D7 unit | is | held as |
|---|---|---|
| `MF309_followup` | `D1/P76` | in atlas |
| `MF311_thick` | `D1/P311_thick` | in atlas |
| `MF312_thick` | `D1/P312_thick` | in atlas |
| `MF318_thick`, `MF318_thin` | `D1/P318_thick`, `D1/P318_thin` | in atlas |
| `P112/P115/P116/P121_HC` | `D1/P112` … | in atlas |

Containment 0.987–1.000, enrichment 35–166×. The Wilcox comparison concluded "not one of the ten
donors is anywhere in our atlas" by string-matching bare donor IDs across two manifests. Vienna
files the same patient as `MF309` in one deposit and `P76` in another — **the handoff's own C1
hazard, in the opposite direction.**

**D7's real contribution is 6 new patients (`P65 P73 P84 P90 P107 P138`) and 13 new libraries** —
the extra libraries being lesional/non-lesional and thin/thick biopsies of patients we already
hold, which is a within-patient design the corpus does not otherwise have. `GSE173205` ships
**no V(D)J**.

## 2 · Other confirmed identities

**Five-way deposits.** The four Vienna healthy controls `P112 P115 P116 P121` are published in
**D1, D3, D7, D8 and D9** — five separate accessions, all containment ≥0.996.

**A duplicate already inside v1.** `D5/skin` == `D1/P303_skin`, containment 1.000, n=6,327 —
**and both are in the built atlas**, as `D5__MFIVB_skin` and `D1__P303_skin`. v1 counts one
patient's skin biopsy twice under two donor IDs. `DATA_PROVENANCE.md` §6 flagged this as a
possibility ("D5 is n=1; its skin fraction may overlap an atlas Rindler donor"); it is confirmed.
`D5/PBMC` == `D3/P303_Blood`, which v1 did not ingest, so the blood copy is unique.

**Pittsburgh renames, none guessable from IDs.**
`D13/SZ29` == `D15/MF25` (0.991) · `D13/SC374` == `D15/PF2_3` (0.899) ·
`D6/SC157` == `D14/MF6` · `D6/SC158` == `D14/MF8` · `D6/SC205` == `D14/MF12` (all ≥0.999).
The last three matter beyond dedup: `SC157/SC158/SC205` are the three D6 donors carrying
**`PROVISIONAL subtype — verify`** in `data/D6_gaydosik2019_skin/meta/samples.tsv`, defaulted to
MF because the deposit had no label. `GSE206123` re-publishes them as `MF6 / MF8 / MF12` with the
paper's metadata — this resolves a `DATA_PROVENANCE.md` §6 provisional-label item.

**`B4/SZ30` == `D13/SZ30`** (0.932) — the collision predicted from the manifests, confirmed.

**`GSE272005` (D16) is the same 8 `PT` patients as Li's MDA sub-cohort**, and Li's object already
contains *both* the Tumor and PP libraries (both match Li's single unit). So D16 adds no cells —
its value is entirely the `scVDJ.zip`, including for **`PT47` and `PT50`, which have no TCR at
all today**.

## 3 · Confirmed non-collisions (the gate's negative results)

- **`B7/SS1`–`SS4` (Moffitt) are not `B4`/`D13`'s `SZ*` (Pittsburgh)** — different patients.
- **`B6/P1`–`P11` (Yale) are not `D11/P1`–`P6` (brentuximab)** — two new cohorts using the same
  bare IDs for different people. Namespacing by deposit is load-bearing, not cosmetic.
- **`B8` (Dorando) does not overlap `B1`/`B2`** despite the shared clinic.
- No cross-family duplicate anywhere in 39,063 pairs.

---

## 4 · Net contribution after dedup

| cohort | units | dup | keep | donors | barcodes (pre-QC) |
|---|---:|---:|---:|---:|---:|
| D7_rindler2021mc_skin | 27 | 14 | 13 | 7 | 124,378 |
| D8_alkon2024_parapsoriasis | 24 | 10 | 14 | 14 | 190,030 |
| D9_lyp_vs_ctcl_skin | 14 | 7 | 7 | 7 | 108,008 |
| D10_jonak2021_discordant | 2 | 0 | 2 | 1 | 13,624 |
| D13_gaydosik2022_skin_blood | 12 | 1 | 11 | 11 | 108,716 |
| D14_gaydosik2023_skin | 12 | 7 | 5 | 5 | 55,783 |
| D15_il4ra_blockade_skin | 23 | 4 | 19 | 16 | 164,012 |
| D16_song2024_transformed | 15 | 12 | 3 | 3 | 6,441 |
| B6_ren2023_blood | 14 | 0 | 14 | 14 | 149,023 |
| B7_harro2023_blood | 7 | 0 | 7 | 7 | 139,127 |
| B8_dorando2026_blood | 15 | 0 | 15 | 6 | 111,576 |
| D11_brentuximab_skin | 13 | 0 | 13 | 6 | 140,093 |
| D12_pacritinib_skin | 2 | 0 | 2 | 1 | 64,583 |
| **total** | **200** | **75** | **125** | **98** | **1,375,394** |

Donor counts are per-cohort; `P65` and `P90` appear in both D7 and D8 and are counted once.
Not all 98 are CTCL patients — `B6/N1-N3`, `D13/HB`, `D15/NS13,14,16,17` are controls, `D8`'s
`SPD*` are small-plaque parapsoriasis, `D10/PCFCL` is a B-cell lymphoma lesion. Entity
assignment is a §4 metadata task, not a gate output.

**Scale implication for the rebuild.** ~1.38 M pre-QC barcodes on top of 1.17 M post-QC cells —
so v2 lands somewhere near **2.2–2.4 M cells and ~274 samples**, roughly double v1. v1's measured
peaks were 92 GB for the concat and 70–82 GB for MrVI; both must be re-provisioned, not reused.

---

## 4b · Findings the sample-sheet build added

- **`GSE206123` (D14) contributes no new CTCL patients at all.** Its titles are
  `CTCL-6/8/12 [reanalyzed]` + `HC-1..HC-9`; the gate shows `CTCL-6 = D6/SC157`,
  `CTCL-8 = D6/SC158`, `CTCL-12 = D6/SC205` and `HC-1..4 = D6`'s four healthy skin samples.
  Net contribution: **5 new healthy skin controls** (`HC-5..HC-9`).
- **…and it resolves three provisional labels in the existing atlas.** `D6`'s
  `SC157 / SC158 / SC205` carry `PROVISIONAL subtype — verify` because GSE128531 stated no
  MF-vs-SS label (`DATA_PROVENANCE.md` §6). GSE206123 re-publishes the same libraries with
  `tumor stage`: **SC157 = IIB (F IIB T3NxB0M0), SC158 = IIB (T3N0B0M0), SC205 = IVA
  (T4NxM0B0)** — all MF, confirming the default that was applied and supplying the stage.
- **`GSE264636` has a 15th library with no sample token in its filename**
  (`GSM9038965_matrix.mtx.gz`); the series matrix names it **P220**. A filename-driven
  gate skips it entirely, so `check_overlap.py` now falls back to the GSM accession.
- **`GSE290557`'s `GSM8816535_999PB022119_barcodes.tsv.gz` is 0 bytes in GEO itself**
  (its own `filelist.txt` records size 0), so that triplet is unreadable. Excluded, with the
  reason recorded; patient 999 is still represented by `999PB040716`.
- **`GSE182861` and `GSE293752` carry hashtag-multiplexed lanes** — `HC1and2_GEX`,
  `HC3and4_GEX`, `SZ29_MF25_GEX_HTO`, `HB1_2_3_GEX_HTO`, `PF2/PF3` — which is stated only in
  the series-matrix titles, not the file names. `HTO1..HTO10` are present in the Antibody
  Capture block, so these demux with the same hashsolo path as B4. Both deposits also carry
  a **20-marker surface ADT panel** (CD3/CD4/CD8/CD25/CD194/CD30/IL4Ra/IL13Ra1…) that is
  currently dropped rather than carried to `obsm['X_adt']` as B5's is — a free modality left
  on the table, noted rather than implemented.

## 4c · Built atlas (v2)

`data/atlas_joint/joint_annotated.h5ad` — **2,157,693 cells x 42,347 genes**, 263 samples,
268 donor-units, **201 patients**, 21 datasets. 290,112 duplicate cells dropped across 45
samples, exactly as the ledger specifies. `X_adt` survives concat (127,861 cells x 128
proteins). Peak RSS 216 GB / 26 min (v1: 92 GB / 25 min).

Compartments: Skin 1,345,527 - Blood 811,024 - LN 1,142.

`tcr_clones.parquet`: **779,401 cells / 227,531 clones** (v1: 360,794 / 84,464). `li24` now
carries **17,563 TCR cells where v1 had none**, grafted from `GSE272005` -- including the
first clonotypes for `PT47` and `PT50`.

**Regression vs `data/_archive_v1/`:** all seven v1 TCR cohorts reproduce **100.0000 %** on
`is_malignant`, delta 0.

## 5 · Consequences for the code

1. **`SHARED_BRUNNER` is superseded.** It lists 6 names; the gate finds **45 identity groups /
   63 units to drop**. `concat_joint:1073` must key on `(dataset, real_donor)` pairs read from
   `tables/atlas_dedup_v2_decisions.csv`, never a bare-string `isin`.
2. **A cross-deposit patient key is now required, not optional.** The gate splits patient `P65`
   across two deposits (lesional library wins in D8, non-lesional only exists in D7), so it would
   enter the atlas as two donors, `D7__P65` and `D8__P65` — the same defect as v1's
   `D5__MFIVB` / `D1__P303`. The alias groups in the decisions table are exactly the mapping
   needed to collapse them. This was scoped out of Phase B as speculative; the gate has now made
   it evidence-backed.
3. **v1 itself needs a decision** on the `D5__MFIVB_skin` / `D1__P303_skin` double-count before
   v2 is built, or v2 inherits it.
4. `clean()` and the raw-`.h5` cell call belong in `atlas_join_helpers`, not only here.
