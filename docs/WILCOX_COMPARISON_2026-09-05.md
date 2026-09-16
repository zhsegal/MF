# Head-to-head: our atlas vs. Wilcox *Blood* 2026

**Sources:** `atlas_samples_manifest.csv` (149 samples) vs. Wang/Wilcox *Blood* 2026,
`blood.2025032507`, supplemental Table S1 (177 rows) and supplemental Figure 1B.
**Companion:** `wilcox_gap_donors.csv` — the 37 patients they have and we don't, with full metadata.

---

## 0 · The answer

**They have 37 CTCL patients we don't. We have 47 donor-units they don't.** Neither atlas
contains the other. The overlap is 81 of their 116 patients.

The single most consequential finding is not a new dataset — it is that **`GSE173205` was
excluded from our atlas on a false premise**, and their manifest proves it.

| | Ours | Theirs |
|---|---|---|
| Cells | 1,173,694 | 2,083,007 (skin 1,538,561 / blood 544,446) |
| CTCL patients | 116 donor IDs (170 counting geskin's 65 HTO pseudo-donors) | 116 |
| Samples/biopsies | 149 | 126 |
| Healthy/reference donors | 10 samples | 50 donors, 5 accessions |
| CTCL source accessions | 10 | 13 + 1 in-house |
| Compartments | Skin 99 / Blood 49 / LN 1 | skin 102 / blood 72 / LN 3 |
| Spatial | 23 Visium sections | none |
| TCR+ cells | 267,764 in-atlas (360,794 clone table) | 539,176 clonal T cells (TCR + inferCNV + Numbat) |

Their larger cell count is partly reference data: `E-MTAB-10026` alone contributes 28 healthy
PBMC donors. On CTCL patients the two atlases are the same size, from substantially different
patients.

---

## 1 · The correction that matters most

### `GSE173205` — 10 patients, excluded from our atlas for a reason that is not true

`DATA_PROVENANCE.md` §2 drops D4 (Rindler, *Mol Cancer* 2021) with the note *"already integrated
in the atlas … Redundant."* Round 2 of the sweep flagged this as uncertain because Li 2024's data
availability does list `GSE173205`.

**The manifests settle it. Not one of the ten donors is anywhere in our atlas:**

`MF309`, `MF311`, `MF312`, `MF318`, `P65`, `P73`, `P84`, `P90`, `P107`, `P138`

Li cited `GSE173205` as a comparison dataset; it was never integrated into the 419,579-cell
object we loaded. Wilcox treats it as a standalone source of 10 skin donors and their donor IDs
confirm no collision with our `chennareddy25`, `brunner24` or `li24` members.

Stages IA/IB/IIB/IVA1 with full treatment annotation — and it is a Vienna cohort, so it dedups
cleanly against the Vienna material we already hold.

**This is the highest-value single fix available: 10 patients, open GEO, no application needed.**

---

## 2 · Their three accessions we never found in five sweep rounds

All three resolve accessions the sweep listed as "NOT LOCATED".

### `GSE197619` — Yale, 11 blood donors → **this is Ren 2023** (candidate R3_05)
`P1`–`P11`, all blood, 5′. Every patient heavily pre-treated: ECP, bexarotene, IFN-α, IFN-γ,
vorinostat, mogamulizumab. Stages IA/IIB/IVA. A treatment-annotated blood cohort — the axis our
corpus is thinnest on outside B2 and B5.

### `GSE207679` — Moffitt / Univ. South Florida, 7 blood donors → **this is Harro 2023** (R3_06)
`MF1`–`MF3`, `SS1_MOFFIT`, `SS2`, `SS3_MOFFIT`, `SS4_MOFFIT`. **All seven are treatment-naive
("none").** Ten of the 37 gap patients are treatment-naive and seven of them are here.

That matters twice over. Treatment-naive baseline is nearly absent from both atlases, and this is
the cohort behind the claim that Sézary syndrome originates from heavily mutated hematopoietic
progenitors — the human statement of the premise your mouse HSPC arm is built on.

### `GSE272005` — Cornell, 8 skin donors → **NOT a gap, but useful**
`PT11`, `PT35`, `PT47`, `PT50`, `PT52`, `PT53`, `PT55`, `PT56` — an exact 8/8 match to our Li
`MDA` sub-cohort. Same patients, all LCT-positive.

The value is elsewhere: §4 records that `PRJNA754592` deposited only BAMs, so V(D)J FASTQs could
not be restored, which is why the TRUST4 workaround exists. `GSE272005` is a **GEO re-deposit of
the same eight donors**. If it carries standard CellRanger VDJ output, it supersedes the workaround
for those 13,025 cells. Worth one look at the GEO supplementary file list.

This also closes sweep item A5: Li did not absorb "8 of 16" Song biopsies arbitrarily — the eight
PT donors are the transformed-CTCL scRNA set, and Wilcox took exactly the same eight.

---

## 3 · Donor-ID collisions — read this before any ingest

Wilcox's Table S1 is, incidentally, a **collision map for this literature**. They had to add
institution suffixes because bare IDs repeat across deposits:

| Bare ID | Appears in | And in |
|---|---|---|
| `MF17`, `MF18`, `MF21` | `HRA000166` (PKU — **inside our Li object**) | `GSE182861` (Pittsburgh — different patients) |
| `CTCL2`–`CTCL6` | `E-MTAB-12303` (Li — ours) | `GSE146586` (Iowa — also ours) |
| `SS1`–`SS4` | `GSE182861` (Pittsburgh) | `GSE207679` (Moffitt) |

Our manifest is already protected on the second row: `borcherding23` donors carry a `B2__` prefix
and Li's carry `Li2024_atlas__`. **The first row is the live hazard.** Our Li PKU donors are stored
with bare `real_donor` values `MF17`, `MF18`, `MF21`. Ingesting `GSE182861` and keying on
`real_donor` would silently merge three pairs of unrelated patients — a Chinese cohort with a
Pittsburgh one.

Note the earlier sweep's tentative read that "Li absorbed Pittsburgh data" was wrong. It was this
collision. Wilcox's `_PKU` suffix is the tell.

---

## 4 · What we have that they don't

Three sources are absent from their atlas entirely — **our two largest members among them**:

| Member | Accession | Donors | Cells | TCR+ |
|---|---|---|---|---|
| geskin26 (dupilumab) | `GSE290850` | 11 lanes / 65 pseudo | 260,455 | 69,386 |
| buus25 | `GSE284075` | 17 | 127,861 | 70,169 |
| borcherding19 | `GSE124899` | 2 | 14,253 | 0 |

We are also deeper on three shared sources — `GSE266862` (22 vs 13), `GSE128531` (9 vs 5), and the
Li object (36 vs 18, because they took only Li's own 18 and re-downloaded `HRA000166` separately
rather than using the integrated object).

And structurally we hold four things they have nothing comparable to:

- **Spatial.** 23 Visium sections with DestVI deconvolution. Their atlas is scRNA/scTCR only.
- **FFPE / 10x Flex at scale.** 10 samples vs their single in-house Flex biopsy.
- **ECCITE.** 14 samples.
- **Recovered V(D)J.** The 13,025 TRUST4-mined cells, plus a dominant-clone malignant call
  validated at 0.981 precision / 0.834 recall against Li's own labels. They call malignancy by
  clonotype >100 cells plus inferCNV plus Numbat, with no external validation set — a defensible
  method, but ours is the one with a measured error rate.

Net ours-only: **47 donor-units, ~493,000 cells.**

---

## 5 · Where they are deeper

- **`GSE269981`** — they have 8 CTCL donors, we have 5. `P178` (blood, IVA1) is in
  `wilcox_gap_donors.csv`; the other two need checking against our `chennareddy25` overlap handling.
- **`GSE171811`** — they list 12 units to our 8, but their IDs are blood/skin-split
  (`SS1B`/`SS1S`), so this is the same 8 patients counted per-compartment. Not a gap.
- **In-house `P1_UM`** — Michigan, MF stage IVA with LCT, 10x Flex, and a **paired DMSO vs
  pacritinib ex vivo treated biopsy**. Deposited at `GSE309805` / `GSE309807`. A drug-perturbation
  paired design does not exist anywhere in our corpus.

---

## 6 · Two things in their results that bear directly on the mouse arm

Not inventory, but the reason to read the paper rather than just mine its manifest.

**Their myeloid meta-program MP2 is "Endocytosis (MARCO, APOE)."** In human CTCL skin
monocytes/macrophages. That is the APOE/LAM through-line you carry as a cross-project link,
appearing independently in a human disease atlas. Their expanded populations are C1QA and STAB1
tissue-resident macrophages and S100A8 monocytes, all enriched for alternative polarization, driven
by an IL-10 → STAT3 axis that they knock down pharmacologically.

**Their CCC output overlaps yours structurally.** CellPhoneDB v5 plus NicheNet, with `CSF1–CSF1R`,
`IL10–IL10R`, `IFNG–IFNGR`, `CCL5–CCR4` and `CD274–PDCD1` as the malignant-to-myeloid edges.
Worth comparing against your blacklists — their edge set is a useful external check on whether the
marker-receptor tautology you catalogued shows up in an independently built pipeline.

---

## 7 · Actions

1. **Fetch `GSE173205`** (10 patients). Correct §2 of `DATA_PROVENANCE.md` — the exclusion
   rationale is wrong, and a reviewer with both manifests can see it.
2. **Fetch `GSE197619`** (11, Yale, treated blood) and **`GSE207679`** (7, Moffitt,
   treatment-naive blood).
3. **Fetch `GSE182861`** (7, Pittsburgh) — **and namespace by accession, not donor ID**, or you
   will merge `MF17/18/21` into the PKU donors of the same name.
4. **Check `GSE272005`** for CellRanger VDJ output. If present it replaces the TRUST4 route for the
   eight `PT` donors and removes a methods caveat.
5. **Check `GSE309805` / `GSE309807`** — their deposit. If the harmonized object is there, it is
   both a benchmark and a shortcut.
6. **Recheck `GSE269981`**: they extracted 8 CTCL donors, we took 5.
7. **Rewrite the scale claim.** "Largest available" is theirs in print. Ours is the atlas with
   spatial, FFPE, ECCITE, recovered V(D)J and a validated malignant call — position on
   modality breadth and provenance rigour, not cell count.

Adding items 1–3 and the in-house donor takes us to **~153 CTCL patients**, against their 116, and
makes ours a strict superset of theirs on every source except their in-house biopsy.
