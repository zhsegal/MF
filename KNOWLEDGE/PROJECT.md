# What this project is

## The disease

Cutaneous T-cell lymphoma — **mycosis fungoides (MF)** and its leukaemic variant **Sézary
syndrome (SS)**. The malignant cell is a clonal **CD4 T cell**. In MF it sits in the skin; in SS
it circulates. The clone is not morphologically distinct from the patient's own reactive T cells,
which is the central analytical problem: in any sample, malignant and benign CD4 cells are mixed
together, and telling them apart is a measurement, not a label.

## The atlas

**v2: 2,157,693 cells, 14 studies, 1,345,527 skin and 811,024 blood.** Built by joining ~20 public
and in-house cohorts onto a common gene space and obs schema. Integration is **MRVI**
(`scvi.external.MRVI`) — not SemanticSCVI, which an earlier generation of this project used.

Skin and blood are analysed **separately** throughout. They differ in chemistry, in depth, in
which objects are full-gene, and in how the clone behaves; pooling them hides more than it gains.

## The estimand almost every question reduces to

Within a patient, malignant and benign CD4 cells see the same microenvironment, the same
sequencing batch and the same donor genetics. So the comparison that answers most questions is
**paired, within-patient, malignant versus benign CD4** — the patient is their own control. This
is why so much of the machinery here is about defining those two arms correctly, and why a
contaminated benign arm is not a cosmetic problem: it is the control group.

For cell–cell communication specifically this has a useful consequence. Both arms face the same
partner cells, so in a malignant-vs-benign contrast the partner term cancels and the differential
interaction *is* a paired differential expression on the CD4 side. Partner cells act only as a
gate — is the cognate gene expressed in that cell type at all.

## What we are trying to find

1. **Surface targets** for CAR-T and bispecifics: a protein high on the malignant clone, low on
   the patient's benign T cells, and low on healthy-donor T cells. The third arm matters — a
   target that spares the tumour's neighbours but ablates a healthy person's T compartment is not
   a target. This is the `40_targets` / `run_target_screen*` line of work, and the payload
   selectivity screen (`reports/51_payload_selectivity.md`).
2. **The communication the clone rewires**: which ligand–receptor genes change on the malignant
   side relative to the patient's own benign CD4, and which partner cell types can receive them
   (`reports/ccc52_mvb/`).
3. **The degradome / protease axis** in the tumour microenvironment — how the clone remodels
   matrix and what that implies for invasion.
4. **Subclonal structure**: TCR clonality (ALICE), CNV, and how subclones differ in signalling.

## How malignancy is called

Four independent calls, never OR-ed: `mal_tcr` (TCR clonality), `mal_cnv` (inferCNV),
`mal_knn` (latent label propagation), and **`mal_cluster`** — per-sample Leiden with one call per
cluster, the headline since 2026-10-22. Each ships a `_defined` mask because a cell with no TCR
is untested, not benign. See `DECISIONS.md`.

## Where the hard parts are

- **The benign arm is not clean by default.** The original lineage label was an else-branch; the
  v1 benign CD4 arm was only ~46% conventional CD4 in skin. Fixed forward by
  `nonmal_T_labels_v1`; see `CAVEATS.md`.
- **The objects are asymmetric.** Blood's T object is a 10k-HVG subset without `CD4`; skin's is
  full-gene. Nothing in the files says so.
- **Cohorts are heterogeneous.** Three chemistries including a probe panel, ex vivo cultures,
  HTO-pooled lanes, five healthy donors mislabelled as SS, and duplicate libraries between
  accessions. `helpers/target_cohort.py` encodes the donor truth; `CAVEATS.md` lists the traps.
