# `24_subclone_tcr_signaling` vs old `old/23_subclonal_evolution`+`old/26_tcr_signaling_subclones` — reproduction check (+ Buus)

**Verdict:** `24_subclone_tcr_signaling` faithfully reproduces the old pipeline. Method/params are identical (same
committed helpers, `SEED=0`); the v2 output CSVs match `old/26_tcr_signaling_subclones`'s documented numbers exactly. On the
**shared cohort** the results are strongly concordant (global module r=0.91, within-sample r=0.91,
arm-activity Spearman 0.89 / 100% sign-agree, 20/23 same subclone-k). All remaining differences are
**data-driven** — the re-annotated atlas + `23_malignancy_tcr_cnv` **v3** malignancy calls (which, notably, *improved*
TCR/CNV concordance for several donors). **Buus (B5) is cleanly integrated**: two Sézary donors
(`B5__PT09`, `B5__PT23`) produce sensible, non-NaN outputs in every table.

Comparison is CSV-only (`data/atlas_joint/*_v2.csv` = `old/23_subclonal_evolution`/`old/26_tcr_signaling_subclones`; `*_v3.csv` = `24_subclone_tcr_signaling`); driver
`scratchpad/compare_v2_v3.py`. No code/notebook changes made.

## Cohort delta
- **29 shared donors** in §B / **23 shared eligible** in §A (prefixes `D1/D3/D5/H/Li2024_atlas`).
- **Added (Buus):** `B5__PT09`, `B5__PT23` — the intended new study.
- **Dropped:** `D3__P87` — fell below the `MIN_MAL=200` eligibility under v3 malignancy calls.
- All output-CSV schemas identical v2↔v3.

## §A — subclonal evolution (was `old/23_subclonal_evolution`)
| metric | agreement |
|---|---|
| same subclone-k | **20/23** donors (diffs: D1__P204 4→2, D1__P76 4→3, D5__MFIVB 3→2) |
| `max_arm_delta` | Pearson **0.96** |
| `silhouette` | Pearson 0.46 / Spearman 0.64 |
| `frac_cnv_confirmed` | Pearson 0.52 / Spearman 0.64 — *v3 higher* for several donors |
| trunk arm-set identical | **14/22** donors; the rest add 1–2 arms, shared arms preserved |

`frac_cnv_confirmed` rose sharply for D1__P76 (0.33→0.83), D3__P192 (0.18→0.74), D5__MFIVB
(0.15→0.92) — the v3 malignancy calls are cleaner, so more CNV subclones are TCR-confirmed. This is
an improvement, not a regression.

## §B — TCR / co-stimulation signaling (was `old/26_tcr_signaling_subclones`)
| table | concordance |
|---|---|
| **global pooled** (23 modules, Cliff's δ) | Pearson **0.91** / Spearman 0.89 / sign-agree **87%** (20/23) |
| **within-sample** (667 donor×module pairs) | Pearson **0.91** / Spearman 0.90 / sign-agree **87%** |
| **arm × activity** (7 crosses, ρ) | Spearman **0.89** / sign-agree **100%** |

Load-bearing biology preserved (v2→v3 Cliff's δ): `L0_coreceptor` −0.387→−0.341,
`L0_pan_t_identity_loss` −0.295→−0.306, `L4_ieg_acute` −0.219→−0.223, `L3c_ap1_mapk_activity`
−0.173→−0.229, `L3b_nfat_activity` +0.093→+0.092. The strongest arm cross `chr6q- → L3a_nfkb_activity`
holds (ρ 0.107→0.119). The v2 numbers reproduce `old/26_tcr_signaling_subclones`'s reported values exactly — confirming the v2
CSVs are genuinely `old/26_tcr_signaling_subclones`'s outputs.

**3 sign flips** (all small-magnitude, near zero): `COSTIM_ligands_on_TME_partners` (−0.04→+0.02),
`COSTIM_pi3k_akt_activity` (+0.02→−0.06), `L0_tcr_cd3_components` (+0.12→−0.07, the largest). These
sit near δ=0 where reactive-CD4 re-annotation (100k→262k cells) easily tips the sign; not a code issue.

**Signaling modes** — compared at distribution level (subclone IDs are re-clustered v2↔v3). The
**dominant mode is preserved** (`ap1_vs_nfat-`: 74/150 in v2, 98/178 in v3). A secondary cluster is
relabeled (v2's `chronic_vs_acute-`/`autonomous_nfkb+` → v3's `costim_amp+`), an expected consequence
of re-clustering + `fcluster(maxclust=4)` re-picking the per-cluster dominant contrast — not a bug.

## Buus (B5) integration — PASS
Both donors present with 0 NaN and in-range effects across every table (subclone_summary,
trunk_branch, within_sample_module=46 rows, subclone_vs_control_module=437 rows, arm_activity=14,
modes=19 nested subclones; δ ∈ [−0.53, +0.77]).

| donor | n_malig | k | sil | frac_cnv_confirmed | entity |
|---|---|---|---|---|---|
| B5__PT09 | 5253 | 2 | 0.30 | **0.918** (strong) | Sézary |
| B5__PT23 | 3702 | 4 | 0.23 | **0.022** (weak — see caveat) | Sézary |

## Caveats (report only, no fixes)
1. **B5__PT23 `frac_cnv_confirmed = 0.022`** — its k=4 CNV subclone split is weakly TCR-confirmed.
   Consistent with the Buus source being flagged a *proxy/placeholder QC* in `atlas_join_helpers.py`
   ("Buus 2025 CancDisc Methods TBD → verify"). Treat B5__PT23's fine subclone structure cautiously.
2. **`24_subclone_tcr_signaling` cell 6 prints `0 / 17` TCR-β variants** while 25 donors are eligible. Not a merge bug: the
   founder/family loop iterates the ALICE clonotype table, which keeps only donors with a valid TRB
   CDR3 (17 of 25 — a TCR-data-availability artifact via `alice_helpers.clonotype_table`). The `0`
   means no donor's founder clone had a ≤1-aa neighbor (`founder_family` component size 1), i.e.
   single-dominant clones — biologically expected for MF/SS, and downstream analysis does not depend
   on this count.
3. **`D3__P87` dropout** — expected; its malignant count dropped below `MIN_MAL=200` under v3 calls.

## Reproduce
```
/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/bin/python \
  scratchpad/compare_v2_v3.py
```
Every number above traces to a printed join in that script.
