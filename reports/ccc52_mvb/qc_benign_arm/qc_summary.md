# QC of the benign CD4 arm

`jobs/run_ccc52_qc.py`, 2026-11-22. Read-only; the v1 arms were re-derived and asserted identical to `../row_labels.npz`.

## Flagged benign clusters (F) -- any of CD8A, CD8B, TRDC, KLRD1, FOXP3 detected in >= 0.05 of the cluster

| tissue | benign cells | flagged | % | CD8 | gd/NK | Treg | remaining | CD8 set >=0.25 | >=0.50 |
|---|---|---|---|---|---|---|---|---|---|
| blood | 27,634 | 15,897 | 57.5% | 8,287 | 1,768 | 8,631 | 11,737 | 5.7% (42) | 2.6% (24) |
| skin | 131,330 | 125,780 | 95.8% | 107,207 | 22,440 | 36,411 | 5,550 | 21.9% (161) | 9.4% (81) |

## Across-donor median detection by group (B)

| tissue | group | cells | med UMI | CD4 | CD40LG | IL7R | CD3E | CD8A | CD8B | TRDC | KLRD1 | NKG7 | FOXP3 | IL2RA | KRT14 | LYZ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blood | CD4_Treg_like | 6,208 | 4563 | 0.358 | 0.086 | 0.677 | 0.976 | 0.008 | 0.014 | 0.004 | 0.002 | 0.024 | 0.402 | 0.263 | 0.000 | 0.003 |
| blood | CD4_benign | 27,634 | 4411 | 0.310 | 0.283 | 0.877 | 0.963 | 0.031 | 0.031 | 0.008 | 0.011 | 0.085 | 0.027 | 0.064 | 0.000 | 0.003 |
| blood | CD4_malignant | 182,329 | 4290 | 0.311 | 0.202 | 0.808 | 0.932 | 0.000 | 0.000 | 0.000 | 0.000 | 0.011 | 0.005 | 0.036 | 0.000 | 0.002 |
| blood | CD8 | 27,274 | 3854 | 0.139 | 0.106 | 0.536 | 0.899 | 0.240 | 0.165 | 0.078 | 0.380 | 0.694 | 0.005 | 0.010 | 0.000 | 0.001 |
| blood | NK | 553 | 4101 | 0.000 | 0.000 | 0.061 | 0.176 | 0.144 | 0.016 | 0.346 | 0.959 | 1.000 | 0.000 | 0.000 | 0.000 | 0.020 |
| skin | CD4_Treg | 35,736 | 2479 | 0.356 | 0.059 | 0.500 | 0.853 | 0.006 | 0.000 | 0.000 | 0.000 | 0.024 | 0.402 | 0.357 | 0.057 | 0.010 |
| skin | CD4_benign | 131,330 | 1982 | 0.162 | 0.137 | 0.596 | 0.732 | 0.146 | 0.041 | 0.035 | 0.104 | 0.258 | 0.083 | 0.124 | 0.070 | 0.015 |
| skin | CD4_malignant | 198,620 | 3412 | 0.183 | 0.195 | 0.411 | 0.773 | 0.010 | 0.000 | 0.001 | 0.003 | 0.053 | 0.010 | 0.124 | 0.072 | 0.017 |
| skin | CD8 | 43,152 | 1756 | 0.027 | 0.020 | 0.346 | 0.797 | 0.636 | 0.247 | 0.018 | 0.146 | 0.848 | 0.005 | 0.016 | 0.065 | 0.015 |

## Conclusion

The benign CD4 arm is not a clean conventional-CD4 population, and what it carries is **real cells, not ambient RNA and not doublets**. In skin 161 benign-arm clusters (21.9% of benign cells) reach CD8A/CD8B detection >= 0.25 and 81 reach >= 0.50 -- whole clusters at det_CD8A 0.85-0.97 with det_NKG7 ~0.9 and det_CD4 <= 0.16, i.e. CD8/NK clusters that nb21's per-cluster hand map assigned to CD4 by default (see `qc_answers.md` section 1). Three readings exclude the alternatives. (i) Inside the benign arm CD4 is detected in 3.2% of CD8A/B+ cells against 20.8% of CD8A/B- cells (5.6% vs 33.4% in blood); ambient and doublets both ADD CD8 transcripts to cells that keep their CD4, so they predict the opposite -- and in the malignant arm, where the few CD8A/B+ cells are exactly that artefact, the two are close (13.9% vs 19.0%). (ii) The benign arm is the SHALLOWER arm (skin median 1982 UMI against 3412) yet its CD8A detection is 14x higher; ambient scales with depth. (iii) Scrublet scores are no higher in CD8A/B+ benign cells than in CD8A/B- ones (skin 0.0260 vs 0.0287), and predicted doublets were deleted upstream, so the 0.25 cut-off is empty by construction. Protein agrees where it exists (buus25, 8 blood donors): the benign arm is CD4-high by ADT like the malignant arm, but a minority of its cells sits on the CD8 side of the donor's own CD4/CD8 midpoint. A second, smaller admixture is Treg: skin FOXP3 detection is 0.083 in the benign arm against 0.010 malignant although CD4_Treg is a separate skin label, and those FOXP3+ benign cells are CD40LG-low (5.3% vs 14.3%), which is Treg phenotype rather than FOXP3 noise. Blood is the same picture at roughly a quarter the size.

**Consequence for v1.** CD8A, CD8B, KLRD1, TRDC and FOXP3 are artefacts of the arm definition and must not be read as malignant-vs-benign biology. Dropping only the whole mislabelled clusters (detection >= 0.25, which keeps 62/71 skin and 37/38 blood donors) moves skin KLRD1 -4.94 -> -1.25, TRDC -3.61 -> -1.34, FOXP3 -3.54 -> -2.08 and CD8A -4.53 -> -3.84, while the reported findings barely move: TOX 1.17 -> 1.49, CD70 1.18 -> 1.97, blood KIR3DL2 4.16 -> 4.22, blood DPP4 -4.36 -> -4.38. The brief's own 0.05 cut-off is too loose to work with -- it removes 96% of the skin benign arm and leaves 14 donors, a self-selected set -- so variant (c) at 0.25 is reported beside it in `qc_sensitivity.csv`. **Confidence: high** that a CD8/NK and Treg admixture is present and drives the cytotoxic and Treg genes; **moderate** on its exact size, because the CD8 and NK reference groups are themselves mixtures (`README.md:85-91`) and the 0.25 cut-off is a judgement, not a calibration.
