# NB28 — biology

*Values are lognorm CP10K. "Clone" = TCR-defined malignant CD4 cells; "reactive" = non-clonal CD4 cells from the same biopsies.*

---

## 1. The malignant clone carries no degradome

No MMP, ADAM, ADAMTS, KLK, PLAU/PLAUR/PLAT/PLG or HPSE is enriched in the clone relative to reactive T cells in the same biopsy. HPSE is higher in the reactive arm (0.0181 vs 0.0372).

What the clone does carry:
- **Protease inhibitors:** TIMP1 (Δ 0.3148, q 0.0008), CSTB (Δ 0.2604, q 0.0004), SERPINB1, SERPINB9 (both q 0.0377).
- **Lysosomal cathepsins:** CTSA, CTSD, CTSH — intracellular, not secreted.
- **Granzymes:** GZMB (q 0.0138), GZMM (q 0.0271).

SERPINB9's canonical role is protecting granzyme-B-producing cells from their own granzyme B, and the clone expresses GZMB, so the pairing reads as cytotoxic self-protection rather than immune evasion.

TIMP1 and its receptor CD63 are both clone-enriched (CD63 Δ 0.5881, q 0.0003). With no MMPs in the clone to inhibit, the TIMP-1–CD63–β1-integrin pro-survival pathway (Jung et al., *EMBO J* 2006) is the more plausible function.

Cathepsins and serpins are higher in MF CD4 cells than in healthy CD4 cells, but equally high in clone and reactive cells (CTSC 0.6750 / 0.7129 / 0.2181; CTSD 0.5566 / 0.5432 / 0.1849; CTSA 0.4025 / 0.3834 / 0.1127; SERPINB1 0.3515 / 0.2985 / 0.1013 — clone / reactive / healthy). The CD4 protease-related signal is a disease feature, not a malignancy feature.

T cells migrate amoeboidly rather than by proteolysis (Wolf/Friedl, *Blood* 2003; Lämmermann, *Nature* 2008), so a protease-null CD4 cell is the baseline expectation. What this refutes is the CTCL-specific model of malignant T cells digesting type IV collagen and laminin to cross the basement membrane. Fibroblast COL4A1–COL4A4 *increase* in MF, which is also inconsistent with active basement-membrane degradation. Epidermotropism in MF therefore needs a non-proteolytic explanation.

---

## 2. The degradome is stromal

**Fibroblasts (MF vs healthy skin)** — proteases up:
- MMP2 (F_inflammatory 1.8517 vs 1.1125; apCAF 1.5233 vs 0.9154), MMP14 (F_inflammatory 0.7550 vs 0.4295), MMP19, MMP1, MMP3, MMP9, MMP12, MMP25
- ADAMTS1/2/4/5/15 (ADAMTS2 F_structural 0.5695 vs 0.2748), ADAMDEC1 (apCAF 0.5276 vs 0.0708)
- Cathepsins B/C/H/K/L/S/Z (CTSB apCAF 1.4825 vs 0.6832; CTSK apCAF 2.2903 vs 1.9597)
- PRSS23 (F_structural 0.8563 vs 0.3542), PLAU, PLAUR, FAP (F_structural 0.5329 vs 0.3098), MME

Inhibitors split: TIMP1 and SERPINE1 up; **TIMP3 down** (F_inflammatory 0.9076 vs 1.4017; apCAF 0.5899 vs 1.0517); SLPI down. TIMP-3 is the physiological inhibitor of the ADAMTSs and MT-MMPs (Kashiwagi, *JBC* 2001; Brew & Nagase, *BBA* 2010) — precisely the proteases that rise. Net pericellular proteolytic capacity tilts upward.

Matrix: provisional matrix up (TNC apCAF 1.7965 vs 0.3566; POSTN F_papillary 2.3600 vs 0.9269; THBS1/2, FN1, COL4A1–4, fibrillar collagens); homeostatic reticular programme down (DCN, GAS6, ASPN, OGN, PRELP, FBLN1, MFAP4, MFAP5, ELN).

**Myeloid (MF vs healthy skin)** — an identity shift, not a protease shift:
- Resident programme lost: FOLR2 (Mac_FOLR2 1.5436 vs 2.0998), LYVE1 (0.0969 vs 0.6056), MRC1, MSR1, CD36 — consistent with loss of FOLR2⁺ resident macrophages (Nalio Ramos, *Cell* 2022).
- Monocyte-derived programme gained: CCL18 (Mac_FOLR2 2.1545 vs 0.3119), CCL17, CXCL9, CXCL10, IDO1, CD274, LAMP3, S100A8/A9.
- Cathepsins and MMP12 up (cDC2 0.9580 vs 0.0940); MMP9, MMP19, PLAUR, PLAU, TIMP1, TIMP3 down.
- IL1A, IL1B, TNF and IL23A all down.

MMP12 up with elastic fibres down (ELN, MFAP4, MFAP5, FBLN1) forms a self-consistent elastolysis axis, but its predicted IL-1/TNF driver is absent. Myeloid MMP9 being lower in MF contrasts with Gaydosik/Fuschiotti (*JCI Insight* 2023), who describe myeloid MMP9 acting on malignant T cells; the resident-to-monocyte-derived shift may account for it, since resident macrophages carry MMP9 constitutively.

---

## 3. The clone does not instruct the stroma

Sixteen candidate fibroblast-activating ligands were examined for their source:

| Ligand | Source | In MF | Stromal receptor |
|---|---|---|---|
| **TNFSF14 (LIGHT)** | **clone-restricted** | up | LTBR ↑, TNFRSF14 ↑ |
| LTA | MF CD4, both arms | up vs healthy | TNFRSF1A/1B ↑ |
| CD40LG | MF CD4, both arms | **down** (0.2401 vs 0.7459 healthy) | CD40 ↑ |
| OSM | myeloid | **down** in DC | OSMR ↑ |
| IL11 | undetectable | — | IL11RA **down** (apCAF 0.5260 vs 0.9997) |
| TGFB1 / TGFB3 | fibroblast, myeloid | up (small) | TGFBR2 ↑ |
| LRRC32 (GARP) | fibroblast; reactive CD4 > clone | up | — |
| ITGB8 / ITGAV | fibroblast | up | — |
| THBS1 / TGM2 | fibroblast, myeloid | up | — |
| PDGFA | clone and fibroblast | up | PDGFRB ↑ |
| PDGFB | myeloid | **down** (Mac_FOLR2 0.1613 vs 0.3739) | PDGFRB ↑ |
| SPP1 | myeloid | up (small) | — |
| ENPP2 (autotaxin) | myeloid, fibroblast | up | LPAR1 ↑ |
| ENTPD1 / NT5E | fibroblast, myeloid | up | ADORA2A/2B ↑ |
| AREG / HBEGF | myeloid | HBEGF **down** | EGFR modest |
| CXCL13 | fibroblast, myeloid | up (small) | — |
| WNT5A / JAG1 | fibroblast | up | ROR2, NOTCH1–3 |
| IL17A/F, IL22, IL26 | undetectable | — | IL17RA ↑ |

One clone-restricted ligand; eleven never expressed by CD4; two disease-general in CD4; five falling in MF. The clone's emitted repertoire is TNFSF14, PDGFA, MIF and CD70 — no TGF-β, no PDGFB, no EGF-family ligand, no IL-1 family, no detectable type-2 cytokine, no CXCL13.

**The TGF-β loop is complete within fibroblasts:** latent ligands (TGFB1, TGFB3), docking molecule (LRRC32), activating integrins (ITGAV, ITGB8), non-integrin activators (THBS1, TGM2), receptor (TGFBR2 F_reticular 0.7106 vs 0.3892) and the full output (FAP, SERPINE1, LOXL2/3, LTBP2, ADAMTS2, PRSS23, TNC, POSTN, collagens). No T-cell input is required. The output and receptor effects are large; the latent ligands and activators are small, as expected for latent TGF-β components.

The CAF programme reads as a self-sustaining autocrine TGF-β loop with myeloid inputs, not as a clone-instructed state.

---

## 4. What the clone is

Forty-three genes separate the clone from reactive T cells in the same biopsy.

**Transcription factors, clone-restricted:** GATA3 (Δ 0.5608, pct_up 0.909, q 0.0001), BATF (Δ 0.4623, q 0.0006), STAT3 (Δ 0.4380, q 0.0008), TOX (q 0.0024; 0.6404 vs 0.2719 reactive). IRF4 and RELB are up in both MF CD4 populations but do not distinguish clone from reactive. RUNX2 is not expressed differentially in CD4 cells at all.

**Glycolysis, clone-restricted:** ENO1 (Δ 0.3032, q 0.0001), PKM (Δ 0.4484, q 0.0004), HIF1A (pct_up 1.00, q 0.0074), plus PPIA, PPIB, BSG.

**Proliferation is shared.** MKI67 and HMMR are up in both MF CD4 populations versus healthy CD4 (MKI67 clone 0.1571, reactive 0.1113, healthy 0.0116) and do not distinguish clone from reactive. The clone's glycolytic state is therefore not a consequence of dividing more than its neighbours — it is metabolically distinct from equally proliferative reactive T cells.

**BSG** is clone-enriched (Δ 0.4952, q 0.0004) but neither lactate transporter tracks it: SLC16A1 is up in both MF CD4 populations without distinguishing them, and SLC16A3 is unchanged. So neither the EMMPRIN-induces-fibroblast-MMPs model (disputed by BSG-knockout work, Marchiq et al. 2015) nor the MCT-chaperone model is supported by co-expression. BSG sits with the glycolytic module; it is not treated as a clone-to-stroma signal.

**Positioning:** CCR7 up; CXCR4, IL7R, CD44, SELL, S1PR1, ITGA4/5/6 down (see §6).

**Identity:** KIR3DL2 up (pct_up 1.00), DPP4/CD26 down, CD3E down, PTPRC down, CD47 up, CD27 and TNFRSF4 up, IL21R up. FOXP3 is higher in reactive T cells — the MF skin clone is not Treg-like.

**Immune checkpoints** (CTLA4, LAG3, TIGIT, HAVCR2, PDCD1, ICOS, IL2RA, TNFRSF9) are elevated on reactive T cells, not on the clone.

---

## 5. Signalling axes

### LIGHT → LTβR / HVEM → RELB

The only clone-to-fibroblast chain with ligand, receptor and downstream effector all present.
- **Ligand:** TNFSF14 clone-restricted versus reactive cells in the same biopsy (q 0.0377, pct_up 1.00), versus reactive cells across donors (0.2715 vs 0.1550), and versus healthy CD4 (0.2715 vs 0.0445).
- **Receptors** up in MF fibroblasts: TNFRSF14 in F_inflammatory (0.3329 vs 0.1947), apCAF (0.4287 vs 0.2454) and F_structural (0.3459 vs 0.2099); LTBR in F_structural (0.4727 vs 0.3431). CD40 and TNFRSF1A/1B rise alongside.
- **Effector:** RELB, the non-canonical NF-κB transcription factor downstream of LTβR, up in F_mesenchymal (0.3164 vs 0.0774), F_inflammatory (0.2901 vs 0.0901) and F_structural (0.2461 vs 0.0674).

Receptor effects are modest; the RELB effects are larger. LIGHT–LTβR signalling is an established driver of skin fibrosis (Herro & Croft). Other TNF-superfamily ligands do not behave the same way: CD40LG falls in MF CD4 while CD40 rises on fibroblasts, and LTA rises in reactive and clonal cells alike.

### Th2 instruction

GATA3 is clone-restricted. IL4, IL5, IL9, IL13, IL31 and IL22 are undetectable in either CD4 population. IL4R is up in five fibroblast subsets (F_structural 0.1496 vs 0.0625; F_inflammatory 0.1471 vs 0.0607; apCAF 0.1816 vs 0.1005; F_mesenchymal 0.0796 vs 0.0270; F_reticular 0.1567 vs 0.0762), with HAS2, POSTN and TNC. Myeloid TSLP *falls* (Mac_FOLR2 0.0062 vs 0.0208), which argues against a periostin–TSLP relay. The picture is a Th2-transcription-factor-high clone and a Th2-responsive stroma, with the cytokines themselves below detection.

### gp130 / STAT3

Fibroblast IL6 (F_inflammatory 0.2560 vs 0.0910), OSMR (0.4112 vs 0.1703), PLAUR, TIMP1, MMP1 and MMP3 up; clone IL6R up (0.2985 vs 0.1994); STAT3 up in the clone and in five fibroblast subsets. The ligands are stromal, so the direction is fibroblast → clone. Myeloid IL6ST falls. This fits the JAK/STAT3 rationale in CTCL (Childs, *JEADV* 2026).

### MIF / CD74 and antigen-presenting fibroblasts

Clone MIF is up and fibroblast CD74 is up (apCAF 2.9121 vs 1.8176; F_structural 1.0776 vs 0.1843). CD74 is primarily the MHC-II invariant chain, and CTSS (which cleaves it to CLIP) and CIITA rise in the same fibroblasts. The fibroblast CD74 signal reads as an intact class-II processing programme — the MHC-II⁺ F2 fibroblasts of Li/Strobl (*Nat Immunol* 2024) — rather than as MIF reception.

### Hyaluronan

Fibroblasts increase HA synthesis (HAS2 F_structural 0.3234 vs 0.1607, F_mesenchymal 0.2622 vs 0.0917; HAS1), HA fragmentation (CEMIP in four subsets; HYAL2 in fibroblasts and across myeloid cells), and CD44 (F_reticular 1.6574 vs 1.2815; F_inflammatory 1.5894 vs 1.1604; F_structural 1.5475 vs 1.2948).

The clone is the only compartment that *loses* CD44 (Δ −0.1838, q 0.0032; clone 1.5813, reactive 1.7270, healthy 2.1328), and it gains HYAL3 (0.0248 vs 0.0102). The model in which low-molecular-weight HA binds CD44 on malignant T cells to drive growth is supported on the HA-production side and contradicted on the receptor side. Bexarotene's reported suppression of fibroblast HAS1/HAS2 fits the stromal half.

### Other non-clonal axes

ENPP2 (autotaxin) is up in myeloid cells (Mac_FOLR2 0.3063 vs 0.0961) and F_reticular, with LPAR1 up in fibroblasts — a myeloid/stromal → fibroblast lysophosphatidic-acid axis. An adenosine module (ENTPD1, NT5E with ADORA2A/2B) rises mainly in fibroblasts, with small effects. PDGFA comes from both the clone and fibroblasts; PDGFRB rises in apCAF and F_mesenchymal.

---

## 6. CXCL12–CXCR4

**On the clone:** CXCR4 is among the strongest clone-versus-reactive differences in the data — Δ −0.8131, pct_up 0.0645 (near-unanimous across biopsies), q < 1e-4. Across donors: clone 1.2292, reactive 1.7231, healthy CD4 2.4447 — a graded loss. About 6.5% of clonal cells carry any CXCR4 transcript.

**On the stroma:** CXCL12 falls in apCAF (1.9629 vs 3.7691) and F_inflammatory (1.9501 vs 2.9714), and in macrophages (Mono_Mac 0.0583 vs 0.3161; Mac_FOLR2 0.1582 vs 0.4590). **F_reticular — the canonical dermal CXCL12 source — is unchanged** (2.3336 vs 2.1232). The loss is confined to the CAF-converting subsets and macrophages; the reticular niche persists, and the clone stops reading it.

**Fibroblasts gain CXCR4** across four subsets (apCAF 0.1471 vs 0.0236; F_inflammatory 0.1391 vs 0.0202; F_reticular 0.1151 vs 0.0235; F_structural 0.0868 vs 0.0428), with small effects.

**The retention module:** CXCR4 falls together with every dermal retention and egress receptor on the clone — CD44 (−0.146), IL7R (−0.790), SELL (−0.449), S1PR1 (−0.132), ITGA4 (−0.154), ITGA5 (−0.060), ITGA6 (−0.055) — while CCR7 rises. The clone is a central-memory-like cell stripped of its anchoring and egress receptors except CCR7. This reads as release from dermal retention niches, consistent with amoeboid, protease-independent migration and with the transition from tissue-confined to disseminating disease.

**Relation to prior work.** Rindler et al. (*Mol Cancer* 2021) identified CXCR4 and IL7R loss as markers of MF lesion progression; the same directions appear here within biopsies. Aronovich et al. (*J Invest Dermatol* 2021) reported the opposite pairing: cultured fibroblasts from early-stage MF secreting high CXCL12, the MyLa line expressing high CXCR4, and AMD3100 blocking MyLa migration and chemoprotection, with a CXCR4-high T-cell subpopulation seen by CyTOF in one biopsy without clonotype assignment. The two datasets agree on FAP and MMP2 and disagree on this pair. Candidate explanations: disease stage (early-stage fibroblasts there, clone-dense lesions here, with CXCR4 loss independently described as a progression marker); serum- and culture-induced CXCL12 in fibroblasts; CXCR4-high as a property of cultured T-cell lines; and transcript versus surface protein, although the low detection rate is a floor effect and the ligand also falls.

---

## 7. Discrepancies with published CTCL work

- **RUNX2** is not differentially expressed in the clone; Danielsen et al. (*Front Oncol* 2024) assigned it to CD4⁺ tumour cells of MF progressors. Here it appears in fibroblasts and Mono_Mac, with small effects.
- **CXCL13** is not clone-enriched; Li/Strobl et al. (*Nat Immunol* 2024) report it as near-specific to malignant T cells and as the B-cell recruitment signal. Here its stromal and myeloid signals are small, so the data are uninformative rather than contradictory.
- **TIMP1**, part of the Park/Choi (*Blood Adv* 2025) 46-gene MF progression signature, is fibroblast- and clone-derived and falls in MF myeloid cells. That signature is otherwise dominated by leukocyte cytoskeletal and proliferation genes (RAC2, WIPF1, CORO1A, LPXN, BIRC5, TFRC, LDHB), with TIMP1 its only matrix gene.
- **Myeloid MMP9** is lower in MF than healthy skin, in contrast to Gaydosik/Fuschiotti (*JCI Insight* 2023).
- **CXCL12–CXCR4** runs opposite to Aronovich et al. (*J Invest Dermatol* 2021), as above.
