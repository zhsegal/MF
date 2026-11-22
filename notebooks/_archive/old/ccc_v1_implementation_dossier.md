# CCC analysis in `38_ccc_subtype_descriptive.ipynb` / `39_ccc_subtype_headline_figures.ipynb`
### Full implementation dossier for external review

Repo: `scvi-tools-neural-nmf/notebooks/MF/`
Code under review: `38_ccc_subtype_descriptive.ipynb`, `39_ccc_subtype_headline_figures.ipynb`,
`ccc_data.py` (v1 config), `ccc_data_sub.py` (this run's config), `ccc_helpers.py` (all logic).
Tool: **LIANA+** (`li.mt.rank_aggregate`), human `consensus` resource, no ortholog mapping.

---

## 1. Biological question

Atlas: Mycosis Fungoides / CTCL skin (Li et al. 2024 + 6 other studies), **749,510 skin cells,
82 donors, 7 studies**. Question: which ligand–receptor edges connect the **malignant CD4 clone**
to its tumour microenvironment, and — new in nb38/39 — **which myeloid / fibroblast sub-state
carries each edge**, where the v1 run (nb35/36) had only pooled `Myeloid` (93,435 cells) and
`Fibroblast` (66,760).

---

## 2. Data object and its construction (`ccc_helpers.build_ccc_object`, run once via `jobs/run_ccc_build.py`)

Source: `data/atlas_joint/joint_annotated.h5ad` — 1,173,694 × 40,821 gene symbols, raw counts
(48 GB, never read whole). Metadata comes from a separate parquet
(`atlas_obs_full.parquet`, 76 cols) because the h5ad obs lacks `cell_type_final`,
`mal_tcr_alice`, `stage_group`.

Build (single streaming pass, `anndata.io.sparse_dataset`, 50k-row chunks):

1. Subset obs to `compartment == "Skin"`.
2. Build the grouping label (§3), drop unlabelled cells.
3. `keep_gene_mask` = genes appearing as any subunit of any consensus-resource interaction →
   **1,832 genes**.
4. **Library size summed over all 40,821 genes, then columns subset**, then
   `X = log1p(CP10K)` float32; raw counts kept as `layers["raw_counts"]` int32.
   (Explicitly flagged in the module docstring as *the* correctness-critical step —
   normalising after subsetting would inflate everything ~16×. nb34 has an equivalence gate.)
5. Same pass accumulates a dense pseudobulk cube keyed `(ccc_celltype × sample_id)` for a
   **deferred** donor-level differential phase (never used in nb38/39).
6. Manifest JSON with `n_obs`, `n_vars`, per-level counts, liana version.

Result: `data/ccc/ccc_skin.h5ad`, **727,156 × 1,832**, `X` = lognorm.

`load_ccc_adata` re-asserts the manifest; `assert_ccc_invariants` checks counts are integral,
`X`/counts share a sparsity pattern, `X.max() < 15`, and every control gene is in `var`.

**nb38/39 do not rebuild the object.** They load it, attach a new grouping column by `cell_id`
join, and subset in memory.

---

## 3. Cell labelling

### 3.1 The malignancy call (v1, unchanged here)

`MALIG_SRC = "mal_tcr_alice"` — from nb30: per-donor founder TRB clonotype plus its
ALICE-significant ≤1-aa CDR3 family (OLGA Pgen model). **CD4 only.** Chosen over
`mal_combined` (= ALICE ∪ inferCNV) deliberately: inferCNV derives its call from smoothed
regional *expression*, i.e. partly the same measurement LIANA then scores, so ALICE keeps the
call independent of the scored data.

`build_ccc_celltype` (helpers:316) collapses `cell_type_final` + the call:

| condition | label | n |
|---|---|---|
| CD4 & call True | `CD4_malignant` | 76,349 (29 donors ≥ MIN_CELLS) |
| CD4 & call False **and** `assessed_tcr_nb30` True | `CD4_reactive` | 72,207 (34 donors) |
| CD4 & call null, **or** False-but-never-testable | `CD4_unassessed` | 190,142 |
| everything else | `cell_type_final` verbatim | — |

The `assessed_tcr_nb30` gate demotes 59,658 CD4 that came back "False" only because no TRB CDR3
was recovered — absence of data, not evidence of benignity. CD8 and Tregs are never split
(ALICE calls none; `mal_combined`'s 21,200 CD8 calls were 100% `cnv_only`).

`malignant_evidence` ∈ {both, tcr_only, cnv_only}: 62,096 of the 76,349 malignant CD4 are
`both`; 14,253 `tcr_only`. Used in nb39 §8 as a nested-subset robustness arm.

### 3.2 The sub-level label (new in nb38/39)

`build_ccc_celltype_sub` (helpers:419): run `build_ccc_celltype`, then **replace** every cell
whose `cell_type_final ∈ {Myeloid, Fibroblast}` with `subtype_ccc` from the nb10c sidecar
`data/atlas_joint/skin_myeloid_fibro_subtypes.csv` (one row per myeloid/fibroblast skin cell,
160,195 rows; cols `cell_id, lineage, leiden_sub, subtype_fine, subtype_ccc`). Null
`subtype_ccc` (UNK / proliferating / pericyte contamination) → dropped. Then everything outside
`KEEP_LEVELS` → NaN → dropped by the caller.

`subtype_ccc` is a **coverage-driven collapse** already performed in nb10c: a fine state that
cleared MIN_CELLS in fewer than MIN_SAMPLES donors was merged into its parent
(cDC1+cDC2→cDC, Mono+moDC→Mono_moDC, Mac_ISG→Mac_infl, F_myofibroblast→F_mesenchymal).

**Roster (`KEEP_LEVELS`, 18 levels):**

- carried over cell-for-cell from v1: `CD4_malignant`, `CD4_reactive`, `CD8`, `B`, `Keratinocyte`
- myeloid (8): `LC`, `cDC`, `DC_LAMP3`, `pDC`, `Mono_moDC`, `Mac_FOLR2`, `Mac_SPP1_TREM2`, `Mac_infl`
- fibroblast (5): `F_papillary`, `F_reticular`, `F_mesenchymal`, `F_inflammatory`, `F_apCAF`

**Dropped from the roster** (scope narrowing, *not* a refutation — their nb35/36 results stand):
`CD4_unassessed`, `Tregs`, `Plasma`, `Vascular`, `Mast`, `Melanocyte`, plus `UNK`/`nan`.
Rationale given: bound the multiple-testing surface so 13 sub-levels don't arrive with a larger
grid than v1's total.

Marker definitions per level are in `ccc_data_sub.CELL_DEFINITIONS` (e.g. `F_inflammatory` =
"CCL19, APOE, CXCL2/3, **CXCL12**, IL6, MMP1…"; `F_apCAF` = "**HLA-DRA**/HLA-DQA1/CD74-high";
`Mac_SPP1_TREM2` = "SPP1/TREM2/APOC1/GPNMB/ACP5").

### 3.3 Integrity asserts in nb38/39 §0

- sidecar's observed levels `==` `MYELOID_LEVELS` / `FIBRO_LEVELS` (catches a stale config).
- every pooled Myeloid/Fibroblast cell is either sub-labelled or dropped for a stated reason
  (count reconciliation against the sidecar).
- the five untouched levels have identical cell counts pooled vs sub — this is what makes the
  B/CD8 axes a regression test.

---

## 4. Run parameters

```
RESOURCE_NAME = "consensus"     # liana human consensus, ~1,832 subunit genes
EXPR_PROP     = 0.1             # sweep [0.05, 0.1, 0.2]
MIN_CELLS     = 25              # liana default 5 judged too permissive
MIN_SAMPLES   = 5               # donors clearing MIN_CELLS before a level may be *claimed*
N_PERMS       = 1000
SEED          = 1337 ; N_JOBS = 8
SUBSAMPLE_MAX_PER_LEVEL           = 8_000    # v1 used 15_000
SUBSAMPLE_MAX_PER_DONOR_PER_LEVEL = 1_000    # v1 used 1_500
DOWNSAMPLE_SIZES = [5_000, 15_000, 40_000] ; N_SHUFFLES = 3 ; TOP_N = 20
PVAL_ALPHA = 0.05 ; HEATMAP_TOP_N = 15 ; DOTPLOT_TOP_N = 20
```

**Subsampling** (`subsample_levels`, helpers:965): per level, first cap each donor's
contribution at 1,000, then cap the level at 8,000, uniform random without replacement, seeded.
Two stated purposes: compute cost, and the fact that **liana's permutation p shrinks with group
size**, so an unbalanced `CD4_malignant` (76,349) would score p=0 on nearly everything purely on n.

**The liana call** (`run_rank_aggregate`, helpers:1105):

```python
li.mt.rank_aggregate(
    adata, groupby=<ccc_celltype_sub>, resource=<consensus df>,
    groupby_pairs=<explicit (source,target) frame>,   # the N² grid is NEVER computed
    expr_prop=0.1, min_cells=25, use_raw=False, layer="lognorm",
    n_perms=1000, aggregate_method="rra", return_all_lrs=False,
    n_jobs=8, seed=1337, key_added=...)
```

Outputs used downstream: `magnitude_rank` (RRA consensus over the magnitude scores of the
member methods), `specificity_rank`, `lr_means`, `cellphone_pvals`.

`build_groupby_pairs` expands `{axis: (senders, receivers)}` into both directions and dedups.

**Resource-name resolution** (`resolve_pairs` / `resolve_controls` / `resolve_panel`,
helpers:155–260). Necessary because the consensus resource does not spell interactions the way
people write them: complex subunits alphabetically sorted; some ligands are complexes
(CSF1→CSF1R is stored `CSF1_IL34 → CSF1R`); some receptors carry obligate extra chains
(IL7→IL7R is `IL7 → IL2RG_IL7R`); some pairs stored with the receptor in the ligand column
(`TIGIT → PVR`, `CD96 → PVR`). Matching is by **subunit-subset**; when a match requires a swap the
orientation is `"flipped"` and `resolve_controls` swaps source/target with it. 47/49 curated
pairs resolve.

---

## 5. nb38 — the descriptive notebook, section by section

**§0 · label + roster + asserts** (above). Also `load_resource(var_names=...)` reports how many
interactions have every subunit present in the data.

**§0b · coverage / axis feasibility.** `cell_count_audit` cross-tabs level × donor;
`coverage_table` reports donors present and donors ≥ MIN_CELLS; `axis_feasibility` gives, per
(sender, receiver), the number of donors where *both* sides clear MIN_CELLS, and a verdict
`claim` / `report_only` / `not_computable`. Computed **inside the CTCL window**.
`CLAIMABLE` = levels with ≥ 5 donors at ≥ 25 cells. `REPORT_ONLY` levels are still computed and
plotted ("absence goes on the record") but excluded from nb39's headline set.

**§0c · study confounding.** Per level: n_studies, top study, top-study fraction, n_donors.
`top_study_frac > 0.8` → flagged `STUDY-DOMINATED`. **The flag travels with the result; it does
not gate it.**

**§0d · balanced subsample**, then a local `run(name, source, key_added, out_name)` helper:
`prepare_analysis` (applies the analysis's disease window + builds `groupby_pairs`) →
`run_rank_aggregate` → CSV to `tables/ccc_sub_<name>.csv`.

**§1 malignant CD4 ↔ 8 myeloid states** · **§2 ↔ 5 fibroblast states** · **§3 ↔ CD8/B/Keratinocyte**
· **§4 ↔ reactive CD4** — each is one `rank_aggregate` run over an explicit bounded grid, plus
two dot plots (panel = sender, x = receiver; colour = `lr_means`, size = −log10 cellphone p).

**§5 · comparator + rank-delta.** Four more runs with **`CD4_reactive` as the sender** against
the same partner sets. `rank_delta` (helpers:1448) merges on `(target, ligand_complex,
receptor_complex)` and computes `delta_rank = magnitude_rank_malignant − magnitude_rank_reactive`
and `malignant_only = reactive rank is NaN`. Stated reading rule: *no claim "malignant CD4
signals X to Y" is reportable unless the same pair is absent or weaker with reactive CD4 as the
sender, against the same partner, in the same tissue.*

**§6 · controls.** `control_report` resolves `POSITIVE_CONTROLS` (11, re-pointed at the sub-level
each edge is *expected* on) and `NEGATIVE_CONTROLS` (3 lineage-impossible: KITLG→KIT, COL1A1→
ITGA2_ITGB1, KRT1→*), reports found / rank position / percentile / top-n membership, and flags
any `LR_CAVEATS` gene sitting in the global top 10 (the MIF→CD74 failure mode). `SPEC_MUST_HAVES`
= the 3 edges the replication spec names (CXCL13→CXCR5, CD40LG→CD40, CD86→CD28, all on the
**untouched** B level) — held to the v1 bar as a regression test.

**§7 · sensitivity (all on the myeloid axis).**
- `expr_prop_sweep` over {0.05, 0.1, 0.2}; reports how many of the top-20 are stable across all three.
- `shuffle_control`: permute the grouping label **within donor**, 3 shuffles, re-run, report
  overlap with the real top-20 (target ≤ 2/20). Within-donor holds each donor's composition
  fixed, so what dissolves is cell-type specificity, not donor identity.
- `downsample_stability` at 5k/15k/40k per level: reports median `cellphone_pvals`,
  fraction of p == 0, and top-20 overlap with the largest run — makes the n-dependence of the
  permutation p explicit.

**§8 · forced curated panel + autocrine check.** `LR_PANEL` = 8 curated groups, 49 pairs
(B-cell/TLS axis, costim TME→malignant, checkpoints, Th2/stroma remodelling, chemokine
recruitment, survival signal-3, malignant→myeloid, adhesion). Plotted **whether or not the pair
cleared expr_prop**, so a negative is reported rather than dropped. `forced_panel_expression`
computes, per (level, gene), the raw detection proportion and mean lognorm directly from the
matrix — separating "gene not detected here" from "just under the 0.1 threshold".
`panel_coverage` separates *absent from the resource* (can never be scored) from *absent from
the data*. Autocrine check: pivot the panel genes' `expr_prop` for the two CD4 levels and report
the malignant/reactive ratio — genes symmetric between them are flagged as autocrine-artifact risk.

**§9 · study diagnostic.** Per-study donor counts per level; studies with ≥3 donors on
`CD4_malignant` and on ≥1 myeloid state get their own `rank_aggregate` run (`run_by_group`);
then a top-20 overlap matrix between studies and, for each of the top-30 pooled pairs, how many
studies recover it.

**§10 · malignancy-definition sensitivity.** Re-label with `mal_combined`, `mal_cnv`, and
li2024's own `tumor_cell` label (CDR3 gate switched **off** for the alternatives on purpose),
re-run the myeloid axis, report top-20 overlap across definitions and which pairs sit in the
top-20 of all four.

**§11 · "where does each named edge land" — the point of the whole re-run.** `RESOLUTION_TESTS`:

| test | pairs | focal | candidates | expected |
|---|---|---|---|---|
| `ccr4_axis` | CCL17→CCR4, CCL22→CCR4 | receiver = malignant | 8 myeloid | cDC, DC_LAMP3 |
| `csf1_axis` | CSF1→CSF1R | sender = malignant | 8 myeloid | Mac_SPP1_TREM2, Mac_FOLR2 |
| `cxcl12_axis` | CXCL12→CXCR4 | receiver = malignant | 5 fibro | F_inflammatory |
| `mhcii_axis` | HLA-DRA→CD4 | receiver = malignant | 13 fibro+myeloid | F_apCAF |
| `tgfb_axis` | TGFB1→TGFBR1_TGFBR2 | sender = malignant | 5 fibro | F_mesenchymal |

For each: build the (source, target, ligand, receptor) tuples for every candidate level, resolve
onto resource spelling, left-join the results, take **min `magnitude_rank` per level**, and
report winner / runner-up / `gap_rank_ratio = runner_up / winner`. Verdict heuristic:

```python
spread_out = n_levels_scored >= max(3, 0.6 * n_candidates)
close      = gap_rank_ratio < 3
verdict    = "uniform" if (spread_out and close) else "concentrated"
```

Four documented outcomes: `concentrated + as_expected`, `concentrated + NOT as_expected`
(the interesting one — the v1 attribution pointed at the wrong cell), `uniform` (the pooled level
was hiding nothing), `not_scored` (coverage failure, not biology).

**§12 · regression against v1.** (a) On the five untouched levels, merge v1's
`ccc_liana_ctcl_all_{core,extended}.csv` against this run and report **Spearman ρ on
`magnitude_rank`** plus the three spec must-haves side by side. (b) For each v1 *pooled* edge
(Myeloid / Fibroblast), find its best-scoring sub-level and flag `lost_at_sublevel` where no
sub-level scored it.

**§13 · healthy-skin negative control.** Fibroblast state → keratinocyte / myeloid state in HC
skin. Hard assert: `CD4_malignant` count in HC == 0.

---

## 6. nb39 — the headline-figure notebook

Same §0 labelling block (copy-paste identical). Then:

**§0b · claim gate applied first, and only two runs.** `focal_window(CTCL)` **before**
subsampling; `CLAIMABLE` = partners with ≥ MIN_SAMPLES donors at ≥ MIN_CELLS in the CTCL window;
gate table written to `tables/ccc_sub_headline_claim_gate.csv`.

- `pairs_mal` = `CD4_malignant` ↔ every partner (both directions) → `res_mal`
- `pairs_rea` = `CD4_reactive` ↔ every partner except itself → `res_rea`

**§1–4** dot plots: malignant ↔ claimable myeloid, ↔ claimable fibroblast, ↔ CD8/B/Keratinocyte
(the visual regression test), then the whole map both directions (top 30).
**§5** the comparator panel + `rank_delta` both directions + a horizontal bar chart of
`−delta_rank` for the top 40 outgoing pairs.
**§6a** `partner_heatmap`: interactions × partner, one panel per direction, colour = `lr_means`
on viridis with **shared vmin/vmax across both panels**, `cmap.set_bad(grey=0.92)` so structurally
absent cells read as missing rather than weak, `*` where cellphone p < 0.05, rows ordered by min
`magnitude_rank` across partners. Coverage table printed underneath.
**§6b** the edge-landing heatmap: `−log10(min magnitude_rank)` per (RESOLUTION_TEST × level).
**§7** forced curated panel dot plots + the negatives table with the three distinct reasons a
curated pair can be missing.
**§8** robustness: control report, spec must-haves, within-donor shuffle control, and the
**evidence == 'both'** arm — restrict `CD4_malignant` to cells that also carry a CNV call
(a *nested* subset under the ALICE primary), re-run, report top-20 overlap and per-pair survival.

**§9 · the reportable set** — five gates ANDed:

| gate | rule |
|---|---|
| 1 | partner level ∈ `CLAIMABLE` |
| 2 | `beats_comparator` = `delta_rank < 0` **or** `malignant_only` |
| 3 | `~caveated` — neither ligand nor receptor subunit ∈ `LR_CAVEATS` |
| 4 | `survives_evidence_both` — the (source,target,L,R) key is present in `res_both` |
| 5 | `p_ok` — `cellphone_pvals < 0.05` |

Output `tables/ccc_sub_reportable_set.csv` + a per-gate drop count. Figures →
`figures/final/ccc_sub_*.svg`.

---

## 7. What the authors explicitly declare absent

- **All donor-level inference.** Every p is LIANA's cell-unit permutation p, pseudoreplicated
  across donors (76,349 malignant CD4 from 29 donors). `CAVEAT_BLOCK` states they "size the dots
  and mark the heatmap cells; they are not evidence."
- Stage / disease / layer contrasts — `ccc_data.FORBIDDEN_CONTRASTS` documents why each is a
  study contrast in disguise (SS skin = buus2025 alone; stage early/advanced = 7v7 donors inside
  li2024 only; HC skin has 1,524 CD4 across 9 donors; assessed-vs-unassessed reads out chemistry).
- The six dropped roster levels.
- Differential abundance / composition.

---

## 8. Points a reviewer should press on

Ordered roughly by how much they could change a conclusion. Items 1–4 are concrete
implementation issues; 5–12 are design questions.

**1. nb38 subsamples *before* windowing; nb39 windows *before* subsampling.**
nb38 §0d calls `subsample_levels(sub_all, …)` on the **all-disease** object, and only then does
`run()` → `prepare_analysis` → `focal_window(disease=CTCL)`. So the 8,000/level and
1,000/donor/level caps are spent partly on HC and other non-CTCL cells, and the effective CTCL n
per level is both smaller than intended and **non-uniformly** reduced across levels (structural
levels — fibroblast, keratinocyte, myeloid — have far more HC representation than CD4_malignant,
which has none). nb39 §0b does it in the correct order. The two notebooks therefore run on
different effective samples, which alone can explain any disagreement between them.

**2. `magnitude_rank` is compared across separate `rank_aggregate` runs.**
`rank_delta`, the §12 v1 regression, the §10 definition-overlap and the §11 landing table all
subtract or rank-correlate `magnitude_rank` values produced by **different** runs with different
`groupby_pairs` grids (and, in §10, different cell compositions). RRA scores are normalised
within the set of interactions tested in that call, so they are not obviously on a common scale
across runs. Within the §11 landing test the comparison *is* within one run (good); across
malignant-vs-reactive runs it is not.

**3. `malignant_only` promotes a coverage failure to evidence.**
Gate 2 in nb39 §9 passes a pair if it is `malignant_only`, i.e. simply not scored in the reactive
run. But a pair can be unscored in the reactive run because the reactive level failed
`expr_prop`/`min_cells` for it — exactly the "coverage, not biology" distinction the pipeline is
careful about everywhere else (§8, §11 `not_scored`, §12 `lost_at_sublevel`). Here it flows
straight into the reportable set.

**4. Gate 5 (`cellphone_pvals < 0.05`) is close to vacuous, and the pipeline says so itself.**
With ~8,000 cells per level and `n_perms=1000`, §7's own `downsample_stability` output exists
precisely to show that `frac_pval_zero` rises with n. A gate whose own documentation says it is
not evidence is still deciding membership of the deliverable table. Related: `specificity_rank`
— arguably the right axis for a "malignant-specific" claim — is computed and printed but never
gates anything.

**5. Circularity between the sub-level definitions and the RESOLUTION_TESTS.**
`F_inflammatory` is defined (nb10c → `CELL_DEFINITIONS`) by a marker set that **includes CXCL12**,
and the `cxcl12_axis` test asks which fibroblast state carries CXCL12→CXCR4.
`F_apCAF` is defined as HLA-DRA/CD74-high, and `mhcii_axis` asks which state carries HLA-DRA→CD4.
The config acknowledges the MHC-II caveat only as "MHC-II tracks myeloid identity"; the sharper
problem is that the receiver level was *named for the ligand being tested*. `csf1_axis` →
Mac_SPP1_TREM2 and `tgfb_axis` → F_mesenchymal are less circular but the states were still
clustered on the same expression matrix LIANA scores. What would a non-circular version look
like — hold-out genes, or defining states on a marker set with the tested L/R genes excluded?

**6. "Which state carries the edge" is confounded with per-state n and per-state depth.**
`magnitude_rank` and `expr_prop` both depend on detection rate, which depends on cells per level
and on sequencing depth. After the cap, levels still differ by an order of magnitude
(CD4_malignant near 8,000 vs pDC / F_apCAF possibly a few hundred). §11 ranks levels against each
other with no power equalisation. A per-test control — downsample every candidate level to the
smallest claimable n and re-run the landing test — is not present, and would be cheap.
Likewise the `gap_rank_ratio < 3` "uniform" threshold is unmotivated and has no null.

**7. No donor-level resampling anywhere.** The single most informative missing robustness check
is arguably a **donor jackknife / bootstrap on §11**: recompute the landing test leaving out each
donor and report how often the winning state still wins. It uses only the existing machinery and
would convert "concentrated on X" from a point estimate into something with a stability
statement, without needing the deferred pseudobulk phase.

**8. Ambient RNA and doublets are never addressed.**
No SoupX/CellBender/decontX step anywhere in the build; `doublet_score` is carried in `OBS_COLS`
but never used to filter in nb38/39. Both are canonical false-positive generators for CCC:
a T-cell/DC doublet co-expresses ligand and receptor, and ambient contamination scales with the
local abundance of the source cell type — which is exactly what differs between donors and
between the malignant/reactive comparison. The lineage-impossible `NEGATIVE_CONTROLS`
(COL1A1, KRT1, KITLG from a T cell) partially probe this, but they gate nothing and there is no
per-level ambient estimate.

**9. The malignant-vs-reactive comparator is not donor-matched.**
`CD4_malignant` clears MIN_CELLS in 29 donors, `CD4_reactive` in 34, and the subsample draws
independently per level. `rank_delta` therefore compares two levels that come from partially
different donor sets (and hence different studies, chemistries, stages). A donor-paired
comparator — restrict both levels to donors with ≥25 of *both*, which the v1 config already
identifies as the one contrast free of study confounding (`CONTRASTS["malignant_vs_reactive"]`,
n_units=26) — is available but not used in the descriptive phase.

**10. Study confounding is diagnosed but never gates or corrects anything.**
§0c flags `top_study_frac > 0.8` and §9 runs the study diagnostic on the myeloid axis only;
neither feeds the reportable set. li2024 is 56% of skin cells. There is no batch-aware CCC
variant (e.g. requiring an edge to replicate in ≥2 studies) in the gates.

**11. Multiple testing is handled only by grid-bounding.**
No FDR anywhere — not across the ~13-level grid, not across the five resolution tests, not across
the curated panel. The stated defence is that `groupby_pairs` keeps the grid small, which
controls the *number* of tests but not the error rate.

**12. Resolution is asymmetric.** Keratinocytes stay pooled (basal / spinous / granular) and CD8
stays pooled, while myeloid and fibroblast are split 8 and 5 ways. Epidermotropism — arguably the
defining MF phenotype and the axis where keratinocyte state should matter most — is therefore
still evaluated at the crudest available resolution.

**13. Gate 3 (`caveated`) silently removes the Th2 biology.**
`LR_CAVEATS` contains IL13, IL4, TNFSF8, PVR, TIGIT, MIF, B2M, CD74, HLA-DRA. Gate 3 drops any
pair touching those genes from the reportable set — including the IL13/IL4 pairs that the forced
panel exists specifically to *report* (§7's own text: "Th2 cytokine mRNA is poorly captured by 3'
10x, which is exactly why it is in the forced panel"). The forced panel and the reportable-set
gate are pulling in opposite directions on the same interactions.

**14. Spatial data is available and unused.** The atlas includes Visium (li2024). Nothing in
nb38/39 uses it — no spatial co-localisation check on the sub-level edges, no
`liana` spatial bivariate / MISTy arm. For claims of the form "malignant CD4 signals to
F_inflammatory", spatial proximity is the most direct available orthogonal validation.

**15. Robustness arms are power-confounded.** The `evidence == 'both'` arm (nb39 §8) and the
alternative-definition arm (nb38 §10) change the number of cells in `CD4_malignant`
(76,349 → 62,096 in the full object; more in the subsample) while comparing top-20 overlaps.
Loss of a pair under the restriction is therefore confounded with loss of power. Downsampling the
primary arm to the restricted arm's n would separate the two.

---

## 9. File map for the reviewer

```
notebooks/MF/
  38_ccc_subtype_descriptive.ipynb        # 13 sections, ~14 rank_aggregate runs
  39_ccc_subtype_headline_figures.ipynb   # 2 rank_aggregate runs + figures + reportable set
  ccc_data.py        (531 l)  # v1 config: paths, keys, panel, caveats, controls, contrasts
  ccc_data_sub.py    (362 l)  # `from ccc_data import *` + roster/axes/controls overrides
  ccc_helpers.py    (1614 l)  # everything: build, labels, windows, audits, runs, controls, figures
  data/ccc/ccc_skin.h5ad                          727,156 x 1,832 (X = lognorm)
  data/ccc/ccc_skin_pseudobulk_full.parquet       stale for these levels; unused here
  data/atlas_joint/skin_myeloid_fibro_subtypes.csv  nb10c sidecar, 160,195 rows
  tables/ccc_sub_*.csv        all nb38/39 outputs
  figures/final/ccc_sub_*.svg all nb39 figures
```

Upstream notebooks referenced: nb10b (cell_type_final), **nb10c** (myeloid/fibro sub-annotation),
nb30 (ALICE TCR malignancy call), nb32 (obs parquet), nb34 (build QC + equivalence gate),
**nb35/36** (the v1 pooled-level CCC run this one is compared against).
