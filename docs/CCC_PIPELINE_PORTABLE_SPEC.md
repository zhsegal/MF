# Donor-replicated cell–cell communication pipeline — portable spec

Reference implementation: `20_skin/27_ccc_subtype.ipynb` + `ccc_utils.py` + `ccc_data{,_sub}.py`
(CTCL skin atlas, ~420k cells, 7 studies, 82 donors, LIANA `consensus` resource).
This document is written so another agent can rebuild the same pipeline on a **different
dataset** without reading that code. Nothing below is CTCL-specific except the worked examples.

---

## 0 · What the pipeline is for, and what it refuses to do

**Goal.** Given a single-cell atlas with (a) a cell-type label, (b) a donor/replicate key,
(c) a study/batch key, and (d) one *focal* cell type of interest, produce a **short, defensible
list of ligand–receptor edges** between the focal type and its neighbours, where every claim is
backed by donor-level replication rather than by a cell-unit p-value.

**The two things it refuses to produce** — this is the design axis of the whole pipeline:

1. **`magnitude_rank` (any expression-magnitude score).** Every LIANA magnitude score is monotone
   in absolute expression, so ranking on it ranks abundance and sequencing depth.
2. **`cellphone_pvals` (cell-unit permutation p).** The permutation null has SD ≈ σ/√n_cells, so
   the p-value measures *how many cells the level has*, not how much evidence there is. With
   10³–10⁵ cells per level everything is p < 0.001.

Ranking is therefore on **`specificity_rank`** only (a robust-rank aggregate of NATMI
`spec_weight`, Connectome `scaled_weight`, and log2FC `lr_logfc` — all three permutation-free),
and **all evidence is donor-level** (§2–§5 below). The implementation *deletes* the two forbidden
columns from the result frame and asserts their absence, so no downstream sort or gate can reach
them. Port that assertion; it is the cheapest guarantee in the pipeline.

**Three structural rules that must survive the port**

| rule | why |
|---|---|
| **One roster, one window, one run** | `liana_pipe` derives `groupby_subset` from `groupby_pairs` and *physically subsets the object*. The null, `mat_mean`, Connectome z-scores, logFC and NATMI sums are then computed on that residual pool. **Scores from two different grids are not on a common scale and must never be subtracted or compared.** Build one `groupby_pairs` over the entire roster, run once per window. |
| **Order: label → window → subsample** | Subsampling before windowing spends the cell budget on cells no run ever sees, and spends it unevenly (structural cell types carry healthy-control cells; a disease-restricted type carries none). |
| **Common n across levels** | Every level enters the ranked run at the *same* number of cells, so a rank difference cannot be a cell-count difference. |

---

## 1 · Inputs and the config contract

### Required object
An `AnnData` with:

| what | reference impl | notes |
|---|---|---|
| lognorm matrix | `.layers["lognorm"]` (also `.X`) | what LIANA scores |
| raw counts | `.layers["raw_counts"]` | what pseudobulk sums |
| grouping label | `.obs["ccc_celltype_sub"]` | the roster; NaN = cell excluded |
| donor key | `.obs["donor"]` | **the replicate unit** — never the sample if donors have >1 sample |
| study/batch key | `.obs["study"]` | reported per level, never gates |
| condition key | `.obs["disease"]` | defines the window |
| library size | `.obs["total_counts"]` | ambient regression covariate |

**Gene-subset gate (blocking, run it first).** If the object was reduced to the resource genes to
save memory, the lognorm size factor **must have been summed over all genes before the columns
were subset**. Doing it the other way rescales every cell (≈6.7× in the reference) and produces
plausible, wrong scores. Test: keep a small full-gene test object (2–3 donors), run the identical
grid on it and on the same cells inside the subset object, and assert
`max|Δ| < 1e-6` on `specificity_rank`, `expr_prod`, `spec_weight`, `scaled_weight`, `lr_logfc`.
Fail loudly. This gate caught a real bug.

### Config module (frozen vocabulary; keep separate from the function library)
- `KEEP_LEVELS` / `DROP_LEVELS` — the roster and what is out of scope (a scope narrowing is not a
  refutation; say so).
- `RESOURCE_NAME` (`"consensus"`), `EXPR_PROP = 0.1`, `MIN_CELLS = 25` (LIANA's default of 5 is far
  too permissive per donor), `MIN_SAMPLES = 5` (donors clearing `MIN_CELLS` before a level may be
  *claimed*), `SEED`, `N_JOBS`, `TOP_N = 20`, `N_SHUFFLES = 3`.
- `POSITIVE_CONTROLS`, `NEGATIVE_CONTROLS` — 4-tuples `(source, target, ligand, receptor)`;
  `"*"` allowed as a wildcard on either side. Negatives must be **lineage-impossible**
  (a T cell does not send `KITLG` or `COL1A1`), not merely "unexpected".
- `MUST_HAVES` — a handful of edges on cell types that did **not** change since the previous
  version of the analysis. These are a regression test, not a result.
- `LR_PANEL` — curated pairs grouped by biology, reported **whether or not they pass** (§6).
- `RESOLUTION_TESTS` — named edges whose *destination* is the question, each
  `{pairs, sender|receiver (the focal type), candidates: [levels], expected: [levels], why}` (§4).
- `CONTRASTS` / `FORBIDDEN_CONTRASTS` — which comparisons are confounded by study and must not be
  run. In the reference, every stage/disease/layer contrast is a study contrast in disguise; only
  the *within-donor* malignant-vs-reactive contrast is clean. **Do this audit for the new dataset
  before writing any code.**
- `LR_CAVEATS` — genes whose resource edges track lineage identity (e.g. MHC-II edges track
  myeloid identity). **Commentary only. Never a blacklist** — in v1 it was silently deleting the
  exact biology the curated panel existed to report.

### The scoring method
```python
SPEC_CONSENSUS = AggregateClass(aggregate_meta, methods=[natmi, connectome, logfc])
SPEC_CONSENSUS(adata, groupby=..., groupby_pairs=pairs, resource=...,
               expr_prop=0.1, min_cells=25, use_raw=False, layer="lognorm",
               n_perms=1,            # inert: all three members carry permute=False
               consensus_opts=None,  # ["Specificity"] raises KeyError in liana 1.8.1
               return_all_lrs=False, aggregate_method="rra", seed=SEED)
```
`n_perms` must be an int, **not None**: with `None`, `liana_pipe` overwrites `consensus_opts` with
`'Magnitude'` and the frame comes back with `magnitude_rank` as the only aggregate. Since all three
member methods carry `permute=False`, no permutation is ever computed and the value is inert.
Immediately after the call: drop `magnitude_rank`, `cellphone_pvals`, `lr_means`; assert they are
gone; add `specificity_pct = rank(pct=True)`.

Key columns everywhere: `KEY_COLS = ["source", "target", "ligand_complex", "receptor_complex"]`.

**Complex rule, applied uniformly:** *a complex is only as good as its worst subunit.* Detection,
LFC, FDR for `TGFBR1_TGFBR2` = the min/worst over its subunits. Never the mean.

---

## 2 · §0 — Roster, window, claim gate

1. Build the grouping label; reconcile it cell-for-cell against whatever sidecar produced it, and
   print the pooled→sub cross-tab. If some levels are carried over untouched from a previous
   version, say so — they become the regression axis.
2. **Window** (`focal_window`): subset to the condition of interest, print
   cells / donors / studies. Compute everything downstream *inside* this window — claimability
   computed on the whole object counts control donors that no run ever sees.
3. **Claim gate** (`claim_gate`): per level, `n_cells`, `n_donors_present`,
   `n_donors ≥ MIN_CELLS`, `n_studies`, `top_study`, `top_study_frac`,
   `claimable = (n_donors ≥ MIN_CELLS) ≥ MIN_SAMPLES`, `study_dominated = top_study_frac > 0.8`.
   **Study dominance travels with the row and never gates** — it is a caveat printed next to the
   claim, not a filter.
4. **Negative-control window**: rerun the gate on the healthy/control window. If fewer than two
   levels are claimable there, *that failure is itself a result* — record "this resolution is a
   disease-window statement" rather than silently skipping the control.
5. **Sanity assertions** as code, not prose: e.g. `0` malignant cells in healthy skin.
6. **Roster trimming for the ranked run.** A level far smaller than the rest sets the common n for
   everybody. Drop it *from the ranked run only* (reference: pDC, 1,111 cells, would have set n for
   all 18 levels), keep it in the coverage table and in the tests that equalise power internally
   (§4), and put its absence on the record.

Outputs: `coverage.csv`, reconciliation table, a printed block in the run log.

---

## 3 · §1 — The single run

```
run_ad, report = subsample_common_n(windowed, roster)
pairs          = build_groupby_pairs(roster)        # every ordered pair, both directions
res            = run_spec_consensus(run_ad, resource, pairs)
```

`subsample_common_n` (seeded, logged):
```
n_raw     = min cells over roster levels
donor_cap = n_raw // 5        # applied FIRST, so no single donor defines a level
n         = min(level total after the donor cap, COMMON_N_CAP)   # COMMON_N_CAP = 3000
every level truncated to n, uniform without replacement
```
Report per level: `n_window`, `n_after_donor_cap`, `n_final`, `n_donors_final`,
`power_deficient`. Persist the full frame to parquet (CSV holds the top 5,000 rows only).

Also run the **control-window** grid (same function, control levels) as a negative control when it
is claimable.

---

## 4 · §2 — Donor reproducibility (the primary evidence axis)

This replaces the permutation p entirely.

Build a **(level, donor, gene) pseudobulk cube** from the *windowed, un-subsampled* object
(subsampling exists to equalise the ranking, not to throw away donor evidence). Compute it by
indicator matmul on the counts layer — never re-read the source atlas:
```
key   = level || donor            → csr indicator matrix `ind`
sums  = ind @ X                   # summed counts
det   = ind @ (X > 0)             # cells with a detected transcript
```
Store `counts` and `n_detected` per (level, donor, gene), drop zero-detection rows, and a `meta`
frame with `n_cells`, `study`, `disease`, `total_counts`, `median_lib` per (level, donor).

Densify into `[n_levels, n_donors, n_genes]` arrays (`prop` = detection fraction, or `log1p CPM`)
with a per-(complex, level) worst-subunit cache — a plane is ~18 MB at 17×73×1832 and the per-edge
loops become numpy indexing instead of 100k pandas `.loc` calls on a MultiIndex.

**Statistic, per edge:**
- *eligible donor* = both levels clear `MIN_CELLS` in that donor;
- *recovered* = every ligand subunit ≥ `expr_prop` in the sender **and** every receptor subunit
  ≥ `expr_prop` in the receiver, in that donor;
- report `donor_frac = recovered / eligible`, a **Wilson 95 % CI**, and
  `n_studies_recovered` (distinct studies among recovering donors).

**Gate 2:** `donor_frac ≥ 0.5` **and** `n_studies_recovered ≥ 2` **and** `n_donors_tested ≥ 5`.

---

## 5 · §3 — Paired within-donor contrast (is the focal side actually up?)

Answers "is this edge a property of the focal cell type, or of the tissue?" It replaces two broken
predecessors, both worth naming so they are not reinvented:
- `rank_delta` — subtracted RRA scores from **two runs on different residual pools and different
  donor sets** (see the one-grid rule);
- `focal_only` — promoted a *coverage failure* (the pair simply not scored in the comparison run)
  to evidence.

**Design.** Donors carrying ≥ `MIN_CELLS` of **both** the focal level and its paired comparator
(reference: malignant vs reactive CD4 in the same donor). Pseudobulk counts, design
`~ donor + celltype`, contrast `focal vs comparator`. Donor, study, chemistry and stage all cancel
inside the donor term — this is typically **the only contrast in a multi-study atlas free of study
confounding**. Fitted with **PyDESeq2** where there is no R/rpy2 (substitutes for edgeR/limma-voom;
log the substitution). Gene filter: detected in ≥ max(3, n_donors//2) pseudo-samples. BH-FDR.

**Per-edge roll-up:** take the **focal-side** gene — the ligand where the focal type sends, the
receptor where it receives — worst subunit for a complex (the subunit with the *largest* FDR).
`contrast_ok = FDR < 0.05 AND lfc > 0`.

**Gate 3:** `contrast_ok`.

*Caveat to log if the object is gene-subset:* size factors are estimated on the resource genes
only. Median-of-ratios is robust to this, but it is not the same fit.

---

## 6 · §4 — Where each named edge lands, power-equalised within donor

The question "which sub-state carries CCL17→CCR4?" cannot be answered by comparing scores across
levels with different n: both `magnitude_rank` and `expr_prop` depend on detection rate, which
depends on cells per level and on depth. (The predecessor verdict — a `gap_rank_ratio < 3`
"uniform vs concentrated" call — had **no null at all** and was deleted.)

**Replacement, per `RESOLUTION_TEST`:**
1. **Within each donor**, take every candidate level clearing `MIN_CELLS` plus the focal level;
   downsample all of them to the **smallest such level in that donor** (`n_eq`).
2. Run the consensus **once** over that donor's candidate grid (so the comparison is always inside
   one run, at equal n).
3. Score each candidate = best (`min`) `specificity_rank` among the test's resolved LR pairs on the
   `(focal → candidate)` or `(candidate → focal)` orientation; rank candidates within the donor.
4. Across donors: **Skillings–Mack** on the (donor × level) within-donor rank matrix — the
   incomplete-block generalisation of Friedman; reduces to Friedman when nothing is missing. *A
   level not claimable in a donor is missing, not zero*, which is exactly what Skillings–Mack is
   for. Plus a **leave-one-donor-out jackknife**: the % of LODO runs in which the winner still wins.
5. Report `median_within_donor_rank`, `n_donors`, `is_winner`, `expected`, `jackknife_win_pct`,
   `skillings_mack_{stat,df,p}`.

Levels excluded from §1 for setting the common n **are candidates here**: this test equalises power
within donor by construction, so the reason for their exclusion does not apply.

Skillings–Mack, for reimplementation: per block with `k` present treatments, rank the present
values, accumulate `A_j += sqrt(12/(k+1)) · (rank − (k+1)/2)`, build `Σ` with `Σ_jj += k−1` and
`Σ_jl −= 1` for present pairs; statistic `A' Σ⁺ A ~ χ²_{k_total−1}`.

**Circularity warning that must be reported, not buried:** if the candidate sub-states were
clustered on the same expression matrix being scored, a test asking which state carries a gene that
helped define it is circular by construction. Report those tests, do not claim them.

---

## 7 · §5 — Controls

| arm | what it does | verdict style |
|---|---|---|
| **positive / negative controls** | rank position + percentile on `specificity_rank`; for anything not found, `grid_coverage` says **why** | positives found; negatives absent or bottom decile |
| **`MUST_HAVES`** | edges on unchanged cell types | regression test vs the previous version |
| **within-donor label shuffle** | permute the label **within donor** (holds each donor's composition fixed, so what dissolves is cell-type specificity rather than donor identity), rerun, `N_SHUFFLES = 3` seeds | count overlap = **commentary**; the *identity* of every edge reaching a shuffled top-N is the gate |
| **ambient regression** | per edge, regress the donor-level score on donor **focal-cell fraction** + `log(median library size)`; BH across edges | `ambient_flag = β_focalfrac > 0 at FDR < 0.05` — **flagged, not deleted** |
| **evidence-subset arm** | rerun the whole grid with the focal level restricted to a nested, higher-confidence subset, at the **same common n** | power-matched by construction: a lost edge is a lost edge, not lost power |
| **doublet arm** | reference: **not computable** — `doublet_score` null for 100 % of the largest study (56 % of cells); a filter would silently be a study contrast | say "not computable and why", never skip silently |

Two recurring principles here:
- **A control that names the offending edges beats a control that returns a pass/fail count.** The
  shuffle and the ambient arm both return per-edge flags that §7 consumes as gates; neither is an
  assertion on a summary number that moves with seed noise.
- **`grid_coverage`** recomputes LIANA's own filter directly off the matrix for a *short* list of
  edges and returns a `reason` ∈ {`scored`, `level_below_min_cells`, `ligand_below_expr_prop`,
  `receptor_below_expr_prop`, `both_below_expr_prop`, `gene_absent_from_data`}. Without it, "not in
  the result table" is uninterpretable.

Donor score used by the ambient regression = `0.5 · (sender ligand + receiver receptor)` mean
log1p-CPM, worst subunit — the pseudobulk analogue of an LR mean. Require ≥ 8 eligible donors.

---

## 8 · §6 — Forced curated panel

The curated pairs are reported **whether or not they cleared `expr_prop`** — reporting the
negatives is the point. Keep the three reasons a curated pair can be missing apart, because they
look identical in a result table:

1. **absent from the resource** — can never be scored however well expressed;
2. **gene absent from the data**;
3. **below `expr_prop` / `min_cells` in this window** — a coverage limit, not a biological negative.

Emit `panel_coverage` (`in_resource`, `in_data` per pair), `panel_detection` (per level × gene:
detection proportion, mean lognorm, `passes_min_cells`, resolved subunits included so a pair that
fails only on an obligate extra chain — e.g. IL2RG on IL7→IL7R — is diagnosable), and the scored
panel edges. A negative then reads as *"detected in 6 % of the sender"*, not as absence.

Print `LR_CAVEATS` here as commentary. It gates nothing.

---

## 9 · §7 — The reportable set: six gates, ANDed

| gate | rule | source |
|---|---|---|
| 1 | both levels claimable (≥5 donors at ≥25 cells, in-window) | §0 |
| 2 | `donor_frac ≥ 0.5` in ≥2 studies (≥5 donors tested) | §2 |
| 3 | paired-contrast FDR < 0.05, focal side up | §3 |
| 4 | not ambient-flagged | §5 |
| 5 | not recoverable under the within-donor label shuffle | §5 |
| 6 | `specificity_rank` in the top decile of the run | §1 |

Every gate is a donor-level or coverage statement; **none is a cell-unit p-value.** Apply them in
order and emit a per-gate drop table (`n_in`, `n_out`, `n_dropped`, `n_failing_alone`) — the drop
profile is as informative as the surviving set.

**Assert, do not hope:**
```python
assert no negative control is in the reportable set
assert no shuffle-recurrent edge survived gate 5
```

Deliverable: `reportable_set.csv` + a self-describing run log.

---

## 10 · Figures — exactly five, and what is deliberately absent

House style: white background, one teal accent (`#0F766E`), no top/right spines, `svg.fonttype:
none`, `layout="constrained"`, `savefig.dpi = 200`. Figure canvases are **sized from the content**
(a helper converts the longest tick label into inches) — sizing after the fact is what made labels
collide. `viridis` throughout; `cmap.set_bad("0.92")` so a structurally absent cell reads as
*missing*, not as *weak*.

**Fig 1 — dotplot, top-N edges per direction.**
x = partner level, y = `ligand → receptor`, **colour = −log10 `specificity_rank`**, **size =
fraction of donors recovering the edge** (`s = 30 + 200·donor_frac`). Two panels stacked
vertically: *focal → partner* and *partner → focal*, `sharex`, **one shared partner axis and one
shared colour scale**, so a column means the same level in both. Rows ordered by best rank. Size
legend drawn with proxy markers at 25/50/100 %. This single encoding — rank in colour, replication
in size — is the whole thesis of the pipeline in one panel.

**Fig 1b — the same dots, blocked by biological program.**
Identical encoding, but rows are the **curated** edges grouped into named programs
(`tls_b_cell_recruitment`, `myeloid_dc_reprogramming`, `dc_skin_homing`,
`fibroblast_stromal_retention`, `canonical_state_regulation`, …), blocked in program order with
alternating grey bands, a teal bracket and the program name written outside the axes.
**No top-N truncation**: a curated edge that never cleared `expr_prop` is a *reported negative*,
returned on the figure's `missing_` attribute and printed, not silently dropped.

**Fig 2 — heatmap, interactions × partner.**
Value = −log10 `specificity_rank`, shared vmin/vmax across both direction panels, **grey =
structurally absent**, `*` where the paired-contrast FDR < 0.05. Annotation colour flips
black/white on cell luminance. Top-15 rows by best rank.

**Fig 3 — landing heatmap.**
`RESOLUTION_TEST` × level, cell = **median within-donor rank** (`viridis_r`, 1 = best), the winner
annotated with its **donor-jackknife win %**. One glance answers "which state carries this edge,
and would it survive dropping a donor".

**Fig 4 — forced curated panel dotplot.**
x = level, y = panel gene, **size = detection proportion**, colour = mean lognorm. Plotted whether
or not the pair cleared `expr_prop`; ~0.30 in per gene so every symbol stays legible.

**Deliberately absent, and worth keeping absent:** no network / chord / circos plot (they encode
nothing the dotplot does not, and they invite reading edge thickness as evidence), no UMAP, no
violin or ridge plot, no per-section exploratory panel.

---

## 11 · Reporting discipline

Every section prints **one fixed-width block** and appends it to `run_log.md`: what ran, n per
level, n donors, n tests, n passing, and the top rows as text. The rule is that the block must be
**readable pasted into a chat by someone with no access to the notebook**. No emoji, no ASCII art,
no progress bars. Truncate the log at the start of every run so a clean-kernel run does not append
to the previous one.

Every section also writes its own CSV. The heavy scored frame goes to parquet; CSVs carry the top
rows.

Close with an explicit **"deliberately absent"** section: confounded contrasts not run, cell types
dropped upstream, arms the dataset cannot support, orthogonal data (e.g. spatial) that exists and
was not used. And a **"read with"** section naming the structural caveats — e.g. splitting a level
into *k* sub-levels multiplies the tested grid by *k* while each sub-level carries fewer cells and
fewer donors, so a pair that was solid on the pooled level can look weaker on every sub-level with
no biology having changed.

---

## 12 · Porting checklist

1. Identify **focal type**, **roster**, **replicate key** (donor, not sample), **batch key**,
   **window**.
2. Audit which contrasts are confounded with batch; write the `FORBIDDEN_CONTRASTS` list **before**
   writing analysis code. Find the one contrast that is within-donor — it is the §3 axis.
3. Run the gene-subset equivalence gate if the object is not full-gene. Blocking.
4. Claim gate in-window; decide the ranked roster; log anything dropped for setting the common n.
5. One grid, one run, common n, `specificity_rank` only; assert the forbidden columns are gone.
6. Pseudobulk cube → donor reproducibility (Wilson CI + study count).
7. Paired within-donor DE → per-edge focal-side roll-up (worst subunit).
8. Within-donor power-equalised landing test → Skillings–Mack + LODO jackknife.
9. Controls: positives/negatives with coverage reasons, must-haves, within-donor shuffle (per-edge
   flags), ambient regression, a power-matched subset arm; state explicitly what the dataset cannot
   support.
10. Six gates ANDed, per-gate drop table, assertions on control leakage.
11. Five figures, one encoding, nothing exploratory.

**Parameters that carried the reference analysis:** `expr_prop = 0.1`, `min_cells = 25`,
`min_samples = 5`, `common_n_cap = 3000`, per-donor cap = `n_raw // 5`, `donor_frac ≥ 0.5`,
`≥ 2 studies`, `≥ 5 donors tested`, `≥ 8 donors` for the ambient regression, `FDR < 0.05`,
top decile on `specificity_rank`, `N_SHUFFLES = 3`, `TOP_N = 20`, one seed for everything.

**Failure modes seen in practice, in order of cost:** wrong lognorm size factor after gene
subsetting (silent, ~6.7× rescale) · subtracting scores across two runs on different grids ·
subsampling before windowing · ranking on magnitude · treating a coverage failure as a biological
negative · a blacklist that deletes the exact biology the analysis exists to report · a
pass/fail control that moves with seed noise · claiming a resolution test that is circular by
construction.
