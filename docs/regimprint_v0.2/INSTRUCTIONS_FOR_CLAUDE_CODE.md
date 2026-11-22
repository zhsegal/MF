# nb31 — instructions for Claude Code: regulatory-imprint models on the CTCL atlas (package v0.2)

v0.2 replaces v0.1. The v0.1 sampler ran at the tree-depth limit on the real input (about 14 h per fit); the
models are now written in coordinates the data identify (see section E). Cancel every job still running on
v0.1, install v0.2, and start with step C0. `inputs.npz` does not need to be rebuilt.

You are given the tested package `regimprint` (this folder). Your job is to run it on the GPU machine, make the
figures, and write the review document. Do not change model, likelihood, scorer or benchmark code. If
something forces a change, stop, record it under "Deviations" in `nb31/review.md`, and rerun the fast tests.

Order of this document: A housekeeping, B consolidation (inputs), C runs, D figures and printouts,
E statistics and decision rules, F what is and is not verified.

## A. Housekeeping

1. Environment: Python >= 3.10. `pip install -e ".[test,adapter]"`, then `pip install -U "jax[cuda12]"`.
   Confirm the GPU is used: `python -c "import jax; print(jax.devices())"`.
2. Verify the installation; paste the last lines of each into `review.md`.
   ```bash
   pytest -q tests -m "not slow"        # 10 tests, under a minute
   pytest -q tests -m slow              # 2 MCMC recovery tests, a few minutes
   ```
3. Layout: `nb31/data/` (inputs.npz and the tables it was built from), `nb31/run_v02/` (pipeline outputs),
   `nb31/figures/`, `nb31/review.md`. Never overwrite a run directory; a rerun gets a new one. Keep the v0.1
   run directories and label them as superseded.
4. Sampler settings on GPU are the defaults in `regimprint/config.py` (4 chains, 1000 warm-up, 1000 draws,
   `target_accept=0.9`, `max_tree_depth=10`, `chain_method="vectorized"`, 64-bit). Do not switch to 32-bit.
5. Convergence gate: every fit prints `div=… rhat=… ess=… steps=… at_max_depth=… pass=…` and writes it to
   `*_diag*.csv`. A fit with `pass=False` is rerun once with `--set target_accept=0.95 num_warmup=2000`. If it
   still fails, report it as failed and do not use its numbers. Never relax the gate, and never cap the tree
   depth or shorten the chains to save time: report the timing instead.
6. Every stage writes `config_used_<stage>.json` and appends to `<out>/log.txt`. Keep both.

## B. Consolidation — `nb31/data/inputs.npz`

Already built; reuse it. For reference, the definitions it must satisfy (rebuild with
`scripts/build_inputs.py -h` only if one of them was not met):

- Donors: the nb30 set. Reference arm: reactive conventional CD4. Treg and cycling cells excluded; cycling
  cells excluded from **both** arms. Subclones from nb23, >= 150 cells each, smaller ones dropped, not merged.
  Donors without a decomposition are one clone. All time points of a geskin2026 patient are one donor.
- Raw integer counts. Genes: present in atlas, DE table and arm table, >= 10 counts in >= 50% of donors,
  without TR/IG, MT/RP, S/G2M and FFPE probe-partition genes. Immediate-early and lineage genes are kept.
- DE table: Stim48hr, with the on-target row for every regulator; off-target-flagged regulators removed.
- Optional arrays: inferCNV calls (only for the `mask` option), FFPE flags, edgeR dispersion (M1 only).

## C. Runs

```bash
R="python -m regimprint.run"; IN=nb31/data/inputs.npz; OUT=nb31/run_v02
python scripts/time_one_fit.py --input $IN                       # 0. timing gate, about 500 iterations
$R final  --input $IN --out $OUT                                 # 1. four fits on all data + tables + checks
for f in 0 1 2 3 4;   do $R b1 --input $IN --out $OUT --fold $f; done   # 2. gene-blocked benchmark, 8 fits per fold
for f in 0 1 2 3 4 5; do $R b2 --input $IN --out $OUT --fold $f; done   # 3. donor-blocked cold start, 3 fits per fold
$R report --out $OUT                                             # 4. report.md
python scripts/sim_interaction_checks.py planted --reps 20       # 5. calibration of the interaction model
python scripts/sim_interaction_checks.py mixture --reps 10
```

**Step 0 decides everything.** Paste its output into `review.md`. Continue only if the steps per iteration
after warm-up are well below 1,023 and no iteration is at the depth limit. In simulation v0.2 needed about
60 steps where v0.1 needed the maximum; nothing has been measured on the real input. If step 0 still
saturates, stop and report the output; do not work around it. From the measured seconds per iteration, state
the projected time of steps 1–3 before launching them.

Steps 2 and 3 are independent per fold: submit them as array jobs. Tuning (`$R tune`) is deferred: run it
only if the budget allows after steps 1–4, with `--out` pointing at a new directory, and record that the
pre-specified defaults were used. The regulator screen now takes seconds (`screen="ridge"`); the elastic-net
screen of v0.1 remains available as `--set screen=enet` but took hours per fold on the real input.

`--set rho_mode=plugin` fixes the gene precisions at their MAP and roughly halves the cost. In simulation it
lowered accuracy and interval coverage (section F), so use it only for a timing comparison or if the default
cannot finish, and say so.

## D. Figures and printouts

Figure code is not part of the package: write it from the CSVs in the run directory. Four figures only, built
one at a time: render, check the preview, then save. Follow the project figure style exactly:

- Output: PNG only, 600 dpi, white background, `bbox_inches="tight"`; no PDF. Multi-panel figures 7.2 in wide,
  height sized to content.
- Typography: Arial (fallback Liberation Sans, DejaVu Sans); base and axis labels 6.5 pt, titles 7 pt normal
  weight (pad 4), ticks and legend 6 pt, in-plot annotations 5.5 pt; panel letters lowercase bold 8 pt at the
  top-left just outside the axes.
- Axes: top and right spines off; axis and tick linewidth 0.5, tick length 2.5; frameless legends outside the
  data area; reference lines thin (0.5) black dotted or dashed; no gridlines, gradients or decoration.
- Colour: Okabe-Ito palette (#E69F00, #56B4E9, #009E73, #D55E00, #CC79A7, #0072B2, black); one fixed colour
  per entity across all figures; single series in #0072B2; background or non-supported points in light grey
  (#e4e4e4, #9a9a9a); diverging data on a scale centred at 0 (`TwoSlopeNorm`).
- Text: never show internal names. Map them with one dictionary, for example `w` -> "population weight",
  `kappa` -> "between-patient spread", `omega` -> "between-subclone spread", `add` -> "subclone-aware additive
  model", `null` -> "model without regulators", `donor` -> "patient-level model", `shared` -> "population-only
  model", `int` -> "interaction model", `pooled_int` -> "pooled-subclone control", `m1` -> "log-fold-change
  model", `m3` -> "nonparametric ceiling", `add_scrambled` -> "scrambled-regulator control". Titles say what is
  shown, not a conclusion. No explanatory text inside a figure beyond n and one key statistic.
- Gene labels: label only regulator groups whose 90% interval excludes zero, plus a pre-defined list supplied
  by the PI; if nothing meets the rule, label only the pre-defined list.
- Figure legends go in `review.md`, not in the image: bold "Figure name. One-line description." then (a),
  (b)… panel descriptions, then ordering, symbols, exclusions, thresholds and n.
- Tables shown as images: minimal black-and-white booktabs style, no colour, no unnecessary numeric columns.

- **F1 trunk/branch map** from `regulators.csv`: x = absolute population weight, y = share of the regulator's
  variation that lies between subclones (`branch_share`), point size = between-subclone spread. Groups whose
  population-weight interval includes zero in grey. Mark groups with p < 0.05 in `ffpe_contrast.csv` with †.
- **F2 benchmark ladder** from `b1_ladder.csv` and `b2_ladder.csv`: difference in held-out log predictive
  density per gene (`per_gene`) with its 90% interval (`lo`, `hi` divided by `n_genes`) for each comparison;
  dashed reference line at 0.
- **F3 external checks** from `checks.csv` and `imprints.csv`: (a) patient-level activity against minus the
  regulator's own log fold change; (b) concordance of activity with chromosome-arm dosage per patient.
- **F4 interaction matrix** from `interactions.csv`, only if the interaction model is adopted (section E).

`review.md` must contain, in this order: the verification output of A2; the output of step C0; the projected
and actual wall time per stage; the ladder tables; number of fits and number failing the gate per stage;
`checks.csv`; the top 25 rows of `regulators.csv` by absolute weight and by between-subclone spread; the
outcome of each decision rule in E with the numbers behind it; the simulation results of step 5; deviations.

## E. Statistics and decision rules

All comparisons are differences in held-out log predictive density summed over held-out genes, with a 90%
block-bootstrap interval over gene clusters. "Supported" means the interval excludes zero.

1. Negative control: `add_scrambled` vs `null` must not be supported. If it is, stop.
2. Regulators matter: `add` vs `null`. Patient layer: `donor` vs `shared`. Subclone layer: `add` vs `donor`
   on the `within` component. `shared` vs `null` can be unsupported even when regulators matter, because a
   model without patient deviations is mis-specified when patients differ; do not read it alone.
3. Interactions are adopted only if all hold: `int` vs `add` supported in B1; `int` vs `add` not negative in
   B2; step 5 `planted` passes (>= 80% of planted pairs recovered with the right sign, <= 10% of other pairs
   flagged); the `int` fit passes the gate. A pair is flagged when `p_practical > 0.9` in `interactions.csv`.
   Pairs flagged only in `interactions_pooled_control.csv` are candidates for subclone mixing.
4. Ceiling: `m3` vs `add`. If not supported, no nonlinear structure in the shared signature is detectable at
   this sample size, whatever rule 3 says.
5. M1 vs M2: `m1_vs_m2_w_spearman` in `checks.csv` should be >= 0.8; if lower, inspect before reporting.
6. External checks: E1 (`median_rho` > 0 with Wilcoxon p < 0.05 across patients) and E2 (`mean_concordance`
   > 0.5). A failed E1 means the model matches profile similarity rather than regulator dosage: say so.

What changed in v0.2 (all in the code; record in `review.md`):
- Same model, new coordinates: the shared gene residual is split into the part orthogonal to the arm
  indicators and knockdown profiles and the part inside their span; the patient centre is integrated out
  analytically; activities are sampled in a rotated basis. Patient-level activity is the mean of that
  patient's subclone activities.
- Model changes: Student-t(4) prior on the population weights instead of the horseshoe (the screen does the
  selection); a floor of 0.05 on the between-patient and between-subclone spreads and the arm-offset scale.
- Each fit starts with a MAP stage that supplies initial values.
- Default screen is multi-response ridge with generalised cross-validation.
- Convergence diagnostics ignore parameters pinned at a floor.

Unchanged from v0.1 relative to the original plan: arm offsets instead of an inferCNV mask; immediate-early
and lineage genes kept; interaction features include module self-products; M3 calibrated on an internal split.

## F. What is verified, and what is not

See `VERIFICATION.md` for the numbers. Everything was run on simulated data in a one-core CPU sandbox.

Not verified, and therefore to be watched:
- Nothing has been run at full size or on a GPU; four vectorised chains are untested.
- The step-count improvement is measured in simulation only; step C0 is the test on real data.
- The cost of early warm-up at full size is unknown.
- The pooled model has not produced spurious interactions in any simulation, so the mixing control is unvalidated.
- Simulation-based calibration (rank histograms) is not implemented.
- `scripts/build_inputs.py` is tested only on a synthetic AnnData.
