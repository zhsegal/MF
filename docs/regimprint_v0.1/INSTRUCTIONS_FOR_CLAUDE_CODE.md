# nb31 — instructions for Claude Code: regulatory-imprint models on the CTCL atlas

You are given the tested package `regimprint` (this folder). Your job is to build its input file from the
atlas, run the pipeline on the GPU machine, make the figures, and write the review document. Do not change
model, likelihood, scorer or benchmark code. If something forces a change, stop, record it under
"Deviations" in `nb31/review.md`, and rerun the fast tests. Where this document and the earlier plan
(`nb31_regulatory_imprint_models_final_plan.md`) disagree, this document and the code win; the differences
are listed in section E.

Order of this document: A housekeeping, B consolidation (inputs), C runs, D figures and printouts,
E statistics and decision rules, F what is and is not verified.

## A. Housekeeping

1. Environment: Python >= 3.10. `pip install -e ".[test,adapter]"`, then `pip install -U "jax[cuda12]"`.
   Confirm the GPU is used: `python -c "import jax; print(jax.devices())"`.
2. Verify before touching real data. All three must succeed; paste their last lines into `review.md`.
   ```bash
   pytest -q tests -m "not slow"
   pytest -q tests -m slow
   python -m regimprint.run demo --out /tmp/demo && python -m regimprint.run final --input /tmp/demo/inputs.npz --config /tmp/demo/config.json --out /tmp/demo
   ```
3. Layout: `nb31/data/` (inputs.npz and the tables it was built from), `nb31/run/` (pipeline outputs),
   `nb31/figures/`, `nb31/review.md`. Never overwrite a run directory; a rerun gets a new one.
4. Sampler settings on GPU are the defaults in `regimprint/config.py` (4 chains, 1000 warm-up, 1000 draws,
   `target_accept=0.9`, `chain_method="vectorized"`, 64-bit). Do not switch to 32-bit.
5. Convergence gate: every fit prints `div=… rhat=… ess=… pass=…` and writes it to `*_diag*.csv`.
   A fit with `pass=False` is rerun once with `--set target_accept=0.95 num_warmup=2000`. If it still
   fails, report it as failed in `review.md` and do not use its numbers. Never relax the gate.
6. Every stage writes `config_used_<stage>.json` and appends to `run/log.txt`. Keep both.

## B. Consolidation — build `nb31/data/inputs.npz`

Use `scripts/build_inputs.py` (run with `-h`). It needs the tables below; build them first.

**Cells.** Donors: the nb30 set (>= 600 CD4 cells in each arm). Reference arm: reactive conventional CD4,
with Treg clusters and cycling cells excluded. Malignant arm: nb23 subclone labels, cycling cells excluded.
Cycling cells are excluded from **both** arms: put boolean obs columns (e.g. `is_treg`, `is_cycling`) in the
AnnData and pass them to `--exclude-cols`. A subclone needs >= 150 cells (`--min-cells`); smaller ones are
dropped, not merged. Malignant cells of donors without an nb23 decomposition must have a missing subclone
label: the script then treats the donor as one clone. All time points of a geskin2026 patient are one donor.
Counts must be raw integer UMI counts (`--counts-layer`).

**Genes.** The script keeps genes present in the atlas, the DE table and the arm table, with >= 10 counts in
>= 50% of donors, and removes TR/IG segments and MT/RP genes by pattern. Pass S/G2M genes and the FFPE
probe-partition genes as text files to `--exclude-genes`. Do **not** remove immediate-early or lineage
genes (decision of the PI).

**DE table (`--de`).** Long format, one stimulation condition (Stim48hr, as in nb29): columns `regulator`,
`gene`, `log2FoldChange`, `lfcSE`, `padj`. It must contain the on-target row (`regulator == gene`) for
every regulator; regulators without it, or with a positive on-target change, are dropped by the script.
Remove regulators the resource flags as off-target before writing the table.

**Arm table (`--gene-arm`).** Two columns `gene`, `arm` (e.g. `CDKN2A 9p`), from gene coordinates and
centromere positions of the genome build used by the atlas. Acrocentric short arms can be omitted.

**Optional.** `--cnv`: inferCNV calls per subclone and arm (`donor, subclone, arm, call`), used only to
define the common scoring set when CNV options are tuned and by the `mask` option. `--ffpe`: FFPE donor ids.
`--phi`: edgeR trended dispersion per gene (design `~ donor + arm` on the pseudobulks); used by M1 only.

**Print into `review.md`:** the script's summary (G, regulators, donors by number of subclones, dropped
donors and warnings), the fraction of missing regulator–gene pairs, and the median library size per arm.
Stop if fewer than 5 donors have more than one subclone: the subclone layer would not be identified.

## C. Runs

```bash
R="python -m regimprint.run"; IN=nb31/data/inputs.npz; OUT=nb31/run
$R tune   --input $IN --out $OUT                                  # 1. tuning (about 70 fits)
for f in 0 1 2 3 4;   do $R b1 --input $IN --out $OUT --fold $f; done   # 2. gene-blocked benchmark, 8 fits per fold
for f in 0 1 2 3 4 5; do $R b2 --input $IN --out $OUT --fold $f; done   # 3. donor-blocked cold start, 3 fits per fold
$R final  --input $IN --out $OUT                                  # 4. four fits on all data + tables + checks
$R report --out $OUT                                              # 5. report.md
python scripts/sim_interaction_checks.py planted --reps 20        # 6. calibration of the interaction model
python scripts/sim_interaction_checks.py mixture --reps 10
```

Steps 2 and 3 are independent jobs per fold: submit them as an array. Step 1 must finish before 2–4
because they read `config_tuned.json`. Tuning runs one coordinate pass in the fixed order
`cnv, shrink, scale, min_de, collapse_r, k_star, frac_within`; gene fold 4 is locked and never used by it.
If time is short, tune a subset with `--tune-params cnv k_star` and say so in `review.md`.

Time the first `b1` fold and report the wall-clock per fit before launching the rest.

## D. Figures and printouts

Figure code is not part of the package: write it from the CSVs in `run/`, in the project design system
(PNG + PDF, 400 dpi, `pdf.fonttype=42`). Four figures only.

- **F1 trunk/branch map** from `regulators.csv`: x = |w|, y = `branch_share`, point size = `omega`,
  label the 15 groups with the largest |w| or omega. Mark groups listed in `ffpe_contrast.csv` with p < 0.05.
- **F2 benchmark ladder** from `b1_ladder_locked_fold.csv` (or `b1_ladder.csv` if no tuning was run) and
  `b2_ladder.csv`: `per_gene` with the 90% interval (`lo`, `hi` divided by `n_genes`) for each comparison.
- **F3 external checks** from `checks.csv` and `imprints.csv`: (a) donor-level activity against minus the
  regulator's own log fold change; (b) concordance of activity with arm dosage per donor.
- **F4 interaction matrix** from `interactions.csv`, only if the interaction model is adopted (section E).

`review.md` must contain, in this order: the verification output of A2; the input summary of B; the tuning
table (`tuning.csv`) with adopted values; the ladder tables; number of fits and number failing the gate per
stage; `checks.csv`; the top 25 rows of `regulators.csv` by |w| and by omega; the outcome of each decision
rule in E with the numbers behind it; the simulation results of step 6; deviations.

## E. Statistics and decision rules

All comparisons are differences in held-out log predictive density summed over held-out genes, with a 90%
block-bootstrap interval over gene clusters (`compare` in `benchmark.py`). "Supported" means the interval
excludes zero. After tuning, only the locked fold is an unbiased benchmark: use
`b1_ladder_locked_fold.csv` for every claim.

1. Negative control: `add_scrambled` vs `null` must not be supported. If it is, stop.
2. Regulators matter: `add` vs `null`. Patient layer: `donor` vs `shared`. Subclone layer: `add` vs
   `donor` on the `within` component. Report each as supported or not. `shared` vs `null` can be
   unsupported even when regulators matter, because a model without patient deviations is mis-specified
   when patients differ; do not read it alone.
3. Interactions are adopted only if all hold: `int` vs `add` supported on the locked fold; `int` vs `add`
   not negative in B2; step 6 `planted` passes (>= 80% of planted pairs recovered with the right sign,
   <= 10% of other pairs flagged); the `int` fit passes the gate. Flag a pair when `p_practical > 0.9`
   in `interactions.csv`. Compare with `interactions_pooled_control.csv`: pairs flagged only in the pooled
   control are candidates for subclone mixing, not interaction.
4. Ceiling: `m3` vs `add`. If not supported, state that no nonlinear structure in the shared signature is
   detectable at this sample size, whatever rule 3 says.
5. M1 vs M2: `m1_vs_m2_w_spearman` in `checks.csv` should be >= 0.8. If lower, list the genes with the
   lowest counts and the lowest between-level precision before reporting anything.
6. External checks: E1 (`median_rho` > 0 with Wilcoxon p < 0.05 across donors) and E2 (`mean_concordance`
   > 0.5). A failed E1 means the model matches profile similarity rather than regulator dosage: say so.

Differences from the earlier plan, already implemented: per-arm dosage offsets estimated by the model
replace the inferCNV gene mask (default `cnv="offsets"`); immediate-early and lineage genes are kept;
pre-processing choices are tuned on held-out data; the regulator screen is a group-sparse multi-task
elastic net with separate coefficients per donor and per subclone; interaction features include each
module's product with itself; shrinkage is implemented in Python (no R); the M3 prediction is calibrated
on an internal split so that it equals the additive model when there is no nonlinear signal; `kappa` and
`omega` are partially pooled across regulators; knockdown profiles are median-centred.

## F. What is verified, and what is not

Verified in a one-core sandbox on simulated data of reduced size (300 genes, 6 donors, 8 regulator groups):
see `VERIFICATION.md` for the numbers.

Not verified, and therefore your responsibility to watch:
- Nothing has been run at full size or on a GPU. `chain_method="vectorized"` and 4 chains are untested here.
- Simulation-based calibration (rank histograms) is not implemented.
- The pooled model did not produce spurious interactions in the one mixture simulation that was run, so
  the mixing artifact is not demonstrated; step 6 `mixture` at full size is the test.
- `scripts/build_inputs.py` is tested only on a synthetic AnnData.
- With 200–400 draws from one chain the reduced fits did not meet the production gate (ESS); they were
  judged on recovery of the simulated truth instead.
