# Verification record (v0.1.0)

Everything below was run in a one-core CPU sandbox (no GPU, 4 GB RAM) on **simulated** data. The simulator
draws counts from a gamma-Poisson process with donor baselines and log-normal gene residuals, not from the
model's own likelihood. Reduced size unless stated: 300 genes, 6 donors (subclones per donor 1, 1, 2, 2, 3,
4), 16 regulators of which 8 active, 4 chromosome arms. Chains were short (150–400 draws, 1–2 chains, tree
depth 6–7), so most fits miss the production gate on effective sample size; they were judged on recovery of
the simulated truth.

## Unit tests (`pytest -q tests -m "not slow"`: 10 passed)

| Test | What it establishes |
|---|---|
| between term vs `scipy.stats.betabinom` | mixture over subclones and size factors are correct (1e-6) |
| within term vs `scipy.stats.dirichlet_multinomial` | correct, and exactly 0 for single-clone donors |
| tree likelihood vs flat multinomial at large precision | the two-level factorisation is exact (1e-3) |
| M1 vs dense `scipy.stats.multivariate_t` | Sherman–Morrison covariance is correct (1e-6) |
| Stage 0 | planted duplicate regulators collapsed; self-targets exactly zero; size factors centre the median ratio; folds balanced; screen keeps >= 6 of 8 active regulators; interaction features orthogonal to main effects; folds reusable across configurations |
| shrinkage | error of knockdown effects reduced by more than 20% (0.137 -> 0.077 in one column) |
| scorer | true parameters score above null in total and within components; pooled data path works; bootstrap interval brackets the estimate |
| projection | closed-form activities correlate > 0.9 with truth |
| adapter | `scripts/build_inputs.py` reproduces pseudobulks, effects, on-target values and self-indices from a synthetic AnnData |

`pytest -q tests -m slow`: 2 passed (M1 and M2 recovery, 87 s).

## Parameter recovery (2 chains x 250 draws, tree depth 7)

| Model | Corr. of activities with truth | 90% interval coverage | Divergences | R-hat max | Time |
|---|---|---|---|---|---|
| M2 `add` | 0.973 | 0.86 | 0 | 1.028 | 337 s |
| M1 `add` | 0.973 | 0.87 | 0 | 1.024 | 89 s |

Shared residual a_g: correlation 0.88 with truth, sigma_a 0.32 (truth 0.30). Arm offsets: correlation 0.90.
Subclone spread omega: the three planted branch regulators are estimated at 0.22–0.43 (truth 0.40), the
others at 0.03–0.05. With six donors the population weights are shrunk (truth 0.59 and 0.78 -> 0.34).

## Model ladder, one held-out gene fold (99 genes, 10 gene blocks)

Differences in held-out log predictive density with 90% block-bootstrap intervals. Two independent runs.

| Comparison | Run 1 (200 draws) | Run 2 (CLI demo, 150 draws) | Expected |
|---|---|---|---|
| scrambled regulators vs null | +0.4 [−3.1, 3.5] | −1.5 [−4.0, 1.1] | ~0 |
| shared vs null | −10.7 [−29.7, 8.3] | −7.9 [−28.2, 12.3] | see note |
| donor vs shared | +29.6 [12.1, 48.9] | +23.9 [10.1, 38.7] | > 0 |
| add vs donor (total) | +64.5 [5.6, 142.6] | +70.2 [12.2, 149.3] | > 0 |
| add vs donor (within) | +69.6 [11.7, 147.5] | +69.8 [11.8, 148.0] | > 0 |
| add vs null | not computed | +86.2 [−0.1, 182.3] | > 0 |
| M1 vs null | +77.3 [4.2, 168.3] | +75.9 [−1.0, 165.0] | > 0 |
| int vs add (no true interaction) | — | +4.7 [−0.5, 10.4] | ~0 |
| M3 vs add (no true nonlinearity) | −17.9 [−28.0, −8.3] before calibration | −0.8 [−1.6, 0.0] after | ~0 |

Notes. (1) "shared vs null" is not expected to be positive when patients differ: a model with population
weights but no patient deviations is mis-specified and is scored with the precision of the full model.
(2) With only 10 gene blocks the intervals are wide; "add vs null" touches zero in run 2. (3) M3 was
overconfident before it was calibrated on an internal split; after calibration BART found out-of-sample
R² = 0 and M3 equals the additive model.

## Interactions

| Simulation | Result |
|---|---|
| Two planted pairs (+4, −4), 1 chain x 400 draws, R-hat <= 1.075 | both recovered with the correct sign (0.26 [0.16, 0.37]; −0.18 [−0.22, −0.15]); with the rule `p_practical > 0.9`: 2 of 2 planted, 0 of 34 others. By "interval excludes zero" alone, 3 further pairs of size 0.02–0.03 were flagged. Activities: correlation 0.978 with truth |
| Second data set (seed 100) through `scripts/sim_interaction_checks.py planted --reps 1 --cpu-small` | 2 of 2 planted recovered with the correct sign, 0 of 34 others flagged |
| First data set again, 1 chain x 200 draws | same five pairs; R-hat 1.81 — chains this short are not usable for the interaction model |
| Additive truth, strongly divergent subclones (omega 0.9), 6 multi-clone donors | pooled model: 0 of 36 flagged; subclone-aware model: 0 of 36 flagged. **The mixing artifact was not reproduced**, so the pooled control is unvalidated |

## Projection and external checks (M1 fit)

Projection of each subclone with population parameters fixed: correlation 0.995 with the full fit, 0.964
with truth. E1 self-consistency: median per-donor Spearman 0.76–0.80, Wilcoxon p = 0.016; the slope is 0.41
where the simulated truth is 1, so the slope is attenuated and must not be read as a calibrated dose. E2 arm
dosage: concordance 0.79–0.82 over 44 regulator–arm pairs, p = 0.016–0.031. M1 vs M2 population weights:
Spearman 0.90.

## End-to-end pipeline (`python -m regimprint.run`, `--small` demo)

`demo`, `b1 --fold 0` (8 MCMC fits + BART, 325 s), `b2 --fold 0` (3 fits, 93 s), `final` (4 fits, 155 s),
`report`, and `tune --tune-params shrink` (4 fits, 121 s) all completed and wrote their outputs. Tuning
result: removing shrinkage lowered held-out log density by 26.2 [−53.6, −3.5], so shrinkage was kept.
One bug was found and fixed here: pandas read the model named `null` as a missing value in the report.

## Not verified

- Full-size problems, GPU execution, `chain_method="vectorized"`, four chains, and the production gate.
- More than one fold of either benchmark; the full tuning grid; the `mask` and `none` CNV options in a fit.
- The mixing artifact and the pooled control (see above). Two replicates of the planted-interaction simulation and one of the mixture simulation.
- Simulation-based calibration (rank histograms) is not implemented.
- The adapter on a real atlas; real Perturb-seq tables; any real data.
- Figure code is not included.
