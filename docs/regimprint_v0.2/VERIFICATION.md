# Verification record (v0.2.0)

Everything below was run in a one-core CPU sandbox (no GPU, 4 GB RAM, background jobs killed after about
12 minutes) on **simulated** data. The simulator draws counts from a gamma-Poisson process with donor
baselines and log-normal gene residuals, not from the model's own likelihood. Nothing has been run on real
data, at full size, on a GPU, or with four vectorised chains.

## Why v0.2 exists

On the real input the v0.1 sampler used 665–1,016 of a possible 1,023 leapfrog steps per iteration (reported
by the user's run), about 14 h per fit. The cause was located by computing the posterior curvature at the
mode on a small simulated problem whose information per regulator matches the real one (150 genes, 8
pseudobulks, 6 collinear regulators). "Steps" is the trajectory length implied by the condition number after
ideal diagonal preconditioning; it is a local Gaussian approximation and does not capture funnels.

| Parameterisation | Condition number | Implied steps | Longest directions |
|---|---|---|---|
| v0.1 (scale x standard normal everywhere, horseshoe, precisions sampled) | 1,380,000 | ~1,170 | gene precisions with activities; shared residual with arm offsets |
| v0.2 coordinates, discarded span of the shared residual left with a wide prior | 4,159 | ~64 | entirely inside the shared residual |
| v0.2 final, precisions fixed at MAP | 437 | ~21 | mixed |
| v0.2 final, precisions sampled (default) | 1,595 | ~40 | gene precisions |

## Sampling on real-like simulated data (200 genes, 10 donors, 20 collinear regulators; 300 warm-up + 100 draws, 1 chain, precisions fixed at MAP)

| | Before the span fix | v0.2 final |
|---|---|---|
| Steps per iteration after warm-up | 220 | 63 |
| Iterations at the depth limit (1,023) | 0% | 0% |
| Wall time | 506 s | 293 s |
| Divergences | 0 | 0 |
| R-hat max / ESS of scale parameters (100 draws) | 1.039 / 22–45 | 1.133 / 25–47 |

No v0.1 sampling run exists on this data set (the attempts exceeded the sandbox's job limit); the v0.1 baseline
is the curvature number above and the user's measurement on real data. Activities correlated 0.84 with truth
and 90% intervals covered 74–76% on this data set, which has collinear profiles and active regulators left
out by the screen; the cause of the under-coverage here was not isolated.

## Unit tests

`pytest -q tests -m "not slow"`: 10 passed on the final code (likelihoods against scipy; Stage 0; scorer;
projection; adapter on a synthetic AnnData). `pytest -q tests -m slow`: 2 passed (M1 and M2 recovery, 135 s).
The marginal prior with the patient centre integrated out matches a dense multivariate normal to 1e-9.

## Parameter recovery (300 genes, 6 donors, 8 screened regulators; 2 chains x 250 draws, tree depth 10)

| Model | Time | R-hat max | ESS min | Corr. with truth | 90% coverage |
|---|---|---|---|---|---|
| M2, precisions sampled (default) | 535 s | 1.037 | 106 | 0.960 | 0.89 |
| M2, precisions fixed at MAP (`rho_mode=plugin`) | 274 s | 1.016 | 157 | 0.934 | 0.81 |
| M1 | 58 s | 1.016 | 201 | 0.962 | 0.90 |
| M2, v0.1 (elastic-net screen, tree depth 7) | 337 s | 1.028 | 117 | 0.973 | 0.86 |

The v0.1 row used a different screen (it kept one more true regulator), so it is not a like-for-like
comparison. Fixing the precisions at their MAP is faster but less accurate and overconfident; it is not the default.

## Screen

On the recovery data the ridge screen kept 6 of the 8 active regulators and dropped one true trunk regulator
that the elastic net kept (7 of 8). Its run time at full size has not been measured; it is one singular value
decomposition of a genes x regulators matrix.

## Model ladder, one held-out gene fold (99 genes, 10 gene blocks; 1 chain x 200 draws)

| Comparison | Difference in held-out log density [90% interval] | Expected |
|---|---|---|
| scrambled regulators vs no regulators | −0.2 [−7.4, 7.5] | ~0 |
| population-only vs no regulators | −12.8 [−35.9, 10.3] | see note |
| patient layer vs population-only | +53.5 [28.7, 81.8] | > 0 |
| subclone layer vs patient layer (total) | +81.4 [7.7, 181.0] | > 0 |
| subclone layer vs patient layer (within) | +88.3 [12.2, 190.6] | > 0 |
| full model vs no regulators | +122.0 [4.9, 253.4] | > 0 |
| M1 vs no regulators | +121.6 [10.2, 246.5] | > 0 |
| M3 vs full model (no true nonlinearity) | −2.5 [−5.1, −0.7] | ~0 |
| interaction vs full model (no true interaction; CLI demo) | +2.7 [−3.5, 9.0] | ~0 |

Fits used 31–63 steps per iteration; none reached the depth limit. Note: "population-only vs no regulators" is
not expected to be positive when patients differ. M3 is slightly below the additive model when there is
nothing nonlinear to find; the difference is small but its interval excludes zero.

## Interactions

Two planted pairs (+4, −4), 1 chain x 300 draws: both recovered with the correct sign (0.25 [0.16, 0.36];
−0.18 [−0.22, −0.15]); with the rule `p_practical > 0.9` exactly those two are flagged, 0 of 34 others. By
"interval excludes zero" alone three further pairs of size 0.02–0.03 are flagged. The preceding additive fit
had R-hat <= 1.016 on every monitored parameter. The mixture simulation (additive truth, divergent subclones)
was run under v0.1 only: neither the pooled nor the subclone-aware model flagged anything, so the mixing
artifact is not demonstrated and the pooled control is unvalidated.

## End-to-end pipeline on v0.2 (`--small` demo: 300 genes, 6 donors, 1 chain x 150 draws)

`final` (4 fits, 328 s), `b2 --fold 0` (3 fits, 160 s; full model vs no regulators +75.3 [16.9, 144.4];
interaction vs full model −8.5 [−27.2, 2.8]), the interaction and pooled-control paths of `b1`, and `report`
all completed. Checks on the final fit: E1 median per-patient Spearman 0.68, Wilcoxon p = 0.016, slope 0.42
(the simulated truth is 1, so the slope is attenuated); E2 concordance 0.80 over 41 regulator–arm pairs,
p = 0.016; M1 vs M2 population weights Spearman 0.98. Not re-run on v0.2: `tune`, and a complete `b1` fold
through the command line (its parts were run separately).

## Not verified

- Real data, full size, GPU, `chain_method="vectorized"`, four chains, the production convergence gate.
- Whether the step-count improvement holds on the real input: `scripts/time_one_fit.py` is the test.
- Cost of early warm-up at full size (before step sizes adapt).
- More than one fold of either benchmark; `tune` on v0.2; the `mask` and `none` CNV options in a fit.
- The mixing artifact and the pooled control. Simulation-based calibration is not implemented.
- The adapter on a real atlas. Figure code is not included.
