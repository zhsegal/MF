# regimprint (v0.2)

Hierarchical regulator-imprint models for malignant subclones. Given pseudobulk counts for each patient's
reactive CD4 arm and 1–5 malignant subclones, and a Perturb-seq knockdown resource, it estimates the
activity of each regulator in each subclone, decomposed into a population, a patient and a subclone layer.

| Model | What it is | Role |
|---|---|---|
| M1 | hierarchy fitted to log fold changes (multivariate Student-t, shared-reference covariance) | fast cross-check; closed-form projection of new samples |
| M2 | hierarchy fitted to counts (beta-binomial between patient arms, Dirichlet-multinomial among subclones) | publication model; variants `null < shared < donor < add < int` |
| M3 | BART on the shared residual (bartz) | ceiling on what any nonlinear function of the knockdown profiles could add |

## Install and verify

```bash
pip install -e ".[test,adapter]"
pip install -U "jax[cuda12]"            # GPU machine only; skip on CPU
pytest -q tests -m "not slow"           # ~20 s: likelihoods vs scipy, Stage 0, scorer, projection, adapter
pytest -q tests -m slow                 # MCMC recovery on a reduced simulation (minutes on GPU, ~10 min on one CPU core)
python -m regimprint.run demo --out demo --small     # then run the stages below on demo/inputs.npz
python scripts/time_one_fit.py --input inputs.npz    # on real data: run this before any pipeline stage
```

## Input file

One `inputs.npz` (see `regimprint/data.py`; build it with `scripts/build_inputs.py`):

| Array | Shape | Meaning |
|---|---|---|
| `genes`, `gene_arm` | G | gene symbols and chromosome arm (e.g. `9p`) |
| `donors`, `y_ref` | D, D×G | raw counts of each donor's reactive conventional CD4 pseudobulk |
| `sub_name`, `sub_donor`, `y_sub` | P, P, P×G | raw counts of each malignant subclone pseudobulk; rows sorted by donor; 1–5 per donor |
| `reg_names`, `beta_lfc`, `beta_se` | K0, G×K0, G×K0 | knockdown log fold changes and standard errors, **natural log** |
| `ontarget_lfc`, `n_de`, `reg_self`, `reg_arm` | K0 | knockdown of the regulator itself (≤ 0), DE gene count, index of its own gene (−1 if absent), its arm |
| `cnv_call`, `cnv_arms` (optional) | P×A, A | inferCNV calls in {−1, 0, 1}; only used by the `mask` option and for scoring CNV options |
| `donor_ffpe` (optional), `phi` (optional) | D, G | FFPE flag; NB dispersion for M1 (default 0) |

## Pipeline

```bash
R="python -m regimprint.run"
$R tune   --input inputs.npz --out run                 # optional; writes run/config_tuned.json and tuning.csv
$R b1     --input inputs.npz --out run --fold 0        # one job per gene fold (0..4)
$R b2     --input inputs.npz --out run --fold 0        # one job per donor fold (0..5)
$R final  --input inputs.npz --out run                 # fits on all data, tables, checks, projection parameters
$R report --out run                                    # report.md + ladder tables
```

Each fit is a MAP stage (initial values) followed by NUTS. Sampler-related settings: `param` ("centered" =
v0.2 coordinates; "noncentered" = v0.1, kept for comparison only), `rho_mode` ("sampled" default; "plugin"
fixes gene precisions at their MAP: about twice as fast, lower accuracy and coverage in simulation), `screen`
("ridge" default, seconds; "enet", more sensitive in simulation, hours at full size), `scale_floor`.

Configuration: `regimprint/config.py`. Override with `--config file.json` or `--set k_star=50 cnv=mask`.
`run/config_tuned.json`, if present, takes precedence over `--config`.

## Outputs of `final`

`regulators.csv` (population weight `w`, patient spread `kappa`, subclone spread `omega`, `branch_share`),
`imprints.csv` (activity per subclone and regulator group with 90% interval), `interactions.csv` and
`interactions_pooled_control.csv`, `checks.csv` (E1 self-consistency, E2 arm dosage, M1-vs-M2 agreement,
variance shares), `ffpe_contrast.csv`, `posterior_*.npz`, `projection_population.json`.

Sign convention: activity > 0 means the subclone looks like a knockdown of that regulator (less activity).

## Module map

`data.py` input container · `prepare.py` Stage 0 (size factors, shrinkage, dose normalisation, duplicate
collapse, screen, modules, interaction features, folds) · `likelihoods.py` the three log-densities ·
`models.py` hierarchy + M1 + M2 · `fit.py` NUTS and the convergence gate · `score.py` held-out scorer ·
`m3.py` BART ceiling · `benchmark.py` B1, B2, block bootstrap · `tune.py` parameter tuning ·
`checks.py` external checks and tables · `project.py` projection · `simulate.py` simulator · `run.py` CLI.
