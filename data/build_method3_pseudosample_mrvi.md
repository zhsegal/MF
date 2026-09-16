# Build spec: Method 3 — pseudo-sample MrVI counterfactual classifier

Third notebook (`nb3_pseudosample_mrvi.ipynb`), comparable to Options 1 and 2A. Reuse the same CONFIG, anchor construction, `calls.csv` handoff, benchmarking (UMAP + per-cell track heatmap + concordance), and anchor-CV conventions from `build_mrvi_malignant_classifiers.md`. Only the method-specific parts are spelled out here.

## Idea

Pseudo-sampling turns malignancy into a legitimate MrVI sample covariate. Split each real sample into pseudo-samples by TCR status, train MrVI with the pseudo-sample as the target covariate, then classify each cell by whether its real expression matches the **malignant** or **benign** counterfactual reconstruction. This is a donor-aware, batch-corrected estimate of the same malignant-vs-benign axis as 2A — expected to mainly tame cross-study variance, not break the early-stage floor.

## CONFIG additions (on top of the shared CONFIG)

```python
NUISANCE_KEY   = "chemistry"   # MrVI nuisance b; concat with STUDY_KEY if both needed. REQUIRED.
MIN_PSEUDO     = 20            # min cells for a malignant or benign pseudo-sample to enter the CONTRAST
MARGIN_MIN     = 0.02          # abstain band on the reconstruction score (or use prob in [0.4,0.6])
```

## Step 1 — pseudo-samples (3-way, every cell in exactly one)

Per real sample `d`, assign `pseudo_sample`:
- `f"{d}_malignant"` if `tcr_label == malignant`
- `f"{d}_benign"`    if `tcr_label == benign`
- `f"{d}_unknown"`   otherwise (no-TCR + ambiguous)

Add `malignant_status` ∈ {`malignant`,`benign`,`unknown`} constant within each pseudo-sample.

## Step 2 — size gates (contrast vs training are different)

- A `_malignant`/`_benign` pseudo-sample enters the **DE contrast** only if it has ≥ `MIN_PSEUDO` cells. Below that, relabel those cells' pseudo-sample to `_unknown` (keep the cells, drop them from the axis).
- `_unknown` pseudo-samples are NEVER in the contrast but ALWAYS in training.
- Report, per donor, whether it contributes both sides (paired contrast), one side (pooled-only), or neither (Flex/no-TCR → unknown only).

## Step 3 — MrVI setup + train (all cells)

- `MRVI.setup_anndata(adata, sample_key="pseudo_sample", batch_key=NUISANCE_KEY)` — verify arg names for the installed version.
- Train on ALL cells (the `_unknown` pseudo-samples included — they shape `u` and the decoder and benefit the manifold the classifier projects through).
- The nuisance covariate is REQUIRED: `u` only corrects the method effect you name. Do not omit it.
- Persist `u = get_latent_representation(give_z=False)`.

## Step 4 — integration QC gate (verify, don't assume)

Before trusting any Flex call: check whether Flex T cells mix with 5′ T cells of the same type in `u` (e.g., per cell type, fraction of KNN neighbours from the other chemistry; or a UMAP colored by chemistry within T cells). 
- If they mix → proceed, treat uniformly.
- If Flex forms its own island → `u` did not bridge the domain shift; mark all Flex calls low-confidence regardless of score. Print the verdict explicitly.

## Step 5 — malignant axis (DE)

```python
model.differential_expression(
    sample_cov_keys=["malignant_status"],
    sample_subset=<list of _malignant and _benign pseudo-samples passing MIN_PSEUDO>,  # excludes _unknown
    store_lfc=True,
)
```
Verify the method/arg names. `_unknown` must not enter the contrast.

## Step 6 — per-cell classification (the actual call)

For every cell (anchors + unknown + Flex), build two class reconstructions and compare to real expression:
1. Class-averaged counterfactual: for each cell, average its counterfactual `z` over the **malignant** contrast pseudo-samples → `z_mal`; over the **benign** ones → `z_ben`. (Use MrVI's counterfactual/local-representation API; if no high-level call exists, drop to the module generative path `z = g_uz(u, s)` and verify.)
2. Decode both to expected expression → `mal_recon`, `ben_recon` (2 decodes/cell).
3. Real expression `R` = MrVI normalized expression if exposed, else log-norm on the MrVI HVGs.
4. Score `s = corr(R, mal_recon) − corr(R, ben_recon)` (Pearson across genes; z-score genes first). `s > 0` leans malignant.
5. Calibrate to `prob_m3` via 1-D logistic fit on anchors (anchor `s` → anchor label), same as 2A.

This is donor-aware because the class averages span all contributing donors' pseudo-samples.

## Step 7 — threshold, HC clamp, abstain

- `thr_m3` from anchor CV, `StratifiedGroupKFold` grouped by donor (report a second run grouped by study). Per fold, refit references/logistic on train anchors only; evaluate held-out; median fold threshold. Report F1/AUC/balanced-acc overall, per stage, per study.
- HC clamp: cells from healthy-control donors → `call_m3 = benign` (known negative; removes false positives the model can't learn to avoid).
- Abstain: `uncertain_m3 = (prob_m3 ∈ [0.4, 0.6])` or `|s| < MARGIN_MIN`. Flex cells flagged low-confidence if Step 4 failed.

## Step 8 — outputs

- Merge `prob_m3, call_m3, uncertain_m3` into `calls.csv` so it joins the M1/M2 comparison automatically.
- Run the shared benchmarking section: UMAP (`tcr_label` | `call_m3`, plus the M1/M2/M3 + agreement panels if all present), per-cell track heatmap (add `prob_m3` as a track), concordance summary vs TCR anchors and pairwise method agreement.

## Decision gates / edge cases

- Stop if CONFIG keys missing; `NUISANCE_KEY` is mandatory.
- A donor with no admissible `_malignant` or `_benign` pseudo-sample contributes only to training, never the axis.
- Never label no-TCR as benign; `_unknown` stays out of the contrast.
- Step 4 QC governs Flex trust — do not silently treat Flex as reliable if it islands in `u`.
- Validate on held-out **donors** (pseudo-samples held out), not just cells; keep HC clamp and abstain.
- Frame in the writeup: this is a cleaner estimate of the 2A axis, expected to stabilize cross-study variance, not to fix early-stage / no-TCR resolution.
