"""The malignancy direction δ̂: which genes build it, and what its 1-D projection can do.

nb38 fits, in a 10-d latent, the donor-averaged clonal direction

    δ̂ = mean_d [ mean(z | ALICE-clonal cells of d) − mean(z | benign anchors of d) ]

over the TCR-labelled MF/SS skin donors, and transfers it onto the TCR-less ones. On the v1 cohort
(nb37, now in ``old/``) that transfer scored leave-3-out per-donor AUROC ~0.94 against a shuffled-δ̂
null of ~0.51. This module answers the two questions that leaves open, held to *different*
standards:

**Part 1 — what genes build δ̂.** A discovery question, so it uses the all-donor models and every
labelled donor; leakage is not a concern because nothing here is a performance claim. The map is
exact rather than heuristic: the ``structural``/``semantic``/``ldvae`` arms decode as

    px_scale = softmax_genes( W z + W_batch · onehot(batch) ),      W = softplus(·) ≥ 0, [G, K]

so a step along δ̂ shifts gene ``g``'s logit by ``(W δ̂)_g``. Two properties of that decoder drive
the implementation:

* **The softmax gauge.** Adding a constant to a column of ``W`` leaves the likelihood unchanged
  (documented at ``scvi/module/_vae.py`` ``get_factor_usage``), so ``A @ δ̂`` carries a meaningless
  global offset and only gene *contrasts* are identifiable. :func:`gene_score` therefore returns
  ``A δ̂ − Σ_g p̄_g (A δ̂)_g`` — the exact first-order ``d log px_scale_g / dt`` along δ̂, with ``p̄``
  the mean expressed fraction. :func:`counterfactual_lfc` is the assumption-free cross-check
  (decode ``μ_ben`` vs ``μ_ben + δ̂`` and take the log ratio); it handles the softmax, the batch
  offset and any decoder curvature exactly, and also works for the nonlinear ``scvi`` arm.
* **Rank ``K``.** With ``n_latent=10`` the gene ranking has ten degrees of freedom, not ``G``. So
  :func:`factor_contributions` / :func:`top_genes_per_factor` report the decomposition alongside
  the gene list rather than implying more resolution than the model has.

**Part 2 — can ``s = z·δ̂/‖δ̂‖`` discriminate.** A performance claim, so it is leakage-free:
:func:`axis_scores` scores each donor on the fold latent where that donor was *held out*, with δ̂
fitted only on that fold's training donors. Because raw ``z`` carries large donor offsets (the very
reason δ̂ is per-donor-differenced), separability is read *within* donor — AUROC is rank-based, so
no centring is needed — and the cost of the offset is quantified separately in :func:`axis_metrics`.

Companion to ``semantic_clonality_helpers`` (``SC``), whose ``fit_delta`` / ``project`` /
``donor_label_table`` primitives are reused rather than reimplemented.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import semantic_clonality_helpers as SC

NB_MF = Path(__file__).resolve().parent.parent
# `_pert_utils`, `benchmark_modalities` and `factor_evidence` live in the scvi-tools-neural-nmf
# fork, not in this repo. Override with SCVI_NB_DIR if that checkout moves.
SCVI_NB = Path(os.environ.get("SCVI_NB_DIR", "/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks"))
# v2 tree. The v1 (nb37-cohort) artifacts stay on disk under `clonality_transfer/` — they are no
# longer the default, and nothing here writes into that directory.
OUT_DIR = NB_MF / "benchmark_results" / "delta_axis_v2"
CONFIG = NB_MF / "jobs" / "clonality_folds_config.json"
GTF = NB_MF / "data" / "cache" / "Homo_sapiens.GRCh38.110.chr.gtf.gz"
GENE_SETS = SCVI_NB / "gene_sets"

LABELS_KEY = "cell_type_T"
DONOR_KEY = SC.DONOR_KEY
EPS = 1e-9


# ---------------------------------------------------------------------------
# Context: the slim input, the fold definitions, the donor roles
# ---------------------------------------------------------------------------
def load_context(config=CONFIG, float32=True):
    """``(adata, cfg, folds, donor_tbl, labelled)`` for the whole notebook.

    ``adata`` is the slim job input nb38 Part A wrote (CD4 cells x 2,500 HVGs, raw counts in ``X``,
    ``is_clonal``/``is_anchor`` already in ``obs``). ``folds`` is read from the job config rather
    than re-derived with ``SC.make_triples`` so part 2 scores exactly the donors the cached fold
    latents were trained without.

    ``labelled`` is the δ-fitting donor set: every donor with at least one clonal *and* one anchor
    cell — the same rule ``SC.run_cv`` applies by default.
    """
    import scanpy as sc

    cfg = json.loads(Path(config).read_text())
    adata = sc.read_h5ad(cfg["input_h5ad"])
    if float32 and adata.X.dtype != np.float32:
        adata.X = adata.X.astype(np.float32)     # halves the footprint; scvi trains in float32

    donor_tbl = SC.donor_label_table(adata.obs)
    donors = adata.obs[DONOR_KEY].astype(str).to_numpy()
    ic = adata.obs[SC.CLONAL_COL].to_numpy(dtype=bool)
    ia = adata.obs[SC.ANCHOR_COL].to_numpy(dtype=bool)
    labelled = sorted({d for d in np.unique(donors)
                       if ic[donors == d].any() and ia[donors == d].any()})
    folds = [list(f) for f in cfg["folds"]]
    return adata, cfg, folds, donor_tbl, labelled


def label_arrays(obs):
    """``(donors, is_clonal, is_anchor)`` as plain numpy, the form every SC primitive wants."""
    return (obs[DONOR_KEY].astype(str).to_numpy(),
            obs[SC.CLONAL_COL].to_numpy(dtype=bool),
            obs[SC.ANCHOR_COL].to_numpy(dtype=bool))


def latent_path(arm, key, out_dir=OUT_DIR):
    return Path(out_dir) / f"z_{arm}_{key}.npy"


def load_latent(arm, key, out_dir=OUT_DIR, n_obs=None):
    """Cached per-(arm, fold) latent, ``(n_obs, n_latent)`` float32."""
    z = np.load(latent_path(arm, key, out_dir))
    if n_obs is not None and z.shape[0] != n_obs:
        raise ValueError(f"{latent_path(arm, key, out_dir).name}: {z.shape=} vs n_obs={n_obs}")
    return z


def load_arm(adata, arm, key="all", cfg=None, out_dir=OUT_DIR):
    """``(z, model)`` for one trained arm: cached latent + the checkpoint reloaded on its own copy.

    Each scvi model needs its own ``adata.copy()`` (UUID collision), and ``setup_anndata`` must use
    the batch key that arm was *trained* with — ``structural_donor`` used ``donor``, everything else
    ``study``. Reading it from the config keeps the two in sync.
    """
    from scvi.model._semantic_scvi import SemanticSCVI   # not re-exported from scvi.model

    cfg = cfg if cfg is not None else json.loads(CONFIG.read_text())
    mcfg = cfg["models"][arm]
    batch_key = mcfg.get("batch_key", cfg["batch_key"])
    cache_dir = Path(cfg["model_cache_dir"]) / arm / key
    if not (cache_dir / "model.pt").exists():
        raise FileNotFoundError(
            f"no checkpoint at {cache_dir}\nsubmit it first: "
            f"bash {NB_MF / 'jobs' / 'run_clonality_folds.sh'} --folds {arm}_{key}")

    own = adata.copy()
    SemanticSCVI.setup_anndata(own, layer=None, labels_key=cfg["labels_key"], batch_key=batch_key)
    model = SemanticSCVI.load(str(cache_dir), adata=own)
    z = load_latent(arm, key, out_dir, n_obs=adata.n_obs)
    print(f"{arm}/{key}: batch={batch_key} n_batch={model.summary_stats.n_batch} z={z.shape}")
    return z, model


def loadings(model, expect_positive=True):
    """``A`` = gene x factor loading DataFrame, with the non-negativity contract asserted.

    ``use_batch_norm=False`` for every arm here, so ``get_loadings()`` is exactly ``softplus(W)``
    and hence ``>= 0``; all sign structure in ``A @ δ̂`` therefore comes from δ̂ itself. If the
    assertion ever fires, a BatchNorm row-scale has been reintroduced (``gamma`` is unconstrained
    and can flip a gene's sign) and the interpretation below needs revisiting.
    """
    A = model.get_loadings()
    if expect_positive and float(np.asarray(A.values).min()) < 0:
        raise AssertionError(
            f"loadings have negative entries (min={np.asarray(A.values).min():.3g}) — "
            "expected softplus(W) >= 0; is use_batch_norm on?")
    return A


# ---------------------------------------------------------------------------
# δ: the direction, and the per-donor directions it averages
# ---------------------------------------------------------------------------
def per_donor_deltas(z, obs, train_donors, shuffle_seed=None):
    """``(deltas, delta_hat)``: per-donor ``δ_d`` as a DataFrame, and their mean δ̂.

    ``SC.fit_delta`` returns only the average and a summary table; the individual ``δ_d`` are what
    make the per-donor consistency analysis possible, so they are kept here. The mean is asserted
    against ``SC.fit_delta`` so the two paths cannot silently diverge.
    """
    donors, ic, ia = label_arrays(obs) if isinstance(obs, pd.DataFrame) else obs
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None

    rows, keys = [], []
    for d in train_donors:
        m = donors == d
        lab = m & (ic | ia)
        if lab.sum() < 2:
            continue
        pos = ic[lab]
        if rng is not None:
            pos = rng.permutation(pos)
        if pos.sum() == 0 or (~pos).sum() == 0:
            continue
        zl = z[lab]
        rows.append(zl[pos].mean(0) - zl[~pos].mean(0))
        keys.append(d)
    if not rows:
        raise ValueError("no training donor had both clonal and anchor cells")

    deltas = pd.DataFrame(np.stack(rows), index=pd.Index(keys, name=DONOR_KEY),
                          columns=[f"Z_{i}" for i in range(z.shape[1])])
    delta_hat = deltas.mean(0).to_numpy()
    ref = SC.fit_delta(z, donors, ic, ia, train_donors, shuffle_seed=shuffle_seed)[0]
    if not np.allclose(delta_hat, ref, atol=1e-6):
        raise AssertionError("per_donor_deltas disagrees with SC.fit_delta")
    return deltas, delta_hat


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / max(np.linalg.norm(v), 1e-12)


# ---------------------------------------------------------------------------
# Part 1: δ -> genes
# ---------------------------------------------------------------------------
def mean_fraction(X, mask=None):
    """``p̄_g``: mean per-cell expressed fraction, the softmax gauge's centring weights.

    Row-normalises counts then averages over cells, so ``p̄`` sums to 1 and estimates
    ``E[px_scale]`` — the point the first-order expansion of the gene softmax is taken at.
    """
    from scipy.sparse import issparse

    Xs = X if mask is None else X[np.asarray(mask, dtype=bool)]
    if issparse(Xs):
        lib = np.asarray(Xs.sum(1)).ravel()
        lib[lib == 0] = 1.0
        p = np.asarray(Xs.multiply(1.0 / lib[:, None]).mean(0)).ravel()
    else:
        Xs = np.asarray(Xs, dtype=np.float64)
        lib = Xs.sum(1)
        lib[lib == 0] = 1.0
        p = (Xs / lib[:, None]).mean(0)
    return p / max(p.sum(), 1e-12)


def detection_frac(X, mask=None):
    """Fraction of cells with a nonzero count, per gene."""
    from scipy.sparse import issparse

    Xs = X if mask is None else X[np.asarray(mask, dtype=bool)]
    if issparse(Xs):
        return np.asarray((Xs > 0).sum(0)).ravel() / max(Xs.shape[0], 1)
    return (np.asarray(Xs) > 0).mean(0)


def expressed_mask(X, mask=None, min_detect=0.05, min_frac=None, p_bar=None):
    """Which genes are expressed enough for a *relative* change to mean anything.

    :func:`gene_score` is ``d log px_scale / dt`` — a relative quantity — so a gene sitting at a
    fraction of 1e-6 can top the ranking on noise. This matters especially for
    ``decoder_mode='structural'``, where ``W`` is a function of the frozen gene embedding rather than
    of expression: a gene the cohort never expresses still gets a full loading row, and genes at the
    extremes of the embedding geometry (keratins, neuroendocrine peptides) end up with large logit
    swings that correspond to no detectable transcript. Gate on detection rate among the labelled
    cells, optionally also on ``p̄``.
    """
    det = detection_frac(X, mask)
    keep = det >= float(min_detect)
    if min_frac is not None:
        if p_bar is None:
            p_bar = mean_fraction(X, mask)
        keep &= np.asarray(p_bar) >= float(min_frac)
    return keep, det


def abundance_change(score, p_bar):
    """``d px_scale_g / dt = p̄_g · s_g`` — the change in *absolute* expressed fraction along δ̂.

    The companion to :func:`gene_score`: same derivative, not log-transformed, so it answers "which
    genes move the most transcript" rather than "which genes move the most in relative terms". It
    downweights unexpressed genes by construction instead of by a threshold, so the two views
    together bracket the answer without either one needing to be the single right choice.
    """
    s = pd.Series(score).astype(float)
    return (s * np.asarray(p_bar, dtype=float)).rename("abundance_change")


def gene_score(A, delta, p_bar, normalize=True):
    """Per-gene ``d log px_scale / dt`` along δ̂ — the malignancy gene signature.

    ``s_g = (A u)_g − Σ_h p̄_h (A u)_h`` with ``u = δ̂/‖δ̂‖`` (``normalize=False`` keeps δ̂'s own
    length, i.e. one full clonal-minus-benign step). The subtraction is the softmax gauge fix, not
    cosmetic: without it the ranking is dominated by the direction the likelihood cannot see.
    """
    A_df = A if isinstance(A, pd.DataFrame) else pd.DataFrame(A)
    u = unit(delta) if normalize else np.asarray(delta, dtype=float)
    raw = A_df.to_numpy() @ u
    p = np.asarray(p_bar, dtype=float)
    if p.shape[0] != raw.shape[0]:
        raise ValueError(f"p_bar has {p.shape[0]} genes, loadings have {raw.shape[0]}")
    return pd.Series(raw - float(p @ raw), index=A_df.index, name="delta_score")


def gauge_residual(A, delta, p_bar, seed=0):
    """Max change in :func:`gene_score` after adding a random constant to every column of ``A``.

    The softmax makes that transformation a no-op on the likelihood, so a correct score must be
    invariant to it. Expected ~1e-12; anything appreciable means the centring is wrong.
    """
    rng = np.random.default_rng(seed)
    A_df = A if isinstance(A, pd.DataFrame) else pd.DataFrame(A)
    shift = rng.normal(size=A_df.shape[1]) * float(np.abs(A_df.to_numpy()).mean())
    s0 = gene_score(A_df, delta, p_bar)
    s1 = gene_score(A_df + shift, delta, p_bar)
    return float(np.abs(s1 - s0).max())


def per_donor_gene_scores(A, deltas, p_bar):
    """One gene signature per donor: ``(n_donors, n_genes)``, rows indexed by donor.

    δ̂ is a mean over donors, so a gene can look strong because every donor moves it or because two
    donors move it a long way. This is the matrix that tells those apart (see :func:`consistency`).
    """
    A_df = A if isinstance(A, pd.DataFrame) else pd.DataFrame(A)
    rows = {d: gene_score(A_df, deltas.loc[d].to_numpy(), p_bar).to_numpy()
            for d in deltas.index}
    return pd.DataFrame(rows, index=A_df.index).T


def consistency(scores_d):
    """Per-gene mean / sd / paired-t / sign agreement across the per-donor signatures.

    ``t = mean / (sd/sqrt(n))`` over donors ranks genes that are strong *and* reproducible, which a
    single averaged δ̂ cannot distinguish from genes driven by a couple of donors. ``frac_up`` is the
    fraction of donors moving the gene in the mean direction — a nonparametric version of the same
    idea, and the one to trust when a donor is an outlier.
    """
    S = scores_d.to_numpy(dtype=float)
    n = S.shape[0]
    mean = S.mean(0)
    sd = S.std(0, ddof=1) if n > 1 else np.zeros_like(mean)
    se = sd / np.sqrt(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, mean / se, 0.0)
    sign = np.sign(mean)
    frac = ((np.sign(S) == sign[None, :]) & (sign[None, :] != 0)).mean(0)
    return pd.DataFrame({"mean": mean, "sd": sd, "t": t, "frac_same_sign": frac,
                        "abs_t": np.abs(t)}, index=scores_d.columns)


def factor_contributions(A, delta, z, module=None):
    """Which of the ``K`` factors δ̂ actually moves, in gauge-invariant units.

    ``contribution_k = u_k · std_genes(A[:, k])`` — the spread of factor ``k``'s loadings times how
    far δ̂ travels along it. Both halves of the LDVAE gauge are handled: rescaling column ``k`` by
    ``c`` while scaling ``z_k`` by ``1/c`` leaves the product fixed, and the additive per-column
    constant drops out of a standard deviation. ``amplitude_k = std_cells(z_k) · std_genes(A[:,k])``
    is the same quantity ``LDVAE.get_factor_usage`` reports for overall factor usage, included so a
    factor δ̂ leans on can be read against how much the data uses it at all.
    """
    A_df = A if isinstance(A, pd.DataFrame) else pd.DataFrame(A)
    u = unit(delta)
    w_std = A_df.to_numpy().std(0)
    z_std = np.asarray(z, dtype=float).std(0)
    contrib = u * w_std
    out = pd.DataFrame({"delta_unit": u, "w_std": w_std, "z_std": z_std,
                        "contribution": contrib,
                        "abs_frac": np.abs(contrib) / max(np.abs(contrib).sum(), 1e-12),
                        "amplitude": z_std * w_std}, index=A_df.columns)
    eff = float("nan")
    if module is not None:
        import torch

        usage = module.get_factor_usage(torch.as_tensor(np.asarray(z), dtype=torch.float32))
        out["usage"] = usage["usage"]
        eff = float(usage["eff_n_factors"])
    out = out.sort_values("abs_frac", ascending=False)
    out.attrs["eff_n_factors"] = eff          # set after sorting: attrs do not always propagate
    return out


def top_genes_per_factor(A, factors=None, n=20):
    """Top-``n`` genes of each factor column — names the programs δ̂ is built from.

    Within one column the additive gauge constant is shared by every gene, so the ranking is
    well defined even though the column's absolute level is not.
    """
    A_df = A if isinstance(A, pd.DataFrame) else pd.DataFrame(A)
    cols = list(A_df.columns) if factors is None else list(factors)
    return pd.DataFrame({c: A_df[c].nlargest(n).index.to_numpy() for c in cols},
                        index=[f"{i + 1}" for i in range(n)])


def factor_expression_support(A, det, n_top=30):
    """Is each factor's top-loading set actually expressed? The dump-factor test.

    Under ``decoder_mode='structural'`` a factor is free to point at a corner of the gene-embedding
    space the cohort never transcribes; its column then looks like a program but carries no
    expression. Comparing the mean detection rate of a factor's top-``n_top`` genes against the
    cohort-wide mean separates a real program from that failure mode — and if δ̂ leans on a factor
    that fails here, its gene ranking is embedding geometry rather than biology.
    """
    A_df = A if isinstance(A, pd.DataFrame) else pd.DataFrame(A)
    d = pd.Series(np.asarray(det, dtype=float), index=A_df.index)
    rows = {c: {"top_mean_detect": float(d.reindex(A_df[c].nlargest(n_top).index).mean()),
                "top_min_detect": float(d.reindex(A_df[c].nlargest(n_top).index).min())}
            for c in A_df.columns}
    out = pd.DataFrame(rows).T
    out["all_gene_mean_detect"] = float(d.mean())
    out["support_ratio"] = out["top_mean_detect"] / max(float(d.mean()), 1e-12)
    return out


def _batch_codes(model, obs, batch_key=None):
    """Integer batch codes for ``obs`` under a fitted model's own categorical mapping."""
    reg = model.adata_manager.get_state_registry("batch")
    cats = [str(c) for c in np.asarray(reg["categorical_mapping"])]
    key = batch_key or str(reg.get("original_key", "batch"))
    lookup = {c: i for i, c in enumerate(cats)}
    vals = obs[key].astype(str).to_numpy()
    missing = sorted(set(vals) - set(lookup))
    if missing:
        raise ValueError(f"batch value(s) {missing[:5]} absent from the model's mapping")
    return np.array([lookup[v] for v in vals], dtype=np.int64), key


def counterfactual_lfc(model, z, obs, delta, donors, pseudo=1e-6):
    """Exact per-gene log2 fold change of a δ̂ step, averaged over donors.

    For each donor: ``μ_ben`` = mean latent of its benign anchors, decode ``μ_ben`` and
    ``μ_ben + δ̂`` through the fitted ``generative()`` at that donor's own batch code and median
    library, and take ``log2(scale_mal / scale_ben)``. Unlike :func:`gene_score` this makes no
    linearity assumption — it absorbs the gene softmax, the batch offset and any decoder curvature
    — so agreement between the two validates reading the loadings directly. Two decoded rows per
    donor, so it is cheap.
    """
    import sys

    if not SCVI_NB.exists():
        raise FileNotFoundError(
            f"SemanticSCVI notebooks dir not found at {SCVI_NB} — set SCVI_NB_DIR to the "
            "scvi-tools-neural-nmf/notebooks checkout."
        )
    if str(SCVI_NB) not in sys.path:
        sys.path.insert(0, str(SCVI_NB))
    from _pert_utils import decode

    d_arr, ic, ia = label_arrays(obs)
    codes, _ = _batch_codes(model, obs)
    counts = np.asarray(obs.get("_total_counts")) if "_total_counts" in obs else None

    parts = []
    for d in donors:
        m = d_arr == d
        ben = m & ia & ~ic
        if ben.sum() < 5:
            continue
        mu = z[ben].mean(0)
        Z = np.stack([mu, mu + np.asarray(delta, dtype=float)])
        lib = np.log(float(np.median(counts[m])) if counts is not None else 1e4)
        bi = np.full(2, int(np.bincount(codes[m]).argmax()), dtype=np.int64)
        sc = decode(model, Z, np.full(2, lib), batch_index=bi, space="scale")
        parts.append(np.log2((sc[1] + pseudo) / (sc[0] + pseudo)))
    if not parts:
        raise ValueError("no donor had >= 5 benign anchor cells")
    var_names = getattr(model.adata, "var_names", None)
    idx = pd.Index(var_names) if var_names is not None else None
    return pd.Series(np.mean(np.stack(parts), axis=0), index=idx, name="counterfactual_lfc")


def pseudobulk_lfc(X, obs, donors, var_names=None, target_sum=1e4, pseudo=1.0):
    """Paired per-donor clonal-vs-anchor log2FC on the *data*, with a one-sample t across donors.

    Each donor contributes ``log2((mean CP10K | clonal) + 1) − log2((mean CP10K | anchor) + 1)``, so
    the contrast is within-donor by construction and donor composition cannot drive it. The t-test
    is over the 31 per-donor fold changes — the model-free reference the δ̂ signature is compared to.
    """
    from scipy.sparse import issparse
    from scipy.stats import ttest_1samp

    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        multipletests = None

    d_arr, ic, ia = label_arrays(obs)

    def _mean_cp(mask):
        Xs = X[mask]
        if issparse(Xs):
            lib = np.asarray(Xs.sum(1)).ravel()
            lib[lib == 0] = 1.0
            return np.asarray(Xs.multiply(target_sum / lib[:, None]).mean(0)).ravel()
        Xs = np.asarray(Xs, dtype=np.float64)
        lib = Xs.sum(1)
        lib[lib == 0] = 1.0
        return (Xs / lib[:, None] * target_sum).mean(0)

    rows, keys = [], []
    for d in donors:
        m = d_arr == d
        cl, an = m & ic, m & ia & ~ic
        if cl.sum() < 10 or an.sum() < 10:
            continue
        rows.append(np.log2(_mean_cp(cl) + pseudo) - np.log2(_mean_cp(an) + pseudo))
        keys.append(d)
    if not rows:
        raise ValueError("no donor had >= 10 cells in both classes")

    L = np.stack(rows)
    t, p = ttest_1samp(L, 0.0, axis=0)
    q = p if multipletests is None else multipletests(p, method="fdr_bh")[1]
    idx = pd.Index(var_names) if var_names is not None else pd.RangeIndex(L.shape[1])
    out = pd.DataFrame({"logfc": L.mean(0), "logfc_sd": L.std(0, ddof=1),
                        "t": t, "pvalue": p, "qvalue": q,
                        "frac_up": (L > 0).mean(0)}, index=idx)
    out.attrs["donors"] = keys
    return out, pd.DataFrame(L, index=pd.Index(keys, name=DONOR_KEY), columns=idx)


# ---------------------------------------------------------------------------
# Part 1b: is the signature CNV dosage or a transcriptional program?
# ---------------------------------------------------------------------------
def arm_annotate(var_names, gtf=GTF):
    """Map gene symbols to hg38 chromosome arms (``chr17q``-style); ``''`` when unplaceable.

    Reuses nb30's machinery exactly — ``infercnvpy.io.genomic_position_from_gtf`` for coordinates,
    then ``skin_T_cnv_helpers._arm_labels`` with the same centromere table — so an arm called here
    is the same arm the CNV notebooks call.
    """
    import anndata as ad
    import infercnvpy as cnv

    import skin_T_cnv_helpers as SK

    names = pd.Index([str(g) for g in var_names])
    tmp = ad.AnnData(np.zeros((1, len(names)), dtype=np.float32),
                     var=pd.DataFrame(index=names))
    cnv.io.genomic_position_from_gtf(str(gtf), adata=tmp, gtf_gene_id="gene_name")
    arms = pd.Series(SK._arm_labels(tmp.var), index=names, name="arm").astype(str)
    n = int((arms != "").sum())
    print(f"arm annotation: {n}/{len(names)} genes placed ({n / len(names):.1%})")
    return arms


CTCL_ARMS = {"chr8q": "gain", "chr17q": "gain", "chr7q": "gain",
             "chr10q": "loss", "chr13q": "loss", "chr17p": "loss", "chr9p": "loss"}
"""Recurrent CTCL/MF arm-level events from the literature, for orienting :func:`arm_enrichment`."""


def arm_enrichment(scores, arms, min_genes=15, n_perm=2000, seed=0):
    """Per-arm mean signature score with a gene-label permutation p-value.

    If δ̂ were tracking copy-number dosage rather than a program, its gene score would be shifted
    coherently across whole arms; a program need not be. Permuting the gene→arm assignment holds the
    score distribution and the arm sizes fixed, so the p-value asks only whether *this* arm's genes
    are unusually shifted. Two-sided, BH-corrected.
    """
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        multipletests = None

    s = pd.Series(scores).astype(float)
    a = pd.Series(arms).reindex(s.index).fillna("").astype(str)
    ok = a.to_numpy() != ""
    s_ok, a_ok = s.to_numpy()[ok], a.to_numpy()[ok]

    sizes = pd.Series(a_ok).value_counts()
    keep = [arm for arm in sizes.index if sizes[arm] >= min_genes]
    rng = np.random.default_rng(seed)
    obs_mean = {arm: float(s_ok[a_ok == arm].mean()) for arm in keep}
    grand = float(s_ok.mean())

    hits = {arm: 0 for arm in keep}
    for _ in range(int(n_perm)):
        perm = rng.permutation(s_ok)
        for arm in keep:
            if abs(float(perm[a_ok == arm].mean()) - grand) >= abs(obs_mean[arm] - grand):
                hits[arm] += 1
    p = np.array([(hits[arm] + 1) / (n_perm + 1) for arm in keep])
    q = p if multipletests is None else multipletests(p, method="fdr_bh")[1]
    return pd.DataFrame({"n_genes": [int(sizes[arm]) for arm in keep],
                         "mean_score": [obs_mean[arm] for arm in keep],
                         "delta_vs_all": [obs_mean[arm] - grand for arm in keep],
                         "pvalue": p, "qvalue": q,
                         "known_ctcl": [CTCL_ARMS.get(arm, "") for arm in keep]},
                        index=pd.Index(keep, name="arm")).sort_values("qvalue")


def enrichment(genes, universe, lib1=None, lib2=None, top_n=8):
    """BH-corrected hypergeometric enrichment of a gene list, via the repo's shared libraries."""
    import sys

    if not SCVI_NB.exists():
        raise FileNotFoundError(
            f"SemanticSCVI notebooks dir not found at {SCVI_NB} — set SCVI_NB_DIR to the "
            "scvi-tools-neural-nmf/notebooks checkout."
        )
    if str(SCVI_NB) not in sys.path:
        sys.path.insert(0, str(SCVI_NB))
    from benchmark_modalities import load_libraries
    from factor_evidence import top_enriched_sets

    lib1 = GENE_SETS / "lib1_immune.gmt" if lib1 is None else lib1
    lib2 = GENE_SETS / "lib2_cd4.gmt" if lib2 is None else lib2
    libs = load_libraries(str(lib1) if Path(lib1).exists() else None,
                          str(lib2) if Path(lib2).exists() else None)
    return pd.DataFrame(top_enriched_sets(list(genes), libs, set(universe), top_n=top_n))


# ---------------------------------------------------------------------------
# Part 2: the 1-D axis, leakage-free
# ---------------------------------------------------------------------------
def axis_scores(arm, folds, obs, labelled, out_dir=OUT_DIR, seed=0, n_null=20):
    """Leakage-free per-cell projection ``s = z·δ̂/‖δ̂‖`` for every held-out donor.

    Donor ``d`` is scored on ``z_<arm>_fold{k}.npy``, the latent of the model trained *without* fold
    ``k``'s donors, with δ̂ fitted on that fold's training donors only — so neither the encoder nor
    the direction has seen ``d``.

    Returns ``(percell, deltas, null_deltas)``. ``null_deltas[fold]`` is a *list* of ``n_null``
    independently permuted δ̂s, deliberately not collapsed: one shuffled direction is shared by every
    held-out donor of its fold, so a single draw gives correlated per-donor AUROCs and a mean that
    wanders far from 0.5 (the v1 run saw 0.385 and read it as a bug — see ``SC.run_cv``'s docstring). In
    10 dimensions a random direction has ``cos`` with δ̂ of order ``1/sqrt(10)``, which is plenty to
    look like signal. The null is a *distribution* over draws, and :func:`axis_metrics` reports it
    that way. ``percell`` carries draw 0 as ``s_null`` for plotting only.
    """
    donors, ic, ia = label_arrays(obs)
    labelled = list(labelled)
    parts, deltas, null_deltas = [], {}, {}
    for k, held in enumerate(folds):
        z = load_latent(arm, f"fold{k}", out_dir, n_obs=len(obs))
        train = [d for d in labelled if d not in set(held)]
        d_real = SC.fit_delta(z, donors, ic, ia, train)[0]
        deltas[f"fold{k}"] = d_real
        null_deltas[k] = [SC.fit_delta(z, donors, ic, ia, train,
                                       shuffle_seed=seed + 1000 + k + 100_000 * j)[0]
                          for j in range(max(1, int(n_null)))]
        for h in held:
            m = donors == h
            if not m.any():
                continue
            parts.append(pd.DataFrame(
                {"donor": h, "fold": k,
                 "s": SC.project(z[m], d_real),
                 "s_null": SC.project(z[m], null_deltas[k][0]),
                 "y": ic[m], "labelled": (ic | ia)[m]},
                index=obs.index[m]))
    if not parts:
        raise ValueError("no held-out donor produced scores")
    return pd.concat(parts), deltas, null_deltas


def logistic_direction(z, obs, train_donors, seed=0):
    """The 10-d discriminant ``SC.fit_rules``' winning rule uses, as a single weight vector.

    ``SC.fit_rules`` fits it on donor-centred ``z``; within one donor the predicted probability is a
    monotone function of ``w·z`` (the centring is a per-donor constant), so ``w`` alone is enough for
    a within-donor AUROC — no held-out donor's benign centroid is needed.
    """
    donors, ic, ia = label_arrays(obs)
    lr = SC.fit_rules(z, donors, ic, ia, train_donors, SC.fit_delta(z, donors, ic, ia,
                                                                   train_donors)[0],
                      seed=seed)["logistic_z"][0]
    return np.asarray(lr.coef_).ravel()


def axis_metrics(scores, obs, arm, folds, labelled, null_deltas=None, out_dir=OUT_DIR, seed=0):
    """``(per_donor, null_draws)``: within-donor AUROC of the 1-D axis, the 10-d bound, and the null.

    Per held-out donor, on that donor's labelled cells only: ``auc_1d`` (the δ̂ projection) and
    ``auc_10d`` (``w·z``, ``w`` from that fold's logistic fit). Within one donor the logistic
    probability is a monotone function of ``w·z`` — the donor-centring it was fitted with is a
    per-donor constant — so ``auc_10d`` needs no benign centroid and is a fair ceiling on what any
    single direction can reach. AUROC is rank-based and computed *within* donor, so the donor offset
    in ``z`` neither helps nor hurts.

    ``null_draws`` is one row per (donor, draw) from :func:`axis_scores`'s permuted δ̂s. Read its
    ``auc`` mean ± sd as the null distribution; the mean must sit at ~0.5.
    """
    S = scores[scores["labelled"].to_numpy(dtype=bool)]
    donors_all, _, _ = label_arrays(obs)
    per_fold = {}
    for k, held in enumerate(folds):
        z = load_latent(arm, f"fold{k}", out_dir, n_obs=len(obs))
        train = [d for d in labelled if d not in set(held)]
        per_fold[k] = (z, logistic_direction(z, obs, train, seed=seed))

    rows, null_rows = [], []
    for (k, d), sub in S.groupby(["fold", "donor"], observed=True):
        y = sub["y"].to_numpy(dtype=bool)
        if not (0 < y.sum() < len(y)):
            continue
        z, w = per_fold[k]
        pos = obs.index.get_indexer(sub.index)
        rows.append({"fold": k, "donor": d, "n_labelled": len(y), "true_frac": float(y.mean()),
                     "auc_1d": SC._auc_raw(y, sub["s"].to_numpy()),
                     "auc_10d": SC._auc_raw(y, z[pos] @ w)})
        for j, dn in enumerate((null_deltas or {}).get(k, [])):
            null_rows.append({"fold": k, "donor": d, "draw": j,
                              "auc": SC._auc_raw(y, SC.project(z[pos], dn))})
    return (pd.DataFrame(rows).sort_values("auc_1d", ascending=False),
            pd.DataFrame(null_rows))


def pooled_auc(scores, center="none"):
    """Pooled AUROC over every held-out labelled cell — the price of the donor offset.

    ``center='none'`` throws all donors onto one global axis; ``center='median'`` first subtracts
    each donor's own median ``s``, a *label-free* shift (no clonal/anchor information used), which
    isolates how much of the pooled loss is donor offset rather than weak separation.
    """
    S = scores[scores["labelled"].to_numpy(dtype=bool)].copy()
    s = S["s"].to_numpy(dtype=float)
    if center == "median":
        s = s - S.groupby("donor", observed=True)["s"].transform("median").to_numpy()
    elif center != "none":
        raise ValueError(f"center must be 'none' or 'median', got {center!r}")
    return SC._auc_raw(S["y"].to_numpy(dtype=bool), s), S["y"].to_numpy(dtype=bool), s


def direction_stability(deltas, delta_all=None, extra=None):
    """Cosine similarity among the fold δ̂s (and δ̂_all / any extra direction).

    A direction refitted on eight overlapping donor subsets should be near-parallel to itself; a
    cosine matrix far from 1 would mean δ̂ is a property of which donors happened to be in the fit
    rather than of malignancy. Including the logistic weight vector as ``extra`` also answers
    whether δ̂ points where the optimal discriminant does.
    """
    cols = dict(deltas)
    if delta_all is not None:
        cols["all"] = delta_all
    for name, v in (extra or {}).items():
        cols[name] = v
    keys = list(cols)
    U = np.stack([unit(cols[k]) for k in keys])
    return pd.DataFrame(U @ U.T, index=keys, columns=keys)
