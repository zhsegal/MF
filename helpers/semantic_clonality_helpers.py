"""Latent δ-arithmetic transfer of TCR clonality to TCR-less MF skin samples (nb37).

The perturbation idea from ``notebooks/03_perturbation_simple.ipynb`` — a held-out unit's latent
state is recoverable as ``z_ctrl + δ`` with ``δ = mean(z|stim) − mean(z|ctrl)`` fitted on
training units — applied to malignancy instead of perturbation:

    δ_d = mean(z | ALICE-clonal cells of donor d) − mean(z | benign anchors of donor d)
    δ̂   = mean_d δ_d                       (per-donor differencing cancels donor offsets)

For a donor with no TCR we do not know its benign centroid, so it is *estimated* from the
donor's own cells (2-component GMM on the δ̂ projection, the same per-donor trick nb30 uses on
``cnv_score``), and the clonal centroid is then **reconstructed** as ``μ̂_mal = μ̂_ben + δ̂``.
Cells are assigned to the nearer centroid.

Validation is repeated **leave-3-samples-out** over the labelled donors; the encoder is
retrained per fold by ``jobs/run_clonality_folds.py`` (held-out donors' cells never enter
training), and this module consumes the resulting per-fold latents.

Companion to ``semantic_malig_helpers`` (whose threshold/scoring primitives are reused).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import semantic_malig_helpers as M

def r2_latent(z_pred_mean, z_real_mean):
    """R² of a predicted latent centroid = pearson² over the latent dims.

    Same definition as ``_pert_utils.r2_latent``, inlined so the eval loop does not have to
    import scvi. Here: does ``μ̂_mal = μ̂_ben + δ̂`` land on the real clonal centroid?
    """
    from scipy.stats import pearsonr

    a = np.asarray(z_pred_mean, dtype=np.float64).ravel()
    b = np.asarray(z_real_mean, dtype=np.float64).ravel()
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(pearsonr(a, b)[0] ** 2)


CLONAL_COL = "is_clonal"    # ALICE malignant CD4 cell
ANCHOR_COL = "is_anchor"    # benign anchor: has_tcr & ~ALICE & ~tcr_is_expanded
DONOR_KEY = "donor"

RULES = ("delta_centroid", "delta_threshold", "logistic_z", "delta_centroid_null")


# ---------------------------------------------------------------------------
# Donor table + leave-3-out folds
# ---------------------------------------------------------------------------
def donor_label_table(obs, donor_key=DONOR_KEY, min_clonal=1, min_anchor=25,
                      eval_min_cells=200, eval_min_clonal=25, eval_min_anchor=25):
    """Per-donor label counts, the role each donor plays, and holdout eligibility.

    ``group`` is one of:
      ``labelled``       — TCR + an ALICE clone + enough benign anchors → δ fitting + eval
      ``labelled_small`` — has an ALICE clone but too few of one class to be a reliable unit →
                           still trains the encoder and still contributes ``δ_d`` if it has both
                           classes, but is never held out **and is not a negative control**
                           (it does carry a clone)
      ``no_clone``       — has TCR and *zero* ALICE-clonal cells → training + negative control
      ``dark``           — no TCR at all → transfer target

    ``eligible`` is the stricter gate for *being held out*: a donor small in either class gives
    a meaningless per-donor AUROC and an unstable per-donor GMM anchor. Ineligible labelled
    donors still contribute cells to the encoder and δ̂ in every fold — the gate restricts who
    we evaluate on, not who we train on.
    """
    g = (obs.groupby(donor_key, observed=True)
         .agg(n_cells=(CLONAL_COL, "size"),
              n_clonal=(CLONAL_COL, "sum"),
              n_anchor=(ANCHOR_COL, "sum"),
              n_tcr=("has_tcr", "sum"),
              study=("study", "first"),
              disease=("disease", "first")))
    for c in ("n_clonal", "n_anchor", "n_tcr"):
        g[c] = g[c].astype(int)
    labelled = (g["n_clonal"] >= min_clonal) & (g["n_anchor"] >= min_anchor)
    # a donor with *any* clonal cell is never a negative control, even if it is too small to
    # be a labelled unit — calling it "no_clone" would assert a clone-bearing sample is clean
    g["group"] = np.select(
        [labelled, g["n_clonal"] > 0, g["n_tcr"] > 0],
        ["labelled", "labelled_small", "no_clone"], default="dark")
    g["clonal_frac"] = (g["n_clonal"] / g["n_cells"]).round(4)
    g["eligible"] = (labelled & (g["n_cells"] >= eval_min_cells)
                     & (g["n_clonal"] >= eval_min_clonal) & (g["n_anchor"] >= eval_min_anchor))
    return g.sort_values(["group", "study", "n_clonal"], ascending=[True, True, False])


def make_triples(donor_tbl, size=3, seed=0, exclude=(), batch_key="study"):
    """Leave-``size``-out folds over the *eligible* labelled donors, every donor tested once.

    Donors are dealt largest-``n_clonal``-first into the currently smallest fold, subject to the
    hard constraint that **no fold may hold out every donor of a batch category** — a held-out
    donor whose batch vanished from training has no trained batch embedding and cannot be
    encoded. Batch totals count *all* donors in ``donor_tbl`` (dark and no-clone donors keep
    their study represented for free).

    ``exclude`` donors are never held out but stay in training. A remainder smaller than
    ``size`` lands in the existing folds rather than forming an undersized fold.
    """
    tbl = donor_tbl
    elig = [d for d in tbl.index[tbl.get("eligible", tbl["group"] == "labelled")]
            if d not in set(exclude)]
    elig = list(tbl.loc[elig].sort_values("n_clonal", ascending=False).index)
    if not elig:
        raise ValueError("no eligible donors")
    batch = tbl[batch_key].astype(str)
    total = batch.value_counts().to_dict()             # donors per batch in the whole cohort

    # a donor that is the ONLY one of its batch can never be held out — holding it out deletes
    # its batch from training. Report it rather than dropping it quietly.
    solo = [d for d in elig if total[batch[d]] <= 1]
    if solo:
        print(f"make_triples: training-only (sole donor of its {batch_key}): "
              + ", ".join(f"{d} [{batch[d]}]" for d in solo))
        elig = [d for d in elig if d not in set(solo)]
        if not elig:
            raise ValueError(f"every eligible donor is the sole donor of its {batch_key}")

    n_folds = max(1, len(elig) // size)
    folds = [[] for _ in range(n_folds)]
    held_batch = [{} for _ in range(n_folds)]
    for d in elig:
        b = batch[d]
        order = sorted(range(n_folds), key=lambda k: (len(folds[k]), k))
        placed = next((k for k in order if held_batch[k].get(b, 0) + 1 < total[b]), None)
        if placed is None:
            raise ValueError(
                f"cannot place {d} ({batch_key}={b}, {total[b]} donor(s) in cohort) without "
                f"emptying that batch — add it to `exclude` or use a coarser batch key")
        folds[placed].append(d)
        held_batch[placed][b] = held_batch[placed].get(b, 0) + 1
    rng = np.random.default_rng(seed)
    for f in folds:
        rng.shuffle(f)
    return folds


def check_batch_coverage(obs, folds, batch_key="study", donor_key=DONOR_KEY):
    """Assert no fold removes every cell of a batch category (would break held-out encoding)."""
    bad = {}
    for k, held in enumerate(folds):
        keep = ~obs[donor_key].isin(held).to_numpy()
        missing = set(obs[batch_key].astype(str)) - set(obs.loc[keep, batch_key].astype(str))
        if missing:
            bad[k] = sorted(missing)
    if bad:
        raise ValueError(f"folds empty a {batch_key} category (add those donors to `exclude`): {bad}")
    return True


# ---------------------------------------------------------------------------
# δ fitting
# ---------------------------------------------------------------------------
def fit_delta(z, donors, is_clonal, is_anchor, train_donors, shuffle_seed=None):
    """Donor-averaged clonal direction ``δ̂`` and the per-donor ``δ_d`` it averages.

    ``shuffle_seed`` permutes the clonal/anchor assignment *within* each training donor —
    the null used as the sanity floor (δ̂ then carries no malignancy information).
    """
    donors = np.asarray(donors)
    is_clonal = np.asarray(is_clonal, dtype=bool)
    is_anchor = np.asarray(is_anchor, dtype=bool)
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None

    rows, deltas = [], []
    for d in train_donors:
        m = donors == d
        lab = m & (is_clonal | is_anchor)
        if lab.sum() < 2:
            continue
        pos = is_clonal[lab]
        if rng is not None:
            pos = rng.permutation(pos)
        if pos.sum() == 0 or (~pos).sum() == 0:
            continue
        zl = z[lab]
        dd = zl[pos].mean(0) - zl[~pos].mean(0)
        deltas.append(dd)
        rows.append({"donor": d, "n_clonal": int(pos.sum()), "n_anchor": int((~pos).sum()),
                     "norm": float(np.linalg.norm(dd))})
    if not deltas:
        raise ValueError("no training donor had both clonal and anchor cells")
    delta = np.mean(np.stack(deltas), axis=0)
    return delta, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-donor benign anchor (the "we don't know this donor's controls" step)
# ---------------------------------------------------------------------------
BENIGN_GATE_FRAC = 0.5
"""Default two-mode gate for :func:`gmm_benign_side`, as a fraction of ``‖δ̂‖``.

The nearer-centroid cut sits at ``‖δ̂‖/2``, so requiring the donor's own two GMM component means
to be at least half of ``‖δ̂‖`` apart asks for no more separation than the rule needs to call a
cell at all. Pass ``gate_frac=None`` anywhere below to restore the ungated behaviour.
"""


def gmm_benign_side(s, seed=0, max_frac=0.9, min_sep=None):
    """Boolean mask of the benign-side cells of a 1-D projection ``s``, via a 2-component GMM.

    Mirrors nb30's per-donor GMM on ``cnv_score``. Falls back to the lower half if the GMM
    degenerates or claims implausibly many cells (> ``max_frac``) for the malignant side.

    ``min_sep`` gates the split. Without it a 2-component GMM is fitted to **every** donor and the
    fallback is a median split, so a donor with zero clonal cells is still cut in half by
    construction — which is why the healthy controls came back at ``pred_clonal_frac`` 0.59-0.87
    and, more insidiously, why ``mu_ben`` was the lower *mode* rather than the donor's benign
    centroid for every donor, shifting every centered score up by half the within-donor spread.
    When the two component means are closer than ``min_sep`` the donor has no resolvable malignant
    mode and every cell is returned as benign side, so ``mu_ben`` becomes the whole-donor mean.
    """
    from sklearn.mixture import GaussianMixture

    s = np.asarray(s, dtype=float)
    ok = np.isfinite(s)
    out = np.zeros(s.shape, dtype=bool)
    if ok.sum() < 50 or np.allclose(s[ok].std(), 0):
        out[ok] = True if min_sep is not None else (s[ok] <= np.median(s[ok]))
        return out
    gm = GaussianMixture(n_components=2, random_state=seed, n_init=1).fit(s[ok, None])
    means = gm.means_.ravel()
    if min_sep is not None and float(np.ptp(means)) < float(min_sep):
        out[ok] = True
        return out
    lo = int(np.argmin(means))
    lab = gm.predict(s[ok, None])
    ben = lab == lo
    if ben.mean() < (1 - max_frac) or ben.mean() > max_frac:
        ben = s[ok] <= np.median(s[ok])
    out[ok] = ben
    return out


def _auc_raw(y, s):
    """AUROC **without** polarity folding, NaN scores dropped.

    ``M.binary_scores`` reports ``max(auc, 1 - auc)``, which is right for comparing predictors
    of unknown sign but silently lifts a random predictor's floor above 0.5 — the shuffled-label
    null lands near 0.7, not 0.5. This is the unfolded number: a real direction must be > 0.5,
    and the null must sit at 0.5.
    """
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(s)
    if ok.sum() == 0 or not (0 < y[ok].sum() < ok.sum()):
        return float("nan")
    return float(roc_auc_score(y[ok], s[ok]))


def project(z, delta):
    """Unit-norm projection of ``z`` on ``δ̂`` (the malignancy axis)."""
    u = np.asarray(delta, dtype=float)
    u = u / max(np.linalg.norm(u), 1e-12)
    return np.asarray(z) @ u


def donor_centroids(z, delta, seed=0, known_benign=None, gate_frac=BENIGN_GATE_FRAC):
    """``(μ_ben, μ̂_mal, s_centered, thr, single_mode)`` for one donor's cells.

    ``known_benign`` (bool mask) uses the donor's real anchors — only available for training
    donors. Otherwise the benign side is estimated with ``gmm_benign_side`` on the δ̂
    projection, which is the whole point: a TCR-less donor has no known controls.

    ``s_centered`` is the projection measured from ``μ_ben``; the nearer-centroid rule in
    ``z`` is exactly ``s_centered > thr`` with ``thr = ‖δ̂‖/2`` (the midpoint hyperplane).

    ``gate_frac`` scales ``‖δ̂‖`` into the two-mode gate described in :func:`gmm_benign_side`;
    ``single_mode`` reports that the gate fired, i.e. this donor showed no resolvable malignant
    mode and ``μ_ben`` is its overall mean. ``None`` disables the gate (pre-fix behaviour).
    """
    z = np.asarray(z, dtype=float)
    s = project(z, delta)
    norm = float(np.linalg.norm(delta))
    if known_benign is not None:
        ben, single_mode = np.asarray(known_benign, dtype=bool), False
    else:
        min_sep = None if gate_frac is None else float(gate_frac) * norm
        ben = gmm_benign_side(s, seed=seed, min_sep=min_sep)
        single_mode = bool(ben.all())
    if ben.sum() == 0:
        ben = s <= np.median(s)
    mu_ben = z[ben].mean(0)
    mu_mal = mu_ben + np.asarray(delta, dtype=float)
    return mu_ben, mu_mal, s - project(mu_ben[None], delta)[0], norm / 2.0, single_mode


# ---------------------------------------------------------------------------
# Scoring rules
# ---------------------------------------------------------------------------
def _train_centered_scores(z, donors, is_clonal, is_anchor, train_donors, delta):
    """Donor-centered projection + label for the labelled cells of the training donors."""
    donors = np.asarray(donors)
    is_clonal = np.asarray(is_clonal, dtype=bool)
    is_anchor = np.asarray(is_anchor, dtype=bool)
    ss, yy, zz = [], [], []
    for d in train_donors:
        m = donors == d
        ben = is_anchor[m] & ~is_clonal[m]
        if not ben.any() or not is_clonal[m].any():
            continue
        lab = m & (is_clonal | is_anchor)
        _, _, sc, _, _ = donor_centroids(z[m], delta, known_benign=ben)
        keep = (is_clonal | is_anchor)[m]
        ss.append(sc[keep])
        yy.append(is_clonal[lab])
        zz.append(z[lab] - z[m][ben].mean(0))
    if not ss:
        raise ValueError("no training donor had both clonal and benign-anchor cells")
    return np.concatenate(ss), np.concatenate(yy), np.concatenate(zz)


def score_donor(rule, z_h, delta, seed=0, fitted=None, centroids=None,
                gate_frac=BENIGN_GATE_FRAC):
    """``(score, call, μ_ben, μ̂_mal, single_mode)`` for one held-out/dark donor under one rule.

    ``fitted`` carries whatever the rule learned on the training donors (a threshold form for
    ``delta_threshold``, a logistic model for ``logistic_z``); ``delta_centroid`` needs none —
    its cut is the geometric midpoint between ``μ̂_ben`` and ``μ̂_mal``. ``centroids`` reuses a
    precomputed ``donor_centroids`` tuple (the GMM is the slow part; rules sharing a δ̂ share it).
    """
    mu_ben, mu_mal, s, thr, single_mode = (
        centroids if centroids is not None
        else donor_centroids(z_h, delta, seed=seed, gate_frac=gate_frac))
    if rule in ("delta_centroid", "delta_centroid_null"):
        return s, s > thr, mu_ben, mu_mal, single_mode
    if rule == "delta_threshold":
        form, params = fitted
        return s, M.apply_threshold_form(s, form, params), mu_ben, mu_mal, single_mode
    if rule == "logistic_z":
        lr, form, params = fitted
        p = lr.predict_proba(np.asarray(z_h) - mu_ben)[:, 1]
        return p, M.apply_threshold_form(p, form, params), mu_ben, mu_mal, single_mode
    raise ValueError(f"unknown rule {rule!r}")


def fit_rules(z, donors, is_clonal, is_anchor, train_donors, delta, seed=0):
    """Fit the training-side part of every rule. Returns ``{rule: fitted}``."""
    from sklearn.linear_model import LogisticRegression

    s_tr, y_tr, z_tr = _train_centered_scores(z, donors, is_clonal, is_anchor, train_donors, delta)
    fitted = {"delta_centroid": None, "delta_centroid_null": None}
    form, params, _ = M.fit_threshold_form(s_tr, y_tr)
    fitted["delta_threshold"] = (form, params)
    lr = LogisticRegression(max_iter=1000, random_state=seed).fit(z_tr, y_tr)
    p_tr = lr.predict_proba(z_tr)[:, 1]
    lform, lparams, _ = M.fit_threshold_form(p_tr, y_tr)
    fitted["logistic_z"] = (lr, lform, lparams)
    return fitted


# ---------------------------------------------------------------------------
# Cross-validation driver
# ---------------------------------------------------------------------------
def run_cv(z_by_fold, obs, folds, *, latent_name="semantic", train_donors_all=None,
           rules=RULES, seed=0, donor_key=DONOR_KEY, n_null_draws=1,
           gate_frac=BENIGN_GATE_FRAC):
    """Score every rule on every held-out donor of every fold.

    ``z_by_fold`` maps fold index → latent array aligned to ``obs`` (the per-fold model's
    encoding of *all* cells; its training set excluded the fold's held-out donors). Pass a
    single array to reuse one latent for every fold (the leakage-reference / dry-run arm).

    ``train_donors_all`` restricts δ fitting to the labelled donors; defaults to every donor
    with both clonal and anchor cells.

    ``n_null_draws`` is how many label permutations the ``*_null`` rules average over. At 1 (the
    old behaviour) the null is ONE draw per fold, shared by every held-out donor in that fold, so
    the per-donor AUCs are correlated and the reported mean is ~8 correlated samples, not 26
    independent ones — which is how a null whose expectation is 0.5 came back at 0.385 and got
    read as a bug. Pass ~20 and read ``auc_raw`` mean ± sd over draws as the null *distribution*.

    ``gate_frac`` is forwarded to :func:`donor_centroids`; ``None`` restores the ungated
    (pre-fix) benign-side estimate.

    Returns ``(results, percell)``:
      ``results`` — one row per (fold, held-out donor, rule, null draw) with ``M.binary_scores``
                    metrics, predicted vs true clonal fraction, and the centroid-recovery R².
      ``percell`` — per-cell score/call for the held-out donors, indexed by ``obs.index``
                    (null rules contribute draw 0 only, so the frame stays one row per cell).
    """
    donors = obs[donor_key].astype(str).to_numpy()
    is_clonal = obs[CLONAL_COL].to_numpy(dtype=bool)
    is_anchor = obs[ANCHOR_COL].to_numpy(dtype=bool)
    labelled = is_clonal | is_anchor
    if train_donors_all is None:
        train_donors_all = sorted({d for d in np.unique(donors)
                                   if is_clonal[donors == d].any() and is_anchor[donors == d].any()})

    rows, cells = [], []
    for k, held in enumerate(folds):
        z = z_by_fold[k] if isinstance(z_by_fold, dict) else z_by_fold
        z = np.asarray(z, dtype=float)
        train = [d for d in train_donors_all if d not in set(held)]
        # a null rule fans out over `n_null_draws` independent permutations; a real rule is one key
        null_keys = [f"null{j}" for j in range(max(1, int(n_null_draws)))]
        rule_keys = {r: (null_keys if r.endswith("_null") else ["real"]) for r in rules}
        deltas = {"real": fit_delta(z, donors, is_clonal, is_anchor, train)[0]}
        for j, key in enumerate(null_keys):
            deltas[key] = fit_delta(z, donors, is_clonal, is_anchor, train,
                                    shuffle_seed=seed + 1000 + k + 100_000 * j)[0]
        fitted = fit_rules(z, donors, is_clonal, is_anchor, train, deltas["real"], seed=seed)

        for h in held:
            m = donors == h
            lab_h = m & labelled
            y = is_clonal[lab_h]
            if y.sum() == 0 or (~y).sum() == 0:
                print(f"[fold {k}] skip {h}: y has one class ({int(y.sum())}/{int(lab_h.sum())})")
                continue
            keep_lab = labelled[m]
            # the GMM benign-side estimate is the slow step — share it across rules on the same δ̂
            cent = {key: donor_centroids(z[m], deltas[key], seed=seed, gate_frac=gate_frac)
                    for key in deltas}
            for r in rules:
                for draw, key in enumerate(rule_keys[r]):
                    s, call, mu_ben, mu_mal_hat, single_mode = score_donor(
                        r, z[m], deltas[key], seed=seed, fitted=fitted.get(r),
                        centroids=cent[key], gate_frac=gate_frac)
                    sc = M.binary_scores(y, call[keep_lab], score=s[keep_lab])
                    # binary_scores' `auc` is polarity-folded (max(a, 1-a)), which floors a random
                    # direction well above 0.5 — keep the unfolded AUC so the null reads honestly.
                    sc["auc_raw"] = _auc_raw(y, s[keep_lab])
                    sc.update(fold=k, donor=h, rule=r, latent=latent_name, null_draw=draw,
                              study=str(obs.loc[m, "study"].iloc[0]),
                              n_cells=int(m.sum()), n_labelled=int(lab_h.sum()),
                              single_mode=single_mode,
                              true_frac=float(y.mean()), pred_frac=float(call[keep_lab].mean()),
                              pred_frac_all=float(call.mean()),
                              centroid_r2=float(r2_latent(mu_mal_hat, z[lab_h][y].mean(0))))
                    rows.append(sc)
                    if draw:            # keep `percell` one row per cell per rule
                        continue
                    cells.append(pd.DataFrame({"score": s, "call": call, "fold": k, "donor": h,
                                               "rule": r, "latent": latent_name,
                                               "labelled": keep_lab, "y": is_clonal[m]},
                                              index=obs.index[m]))
    results = pd.DataFrame(rows)
    front = ["latent", "rule", "fold", "donor", "study", "null_draw", "auc", "auc_raw", "f1",
             "balanced_acc", "precision", "recall", "specificity", "jaccard", "true_frac",
             "pred_frac", "centroid_r2", "single_mode", "n", "n_cells", "n_labelled"]
    results = results[[c for c in front if c in results] +
                      [c for c in results.columns if c not in front]]
    return results, pd.concat(cells) if cells else pd.DataFrame()


def summarize(results, by=("latent", "rule")):
    """mean ± sd of the headline metrics over held-out donors.

    With ``n_null_draws > 1`` a null rule contributes one row per (donor, draw); ``n_donors``
    therefore counts rows, and ``n_null_draws`` reports the fan-out so the two are not confused.
    ``auc_raw_sd`` for a null rule is then the spread of the null itself — the number to compare
    0.5 against, rather than a single draw.
    """
    cols = ["auc", "auc_raw", "f1", "balanced_acc", "recall", "specificity", "centroid_r2"]
    g = results.groupby(list(by), observed=True)
    out = pd.DataFrame({"n_donors": g.size()})
    if "null_draw" in results:
        out["n_null_draws"] = g["null_draw"].nunique()
    if "single_mode" in results:
        out["frac_single_mode"] = g["single_mode"].mean().round(3)
    for c in cols:
        out[f"{c}_mean"] = g[c].mean().round(3)
        out[f"{c}_sd"] = g[c].std().round(3)
    return out.sort_values("auc_mean", ascending=False)


# ---------------------------------------------------------------------------
# Applied call on the dark donors
# ---------------------------------------------------------------------------
def apply_to_donors(z, obs, target_donors, delta, *, seed=0, donor_key=DONOR_KEY,
                    rule="delta_centroid", fitted=None, gate_frac=BENIGN_GATE_FRAC):
    """Per-target-donor call, using each donor's own GMM-estimated benign centroid.

    ``rule`` defaults to the parameter-free ``delta_centroid``; pass any other rule together with
    the ``fitted`` object ``fit_rules`` produced on the same latent to apply it instead.

    ``gate_frac`` forwards the two-mode gate (see :func:`gmm_benign_side`). ``single_mode`` in the
    per-donor frame flags donors where it fired — for a healthy control that is the *expected*
    outcome, and a run where no control is flagged should be read as the gate not working rather
    than as every control carrying a clonal population.

    Returns ``(percell, per_donor)``. Nothing here is validated against external truth —
    these donors have no TCR and no published malignant label.
    """
    donors = obs[donor_key].astype(str).to_numpy()
    parts, rows = [], []
    for d in target_donors:
        m = donors == d
        if m.sum() < 50:
            print(f"skip {d}: {int(m.sum())} cells")
            continue
        s, call, _, _, single_mode = score_donor(rule, z[m], delta, seed=seed, fitted=fitted,
                                                 gate_frac=gate_frac)
        parts.append(pd.DataFrame({"donor": d, "clonal_score": s, "clonal_call": call},
                                  index=obs.index[m]))
        rows.append({"donor": d, "n_cells": int(m.sum()), "pred_clonal_frac": float(call.mean()),
                     "single_mode": single_mode,
                     "study": str(obs.loc[m, "study"].iloc[0]),
                     "disease": str(obs.loc[m, "disease"].iloc[0])})
    return (pd.concat(parts) if parts else pd.DataFrame()), pd.DataFrame(rows)
