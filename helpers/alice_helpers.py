"""ALICE-style TCR-neighborhood enrichment for the CTCL atlas (TRB, V-agnostic).

ALICE (Pogorelyy et al., PNAS 2019): vertices = clonotypes, edges join CDR3s
differing by <=1 amino-acid substitution; flag clonotypes with **more neighbors
than a generative VDJ-recombination null predicts**. Null = OLGA Pgen (humanTRB)
scaled by a thymic-selection factor Q; significance = Poisson survival, BH-FDR.

This module is the single-snapshot, control-free variant: no empirical/healthy
background, no TCRNET. TRB chain only. We marginalize Pgen over V/J (the atlas
clone table carries no V-gene), i.e. V-agnostic neighbor counting + Pgen.

Used by ``21_tcr_alice_neighborhood.ipynb`` for three analyses:
  Use 1  per-patient malignant subclone-family recovery (founder + <=1-aa variants)
  Use 2  cross-patient convergence of founders (expected negative)
  Use 3  reactive antigen-driven clusters in the CD8 infiltrate

Heavy step = the OLGA Pgen sweep over single-mismatch variants. We only compute it
for clonotypes that actually have >=1 observed neighbor (a 0-neighbor clonotype can
never be enriched), which collapses the work by ~orders of magnitude.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import poisson
from statsmodels.stats.multitest import fdrcorrection

AA = "ACDEFGHIKLMNPQRSTVWY"           # 20 canonical amino acids
_AASET = set(AA)


# --------------------------------------------------------------------------- #
# OLGA generative model (human TRB, default IGoR model shipped with olga)
# --------------------------------------------------------------------------- #
def load_olga_trb():
    """Load the default human-TRB OLGA Pgen model. Returns a GenerationProbabilityVDJ."""
    import olga.load_model as load_model
    import olga.generation_probability as pgen
    import olga

    mdir = os.path.join(os.path.dirname(olga.__file__), "default_models", "human_T_beta")
    params = os.path.join(mdir, "model_params.txt")
    marginals = os.path.join(mdir, "model_marginals.txt")
    v_anchor = os.path.join(mdir, "V_gene_CDR3_anchors.csv")
    j_anchor = os.path.join(mdir, "J_gene_CDR3_anchors.csv")

    gen = load_model.GenomicDataVDJ()
    gen.load_igor_genomic_data(params, v_anchor, j_anchor)
    model = load_model.GenerativeModelVDJ()
    model.load_and_process_igor_model(marginals)
    return pgen.GenerationProbabilityVDJ(model, gen)


def make_pgen(model):
    """Memoized scalar Pgen(cdr3) (V/J-marginalized). Returns a cached callable."""

    @lru_cache(maxsize=None)
    def _pgen(cdr3: str) -> float:
        try:
            return float(model.compute_aa_CDR3_pgen(cdr3))
        except Exception:
            return 0.0

    return _pgen


# --------------------------------------------------------------------------- #
# Single-mismatch neighborhood (classic ALICE: substitutions, same length)
# --------------------------------------------------------------------------- #
def one_mismatch_variants(cdr3: str):
    """Yield every single-substitution variant of ``cdr3`` (19*L, excludes self)."""
    for i, c in enumerate(cdr3):
        for a in AA:
            if a != c:
                yield cdr3[:i] + a + cdr3[i + 1:]


def _valid(seq: str) -> bool:
    return bool(seq) and set(seq) <= _AASET


def neighbor_graph(seqs) -> nx.Graph:
    """Graph over unique CDR3s; edge iff Hamming distance == 1 (same length).

    Built by wildcard-bucket hashing: mask one position at a time; seqs sharing a
    (length, masked-pattern) bucket differ by exactly one residue. O(N*L), not O(N^2).
    """
    seqs = [s for s in dict.fromkeys(seqs) if _valid(s)]
    g = nx.Graph()
    g.add_nodes_from(seqs)
    buckets: dict[tuple, list[str]] = {}
    for s in seqs:
        for i in range(len(s)):
            buckets.setdefault((len(s), i, s[:i] + "*" + s[i + 1:]), []).append(s)
    for members in buckets.values():
        if len(members) > 1:
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    g.add_edge(members[a], members[b])
    return g


def observed_degree(seqs) -> dict:
    """seq -> number of distinct <=1-substitution neighbors among ``seqs``."""
    g = neighbor_graph(seqs)
    return dict(g.degree())


# --------------------------------------------------------------------------- #
# ALICE test
# --------------------------------------------------------------------------- #
def expected_pgen_sum(cdr3: str, pgen_fn) -> float:
    """Per-unit expected mass = sum of Pgen over all single-mismatch variants."""
    return float(sum(pgen_fn(v) for v in one_mismatch_variants(cdr3) if _valid(v)))


def calibrate_Q(obs_deg: np.ndarray, lam_unit: np.ndarray) -> float:
    """Empirical thymic-selection scale Q.

    Under the null, E[obs] = Q * lam_unit. We fit Q on the *bulk* (non-outlier)
    clonotypes via the median ratio of total observed to total expected mass,
    trimming the top decile of obs so true ALICE hits don't inflate the scale.
    """
    obs_deg = np.asarray(obs_deg, float)
    lam_unit = np.asarray(lam_unit, float)
    m = (lam_unit > 0)
    if m.sum() < 5:
        return 1.0
    o, l = obs_deg[m], lam_unit[m]
    keep = o <= np.quantile(o, 0.90)
    denom = l[keep].sum()
    return float(o[keep].sum() / denom) if denom > 0 else 1.0


def alice_test(clonotypes: pd.DataFrame, pgen_fn, Q: float | None = None,
               alpha: float = 0.05, seq_col: str = "cdr3") -> pd.DataFrame:
    """Run ALICE on one repertoire (one row per unique clonotype).

    ``clonotypes`` must have ``seq_col``; extra cols (e.g. n_cells, is_founder)
    are carried through. Returns the frame + obs_deg, exp_deg, pois_p, q_bh,
    significant. Expected mass is computed only for clonotypes with obs_deg>=1.
    """
    df = clonotypes.copy()
    seqs = df[seq_col].astype(str).tolist()
    deg = observed_degree(seqs)
    df["obs_deg"] = df[seq_col].map(lambda s: deg.get(str(s), 0)).astype(int)

    n_unique = df[seq_col].nunique()
    has_nb = df["obs_deg"] >= 1
    lam_unit = np.zeros(len(df))
    lam_unit[has_nb.values] = [
        n_unique * expected_pgen_sum(str(s), pgen_fn)
        for s in df.loc[has_nb, seq_col]
    ]
    if Q is None:
        Q = calibrate_Q(df["obs_deg"].values, lam_unit)
    df["Q"] = Q
    df["exp_deg"] = Q * lam_unit
    # P(X >= obs) under Poisson(exp_deg); clonotypes with no neighbors -> p=1
    df["pois_p"] = 1.0
    df.loc[has_nb, "pois_p"] = poisson.sf(
        df.loc[has_nb, "obs_deg"].values - 1, df.loc[has_nb, "exp_deg"].values)
    p = df["pois_p"].fillna(1.0).values
    rej, q = fdrcorrection(p, alpha=alpha)
    df["q_bh"] = q
    df["significant"] = rej & has_nb.values
    return df.sort_values(["significant", "obs_deg"], ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Cohort -> per-clonotype tables (from skin_T_tcr_annotated obs)
# --------------------------------------------------------------------------- #
def clonotype_table(obs: pd.DataFrame, group: str = "donor",
                    key_col: str = "tcr_clone_id", seq_col: str = "trb_cdr3") -> pd.DataFrame:
    """Collapse per-cell obs to unique (group, TRB-CDR3) clonotypes.

    The TRB CDR3 is taken from the **unified clone key** ``tcr_clone_id`` (set by
    ``recompute_dominant_clone`` for every cohort, ``TRB:<cdr3>``; Li2024 cells carry
    their key only here, not in ``trb_cdr3``), falling back to ``trb_cdr3``. TRA-only
    clones (no TRB) are dropped — TRB-only analysis.

    Expects obs with ``has_tcr``, ``group``, ``key_col`` and/or ``seq_col``, and
    (optional) ``tcr_is_dominant_clone``. Returns: group, cdr3, n_cells, is_founder.
    """
    o = obs.copy()
    o = o[o["has_tcr"].astype(bool)]
    trb = pd.Series("", index=o.index)
    if key_col in o.columns:
        trb = o[key_col].astype(str).str.extract(r"^TRB:([A-Z]+)$", expand=False).fillna("")
    if seq_col in o.columns:
        trb = trb.mask(trb == "", o[seq_col].astype(str))
    o["cdr3"] = trb
    o = o[o["cdr3"].map(_valid)]
    founder = ("tcr_is_dominant_clone" in o.columns)
    rows = []
    for gval, sub in o.groupby(group, observed=True):
        vc = sub["cdr3"].value_counts()
        fset = (set(sub.loc[sub["tcr_is_dominant_clone"].astype(bool), "cdr3"])
                if founder else set())
        for cdr3, n in vc.items():
            rows.append({group: gval, "cdr3": cdr3, "n_cells": int(n),
                         "is_founder": cdr3 in fset})
    return pd.DataFrame(rows)


def run_alice_by_group(clono: pd.DataFrame, pgen_fn, group: str = "donor",
                       Q: float | None = None, alpha: float = 0.05) -> pd.DataFrame:
    """alice_test per group (e.g. per donor). Concatenates results with the group key."""
    out = []
    for gval, sub in clono.groupby(group, observed=True):
        if sub["cdr3"].nunique() < 2:
            continue
        res = alice_test(sub.drop(columns=[group]), pgen_fn, Q=Q, alpha=alpha)
        res.insert(0, group, gval)
        out.append(res)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def founder_family(clono_group: pd.DataFrame, seeds=None) -> set:
    """CDR3s in the <=1-substitution connected component(s) of the seed clone(s).

    ``clono_group`` = one group's clonotype rows (must have cdr3). Seeds default to
    the ``is_founder`` clones; pass ``seeds`` to use an explicit set instead (e.g. the
    largest clonotype of a non-malignant donor). Returns the set of CDR3s reachable
    from any seed (the subclone family), including the seeds themselves; empty if no seed.
    """
    seqs = clono_group["cdr3"].astype(str).tolist()
    g = neighbor_graph(seqs)
    if seeds is None:
        seeds = clono_group.loc[clono_group["is_founder"].astype(bool), "cdr3"].astype(str)
    fam: set = set()
    for f in set(seeds):
        if f in g:
            fam |= nx.node_connected_component(g, f)
    return fam
