#!/usr/bin/env python
"""Donor-level long export of the pay51 payload-selectivity screen (report 51).

Purpose: give the figures a per-donor frame so Gate 1-3 can be redrawn as paired per-donor
strips instead of medians. This re-runs the AGGREGATION of `jobs/run_payload_screen.py` only --
same object, same arms, same `mal_cluster` call, same log1p CP10K on `obs.total_counts`, same
cohort filter, same coverage mask, same gene set -- and writes three new tables. It derives no
new statistic beyond what report 51 already published, and it overwrites nothing.

What it imports rather than copies: `run_payload_screen.pass_t` / `pass_joint` (the matrix pass
and the arm definitions), `.gate1` (the paired tests), `.coverage_mask`, and the project helpers
`target_cohort` / `target_screen`. The only change made to `run_payload_screen.py` is a
`min_cells=` argument on `pass_t` and `pass_joint` (default unchanged): this export needs the
sub-20-cell donor-arms so it can FLAG them rather than drop them. Aggregation is per group, so
every group at or above the floor is bit-identical to the published run.

Determinism: the depth-matching step reuses the existing seed, `run_payload_screen.SEED = 0`,
through one `np.random.default_rng(SEED)` created here and handed to `pass_t` exactly as
`main()` does. Thinning is applied to the whole cell block before grouping, so it does not
depend on which donor-arms pass the cell floor.

n_up / n_down attribution: in `pay51_cd4_paired.csv` these come from
`paired_counts(..., value="mean_all")` -- they belong to the MEAN-EXPRESSION test and count
donors whose per-donor mean log1p CP10K is higher / lower on the malignant arm. They do not
belong to either detection test. They are emitted here as `n_up_mean` / `n_down_mean` so the
attribution cannot be lost again. (`reports/51_payload_selectivity.md` prints them beside the
expression columns and makes no claim to the contrary; the misattribution is not in it.)

Two deviations from the export brief, both deliberate:
  * The brief says 53 panel genes. All 58 are carried, with `gene_set` naming the family, because
    the 5 audit genes include GAPDH -- the ONLY gene in the whole set that the coverage mask
    removes, in 31 li2024 donors. Those NaNs are the evidence that the NaN rule works. Filter
    `gene_set != "audit"` for exactly the 53.
  * `median_total_counts` stays float, not int. A per-donor-arm median UMI over an even number of
    cells is a half-integer (the published arm medians are 3330.75, 2038.5); rounding it would
    break the 1e-6 verification against `pay51_arms.csv`.

Outputs (new files only):
  tables/pay51_donor_long.csv         (donor, arm, gene) grid, NaN where not evaluated
  tables/pay51_gate2_reach_donor.csv  (donor, arm, payload) complete-complex reach
  tables/pay51_gene_stats.csv         one row per gene, three tests, suffixed per test
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, os.environ.get("TARGET_SCREEN_THREADS", "4"))
os.environ.setdefault("MPLBACKEND", "Agg")

import h5py
import numpy as np
import pandas as pd


def _nb_dir():
    for p in [Path(__file__).resolve().parent] + list(Path(__file__).resolve().parents):
        if p.name == "MF" and (p / "data").is_dir():
            return p
    raise RuntimeError("MF project root not found")


NB_DIR = _nb_dir()
sys.path.insert(0, str(NB_DIR / "helpers"))
sys.path.insert(0, str(NB_DIR / "jobs"))

import cluster_malig_helpers as CM          # noqa: E402
import malignancy as MAL                    # noqa: E402
import target_cohort as TC                  # noqa: E402
import run_payload_screen as P              # noqa: E402

TOL = 1e-6
FAILURES: list[str] = []


def hr(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78, flush=True)


def check(label, dev, n):
    """Record a recomputation check. Above TOL is an error, not a warning."""
    ok = np.isfinite(dev) and dev <= TOL
    print(f"  [{'OK ' if ok else 'FAIL'}] {label:<58} max|dev| = {dev:.3e}  over {n:,} values")
    if not ok:
        FAILURES.append(f"{label}: max|dev| = {dev:.3e} over {n:,} values")


def dev_of(new, old, cols, keys, label):
    """Max absolute deviation between a recomputed frame and a published table."""
    m = new.merge(old, on=keys, how="outer", suffixes=("_new", "_pub"), indicator=True)
    if (m["_merge"] != "both").any():
        bad = m.loc[m["_merge"] != "both", keys].astype(str).agg(" ".join, axis=1).tolist()
        FAILURES.append(f"{label}: {len(bad)} key(s) present on one side only: {bad[:8]}")
        print(f"  [FAIL] {label}: keys only on one side: {bad[:8]}")
        m = m[m["_merge"] == "both"]
    worst, n = 0.0, 0
    for c in cols:
        a = pd.to_numeric(m[f"{c}_new"], errors="coerce").to_numpy(float)
        b = pd.to_numeric(m[f"{c}_pub"], errors="coerce").to_numpy(float)
        both_nan = np.isnan(a) & np.isnan(b)
        d = np.abs(a - b)
        d[both_nan] = 0.0
        if np.isnan(d).any():
            FAILURES.append(f"{label}/{c}: {int(np.isnan(d).sum())} value(s) NaN on one side only")
        worst = max(worst, float(np.nanmax(d)) if len(d) else 0.0)
        n += int((~both_nan).sum())
    check(label, worst, n)


# --------------------------------------------------------------------- aggregation
def aggregate_everything():
    """The published aggregation, with the cell floor lifted so sub-threshold arms survive."""
    args = SimpleNamespace(max_cells=None, skip_joint=False)
    rng = np.random.default_rng(P.SEED)
    CM.use_tissue("skin")
    MAL.use_tissue("skin")
    assert MAL.CLUSTER_PQ.name.startswith("skin_"), MAL.CLUSTER_PQ

    with h5py.File(P.T_OBJ, "r") as h5:
        genes_all = pd.Index(CM._read_elem(h5["var"][h5["var"].attrs["_index"]])).astype(str)
    missing = [g for g in P.GENES if g not in set(genes_all)]
    assert not missing, f"genes absent from the object var: {missing}"
    gi = genes_all.get_indexer(P.GENES)

    lr, rr, _ = P.pass_t(args, genes_all, gi, rng, min_cells=0)
    lr2, rr2, _ = P.pass_joint(args, genes_all, gi, min_cells=0)
    return pd.concat(lr + lr2, ignore_index=True), pd.DataFrame(rr + rr2)


def cohort_and_mask(long_all, reach_all):
    """The published cohort filter and coverage mask, with masked cells set to NaN, not dropped."""
    don = TC.skin_donor_table()
    with h5py.File(P.T_OBJ, "r") as h5:
        o = CM.read_obs(h5, cols=["donor", "study"])
    o = o.astype({"donor": str, "study": str}).drop_duplicates("donor")
    d2s = o.set_index("donor")["study"]
    long_all["study"] = long_all["donor"].map(d2s)
    assert long_all["study"].notna().all(), "a donor key has no study"
    long_all["cohort"] = long_all["donor"].str.split("__").str[0]
    chk = long_all.groupby("cohort")["study"].nunique()
    assert (chk == 1).all(), f"cohort prefix is not 1:1 with study: {chk[chk > 1].to_dict()}"
    c2s = long_all.drop_duplicates("cohort").set_index("cohort")["study"].to_dict()

    long_c = TC.filter_long(long_all, don)
    reach_c = reach_all[reach_all["donor"].isin(set(long_c["donor"]))].copy()
    reach_c["study"] = reach_c["donor"].map(d2s)

    cov = P.coverage_mask({c: s for c, s in c2s.items() if c in P.COHORT_SRC})
    meas = cov.set_index(["study", "gene"])["measured"]
    key = pd.MultiIndex.from_arrays([long_c["study"], long_c["gene"]])
    long_c["measured"] = meas.reindex(key).to_numpy()
    viol = int(((~long_c["measured"]) & (long_c["pct_expr"] > 0)).sum())
    print(f"coverage mask: {int((~long_c['measured']).sum()):,} of {len(long_c):,} "
          f"(donor-arm, gene) cells unmeasured; {viol} contradict the mask")
    assert viol == 0, "a gene the source object does not carry has non-zero counts"
    for c in ("pct_expr", "mean_all", "sd_all", "pct_expr_ds"):
        long_c.loc[~long_c["measured"], c] = np.nan

    return long_c, reach_c, don, d2s, cov


# --------------------------------------------------------------------- exports
def build_donor_long(long_c, d2s):
    """One row per (donor, arm, gene) the analysis could have evaluated. Absent -> NaN."""
    donors = sorted(set(long_c["donor"]))
    arms = [a for a in P.ARM_ORDER]
    idx = pd.MultiIndex.from_product([donors, arms, P.GENES], names=["donor", "arm", "gene"])
    base = long_c.set_index(["donor", "arm", "gene"])
    assert base.index.is_unique, "duplicate (donor, arm, gene) in the long frame"
    out = base.reindex(idx).reset_index()

    out["study"] = out["donor"].map(d2s)
    out["gene_set"] = out["gene"].map(P.FAMILY_OF)
    is_cd4 = out["arm"].isin([P.MALIG, P.REACT])
    out["pct_expr_depthmatched"] = np.where(is_cd4, out["pct_expr_ds"], np.nan)
    out = out.rename(columns={"donor": "donor_key", "mean_all": "mean_log1p_cp10k"})
    out["n_cells"] = out["n_cells"].astype("Int64")
    out["passes_min_cells"] = out["n_cells"] >= P.MIN_CELLS   # NA where the arm is absent

    cols = ["study", "donor_key", "arm", "gene", "gene_set", "n_cells", "median_total_counts",
            "mean_log1p_cp10k", "pct_expr", "pct_expr_depthmatched", "passes_min_cells"]
    return out[cols].sort_values(["arm", "donor_key", "gene"], ignore_index=True)


def build_gate2_donor(reach_c, long_c, d2s):
    """One row per (donor, arm, payload). Same cell-level co-expression as report 51 §4b."""
    chains = sorted({c for ch in P.COMPLEX.values() for c in ch})
    unmeasured = long_c.loc[~long_c["measured"] & long_c["gene"].isin(chains), "gene"].unique()
    assert len(unmeasured) == 0, f"a complex chain is coverage-masked: {sorted(unmeasured)}"

    payloads = [p for p in P.COMPLEX if p in reach_c.columns]
    donors = sorted(set(long_c["donor"]))
    arms = [a for a in P.ARM_ORDER]
    idx = pd.MultiIndex.from_product([donors, arms, payloads],
                                     names=["donor", "arm", "payload"])
    lng = (reach_c.melt(id_vars=["arm", "donor", "n_cells"], value_vars=payloads,
                        var_name="payload", value_name="frac_reachable")
           .set_index(["donor", "arm", "payload"]))
    assert lng.index.is_unique, "duplicate (donor, arm, payload)"
    out = lng.reindex(idx).reset_index()
    out["study"] = out["donor"].map(d2s)
    out = out.rename(columns={"donor": "donor_key"})
    out["n_cells"] = out["n_cells"].astype("Int64")
    return out[["study", "donor_key", "arm", "payload", "n_cells", "frac_reachable"]
               ].sort_values(["payload", "arm", "donor_key"], ignore_index=True)


def build_gene_stats(long_q):
    """One row per gene: the three tests report 51 runs, each column suffixed with its test."""
    g1 = pd.concat([P.gate1(long_q, gs, f) for f, gs in P.FAMILIES.items()], ignore_index=True)
    mean_cols = {"log2fc_paired": "log2fc_paired_mean", "p": "p_mean", "q": "q_mean",
                 "n_up": "n_up_mean", "n_down": "n_down_mean", "n_tie": "n_tie_mean",
                 "n_donors_paired": "n_donors_paired_mean",
                 "frac_donors_up": "frac_donors_up_mean"}
    det_cols = {"log2fc_detect": "log2fc_detect_raw", "p_detect": "p_detect_raw",
                "q_detect": "q_detect_raw",
                "n_donors_paired_detect": "n_donors_paired_detect_raw"}
    out = g1.rename(columns={**mean_cols, **det_cols})
    out = out[["gene", "family"] + list(mean_cols.values()) + list(det_cols.values())]

    dsl = long_q.dropna(subset=["pct_expr_ds"]).copy()
    dsl["pct_expr"] = dsl["pct_expr_ds"]
    dm = pd.concat([P.gate1(dsl, gs, f) for f, gs in P.FAMILIES.items()], ignore_index=True)
    dm = dm[["gene", "log2fc_detect", "p_detect", "q_detect", "n_donors_paired_detect"]].rename(
        columns={"log2fc_detect": "log2fc_detect_depthmatched",
                 "p_detect": "p_detect_depthmatched",
                 "q_detect": "q_detect_depthmatched",
                 "n_donors_paired_detect": "n_donors_paired_detect_depthmatched"})
    out = out.merge(dm, on="gene", how="outer").rename(columns={"family": "gene_set"})
    order = {g: i for i, g in enumerate(P.GENES)}
    return out.sort_values("gene", key=lambda s: s.map(order), ignore_index=True)


# --------------------------------------------------------------------- verification
def verify_published(long_q, reach_c, stats):
    hr("VERIFICATION 1 -- every published summary, recomputed from the exported long frame")

    arms_pub = pd.read_csv(P.TAB_DIR / "pay51_arms.csv")
    arms_new = (long_q.drop_duplicates(["arm", "donor"])
                .groupby("arm")
                .agg(n_donors=("donor", "nunique"), n_cells=("n_cells", "sum"),
                     n_studies=("study", "nunique"),
                     median_total_counts=("median_total_counts", "median"),
                     median_n_genes=("median_n_genes", "median"))
                .reset_index())
    dev_of(arms_new, arms_pub,
           ["n_donors", "n_cells", "n_studies", "median_total_counts", "median_n_genes"],
           ["arm"], "pay51_arms.csv (per-arm donors, cells, depth)")

    g1_pub = pd.read_csv(P.TAB_DIR / "pay51_cd4_paired.csv")
    prev = []
    for a, tag in ((P.MALIG, "malig"), (P.REACT, "react"), (P.HC, "hc")):
        p = TS_prevalence(long_q, a)[["gene", "median_pct_expr", "frac_donors_positive",
                                      "n_donors"]]
        prev.append(p.rename(columns={c: f"{tag}_{c}" for c in p.columns if c != "gene"}))
    prev_new = prev[0].merge(prev[1], on="gene").merge(prev[2], on="gene")
    dev_of(prev_new, g1_pub,
           ["malig_median_pct_expr", "react_median_pct_expr", "hc_median_pct_expr",
            "malig_frac_donors_positive", "malig_n_donors", "react_n_donors", "hc_n_donors"],
           ["gene"], "pay51_cd4_paired.csv (per-gene median pct_expr, CD4 arms)")

    dev_of(stats.rename(columns={"log2fc_paired_mean": "log2fc_paired", "q_mean": "q",
                                 "p_mean": "p", "n_up_mean": "n_up", "n_down_mean": "n_down",
                                 "n_donors_paired_mean": "n_donors_paired",
                                 "log2fc_detect_raw": "log2fc_detect",
                                 "q_detect_raw": "q_detect"}),
           g1_pub,
           ["log2fc_paired", "p", "q", "n_up", "n_down", "n_donors_paired", "log2fc_detect",
            "q_detect"],
           ["gene"], "pay51_cd4_paired.csv (Gate 1 tests)")

    dm_pub = pd.read_csv(P.TAB_DIR / "pay51_sens_depth_matched.csv")
    dev_of(stats[["gene", "log2fc_detect_depthmatched", "q_detect_depthmatched"]], dm_pub,
           ["log2fc_detect_depthmatched", "q_detect_depthmatched"], ["gene"],
           "pay51_sens_depth_matched.csv (depth-matched detection)")

    g2_pub = pd.read_csv(P.TAB_DIR / "pay51_gate2_summary.csv")
    payloads = [p for p in P.COMPLEX if p in reach_c.columns]
    rl = reach_c.melt(id_vars=["arm", "donor", "n_cells"], value_vars=payloads,
                      var_name="payload", value_name="frac")
    rl = rl[rl["n_cells"] >= P.MIN_CELLS]
    g2_new = (rl.groupby(["payload", "arm"])
              .agg(n_donors=("donor", "nunique"), median_frac=("frac", "median"),
                   mean_frac=("frac", "mean"), max_frac=("frac", "max")).reset_index())
    g2_new = g2_new.merge(rl.assign(_h=rl["frac"] >= 0.10)
                          .groupby(["payload", "arm"])["_h"].mean()
                          .rename("donor_prev_ge10pct").reset_index(),
                          on=["payload", "arm"], how="left")
    dev_of(g2_new, g2_pub, ["n_donors", "median_frac", "mean_frac", "max_frac",
                            "donor_prev_ge10pct"],
           ["payload", "arm"], "pay51_gate2_summary.csv (reach median, donor share >=10%)")

    g3_pub = pd.read_csv(P.TAB_DIR / "pay51_gate3_summary.csv")
    lig = long_q[long_q["gene"].isin(P.C_LIGANDS)]
    g3_new = (lig.groupby(["gene", "arm"])
              .agg(n_donors=("donor", "nunique"),
                   median_pct_expr=("pct_expr", "median"),
                   mean_pct_expr=("pct_expr", "mean"),
                   median_mean_all=("mean_all", "median"),
                   frac_donors_pos=("pct_expr",
                                    lambda s: float((s >= P.DETECT_FLOOR).mean())))
              .reset_index())
    dev_of(g3_new, g3_pub, ["n_donors", "median_pct_expr", "mean_pct_expr", "median_mean_all",
                            "frac_donors_pos"],
           ["gene", "arm"], "pay51_gate3_summary.csv (ligand pseudobulk per arm)")


def TS_prevalence(long, arm):
    import target_screen as TS
    return TS.prevalence(long, arm, detect_floor=P.DETECT_FLOOR)


def verify_pairing(long_q, cov):
    hr("VERIFICATION 2 -- is n_donors_paired = 75 for every gene, or is it zero-fill?")
    w = (long_q[long_q["arm"].isin([P.MALIG, P.REACT])]
         .pivot_table(index=["gene", "donor"], columns="arm", values="mean_all"))
    both = (w.dropna(subset=[P.MALIG, P.REACT]).groupby(level="gene").size()
            .rename("n_donors_both_cd4_arms").reset_index())
    both["gene_set"] = both["gene"].map(P.FAMILY_OF)
    vc = both["n_donors_both_cd4_arms"].value_counts().sort_index()
    print("donors with a non-NaN value in BOTH CD4 arms, over the 58 genes:")
    print("  " + "; ".join(f"{n} donors -> {k} gene(s)" for n, k in vc.items()))
    panel = both[both["gene_set"] != "audit"]
    if panel["n_donors_both_cd4_arms"].nunique() == 1:
        print(f"  all {len(panel)} panel genes have exactly "
              f"{int(panel['n_donors_both_cd4_arms'].iloc[0])} paired donors -- stated "
              "explicitly, as asked.")
    odd = both[both["n_donors_both_cd4_arms"] != both["n_donors_both_cd4_arms"].max()]
    print("\ngenes that do NOT reach the maximum (this is the mask working, not zero-fill):")
    print(odd.to_string(index=False) if len(odd) else "  none")

    unmeas = cov[~cov["measured"]][["gene", "study", "cohort", "n_genes_in_source"]]
    print("\ncross-check against pay51_gene_coverage.csv -- (gene, study) pairs not measured:")
    print(unmeas.to_string(index=False) if len(unmeas) else "  none")
    cov_pub = pd.read_csv(P.TAB_DIR / "pay51_gene_coverage.csv")
    zero = cov_pub[~cov_pub["measured"]][["gene", "study", "n_donors_measured"]]
    print("published gene_coverage rows with measured = False:")
    print(zero.to_string(index=False) if len(zero) else "  none")
    for g in unmeas["gene"].unique():
        lost = int(both.loc[both["gene"].eq(g), "n_donors_both_cd4_arms"].iloc[0])
        print(f"  {g}: {both['n_donors_both_cd4_arms'].max() - lost} paired donors removed by "
              f"the mask ({lost} remain) -- these would have been zero-filled.")


def verify_nan(dl, g2d):
    hr("VERIFICATION 3 -- NaN versus 0, by reason")
    n = len(dl)
    arm_absent = dl["n_cells"].isna()
    unmeasured = (~arm_absent) & dl["pct_expr"].isna()
    zero = (~arm_absent) & (~unmeasured) & (dl["pct_expr"] == 0)
    pos = (~arm_absent) & (~unmeasured) & (dl["pct_expr"] > 0)
    print(f"pay51_donor_long.csv: {n:,} rows "
          f"({dl['donor_key'].nunique()} donors x {dl['arm'].nunique()} arms x "
          f"{dl['gene'].nunique()} genes)")
    for lab, m in (("NaN  - arm absent in that donor (no cells of it at all)", arm_absent),
                   ("NaN  - gene not in that study's source object (coverage mask)", unmeasured),
                   ("0    - measured, genuinely zero in every cell of the arm", zero),
                   ("> 0  - measured and detected", pos)):
        print(f"  {lab:<62} {int(m.sum()):>8,}  ({m.mean():6.2%})")
    assert int((arm_absent | unmeasured | zero | pos).sum()) == n, "row classes do not partition"

    sub = dl[~arm_absent]
    below = sub["passes_min_cells"].eq(False)
    print(f"\nsub-threshold donor-arms kept and flagged (passes_min_cells = False): "
          f"{int(below.sum()):,} rows, "
          f"{sub.loc[below, ['arm', 'donor_key']].drop_duplicates().shape[0]} donor-arms")
    print(sub[below].drop_duplicates(["arm", "donor_key"])["arm"].value_counts().to_string())

    dm = dl[dl["arm"].isin([P.MALIG, P.REACT])]
    print(f"\npct_expr_depthmatched: non-NaN on {int(dm['pct_expr_depthmatched'].notna().sum()):,} "
          f"of {len(dm):,} CD4 rows; NaN by construction on all "
          f"{int(len(dl) - len(dm)):,} non-CD4 rows (depth matching not applicable). "
          "NaN inside the CD4 arms = no cell of that donor-arm reached "
          f"{P.DS_DEPTH:,} UMI.")

    print(f"\npay51_gate2_reach_donor.csv: {len(g2d):,} rows; "
          f"{int(g2d['frac_reachable'].isna().sum()):,} NaN (arm absent in that donor), "
          f"{int((g2d['frac_reachable'] == 0).sum()):,} genuine 0.")


def verify_zero_lfc(long_q, stats):
    hr("VERIFICATION 4 -- CCL19, CCL21, CXCL9: significant depth-matched q, median log2FC = 0")
    genes = ["CCL19", "CCL21", "CXCL9"]
    print(stats[stats["gene"].isin(genes)][
        ["gene", "gene_set", "log2fc_detect_depthmatched", "p_detect_depthmatched",
         "q_detect_depthmatched", "n_donors_paired_detect_depthmatched",
         "log2fc_detect_raw", "q_detect_raw", "log2fc_paired_mean", "q_mean"]].to_string(
        index=False))
    w = (long_q[long_q["arm"].isin([P.MALIG, P.REACT]) & long_q["gene"].isin(genes)]
         .pivot_table(index=["gene", "donor"], columns="arm", values="pct_expr_ds")
         .dropna())
    w["diff_malig_minus_react"] = w[P.MALIG] - w[P.REACT]
    for g in genes:
        d = w.loc[g].sort_values("diff_malig_minus_react")
        nz = d["diff_malig_minus_react"] != 0
        print(f"\n{g}: {len(d)} paired donors, {int((d['diff_malig_minus_react'] > 0).sum())} up, "
              f"{int((d['diff_malig_minus_react'] < 0).sum())} down, "
              f"{int((~nz).sum())} tied at exactly 0; "
              f"median diff = {d['diff_malig_minus_react'].median():.6g}")
        print("  per-donor depth-matched detection (malignant, reactive, difference), "
              "non-tied donors only:")
        print(d[nz].to_string() if nz.any() else "   none")


def verify_xcl(long_q):
    hr("VERIFICATION 5 -- XCL1 and XCL2: medians near 1% while means sit at the reactive level")
    for g in ("XCL1", "XCL2"):
        w = (long_q[long_q["arm"].isin([P.MALIG, P.REACT]) & long_q["gene"].eq(g)]
             .pivot_table(index="donor", columns="arm", values="mean_all").dropna())
        p = (long_q[long_q["arm"].isin([P.MALIG, P.REACT]) & long_q["gene"].eq(g)]
             .pivot_table(index="donor", columns="arm", values="pct_expr").dropna())
        w.columns = [f"mean_{c}" for c in w.columns]
        p.columns = [f"pct_{c}" for c in p.columns]
        t = w.join(p)
        t["d_mean"] = t[f"mean_{P.MALIG}"] - t[f"mean_{P.REACT}"]
        nz_m = int((t[f"mean_{P.MALIG}"] > 0).sum())
        nz_r = int((t[f"mean_{P.REACT}"] > 0).sum())
        print(f"\n{g}: {len(t)} paired donors. "
              f"malignant non-zero in {nz_m}, reactive non-zero in {nz_r}.")
        print(f"  median mean_all  malignant {t[f'mean_{P.MALIG}'].median():.6g}   "
              f"reactive {t[f'mean_{P.REACT}'].median():.6g}")
        print(f"  mean   mean_all  malignant {t[f'mean_{P.MALIG}'].mean():.6g}   "
              f"reactive {t[f'mean_{P.REACT}'].mean():.6g}")
        print(f"  median pct_expr  malignant {t[f'pct_{P.MALIG}'].median():.6g}   "
              f"reactive {t[f'pct_{P.REACT}'].median():.6g}")
        up = int((t["d_mean"] > 0).sum())
        m = t[f"mean_{P.MALIG}"].sort_values(ascending=False)
        top5 = float(m.head(5).sum() / m.sum()) if m.sum() > 0 else float("nan")
        print(f"  donors with malignant > reactive mean_all: {up} of {len(t)}; "
              f"the 5 highest-expressing malignant arms carry {top5:.1%} of all malignant "
              f"{g} signal -- that is what holds the mean up while the median sits near 1%.")
        top = t.reindex(t["d_mean"].abs().sort_values(ascending=False).index)
        share = (top["d_mean"].abs().cumsum() / top["d_mean"].abs().sum())
        k = int((share < 0.80).sum()) + 1
        print(f"  {k} of {len(t)} donors carry 80% of the total |malignant - reactive| in "
              f"mean_all. Those donors:")
        print(top.head(k).to_string())


# --------------------------------------------------------------------- main
def main():
    print(f"pay51 donor-level export -- seed {P.SEED}, depth {P.DS_DEPTH:,} UMI, "
          f"cell floor {P.MIN_CELLS} recorded not applied", flush=True)
    long_all, reach_all = aggregate_everything()
    long_c, reach_c, don, d2s, cov = cohort_and_mask(long_all, reach_all)

    long_q = long_c.dropna(subset=["mean_all"])
    long_q = long_q[long_q["n_cells"] >= P.MIN_CELLS]

    dl = build_donor_long(long_c, d2s)
    g2d = build_gate2_donor(reach_c, long_c, d2s)
    stats = build_gene_stats(long_q)

    f1 = P.save(dl, "donor_long")
    f2 = P.save(g2d, "gate2_reach_donor")
    f3 = P.save(stats, "gene_stats")
    print(f"\nwrote {f1}  {dl.shape}\nwrote {f2}  {g2d.shape}\nwrote {f3}  {stats.shape}")

    verify_published(long_q, reach_c, stats)
    verify_pairing(long_q, cov)
    verify_nan(dl, g2d)
    verify_zero_lfc(long_q, stats)
    verify_xcl(long_q)

    hr("RESULT")
    if FAILURES:
        for f in FAILURES:
            print("FAIL " + f)
        raise SystemExit(f"{len(FAILURES)} verification failure(s) above tolerance {TOL}")
    print(f"all recomputation checks within {TOL}; three tables written.")


if __name__ == "__main__":
    main()
