"""Is this antigen real, or is it contamination that scales with arm composition?

nb43's first blood ranking put TPO, MBP, ADGRE1, PTH2R, NRROS, EVC2 and FGFR1 above CCR4. TPO
and MBP are not surface proteins; ADGRE1 is eosinophil-restricted in humans; NRROS is myeloid.
And `frac_donors_up` was inverted against plausibility -- MBP 0.962, EVC2 0.936, CCR4 0.744.

That ordering is the signature of a technical effect, not of biology. Blood in SS is ~92%
malignant CD4, so malignant x monocyte doublets outnumber reactive x monocyte doublets by about
the arm-abundance ratio. Every minor population in the sample preferentially contaminates the
malignant arm, in every donor, which makes the artifact *more* reproducible across donors than
real inter-patient biology. Ambient RNA works the other way (it is shared by both arms) and
cannot produce this.

Six tests, each with its own failure mode, and an empirical null rather than chosen thresholds:

  fails_trim           the effect collapses when the top decile of cells is trimmed, so it is
                       carried by a handful of cells rather than by the population
  fails_concentration  most of the arm's signal for this gene sits in <1% of its cells
  fails_lineage        per-cell expression tracks a contaminant module score (monocyte in
                       blood; keratinocyte, fibroblast or myeloid in skin) inside the tumour arm
  fails_umi            it tracks per-cell library size, or is confined to the top-UMI decile
  fails_clean          it does not survive dropping lineage-positive and high-doublet cells
  fails_burden         its per-donor effect scales with the malignant:reactive abundance ratio

The last one is the mechanistic test: contamination magnitude is proportional to that ratio by
construction, and a real antigen's selectivity is not.

The null is `contamination_band`: the same statistics computed on genes that are *certainly*
not CD4 T-cell antigens -- LYZ, S100A8, CD14, VCAN, PPBP, HBB, MS4A1. Whatever they score is
what pure contamination scores in this dataset, so every threshold is read off them.

    band = TQ.contamination_band(sel, trim, clean, qc_cell, burden_r)
    ev   = TQ.artifact_evidence(sel, trim, clean, qc_cell, burden_r, band)
    ev["artifact_flag"].value_counts()
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------- the tissue's contaminants
# Lineage markers that a CD4 T cell does not express. Confirmed present in the var of the object
# each compartment screens. `strict` members are the ones used to call a cell contaminated: a
# single count is already evidence, because these are high-expressors in their own lineage.
#
# The two compartments are contaminated by different things, which is the whole reason this is
# parametrised. Blood: monocytes, platelets, erythrocytes, B cells. Skin: keratinocytes above
# all -- epidermal dissociation shears them and keratin is the classic skin ambient transcript --
# then fibroblasts, endothelium, melanocytes, mast cells and myeloid.
LINEAGE_MODULES_BY_TISSUE = {
    "blood": {
        "myeloid": dict(
            genes=["LYZ", "S100A8", "S100A9", "CD14", "VCAN", "FCN1", "CST3", "LST1", "AIF1",
                   "TYROBP", "FCER1G", "CD68", "MNDA", "HLA-DRA"],
            strict=["LYZ", "S100A8", "S100A9", "CD14", "VCAN", "FCN1", "CD68", "MNDA"]),
        "platelet": dict(genes=["PPBP", "PF4"], strict=["PPBP", "PF4"]),
        "rbc": dict(genes=["HBB"], strict=["HBB"]),
        "bcell": dict(genes=["MS4A1", "CD79A", "CD79B"], strict=["MS4A1", "CD79A"]),
    },
    "skin": {
        "keratinocyte": dict(
            genes=["KRT14", "KRT5", "KRT1", "KRT10", "KRT17", "KRT6A", "SFN", "KRTDAP", "DMKN",
                   "S100A2"],
            strict=["KRT14", "KRT5", "KRT1", "KRT10", "SFN", "KRTDAP", "DMKN"]),
        "fibroblast": dict(
            genes=["COL1A1", "COL1A2", "COL3A1", "COL6A2", "DCN", "LUM", "PDGFRB", "SFRP2"],
            strict=["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM"]),
        # PECAM1 is CD31 and sits on naive T cells, so it scores the module but never drops a
        # cell and never enters the null.
        "endothelial": dict(genes=["PECAM1", "VWF", "CDH5", "CLDN5", "EGFL7"],
                            strict=["VWF", "CDH5", "CLDN5"]),
        "melanocyte": dict(genes=["PMEL", "MLANA", "TYRP1", "DCT"],
                           strict=["PMEL", "MLANA", "TYRP1", "DCT"]),
        # FCER1G, TYROBP and ITGAX are carried by NK and by some activated T cells: scored, not
        # strict.
        "myeloid": dict(
            genes=["LYZ", "CD14", "CD68", "AIF1", "C1QA", "C1QB", "CD163", "FCER1G", "TYROBP",
                   "ITGAX", "CST3"],
            strict=["LYZ", "CD14", "CD68", "C1QA", "C1QB", "CD163"]),
        "mast": dict(genes=["TPSAB1", "TPSB2", "CPA3", "MS4A2"],
                     strict=["TPSAB1", "TPSB2", "CPA3"]),
        "bcell": dict(genes=["MS4A1", "CD79A", "CD79B", "JCHAIN", "MZB1"],
                      strict=["MS4A1", "CD79A", "JCHAIN", "MZB1"]),
        "rbc": dict(genes=["HBB", "HBA2"], strict=["HBB", "HBA2"]),
    },
}

# NK markers are a separate case: a CD4 T cell can be NKG7-low, and the CD8/NK arm of this atlas
# is a known mixture, so they are scored but never used to drop a cell.
NK_MARKERS_BY_TISSUE = {"blood": ["NKG7", "GNLY", "KLRD1"], "skin": ["NKG7", "GNLY", "KLRD1"]}

#: genes used as the contamination null: what pure contamination scores in this cohort. Listed
#: rather than derived, because the band decides every threshold and a gene silently entering or
#: leaving it moves them. Excluded on purpose: `HLA-DRA` and `PECAM1` (activated and naive T
#: cells carry them), the NK markers, and in skin `S100A2` and `ITGAX`, whose families are not
#: clean of T cells.
BAND_GENES_BY_TISSUE = {
    "blood": ["LYZ", "S100A8", "S100A9", "CD14", "VCAN", "FCN1", "CST3", "LST1", "AIF1",
              "TYROBP", "FCER1G", "CD68", "MNDA", "PPBP", "PF4", "HBB", "MS4A1", "CD79A"],
    "skin": ["KRT14", "KRT5", "KRT1", "KRT10", "KRT17", "KRT6A", "SFN", "KRTDAP", "DMKN",
             "COL1A1", "COL1A2", "COL3A1", "COL6A2", "DCN", "LUM", "SFRP2",
             "VWF", "CDH5", "CLDN5", "EGFL7",
             "PMEL", "MLANA", "TYRP1", "DCT",
             "LYZ", "CD14", "CD68", "C1QA", "C1QB", "CD163",
             "TPSAB1", "TPSB2", "CPA3", "MS4A2",
             "MS4A1", "CD79A", "JCHAIN", "MZB1", "HBB", "HBA2"],
}

#: the module scores each compartment regresses per-cell expression on, inside the malignant arm.
#: One per abundant contaminant: in blood the monocyte is the only one that matters, in skin the
#: keratinocyte, the fibroblast and the myeloid cell all do.
COUPLING_MODULES = {"blood": ["myeloid"], "skin": ["keratinocyte", "fibroblast", "myeloid"]}

#: the genes that prompted all of this, per compartment. Carried so a notebook can print the
#: verdict on them by name without retyping the list.
#:  blood -- the first ranking put these above CCR4.
#:  skin  -- the v1 (pre-QC) ranking's implausible passers: a fibroblast collagen, a secreted
#:           lectin, a lysosomal protein, and four cytoplasmic or ER proteins that cannot be
#:           antibody targets whatever their transcript does.
SUSPECTS_BY_TISSUE = {
    "blood": ["TPO", "MBP", "ADGRE1", "PTH2R", "NRROS", "EVC2", "FGFR1"],
    "skin": ["COL6A3", "LGALS3", "GRN", "QSOX1", "CALU", "VDAC1", "GPI", "SYPL1"],
}

TISSUE = "blood"
LINEAGE_MODULES = LINEAGE_MODULES_BY_TISSUE[TISSUE]
NK_MARKERS = NK_MARKERS_BY_TISSUE[TISSUE]
BAND_GENES = BAND_GENES_BY_TISSUE[TISSUE]
SUSPECTS = SUSPECTS_BY_TISSUE[TISSUE]


def use_tissue(tissue="blood"):
    """Point the module at one compartment's contaminants. Call it before anything else.

    Same contract as `cluster_malig_helpers.use_tissue` and `malignancy.use_tissue`: it rebinds
    module-level names, so a notebook that imported the module (not the names) sees the change.
    """
    global TISSUE, LINEAGE_MODULES, NK_MARKERS, BAND_GENES, SUSPECTS
    if tissue not in LINEAGE_MODULES_BY_TISSUE:
        raise ValueError(f"unknown tissue {tissue!r}")
    TISSUE = tissue
    LINEAGE_MODULES = LINEAGE_MODULES_BY_TISSUE[tissue]
    NK_MARKERS = NK_MARKERS_BY_TISSUE[tissue]
    BAND_GENES = BAND_GENES_BY_TISSUE[tissue]
    SUSPECTS = SUSPECTS_BY_TISSUE[tissue]
    return TISSUE


def coupling_cols(tissue=None) -> list[str]:
    """The per-cell coupling columns `run_target_screen_qc.py` writes for this compartment."""
    return [f"r_{m}" for m in COUPLING_MODULES[tissue or TISSUE]]


FAIL_COLS = ["fails_trim", "fails_concentration", "fails_lineage", "fails_umi",
             "fails_clean", "fails_burden"]


def lineage_genes(include_nk=False) -> list[str]:
    g = [x for m in LINEAGE_MODULES.values() for x in m["genes"]]
    if include_nk:
        g += NK_MARKERS
    return sorted(set(g))


def strict_genes() -> list[str]:
    return sorted({x for m in LINEAGE_MODULES.values() for x in m["strict"]})


# ------------------------------------------------------------------ per-gene inputs
def burden_scaling(long, burden, arm_pos="CD4_malignant", arm_neg="CD4_reactive",
                   value="mean_all", pseudo=0.05, min_donors=5) -> pd.DataFrame:
    """Per gene: does the within-donor effect grow with the donor's malignant:reactive ratio?

    `burden` is `target_cohort.burden(long)`. A contaminating transcript enters the malignant arm
    in proportion to how much it outnumbers the reactive arm, so its per-donor log2FC rises with
    `log_ratio`. A real antigen's selectivity has no reason to.
    """
    w = (long[long["arm"].isin([arm_pos, arm_neg])]
         .pivot_table(index=["gene", "donor"], columns="arm", values=value))
    w = w.dropna(subset=[c for c in (arm_pos, arm_neg) if c in w.columns])
    if arm_pos not in w.columns or arm_neg not in w.columns:
        return pd.DataFrame(columns=["gene", "rho_burden", "p_burden", "n_donors_burden"])
    w["lfc"] = np.log2((w[arm_pos] + pseudo) / (w[arm_neg] + pseudo))
    lr = burden["log_ratio"]

    rows = []
    for gene, d in w.groupby(level="gene"):
        don = d.index.get_level_values("donor")
        x = lr.reindex(don).to_numpy()
        y = d["lfc"].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < min_donors or np.ptp(y[ok]) == 0:
            continue
        r = stats.spearmanr(x[ok], y[ok])
        rows.append({"gene": gene, "rho_burden": float(r.statistic),
                     "p_burden": float(r.pvalue), "n_donors_burden": int(ok.sum())})
    return pd.DataFrame(rows)


def concentration(long, arm="CD4_malignant") -> pd.DataFrame:
    """Median over the arm's donors of `top1_share` / `top5_share` -- how few cells carry it."""
    a = long[long["arm"].eq(arm)]
    have = [c for c in ("top1_share", "top5_share", "pct_expr_ge2", "pct_expr") if c in a.columns]
    if not have:
        return pd.DataFrame(columns=["gene"] + have)
    out = a.groupby("gene")[have].median().reset_index()
    return out.rename(columns={"pct_expr": "median_pct_expr"})


def prevalence_residual(ev, cols=None, ref="median_pct_expr", bins=20,
                        log_cols=("or_topdecile",)):
    """Each statistic minus the cohort median of genes with the same detection rate.

    Every one of these quantities is confounded by how often the gene is detected at all. A gene
    seen in two cells necessarily has `top1_share` near 1; a gene seen in 80% of cells
    necessarily has a large top-UMI-decile odds ratio, because detection saturates with depth.
    Comparing a candidate against the raw contamination band would therefore flag CD7 and
    exonerate a gene detected once. The residual compares like with like: what is unusual about
    this gene *given how often it is seen*.
    """
    cols = ("top1_share", "r_umi", "or_topdecile", *coupling_cols()) if cols is None else cols
    out = ev.copy()
    if ref not in out.columns:
        return out
    x = np.log10(out[ref].clip(lower=1e-4))
    grp = pd.cut(x, bins=bins, duplicates="drop")
    for c in cols:
        if c not in out.columns:
            continue
        v = np.log(out[c].clip(lower=1e-6)) if c in log_cols else out[c]
        out[f"{c}_excess"] = (v - v.groupby(grp, observed=True).transform("median")).to_numpy()
    return out


def cluster_confinement(cluster_long, arm="CD4_malignant", value="mean_all",
                        min_clusters=3) -> pd.DataFrame:
    """Share of an arm's signal held by its single strongest cluster.

    Doublets cluster together, and the blood malignancy call is made *per cluster* (`mal_cluster`),
    so a gene whose whole malignant-arm signal lives in one or two Leiden clusters is a candidate
    for a cluster-level call artifact rather than a property of the tumour.
    """
    a = cluster_long[cluster_long["arm"].eq(arm)]
    if a.empty or value not in a.columns:
        return pd.DataFrame(columns=["gene", "top_cluster_share", "n_clusters"])
    m = a.groupby(["gene", "cluster"])[value].mean()
    rows = []
    for gene, s in m.groupby(level="gene"):
        v = np.sort(s.to_numpy())[::-1]
        tot = v.sum()
        if len(v) < min_clusters or tot <= 0:
            continue
        rows.append({"gene": gene, "top_cluster_share": float(v[0] / tot),
                     "top2_cluster_share": float(v[:2].sum() / tot), "n_clusters": int(len(v))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ the null and the verdict
def _ratio(num, den, floor=1e-3):
    """Effect retained, as a fraction. Negative or absent numerator counts as full collapse."""
    den = np.where(np.abs(den) < floor, np.nan, den)
    return np.clip(np.asarray(num, float) / den, -1, 2)


def assemble(sel, sel_trim=None, sel_clean=None, qc_cell=None, burden_r=None,
             conc=None, confine=None) -> pd.DataFrame:
    """One row per gene with every quantity the six tests read, whatever is available."""
    out = sel[["gene", "log2fc_paired", "frac_donors_up", "n_donors_paired", "q"]].copy()
    if sel_trim is not None:
        out = out.merge(sel_trim[["gene", "log2fc_paired"]].rename(
            columns={"log2fc_paired": "log2fc_trim"}), on="gene", how="left")
        out["retained_trim"] = _ratio(out["log2fc_trim"], out["log2fc_paired"])
    if sel_clean is not None:
        out = out.merge(sel_clean[["gene", "log2fc_paired"]].rename(
            columns={"log2fc_paired": "log2fc_clean"}), on="gene", how="left")
        out["retained_clean"] = _ratio(out["log2fc_clean"], out["log2fc_paired"])
    for extra in (qc_cell, burden_r, conc, confine):
        if extra is not None and len(extra):
            out = out.merge(extra, on="gene", how="left")
    return out


def contamination_band(ev: pd.DataFrame, genes=None, quantiles=(0.5, 0.9, 0.975)) -> pd.DataFrame:
    """What pure contamination scores in this dataset, per statistic.

    Returns a frame of quantiles over the lineage-control genes. If these genes do *not* come out
    positive and reproducible, the differential-contamination model is wrong for this cohort and
    nothing below should be used -- `band_is_calibrated` says so explicitly.
    """
    genes = BAND_GENES if genes is None else list(genes)
    sub = ev[ev["gene"].isin(genes)]
    want = ["log2fc_paired", "frac_donors_up", "retained_trim", "retained_clean",
            "top1_share", "top1_share_excess", "top5_share"]
    for c in coupling_cols():
        want += [c, f"{c}_excess"]
    want += ["r_umi", "r_umi_excess", "r_dbl", "or_topdecile", "or_topdecile_excess",
             "rho_burden", "top_cluster_share"]
    cols = [c for c in want if c in sub.columns]
    band = sub[cols].quantile(list(quantiles))
    band.index = [f"p{int(q * 1000) / 10:g}" for q in quantiles]
    band.attrs["n_genes"] = int(sub["gene"].nunique())
    band.attrs["genes_found"] = sorted(sub["gene"].unique())
    return band


def band_is_calibrated(ev, band, genes=None, min_median_lfc=0.1) -> tuple[bool, str]:
    """Sanity gate: the contamination controls must behave like contamination."""
    genes = BAND_GENES if genes is None else list(genes)
    sub = ev[ev["gene"].isin(genes)]
    if sub.empty:
        return False, "no lineage-control gene reached the paired test"
    med = float(sub["log2fc_paired"].median())
    up = float(sub["frac_donors_up"].median())
    ok = med > min_median_lfc and up > 0.5
    msg = (f"{len(sub)} lineage controls: median log2FC {med:+.2f}, median frac_donors_up "
           f"{up:.2f} -- " + ("contamination behaves as predicted, the band is usable."
                              if ok else
                              "the controls are NOT up in the malignant arm, so differential "
                              "contamination is not what this cohort shows. Do not exclude any "
                              "gene on this basis; find the real mechanism first."))
    return ok, msg


def artifact_evidence(ev: pd.DataFrame, band: pd.DataFrame, retain_trim=0.5, retain_clean=0.3,
                      rho_burden=0.5, band_q="p90") -> pd.DataFrame:
    """The six booleans and the verdict, added to `ev`.

    Thresholds that have an empirical null (`top1_share`, the lineage couplings, `r_umi`,
    `or_topdecile`)
    are read off the contamination band at `band_q`. The three that do not -- how much of the
    effect must survive trimming or cleaning, and how steeply it may scale with burden -- are
    explicit arguments, printed by the notebook next to the result.
    """
    out = ev.copy()

    def _band(col, default=np.inf):
        if band is None or col not in band.columns or band_q not in band.index:
            return default
        v = band.loc[band_q, col]
        return float(v) if np.isfinite(v) else default

    out["fails_trim"] = (out["retained_trim"] < retain_trim
                         if "retained_trim" in out else False)
    out["fails_clean"] = (out["retained_clean"] < retain_clean
                          if "retained_clean" in out else False)
    # Prefer the prevalence-matched residual wherever it was computed: the raw statistic is a
    # function of how often the gene is detected, and would rank CD7 beside LYZ.
    def _col(name):
        return f"{name}_excess" if f"{name}_excess" in out.columns else name

    c_conc, c_umi, c_or = _col("top1_share"), _col("r_umi"), _col("or_topdecile")
    out["fails_concentration"] = (out[c_conc] >= _band(c_conc)
                                  if c_conc in out else False)
    # One compartment, several contaminants: the gene fails if it tracks ANY of them beyond what
    # the lineage controls do. In blood that list is one module long, so this is the old test.
    lin = pd.Series(False, index=out.index)
    for m in coupling_cols():
        c = _col(m)
        if c in out:
            lin |= out[c] >= _band(c)
    out["fails_lineage"] = lin
    umi = pd.Series(False, index=out.index)
    if c_umi in out:
        umi |= out[c_umi] >= _band(c_umi)
    if c_or in out:
        umi |= out[c_or] >= _band(c_or)
    out["fails_umi"] = umi
    out["fails_burden"] = (out["rho_burden"] >= rho_burden
                           if "rho_burden" in out else False)

    for c in FAIL_COLS:
        out[c] = out[c].fillna(False).astype(bool)
    out["n_fails"] = out[FAIL_COLS].sum(axis=1)
    out["artifact_flag"] = np.select(
        [out["n_fails"] >= 2, out["n_fails"] == 1], ["artifact", "suspect"], "clean")
    out["fail_reasons"] = [", ".join(c[6:] for c in FAIL_COLS if r[c]) or "-"
                           for _, r in out.iterrows()]
    # A gene that is DOWN in the malignant arm cannot be explained by contamination entering it,
    # so the flag only ever applies to the up direction.
    down = out["log2fc_paired"] <= 0
    out.loc[down, ["artifact_flag", "fail_reasons"]] = ["clean", "- (down in malignant)"]
    return out


def verdict_text(out: pd.DataFrame, suspects=None) -> str:
    """The block nb43 prints: counts, the named suspects, and every excluded gene with its reason."""
    suspects = SUSPECTS if suspects is None else list(suspects)
    L = []
    P = L.append
    P(out["artifact_flag"].value_counts().to_string())
    P("")
    P("the genes that prompted this, by name:")
    cols = [c for c in ("gene", "log2fc_paired", "frac_donors_up", "median_pct_expr",
                        "retained_trim", "retained_clean", "top1_share_excess",
                        *(f"{c}_excess" for c in coupling_cols()),
                        "or_topdecile_excess", "rho_burden",
                        "n_fails", "artifact_flag", "fail_reasons") if c in out.columns]
    P(out[out["gene"].isin(suspects)][cols].to_string(index=False))
    P("")
    bad = out[out["artifact_flag"].ne("clean") & (out["log2fc_paired"] > 0)]
    P(f"excluded from the headline ({len(bad)} genes), every one with its reason:")
    P(bad.sort_values("log2fc_paired", ascending=False)[cols].to_string(index=False))
    return "\n".join(L)
