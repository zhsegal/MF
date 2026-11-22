#!/usr/bin/env python
"""Step 1 + Step 2 of the cluster-level malignancy call, as a job (after run_cluster_malig.py).

Step 1, on the ALICE-labelled samples: label each cluster by the ALICE family share of its TCR+
cells; compare four scores leave-one-study-out (learned gene program, nb26 a-priori markers,
V-gene skew, program + V skew); cutoffs from the held-out probabilities.
Step 2: score every cluster. HC -> benign, ALICE samples keep their ALICE label, clear -> auto,
the rest -> review queue + one review sheet per sample.

Writes (paths in cluster_malig_helpers):
  tables/cluster_malig_cv_v1.csv          held-out comparison, candidate x fold
  tables/cluster_malig_queue_v1.csv       review load per candidate x MAX_ERR   <- your decision
  tables/cluster_malig_program_v1.csv     the learned gene program
  tables/cluster_malig_clusters_v1.csv    every cluster: features, P per candidate, source, call
  tables/cluster_malig_review_v1.csv      review queue (rows you already filled are kept)
  figures/nb26b_review/<sample>.png + _all_review_sheets.pdf
  figures/nb26b_candidates_auto_share.png
  data/atlas_joint/skin_T_malignancy_cluster_v1.parquet   per-cell call, manual calls merged
  data/atlas_joint/skin_T_cluster_malig_model_v1.json     settings, cutoffs, counts

Usage:
  python run_cluster_malig_model.py [--program auto|program|ctcl_apriori|vskew|program+vskew]
                                    [--max-err 0.02] [--no-clone-v] [--in-dir DIR --out-dir DIR]
--out-dir sends every output to one flat directory (dry runs).
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("CLUSTER_MALIG_THREADS", "4"))

NB_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NB_DIR / "helpers"))

import anndata as ad                   # noqa: E402
import matplotlib                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402
import pandas as pd                    # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages   # noqa: E402

import cluster_malig_helpers as CM     # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tissue", choices=sorted(CM.CONFIG), default="skin")
    p.add_argument("--program", default="auto", choices=["auto", *CM.CANDIDATES])
    p.add_argument("--max-err", type=float, default=CM.MAX_ERR)
    p.add_argument("--in-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--no-clone-v", action="store_true",
                   help="skip the clone-V check (auto-benign clusters carrying the sample's clone V)")
    p.add_argument("--skip-sheets", action="store_true")
    args = p.parse_args()
    CM.use_tissue(args.tissue)          # before any CM path is read
    if args.in_dir is None:
        args.in_dir = CM.ATLAS_JOINT
    return args


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    args = parse_args()
    P = (lambda path: path) if args.out_dir is None else (lambda path: args.out_dir / path.name)
    review_dir = CM.REVIEW_DIR if args.out_dir is None else args.out_dir / "review"
    fig_dir = CM.FIG_DIR if args.out_dir is None else args.out_dir
    for d in (review_dir, fig_dir, P(CM.CV_CSV).parent, P(CM.CALLS_PQ).parent):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ load + labels
    pb = ad.read_h5ad(args.in_dir / CM.PSEUDOBULK_H5AD.name)
    cells = pd.read_parquet(args.in_dir / CM.CLUSTERS_PQ.name)
    man = json.loads((args.in_dir / CM.MANIFEST_JSON.name).read_text())
    assert len(cells) == man["n_cells"] and pb.n_obs == man["n_clusters"], "job outputs out of sync"
    log(f"{man['n_samples']} samples / {man['n_clusters']:,} clusters / {man['n_cells']:,} cells "
        f"| skipped {len(man['skipped'])} | failed {len(man['failed'])}")

    cl = pb.obs.copy()
    for c in ["sample_id", *CM.META_COLS]:
        cl[c] = cl[c].astype(str)
    for c in ("stage_class", "tech"):           # skin design columns; blood has neither
        if c not in cl.columns:
            cl[c] = ""
    cl["n_cells"] = cl["n_cells"].astype(int)
    clon = pd.read_csv(CM.CLONALITY_CSV).set_index("sample_id")
    tcr_samples = set(clon.index[clon["tcr_cohort"].astype(bool) & clon["is_clonal"].notna()])
    clonal_samples = set(clon.index[clon["is_clonal"].astype(str).eq("True")])
    in_tcr = cl["sample_id"].isin(tcr_samples)
    is_hc = cl["disease"].eq("HC")
    cl["tcr_label"] = np.where(in_tcr, CM.tcr_cluster_labels(cl["n_tcr"], cl["n_family"],
                                                             CM.MIN_TCR, CM.FAM_HI, CM.FAM_LO),
                               "none")
    cl.loc[is_hc, "tcr_label"] = "benign"
    cl["role"] = np.select(
        [cl["entity"].isin(CM.DROP_ENTITIES), is_hc,
         in_tcr & cl["sample_id"].isin(clonal_samples), in_tcr],
        ["excluded", "train_hc", "train", "tcr_nonclonal"], "apply")
    roles = cl.groupby("role").agg(samples=("sample_id", "nunique"), clusters=("n_cells", "size"),
                                   cells=("n_cells", "sum"))
    log("roles:\n" + roles.to_string())
    t = cl[cl["role"].eq("train")]
    mixed = float(t.loc[t["tcr_label"].eq("mixed"), "n_tcr"].sum() / t["n_tcr"].sum())
    ceiling = float(np.maximum(t["n_family"], t["n_tcr"] - t["n_family"]).sum() / t["n_tcr"].sum())
    log(f"train: TCR+ cells in mixed clusters {mixed:.1%} | cluster-call purity ceiling {ceiling:.1%}")

    # ------------------------------------------------------------ fold-independent features
    genes_all = pb.var_names.to_numpy()
    keep = np.asarray(pb.layers["n_detected"].sum(0)).ravel() / cl["n_cells"].sum() >= 1e-3
    LC, genes = CM.logcpm(pb.X, keep), genes_all[keep]
    feats = pd.DataFrame(index=cl.index)
    feats["ctcl"] = CM.program_score(LC, genes, CM.CTCL_UP, CM.CTCL_DOWN)
    vb, va = CM.vskew(cells, "top_trbv"), CM.vskew(cells, "top_trav")
    feats["v_z"] = np.fmax(vb["v_z"].reindex(cl.index), va["v_z"].reindex(cl.index))
    cl["top_v"] = vb["top_v"].reindex(cl.index)
    cl["top_v_share"] = vb["top_v_share"].reindex(cl.index)
    # mal_knn is a skin-only comparison arm (retired nb26). Where there is none, the column is
    # carried as NaN so every figure and table below keeps its shape.
    if CM.KNN_PQ is not None and Path(CM.KNN_PQ).exists():
        knn = pd.read_parquet(CM.KNN_PQ, columns=["call_mrvi10k"])
        kc = cells[["cell_id", "cluster"]].join(knn["call_mrvi10k"].astype(float), on="cell_id")
        feats["knn_frac"] = kc.groupby("cluster")["call_mrvi10k"].mean().reindex(cl.index)
    else:
        log("no mal_knn parquet for this tissue -- knn_frac left NaN (comparison arm only)")
        feats["knn_frac"] = np.nan
    vcov = feats["v_z"].notna().groupby(cl["study"]).mean().round(2)
    log(f"logCPM {LC.shape} | clusters with V skew by study: {vcov.to_dict()}")

    # ------------------------------------------------------------ Step 1: leave one study out
    y = cl["tcr_label"].eq("malignant").to_numpy()
    w = cl["n_cells"].to_numpy(float)
    TRAIN = (cl["role"].isin(["train", "train_hc"])
             & cl["tcr_label"].isin(["malignant", "benign"])).to_numpy()
    w_fit = 1.0 / cl.groupby("donor")["n_cells"].transform("size").to_numpy()
    STUDY, SAMPLE = cl["study"].to_numpy(), cl["sample_id"].to_numpy()
    FOLDS = sorted(set(STUDY[TRAIN]))
    oof = pd.DataFrame(np.nan, index=cl.index, columns=[*CM.CANDIDATES, "knn_frac"])
    sel_count = {}
    for held in FOLDS:
        te, tr = TRAIN & (STUDY == held), TRAIN & (STUDY != held)
        tab = CM.consensus_signature(pb.X, genes_all, cl, tr, CM.N_UP, CM.N_DOWN)
        up = tab.loc[tab["direction"].eq("up"), "gene"]
        dn = tab.loc[tab["direction"].eq("down"), "gene"]
        for g in [*up, *dn]:
            sel_count[g] = sel_count.get(g, 0) + 1
        f = feats.assign(program=CM.program_score(LC, genes, up, dn))
        for cand, cols in CM.CANDIDATES.items():     # V skew alone stays NaN where no V reads (li2024)
            oof[cand] = np.where(te, CM.fit_predict(f, cols, tr, te, y, w_fit), oof[cand])
        log(f"fold {held:<16} train donors {tab.attrs['n_donors']:>3} | program {len(up)} up / "
            f"{len(dn)} down | held-out clusters {int(te.sum())} ({int(y[te].sum())} malignant)")
    oof["knn_frac"] = feats["knn_frac"]

    # ------------------------------------------------------------ final program + per-candidate P
    tab = CM.consensus_signature(pb.X, genes_all, cl, TRAIN, CM.N_UP, CM.N_DOWN)
    UP = tab[tab["direction"].eq("up")].sort_values("lfc_median", ascending=False)
    DN = tab[tab["direction"].eq("down")].sort_values("lfc_median")
    prog = pd.concat([UP, DN])
    prog["n_folds_selected"] = prog["gene"].map(sel_count).fillna(0).astype(int)
    prog.round(3).to_csv(P(CM.PROGRAM_CSV), index=False)
    log(f"program: {len(UP)} up / {len(DN)} down from {tab.attrs['n_donors']} donors\n"
        f"  up:   {', '.join(UP['gene'].head(30))}\n  down: {', '.join(DN['gene'].head(30))}")
    feats["program"] = CM.program_score(LC, genes, UP["gene"], DN["gene"])
    prob = pd.DataFrame(np.nan, index=cl.index, columns=list(CM.CANDIDATES))
    everything = np.ones(len(cl), bool)
    for cand, cols in CM.CANDIDATES.items():
        prob[cand] = CM.fit_predict(feats, cols, TRAIN, everything, y, w_fit)
        prob.loc[TRAIN, cand] = oof.loc[TRAIN, cand]      # labelled clusters: held-out P only

    # ------------------------------------------------------------ evaluation + review load
    has_tcr_label = (cl["sample_id"].isin(tcr_samples)
                     & cl["tcr_label"].isin(["malignant", "benign"])).to_numpy()
    fixed = cl["role"].eq("excluded").to_numpy() | is_hc.to_numpy() | has_tcr_label
    rows, queue = [], []
    for cand in oof.columns:
        p = oof[cand].to_numpy()
        for me in sorted({*CM.MAX_ERR_GRID, args.max_err}):
            lo, hi = CM.pick_cutoffs(p[TRAIN], y[TRAIN], w[TRAIN], me)
            ev = CM.evaluate(p[TRAIN], y[TRAIN], w[TRAIN], SAMPLE[TRAIN], lo, hi)
            if me == args.max_err:
                rows.append({"candidate": cand, "fold": "pooled", **ev})
                for held in FOLDS:
                    mk = TRAIN & (STUDY == held)
                    rows.append({"candidate": cand, "fold": held,
                                 **CM.evaluate(p[mk], y[mk], w[mk], SAMPLE[mk], lo, hi)})
            if cand in CM.CANDIDATES:
                pr = prob[cand].to_numpy()
                rev = ~fixed & ~(pr >= hi) & ~(pr <= lo)          # NaN P is never clear
                queue.append({"candidate": cand, "max_err": me, "p_lo": lo, "p_hi": hi,
                              "heldout_auto_cells_frac": ev["auto_cells_frac"],
                              "heldout_err_mal": ev["err_mal"], "heldout_err_ben": ev["err_ben"],
                              "review_clusters": int(rev.sum()),
                              "review_cells": int(w[rev].sum()),
                              "review_samples": int(pd.Series(SAMPLE[rev]).nunique()),
                              "auto_clusters": int((~fixed & ~rev).sum())})
    cv = pd.DataFrame(rows)
    cv.to_csv(P(CM.CV_CSV), index=False)
    queue = pd.DataFrame(queue)
    queue.to_csv(P(CM.QUEUE_CSV), index=False)
    pooled = cv[cv["fold"].eq("pooled")].set_index("candidate")
    best = (args.program if args.program != "auto"
            else pooled.loc[list(CM.CANDIDATES), "auto_cells_frac"].idxmax())
    P_LO, P_HI = (float(x) for x in pooled.loc[best, ["p_lo", "p_hi"]])
    log(f"held-out (MAX_ERR {args.max_err:.0%}):\n"
        + pooled.drop(columns="n_within").round(3).to_string()
        + f"\n-> {best}: p_lo {P_LO:.3f} p_hi {P_HI:.3f}\nreview load:\n"
        + queue.round(3).to_string(index=False))

    # ------------------------------------------------------------ Step 2: calls
    cl["prob"] = prob[best]
    cl["program"], cl["v_z"], cl["ctcl"] = feats["program"], feats["v_z"], feats["ctcl"]
    for cand in CM.CANDIDATES:
        cl[f"p_{cand}"] = prob[cand]
    clear_m, clear_b = cl["prob"].ge(P_HI), cl["prob"].le(P_LO)
    cl["source"] = np.select([cl["role"].eq("excluded"), is_hc, has_tcr_label, clear_m | clear_b],
                             ["unassessed", "hc", "tcr", "auto"], "review")
    cl["call"] = pd.Series(pd.NA, index=cl.index, dtype="boolean")
    cl.loc[cl["source"].eq("hc"), "call"] = False
    cl.loc[cl["source"].eq("tcr"), "call"] = cl["tcr_label"].eq("malignant")
    cl.loc[cl["source"].eq("auto"), "call"] = clear_m

    # ---- clone-V check: an auto-benign cluster carrying the sample's clone V goes to review ----
    cl["clone_v"], cl["shares_clone_v"] = CM.clone_v(cl)
    # validate where it matters: ALICE clusters the held-out score would have auto-called benign
    ab = pd.Series(TRAIN, index=cl.index) & cl["prob"].le(P_LO) & cl["clone_v"].notna()
    sh, mal_ = cl.loc[ab, "shares_clone_v"], cl.loc[ab, "tcr_label"].eq("malignant")
    cv_prec = float(mal_[sh].mean()) if sh.any() else float("nan")
    cv_ben = float(sh[~mal_].mean()) if (~mal_).any() else float("nan")
    log(f"clone-V check, held-out auto-benign ALICE clusters: {int(ab.sum())} ({int(mal_.sum())} "
        f"ALICE-malignant) | flagged {int(sh.sum())}, of them malignant {int((sh & mal_).sum())}")
    would = cl["source"].eq("auto") & cl["call"].eq(False).fillna(False) & cl["shares_clone_v"]
    flip = would & (not args.no_clone_v)
    cl.loc[flip, "source"] = "review"
    cl.loc[flip, "call"] = pd.NA
    cl["reason"] = np.where(cl["source"].eq("review"),
                            np.where(flip, "clone_v", "uncertain"), "")
    log(f"clone-V check {'OFF' if args.no_clone_v else 'on'}: {int(would.sum())} auto-benign clusters "
        f"({int(cl.loc[would, 'n_cells'].sum()):,} cells) carry the clone V, {int(flip.sum())} sent to review")
    cl.to_csv(P(CM.CLUSTERS_CSV))
    src = cl.groupby("source")["n_cells"].agg(clusters="size", cells="sum")
    hc = cl[is_hc & cl["prob"].notna()]
    hc_fp = float(hc.loc[hc["prob"].ge(P_HI), "n_cells"].sum() / max(hc["n_cells"].sum(), 1))
    tt = cl[cl["source"].eq("tcr")]
    dis = tt[(tt["tcr_label"].eq("benign") & tt["prob"].ge(P_HI))
             | (tt["tcr_label"].eq("malignant") & tt["prob"].le(P_LO))]
    log("sources:\n" + src.to_string()
        + f"\nHC cells the score alone would call malignant: {hc_fp:.1%}"
        + f"\nALICE clusters where the score is confidently the other way: {len(dis)}"
          f" ({dis['n_cells'].sum() / max(tt['n_cells'].sum(), 1):.1%} of cells; ALICE kept)")

    # ------------------------------------------------------------ review queue (keep filled rows)
    REV_COLS = ["sample_id", "cluster_idx", "reason", "n_cells", "prob", "program", "top_v",
                "top_v_share", "clone_v",
                "frac_CD4", "frac_CD4_Treg", "frac_CD8", "n_tcr", "n_family", "tcr_label",
                "study", "disease", "stage_class"]
    rev = (cl.loc[cl["source"].eq("review"), REV_COLS].round(3).assign(user_call="", note="")
           .sort_values(["sample_id", "cluster_idx"]))
    review_csv = P(CM.REVIEW_CSV)
    if review_csv.exists():
        old = pd.read_csv(review_csv, index_col="cluster", dtype=str, keep_default_na=False)
        filled = old[old["user_call"].str.strip().ne("")]
        k = filled.index.intersection(rev.index)
        rev.loc[k, ["user_call", "note"]] = filled.loc[k, ["user_call", "note"]]
        gone = filled.index.difference(rev.index)
        if len(gone):
            log(f"{len(gone)} filled review rows no longer in the queue (kept, still applied)")
            rev = pd.concat([rev, old.loc[gone]])
    rev.index.name = "cluster"
    rev.to_csv(review_csv)
    open_rows = rev[rev["user_call"].astype(str).str.strip().eq("")]
    open_samples = list(dict.fromkeys(open_rows["sample_id"]))
    log(f"review queue: {len(open_rows)} open clusters in {len(open_samples)} samples")

    # ------------------------------------------------------------ figures
    order = [*CM.CANDIDATES, "knn_frac"]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(range(len(order)), pooled.loc[order, "auto_cells_frac"],
           color=["#c0392b" if c == best else ("#bbb" if c == "knn_frac" else "#7f8c8d")
                  for c in order])
    for i, c in enumerate(order):
        ax.text(i, pooled.loc[c, "auto_cells_frac"] + 0.01, f"AUC {pooled.loc[c, 'auc_cells']:.2f}",
                ha="center", fontsize=7)
    ax.set_xticks(range(len(order)), [c.replace("_", "\n") for c in order], fontsize=7)
    ax.set_ylabel(f"labelled cells auto-called\n(held-out, <= {args.max_err:.0%} wrong)")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"{best} auto-calls {pooled.loc[best, 'auto_cells_frac']:.0%} of ALICE-labelled "
                 f"cells (knn_frac not held out)", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{CM.REVIEW_DIR.name.replace('_review', '')}_candidates_auto_share.png",
                dpi=150)
    plt.close(fig)

    if not args.skip_sheets:
        gset = set(genes)
        d_up = [g for g in UP["gene"].head(12) if g in gset]
        d_dn = [g for g in DN["gene"].head(12) if g in gset]
        d_mk = [g for g in CM.MARKERS if g in gset and g not in d_up + d_dn]
        dot = d_up + d_dn + d_mk
        breaks = (len(d_up), len(d_up) + len(d_dn))
        expr = pd.DataFrame(LC[:, pd.Index(genes).get_indexer(dot)], index=cl.index, columns=dot)
        pct = pd.DataFrame(pb.layers["n_detected"][:, pd.Index(genes_all).get_indexer(dot)].toarray()
                           / cl["n_cells"].to_numpy()[:, None], index=cl.index, columns=dot)
        status = np.where(cl["source"].eq("review"), "review",
                          np.where(cl["call"].fillna(False), "malignant", "benign"))
        sheet = cl.assign(status=status, alice_ok=cl["sample_id"].isin(tcr_samples))
        for old_png in review_dir.glob("*.png"):
            old_png.unlink()
        by_sample = cells.groupby("sample_id")
        with PdfPages(review_dir / "_all_review_sheets.pdf") as pdf:
            for sid in open_samples:
                f = CM.review_figure(sid, by_sample.get_group(sid), sheet[sheet["sample_id"].eq(sid)],
                                     expr, pct, P_LO, P_HI, breaks=breaks)
                f.savefig(review_dir / (re.sub(r"[^A-Za-z0-9_.-]", "_", sid) + ".png"), dpi=110)
                pdf.savefig(f)
                plt.close(f)
        log(f"wrote {len(open_samples)} review sheets -> {review_dir}")
        audit = [s_ for s_ in cl.loc[cl["source"].eq("auto"), "sample_id"].unique()
                 if s_ not in set(open_samples)]
        audit = sorted(audit, key=lambda s_: (cl.loc[cl["sample_id"].eq(s_), "study"].iat[0], s_))
        with PdfPages(review_dir / "_audit_auto_samples.pdf") as pdf:
            for sid in audit:
                f = CM.review_figure(sid, by_sample.get_group(sid), sheet[sheet["sample_id"].eq(sid)],
                                     expr, pct, P_LO, P_HI, breaks=breaks)
                pdf.savefig(f)
                plt.close(f)
        log(f"wrote audit PDF of {len(audit)} fully auto-called samples")

    # ------------------------------------------------------------ per-cell call (manual calls merged)
    out, final = CM.finalize_calls(cl[["sample_id", "study", "n_cells", "source", "call", "prob"]],
                                   cells, review_csv, P(CM.EXCLUDE_CSV))
    out.to_parquet(P(CM.CALLS_PQ), index=False)
    log(f"wrote {P(CM.CALLS_PQ).name}: {len(out):,} cells, mal_cluster defined on "
        f"{out['mal_cluster'].notna().mean():.1%}")

    summary = {"written": time.strftime("%Y-%m-%d %H:%M"), "program": best,
               "program_arg": args.program, "max_err": args.max_err, "p_lo": P_LO, "p_hi": P_HI,
               "folds": FOLDS, "n_train_donors_program": int(tab.attrs["n_donors"]),
               "mixed_tcr_frac": mixed, "purity_ceiling": ceiling, "hc_fp_cells": hc_fp,
               "tcr_disagree_clusters": int(len(dis)),
               "clone_v_precision_alice": cv_prec, "clone_v_alice_benign_flagged": cv_ben,
               "clone_v_flipped_to_review": int(flip.sum()), "clone_v_check": not args.no_clone_v,
               "roles": roles.reset_index().to_dict("records"),
               "sources": final.groupby("source")["n_cells"].agg(["size", "sum"])
                               .reset_index().to_dict("records"),
               "review_open_clusters": int(len(open_rows)),
               "review_open_samples": len(open_samples)}
    P(CM.MODEL_JSON).write_text(json.dumps(summary, indent=1, default=str))
    log("done")


if __name__ == "__main__":
    main()
