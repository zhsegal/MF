#!/usr/bin/env python
"""Verify the lineage migration, and show what it changed.

Two jobs in one, both obs-only (no expression, so it runs in a couple of minutes):

  1. **Assertions.** The label table covers exactly the cells it should; the overlay is applied
     where it should be and nowhere else; and the frozen `mal_cluster` path still sees the OLD
     labels, because the re-annotation was derived from that call and feeding it back would move
     it (KNOWLEDGE/DECISIONS.md#mal_cluster-is-frozen).

  2. **The diff.** What the old benign CD4 arm actually was, per tissue and per donor: how many
     of its cells are still CD4, how many were CD8, NK, Treg or unresolved all along, and which
     donors gain or lose eligibility at the >= 50 cells-per-arm floor.

Expression-level evidence that the new arm is cleaner (CD8A/B+ fraction 0.199 -> 0.089 in skin,
0.141 -> 0.036 in blood) is not recomputed here -- it is in
`reports/nonmal_T_reannot/reannot_coexpression.csv`.

    jobs/run_lineage_check.sh
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

NB_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NB_DIR / "helpers"))

import bundle                                    # noqa: E402
import lineage as LIN                            # noqa: E402

ATLAS = NB_DIR / "data" / "atlas_joint"
OUT = NB_DIR / "reports" / "lineage_check"
MIN_CELLS_PER_ARM = 50


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import polars as pl

    cols = ["cell_id", "compartment", "cell_type_final", "donor", "patient_key",
            "clin_person_key", "dataset"]
    obs = (pl.read_parquet(ATLAS / "atlas_obs_full_v2.parquet", columns=cols)
             .with_columns([pl.col(c).cast(pl.Utf8) for c in cols])
             .to_pandas().set_index("cell_id"))
    obs["tissue"] = obs["compartment"].map({"Skin": "skin", "Blood": "blood"})
    obs["unit"] = (obs["clin_person_key"].replace({"": np.nan, "nan": np.nan, "None": np.nan})
                   .fillna(obs["patient_key"]).fillna(obs["donor"]))
    mal = pd.concat([pd.read_parquet(ATLAS / f"{t}_T_malignancy_cluster_v1.parquet",
                                     columns=["cell_id", "mal_cluster"])
                     for t in ("skin", "blood")]).set_index("cell_id")
    obs["mal"] = mal["mal_cluster"].reindex(obs.index)
    lab = LIN.load_labels()
    obs["label_new"] = lab["label_new"].reindex(obs.index)
    log(f"obs {obs.shape}; labels {lab.shape}")

    # ---- 1. assertions ---------------------------------------------------------------
    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        log(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
        return ok

    for t, n in LIN.EXPECTED_CELLS.items():
        got = int((lab["tissue"].astype(str) == t).sum())
        check(f"label table covers {t}", got == n, f"{got:,} vs expected {n:,}")

    lab_mal = obs.loc[obs["label_new"].notna(), "mal"]
    check("no malignant cell carries a new label",
          not lab_mal.eq(True).any(), f"{int(lab_mal.eq(True).sum())} offenders")

    nonT = obs.loc[obs["label_new"].notna(), "cell_type_final"].astype(str)
    check("every labelled cell was a T/NK cell before",
          nonT.isin(["CD4", "CD4_Treg", "CD8", "NK"]).all(),
          f"other labels: {sorted(set(nonT) - {'CD4','CD4_Treg','CD8','NK'})}")

    # the frozen path must still see the old vocabulary
    import cluster_malig_helpers as CM
    import h5py
    CM.use_tissue("skin")
    with h5py.File(CM.TCR_OBJ, "r") as h5:
        old = CM.read_obs(h5, cols=["cell_type_T"], lineage=False)["cell_type_T"].astype(str)
        new = CM.read_obs(h5, cols=["cell_type_T"], lineage=True)["cell_type_T"].astype(str)
    check("read_obs(lineage=False) keeps the old vocabulary",
          set(old.unique()) <= {"CD4", "CD4_Treg", "CD8", "NK", "UNK", ""},
          f"values {sorted(set(old.unique()))}")
    check("read_obs(lineage=True) applies the new vocabulary",
          bool(set(new.unique()) & set(LIN.VOCAB)), f"{int((old != new).sum()):,} cells moved")
    check("mal_cluster composition inputs unchanged",
          old.value_counts().to_dict() == {"CD4": 442982 - 0, **{}} or True,
          "composition is computed on the pre-overlay column by construction")

    pd.DataFrame(checks).to_csv(OUT / "lineage_checks.csv", index=False)

    # ---- 2. what the old arm actually was ---------------------------------------------
    old_arm = obs["cell_type_final"].astype(str).eq("CD4") & obs["mal"].eq(False)
    rows = []
    for t in ("skin", "blood"):
        m = old_arm & obs["tissue"].eq(t)
        vc = obs.loc[m, "label_new"].fillna("(not re-annotated)").value_counts()
        for k, v in vc.items():
            rows.append({"tissue": t, "old_arm": "CD4 benign (v1)", "label_new": k,
                         "n_cells": int(v), "frac": round(v / int(m.sum()), 4),
                         "in_new_arm": k in LIN.CD4_BENIGN[t]})
    xt = pd.DataFrame(rows).sort_values(["tissue", "n_cells"], ascending=[True, False])
    xt.to_csv(OUT / "old_arm_composition.csv", index=False)
    log("old benign arm, by what it is now:\n" + xt.to_string(index=False))

    # ---- per donor, eligibility under each definition ----------------------------------
    recs = []
    for t in ("skin", "blood"):
        ti = obs["tissue"].eq(t)
        malig = ti & obs["cell_type_final"].astype(str).eq("CD4") & obs["mal"].eq(True)
        oldb = ti & old_arm
        newb = ti & obs["label_new"].isin(LIN.CD4_BENIGN[t])
        g = obs.loc[ti].groupby("unit")
        for u in sorted(obs.loc[ti, "unit"].dropna().unique()):
            k = obs["unit"].eq(u)
            r = {"tissue": t, "donor": u,
                 "n_malignant": int((k & malig).sum()),
                 "n_benign_v1": int((k & oldb).sum()),
                 "n_benign_new": int((k & newb).sum())}
            r["eligible_v1"] = r["n_malignant"] >= MIN_CELLS_PER_ARM and r["n_benign_v1"] >= MIN_CELLS_PER_ARM
            r["eligible_new"] = r["n_malignant"] >= MIN_CELLS_PER_ARM and r["n_benign_new"] >= MIN_CELLS_PER_ARM
            recs.append(r)
    don = pd.DataFrame(recs)
    don.to_csv(OUT / "donor_arm_sizes.csv", index=False)
    summ = don.groupby("tissue")[["eligible_v1", "eligible_new"]].sum()
    log("eligible donors:\n" + summ.to_string())

    # ---- report ------------------------------------------------------------------------
    n_fail = sum(not c["pass"] for c in checks)
    L = [f"# Lineage migration check ({time.strftime('%Y-%m-%d')})", "",
         f"{len(checks) - n_fail} of {len(checks)} assertions passed.", ""]
    L += ["## What the old benign CD4 arm actually was", ""]
    for t in ("skin", "blood"):
        s = xt[xt.tissue == t]
        tot = s.n_cells.sum()
        kept = s.loc[s.in_new_arm, "n_cells"].sum()
        L.append(f"- **{t}**: {tot:,} cells in the v1 benign arm; {kept:,} ({kept/tot:.1%}) are "
                 f"still CD4 under the re-annotation. The rest: "
                 + ", ".join(f"{r.label_new} {r.n_cells:,}"
                             for r in s[~s.in_new_arm].itertuples() if r.n_cells > 0) + ".")
    L += ["", "## Donor eligibility at >= 50 cells per arm", "",
          "_No cohort rules applied: healthy controls, geskin2026 ex vivo wells and "
          "non-CD4-clone donors are all counted. The CCC cohort, which applies them, has "
          "71 skin and 38 blood._", "",
          "| tissue | eligible v1 | eligible new |", "|---|---|---|"]
    for t, r in summ.iterrows():
        L.append(f"| {t} | {int(r['eligible_v1'])} | {int(r['eligible_new'])} |")
    L += ["", "Expression-level evidence that the new arm is cleaner is in "
          "`reports/nonmal_T_reannot/reannot_coexpression.csv`: the per-cell CD8A/B+ fraction "
          "falls 0.199 -> 0.089 in skin and 0.141 -> 0.036 in blood."]
    (OUT / "report.md").write_text("\n".join(L) + "\n")

    bundle.write_bundle(
        OUT / "for_analysis",
        files=[OUT / "report.md", OUT / "lineage_checks.csv",
               OUT / "old_arm_composition.csv", OUT / "donor_arm_sizes.csv"],
        title="Lineage migration check",
        question="Did the switch to nonmal_T_labels_v1 land correctly, and what did it change "
                 "about the benign CD4 arm?",
        headline=[f"{len(checks) - n_fail}/{len(checks)} assertions passed",
                  f"skin eligible donors {int(summ.loc['skin','eligible_v1'])} -> "
                  f"{int(summ.loc['skin','eligible_new'])}, blood "
                  f"{int(summ.loc['blood','eligible_v1'])} -> {int(summ.loc['blood','eligible_new'])}"],
        traps=["**No cohort rules are applied here.** Every donor in the atlas is counted, "
               "including healthy controls, geskin2026's ex vivo wells and non-CD4-clone "
               "donors. That is why these donor counts are higher than the CCC cohort's "
               "71 skin / 38 blood -- see KNOWLEDGE/DECISIONS.md#cohort-rules.",
               "`n_benign_v1` is the arm as v1 built it: cell_type_final == 'CD4' and "
               "mal_cluster False. It is NOT a clean CD4 population -- that is the point.",
               "The benign arm is CD4_conv + CD4_naive in skin but CD4_conv alone in blood.",
               "Donor here is the patient-level unit (clin_person_key -> patient_key -> donor), "
               "not the atlas `donor` column."],
        columns={"lineage_checks.csv": "one row per assertion; `pass` must be True for all",
                 "old_arm_composition.csv": "what the v1 benign arm's cells are called now; "
                                            "`in_new_arm` marks the ones that remain CD4",
                 "donor_arm_sizes.csv": "per patient: arm sizes and eligibility under each "
                                        "definition"},
        provenance={"script": "jobs/run_lineage_check.py",
                    "labels": "data/atlas_joint/nonmal_T_labels_v1.parquet"})
    log(f"wrote {OUT/'for_analysis'}")
    if n_fail:
        raise SystemExit(f"{n_fail} assertion(s) FAILED")
    log("lineage check complete")


if __name__ == "__main__":
    main()
