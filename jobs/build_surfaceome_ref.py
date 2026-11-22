#!/usr/bin/env python
"""Collapse the four raw surfaceome sources into one table of surface evidence per gene.

    python jobs/build_surfaceome_ref.py        # reads data/external_ref/surfaceome/raw/
                                               # writes  data/external_ref/surfaceome/surfaceome.parquet

Sources and what each one is actually worth:

  SURFY   Bausch-Fluck 2018 PNAS. A machine-learning classifier over 20,193 UniProt entries; the
          `Surfaceome Label` is a PREDICTION except where the source says it came from a training
          set. Genome-wide, so it is the recall arm. Carries topology, TM-domain count, CD number
          and the Almen class, all kept.
  CSPA    Cell Surface Protein Atlas, Bausch-Fluck 2015 PLoS ONE. Mass spectrometry of labelled
          surface glycoproteins on INTACT cells -- 1,492 human entries, the only experimental
          evidence of the four, and the only one that proves a protein was reachable from outside.
          Its own confidence grades are kept verbatim ("1 - high confidence" is not the same claim
          as "3 - unspecific").
  HPA     Human Protein Atlas subcellular location by immunofluorescence; "Plasma membrane" in the
          main or additional location, with HPA's own reliability grade.
  GO      QuickGO, GO:0009986 `cell surface` with descendants, human proteins.

GO:0005886 `plasma membrane` is deliberately NOT a source. It truncates at QuickGO's 50,000-row
download cap (so membership would depend on alphabetical accession order), and the term includes
proteins facing the cytoplasm, which is not what a CAR binds.

`n_evidence` counts the four sources. One source is a prediction; two is a claim. It is a
screening filter, not a fact about a protein: a target still has to be argued from its topology
and from the literature, and CSPA membership outweighs the count.
"""
from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

NB_DIR = Path(__file__).resolve().parent.parent
SURF_DIR = NB_DIR / "data" / "external_ref" / "surfaceome"
RAW = SURF_DIR / "raw"
OUT_PQ = SURF_DIR / "surfaceome.parquet"


# SURFY was built in 2018 and carries the gene symbols of its day. Left unmapped, GPR56 and
# C10orf54 simply vanish from the reference under their current names, and so does any screen
# result for them. Only renames that touch the surface panel are listed; add as they surface.
ALIASES = {
    "GPR56": "ADGRG1", "C10orf54": "VSIR", "CD97": "ADGRE5", "GPR97": "ADGRG3",
    "KIAA0319L": "AAVR", "MARCH1": "MARCHF1", "MARCH5": "MARCHF5", "MARCH8": "MARCHF8",
    "SEPT7": "SEPTIN7", "HRD1": "SYVN1", "CD158k": "KIR3DL2", "FAM159B": "SHISAL2B",
    "GPR114": "ADGRG5", "GPR64": "ADGRG2", "EMR1": "ADGRE1", "EMR2": "ADGRE2",
    "TMEM180": "MFSD13A", "LEPROT": "LEPROT", "SLAMF4": "CD244",
}


def _canon(sym: pd.Series) -> pd.Series:
    """Current HGNC symbol where the source uses a retired one."""
    return sym.astype(str).str.strip().replace(ALIASES)


def _need(p: Path) -> Path:
    if not p.exists():
        sys.exit(f"missing {p}\nrun jobs/download_surfaceome.sh first")
    return p


def load_surfy():
    f = _need(RAW / "surfy" / "table_S3_surfaceome.xlsx")
    d = pd.read_excel(f, sheet_name="SurfaceomeMasterTable", header=1)
    d = d.rename(columns={
        "UniProt gene": "gene", "Surfaceome Label": "label",
        "Surfaceome Label Source": "label_source", "CD number": "cd_number",
        "Membranome Almen main-class": "almen_class", "topology": "topology",
        "TM domains": "n_tm", "MachineLearning FPR class (1=1%, 2=5%, 3=15%)": "ml_fpr_class",
    })
    raw_sym = d["gene"].astype(str).str.strip()
    d["gene"] = _canon(raw_sym)
    n_alias = int((d["gene"] != raw_sym).sum())
    d = d[d["gene"].ne("") & d["gene"].ne("nan")]
    surf = d[d["label"].astype(str).eq("surface")].copy()
    surf["in_surfy"] = True
    # A gene maps to several UniProt entries; keep the most confident row per gene.
    order = {"pos. trainingset": 0, "GPI (UniProt)": 1, "machine learning": 2, "neg. trainingset": 3}
    surf["_rank"] = surf["label_source"].map(order).fillna(9)
    surf = surf.sort_values("_rank").drop_duplicates("gene")
    cols = ["gene", "in_surfy", "label_source", "ml_fpr_class", "cd_number", "almen_class",
            "topology", "n_tm"]
    out = surf[[c for c in cols if c in surf.columns]].rename(
        columns={"label_source": "surfy_source", "ml_fpr_class": "surfy_ml_fpr",
                 "almen_class": "surfy_class"})
    # A SURFY training-set or GPI entry is not a prediction: it is a protein already known to be
    # at the surface. That distinction is what keeps KIR3DL2 -- a phase-2 CTCL target with no CSPA,
    # HPA or GO cell-surface record -- from being filtered out as single-source.
    out["surfy_curated"] = out["surfy_source"].isin(["pos. trainingset", "GPI (UniProt)"])
    print(f"SURFY : {len(d):,} entries -> {len(out):,} genes labelled surface "
          f"({int(out['surfy_curated'].sum()):,} curated, not predicted; "
          f"{n_alias} symbols remapped to current HGNC)")
    return out


def load_cspa():
    f = _need(RAW / "cspa" / "S2_File.xlsx")
    d = pd.read_excel(f, sheet_name="Table A")
    d = d[d["organism"].astype(str).str.lower().eq("human")].copy()
    d["gene"] = _canon(d["ENTREZ gene symbol"])
    d = d[d["gene"].ne("") & d["gene"].ne("nan")]
    d["cspa_category"] = d["CSPA category"].astype(str)
    d = d.sort_values("cspa_category").drop_duplicates("gene")      # "1 - high confidence" first
    out = d[["gene", "cspa_category", "CD"]].rename(columns={"CD": "cspa_cd"})
    out["in_cspa"] = True
    out["cspa_high_confidence"] = out["cspa_category"].str.startswith("1")
    print(f"CSPA  : {len(out):,} human genes detected by MS on intact cells "
          f"({int(out['cspa_high_confidence'].sum()):,} high confidence)")
    return out


def load_hpa():
    f = _need(RAW / "hpa" / "subcellular_location.tsv.zip")
    with zipfile.ZipFile(f) as z:
        with z.open("subcellular_location.tsv") as fh:
            d = pd.read_csv(fh, sep="\t")
    loc = (d["Main location"].fillna("") + ";" + d["Additional location"].fillna(""))
    pm = d[loc.str.contains("Plasma membrane", regex=False)].copy()
    pm["gene"] = _canon(pm["Gene name"])
    pm["in_hpa_pm"] = True
    pm["hpa_main"] = pm["Main location"].fillna("").str.contains("Plasma membrane", regex=False)
    out = (pm[["gene", "in_hpa_pm", "hpa_main", "Reliability"]]
           .rename(columns={"Reliability": "hpa_reliability"})
           .drop_duplicates("gene"))
    print(f"HPA   : {len(out):,} genes with plasma-membrane IF localisation")
    return out


def load_go():
    f = _need(RAW / "go" / "GO_0009986_cell_surface.tsv")
    d = pd.read_csv(f, sep="\t")
    n = len(d)
    if n >= 50000:
        sys.exit(f"{f.name} has {n:,} rows: it hit QuickGO's download cap and is truncated. "
                 "Membership would depend on accession order -- fix the download before building.")
    d["gene"] = _canon(d["SYMBOL"])
    d = d[d["gene"].ne("") & d["gene"].ne("nan")]
    # IEA is electronic annotation; keep it, but say which genes have only that.
    exp = ~d["GO EVIDENCE CODE"].astype(str).isin(["IEA"])
    out = (d.assign(_exp=exp).groupby("gene")
           .agg(go_cell_surface_experimental=("_exp", "any")).reset_index())
    out["in_go_cs"] = True
    print(f"GO    : {len(out):,} genes annotated GO:0009986 cell surface "
          f"({int(out['go_cell_surface_experimental'].sum()):,} with non-IEA evidence)")
    return out


def main():
    parts = [load_surfy(), load_cspa(), load_hpa(), load_go()]
    df = parts[0]
    for p in parts[1:]:
        df = df.merge(p, on="gene", how="outer")

    flags = ["in_surfy", "in_cspa", "in_hpa_pm", "in_go_cs"]
    for c in (*flags, "cspa_high_confidence", "hpa_main", "go_cell_surface_experimental",
              "surfy_curated"):
        df[c] = df[c].astype("object").where(df[c].notna(), False).astype(bool)
    # Mixed-type annotation columns ("2" and "2 (nonTM)" in the same SURFY column) go to string,
    # otherwise parquet refuses the frame.
    for c in ("surfy_ml_fpr", "cd_number", "cspa_cd", "n_tm"):
        if c in df.columns:
            df[c] = df[c].astype(str).replace({"nan": "", "None": ""})
    df["n_evidence"] = df[flags].sum(axis=1).astype(int)
    # The screening gate. Two independent sources, OR one source that is itself evidence rather
    # than prediction: mass spec on intact cells, or a SURFY curated/GPI entry.
    df["surface_confident"] = (df["n_evidence"] >= 2) | df["cspa_high_confidence"] | df["surfy_curated"]
    df = df.sort_values(["surface_confident", "n_evidence", "gene"],
                        ascending=[False, False, True]).reset_index(drop=True)

    SURF_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PQ, index=False)

    prov = {
        "built": time.strftime("%Y-%m-%d %H:%M"),
        "n_genes": int(len(df)),
        "by_n_evidence": {int(k): int(v) for k, v in df["n_evidence"].value_counts().items()},
        "sources": {p.parent.name + "/" + p.name: json.loads(p.read_text())
                    for p in sorted(RAW.rglob("*.provenance.json"))},
        "note": ("GO:0005886 plasma membrane is excluded on purpose: it truncates at QuickGO's "
                 "50k download cap and includes cytoplasm-facing proteins."),
    }
    (SURF_DIR / "surfaceome_provenance.json").write_text(json.dumps(prov, indent=1) + "\n")

    print(f"\nwrote {OUT_PQ}  {df.shape}")
    print(df["n_evidence"].value_counts().sort_index(ascending=False).to_string())
    for n in (4, 3, 2):
        print(f"  >= {n} sources: {int((df['n_evidence'] >= n).sum()):,} genes")

    # The screen is only as good as its positive controls: these must be in the reference.
    sys.path.insert(0, str(NB_DIR / "helpers"))
    import surface_targets as ST                                   # noqa: E402
    have = set(df.loc[df["surface_confident"], "gene"])
    missing = [g for g in ST.POSITIVE_CONTROLS if g not in have]
    print(f"surface_confident: {int(df['surface_confident'].sum()):,} genes")
    print("\npositive controls in the confident surfaceome: "
          f"{len(ST.POSITIVE_CONTROLS) - len(missing)}/{len(ST.POSITIVE_CONTROLS)}"
          + (f"  MISSING {missing}" if missing else ""))
    panel = pd.Index(ST.PANEL_GENES)
    print(f"a-priori panel covered: {panel.isin(have).sum()}/{len(panel)} genes "
          f"(not covered: {sorted(set(panel) - have)})")


if __name__ == "__main__":
    main()
