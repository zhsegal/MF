#!/usr/bin/env python
"""Barcode-overlap gate — the function `detect_duplicate_cells` claims to be but isn't.

`atlas_join_helpers.detect_duplicate_cells` (:1032) says in its docstring that it compares
cleaned barcodes; the body only counts cells by bare `real_donor` for D1/D3 against a
hardcoded 6-name list, so it can confirm an overlap you already know about but never
discover one. This does the actual comparison, straight off the raw deposits.

Why a plain intersection is enough: 10x barcodes are drawn from a fixed ~737k whitelist, so
two unrelated 5k-cell libraries share ~0.7% of barcodes by chance while a re-deposit of the
same library shares ~100%. The two regimes are two orders of magnitude apart.

Runs all pairs *within* a deposit family (Vienna / Pittsburgh / WashU / MDA) — cheap, and it
catches cross-deposit renames that a hand-written candidate list would miss (Vienna files the
same patient as `MF311` in one deposit and `P311` in another).

  python check_overlap.py                      # every family, write tables/atlas_dedup_v2.csv
  python check_overlap.py --family vienna      # one family
  python check_overlap.py --list               # show discovered (cohort, donor) units only
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import re
import sys
import zipfile
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent.parent
DATA = NB_DIR / "data"
OUT_CSV = NB_DIR / "tables" / "atlas_dedup_v2.csv"

# containment = |A n B| / min(|A|,|B|)  -- robust to very different library sizes
DUP_THRESH = 0.50      # >= this  -> same library / same patient material
CHANCE_MAX = 0.10      # <= this  -> chance-level (observed max for a true non-pair here: 0.061)

# ---------------------------------------------------------------------------- cohort specs
# donor_re: applied to the raw filename; group(1) is the donor token.
# kind:     how to pull the barcode list out of that file.
COHORTS = {
    # ---- already in the atlas -------------------------------------------------------
    "D1_chennareddy2025":          dict(fam="vienna",     kind="bctsv",  re=r"^GSM\d+_(.+?)_barcodes\.tsv\.gz$"),
    "D3_brunner2024":              dict(fam="vienna",     kind="bctsv",  re=r"^GSM\d+_(.+?)_barcodes\.tsv\.gz$"),
    "D5_rindler2021frontimm_multitissue": dict(fam="vienna", kind="bctsv", re=r"^GSM\d+_(.+?)_barcodes\.tsv\.gz$"),
    "D6_gaydosik2019_skin":        dict(fam="pittsburgh", kind="genesxcells", re=r"^GSM\d+_(.+?)\.csv\.gz$"),
    "B4_geskin2026_dupilumab_blood": dict(fam="pittsburgh", kind="cr_h5", re=r"^GSM\d+_filtered_feature_bc_matrix(.+?)\.h5$"),
    "B1_borcherding2019_blood":    dict(fam="washu",      kind="bctsv",  re=r"^GSM\d+_(.+?)_barcodes\.tsv\.gz$"),
    "B2_borcherding2023_blood":    dict(fam="washu",      kind="bctsv",  re=r"^GSM\d+_(.+?)_barcodes\.tsv\.gz$"),
    "Li2024_atlas":                dict(fam="mda",        kind="portal_h5ad"),
    # ---- new -------------------------------------------------------------------------
    "D7_rindler2021mc_skin":       dict(fam="vienna",     kind="bctsv",  re=r"^GSM\d+_(.+?)_barcodes\.tsv\.gz$"),
    "D8_alkon2024_parapsoriasis":  dict(fam="vienna",     kind="bctsv",  re=r"^GSM\d+_(.+?)_barcodes\.tsv\.gz$"),
    "D9_lyp_vs_ctcl_skin":         dict(fam="vienna",     kind="bctsv",  re=r"^GSM\d+_(.+?)_barcodes\.tsv\.gz$"),
    "D10_jonak2021_discordant":    dict(fam="vienna",     kind="bctsv",  re=r"^GSM\d+_(.+?)_barcodes\.tsv\.gz$"),
    "D13_gaydosik2022_skin_blood": dict(fam="pittsburgh", kind="cr_h5",  re=r"^GSM\d+_(.+?)_?raw_feature_bc_matrix\.h5$"),
    "D14_gaydosik2023_skin":       dict(fam="pittsburgh", kind="cr_h5",  re=r"^GSM\d+_(.+?)_?raw_feature_bc_matrix\.h5$"),
    "D15_il4ra_blockade_skin":     dict(fam="pittsburgh", kind="cr_h5",  re=r"^GSM\d+_filtered_feature_bc_matrix(.+?)\.h5$"),
    "B8_dorando2026_blood":        dict(fam="washu",      kind="bctsv",  re=r"^GSM\d+_(.+?)_barcodes\.tsv\.gz$"),
    "D16_song2024_transformed":    dict(fam="mda",        kind="zip",    re=r"^GSM\d+_(.+?)_scRNA\.zip$"),
    # no known deposit family, but they still carry colliding bare IDs (`P1`..`P11`,
    # `SS1`..`SS4`) -- they are only ever checked under --all-pairs, which is the default.
    "B6_ren2023_blood":            dict(fam="other",      kind="bctsv",  re=r"^GSM\d+_(.+?)_barcodes\.tsv\.gz$"),
    "B7_harro2023_blood":          dict(fam="other",      kind="bctsv",  re=r"^GSM\d+_(.+?)_barcodes\.tsv\.gz$"),
    "D11_brentuximab_skin":        dict(fam="other",      kind="bctsv",  re=r"^GSM\d+_(.+?)_barcodes\.tsv\.gz$"),
    "D12_pacritinib_skin":         dict(fam="other",      kind="bctsv",  re=r"^GSM\d+_(.+?)_barcodes\.tsv\.gz$"),
}

# unit token -> donor token. A deposit publishes several libraries per patient
# (lesional/non-lesional, thin/thick, Tumor/PP, serial dates); barcode overlap can only
# identify the same *library*, so patient identity needs this on top.
DONOR_OF = {
    "D1":  lambda u: u.replace("_skin", "").replace("_thick", "_thick").replace("P_GS", "PGS"),
    "D3":  lambda u: u.split("_")[0],
    "D5":  lambda u: "MFIVB",
    "D6":  lambda u: re.sub(r"^Labeled_|dataframe$", "", u).split("_")[0],
    "B1":  lambda u: u, "B2": lambda u: u, "B4": lambda u: u,
    "Li2024": lambda u: u,
    "D7":  lambda u: u.split("_")[0],
    "D8":  lambda u: u.split("_")[0],
    "D9":  lambda u: u,
    "D10": lambda u: "JONAK1",
    "D11": lambda u: u.split("_")[0],
    "D12": lambda u: "P1_UM",
    "D13": lambda u: u,
    "D14": lambda u: u,
    "D15": lambda u: u.replace("culture", "").replace("_3", ""),
    "D16": lambda u: u.split("_")[0],
    "B6":  lambda u: u, "B7": lambda u: u,
    "B8":  lambda u: re.match(r"(\d+)PB", u).group(1) if re.match(r"(\d+)PB", u) else u,
}


def donor_of(code: str, unit: str) -> str:
    fn = DONOR_OF.get(code)
    return fn(unit) if fn else unit


FAMILIES = sorted({c["fam"] for c in COHORTS.values()})
_LANE = re.compile(r"-\d+$")
_CORE = re.compile(r"[ACGT]{16}")
WHITELIST = 737_280          # 10x v3 barcode whitelist size, for the chance baseline


def clean(bc: str) -> str:
    """Reduce any deposit's barcode string to the bare 16-nt core.

    Extends atlas_join_helpers._clean_bc (:246), which only strips the `-N` lane suffix.
    Deposits also prefix the sample: D6 writes `SC50nor_AAACCTGAGCCCGAAA`, and without
    stripping that, D6 shares zero barcodes with every other cohort and its genuine
    re-deposit in GSE206123 is invisible.
    """
    bc = _LANE.sub("", bc.strip().strip('"'))
    m = _CORE.search(bc)
    return m.group(0) if m else bc


# ------------------------------------------------------------------------------- readers
def _bctsv(p: Path) -> set[str]:
    with gzip.open(p, "rt") as fh:
        return {clean(l) for l in fh if l.strip()}


RAW_H5_MIN_GENES = 200      # matches qc_filter's default min_genes (atlas_join_helpers.py:712)
RAW_H5_WHITELIST = 100_000  # above this many barcodes the .h5 is an unfiltered raw matrix


def _cr_h5(p: Path) -> set[str]:
    """CellRanger .h5. Unfiltered `raw_feature_bc_matrix.h5` holds the ENTIRE ~737k 10x
    whitelist, so a naive barcode set makes every comparison a trivial 1.000 containment.
    Detect that case and apply a crude cell call from `indptr` alone (genes detected per
    barcode) -- no need to read the counts, and it matches the pipeline's own min_genes."""
    import h5py
    import numpy as np
    with h5py.File(p, "r") as f:
        grp = "matrix" if "matrix" in f else next(iter(f.keys()))
        bcs = f[grp]["barcodes"][:]
        if len(bcs) <= RAW_H5_WHITELIST:
            return {clean(b.decode()) for b in bcs}
        indptr = f[grp]["indptr"][:]                       # CSC: one entry per barcode
    n_genes = np.diff(indptr)
    keep = n_genes >= RAW_H5_MIN_GENES
    print(f"      (raw h5 {p.name}: {len(bcs):,} whitelist -> {int(keep.sum()):,} cells "
          f"at >={RAW_H5_MIN_GENES} genes)", file=sys.stderr)
    return {clean(b.decode()) for b in bcs[keep]}


def _genesxcells(p: Path) -> set[str]:
    """D6: genes x cells CSV -- barcodes are the header row."""
    with gzip.open(p, "rt") as fh:
        hdr = next(csv.reader(fh))
    return {clean(x) for x in hdr[1:] if x.strip()}


def _zip(p: Path) -> set[str]:
    with zipfile.ZipFile(p) as z:
        cand = [n for n in z.namelist() if n.endswith("barcodes.tsv.gz") or n.endswith("barcodes.tsv")]
        if not cand:
            return set()
        raw = z.read(cand[0])
        if cand[0].endswith(".gz"):
            raw = gzip.decompress(raw)
        return {clean(l) for l in raw.decode().splitlines() if l.strip()}


READERS = {"bctsv": _bctsv, "cr_h5": _cr_h5, "genesxcells": _genesxcells, "zip": _zip}


def _portal_units() -> dict[str, set[str]]:
    """Li's processed portal object: index is '<barcode>-0_<donor>_..' ."""
    import h5py
    import numpy as np
    h5 = DATA / "CTCL_all_final_portal_tags.h5ad"
    if not h5.exists():
        return {}
    out: dict[str, set[str]] = {}
    with h5py.File(h5, "r") as f:
        idx = f["obs"][f["obs"].attrs["_index"]][:]
        g = f["obs"]["donor"]
        cats = np.array([c.decode() for c in g["categories"][:]])
        donors = cats[g["codes"][:]]
    for bc, d in zip(idx, donors):
        out.setdefault(d, set()).add(clean(bc.decode().split("_")[0]))
    return out


# ------------------------------------------------------------------------- unit discovery
def units_for(label: str) -> dict[str, set[str]]:
    """-> {donor_token: barcode set} for one cohort."""
    spec = COHORTS[label]
    if spec["kind"] == "portal_h5ad":
        return _portal_units()
    raw = DATA / label / "raw"
    if not raw.is_dir():
        return {}
    pat, rd = re.compile(spec["re"]), READERS[spec["kind"]]
    # a deposit can ship a library with no sample token: GSE264636's 15th sample is just
    # `GSM9038965_barcodes.tsv.gz` (it is patient P220). Fall back to the GSM accession so
    # the gate still compares it -- otherwise a whole donor skips the dedup check.
    bare = re.compile(r"^(GSM\d+)_barcodes\.tsv\.gz$")
    out: dict[str, set[str]] = {}
    for p in sorted(raw.iterdir()):
        m = pat.match(p.name) or bare.match(p.name)
        if not m:
            continue
        try:
            bcs = rd(p)
        except Exception as e:                                    # noqa: BLE001
            print(f"    ! {label}/{p.name}: {e!r}", file=sys.stderr)
            continue
        if bcs:
            out.setdefault(m.group(1), set()).update(bcs)
    return out


def verdict(cont: float, enrich: float) -> str:
    """Containment alone is size-biased: a 400-barcode set is ~10% contained in a 40k set
    purely by chance. Require the overlap to also be far above the whitelist expectation."""
    if enrich < 3.0:
        return "chance"
    if cont >= DUP_THRESH:
        return "DUPLICATE"
    if cont <= CHANCE_MAX:
        return "chance"
    return "PARTIAL-REVIEW"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=FAMILIES, action="append",
                    help="restrict to one deposit family (default: all cohorts, all pairs)")
    ap.add_argument("--by-family", action="store_true",
                    help="only compare within a family (faster, but assumes the family map "
                         "is right -- assuming which pairs can collide is exactly how the "
                         "GSE173205 patient count was got wrong)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--min-cells", type=int, default=100)
    args = ap.parse_args()
    fams = args.family or FAMILIES

    loaded: dict[str, dict[str, set[str]]] = {}
    for lab, spec in COHORTS.items():
        if spec["fam"] not in fams:
            continue
        u = units_for(lab)
        u = {d: b for d, b in u.items() if len(b) >= args.min_cells}
        if u:
            loaded[lab] = u
        print(f"[{spec['fam']:11s}] {lab:32s} {len(u):3d} units  "
              f"{sum(len(b) for b in u.values()):>8,} barcodes")
    if args.list:
        for lab, u in loaded.items():
            print(f"\n{lab}: " + " ".join(f"{d}({len(b)})" for d, b in sorted(u.items())))
        return 0

    rows = []
    blocks = ([[l for l in loaded if COHORTS[l]["fam"] == f] for f in fams]
              if args.by_family else [list(loaded)])
    for labs in blocks:
        for i, la in enumerate(labs):
            fam = COHORTS[la]["fam"]
            for lb in labs[i + 1:]:
                for da, ba in sorted(loaded[la].items()):
                    for db, bb in sorted(loaded[lb].items()):
                        inter = len(ba & bb)
                        if not inter:
                            continue
                        cont = inter / min(len(ba), len(bb))
                        jac = inter / len(ba | bb)
                        exp = len(ba) * len(bb) / WHITELIST      # expected by chance
                        enr = inter / exp if exp else 0.0
                        rows.append(dict(family=(fam if COHORTS[lb]["fam"] == fam
                                                 else f"{fam}|{COHORTS[lb]['fam']}"),
                                         cohort_a=la, donor_a=da, n_a=len(ba),
                                         cohort_b=lb, donor_b=db, n_b=len(bb),
                                         n_shared=inter, containment=round(cont, 4),
                                         jaccard=round(jac, 4), enrichment=round(enr, 2),
                                         verdict=verdict(cont, enr)))
    rows.sort(key=lambda r: -r["containment"])
    OUT_CSV.parent.mkdir(exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else
                           ["family", "cohort_a", "donor_a", "n_a", "cohort_b", "donor_b",
                            "n_b", "n_shared", "containment", "jaccard", "verdict"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n-> {OUT_CSV}  ({len(rows)} pairs with any overlap)")
    hits = [r for r in rows if r["verdict"] != "chance"]
    print(f"\n{'':4}{'family':<11} {'A':<30} {'B':<30} {'shared':>7} {'cont':>6} {'enr':>7}  verdict")
    for r in hits[:120]:
        print(f"{'':4}{r['family']:<11} {r['cohort_a'].split('_')[0]+'/'+r['donor_a']:<30} "
              f"{r['cohort_b'].split('_')[0]+'/'+r['donor_b']:<30} "
              f"{r['n_shared']:>7,} {r['containment']:>6.3f} {r['enrichment']:>7.1f}  {r['verdict']}")
    if not hits:
        print("    (no pair above chance level)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
