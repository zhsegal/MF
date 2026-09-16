#!/usr/bin/env python
"""Turn tables/atlas_dedup_v2.csv (pairwise overlaps) into a per-unit keep/drop ledger.

Union-find over the DUPLICATE pairs gives identity groups = one real library/patient across
however many deposits re-published it. Within each group exactly one deposit wins; the rest
are dropped. Preference order:
  1. a unit already in the v1 atlas  (never re-ingest what we hold)
  2. the deposit that also ships V(D)J for that unit
  3. the deposit with the most barcodes (least aggressively filtered)

Writes tables/atlas_dedup_v2_decisions.csv -- the file atlas_join_helpers reads in place of
the hardcoded SHARED_BRUNNER list.
"""
import collections
import itertools
import csv
from pathlib import Path

import polars as pl

NB = Path(__file__).resolve().parent.parent
PAIRS = NB / "tables" / "atlas_dedup_v2.csv"
OUT = NB / "tables" / "atlas_dedup_v2_decisions.csv"
OBS = NB / "data" / "atlas_joint" / "atlas_obs_full.parquet"

HELD = {"D1_chennareddy2025", "D3_brunner2024", "D5_rindler2021frontimm_multitissue",
        "D6_gaydosik2019_skin", "B1_borcherding2019_blood", "B2_borcherding2023_blood",
        "B4_geskin2026_dupilumab_blood", "Li2024_atlas"}
CODE = {c: c.split("_")[0] for c in HELD}
CODE["Li2024_atlas"] = "Li2024"

# A held cohort always wins over a new deposit -- never re-ingest material we hold.
# Within the held set, the tie-break is the cohort the v1 atlas actually took.
HELD_RANK = {"Li2024_atlas": 0, "D1_chennareddy2025": 1, "D6_gaydosik2019_skin": 2,
             "B4_geskin2026_dupilumab_blood": 3, "B2_borcherding2023_blood": 4,
             "B1_borcherding2019_blood": 5, "D3_brunner2024": 6,
             "D5_rindler2021frontimm_multitissue": 7}

# check_overlap.py keys units by the raw-file token, which is not always the atlas sample_id
# token. Three cohorts need an explicit bridge (obs sample_id suffix -> raw token).
UNIT_ALIAS = {
    "Li2024": lambda t: t,                                   # Li2024_atlas__PT47 -> Li2024/PT47
    "D5": lambda t: t.replace("MFIVB_", ""),                 # D5__MFIVB_skin     -> D5/skin
    "D6": {"SC50_NOR": "Labeled_SC50_011917_SK_NOR_GRCh38raw",
           "SC68_NOR": "Labeled_SC68_051517_SK_NOR_GRCh38raw",
           "SC124_NOR": "Labeled_SC124_080317_SK_NOR_GRCh38raw",
           "SC125_NOR": "Labeled_SC125_080317_SK_NOR_GRCh38raw",
           "SC67_MF2": "Labeled_SC67_050517_SK_MF2_GRCh38raw",
           "SC82_MF5": "Labeled_SC82_060617_SK_MF5_GRCh38raw",
           "SC157": "SC157dataframe", "SC158": "SC158dataframe",
           "SC205": "SC205dataframe"}.get,
}

# Cohort-level V(D)J is too coarse to pick a winner: GSE182861 ships a contig file for MF24
# with the CDR3 columns stripped, so a cohort-level flag elects that copy and drops
# GSE293752's usable one, losing the patient's clones entirely. Resolved per unit below.
HAS_VDJ_COHORT = {"D1_chennareddy2025", "D3_brunner2024", "D5_rindler2021frontimm_multitissue",
                  "B2_borcherding2023_blood", "B4_geskin2026_dupilumab_blood",
                  "B6_ren2023_blood", "B7_harro2023_blood", "B8_dorando2026_blood",
                  "D13_gaydosik2022_skin_blood", "D15_il4ra_blockade_skin",
                  "D16_song2024_transformed"}

_VDJ_CACHE: dict[str, dict[str, bool]] = {}


def _usable_contig(path: Path) -> bool:
    """A contig file counts only if it actually carries CDR3 amino-acid calls."""
    import gzip
    if not path.exists():
        return False
    if path.suffix in (".zip", ".gz") and ".tar" in path.name:
        return True                     # archives were validated to contain a 10x contig csv
    if path.suffix == ".zip":
        return True
    try:
        op = gzip.open if path.suffix == ".gz" else open
        with op(path, "rt", errors="replace") as fh:
            hdr = fh.readline().lower()
    except Exception:                                            # noqa: BLE001
        return False
    return "cdr3" in [c.strip().strip('"') for c in hdr.split(",")]


def unit_has_vdj(cohort: str, token: str) -> bool:
    """Per-unit V(D)J, read from that cohort's samples.tsv contig_file."""
    if cohort not in _VDJ_CACHE:
        m: dict[str, bool] = {}
        meta = NB / "data" / cohort / "meta" / "samples.tsv"
        if meta.exists():
            try:
                with meta.open(newline="") as fh:
                    for r in csv.DictReader(fh, delimiter="\t"):
                        cf = (r.get("contig_file") or "").strip()
                        m[(r.get("sample_id") or "").strip()] = bool(cf) and _usable_contig(
                            NB / "data" / cohort / "raw" / cf)
            except Exception as e:                               # noqa: BLE001
                print(f"    ! unit_has_vdj({cohort}): samples.tsv unreadable ({e!r}) "
                      f"-> falling back to the cohort-level flag")
        _VDJ_CACHE[cohort] = m
    known = _VDJ_CACHE[cohort]
    if known:
        return known.get(token, False)
    return cohort in HAS_VDJ_COHORT      # held cohorts with a different meta schema


def atlas_units() -> set[str]:
    """(dataset-code, raw donor token) actually present in the v1 atlas, as 'CODE/token'.

    sample_id in obs is '<CODE>__<raw sample token>', which is exactly the token
    check_overlap.py parses out of the raw filenames -- so they join directly.
    """
    sids = (pl.scan_parquet(OBS).select("sample_id").unique().collect()["sample_id"].to_list())
    out = set()
    for sid in sids:
        code, _, tok = sid.partition("__")
        code = "Li2024" if code == "Li2024_atlas" else code
        fn = UNIT_ALIAS.get(code)
        tok = (fn(tok) or tok) if fn else tok
        out.add(f"{code}/{tok}")
    return out


# reverse of UNIT_ALIAS: raw-file token -> the token that appears in obs.sample_id,
# so concat_joint can act on the ledger without re-deriving anything.
SAMPLE_ALIAS = {
    # D3's raw files are named by clinical group (`P112_HC1`) but its concat.h5ad -- and
    # therefore obs.sample_id -- uses tissue (`P112_Skin`). Without this the drop list
    # silently matches nothing, which is the failure mode this rewrite exists to remove.
    "D3": {"P112_HC1": "P112_Skin", "P115_HC2": "P115_Skin",
           "P116_HC3": "P116_Skin", "P121_HC4": "P121_Skin"},
    "D5": {"skin": "MFIVB_skin", "PBMC": "MFIVB_PBMC", "LN": "MFIVB_LN"},
    "D6": {"Labeled_SC50_011917_SK_NOR_GRCh38raw": "SC50_NOR",
           "Labeled_SC68_051517_SK_NOR_GRCh38raw": "SC68_NOR",
           "Labeled_SC124_080317_SK_NOR_GRCh38raw": "SC124_NOR",
           "Labeled_SC125_080317_SK_NOR_GRCh38raw": "SC125_NOR",
           "Labeled_SC67_050517_SK_MF2_GRCh38raw": "SC67_MF2",
           "Labeled_SC82_060617_SK_MF5_GRCh38raw": "SC82_MF5",
           "SC157dataframe": "SC157", "SC158dataframe": "SC158",
           "SC205dataframe": "SC205"},
}


def sample_id_of(code: str, unit: str) -> str:
    """'<CODE>__<token>' exactly as it appears in obs.sample_id."""
    tok = SAMPLE_ALIAS.get(code, {}).get(unit, unit)
    return f"{'Li2024_atlas' if code == 'Li2024' else code}__{tok}"


def write_patient_key(units, dup_pairs, out_path, cohort_of) -> None:
    """Cross-deposit patient identity.

    Two sources of evidence, unioned:
      1. barcode-identical pairs  -> the SAME LIBRARY in two deposits
         (this is what catches Vienna's MF309 == P76 rename)
      2. same (deposit family, donor token) -> the same patient's several libraries
         (barcodes cannot link two different biopsies of one person)

    Without this, patient P65 enters the atlas twice -- its lesional library wins in D8
    and its non-lesional library exists only in D7 -- which is the same defect as v1's
    D5__MFIVB / D1__P303.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("co", NB / "check_overlap.py")
    co = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(co)

    par: dict[str, str] = {}

    def find(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    def uni(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            par[ra] = rb

    # Only merge donor tokens ACROSS deposits where the deposits demonstrably share one
    # patient registry: Vienna P###, Pittsburgh MF##/SC###/SZ##, MDA PT##, WashU (B1 is a
    # documented subset of B2). "other" is a bucket, not a registry -- B6/P1 (Yale) and
    # D11/P1 (brentuximab) are different people, which the barcode gate confirms. Merging
    # them by token would reintroduce exactly the collision this whole gate exists to catch.
    SHARED_REGISTRY = {"vienna", "pittsburgh", "mda", "washu"}
    fam_of = {c.split("_")[0] if c != "Li2024_atlas" else "Li2024": v["fam"]
              for c, v in co.COHORTS.items()}
    for u in units:
        code, tok = u.split("/", 1)
        fam = fam_of.get(code, code)
        scope = fam if fam in SHARED_REGISTRY else f"{fam}/{code}"
        uni(u, f"@{scope}:{co.donor_of(code, tok)}")
    for a, b in dup_pairs:
        uni(a, b)

    grp = collections.defaultdict(list)
    for k in list(par):
        grp[find(k)].append(k)

    rows = []
    for members in grp.values():
        real = sorted(m for m in members if not m.startswith("@"))
        if not real:
            continue
        anchors = sorted(m[1:] for m in members if m.startswith("@"))
        if anchors:
            # canonical name: the anchor token backing the most units; ties go to the
            # cohort the atlas already prefers (D1 over D5, so `vienna:P303` not `:MFIVB`)
            def _score(a):
                tok = a.split(":", 1)[1]
                hits = [r for r in real
                        if co.donor_of(r.split("/")[0], r.split("/", 1)[1]) == tok]
                rank = min((HELD_RANK.get(cohort_of.get(r, ""), 99) for r in hits),
                           default=99)
                return (-len(hits), rank, a)
            key = min(anchors, key=_score)
        else:
            key = real[0]
        for m in real:
            code, tok = m.split("/", 1)
            rows.append(dict(unit=m, dataset=code, unit_token=tok,
                             sample_id=sample_id_of(code, tok),
                             donor_token=co.donor_of(code, tok),
                             patient_key=key,
                             n_deposits=len({x.split("/")[0] for x in real})))
    rows.sort(key=lambda r: (r["patient_key"], r["unit"]))
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    multi = {r["patient_key"] for r in rows if r["n_deposits"] > 1}
    print(f"-> {out_path}   {len({r['patient_key'] for r in rows})} patients, "
          f"{len(rows)} units, {len(multi)} patients spanning >1 deposit")


def main() -> None:
    in_atlas = atlas_units()
    rows = list(csv.DictReader(PAIRS.open()))
    dups = [r for r in rows if r["verdict"] == "DUPLICATE"]

    size, cohort_of = {}, {}
    for r in rows:
        for side in ("a", "b"):
            k = f"{CODE.get(r[f'cohort_{side}'], r[f'cohort_{side}'].split('_')[0])}/{r[f'donor_{side}']}"
            size[k] = int(r[f"n_{side}"])
            cohort_of[k] = r[f"cohort_{side}"]

    par: dict[str, str] = {}

    def find(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for r in dups:
        a = f"{CODE.get(r['cohort_a'], r['cohort_a'].split('_')[0])}/{r['donor_a']}"
        b = f"{CODE.get(r['cohort_b'], r['cohort_b'].split('_')[0])}/{r['donor_b']}"
        ra, rb = find(a), find(b)
        if ra != rb:
            par[ra] = rb

    groups = collections.defaultdict(list)
    for k in list(par):
        groups[find(k)].append(k)

    out, gid = [], 0
    for members in groups.values():
        gid += 1
        members = sorted(members)
        held = [m for m in members if cohort_of[m] in HELD]
        vdj = [m for m in members
               if unit_has_vdj(cohort_of[m], m.split("/", 1)[1])]
        if held:
            # Being IN the v1 atlas outranks the cohort preference. A held cohort can
            # contain raw units it never ingested (D3's P303_Blood has no samples.tsv row),
            # and letting one of those win would drop a real sample in favour of a unit
            # that is never built -- silently losing data.
            win = min(held, key=lambda m: (m not in in_atlas, HELD_RANK[cohort_of[m]], m))
            why = ("already in v1 atlas" if win in in_atlas
                   else "held cohort (this unit not ingested in v1)")
        elif vdj:
            win, why = max(vdj, key=lambda m: size[m]), "has usable V(D)J (CDR3 present)"
        else:
            win, why = max(members, key=lambda m: size[m]), "most barcodes"
        for m in members:
            out.append(dict(
                group=f"G{gid:03d}", unit=m, cohort=cohort_of[m], n_barcodes=size[m],
                in_v1_atlas=m in in_atlas,
                status="KEEP" if m == win else "DROP",
                reason=(f"winner ({why})" if m == win
                        else f"duplicate of {win} ({why})"),
                aliases=" ".join(x for x in members if x != m)))

    for r in out:
        code, tok = r["unit"].split("/", 1)
        r["sample_id"] = sample_id_of(code, tok)
    out.sort(key=lambda r: (r["group"], r["status"] != "KEEP", r["unit"]))
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    n_drop = sum(r["status"] == "DROP" for r in out)
    print(f"-> {OUT}   {gid} identity groups, {len(out)} units, {n_drop} DROP\n")

    intra = [r for r in out if r["status"] == "DROP" and r["in_v1_atlas"]]
    if intra:
        print("!! DROP units that are ALREADY IN THE v1 ATLAS (v1 double-counts these):")
        for r in intra:
            print(f"   {r['unit']:<24} {r['reason']}")
        print()

    print("identity groups spanning >=3 deposits:")
    for g, rs in itertools.groupby(out, key=lambda r: r["group"]):
        rs = list(rs)
        if len(rs) >= 3:
            print(f"   {g}: " + "  ".join(f"{r['unit']}{'*' if r['status']=='KEEP' else ''}" for r in rs))


    all_units = sorted(size)
    pairs = [(f"{CODE.get(r['cohort_a'], r['cohort_a'].split('_')[0])}/{r['donor_a']}",
              f"{CODE.get(r['cohort_b'], r['cohort_b'].split('_')[0])}/{r['donor_b']}")
             for r in dups]
    print()
    write_patient_key(all_units, pairs, NB / "tables" / "atlas_patient_key.csv", cohort_of)


if __name__ == "__main__":
    main()
