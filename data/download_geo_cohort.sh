#!/usr/bin/env bash
# Generic GEO series downloader for one atlas cohort dir.
# Modelled on data/B5_buus2025_blood_skin/download.sh (the only vetted precedent):
# egress smoke test -> series_matrix -> resumable RAW.tar -> extract -> md5.
#
# RUN ON A NODE WITH EGRESS -- NOT the fragile login node. Use jobs/run_download_new_cohorts.sh.
#   usage: bash download_geo_cohort.sh <LABEL> <GSE>
set -euo pipefail

LABEL="${1:?usage: download_geo_cohort.sh <LABEL> <GSE>}"
GSE="${2:?usage: download_geo_cohort.sh <LABEL> <GSE>}"
DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COHORT="$DATA_DIR/$LABEL"
STEM="${GSE%???}nnn"
BASE="https://ftp.ncbi.nlm.nih.gov/geo/series/${STEM}/${GSE}"

mkdir -p "$COHORT"/{raw,meta,processed}
cd "$COHORT"

echo "=========== [$LABEL / $GSE] $(date) ==========="

echo "[0] egress smoke test"
curl -sSf -I "${BASE}/suppl/filelist.txt" >/dev/null && echo "    egress OK" \
  || { echo "    NO EGRESS from this node"; exit 1; }

echo "[1] series matrix + filelist -> meta/ , raw/"
wget -q -c "${BASE}/matrix/${GSE}_series_matrix.txt.gz" -P meta/ || echo "    (no series_matrix)"
# filelist.txt is the per-GSM provenance ledger -- keep it (the handoff's -A '*.tar,*.gz' drops it)
wget -q "${BASE}/suppl/filelist.txt" -O raw/filelist.txt

echo "[2] RAW.tar -> raw/ (resumable; NCBI resets long transfers)"
wget -c --tries=20 --waitretry=15 --read-timeout=120 --timeout=120 \
     "${BASE}/suppl/${GSE}_RAW.tar" -P raw/

# series-level supplementary files that live outside RAW.tar (e.g. HTO feature references)
echo "[2b] series-level extras"
for f in $(sed -n '2,$p' raw/filelist.txt | cut -f2 \
             | grep -v '^GSM' | grep -v "^${GSE}_RAW.tar$" | grep -v '^filelist.txt$'); do
  echo "    $f"; wget -q -c "${BASE}/suppl/${f}" -P raw/ || echo "    (failed: $f)"
done

echo "[3] extract"
tar -xf "raw/${GSE}_RAW.tar" -C raw/
echo "    files in raw/: $(ls raw/ | wc -l)"
echo "    mtx:     $(ls raw/*matrix.mtx* 2>/dev/null | wc -l)"
echo "    h5:      $(ls raw/*.h5 2>/dev/null | wc -l)"
echo "    contigs: $(ls raw/*contig_annotations* 2>/dev/null | wc -l)"
echo "    zips:    $(ls raw/*.zip 2>/dev/null | wc -l)"
echo "    tar.gz:  $(ls raw/*.tar.gz 2>/dev/null | wc -l)"

echo "[4] md5"
md5sum "raw/${GSE}_RAW.tar" | tee meta/RAW.tar.md5

echo "[5] meta/meta.json"
python3 - "$LABEL" "$GSE" <<'PY'
import json, sys, pathlib, datetime
label, gse = sys.argv[1], sys.argv[2]
p = pathlib.Path("meta/meta.json")
man = {r.split("\t")[0]: r.split("\t") for r in
       pathlib.Path("../_new_cohorts.tsv").read_text().strip().split("\n")[1:]}
row = man.get(label, [label, gse, "", "", ""])
json.dump({"label": label, "accession": gse, "repo": "GEO", "access": "open",
           "compartment": row[2], "dedup_family": row[3], "note": row[4],
           "download_date": datetime.date.today().isoformat(),
           "url": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse}"},
          p.open("w"), indent=2)
print("   wrote", p.resolve())
PY

echo "[DONE $LABEL] $(date)"
