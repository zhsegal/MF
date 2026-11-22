#!/usr/bin/env bash
# Download the four surfaceome reference sources into data/external_ref/surfaceome/raw/.
#
# Small (tens of MB) and network-only, so this runs on the login node -- no bsub.
# Each file lands in its own directory with a provenance record (URL, date, sha256). Nothing here
# is parsed; jobs/build_surfaceome_ref.py turns raw/ into surfaceome.parquet.
#
#   SURFY     Bausch-Fluck et al. PNAS 2018, the in-silico human surfaceome (prediction, genome-wide)
#   CSPA      Cell Surface Protein Atlas, Bausch-Fluck et al. PLoS ONE 2015 -- mass spectrometry on
#             INTACT cells. The only experimental line of evidence of the four
#   HPA        Human Protein Atlas subcellular location, "Plasma membrane" (IF, with a reliability)
#   QuickGO   GO:0009986 cell surface + GO:0005886 plasma membrane, human, with descendants
#
# Usage:  ./download_surfaceome.sh  [--force]
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$JOB_DIR/../data/external_ref/surfaceome/raw"
FORCE="${1:-}"
mkdir -p "$OUT"

fetch() {   # fetch <subdir> <filename> <url>
    local dir="$OUT/$1" file="$1/$2" url="$3"
    mkdir -p "$dir"
    if [[ -s "$OUT/$file" && "$FORCE" != "--force" ]]; then
        echo "have   $file"
        return
    fi
    echo "get    $file"
    # --fail is not enough: wlab.ethz.ch answers 200 with its landing page for a missing file
    # (it 301s to wollscheidlab.org and drops the filename), so the content type is checked too.
    curl -fsSL --max-time 600 --retry 3 -A "MF-atlas/1.0" -H "Accept: text/tsv" \
         -o "$OUT/$file" "$url"
    if head -c 512 "$OUT/$file" | grep -qiE '<!doctype html|<html'; then
        rm -f "$OUT/$file"
        echo "ERROR: $url served an HTML page, not a data file -- the source has moved." >&2
        exit 2
    fi
    python3 - "$OUT/$file" "$url" <<'PY'
import hashlib, json, sys, time
from pathlib import Path
p, url = Path(sys.argv[1]), sys.argv[2]
h = hashlib.sha256(p.read_bytes()).hexdigest()
p.with_suffix(p.suffix + ".provenance.json").write_text(json.dumps(
    {"url": url, "downloaded": time.strftime("%Y-%m-%d %H:%M"),
     "bytes": p.stat().st_size, "sha256": h}, indent=1) + "\n")
print(f"       {p.stat().st_size:,} bytes  sha256 {h[:16]}...")
PY
}

# wlab.ethz.ch moved to wollscheidlab.org in 2025 and its redirect drops the filename.
fetch surfy table_S3_surfaceome.xlsx \
      "https://wollscheidlab.org/SURFY/table_S3_surfaceome.xlsx"

fetch cspa S2_File.xlsx \
      "https://wollscheidlab.org/CSPA/data/S2_File.xlsx"

fetch hpa subcellular_location.tsv.zip \
      "https://www.proteinatlas.org/download/tsv/subcellular_location.tsv.zip"

GOBASE="https://www.ebi.ac.uk/QuickGO/services/annotation/downloadSearch"
# downloadLimit is capped at 50000 by QuickGO; both terms come in well under it.
GOARGS="taxonId=9606&geneProductType=protein&goUsage=descendants&downloadLimit=50000"
fetch go GO_0009986_cell_surface.tsv   "${GOBASE}?${GOARGS}&goId=GO:0009986"
# GO:0005886 plasma membrane is deliberately NOT fetched: it has ~90k human annotations and
# truncates at the 50k cap, so membership would depend on accession order. It also includes
# cytoplasm-facing proteins, which is not what an antibody can reach.

echo
echo "raw sources in $OUT"
du -sh "$OUT"/* 2>/dev/null || true
echo "next: python jobs/build_surfaceome_ref.py"
