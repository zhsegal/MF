#!/usr/bin/env bash
# Download the genome-scale CD4+ T-cell Perturb-seq *DE estimates* (Zhu, Dann et al., Cell 2026;
# "GWCD4i") used by notebooks/20_skin/29_perturbseq_reverse_engineering.ipynb.
#
#   Zhu R., Dann E., Yan J., ... Pritchard J.K., Marson A. (2026) "Genome-scale perturb-seq in
#   primary human CD4+ T cells maps context-specific regulators of T cell programs and human
#   immune traits". Cell, doi:10.1016/j.cell.2026.08.002 (preprint doi:10.64898/2025.12.23.696273).
#   Data: CZI Virtual Cells Platform, MIT licence; public S3 bucket, no credentials.
#
# What is fetched (~15.7 GB): the per-perturbation DESeq2 results (`GWCD4i.DE_stats.h5ad`,
# 33,983 perturbation x condition rows x 10,282 genes) and the small supplementary tables.
# NOT fetched: the 12 cell-level objects (~1.6 TB), the 41.5 GB pseudobulk and the by-guide /
# by-donor MuData -- the analysis only needs the DE estimates.
# Also fetched from GitHub: the Arce 2024 TCR-activation DESeq2 table the authors use as a
# control signature (5 MB).  The model class itself is vendored in helpers/pert2state_model.py.
#
# Resumable: `aws s3 cp` is skipped for a file whose size already matches the manifest.
#
# Usage:
#   ./download_perturbseq_gwcd4i.sh            # runs in the foreground (network-bound, ~1 h)
#   bsub -q short -W 6:00 -o download_perturbseq_gwcd4i.bsub.log ./download_perturbseq_gwcd4i.sh
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$(dirname "$JOB_DIR")/data/external_ref/perturbseq_gwcd4i"
BUCKET="s3://genome-scale-tcell-perturb-seq/marson2025_data"
AWS="${AWS:-/usr/local/bin/aws}"
HTTPS="https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data"
# the AWS CLI exists on the login nodes only; compute nodes reach the public bucket over HTTPS
if [[ -x "$AWS" ]]; then TRANSPORT=aws; else TRANSPORT=curl; fi
echo "transport: $TRANSPORT"
PYBIN="${PYBIN:-/home/projects/nyosef/zvise/.local/share/mamba/envs/neural_nmf_env/bin/python}"
mkdir -p "$OUT_DIR/suppl_tables"

# file -> expected bytes (S3 listing 2026-11-05)
declare -A MANIFEST=(
    ["GWCD4i.DE_stats.h5ad"]=0
    ["data_sharing_readme.md"]=0
    ["suppl_tables/DE_stats.suppl_table.csv"]=0
    ["suppl_tables/clustering_results_and_annotations.csv"]=0
    ["suppl_tables/clustering_downstream_genes.csv.gz"]=0
    ["suppl_tables/sample_metadata.suppl_table.csv"]=0
    ["suppl_tables/sgrna_library_metadata.suppl_table.csv"]=0
    ["suppl_tables/guide_kd_efficiency.suppl_table.csv"]=0
    ["suppl_tables/table_name_conversion.md"]=0
)

# free-space guard: the h5ad is 15.6 GB and the volume runs near capacity
FREE_GB="$(df -BG --output=avail "$OUT_DIR" | tail -1 | tr -dc '0-9')"
if (( FREE_GB < 40 )); then
    echo "only ${FREE_GB} GB free under $OUT_DIR -- need >= 40 GB" >&2
    exit 2
fi

remote_size() {  # bytes of one S3 object
    if [[ "$TRANSPORT" == aws ]]; then
        "$AWS" s3 ls --no-sign-request "$BUCKET/$1" | awk -v f="$(basename "$1")" '$4==f {print $3}'
    else
        curl -sIL --retry 5 --retry-delay 5 "$HTTPS/$1" \
            | awk 'tolower($1)=="content-length:" {gsub(/\r/,"",$2); n=$2} END {print n}'
    fi
}

fetch() {  # $1 = key relative to the bucket prefix, $2 = destination path, $3 = expected bytes
    if [[ "$TRANSPORT" == aws ]]; then
        "$AWS" s3 cp --no-sign-request --only-show-errors "$BUCKET/$1" "$2"
        return
    fi
    # The S3 endpoint drops long transfers (curl exit 56 at ~2.7 GB of the 15.6 GB h5ad), and
    # --retry restarts the request rather than the byte range, so resume explicitly with -C -.
    local want="${3:-0}" try
    for try in $(seq 1 40); do
        curl -L --no-progress-meter --fail --retry 5 --retry-delay 5 --retry-all-errors \
             --speed-limit 10240 --speed-time 120 -C - -o "$2" "$HTTPS/$1" && :
        local have=0
        [[ -f "$2" ]] && have="$(stat -c %s "$2")"
        if [[ "$want" == 0 || "$have" == "$want" ]]; then return 0; fi
        echo "  attempt $try: $have / $want bytes, resuming" >&2
        sleep 5
    done
    echo "gave up on $1 after 40 attempts" >&2
    return 1
}

echo "[$(date)] -> $OUT_DIR"
for f in "${!MANIFEST[@]}"; do
    want="$(remote_size "$f")"
    [[ -n "$want" ]] || { echo "not in bucket: $f" >&2; exit 1; }
    MANIFEST["$f"]="$want"
    if [[ -f "$OUT_DIR/$f" ]] && [[ "$(stat -c %s "$OUT_DIR/$f")" == "$want" ]]; then
        echo "already complete: $f ($want bytes)"; continue
    fi
    echo "[$(date)] $f ($want bytes)"
    fetch "$f" "$OUT_DIR/$f" "$want"
    got="$(stat -c %s "$OUT_DIR/$f")"
    [[ "$got" == "$want" ]] || { echo "SIZE MISMATCH $f: got $got, expected $want" >&2; exit 1; }
done

# Arce 2024 Teff stimulated-vs-resting DESeq2 table (authors' control signature; GitHub, MIT)
ARCE="Arce2024_20230130_DESeq2_output_AAVS1_Teff_Stimulation_vs_Resting.csv"
ARCE_URL="https://raw.githubusercontent.com/emdann/GWT_perturbseq_analysis_2025/master/metadata/$ARCE"
if [[ ! -s "$OUT_DIR/$ARCE" ]]; then
    echo "[$(date)] $ARCE (GitHub)"
    curl -L --no-progress-meter --fail --retry 5 --retry-delay 10 -o "$OUT_DIR/$ARCE" "$ARCE_URL"
fi
head -1 "$OUT_DIR/$ARCE" | grep -q '"gene_id","baseMean","log2FoldChange"' \
    || { echo "unexpected header in $ARCE" >&2; exit 1; }

# provenance
"$PYBIN" - "$OUT_DIR" "$BUCKET" "$ARCE_URL" <<'PY'
import json, sys, datetime, pathlib
out, bucket, arce = sys.argv[1:4]
out = pathlib.Path(out)
files = {str(p.relative_to(out)): p.stat().st_size for p in out.rglob("*") if p.is_file() and p.name != "provenance.json"}
json.dump({
    "dataset": "Primary Human CD4+ T Cell Perturb-seq (GWCD4i) v1.0, DE estimates + supplementary tables",
    "paper": "Zhu, Dann et al. Cell 2026, doi:10.1016/j.cell.2026.08.002",
    "preprint": "doi:10.64898/2025.12.23.696273",
    "portal": "https://virtualcellmodels.cziscience.com/dataset/genome-scale-tcell-perturb-seq",
    "s3": bucket, "licence": "MIT", "arce2024_activation_signature": arce,
    "model_code": "https://github.com/emdann/pert2state_model (vendored: helpers/pert2state_model.py)",
    "downloaded": datetime.date.today().isoformat(), "files_bytes": files,
    "not_downloaded": ["D*_*.assigned_guide.h5ad (cell level, ~1.6 TB)", "GWCD4i.pseudobulk_merged.h5ad (41.5 GB)",
                       "GWCD4i.DE_stats.by_guide.h5mu", "GWCD4i.DE_stats.by_donors.h5mu"],
}, open(out / "provenance.json", "w"), indent=2)
print("wrote", out / "provenance.json")
PY
echo "[$(date)] done"
