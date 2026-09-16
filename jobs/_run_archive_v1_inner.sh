#!/usr/bin/env bash
set -euo pipefail
JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NB_DIR="$(cd "$JOB_DIR/.." && pwd)"
SRC="$NB_DIR/data"
DST="$NB_DIR/data/_archive_v1"

echo "[$(date)] host=$(hostname)  archiving v1 -> $DST"
mkdir -p "$DST"

# rsync, not cp: restartable, and --link-dest is deliberately NOT used -- hardlinks would
# make the archive alias the live files, so an in-place v2 rebuild would silently rewrite
# the archive too. This must be a real, independent copy.
for item in atlas_joint sample_metadata_final.csv sample_metadata.csv INVENTORY.tsv; do
  [ -e "$SRC/$item" ] || { echo "  (absent: $item)"; continue; }
  echo "[$(date)] copying $item"
  rsync -a --info=progress2 --no-inc-recursive "$SRC/$item" "$DST/"
done
echo "[$(date)] copying tables/"
rsync -a "$NB_DIR/tables" "$DST/"

echo "[$(date)] verifying"
for item in atlas_joint tables; do
  a=$(find "$SRC/$item" -type f 2>/dev/null | wc -l); [ "$item" = tables ] && a=$(find "$NB_DIR/tables" -type f | wc -l)
  b=$(find "$DST/$item" -type f 2>/dev/null | wc -l)
  sa=$(du -sb "$SRC/$item" 2>/dev/null | cut -f1); [ "$item" = tables ] && sa=$(du -sb "$NB_DIR/tables" | cut -f1)
  sb=$(du -sb "$DST/$item" 2>/dev/null | cut -f1)
  echo "  $item: files $a -> $b | bytes $sa -> $sb"
  [ "$a" = "$b" ] || { echo "  FILE COUNT MISMATCH"; exit 1; }
  [ "$sa" = "$sb" ] || { echo "  BYTE COUNT MISMATCH"; exit 1; }
done
date > "$DST/ARCHIVED_AT.txt"
echo "v1 atlas frozen before the v2 rebuild. Do not write here." >> "$DST/ARCHIVED_AT.txt"
echo "[$(date)] DONE archive verified"
