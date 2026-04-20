#!/usr/bin/env bash
# move_glm_dirs.sh
# Copies specified first-level directories across volumes using rsync.
# Source directories are NOT removed after transfer.
# Everything is logged to LOG_FILE.

set -euo pipefail

# ??? Configuration ????????????????????????????????????????????????????????????

DIRS_TO_MOVE=(
    "glm2_all_runs_scrubbed_2025-12-11-12-44"
    "glm2_chosen_all_runs_scrubbed_2025-12-11-11-22"
)

declare -A VOLUME_MAP=(
    ["/mnt/data2/learning-habits/spm_format_noSDC/outputs"]="/mnt/data/learning-habits/spm_format/outputs"
    ["/mnt/data2/learning-habits/spm_outputs_noSDC"]="/mnt/data/learning-habits/spm_outputs"
)

LOG_DIR="$(dirname "$0")/logs"
LOG_FILE="${LOG_DIR}/move_glm_dirs_$(date +%Y-%m-%d_%H-%M-%S).log"

# ??? Setup ????????????????????????????????????????????????????????????????????

mkdir -p "$LOG_DIR"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

log "=========================================================="
log "move_glm_dirs.sh started"
log "Log file: $LOG_FILE"
log "=========================================================="

OVERALL_STATUS=0

# ??? Main loop ????????????????????????????????????????????????????????????????

for dir_name in "${DIRS_TO_MOVE[@]}"; do
    log "----------------------------------------------------------"
    log "Processing: $dir_name"

    FOUND=false

    for src_base in "${!VOLUME_MAP[@]}"; do
        src="${src_base}/${dir_name}"
        dest_base="${VOLUME_MAP[$src_base]}"
        dest="${dest_base}/${dir_name}"

        if [[ ! -d "$src" ]]; then
            log "  Not found in: $src_base ? skipping"
            continue
        fi

        FOUND=true
        log "  Source : $src"
        log "  Dest   : $dest"

        mkdir -p "$dest_base"

        # ?? rsync transfer ??
        log "  Starting rsync..."

        if rsync -ah --stats --checksum "${src}/" "${dest}/" >> "$LOG_FILE" 2>&1; then
            log "  rsync completed successfully."
        else
            log "  ERROR: rsync failed for $dir_name."
            OVERALL_STATUS=1
            continue
        fi

        # ?? Post-transfer verification (file count + size) ??
        src_count=$(find "$src" -type f | wc -l)
        dest_count=$(find "$dest" -type f | wc -l)
        src_size=$(du -sb "$src" | cut -f1)
        dest_size=$(du -sb "$dest" | cut -f1)

        log "  Verification: src_files=$src_count  dest_files=$dest_count"
        log "  Verification: src_bytes=$src_size   dest_bytes=$dest_size"

        if [[ "$src_count" -ne "$dest_count" || "$src_size" -ne "$dest_size" ]]; then
            log "  ERROR: File count or size mismatch ? manual check required."
            OVERALL_STATUS=1
        else
            log "  Verification passed."
        fi
    done

    if ! $FOUND; then
        log "  WARNING: '$dir_name' not found in any source ? skipping."
    fi
done

# ??? Summary ??????????????????????????????????????????????????????????????????

log "=========================================================="
if [[ "$OVERALL_STATUS" -eq 0 ]]; then
    log "All operations completed successfully."
else
    log "One or more operations FAILED. Review the log: $LOG_FILE"
fi
log "move_glm_dirs.sh finished"
log "=========================================================="

exit $OVERALL_STATUS