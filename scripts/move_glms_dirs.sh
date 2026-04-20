#!/usr/bin/env bash
# move_glm_dirs.sh
# Copies specified first-level directories across volumes using rsync.
# Rewrites any absolute symlinks in the destination to reflect new paths.
# Source directories are NOT removed after transfer.
# Everything is logged to LOG_FILE.

set -euo pipefail

# Configuration

DIRS_TO_MOVE=(
    "PPI"
    "glm2_all_runs_scrubbed_2025-12-11-12-44"
    "glm2_chosen_all_runs_scrubbed_2025-12-11-11-22"
)

declare -A VOLUME_MAP=(
    ["/mnt/data2/learning-habits/spm_format_noSDC/outputs"]="/mnt/data/learning-habits/spm_format/outputs"
    ["/mnt/data2/learning-habits/spm_outputs_noSDC"]="/mnt/data/learning-habits/spm_outputs"
)

# Symlink target rewrites: old substring -> new substring
# Applied in order to every absolute symlink found in the destination.
declare -A SYMLINK_REWRITES=(
    ["/mnt/data2/learning-habits/spm_format_noSDC/outputs"]="/mnt/data/learning-habits/spm_format/outputs"
    ["/mnt/data2/learning-habits/spm_outputs_noSDC"]="/mnt/data/learning-habits/spm_outputs"
    ["/mnt/data/learning-habits/spm_format_noSDC/outputs"]="/mnt/data/learning-habits/spm_format/outputs"
    ["/mnt/data/learning-habits/spm_outputs_noSDC"]="/mnt/data/learning-habits/spm_outputs"
)

LOG_DIR="$(dirname "$0")/logs"
LOG_FILE="${LOG_DIR}/move_glm_dirs_$(date +%Y-%m-%d_%H-%M-%S).log"

# Setup

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

# Main loop

for dir_name in "${DIRS_TO_MOVE[@]}"; do
    log "----------------------------------------------------------"
    log "Processing: $dir_name"

    FOUND=false

    for src_base in "${!VOLUME_MAP[@]}"; do
        src="${src_base}/${dir_name}"
        dest_base="${VOLUME_MAP[$src_base]}"
        dest="${dest_base}/${dir_name}"

        if [[ ! -d "$src" ]]; then
            log "  Not found in: $src_base - skipping"
            continue
        fi

        FOUND=true
        log "  Source : $src"
        log "  Dest   : $dest"

        mkdir -p "$dest_base"

        # rsync transfer
        log "  Starting rsync..."

        if rsync -ah --stats --checksum "${src}/" "${dest}/" >> "$LOG_FILE" 2>&1; then
            log "  rsync completed successfully."
        else
            log "  ERROR: rsync failed for $dir_name."
            OVERALL_STATUS=1
            continue
        fi

        # Post-transfer verification (file count only)
        src_count=$(find "$src" -type f | wc -l)
        dest_count=$(find "$dest" -type f | wc -l)

        log "  Verification: src_files=$src_count  dest_files=$dest_count"

        if [[ "$src_count" -ne "$dest_count" ]]; then
            log "  ERROR: File count mismatch - manual check required."
            OVERALL_STATUS=1
            continue
        fi

        log "  Verification passed."

        # Symlink rewriting
        symlink_count=$(find "$dest" -type l | wc -l)
        if [[ "$symlink_count" -eq 0 ]]; then
            log "  No symlinks found - skipping rewrite step."
            continue
        fi

        log "  Rewriting $symlink_count symlinks..."
        rewritten=0
        broken=0

        while IFS= read -r link; do
            target=$(readlink "$link")

            # Only rewrite absolute symlinks
            if [[ "$target" != /* ]]; then
                continue
            fi

            new_target="$target"
            for old in "${!SYMLINK_REWRITES[@]}"; do
                new="${SYMLINK_REWRITES[$old]}"
                new_target="${new_target//$old/$new}"
            done

            if [[ "$new_target" != "$target" ]]; then
                ln -sf "$new_target" "$link"
                ((rewritten++)) || true
            fi

            if [[ ! -e "$link" ]]; then
                log "  WARNING: symlink still broken after rewrite: $link -> $new_target"
                ((broken++)) || true
            fi
        done < <(find "$dest" -type l)

        log "  Symlinks rewritten: $rewritten  Still broken: $broken"
        if [[ "$broken" -gt 0 ]]; then
            OVERALL_STATUS=1
        fi
    done

    if ! $FOUND; then
        log "  WARNING: '$dir_name' not found in any source - skipping."
    fi
done

# Summary

log "=========================================================="
if [[ "$OVERALL_STATUS" -eq 0 ]]; then
    log "All operations completed successfully."
else
    log "One or more operations FAILED. Review the log: $LOG_FILE"
fi
log "move_glm_dirs.sh finished"
log "=========================================================="

exit $OVERALL_STATUS