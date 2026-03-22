#!/bin/bash
# Download nuScenes trainval blobs (parts 01-10), extract only camera samples,
# place into the correct directory, and delete the tgz to save space.
#
# Usage: bash scripts/download_nuscenes.sh [MAX_PARALLEL]
#   MAX_PARALLEL: number of concurrent downloads (default: 3)
#   Each parallel job needs ~30GB disk for its tgz, so 3 jobs ≈ 90GB peak.

MAX_JOBS=${1:-3}
BASE_URL="https://motional-nuscenes.s3.amazonaws.com/public/v1.0"
TARGET_DIR="/scratch/gpfs/ZHUANGL/el5267/thesis-research/DriveBench/data/nuscenes"
DOWNLOAD_DIR="/scratch/gpfs/ZHUANGL/el5267/thesis-research/DriveBench/data"
LOG_DIR="${DOWNLOAD_DIR}/download_logs"

mkdir -p "$TARGET_DIR/samples/CAM_BACK" \
         "$TARGET_DIR/samples/CAM_BACK_LEFT" \
         "$TARGET_DIR/samples/CAM_BACK_RIGHT" \
         "$TARGET_DIR/samples/CAM_FRONT" \
         "$TARGET_DIR/samples/CAM_FRONT_LEFT" \
         "$TARGET_DIR/samples/CAM_FRONT_RIGHT" \
         "$LOG_DIR"

process_part() {
    local i=$1
    local FILENAME="v1.0-trainval${i}_blobs.tgz"
    local URL="${BASE_URL}/${FILENAME}"
    local FILEPATH="${DOWNLOAD_DIR}/${FILENAME}"
    local LOG="${LOG_DIR}/part${i}.log"

    echo "[Part ${i}] Starting..." | tee "$LOG"

    # Download
    if [ -f "$FILEPATH" ]; then
        echo "[Part ${i}] Already downloaded, skipping." | tee -a "$LOG"
    else
        echo "[Part ${i}] Downloading..." | tee -a "$LOG"
        wget -c -q --show-progress -O "$FILEPATH" "$URL" 2>&1 | tail -1 | tee -a "$LOG"
    fi

    # Extract only camera sample files
    echo "[Part ${i}] Extracting camera samples..." | tee -a "$LOG"
    tar xzf "$FILEPATH" -C "$TARGET_DIR" --wildcards 'samples/CAM_*' --strip-components=0 2>/dev/null || true

    # Delete the tgz
    rm -f "$FILEPATH"

    echo "[Part ${i}] Done." | tee -a "$LOG"
}

export -f process_part
export BASE_URL TARGET_DIR DOWNLOAD_DIR LOG_DIR

echo "Downloading nuScenes parts 01-10 with ${MAX_JOBS} parallel jobs..."
echo "Logs in: ${LOG_DIR}/"
echo ""

seq -w 1 10 | xargs -P "$MAX_JOBS" -I {} bash -c 'process_part "$@"' _ {}

echo ""
echo "=========================================="
echo "All parts downloaded and extracted."
echo "Camera files in: ${TARGET_DIR}/samples/"
ls -d "${TARGET_DIR}/samples/CAM_"* 2>/dev/null | while read d; do
    echo "  $(basename "$d"): $(ls "$d" | wc -l) files"
done
echo "=========================================="
