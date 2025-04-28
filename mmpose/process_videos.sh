#!/bin/bash

# Unified script to process videos with pose estimation
# Simple usage: ./process_videos.sh [directory_of_videos]
# Advanced usage: ./process_videos.sh --input-dir [dir] [other options]

# Default values
VIDEO_DIR="./videos"
OUTPUT_DIR="vis_results"
CONFIG_DET="mmpose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py"
CHECKPOINT_DET="https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
CONFIG_POSE="mmpose/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py"
CHECKPOINT_POSE="https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"
SHOW=false
SAVE_PREDICTIONS=true
DRAW_HEATMAP=true
FORCE_REPROCESS=false

# Simple mode: If only one parameter and it's not an option, assume it's the input directory
if [ $# -eq 1 ] && [[ ! "$1" == --* ]]; then
  VIDEO_DIR="$1"
  echo "Simple mode: Processing videos in $VIDEO_DIR"
else
  # Parse arguments for advanced mode
  while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
      --input-dir)
        VIDEO_DIR="$2"
        shift 2
        ;;
      --output-dir)
        OUTPUT_DIR="$2"
        shift 2
        ;;
      --show)
        SHOW=true
        shift
        ;;
      --no-save)
        SAVE_PREDICTIONS=false
        shift
        ;;
      --no-heatmap)
        DRAW_HEATMAP=false
        shift
        ;;
      --force)
        FORCE_REPROCESS=true
        shift
        ;;
      --help|-h)
        echo "Usage:"
        echo "  Simple: ./process_videos.sh [directory_of_videos]"
        echo "  Advanced: ./process_videos.sh [options]"
        echo ""
        echo "Options:"
        echo "  --input-dir DIR         Directory containing videos (default: ./videos)"
        echo "  --output-dir DIR        Output directory for results (default: vis_results)"
        echo "  --show                  Display visualization during processing"
        echo "  --no-save               Don't save prediction results"
        echo "  --no-heatmap            Don't draw heatmap in visualization"
        echo "  --force                 Force reprocessing even if output files exist"
        echo "  --help,-h               Show this help message"
        exit 0
        ;;
      *)
        echo "Unknown option: $key"
        echo "Use --help for usage information"
        exit 1
        ;;
    esac
  done
fi

# Check if directory exists
if [ ! -d "$VIDEO_DIR" ]; then
  echo "Error: Directory $VIDEO_DIR does not exist"
  exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Process video extensions
VIDEO_EXTENSIONS=("*.mp4" "*.avi" "*.mov" "*.mkv" "*.webm")

# Get total number of videos
TOTAL_VIDEOS=0
for EXT in "${VIDEO_EXTENSIONS[@]}"; do
  COUNT=$(find "$VIDEO_DIR" -name "$EXT" 2>/dev/null | wc -l)
  TOTAL_VIDEOS=$((TOTAL_VIDEOS + COUNT))
done

echo "Found $TOTAL_VIDEOS videos to process in $VIDEO_DIR"
if [ $TOTAL_VIDEOS -eq 0 ]; then
  echo "No videos found. Please check the directory path."
  exit 1
fi

echo "Processing will start in 3 seconds..."
sleep 3

# Count for progress tracking
CURRENT=0
SKIPPED=0

# Build command options
CMD_OPTIONS=""
if [ "$SHOW" = true ]; then
  CMD_OPTIONS="$CMD_OPTIONS --show"
fi
if [ "$SAVE_PREDICTIONS" = true ]; then
  CMD_OPTIONS="$CMD_OPTIONS --save-predictions"
fi
if [ "$DRAW_HEATMAP" = true ]; then
  CMD_OPTIONS="$CMD_OPTIONS --draw-heatmap"
fi

# Loop through all videos in directory
for EXT in "${VIDEO_EXTENSIONS[@]}"; do
  for VIDEO in "$VIDEO_DIR"/$EXT; do
    # Check if file exists (handles case when no files match pattern)
    if [ -f "$VIDEO" ]; then
      CURRENT=$((CURRENT + 1))
      BASENAME=$(basename "$VIDEO")
      FILENAME="${BASENAME%.*}"

      # Define expected output files
      VIS_VIDEO="$OUTPUT_DIR/${FILENAME}.mp4"
      JSON_FILE="$OUTPUT_DIR/results_${FILENAME}.json"

      # Check if both output files exist and we're not forcing reprocessing
      if [ -f "$VIS_VIDEO" ] && [ -f "$JSON_FILE" ] && [ "$FORCE_REPROCESS" = false ]; then
        echo "[$CURRENT/$TOTAL_VIDEOS] Skipping: $BASENAME (already processed)"
        SKIPPED=$((SKIPPED + 1))
        continue
      fi

      echo "[$CURRENT/$TOTAL_VIDEOS] Processing: $BASENAME"

      # Run pose estimation command for this video
      python mmpose/demo/topdown_demo_with_mmdet.py \
        "$CONFIG_DET" \
        "$CHECKPOINT_DET" \
        "$CONFIG_POSE" \
        "$CHECKPOINT_POSE" \
        --input "$VIDEO" \
        --output-root "$OUTPUT_DIR" \
        $CMD_OPTIONS

      echo "Completed processing $BASENAME"
      echo "----------------------------------------------------------------"
    fi
  done
done

echo "All videos processed. Results are in $OUTPUT_DIR directory."
echo "Total videos: $TOTAL_VIDEOS, Processed: $((CURRENT - SKIPPED)), Skipped: $SKIPPED"
