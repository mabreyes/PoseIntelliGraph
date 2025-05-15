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
VERBOSE=false
DRY_RUN=false
CHECK_MISSING=false

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
      --verbose)
        VERBOSE=true
        shift
        ;;
      --dry-run)
        DRY_RUN=true
        shift
        ;;
      --check-missing)
        CHECK_MISSING=true
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
        echo "  --verbose               Show detailed information about processing decisions"
        echo "  --dry-run               Check which files would be processed without actually processing them"
        echo "  --check-missing         Check for missing or incomplete processing results"
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

# Function to log detailed information in verbose mode
log_verbose() {
    if [ "$VERBOSE" = true ]; then
        echo "  [VERBOSE] $1"
    fi
}

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
  # Use -maxdepth 1 to prevent searching in subdirectories
  COUNT=$(find "$VIDEO_DIR" -maxdepth 1 -name "$EXT" 2>/dev/null | wc -l)
  TOTAL_VIDEOS=$((TOTAL_VIDEOS + COUNT))
done

echo "Found $TOTAL_VIDEOS videos to process in $VIDEO_DIR (excluding subdirectories)"
if [ $TOTAL_VIDEOS -eq 0 ]; then
  echo "No videos found. Please check the directory path."
  exit 1
fi

if [ "$DRY_RUN" = true ]; then
  echo "Running in dry-run mode - no actual processing will occur"
  echo "List of videos that would be processed:"
fi

echo "Processing will start in 3 seconds..."
sleep 3

# Count for progress tracking
CURRENT=0
SKIPPED=0
WOULD_PROCESS=0

# Arrays to store file lists
SKIPPED_FILES=()
PROCESSED_FILES=()
WOULD_PROCESS_FILES=()
MISSING_FILES=()

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
  # Use -maxdepth 1 to prevent searching in subdirectories
  for VIDEO in $(find "$VIDEO_DIR" -maxdepth 1 -name "$EXT" 2>/dev/null); do
    # Check if file exists (handles case when no files match pattern)
    if [ -f "$VIDEO" ]; then
      CURRENT=$((CURRENT + 1))
      BASENAME=$(basename "$VIDEO")
      FILENAME="${BASENAME%.*}"
      EXTENSION="${BASENAME##*.}"

      # Define expected output files - preserve original extension
      VIS_VIDEO="$OUTPUT_DIR/${FILENAME}.${EXTENSION}"
      JSON_FILE="$OUTPUT_DIR/results_${FILENAME}.json"

      # Verbose logging of paths being checked
      log_verbose "Checking for processed files:"
      log_verbose "  Video: $VIS_VIDEO (exists: $([ -f "$VIS_VIDEO" ] && echo "YES" || echo "NO"))"
      log_verbose "  JSON: $JSON_FILE (exists: $([ -f "$JSON_FILE" ] && echo "YES" || echo "NO"))"

      # Check if both output files exist and we're not forcing reprocessing
      if [ -f "$VIS_VIDEO" ] && [ -f "$JSON_FILE" ] && [ "$FORCE_REPROCESS" = false ]; then
        echo "[$CURRENT/$TOTAL_VIDEOS] Skipping: $BASENAME (already processed)"
        log_verbose "  - Both output files exist and force mode is off"
        SKIPPED=$((SKIPPED + 1))
        SKIPPED_FILES+=("$BASENAME")
        continue
      fi

      # If we're checking for missing files, add files with incomplete processing to the list
      if [ "$CHECK_MISSING" = true ] || [ "$DRY_RUN" = true ]; then
        if [[ -f "$VIS_VIDEO" && ! -f "$JSON_FILE" ]]; then
          MISSING_FILES+=("$BASENAME (incomplete - missing JSON)")
        elif [[ ! -f "$VIS_VIDEO" && -f "$JSON_FILE" ]]; then
          MISSING_FILES+=("$BASENAME (incomplete - missing video)")
        elif [[ ! -f "$VIS_VIDEO" && ! -f "$JSON_FILE" ]]; then
          MISSING_FILES+=("$BASENAME (not processed)")
        fi
      fi

      # Log detailed reason for processing
      if [ ! -f "$VIS_VIDEO" ] && [ ! -f "$JSON_FILE" ]; then
        log_verbose "  - Processing because both output files are missing"
      elif [ ! -f "$VIS_VIDEO" ]; then
        log_verbose "  - Processing because processed video file is missing"
      elif [ ! -f "$JSON_FILE" ]; then
        log_verbose "  - Processing because JSON prediction file is missing"
      elif [ "$FORCE_REPROCESS" = true ]; then
        log_verbose "  - Processing because force mode is enabled"
      fi

      echo "[$CURRENT/$TOTAL_VIDEOS] Processing: $BASENAME"

      # In dry run mode, just count what would be processed and add to list
      if [ "$DRY_RUN" = true ]; then
        WOULD_PROCESS=$((WOULD_PROCESS + 1))
        WOULD_PROCESS_FILES+=("$BASENAME")
        continue
      fi

      # Run pose estimation command for this video
      python -W ignore mmpose/demo/topdown_demo_with_mmdet.py \
        "$CONFIG_DET" \
        "$CHECKPOINT_DET" \
        "$CONFIG_POSE" \
        "$CHECKPOINT_POSE" \
        --input "$VIDEO" \
        --output-root "$OUTPUT_DIR" \
        $CMD_OPTIONS

      PROCESSED_FILES+=("$BASENAME")
      echo "Completed processing $BASENAME"
      echo "----------------------------------------------------------------"
    fi
  done
done

if [ "$DRY_RUN" = true ]; then
  echo "Dry run complete."
  echo "Total videos: $TOTAL_VIDEOS"
  echo "Would process: $WOULD_PROCESS"
  echo "Would skip: $SKIPPED"

  if [ ${#WOULD_PROCESS_FILES[@]} -gt 0 ]; then
    echo ""
    echo "List of videos that would be processed:"
    for VIDEO in "${WOULD_PROCESS_FILES[@]}"; do
      echo "  - $VIDEO"
    done
  fi

  if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo ""
    echo "List of videos with missing or incomplete processing:"
    for VIDEO in "${MISSING_FILES[@]}"; do
      echo "  - $VIDEO"
    done
  fi
else
  echo "All videos processed. Results are in $OUTPUT_DIR directory."
  echo "Total videos: $TOTAL_VIDEOS, Processed: $((CURRENT - SKIPPED)), Skipped: $SKIPPED"

  if [ ${#PROCESSED_FILES[@]} -gt 0 ]; then
    echo ""
    echo "List of videos processed in this session:"
    for VIDEO in "${PROCESSED_FILES[@]}"; do
      echo "  - $VIDEO"
    done
  fi
fi

# Verify skipping logic by checking for any inconsistencies
if [ "$VERBOSE" = true ] || [ "$CHECK_MISSING" = true ]; then
  echo ""
  echo "Missing Videos Report:"
  echo "------------------------------"
  INCOMPLETE=0
  INCOMPLETE_FILES=()

  for EXT in "${VIDEO_EXTENSIONS[@]}"; do
    for VIDEO in $(find "$VIDEO_DIR" -maxdepth 1 -name "$EXT" 2>/dev/null); do
      if [ -f "$VIDEO" ]; then
        BASENAME=$(basename "$VIDEO")
        FILENAME="${BASENAME%.*}"
        EXTENSION="${BASENAME##*.}"

        VIS_VIDEO="$OUTPUT_DIR/${FILENAME}.${EXTENSION}"
        JSON_FILE="$OUTPUT_DIR/results_${FILENAME}.json"

        # Check for partial processing (one file exists but not the other)
        if [ -f "$VIS_VIDEO" ] && [ ! -f "$JSON_FILE" ]; then
          echo "  WARNING: Incomplete processing detected for $BASENAME - Video exists but JSON missing"
          INCOMPLETE=$((INCOMPLETE + 1))
          INCOMPLETE_FILES+=("$BASENAME (Video exists but JSON missing)")
        elif [ ! -f "$VIS_VIDEO" ] && [ -f "$JSON_FILE" ]; then
          echo "  WARNING: Incomplete processing detected for $BASENAME - JSON exists but Video missing"
          INCOMPLETE=$((INCOMPLETE + 1))
          INCOMPLETE_FILES+=("$BASENAME (JSON exists but Video missing)")
        fi
      fi
    done
  done

  if [ $INCOMPLETE -eq 0 ]; then
    echo "  No inconsistencies found. Skip logic is working correctly."
  else
    echo "  Found $INCOMPLETE videos with inconsistent processing status."
    echo "  Consider using --force to reprocess these files."

    echo ""
    echo "  List of incompletely processed videos:"
    for VIDEO in "${INCOMPLETE_FILES[@]}"; do
      echo "    - $VIDEO"
    done
  fi
fi

# Check for specific videos mentioned by the user
if [ "$CHECK_MISSING" = true ] || [ "$DRY_RUN" = true ]; then
  echo ""
  echo "Checking specific videos of interest..."

  # List of specific videos to check
  VIDEOS_TO_CHECK=("71.mp4" "98.mp4")

  for VIDEO_NAME in "${VIDEOS_TO_CHECK[@]}"; do
    VIDEO_PATH="$VIDEO_DIR/$VIDEO_NAME"
    if [ -f "$VIDEO_PATH" ]; then
      FILENAME="${VIDEO_NAME%.*}"
      EXTENSION="${VIDEO_NAME##*.}"
      VIS_VIDEO="$OUTPUT_DIR/${FILENAME}.${EXTENSION}"
      JSON_FILE="$OUTPUT_DIR/results_${FILENAME}.json"

      echo "  Checking $VIDEO_NAME:"
      echo "    - Original video exists: YES"
      echo "    - Processed video exists: $([ -f "$VIS_VIDEO" ] && echo "YES" || echo "NO")"
      echo "    - JSON results exist: $([ -f "$JSON_FILE" ] && echo "YES" || echo "NO")"

      if [[ ! -f "$VIS_VIDEO" || ! -f "$JSON_FILE" ]]; then
        echo "    - Status: NEEDS PROCESSING"
      else
        echo "    - Status: FULLY PROCESSED"
      fi
    else
      echo "  Checking $VIDEO_NAME: Original video not found in $VIDEO_DIR"
    fi
  done
fi
