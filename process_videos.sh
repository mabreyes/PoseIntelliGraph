#!/bin/bash

# process_videos.sh - Run demo_video.py on all videos in a directory
# Usage: ./process_videos.sh /path/to/videos/directory [options]

# Exit on error
set -e

# Function to check if a video file is corrupted
check_video_corruption() {
    local video_file="$1"
    # Try to get video information using ffprobe
    if ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=nw=1:nk=1 "$video_file" >/dev/null 2>&1; then
        return 0 # File is not corrupted
    else
        return 1 # File is corrupted
    fi
}

# Function to check if file size is reasonable
check_file_size() {
    local original_file="$1"
    local processed_file="$2"
    local min_ratio=0.1 # Processed file should be at least 10% of the original size

    # Get file sizes in bytes
    local original_size=$(stat -f%z "$original_file" 2>/dev/null || stat --format=%s "$original_file")
    local processed_size=$(stat -f%z "$processed_file" 2>/dev/null || stat --format=%s "$processed_file")

    # Calculate ratio (processed/original)
    local ratio=$(echo "scale=2; $processed_size / $original_size" | bc -l)

    # Check if ratio is at least min_ratio
    if (( $(echo "$ratio < $min_ratio" | bc -l) )); then
        return 1 # Size is not reasonable
    else
        return 0 # Size is reasonable
    fi
}

# Check if directory argument is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <directory> [options]"
    echo "Options:"
    echo "  --no_hands       No hand pose detection"
    echo "  --no_body        No body pose detection"
    echo "  --percent=N      Process only N% of videos (default: 100%)"
    echo "  --process-all    Process all videos, including ones already processed (default: skip existing)"
    exit 1
fi

# Get directory argument and check if it exists
VIDEO_DIR="$1"
shift # Remove directory argument, leaving only options

if [ ! -d "$VIDEO_DIR" ]; then
    echo "Error: '$VIDEO_DIR' is not a directory"
    exit 1
fi

# Default values
PERCENT=100
SKIP_EXISTING=1  # Default is to skip existing processed videos

# Parse special options for this script
DEMO_OPTIONS=""
for arg in "$@"; do
    if [[ $arg == --percent=* ]]; then
        PERCENT="${arg#*=}"
        # Validate percentage is a number between 1 and 100
        if ! [[ "$PERCENT" =~ ^[0-9]+$ ]] || [ "$PERCENT" -lt 1 ] || [ "$PERCENT" -gt 100 ]; then
            echo "Error: Percentage must be a number between 1 and 100"
            exit 1
        fi
    elif [[ $arg == --process-all ]]; then
        SKIP_EXISTING=0
    elif [[ $arg == --skip-existing ]]; then
        echo "Note: --skip-existing is now the default behavior. Use --process-all to process all videos."
        SKIP_EXISTING=1
    else
        DEMO_OPTIONS="$DEMO_OPTIONS $arg"
    fi
done

# Video file extensions to process
VIDEO_EXTENSIONS=("mp4" "avi" "mov" "mkv" "webm" "flv")

# Find video files in the directory
echo "Searching for videos in: $VIDEO_DIR"

# Create a temporary file for storing video paths
TMPFILE=$(mktemp)
trap 'rm -f "$TMPFILE"' EXIT  # Auto-remove temp file on exit

# Find all files and filter by extension
find "$VIDEO_DIR" -type f | sort | while read -r file; do
    # Get the file extension (lowercase)
    filename=$(basename "$file")
    base_name="${filename%.*}"
    ext="${filename##*.}"
    ext_lower=$(echo "$ext" | tr '[:upper:]' '[:lower:]')

    # Check if it's in our list of video extensions
    for valid_ext in "${VIDEO_EXTENSIONS[@]}"; do
        if [ "$ext_lower" = "$valid_ext" ]; then
            # Check if we should skip existing processed videos
            if [ "$SKIP_EXISTING" -eq 1 ]; then
                # Check for any processed file with this base name
                processed_files=()
                while IFS= read -r processed_file; do
                    processed_files+=("$processed_file")
                done < <(find "$(dirname "$file")" -name "$(basename "${file%.*}").processed.*" -type f)

                if [ ${#processed_files[@]} -gt 0 ]; then
                    # Take the first processed file found
                    processed_file="${processed_files[0]}"

                    # Check if the processed file is corrupted
                    if ! check_video_corruption "$processed_file"; then
                        echo "Processed file exists but is corrupted: $processed_file - will reprocess"
                    # Check if the file size is reasonable
                    elif ! check_file_size "$file" "$processed_file"; then
                        echo "Processed file exists but has unreasonable size: $processed_file - will reprocess"
                    else
                        echo "Skipping $file (already processed: $processed_file)"
                        continue 2  # Skip to next file in outer loop
                    fi
                fi
            fi
            echo "$file" >> "$TMPFILE"
            break
        fi
    done
done

# Check if any videos were found
if [ ! -s "$TMPFILE" ]; then
    echo "No video files found in $VIDEO_DIR"
    exit 1
fi

# Count total videos
TOTAL_VIDEOS=$(wc -l < "$TMPFILE" | tr -d ' ')
echo "Found $TOTAL_VIDEOS video files"

# Calculate how many videos to process based on percentage
VIDEOS_TO_PROCESS=$(( TOTAL_VIDEOS * PERCENT / 100 ))
if [ "$VIDEOS_TO_PROCESS" -lt 1 ]; then
    VIDEOS_TO_PROCESS=1
fi

echo "Will process $VIDEOS_TO_PROCESS videos ($PERCENT% of total)"

# Process each video
COUNTER=1
while IFS= read -r video && [ $COUNTER -le $VIDEOS_TO_PROCESS ]; do
    echo "[$COUNTER/$VIDEOS_TO_PROCESS] Processing: $(basename "$video")"
    if ! python demo_video.py "$video" $DEMO_OPTIONS; then
        echo "Warning: Error processing $video"
        read -p "Continue with next video? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Stopping processing at user request."
            exit 1
        fi
    fi

    COUNTER=$((COUNTER+1))
    echo "-----------------------------------"
done < "$TMPFILE"

echo "All videos processed!"
