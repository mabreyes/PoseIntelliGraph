#!/bin/bash

# process_videos.sh - Run demo_video.py on all videos in a directory
# Usage: ./process_videos.sh /path/to/videos/directory [options]

# Exit on error
set -e

# Check if directory argument is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <directory> [options]"
    echo "Options:"
    echo "  --no_hands       No hand pose detection"
    echo "  --no_body        No body pose detection"
    echo "  --percent=N      Process only N% of videos (default: 100%)"
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
    ext="${filename##*.}"
    ext_lower=$(echo "$ext" | tr '[:upper:]' '[:lower:]')

    # Check if it's in our list of video extensions
    for valid_ext in "${VIDEO_EXTENSIONS[@]}"; do
        if [ "$ext_lower" = "$valid_ext" ]; then
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
