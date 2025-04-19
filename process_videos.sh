#!/bin/bash

# process_videos.sh - Run demo_video.py on all videos in a directory
# Usage: ./process_videos.sh /path/to/videos/directory [options]

# Check if directory argument is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <directory> [options]"
    echo "Options:"
    echo "  --no_hands    No hand pose detection"
    echo "  --no_body     No body pose detection"
    exit 1
fi

# Get directory argument and check if it exists
VIDEO_DIR="$1"
shift # Remove directory argument, leaving only options

if [ ! -d "$VIDEO_DIR" ]; then
    echo "Error: '$VIDEO_DIR' is not a directory"
    exit 1
fi

# Get additional options to pass to demo_video.py
OPTIONS="$@"

# Video file extensions to process
VIDEO_EXTENSIONS=("mp4" "avi" "mov" "mkv" "webm" "flv")

# Find video files in the directory
echo "Searching for videos in: $VIDEO_DIR"
VIDEO_FILES=()

for ext in "${VIDEO_EXTENSIONS[@]}"; do
    # Find files with this extension and add to array
    while IFS= read -r file; do
        if [ -f "$file" ]; then
            VIDEO_FILES+=("$file")
        fi
    done < <(find "$VIDEO_DIR" -type f -name "*.${ext}" -o -name "*.${ext^^}")
done

# Check if any videos were found
TOTAL_VIDEOS=${#VIDEO_FILES[@]}
if [ $TOTAL_VIDEOS -eq 0 ]; then
    echo "No video files found in $VIDEO_DIR"
    exit 1
fi

echo "Found $TOTAL_VIDEOS video files"

# Process each video
COUNTER=1
for video in "${VIDEO_FILES[@]}"; do
    echo "[$COUNTER/$TOTAL_VIDEOS] Processing: $(basename "$video")"
    python demo_video.py "$video" $OPTIONS

    # Check if processing was successful
    if [ $? -ne 0 ]; then
        echo "Warning: Error processing $video"
    fi

    COUNTER=$((COUNTER+1))
    echo "-----------------------------------"
done

echo "All videos processed!"
