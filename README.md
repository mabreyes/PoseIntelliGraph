# Video Pose Estimation

A Python application that performs pose estimation on videos using OpenPose.

## Features

- Process video files or live camera feed
- Detect human poses using OpenPose
- Visualize pose keypoints and connections
- Save processed video with pose estimation overlay

## Requirements

This project uses `uv` from Astral.sh for dependency management instead of pip.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/pose-estimation.git
   cd pose-estimation
   ```

2. Install dependencies using uv:
   ```
   uv pip install -e .
   ```

   This will install the required packages:
   - opencv-python
   - numpy
   - opencv-contrib-python

## Usage

Run the application from the command line:

```
python main.py [input] [options]
```

### Arguments

- `input`: Path to input video file or "camera" to use webcam

### Options

- `-o, --output`: Path to output video file
- `-m, --model-path`: Path to OpenPose model directory
- `-nd, --no-display`: Don't display output while processing

### Examples

Process a video file and save the result:
```
python main.py path/to/video.mp4 -o output.mp4
```

Use webcam as input:
```
python main.py camera
```

## How It Works

The application uses OpenPose, which is a real-time multi-person keypoint detection library. The program:

1. Loads the OpenPose models (downloading them if necessary)
2. Processes each frame of the video to detect human poses
3. Draws the detected keypoints and connections on each frame
4. Displays the processed frames and/or saves to an output video

## License

MIT
