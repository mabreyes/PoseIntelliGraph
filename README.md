# Video Pose Estimation

A Pythonic application that performs pose estimation on videos using OpenPose, following the Single Responsibility Principle.

## Features

- Process video files or live camera feed
- Detect human poses using OpenPose
- Visualize pose keypoints and connections
- Save processed video with pose estimation overlay
- Modular architecture following Single Responsibility Principle
- Type annotations and comprehensive documentation
- Code quality ensured with pre-commit hooks

## Project Structure

```
pose-estimation/
├── main.py                      # Simple entry point
├── pose_estimation/             # Package directory
│   ├── __init__.py              # Package initialization
│   ├── cli.py                   # Command-line interface
│   ├── models/                  # Model-related modules
│   │   ├── __init__.py
│   │   └── openpose.py          # OpenPose model handling
│   ├── processors/              # Processing modules
│   │   ├── __init__.py
│   │   ├── frame_processor.py   # Frame processing
│   │   └── video_processor.py   # Video processing
│   └── utils/                   # Utility functions
│       └── __init__.py
├── pyproject.toml               # Project metadata and dependencies
├── requirements.lock            # Locked dependencies
└── Makefile                     # Project automation
```

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
   make install
   ```

   Or manually:
   ```
   uv pip install -e .
   ```

3. Download the required models:
   ```
   make prepare-models
   ```

## Usage

### Command Line

Run the application from the command line:

```
python main.py [input] [options]
```

Or after installation:

```
pose-estimation [input] [options]
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

### Using the Makefile

The project includes a Makefile for common tasks:

```bash
# Setup environment and install dependencies
make setup

# Install dependencies with uv
make install

# Download models
make prepare-models

# Download sample videos
make prepare-samples

# Run on sample video
make run-sample

# Run on video with multiple people
make run-multi

# Use webcam
make run-webcam

# Clean generated files
make clean

# Format code
make format

# Install pre-commit hooks
make pre-commit-install

# Run pre-commit hooks on staged files
make pre-commit

# Run pre-commit hooks on all files
make pre-commit-all
```

## How It Works

The application uses a modular architecture:

1. **OpenPoseModel** - Loads the model and handles keypoint detection
2. **FrameProcessor** - Processes individual frames for pose estimation
3. **VideoProcessor** - Manages video sources and outputs
4. **CLI** - Provides the command-line interface

The application follows the Single Responsibility Principle, with each class having a single, well-defined responsibility.

## Development

### Code Quality Tools

This project uses several code quality tools:

- **black**: Code formatting
- **isort**: Import sorting
- **ruff**: Fast linting
- **mypy**: Type checking
- **pre-commit**: Automated checks before commits

To install the pre-commit hooks:

```bash
make pre-commit-install
```

## License

MIT
