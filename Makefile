.PHONY: setup install run-sample run-webcam run-directory clean format help pre-commit pre-commit-install pre-commit-all

PYTHON = python3
UV = uv
MODEL_PATH = models
VENV = venv
SAMPLE_VIDEO = samples/sample_video.mp4
SAMPLE_OUTPUT = samples/output_video.mp4
MULTI_PEOPLE_VIDEO = samples/multiple_people.mp4
MULTI_PEOPLE_OUTPUT = samples/output_multiple_people.mp4
SAMPLE_DIR = samples
SAMPLE_DIR_OUTPUT = samples_openpose

help:
	@echo "Pose Estimation Project Makefile"
	@echo ""
	@echo "Available commands:"
	@echo "  make setup           - Create virtual environment and install dependencies using uv"
	@echo "  make install         - Install dependencies using uv"
	@echo "  make prepare-models  - Download required models"
	@echo "  make run-sample      - Run pose estimation on sample video"
	@echo "  make run-multi       - Run pose estimation on multiple people video"
	@echo "  make run-directory   - Run pose estimation on all videos in a directory"
	@echo "  make run-webcam      - Run pose estimation on webcam feed"
	@echo "  make clean           - Remove generated files and cache"
	@echo "  make format          - Format code with black (if installed)"
	@echo "  make pre-commit-install - Install pre-commit hooks"
	@echo "  make pre-commit      - Run pre-commit hooks on staged files"
	@echo "  make pre-commit-all  - Run pre-commit hooks on all files"
	@echo "  make help            - Show this help message"

setup:
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created in ./$(VENV)/"
	@echo "Activate it with: source $(VENV)/bin/activate"
	. $(VENV)/bin/activate && $(UV) pip install -e .
	. $(VENV)/bin/activate && $(UV) pip install pre-commit black isort mypy ruff

install:
	$(UV) pip install -e .
	$(UV) pip install pre-commit black isort mypy ruff

prepare-models:
	@echo "Creating model directories..."
	mkdir -p $(MODEL_PATH)/pose/coco
	@echo "Downloading model files..."
	curl -L -o $(MODEL_PATH)/pose/coco/pose_deploy_linevec.prototxt "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt"
	curl -L -o $(MODEL_PATH)/pose/coco/pose_iter_440000.caffemodel "https://www.dropbox.com/s/2h2bv29a130sgrk/pose_iter_440000.caffemodel?dl=1"
	@echo "Model files downloaded successfully!"

prepare-samples:
	@echo "Downloading sample videos..."
	mkdir -p samples
	curl -L -o $(SAMPLE_VIDEO) "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_1MB.mp4"
	curl -L -o $(MULTI_PEOPLE_VIDEO) "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4"
	@echo "Sample videos downloaded successfully!"

run-sample:
	$(PYTHON) main.py $(SAMPLE_VIDEO) -o $(SAMPLE_OUTPUT)

run-multi:
	$(PYTHON) main.py $(MULTI_PEOPLE_VIDEO) -o $(MULTI_PEOPLE_OUTPUT)

run-directory:
	$(PYTHON) main.py $(SAMPLE_DIR)

run-webcam:
	$(PYTHON) main.py camera

clean:
	@echo "Cleaning generated files..."
	rm -rf samples/output_*.mp4
	rm -rf $(SAMPLE_DIR_OUTPUT)
	rm -rf __pycache__/
	rm -rf *.egg-info/
	rm -rf build/ dist/
	@echo "Cleaned generated files!"

clean-all: clean
	@echo "Cleaning models and virtual environment..."
	rm -rf $(MODEL_PATH)/
	rm -rf $(VENV)/
	@echo "Cleaned all generated files and dependencies!"

format:
	@echo "Formatting code with black..."
	black *.py pose_estimation
	@echo "Code formatted successfully!"

pre-commit-install:
	@echo "Installing pre-commit hooks..."
	pre-commit install
	@echo "Pre-commit hooks installed successfully!"

pre-commit:
	@echo "Running pre-commit hooks on staged files..."
	pre-commit run
	@echo "Pre-commit checks complete!"

pre-commit-all:
	@echo "Running pre-commit hooks on all files..."
	pre-commit run --all-files
	@echo "Pre-commit checks complete!"

# Default command
.DEFAULT_GOAL := help
