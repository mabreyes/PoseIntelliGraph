"""
Configuration file for the Violence Detection GNN project.

This file centralizes all global constants and configuration values
for data paths, training hyperparameters, model parameters, and
evaluation/inference settings.
"""

from pathlib import Path
import torch

# ------------------
# Data Configuration
# ------------------
# Base data path conditional on GPU availability
if torch.cuda.is_available():
    # GPU detected paths
    DATA_PATH = Path("./json")
    # Real Life Violence Dataset paths (assuming relative to project root or a specific data directory)
    REAL_LIFE_VIOLENCE_PATH = Path("Real_Life_Violence_Dataset/Violence/processed")
    REAL_LIFE_NONVIOLENCE_PATH = Path(
        "Real_Life_Violence_Dataset/NonViolence/processed"
    )
else:
    # Local paths (no GPU) - Adjust these paths as per your local setup
    DATA_PATH = Path("/Volumes/MARCREYES/violence-detection-dataset")
    # Real Life Violence Dataset paths (local)
    REAL_LIFE_VIOLENCE_PATH = Path(
        "/Volumes/MARCREYES/archive/Real_Life_Violence_Dataset/processed/violent/Real_Life_Violence_Dataset/Violence/processed"
    )
    REAL_LIFE_NONVIOLENCE_PATH = Path(
        "/Volumes/MARCREYES/archive/Real_Life_Violence_Dataset/processed/nonviolent/Real_Life_Violence_Dataset/NonViolence/processed"
    )

# Specific dataset paths (derived from DATA_PATH)
VIOLENT_PATH_CAM1 = DATA_PATH / "violent/cam1/processed"  # Adjusted based on train.py local paths
NON_VIOLENT_PATH_CAM1 = DATA_PATH / "non-violent/cam1/processed"  # Adjusted
VIOLENT_PATH_CAM2 = DATA_PATH / "violent/cam2/processed"  # Adjusted
NON_VIOLENT_PATH_CAM2 = DATA_PATH / "non-violent/cam2/processed"  # Adjusted


# -------------------------
# Training Hyperparameters
# -------------------------
BATCH_SIZE = 32
NUM_EPOCHS = 1  # Default, can be overridden by command line args
LEARNING_RATE = 0.001  # Default, can be overridden by command line args
SAMPLE_PERCENTAGE = 100  # Percentage of data to use (1-100).
# The original train.py used SAMPLE_PERCENTAGE = 1.
# This will be overridden by argparse in train.py if specified.

# ------------------
# Model Parameters
# ------------------
MODEL_IN_CHANNELS = 2  # Input channels for the GNN (e.g., x, y coordinates)
MODEL_HIDDEN_CHANNELS = 64  # Default, can be overridden by command line args in train.py
MODEL_TRANSFORMER_HEADS = 4
MODEL_TRANSFORMER_LAYERS = 2  # Default, can be overridden by command line args in train.py

# ---------------------
# Evaluation Constants
# ---------------------
TEST_SPLIT_RATIO = 0.2
VALIDATION_SPLIT_RATIO = 0.25  # Relative to the training set after initial test split
RANDOM_SEED = 42

# ---------------------
# Inference Constants
# ---------------------
DEFAULT_MODEL_PATH = "violence_detection_model.pt"
DEFAULT_OUTPUT_PATH = "violence_scores.json"
THRESHOLD_MARGIN = 0.2  # Margin for interpretation confidence levels (e.g., "Likely violent")

# Note:
# Some constants (NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, MODEL_HIDDEN_CHANNELS,
# MODEL_TRANSFORMER_LAYERS, SAMPLE_PERCENTAGE) are defined as defaults for
# argparse in train.py. They are included here for centralization, but command-line
# arguments in train.py will take precedence during training.
#
# SAMPLE_PERCENTAGE in train.py was 1; here it's 100 as a general default.
# This value is overridden by argparse in train.py if specified there.
#
# Path definitions in train.py for cam1/cam2 when GPU is available were
# slightly different (e.g., DATA_PATH / "violent/cam1" vs
# DATA_PATH / "violent/cam1/processed"). The "processed" version is used here
# for consistency with local paths. This might need adjustment if the directory
# structure is different on GPU-enabled machines.
#
# MODEL_IN_CHANNELS was previously only in inference.py; added for completeness.
# MODEL_HIDDEN_CHANNELS, MODEL_TRANSFORMER_HEADS, MODEL_TRANSFORMER_LAYERS
# were in both train.py and inference.py; values were consistent.
# DEFAULT_MODEL_PATH, DEFAULT_OUTPUT_PATH, THRESHOLD_MARGIN were only in inference.py.
# TEST_SPLIT_RATIO, VALIDATION_SPLIT_RATIO, RANDOM_SEED were only in train.py.
# Data path logic (DATA_PATH, specific camera paths, REAL_LIFE_VIOLENCE_PATH, etc.)
# from train.py has been replicated here.
# BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE from train.py are replicated.
