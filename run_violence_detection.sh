#!/bin/bash
# Run script for violence detection model
# This script sets the necessary environment variables before running the model

# Set environment variable to enable MPS fallback to CPU for unsupported operations
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Run the model
python violence_detection_model.py

echo "Model execution completed" 