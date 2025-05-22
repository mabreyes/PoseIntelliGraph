#!/usr/bin/env python3
"""
Violence Detection using Graph Neural Networks.

This module implements a GNN model to detect violent behavior from human pose data
in MMPose JSON format. It includes functionality for:
- Data loading and preprocessing from MMPose JSON format
- Graph construction from pose keypoints
- Model training with metrics tracking
- Model evaluation and optimal threshold selection
- Result visualization and model persistence
"""

from __future__ import annotations

import argparse
# import json # No longer directly used in train.py
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
# confusion_matrix, f1_score, precision_recall_curve, roc_curve were for find_optimal_threshold (now in evaluation_utils)
from sklearn.metrics import roc_auc_score # Retained for train_model, evaluate_model
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data # Retained for main() -> graph.y assignment
from torch_geometric.loader import DataLoader
from tqdm import tqdm # Retained for train_model, evaluate_model loops

import visualization as viz

# Import components from separate files
from .graph_utils import load_mmpose_data # create_pose_graph is not directly used here anymore
from .evaluation_utils import find_optimal_threshold
from model import ViolenceDetectionGNN, get_device

# Import configuration values
from .config import (
    # DATA_PATH,  # Not directly used in train.py; paths are constructed from specific constants
    VIOLENT_PATH_CAM1,
    NON_VIOLENT_PATH_CAM1,
    VIOLENT_PATH_CAM2,
    NON_VIOLENT_PATH_CAM2,
    REAL_LIFE_VIOLENCE_PATH,
    REAL_LIFE_NONVIOLENCE_PATH,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    SAMPLE_PERCENTAGE,
    MODEL_HIDDEN_CHANNELS,
    MODEL_TRANSFORMER_HEADS,
    MODEL_TRANSFORMER_LAYERS,
    TEST_SPLIT_RATIO,
    VALIDATION_SPLIT_RATIO,
    RANDOM_SEED,
    MODEL_IN_CHANNELS, # Added as it's used by ViolenceDetectionGNN
)

# find_optimal_threshold function has been moved to src.evaluation_utils
# load_mmpose_data function has been moved to src.graph_utils


def train_model(
    model: ViolenceDetectionGNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 50,
) -> Dict[str, List[float]]:
    """
    Train the GNN model and track metrics.

    Implements a training loop with both training and validation phases.
    For each epoch, the model is trained on the training set and evaluated
    on the validation set. Metrics including loss and AUC are tracked.

    Args:
        model: The GNN model
        train_loader: Training data loader containing graph objects with .y attributes
        val_loader: Validation data loader containing graph objects with .y attributes
        device: Device to train on (CPU/GPU/MPS)
        optimizer: Optimizer for training
        num_epochs: Number of training epochs

    Returns:
        Dictionary of training and validation metrics
    """
    # Training metrics
    metrics: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_auc": []}

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        total_samples = 0

        # Process batches
        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"
        ):
            # Move batch to device
            batch = batch.to(device)

            optimizer.zero_grad()

            # Forward pass
            out = model(batch.x, batch.edge_index, batch.batch)
            target = batch.y.to(device)

            # Ensure shapes match for binary_cross_entropy
            if out.size() != target.size():
                target = target.view(out.size())

            # Calculate loss
            loss = F.binary_cross_entropy(out, target)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track loss
            total_loss += loss.item() * batch.num_graphs
            total_samples += batch.num_graphs

        # Calculate average training loss
        avg_train_loss = total_loss / total_samples
        metrics["train_loss"].append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        val_samples = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"
            ):
                # Move batch to device
                batch = batch.to(device)

                # Forward pass
                out = model(batch.x, batch.edge_index, batch.batch)
                target = batch.y.to(device)

                # Ensure shapes match for binary_cross_entropy
                if out.size() != target.size():
                    target = target.view(out.size())

                # Calculate loss
                loss = F.binary_cross_entropy(out, target)
                val_loss += loss.item() * batch.num_graphs
                val_samples += batch.num_graphs

                # Store predictions and targets for metrics
                all_preds.extend(out.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        # Calculate validation metrics
        avg_val_loss = val_loss / val_samples
        val_auc = roc_auc_score(all_targets, all_preds)

        metrics["val_loss"].append(avg_val_loss)
        metrics["val_auc"].append(val_auc)

        # Print epoch results
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val AUC: {val_auc:.4f}")

    return metrics


def evaluate_model(
    model: ViolenceDetectionGNN, test_loader: DataLoader, device: torch.device
) -> Tuple[float, float, float, Dict[str, float]]:
    """
    Evaluate the model on the test set.

    Performs a comprehensive evaluation of the trained model on the test set,
    calculating loss, AUC, and finding the optimal classification threshold.

    Args:
        model: The trained GNN model
        test_loader: Test data loader containing graph objects with .y attributes
        device: Device for evaluation

    Returns:
        Tuple of (auc_score, accuracy_score, optimal_threshold, threshold_metrics)
    """
    model.eval()
    test_loss = 0
    test_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Move batch to device
            batch = batch.to(device)

            # Forward pass
            out = model(batch.x, batch.edge_index, batch.batch)
            target = batch.y.to(device)

            # Ensure shapes match for binary_cross_entropy
            if out.size() != target.size():
                target = target.view(out.size())

            # Calculate loss
            loss = F.binary_cross_entropy(out, target)
            test_loss += loss.item() * batch.num_graphs
            test_samples += batch.num_graphs

            # Store predictions and targets for metrics
            all_preds.extend(out.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Convert lists to arrays for sklearn metrics
    all_preds_array = np.array(all_preds)
    all_targets_array = np.array(all_targets)

    # Calculate AUC
    test_auc = roc_auc_score(all_targets_array, all_preds_array)

    # Calculate accuracy at default threshold (0.5)
    y_pred = (all_preds_array >= 0.5).astype(int)
    accuracy = np.mean((y_pred == all_targets_array).astype(float))

    # Find optimal classification threshold
    optimal_threshold, threshold_metrics = find_optimal_threshold(
        all_targets_array, all_preds_array
    )

    return test_auc, accuracy, optimal_threshold, threshold_metrics


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the training script.

    Defines and processes command-line arguments for training configuration,
    including number of epochs, batch size, and data sampling percentage.

    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description="Train Violence Detection GRNN Model")
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,  # Now from config
        help=f"Number of training epochs (default: {NUM_EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,  # Now from config
        help=f"Training batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,  # Now from config
        help=f"Learning rate (default: {LEARNING_RATE})",
    )
    parser.add_argument(
        "--sample-percentage",
        type=int,
        default=SAMPLE_PERCENTAGE,  # Now from config
        help=f"Percentage of data to use (default: {SAMPLE_PERCENTAGE})",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=MODEL_HIDDEN_CHANNELS,  # Now from config
        help=f"Hidden channels in model (default: {MODEL_HIDDEN_CHANNELS})",
    )
    parser.add_argument(
        "--transformer-layers",
        type=int,
        default=MODEL_TRANSFORMER_LAYERS,  # Now from config
        help=f"Transformer layers in model (default: {MODEL_TRANSFORMER_LAYERS})",
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default="violence_detection_model.pt",
        help="Path for saving the model (default: violence_detection_model.pt)",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main function to execute the training pipeline.

    This function orchestrates the entire training process:
    1. Loads and preprocesses data from violent and non-violent samples
    2. Splits data into training, validation, and test sets
    3. Creates and trains the GNN model
    4. Evaluates the model and calculates optimal threshold
    5. Visualizes training metrics and model performance
    6. Saves the final model with its threshold information
    """
    # Parse command line arguments
    args = parse_arguments()

    # Apply command line arguments
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    sample_percentage = args.sample_percentage
    hidden_channels = args.hidden_channels
    transformer_layers = args.transformer_layers
    model_output_path = args.model_output

    print(f"Starting training with {num_epochs} epochs, batch size {batch_size}")
    print(f"Using {sample_percentage}% of the dataset")

    # Load data from both cameras if available
    all_graphs = []
    all_labels = []

    # Camera 1 data (always required)
    try:
        print("Loading data from Camera 1...")
        graphs_cam1, labels_cam1 = load_mmpose_data(
            VIOLENT_PATH_CAM1, NON_VIOLENT_PATH_CAM1, sample_percentage
        )
        all_graphs.extend(graphs_cam1)
        all_labels.extend(labels_cam1)
    except Exception as e:
        print(f"Error loading Camera 1 data: {e}")
        if not all_graphs:  # If no Camera 1 data and this is the first load attempt
            raise

    # Camera 2 data (optional)
    try:
        if VIOLENT_PATH_CAM2.exists() and NON_VIOLENT_PATH_CAM2.exists():
            print("Loading data from Camera 2...")
            graphs_cam2, labels_cam2 = load_mmpose_data(
                VIOLENT_PATH_CAM2, NON_VIOLENT_PATH_CAM2, sample_percentage
            )
            all_graphs.extend(graphs_cam2)
            all_labels.extend(labels_cam2)
    except Exception as e:
        print(f"Error loading Camera 2 data: {e}")
        # Continue with Camera 1 data only

    # Real Life Violence Dataset (optional)
    try:
        if REAL_LIFE_VIOLENCE_PATH.exists() and REAL_LIFE_NONVIOLENCE_PATH.exists():
            print("Loading data from Real Life Violence Dataset...")
            graphs_real_life, labels_real_life = load_mmpose_data(
                REAL_LIFE_VIOLENCE_PATH, REAL_LIFE_NONVIOLENCE_PATH, sample_percentage
            )
            all_graphs.extend(graphs_real_life)
            all_labels.extend(labels_real_life)
    except Exception as e:
        print(f"Error loading Real Life Violence Dataset: {e}")
        # Continue with existing data

    # Split data into train/val/test sets
    train_graphs, test_graphs, train_labels, test_labels = train_test_split(
        all_graphs, all_labels, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_SEED
    )

    # Further split training data to create a validation set
    train_graphs, val_graphs, train_labels, val_labels = train_test_split(
        train_graphs,
        train_labels,
        test_size=VALIDATION_SPLIT_RATIO,
        random_state=RANDOM_SEED,
    )

    # Attach labels to graph objects
    for i, graph in enumerate(train_graphs):
        graph.y = torch.tensor([train_labels[i]], dtype=torch.float)

    for i, graph in enumerate(val_graphs):
        graph.y = torch.tensor([val_labels[i]], dtype=torch.float)

    for i, graph in enumerate(test_graphs):
        graph.y = torch.tensor([test_labels[i]], dtype=torch.float)

    # Create data loaders
    train_loader = DataLoader(
        train_graphs, batch_size=batch_size, shuffle=True, follow_batch=["x"]
    )
    val_loader = DataLoader(val_graphs, batch_size=batch_size, follow_batch=["x"])
    test_loader = DataLoader(test_graphs, batch_size=batch_size, follow_batch=["x"])

    # Print dataset statistics
    print(f"Dataset loaded: {len(all_graphs)} total samples")
    print(
        f"Training: {len(train_graphs)}, Validation: {len(val_graphs)}, "
        f"Test: {len(test_graphs)}"
    )

    # Create model
    device = get_device()
    print(f"Using device: {device}")

    model = ViolenceDetectionGNN(
        in_channels=MODEL_IN_CHANNELS,  # Now from config
        hidden_channels=hidden_channels,
        transformer_heads=MODEL_TRANSFORMER_HEADS, # Now from config
        transformer_layers=transformer_layers,
    ).to(device)

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    print("Training model...")
    metrics = train_model(
        model, train_loader, val_loader, device, optimizer, num_epochs
    )

    # Evaluate on test set
    print("Evaluating model on test set...")
    auc, accuracy, threshold, threshold_metrics = evaluate_model(
        model, test_loader, device
    )

    print(f"Test AUC: {auc:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Optimal Threshold: {threshold:.4f}")
    print(f"Sensitivity: {threshold_metrics['sensitivity']:.4f}")
    print(f"Specificity: {threshold_metrics['specificity']:.4f}")
    print(f"F1 Score: {threshold_metrics['f1_score']:.4f}")

    # Save model with threshold information
    model_dict = {
        "model_state_dict": model.state_dict(),
        "threshold": threshold,
        "threshold_metrics": threshold_metrics,
        "test_auc": auc,
        "test_accuracy": accuracy,
    }
    torch.save(model_dict, model_output_path)
    print(f"Model saved to {model_output_path}")

    # Visualize training metrics
    viz.plot_training_metrics(metrics, "training_metrics.png")
    print("Training metrics visualization saved to training_metrics.png")


if __name__ == "__main__":
    main()
