#!/usr/bin/env python3
"""
Violence Detection using Graph Neural Networks.

This module implements a GNN model to detect violent behavior from human pose data
in MMPose JSON format. It includes functionality for graph construction, model
training, and evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Import components from separate files
from gnn import PoseGNN, create_pose_graph
from transformer import TransformerEncoder

# Configuration constants
# Use Path objects for better path handling
DATA_PATH = Path("/Volumes/MARCREYES/violence-detection-dataset")
VIOLENT_PATH = DATA_PATH / "violent/cam1/processed"
NON_VIOLENT_PATH = DATA_PATH / "non-violent/cam1/processed"

# Training hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001


class ViolenceDetectionGNN(nn.Module):
    """
    Full model architecture for violence detection from pose data.

    This model processes pose keypoints using a pipeline of:
    1. Graph Neural Network to process pose graph structure
    2. Transformer to capture contextual patterns
    3. Classifier to produce violence score
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
    ):
        """
        Initialize the full model.

        Args:
            in_channels: Number of input features per node
                         (typically 2 for x,y coordinates)
            hidden_channels: Size of hidden representations
            transformer_heads: Number of attention heads in transformer
            transformer_layers: Number of transformer layers
        """
        super(ViolenceDetectionGNN, self).__init__()

        # GNN component
        self.gnn = PoseGNN(in_channels, hidden_channels)

        # Transformer component
        self.transformer = TransformerEncoder(
            input_dim=hidden_channels,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            output_dim=hidden_channels,
        )

        # Final prediction layers
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = nn.Linear(hidden_channels // 2, 1)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the full model.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment for nodes [num_nodes]

        Returns:
            Violence score between 0 and 1 [batch_size, 1]
        """
        # Process through GNN to get graph embeddings
        x = self.gnn(x, edge_index, batch)

        # Process through transformer to capture contextual patterns
        x = self.transformer(x)

        # Final predictions
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin2(x)

        # Output violence score between 0 and 1
        return torch.sigmoid(x)


def load_mmpose_data(
    violent_path: Path, non_violent_path: Path, sample_percentage: int = 100
) -> Tuple[List[Data], List[float]]:
    """
    Load MMPose JSON files and convert them to graph data.

    Args:
        violent_path: Path to violent pose JSON files
        non_violent_path: Path to non-violent pose JSON files
        sample_percentage: Percentage of files to process (1-100)

    Returns:
        Tuple of (list of graph Data objects, list of corresponding labels)
    """
    # Validate sample percentage
    if not 1 <= sample_percentage <= 100:
        raise ValueError("sample_percentage must be between 1 and 100")

    all_graphs = []
    all_labels = []

    # Get all JSON files from the violent directory
    violent_files = list(violent_path.glob("*.json"))
    if not violent_files:
        raise ValueError(f"No JSON files found in violent directory: {violent_path}")

    print(f"Found {len(violent_files)} violent JSON files")

    # Calculate number of files to process based on percentage
    num_violent_files = max(1, int(len(violent_files) * sample_percentage / 100))

    # Process violent samples
    for json_file in tqdm(
        violent_files[:num_violent_files], desc="Processing violent samples"
    ):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Process each frame in the JSON file
        for frame_data in data.get("instance_info", []):
            # Get frame ID (not used but kept for consistency)
            _ = frame_data.get("frame_id")
            instances = frame_data.get("instances", [])

            for instance in instances:
                keypoints = instance.get("keypoints", [])
                if keypoints:
                    # Convert to numpy array
                    keypoints_np = np.array(keypoints)

                    # Create graph from keypoints
                    graph = create_pose_graph(keypoints_np)
                    if graph is not None:
                        all_graphs.append(graph)
                        all_labels.append(1.0)  # Violent label

    # Process non-violent data
    non_violent_files = list(non_violent_path.glob("*.json"))
    if not non_violent_files:
        raise ValueError(
            f"No JSON files found in non-violent directory: {non_violent_path}"
        )

    print(f"Found {len(non_violent_files)} non-violent JSON files")

    # Calculate number of files to process based on percentage
    num_nonviolent_files = max(1, int(len(non_violent_files) * sample_percentage / 100))

    # Process non-violent samples
    for json_file in tqdm(
        non_violent_files[:num_nonviolent_files], desc="Processing non-violent samples"
    ):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Process each frame in the JSON file
        for frame_data in data.get("instance_info", []):
            # Get frame ID (not used but kept for consistency)
            _ = frame_data.get("frame_id")
            instances = frame_data.get("instances", [])

            for instance in instances:
                keypoints = instance.get("keypoints", [])
                if keypoints:
                    # Convert to numpy array
                    keypoints_np = np.array(keypoints)

                    # Create graph from keypoints
                    graph = create_pose_graph(keypoints_np)
                    if graph is not None:
                        all_graphs.append(graph)
                        all_labels.append(0.0)  # Non-violent label

    return all_graphs, all_labels


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

    Args:
        model: The GNN model
        train_loader: Training data loader
        val_loader: Validation data loader
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

        # Process batches
        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"
        ):
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            out = model(batch.x, batch.edge_index, batch.batch)
            target = batch.y.view(-1, 1).to(device)

            # Calculate loss
            loss = F.binary_cross_entropy(out, target)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs

        # Calculate average training loss
        avg_train_loss = total_loss / len(train_loader.dataset)
        metrics["train_loss"].append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"
            ):
                batch = batch.to(device)

                # Forward pass
                out = model(batch.x, batch.edge_index, batch.batch)
                target = batch.y.view(-1, 1).to(device)

                # Calculate loss
                loss = F.binary_cross_entropy(out, target)
                val_loss += loss.item() * batch.num_graphs

                # Store predictions and targets for metrics
                all_preds.extend(out.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())

        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_auc = roc_auc_score(all_targets, all_preds)

        metrics["val_loss"].append(avg_val_loss)
        metrics["val_auc"].append(val_auc)

        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val AUC: {val_auc:.4f}")

    return metrics


def get_device() -> torch.device:
    """
    Determine the optimal device for training/inference.

    Returns:
        torch.device: CUDA if available, MPS if on Apple Silicon, otherwise CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def find_optimal_threshold(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate the optimal classification threshold using multiple methods.

    Args:
        y_true: Ground truth binary labels
        y_score: Predicted scores (probabilities)

    Returns:
        Tuple of (optimal threshold, dictionary of metrics at that threshold)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # Calculate Youden's J statistic (J = Sensitivity + Specificity - 1)
    j_scores = tpr - fpr
    optimal_idx_j = np.argmax(j_scores)
    optimal_threshold_j = thresholds[optimal_idx_j]

    # Calculate distance to (0,1) point in ROC space
    distances = np.sqrt((1 - tpr) ** 2 + fpr**2)
    optimal_idx_d = np.argmin(distances)
    optimal_threshold_d = thresholds[optimal_idx_d]

    # Calculate F1 score at different thresholds
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)

    # Calculate F1 for all possible thresholds
    f1_scores = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)

    optimal_idx_f1 = np.argmax(f1_scores)
    optimal_threshold_f1 = thresholds[optimal_idx_f1]

    # Choose Youden's J as the primary method (most common in academic literature)
    optimal_threshold = optimal_threshold_j

    # Calculate confusion matrix at optimal threshold
    y_pred = (y_score >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate various metrics at the optimal threshold
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Create metrics dictionary
    metrics = {
        "threshold_j": optimal_threshold_j,
        "threshold_distance": optimal_threshold_d,
        "threshold_f1": optimal_threshold_f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision_val,
        "f1_score": f1_scores[optimal_idx_j],
        "youdens_j": j_scores[optimal_idx_j],
    }

    return optimal_threshold, metrics


def evaluate_model(
    model: ViolenceDetectionGNN, test_loader: DataLoader, device: torch.device
) -> Tuple[float, float, float, Dict[str, float]]:
    """
    Evaluate the model on the test set.

    Args:
        model: The trained GNN model
        test_loader: Test data loader
        device: Device for evaluation

    Returns:
        Tuple of (test_loss, test_auc, optimal_threshold, threshold_metrics)
    """
    model.eval()
    test_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(device)

            # Forward pass
            out = model(batch.x, batch.edge_index, batch.batch)
            target = batch.y.view(-1, 1).to(device)

            # Calculate loss
            loss = F.binary_cross_entropy(out, target)
            test_loss += loss.item() * batch.num_graphs

            # Store predictions and targets for metrics
            all_preds.extend(out.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy().flatten())

    avg_test_loss = test_loss / len(test_loader.dataset)
    test_auc = roc_auc_score(all_targets, all_preds)

    # Find optimal classification threshold
    optimal_threshold, threshold_metrics = find_optimal_threshold(
        np.array(all_targets), np.array(all_preds)
    )

    return avg_test_loss, test_auc, optimal_threshold, threshold_metrics


def plot_training_metrics(
    metrics: Dict[str, List[float]],
    test_auc: float,
    output_path: Path = Path("training_metrics.png"),
) -> None:
    """
    Plot and save training and validation metrics.

    Args:
        metrics: Dictionary containing training metrics
        test_auc: AUC score on test set
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 4))

    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(metrics["train_loss"], label="Train Loss")
    plt.plot(metrics["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    # Plot AUC curves
    plt.subplot(1, 2, 2)
    plt.plot(metrics["val_auc"], label="Validation AUC")
    plt.axhline(
        y=test_auc, color="r", linestyle="--", label=f"Test AUC: {test_auc:.4f}"
    )
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()
    plt.title("Validation AUC")

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Training metrics plot saved to {output_path}")


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    output_path: Path = Path("model_performance_curves.png"),
) -> None:
    """
    Plot ROC and Precision-Recall curves with threshold information.

    Args:
        y_true: Ground truth binary labels
        y_score: Predicted scores (probabilities)
        threshold: Optimal classification threshold
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 5))

    # ROC Curve
    plt.subplot(1, 2, 1)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_true, y_score):.4f})")

    # Add threshold point
    threshold_idx = np.argmin(np.abs(thresholds - threshold))
    plt.plot(
        fpr[threshold_idx],
        tpr[threshold_idx],
        "ro",
        label=f"Threshold = {threshold:.4f}",
    )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")

    # Precision-Recall Curve
    plt.subplot(1, 2, 2)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)

    # Find closest threshold value in PR curve
    pr_thresholds = np.append(
        pr_thresholds, 1.0
    )  # Add 1.0 to match precision/recall arrays
    threshold_idx_pr = np.argmin(np.abs(pr_thresholds - threshold))

    plt.plot(recall, precision, label="Precision-Recall Curve")
    plt.plot(
        recall[threshold_idx_pr],
        precision[threshold_idx_pr],
        "ro",
        label=f"Threshold = {threshold:.4f}",
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Performance curves saved to {output_path}")


def main() -> None:
    """Main function to train and evaluate the violence detection model."""
    # Use the best available device (GPU, MPS, or CPU)
    device = get_device()
    print(f"Using device: {device}")

    # Verify data paths exist
    if not VIOLENT_PATH.exists():
        print(f"Error: Violent data path does not exist: {VIOLENT_PATH}")
        return

    if not NON_VIOLENT_PATH.exists():
        print(f"Error: Non-violent data path does not exist: {NON_VIOLENT_PATH}")
        return

    # Set sample percentage for data loading (for testing purposes)
    sample_percentage = 100  # Default: process all data

    # Load and preprocess data
    print("Loading and preprocessing data...")
    try:
        graphs, labels = load_mmpose_data(
            VIOLENT_PATH, NON_VIOLENT_PATH, sample_percentage
        )
    except ValueError as e:
        print(f"Error loading data: {e}")
        return

    if not graphs:
        print("No valid graphs were created. Check your data.")
        return

    # Print dataset statistics
    print(f"Total graphs: {len(graphs)}")
    print(f"Positive (violent) samples: {sum(labels)}")
    print(f"Negative (non-violent) samples: {len(labels) - sum(labels)}")

    # Convert labels to tensors and add to graph data
    for i, graph in enumerate(graphs):
        graph.y = torch.tensor([labels[i]], dtype=torch.float)

    # Split data into train, validation, and test sets
    train_graphs, test_graphs = train_test_split(
        graphs, test_size=0.2, random_state=42, stratify=labels
    )
    train_graphs, val_graphs = train_test_split(
        train_graphs, test_size=0.25, random_state=42
    )

    print(f"Training graphs: {len(train_graphs)}")
    print(f"Validation graphs: {len(val_graphs)}")
    print(f"Test graphs: {len(test_graphs)}")

    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE)

    # Get input feature dimension from the first graph
    in_channels = train_graphs[0].x.shape[1]

    # Initialize model with GNN and Transformer components
    model = ViolenceDetectionGNN(
        in_channels=in_channels,
        hidden_channels=64,
        transformer_heads=4,
        transformer_layers=2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train model
    print("Training model...")
    metrics = train_model(
        model, train_loader, val_loader, device, optimizer, num_epochs=NUM_EPOCHS
    )

    # Evaluate on test set
    avg_test_loss, test_auc, optimal_threshold, threshold_metrics = evaluate_model(
        model, test_loader, device
    )
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Optimal classification threshold: {optimal_threshold:.4f}")
    print("Threshold metrics:")
    for metric, value in threshold_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Save model
    model_path = Path("violence_detection_model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "threshold": optimal_threshold,
            "metrics": threshold_metrics,
        },
        model_path,
    )
    print(f"Model saved to {model_path}")

    # Plot training metrics
    plot_training_metrics(metrics, test_auc)

    # Extract all predictions and targets from test set for curve plotting
    all_preds = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            all_preds.extend(out.cpu().numpy().flatten())
            all_targets.extend(batch.y.cpu().numpy().flatten())

    plot_roc_pr_curves(np.array(all_targets), np.array(all_preds), optimal_threshold)


if __name__ == "__main__":
    main()
