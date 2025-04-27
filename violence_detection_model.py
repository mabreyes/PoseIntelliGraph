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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from tqdm import tqdm

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
    Graph Neural Network model for violence detection from pose data.

    This model processes pose keypoints represented as graphs and outputs
    a violence score between 0 and 1.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64):
        """
        Initialize the GNN model.

        Args:
            in_channels: Number of input features per node
                         (typically 2 for x,y coordinates)
            hidden_channels: Size of hidden representations
        """
        super(ViolenceDetectionGNN, self).__init__()
        # Graph convolutional layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        # Final prediction layers
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = nn.Linear(hidden_channels // 2, 1)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the GNN.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment for nodes [num_nodes]

        Returns:
            Violence score between 0 and 1 [batch_size, 1]
        """
        # Graph convolutions
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # Global mean pooling (graph-level output)
        x = global_mean_pool(x, batch)

        # Final predictions
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin2(x)

        # Output violence score between 0 and 1
        return torch.sigmoid(x)


def create_pose_graph(keypoints: np.ndarray) -> Optional[Data]:
    """
    Convert keypoints into a graph representation for GNN processing.

    Args:
        keypoints: NumPy array of shape [num_keypoints, 2] containing (x, y) coordinates

    Returns:
        PyTorch Geometric Data object or None if the graph cannot be created
    """
    # Filter out any invalid keypoints (indicated by zeros or NaNs)
    valid_mask = ~np.isnan(keypoints).any(axis=1) & (keypoints != 0).any(axis=1)
    valid_keypoints = keypoints[valid_mask]

    if len(valid_keypoints) < 3:  # Need at least 3 points for a meaningful graph
        return None

    # Node features are the 2D coordinates
    x = torch.tensor(valid_keypoints, dtype=torch.float)

    # Create edges - we connect keypoints based on human body structure
    # For simplicity, connect each keypoint to all others (fully connected graph)
    num_nodes = len(valid_keypoints)
    edge_list = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            edge_list.append([i, j])
            edge_list.append([j, i])  # Bidirectional edges

    if not edge_list:
        return None

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Create PyTorch Geometric Data object
    return Data(x=x, edge_index=edge_index)


def load_mmpose_data(
    violent_path: Path, non_violent_path: Optional[Path] = None
) -> Tuple[List[Data], List[float]]:
    """
    Load MMPose JSON files and convert them to graph data.

    Args:
        violent_path: Path to violent pose JSON files
        non_violent_path: Path to non-violent pose JSON files

    Returns:
        Tuple of (list of graph Data objects, list of corresponding labels)
    """
    all_graphs = []
    all_labels = []

    # Get all JSON files from the violent directory
    violent_files = list(violent_path.glob("*.json"))
    print(f"Found {len(violent_files)} violent JSON files")

    # Process violent samples (limit to 100 for testing)
    for json_file in tqdm(violent_files[:100], desc="Processing violent samples"):
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

    # If non-violent data is provided, process it as well
    if non_violent_path and non_violent_path.exists():
        non_violent_files = list(non_violent_path.glob("*.json"))
        print(f"Found {len(non_violent_files)} non-violent JSON files")

        # Process non-violent samples (limit to 100 for testing)
        for json_file in tqdm(
            non_violent_files[:100], desc="Processing non-violent samples"
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
    else:
        # If no non-violent data, use some violent data
        # as negative examples (for testing)
        print(
            "No non-violent data provided. \
              Creating synthetic non-violent examples."
        )

        # Use different files from violent data as non-violent samples
        # This is just for testing purposes and should be
        #  replaced with real non-violent data
        sample_files = violent_files[100:150]

        for json_file in tqdm(
            sample_files, desc="Creating synthetic non-violent samples"
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


def evaluate_model(
    model: ViolenceDetectionGNN, test_loader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate the model on the test set.

    Args:
        model: The trained GNN model
        test_loader: Test data loader
        device: Device for evaluation

    Returns:
        Tuple of (test_loss, test_auc)
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

    return avg_test_loss, test_auc


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


def main() -> None:
    """Main function to train and evaluate the violence detection model."""
    # Use the best available device (GPU, MPS, or CPU)
    device = get_device()
    print(f"Using device: {device}")

    # Load and preprocess data
    print("Loading and preprocessing data...")
    graphs, labels = load_mmpose_data(VIOLENT_PATH, NON_VIOLENT_PATH)

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

    # Initialize model
    model = ViolenceDetectionGNN(in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train model
    print("Training model...")
    metrics = train_model(
        model, train_loader, val_loader, device, optimizer, num_epochs=NUM_EPOCHS
    )

    # Evaluate on test set
    avg_test_loss, test_auc = evaluate_model(model, test_loader, device)
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

    # Save model
    model_path = Path("violence_detection_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Plot training metrics
    plot_training_metrics(metrics, test_auc)


if __name__ == "__main__":
    main()
