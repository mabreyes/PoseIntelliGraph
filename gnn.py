#!/usr/bin/env python3
"""
Graph Neural Network component for the violence detection system.

This module contains the GNN model that processes human pose data
represented as graphs.
"""
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool


class PoseGNN(nn.Module):
    """
    Graph Neural Network model for processing pose data.

    This model processes pose keypoints represented as graphs and outputs
    feature embeddings for further processing.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64):
        """
        Initialize the GNN model.

        Args:
            in_channels: Number of input features per node
                         (typically 2 for x,y coordinates)
            hidden_channels: Size of hidden representations
        """
        super(PoseGNN, self).__init__()
        # Graph convolutional layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        # Output dimension is hidden_channels after pooling
        self.out_channels = hidden_channels

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
            Graph embeddings [batch_size, hidden_channels]
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
        
        return x


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