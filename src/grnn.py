#!/usr/bin/env python3
"""
Graph Recurrent Neural Network (GRNN) for violence detection.

This module implements a GRNN model that processes graph-structured data over time,
combining the representational power of GNNs with the sequential modeling capabilities
of RNNs.

Key references:
- Y. Seo et al. "Structured Sequence Modeling with Graph RNNs" (2018)
- F. Scarselli et al. "The Graph Neural Network Model" (2009)
- Y. Li et al. "Gated Graph Sequence Neural Networks" (2016)
"""

from typing import List, Optional

# import numpy as np # No longer directly used in this file
import torch
import torch.nn as nn
# from torch_geometric.data import Data # No longer directly used in this file
from torch_geometric.nn import GCNConv, global_mean_pool


# create_pose_graph function has been moved to src.graph_utils


class GraphRNNCell(nn.Module):
    """
    Graph RNN Cell that implements a single time step of a GRNN.

    This cell combines graph-structured message passing with recurrent processing,
    maintaining a hidden state that is updated at each time step based on both
    the current graph input and the previous hidden state.

    Unlike traditional RNNs that operate on sequential data, GRNNs operate on
    graph-structured data that evolves over time, allowing them to capture
    both spatial and temporal dependencies.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize a Graph RNN Cell.

        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden state
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Graph convolution for processing the input graph
        # GCNConv is used to aggregate neighborhood information (spatial component)
        self.input_conv = GCNConv(input_dim, hidden_dim)

        # Graph convolution for processing the hidden state
        # This allows the hidden state to incorporate graph structure
        self.hidden_conv = GCNConv(hidden_dim, hidden_dim)

        # Update gate: controls what portion of the previous hidden state is retained
        # Implemented as a graph convolution to respect the graph structure
        self.update_gate = GCNConv(input_dim + hidden_dim, hidden_dim)

        # Reset gate: controls how much of the previous hidden state influences
        # the new candidate state
        # Also implemented as a graph convolution to respect the graph structure
        self.reset_gate = GCNConv(input_dim + hidden_dim, hidden_dim)

        # Candidate activation: generates new candidate hidden state
        self.candidate = GCNConv(input_dim + hidden_dim, hidden_dim)

        # Layer normalization for better training stability
        # Applied after each graph convolution operation
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for a single time step of the Graph RNN Cell.

        Args:
            x: Node features tensor [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            h_prev: Previous hidden state [num_nodes, hidden_dim]

        Returns:
            New hidden state [num_nodes, hidden_dim]
        """
        batch_size, num_nodes, _ = x.size()

        # Initialize hidden state if not provided
        if h_prev is None:
            h_prev = torch.zeros(
                batch_size, num_nodes, self.hidden_dim, device=x.device
            )

        # Flatten batch and node dimensions for graph convolution
        x_flat = x.view(-1, x.size(-1))
        h_prev_flat = h_prev.view(-1, h_prev.size(-1))

        # Reshape edge_index for batched processing
        # Each batch has its own graph structure
        edge_index_batched = []
        total_nodes = batch_size * num_nodes

        for i in range(batch_size):
            # Offset node indices for each batch
            offset = i * num_nodes
            batch_edge_index = edge_index + offset

            # Safety check: ensure node indices are within valid range
            # This prevents out-of-bounds indexing that causes CUDA device-side asserts
            valid_edges_mask = (
                (batch_edge_index[0] < total_nodes)
                & (batch_edge_index[1] < total_nodes)
                & (batch_edge_index[0] >= 0)
                & (batch_edge_index[1] >= 0)
            )

            # Only keep valid edges
            if not valid_edges_mask.all():
                batch_edge_index = batch_edge_index[:, valid_edges_mask]

            edge_index_batched.append(batch_edge_index)

        if len(edge_index_batched) > 0:
            edge_index_batched = torch.cat(edge_index_batched, dim=1)
        else:
            # Create fallback empty edge index if no valid edges
            edge_index_batched = torch.zeros((2, 0), dtype=torch.long, device=x.device)

        # Concatenate input and previous hidden state for gate computations
        # This allows gates to consider both current input and previous state
        xh_concat = torch.cat([x_flat, h_prev_flat], dim=1)

        # Update gate: determines how much of the previous hidden state to keep
        # σ(W_z·[x_t, h_{t-1}] + b_z)
        z = torch.sigmoid(self.update_gate(xh_concat, edge_index_batched))

        # Reset gate: controls the influence of the previous hidden state
        # σ(W_r·[x_t, h_{t-1}] + b_r)
        r = torch.sigmoid(self.reset_gate(xh_concat, edge_index_batched))

        # Apply reset gate to previous hidden state
        # This filters what information from previous state is relevant
        r_h = r * h_prev_flat

        # Candidate hidden state: potential new information to add
        # tanh(W_c·[x_t, r_t ⊙ h_{t-1}] + b_c)
        xrh_concat = torch.cat([x_flat, r_h], dim=1)
        candidate_h = torch.tanh(self.candidate(xrh_concat, edge_index_batched))

        # Final hidden state computation: weighted combination of previous and candidate
        # h_t = (1-z_t) ⊙ h_{t-1} + z_t ⊙ \tilde{h}_t
        h_new = (1 - z) * h_prev_flat + z * candidate_h

        # Apply layer normalization for stability
        h_new = self.layer_norm(h_new)

        # Reshape back to [batch_size, num_nodes, hidden_dim]
        h_new = h_new.view(batch_size, num_nodes, -1)

        return h_new


class GRNN(nn.Module):
    """
    Graph Recurrent Neural Network for processing graph sequences.

    This model combines GNN-based spatial processing with RNN-based temporal processing,
    enabling the capture of both spatial structure and temporal dynamics in graph data.

    The model processes a sequence of graphs through a recurrent cell that updates
    a hidden state, producing a final representation that captures the full
    spatiotemporal context.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        """
        Initialize the GRNN model.

        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden state
            num_layers: Number of recurrent layers
            dropout: Dropout rate between layers
            bidirectional: Whether to use bidirectional processing
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        # Multiple layers of Graph RNN Cells
        # Each layer processes the output of the previous layer
        self.cells = nn.ModuleList()

        # Input dimension for the first layer
        layer_input_dim = input_dim

        for i in range(num_layers):
            # Create a Graph RNN Cell for this layer
            cell = GraphRNNCell(layer_input_dim, hidden_dim)
            self.cells.append(cell)

            # For subsequent layers, input is the hidden state of the previous layer
            layer_input_dim = hidden_dim

            # Double the input dimension for bidirectional layers (except the first)
            if bidirectional and i > 0:
                layer_input_dim *= 2

        # Backward direction cells for bidirectional GRNN
        if bidirectional:
            self.backward_cells = nn.ModuleList()
            layer_input_dim = input_dim

            for i in range(num_layers):
                cell = GraphRNNCell(layer_input_dim, hidden_dim)
                self.backward_cells.append(cell)
                layer_input_dim = hidden_dim
                if i > 0:
                    layer_input_dim *= 2

        # Dropout layer between recurrent layers
        self.dropout_layer = nn.Dropout(dropout)

        # Final output dimension
        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(
        self, x_sequence: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the GRNN.

        Args:
            x_sequence: Node features with shape [batch_size, seq_len,
                num_nodes, feat_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment for nodes [num_nodes]

        Returns:
            Final graph embeddings [batch_size, output_dim]
        """
        batch_size, seq_len, num_nodes, _ = x_sequence.size()
        device = x_sequence.device

        # Initialize list to store outputs of each layer
        layer_outputs: List[torch.Tensor] = []

        # Initialize hidden states for each layer
        h_states = [None] * self.num_layers

        # Forward direction processing
        for t in range(seq_len):
            # Extract the graph at this time step
            x_t = x_sequence[:, t, :, :]

            # Store outputs of current time step for each layer
            layer_t_outputs: List[torch.Tensor] = []

            # Process through each layer
            for layer in range(self.num_layers):
                if layer == 0:
                    # First layer takes the input directly
                    layer_input = x_t
                else:
                    # Subsequent layers take the output of the previous layer
                    layer_input = layer_t_outputs[-1]

                    # Apply dropout between layers
                    layer_input = self.dropout_layer(layer_input)

                # Process through the Graph RNN Cell
                h_new = self.cells[layer](layer_input, edge_index, h_states[layer])

                # Update hidden state for this layer
                h_states[layer] = h_new

                # Add to layer outputs
                layer_t_outputs.append(h_new)

            # Store the outputs of the last layer for this time step
            layer_outputs.append(layer_t_outputs[-1])

        # Backward direction processing (if bidirectional)
        backward_outputs: List[torch.Tensor] = []
        if self.bidirectional:
            # Initialize hidden states for backward pass
            h_states_bwd = [None] * self.num_layers

            # Process sequence in reverse order
            for t in range(seq_len - 1, -1, -1):
                x_t = x_sequence[:, t, :, :]
                layer_t_outputs_bwd: List[torch.Tensor] = []

                for layer in range(self.num_layers):
                    if layer == 0:
                        layer_input = x_t
                    else:
                        layer_input = layer_t_outputs_bwd[-1]
                        layer_input = self.dropout_layer(layer_input)

                    h_new = self.backward_cells[layer](
                        layer_input, edge_index, h_states_bwd[layer]
                    )
                    h_states_bwd[layer] = h_new
                    layer_t_outputs_bwd.append(h_new)

                backward_outputs.append(layer_t_outputs_bwd[-1])

            # Reverse backward outputs to align with forward outputs
            backward_outputs = backward_outputs[::-1]

        # Get the final hidden states (last time step)
        final_hidden = layer_outputs[-1]  # [batch_size, num_nodes, hidden_dim]

        # Combine forward and backward states if bidirectional
        if self.bidirectional:
            final_hidden = torch.cat([final_hidden, backward_outputs[-1]], dim=-1)

        # Global pooling to get a graph-level representation
        # Reshape for pooling: [batch_size * num_nodes, hidden_dim]
        final_hidden_flat = final_hidden.view(-1, final_hidden.size(-1))

        # Create appropriate batch vector for pooling
        # This assigns each node to its corresponding graph
        batch_pooling = torch.arange(batch_size, device=device).repeat_interleave(
            num_nodes
        )

        # Apply global mean pooling to get graph-level embedding
        pooled = global_mean_pool(final_hidden_flat, batch_pooling)

        return pooled


class ViolenceDetectionGRNN(nn.Module):
    """
    Violence detection model using Graph Recurrent Neural Networks.

    This model processes sequences of pose graphs through a GRNN to detect violent
    behavior, outputting a continuous score between 0 and 1 representing violence
    probability.

    Unlike models that use a standard classification head with cross-entropy loss,
    this model is designed for direct regression of violence scores, which is more
    appropriate for tasks where violence exists on a spectrum rather than as a binary
    state.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        """
        Initialize the violence detection GRNN model.

        Args:
            in_channels: Number of input features per node (typically 2 for x,y coords)
            hidden_channels: Size of hidden representations
            num_layers: Number of GRNN layers
            dropout: Dropout rate for regularization
            bidirectional: Whether to use bidirectional GRNN
        """
        super().__init__()

        # GRNN component for spatial-temporal processing
        self.grnn = GRNN(
            input_dim=in_channels,
            hidden_dim=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        # Final output dimension from GRNN
        grnn_output_dim = hidden_channels * (2 if bidirectional else 1)

        # Score regression network
        # Instead of a classification head, we use a regression network
        # that outputs a continuous value between 0 and 1
        self.score_network = nn.Sequential(
            nn.Linear(grnn_output_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(
        self, x_sequence: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the violence detection model.

        Args:
            x_sequence: Node features [batch_size, seq_len, num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment for nodes [num_nodes]

        Returns:
            Violence score between 0 and 1 [batch_size, 1]
        """
        # Process through GRNN to get graph sequence embeddings
        embeddings = self.grnn(x_sequence, edge_index, batch)

        # Compute violence score through regression network
        scores = self.score_network(embeddings)

        # Use sigmoid to constrain output between 0 and 1
        # This is a key difference from classification models:
        # - The output is a continuous score rather than a class probability
        # - Appropriate for violence detection where violence exists on a spectrum
        # - Allows for more nuanced assessment of violent content
        violence_score = torch.sigmoid(scores)

        return violence_score
