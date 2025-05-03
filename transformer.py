#!/usr/bin/env python3
"""
Transformer component for the violence detection system.

This module contains the transformer model that processes embeddings
from the GNN before final classification.
"""
import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for processing graph embeddings.
    
    This model takes embeddings from the GNN and applies self-attention
    to capture temporal and contextual relationships.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        num_heads: int = 4, 
        num_layers: int = 2, 
        dropout: float = 0.1,
        output_dim: int = 64
    ):
        """
        Initialize the transformer encoder.
        
        Args:
            input_dim: Dimension of input features
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            output_dim: Dimension of output features
        """
        super(TransformerEncoder, self).__init__()
        
        # Add batch dimension info for the transformer
        self.position_embedding = nn.Parameter(torch.zeros(1, 1, input_dim))
        
        # Create transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        
        # Create transformer encoder
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection if needed
        self.out_projection = None
        if output_dim != input_dim:
            self.out_projection = nn.Linear(input_dim, output_dim)
            
        self.out_channels = output_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Transformed representation [batch_size, output_dim]
        """
        # Add sequence dimension for transformer
        x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        # Add positional embedding
        x = x + self.position_embedding
        
        # Apply transformer
        x = self.transformer(x)
        
        # Remove sequence dimension
        x = x.squeeze(1)  # [batch_size, input_dim]
        
        # Apply output projection if needed
        if self.out_projection is not None:
            x = self.out_projection(x)
            
        return x 