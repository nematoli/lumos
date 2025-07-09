from typing import List

import torch
import torch.nn as nn

from lumos.utils.nn_utils import flatten_batch, get_activation, unflatten_batch


class InverseDynamics(nn.Module):
    """
    Inverse dynamics network that predicts actions from consecutive embedding pairs.

    Args:
        in_dim: Input dimension (embedding size)
        out_dim: Output dimension (action space size)
        hidden_dims: List of hidden layer dimensions
        activation: Activation function name
        dropout: Dropout probability (optional)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dims = hidden_dims
        self.activation = get_activation(activation)
        self.dropout = dropout

        layers = []
        current_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, out_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, embeddings):
        """
        Forward pass through inverse dynamics network.

        Args:
            embeddings: Input embeddings of shape (seq, batch, in_dim)

        Returns:
            predicted_actions: Predicted actions of shape (seq-1, batch, out_dim)
        """

        current_embeds = embeddings[:-1]
        next_embeds = embeddings[1:]

        embed_pairs = torch.cat([current_embeds, next_embeds], dim=-1)

        embed_pairs_flat, bd = flatten_batch(embed_pairs)
        predicted_actions = self.mlp(embed_pairs_flat)
        predicted_actions = unflatten_batch(predicted_actions, bd)

        return predicted_actions
