"""Exact Phase-2 neural-network architectures for inference.

These classes are matched to the frozen checkpoints:
- cnn_phase2_v2_best.pt
- gnn_phase2_best.pt

No preprocessing, Streamlit code, or training logic belongs in this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from torch_geometric.nn import GCNConv, global_mean_pool
except ImportError:
    GCNConv = None
    global_mean_pool = None


def _extract_state_dict(checkpoint: Any) -> Mapping[str, Tensor]:
    """Return a plain state_dict from common PyTorch checkpoint formats."""
    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "Expected a PyTorch state_dict or a dictionary containing "
            "a 'state_dict' entry."
        )

    # Handle checkpoints saved from DataParallel.
    if any(str(key).startswith("module.") for key in checkpoint):
        checkpoint = {
            str(key).removeprefix("module."): value
            for key, value in checkpoint.items()
        }

    return checkpoint


class MutationAwareCNNv2(nn.Module):
    """Mutation-aware CNN used by the frozen Phase-2 CNN checkpoint.

    Inputs
    ------
    x_map:
        Tensor of shape ``(batch, 2, 128, 128)``. Channel 1 is the
        residue-contact map and channel 2 is the mutation-site mask.
    mut_vec4:
        Tensor of shape ``(batch, 4)`` containing, in this exact order:
        delta hydrophobicity, delta volume, delta charge, and BLOSUM62 score.
    """

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),   # cnn.0
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # cnn.3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # cnn.6
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # cnn.9
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.mut_mlp = nn.Sequential(
            nn.Linear(4, 16),   # mut_mlp.0
            nn.ReLU(),
            nn.Linear(16, 16),  # mut_mlp.2
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(80, 64),  # head.0: 64 CNN + 16 mutation features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),   # head.3
        )

    def forward(self, x_map: Tensor, mut_vec4: Tensor) -> Tensor:
        cnn_features = self.cnn(x_map).flatten(start_dim=1)
        mutation_features = self.mut_mlp(mut_vec4)
        combined = torch.cat((cnn_features, mutation_features), dim=1)
        return self.head(combined).squeeze(1)


class MutationAwareGNN(nn.Module):
    """Mutation-aware GCN used by the frozen Phase-2 GNN checkpoint.

    Each residue is represented by 25 node features:
    20 amino-acid identity features plus five mutation-aware features.
    """

    def __init__(self, in_dim: int = 25, dropout: float = 0.2) -> None:
        super().__init__()

        if in_dim != 25:
            raise ValueError(
                f"The frozen Phase-2 GNN expects exactly 25 node features; "
                f"received in_dim={in_dim}."
            )
        if GCNConv is None or global_mean_pool is None:
            raise ImportError(
                "torch-geometric is required to instantiate the Phase-2 GNN."
            )

        self.conv1 = GCNConv(25, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 32)

        self.head = nn.Sequential(
            nn.Linear(32, 32),  # head.0
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),   # head.3
        )

    def forward(self, data: Any) -> Tensor:
        x = data.x
        edge_index = data.edge_index

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(
                x.size(0), dtype=torch.long, device=x.device
            )

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        graph_features = global_mean_pool(x, batch)

        return self.head(graph_features).squeeze(1)


def load_cnn_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> MutationAwareCNNv2:
    """Load the frozen CNN with strict key and shape validation."""
    model = MutationAwareCNNv2().to(device)
    checkpoint = torch.load(
        str(checkpoint_path), map_location=device, weights_only=False
    )
    model.load_state_dict(_extract_state_dict(checkpoint), strict=True)
    model.eval()
    return model


def load_gnn_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> MutationAwareGNN:
    """Load the frozen GNN with strict key and shape validation."""
    model = MutationAwareGNN(in_dim=25).to(device)
    checkpoint = torch.load(
        str(checkpoint_path), map_location=device, weights_only=False
    )
    model.load_state_dict(_extract_state_dict(checkpoint), strict=True)
    model.eval()
    return model
