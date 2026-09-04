"""StateEncoder (v2): encode robot-only kinematics to z_robot [B, 256].

Input (33 dims):
  robot_qpos/qvel   [7+7]
  ee_pos/rot6d/vel  [3+6+3+3]
  gripper_qpos/qvel [2+2]

Object state has moved to ObjectEncoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

ROBOT_INPUT_DIM = 33   # 7+7+3+6+3+3+2+2


class StateEncoder(nn.Module):
    """Encode robot kinematics to a fixed-length token."""

    def __init__(self, token_dim: int = 256, hidden_dim: int = 256) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.encoder = nn.Sequential(
            nn.Linear(ROBOT_INPUT_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_dim),
        )
        self.decoder = nn.Linear(token_dim, ROBOT_INPUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x [B, 33]. Returns z_robot [B, token_dim]."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
