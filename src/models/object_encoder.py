"""ObjectEncoder: shared per-object encoder for all N scene objects.

Input (57 dims per object):
  role_embedding  [32]  nn.Embedding(5, 32)
  position        [3]
  rot_6d          [6]   6D rotation from quat_xyzw_to_6d
  linear_velocity [3]
  angular_velocity[3]
  mass_log1p      [1]
  inertia_log1p   [3]
  friction_log1p  [3]
  is_grasped      [1]
  dist_to_ee      [1]
  dist_to_goal    [1]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from utils.task_role_parser import ROLE_TARGET, ROLE_GOAL_SURFACE, ROLE_DISTRACTOR, ROLE_STATIC

OBJ_CONT_DIM  = 25   # all continuous features (position onward)
OBJ_ROLE_DIM  = 32   # role embedding dimension
OBJ_INPUT_DIM = OBJ_ROLE_DIM + OBJ_CONT_DIM   # 57

# Scene-context pooling weights by role
ROLE_POOL_WEIGHTS = {
    ROLE_TARGET:       2.0,
    ROLE_GOAL_SURFACE: 1.5,
    ROLE_DISTRACTOR:   0.5,
    ROLE_STATIC:       0.2,
}


class ObjectEncoder(nn.Module):
    """Encode a single object (or a padded batch of objects) to [B, D] tokens.

    For a batch of N objects with padding, call forward_set() to get [B, N, D].
    """

    def __init__(self, token_dim: int = 256, hidden_dim: int = 256) -> None:
        super().__init__()
        self.token_dim = token_dim

        self.role_emb = nn.Embedding(5, OBJ_ROLE_DIM)

        self.encoder = nn.Sequential(
            nn.Linear(OBJ_INPUT_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_dim),
        )

        # Decoder heads for supervised verification
        self.pose_decoder   = nn.Linear(token_dim, 9)   # position[3] + rot_6d[6]
        self.role_classifier = nn.Linear(token_dim, 5)  # role classification
        self.param_decoder  = nn.Linear(token_dim, 7)   # mass[1]+inertia[3]+friction[3]

    def forward(self, roles: torch.Tensor, cont: torch.Tensor) -> torch.Tensor:
        """
        Args:
            roles: [B] or [B, N]  long tensor of role indices.
            cont:  [B, OBJ_CONT_DIM] or [B, N, OBJ_CONT_DIM]  continuous features.
        Returns:
            [B, token_dim] or [B, N, token_dim]
        """
        r_emb = self.role_emb(roles)                    # [..., 32]
        x = torch.cat([r_emb, cont], dim=-1)            # [..., 57]
        return self.encoder(x)

    def forward_set(
        self,
        roles: torch.Tensor,      # [B, N] long
        cont:  torch.Tensor,      # [B, N, OBJ_CONT_DIM]
        mask:  torch.Tensor,      # [B, N] float  1=valid, 0=padding
    ) -> torch.Tensor:
        """Encode a padded set of objects. Returns [B, N, token_dim]."""
        return self.forward(roles, cont)                # broadcasts over N

    def decode_pose(self, z: torch.Tensor) -> torch.Tensor:
        return self.pose_decoder(z)

    def classify_role(self, z: torch.Tensor) -> torch.Tensor:
        return self.role_classifier(z)

    def decode_params(self, z: torch.Tensor) -> torch.Tensor:
        return self.param_decoder(z)


class SceneContextEncoder(nn.Module):
    """Aggregate N object tokens into a single scene-context token via role-weighted pooling."""

    def __init__(self, token_dim: int = 256) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )
        # Learnable per-role weight scalars (initialized from prior)
        init_w = torch.tensor([2.0, 1.5, 0.5, 0.2, 0.1])
        self.role_weights = nn.Parameter(init_w)

    def forward(
        self,
        obj_tokens: torch.Tensor,   # [B, N, token_dim]
        roles:      torch.Tensor,   # [B, N] long
        mask:       torch.Tensor,   # [B, N] float
    ) -> torch.Tensor:
        """Returns z_scene_ctx: [B, token_dim]."""
        w = torch.clamp(self.role_weights[roles], min=0.0)  # [B, N]
        w = w * mask                                         # zero out padding
        w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-6)  # [B, 1]
        weighted = (obj_tokens * w.unsqueeze(-1)).sum(dim=1) / w_sum  # [B, D]
        return self.proj(weighted)
