"""PhysicalTokenizer (v2): generalized [B, 7+N, D] token assembly.

Base slots (7, fixed):
  [0] z_robot      robot kinematics
  [1] z_contact    role-aware contact set
  [2] z_dyn        TARGET delta + relative object changes
  [3] z_risk       rule-derived risk (P1) / rollout cost (P3)
  [4] z_afford     task phase + events
  [5] z_grad       mask token (P4: real gradient)
  [6] z_scene_ctx  role-weighted pooling of all object tokens

Object slots (N, variable, max_N=16):
  [7..7+N-1] z_obj_i  per-object token (role + pose + vel + params)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from models.state_encoder import StateEncoder, ROBOT_INPUT_DIM
from models.contact_set_encoder import ContactSetEncoder
from models.object_encoder import (
    ObjectEncoder, SceneContextEncoder, OBJ_CONT_DIM,
)
from utils.task_role_parser import ROLE_TARGET

NUM_BASE_TOKENS = 7
MAX_N_OBJECTS   = 16


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


class PhysicalTokenizer(nn.Module):
    """Produce z_phy [B, 7+N, D] from a generalized physics observation dict."""

    def __init__(
        self,
        token_dim: int = 256,
        hidden_dim: int = 256,
        max_n_objects: int = MAX_N_OBJECTS,
    ) -> None:
        super().__init__()
        self.token_dim     = token_dim
        self.max_n_objects = max_n_objects

        # Base slot encoders
        self.robot_encoder   = StateEncoder(token_dim, hidden_dim)
        self.contact_encoder = ContactSetEncoder(token_dim, hidden_dim // 2)

        # z_dyn: two-path encoder
        DYN_TARGET_DIM  = 16   # TARGET abs delta: pos[3]+rot6d[6]+lv[3]+av[3]+dgoal[1]
        DYN_REL_DIM     = 36   # per other-obj: Δrel_pos[3]+Δrel_dist[1]+role_emb[32]
        self.dyn_target_mlp  = _mlp(DYN_TARGET_DIM, hidden_dim // 2, token_dim // 2)
        self.dyn_context_mlp = _mlp(DYN_REL_DIM,   hidden_dim // 2, token_dim // 4)
        self.dyn_proj        = nn.Linear(token_dim // 2 + token_dim // 4, token_dim)

        # z_risk: 6-dim rule features
        self.risk_encoder = _mlp(6, hidden_dim // 2, token_dim)
        self.risk_head    = nn.Linear(token_dim, 6)

        # z_afford: phase one-hot[8] + events[5]
        self.afford_encoder = _mlp(13, hidden_dim // 2, token_dim)
        self.phase_head     = nn.Linear(token_dim, 8)

        # z_grad: learned zero mask token
        self.grad_token = nn.Parameter(torch.zeros(token_dim))

        # Object tokens
        self.obj_encoder     = ObjectEncoder(token_dim, hidden_dim)
        self.scene_ctx_enc   = SceneContextEncoder(token_dim)

    # ------------------------------------------------------------------ #

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Required batch keys:
          robot_vec      [B, 33]
          contacts       [B, C, 28]
          contact_mask   [B, C]
          dyn_target     [B, 16]    TARGET state delta
          dyn_rel_cont   [B, N, 36] other-object relative deltas (padded)
          dyn_rel_mask   [B, N]     valid mask for dyn_rel_cont
          risk_vec       [B, 6]
          afford_vec     [B, 13]
          obj_roles      [B, N]     long, role per object
          obj_cont       [B, N, 25] continuous object features
          obj_mask       [B, N]     float, 1=valid

        Returns z_phy: [B, 7+N, token_dim]
        """
        B = batch["robot_vec"].shape[0]
        N = batch["obj_cont"].shape[1]

        tokens: List[torch.Tensor] = []

        # [0] z_robot
        tokens.append(self.robot_encoder(batch["robot_vec"]))

        # [1] z_contact
        tokens.append(self.contact_encoder(batch["contacts"], batch["contact_mask"]))

        # [2] z_dyn
        z_dyn_t = self.dyn_target_mlp(batch["dyn_target"])    # [B, D/2]
        dyn_rel  = self.dyn_context_mlp(batch["dyn_rel_cont"]) # [B, N, D/4]
        rel_mask = batch["dyn_rel_mask"].unsqueeze(-1)
        dyn_ctx  = (dyn_rel * rel_mask).sum(1) / rel_mask.sum(1).clamp(min=1)
        tokens.append(self.dyn_proj(torch.cat([z_dyn_t, dyn_ctx], dim=-1)))

        # [3] z_risk
        tokens.append(self.risk_encoder(batch["risk_vec"]))

        # [4] z_afford
        tokens.append(self.afford_encoder(batch["afford_vec"]))

        # [5] z_grad (mask token)
        tokens.append(self.grad_token.unsqueeze(0).expand(B, -1))

        # Object tokens [B, N, D]
        obj_tokens = self.obj_encoder.forward_set(
            batch["obj_roles"], batch["obj_cont"], batch["obj_mask"]
        )   # [B, N, token_dim]

        # [6] z_scene_ctx
        tokens.append(self.scene_ctx_enc(obj_tokens, batch["obj_roles"], batch["obj_mask"]))

        # Stack base tokens [B, 7, D]
        base = torch.stack(tokens, dim=1)

        # Pad or truncate object tokens to max_n_objects
        if N < self.max_n_objects:
            pad = torch.zeros(B, self.max_n_objects - N, self.token_dim,
                              device=base.device, dtype=base.dtype)
            obj_tokens = torch.cat([obj_tokens, pad], dim=1)
        else:
            obj_tokens = obj_tokens[:, :self.max_n_objects]

        return torch.cat([base, obj_tokens], dim=1)   # [B, 7+max_N, D]

    # ------------------------------------------------------------------ #
    # Decoder helpers
    # ------------------------------------------------------------------ #

    def decode_robot(self, z_phy: torch.Tensor) -> torch.Tensor:
        return self.robot_encoder.decode(z_phy[:, 0])

    def decode_contact_relations(self, z_phy: torch.Tensor) -> torch.Tensor:
        return self.contact_encoder.decode_relations(z_phy[:, 1])

    def decode_risk(self, z_phy: torch.Tensor) -> torch.Tensor:
        return self.risk_head(z_phy[:, 3])

    def classify_phase(self, z_phy: torch.Tensor) -> torch.Tensor:
        return self.phase_head(z_phy[:, 4])

    def decode_obj_pose(self, z_phy: torch.Tensor, obj_idx: int) -> torch.Tensor:
        return self.obj_encoder.decode_pose(z_phy[:, 7 + obj_idx])

    def classify_obj_role(self, z_phy: torch.Tensor, obj_idx: int) -> torch.Tensor:
        return self.obj_encoder.classify_role(z_phy[:, 7 + obj_idx])

    def decode_obj_params(self, z_phy: torch.Tensor, obj_idx: int) -> torch.Tensor:
        return self.obj_encoder.decode_params(z_phy[:, 7 + obj_idx])
