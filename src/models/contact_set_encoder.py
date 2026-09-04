"""ContactSetEncoder (v2): role-aware contact set encoder.

Relation flags (indices 21-27) are now role-based, not name-based:
  [21] involves_gripper
  [22] involves_target
  [23] involves_goal
  [24] involves_distractor
  [25] target_gripper    (TARGET ↔ GRIPPER)
  [26] target_goal       (TARGET ↔ GOAL_SURFACE)
  [27] target_static     (TARGET ↔ STATIC)

Continuous [0:17] and ID [17:21] layout unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

CONT_DIM    = 17
ID_DIM      = 4
REL_DIM     = 7
CONTACT_DIM = 28


class ContactSetEncoder(nn.Module):
    """Encode padded contact set [B, C, 28] → [B, token_dim]."""

    def __init__(
        self,
        token_dim:   int = 256,
        hidden_dim:  int = 128,
        rel_emb_dim: int = 16,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim

        self.rel_embeddings = nn.ModuleList([
            nn.Embedding(2, rel_emb_dim) for _ in range(REL_DIM)
        ])

        per_contact_dim = CONT_DIM + ID_DIM + REL_DIM * rel_emb_dim

        self.contact_mlp = nn.Sequential(
            nn.Linear(per_contact_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.output_proj = nn.Linear(hidden_dim, token_dim)

        # Decoder: classify contact relation type (7 binary heads)
        self.relation_heads = nn.Linear(token_dim, REL_DIM)

    def forward(
        self,
        contacts: torch.Tensor,   # [B, C, 28]
        mask:     torch.Tensor,   # [B, C]
    ) -> torch.Tensor:
        cont    = contacts[:, :, :CONT_DIM]
        ids     = contacts[:, :, CONT_DIM:CONT_DIM + ID_DIM]
        rel_raw = contacts[:, :, CONT_DIM + ID_DIM:]       # [B, C, 7]

        rel_idx  = rel_raw.long().clamp(0, 1)
        rel_embs = torch.cat(
            [self.rel_embeddings[i](rel_idx[:, :, i]) for i in range(REL_DIM)],
            dim=-1,
        )

        per_contact = torch.cat([cont, ids, rel_embs], dim=-1)
        h = self.contact_mlp(per_contact)                  # [B, C, hidden]

        m     = mask.unsqueeze(-1).float()
        count = m.sum(dim=1).clamp(min=1.0)
        pooled = (h * m).sum(dim=1) / count                # [B, hidden]

        return self.output_proj(pooled)

    def decode_relations(self, z: torch.Tensor) -> torch.Tensor:
        """Predict 7 binary relation flags from z_contact."""
        return self.relation_heads(z)                       # [B, 7]
