"""P5 — PhysicsTokenProjector: 将 z_phy [B, N, 256] 投影到 VLM 隐藏维度。

将 PhysicalTokenizer 产生的物理 token 投影到 StarVLA 的 Qwen-VL 隐藏空间，
随后拼接至 VLM last_hidden_state，供 DiT action head 通过 cross-attention 使用。

参数量 ~1.6M（Linear 256→2048 + GELU + Linear 2048→2048 + LayerNorm + pos_embed）。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PhysicsTokenProjector(nn.Module):
    """z_phy [B, N_phy, 256] → z_proj [B, N_phy, vla_hidden_dim].

    Architecture:
        Linear(phy_dim → vla_dim) → GELU → Linear(vla_dim → vla_dim) → LayerNorm
        + learnable positional embedding [1, N_phy, vla_dim]

    Only this module is trainable in P5-v1 (VLM + DiT frozen).
    N_phy = 7 base slots + up to 16 object slots = 23 total.
    """

    def __init__(
        self,
        phy_token_dim: int = 256,
        vla_hidden_dim: int = 2048,
        num_phy_tokens: int = 23,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.phy_token_dim  = phy_token_dim
        self.vla_hidden_dim = vla_hidden_dim
        self.num_phy_tokens = num_phy_tokens

        self.proj = nn.Sequential(
            nn.Linear(phy_token_dim, vla_hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(vla_hidden_dim, vla_hidden_dim, bias=True),
            nn.LayerNorm(vla_hidden_dim),
        )

        # 可学习位置编码：让 action head 区分各物理 token 的语义槽位
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_phy_tokens, vla_hidden_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, z_phy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_phy: [B, N_phy, phy_token_dim]  float32

        Returns:
            z_proj: [B, N_phy, vla_hidden_dim]
        """
        z = self.proj(z_phy)                         # [B, N_phy, vla_dim]
        z = z + self.pos_embed[:, :z.size(1), :]     # 位置编码（支持动态长度）
        return z

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
