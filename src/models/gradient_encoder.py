"""P4 — GradientEncoder：将 ∂L_phy/∂ctrl 编码为 z_grad [B, 256]。

输入特征（来自 WarpGradComputer 的 grad_info）：
  grad_seq    [H, nu]   — 每步梯度
  grad_norm   [H]       — 每步梯度 L2 范数
  loss_val    [1]       — 当前 loss 值（归一化）
  clip_flag   [1]       — 是否被裁剪（梯度范数>10）

维度拆分（nu=7：xyz[3] + rot[3] + gripper[1]）：
  trans_ratio [1]   — 平移维梯度占比
  rot_ratio   [1]   — 旋转维梯度占比
  grip_ratio  [1]   — 夹爪维梯度占比

总输入：MLP([H*nu + H + 1 + 1 + 3] → 256)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import numpy as np


class GradientEncoder(nn.Module):
    """GradientEncoder：grad_info → z_grad [B, 256]。

    P1 中 z_grad 为全零 mask token；P4 替换为此编码。
    """

    D_TOKEN: int = 256

    def __init__(
        self,
        H:   int = 30,    # rollout horizon
        nu:  int = 7,     # ctrl 维度
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.H  = H
        self.nu = nu

        # 动作维度划分（前3平移，接着3旋转，最后1夹爪）
        self.trans_dims  = list(range(0, 3))
        self.rot_dims    = list(range(3, 6))
        self.gripper_dim = [6] if nu >= 7 else []

        # 输入维度：
        #   grad_seq  : H*nu
        #   grad_norm : H
        #   loss_val  : 1
        #   clip_flag : 1
        #   ratio×3   : 3
        in_dim = H * nu + H + 1 + 1 + 3

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, self.D_TOKEN),
        )

        # 解码头（梯度方向重建验证）
        self.decode_head = nn.Linear(self.D_TOKEN, H * nu)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # 特征拼接
    # ------------------------------------------------------------------
    @staticmethod
    def build_input(
        grad_seq:  np.ndarray | torch.Tensor,  # [H, nu]
        grad_norm: np.ndarray | torch.Tensor,  # [H]
        loss_val:  float,
        clipped:   bool = False,
        nu:        int  = 7,
    ) -> torch.Tensor:
        """组装单条 [feat_dim] 特征向量（供 DataLoader 使用）。"""
        if isinstance(grad_seq, np.ndarray):
            grad_seq  = torch.from_numpy(grad_seq).float()
        if isinstance(grad_norm, np.ndarray):
            grad_norm = torch.from_numpy(grad_norm).float()

        H = grad_seq.shape[0]
        g = grad_seq   # [H, nu]

        # 维度占比
        eps = 1e-8
        total_energy = g.pow(2).sum() + eps
        trans_ratio  = g[:, :3].pow(2).sum() / total_energy
        rot_ratio    = g[:, 3:6].pow(2).sum() / total_energy if nu >= 6 else torch.zeros(1)
        grip_ratio   = g[:, 6:7].pow(2).sum() / total_energy if nu >= 7 else torch.zeros(1)

        feat = torch.cat([
            g.flatten(),                                     # H*nu
            grad_norm,                                       # H
            torch.tensor([loss_val]),                        # 1
            torch.tensor([1.0 if clipped else 0.0]),         # 1
            torch.stack([trans_ratio, rot_ratio.squeeze(),
                         grip_ratio.squeeze()]),             # 3
        ])
        return feat

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: [B, in_dim] → z_grad: [B, 256]."""
        return self.mlp(feat)

    def decode(self, z_grad: torch.Tensor) -> torch.Tensor:
        """z_grad [B,256] → grad_seq_pred [B, H*nu]（用于重建监督）。"""
        return self.decode_head(z_grad)

    # ------------------------------------------------------------------
    # 零 token（P1 兼容用）
    # ------------------------------------------------------------------
    @classmethod
    def zero_token(cls, B: int, device: str = "cpu") -> torch.Tensor:
        """返回全零 z_grad [B, 256]（P1 占位）。"""
        return torch.zeros(B, cls.D_TOKEN, device=device)
