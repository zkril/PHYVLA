"""P5 — StarVLA 物理增强包装器。

将 PhysicsTokenProjector 集成到 StarVLA（QwenGR00T）中：
将 z_phy 投影后拼接到 VLM last_hidden_state，DiT action head 通过 cross-attention 感知物理 token。

设计原则：
  - 冻结 VLM（qwen_vl_interface）：不更新视觉/语言 backbone
  - 可训练：PhysicsTokenProjector + DiT action head（可选）
  - 组合模式而非继承，避免与 StarVLA 自定义 from_pretrained 冲突

用法：
    wrapper = StarVLAPhysicsWrapper.load(ckpt_path, device="cuda")
    # 推理
    a_pred = wrapper.predict(examples, z_phy)    # → [B, 8, 7]
    # 训练前向
    loss   = wrapper.training_forward(examples, z_phy, a_gt)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

# ── 添加 StarVLA 第三方代码到 sys.path ──────────────────────────────────────
_PHYVLA_ROOT  = Path(__file__).resolve().parents[2]
_STARVLA_ROOT = _PHYVLA_ROOT / "third_party" / "starVLA"
_STARVLA_WS   = Path("/home/jiangnan-luxitech/workspace/starVLA")  # 权重/基底模型路径

for _p in [str(_STARVLA_ROOT), str(_STARVLA_WS)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.physics_vla_projector import PhysicsTokenProjector   # noqa: E402

# StarVLA 导入（需要 starVLA conda 环境）
try:
    from starVLA.model.framework.base_framework import baseframework
    _STARVLA_AVAILABLE = True
except ImportError:
    _STARVLA_AVAILABLE = False
    print("[StarVLAPhysicsWrapper] ⚠️  starVLA 未安装，仅投影器可用")


# ─────────────────────────────────────────────────────────────────────────────
# 工具：从 QwenGR00T 提取 VLM 隐藏状态（绕过 @inference_mode 的训练路径）
# ─────────────────────────────────────────────────────────────────────────────

def _encode_vlm(vla_model, examples: List[dict]) -> tuple:
    """提取 Qwen-VL last_hidden_state + attention_mask（不运行 action head）。

    Returns:
        last_hidden:  [B, L, H]  float32/bfloat16
        attn_mask:    [B, L]     bool | None
    """
    from deployment.model_server.tools.image_tools import to_pil_preserve
    from starVLA.training.trainer_utils.trainer_tools import resize_images

    batch_images   = [to_pil_preserve(ex["image"]) for ex in examples]
    instructions   = [ex["lang"]  for ex in examples]

    train_size = getattr(
        getattr(getattr(vla_model.config, "datasets", None), "vla_data", None),
        "obs_image_size", None,
    )
    if train_size:
        batch_images = resize_images(batch_images, target_size=train_size)

    qwen_inputs = vla_model.qwen_vl_interface.build_qwenvl_inputs(
        images=batch_images, instructions=instructions
    )
    attn_mask = qwen_inputs.get("attention_mask", None)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = vla_model.qwen_vl_interface(
            **qwen_inputs,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )
    last_hidden = out.hidden_states[-1]   # [B, L, H]
    return last_hidden, attn_mask


# ─────────────────────────────────────────────────────────────────────────────
# 核心包装器
# ─────────────────────────────────────────────────────────────────────────────

class StarVLAPhysicsWrapper(nn.Module):
    """物理增强 StarVLA：VLM 冻结，physics projector 可训练。

    推理流程：
        images + lang + state
            → Qwen-VL encoder → last_hidden [B, L, H]
            ↓ (z_phy 可选)
        z_phy [B, N, 256]
            → PhysicsTokenProjector → z_proj [B, N, H]
            → concat → last_hidden_aug [B, L+N, H]
            → DiT action head (flow-matching)
            → a_pred [B, T_chunk, 7]
    """

    # 默认配置（与 StarVLA LIBERO-4in1 checkpoint 对齐）
    DEFAULT_VLA_HIDDEN  = 2048
    DEFAULT_PHY_DIM     = 256
    DEFAULT_NUM_PHY_TOK = 23

    def __init__(
        self,
        vla_model,                                          # baseframework (Qwen_GR00T)
        phy_token_dim:  int  = DEFAULT_PHY_DIM,
        vla_hidden_dim: int  = DEFAULT_VLA_HIDDEN,
        num_phy_tokens: int  = DEFAULT_NUM_PHY_TOK,
        freeze_action_head: bool = False,                   # P5-v1 仅训练 projector
    ) -> None:
        super().__init__()
        self.vla = vla_model

        # 冻结 VLM
        for p in self.vla.qwen_vl_interface.parameters():
            p.requires_grad_(False)

        # 可选冻结 action head
        if freeze_action_head:
            for p in self.vla.action_model.parameters():
                p.requires_grad_(False)

        # 唯一新增可训练模块
        self.phy_projector = PhysicsTokenProjector(
            phy_token_dim=phy_token_dim,
            vla_hidden_dim=vla_hidden_dim,
            num_phy_tokens=num_phy_tokens,
        )

        self.vla_hidden_dim = vla_hidden_dim
        self.action_dim     = self.vla.config.framework.action_model.action_dim      # 7
        self.action_horizon = self.vla.action_horizon                                # 8

    # ── 加载 ─────────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        ckpt_path: Union[str, Path],
        device: str = "cuda",
        freeze_action_head: bool = False,
        **kwargs,
    ) -> "StarVLAPhysicsWrapper":
        """从 StarVLA 格式 checkpoint 加载并包装。"""
        assert _STARVLA_AVAILABLE, "安装 starVLA conda 环境后再加载模型"
        ckpt_path = str(ckpt_path)
        print(f"[StarVLAPhysicsWrapper] 加载 VLA checkpoint: {ckpt_path}")
        vla = baseframework.from_pretrained(ckpt_path)
        vla = vla.to(device).eval()

        wrapper = cls(vla, freeze_action_head=freeze_action_head, **kwargs)
        wrapper = wrapper.to(device)
        print(
            f"[StarVLAPhysicsWrapper] 加载完成\n"
            f"  action_dim={wrapper.action_dim}  horizon={wrapper.action_horizon}\n"
            f"  projector 参数量 = {wrapper.phy_projector.n_params:,}"
        )
        return wrapper

    # ── 保存 / 恢复 projector ─────────────────────────────────────────────────

    def save_projector(self, path: Union[str, Path]) -> None:
        torch.save(self.phy_projector.state_dict(), str(path))

    def load_projector(self, path: Union[str, Path]) -> None:
        self.phy_projector.load_state_dict(
            torch.load(str(path), map_location="cpu")
        )
        print(f"[StarVLAPhysicsWrapper] Projector loaded from {path}")

    # ── 推理 ─────────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def predict(
        self,
        examples: List[dict],
        z_phy: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Args:
            examples: list of {"image": PIL | np.ndarray, "lang": str, "state": [7]}
            z_phy:    [B, N_phy, 256] float32 Tensor | None（不提供则退化为原始 VLA）

        Returns:
            normalized_actions: np.ndarray [B, action_horizon, 7]
        """
        state_raw = [ex.get("state") for ex in examples]
        state = None
        if state_raw[0] is not None:
            dev = next(self.vla.parameters()).device
            state = torch.from_numpy(np.array(state_raw)).to(dev, dtype=torch.float32)
            state = state.unsqueeze(1)   # [B, 1, 7]

        last_hidden, attn_mask = _encode_vlm(self.vla, examples)
        B = last_hidden.shape[0]

        if z_phy is not None:
            z_phy = z_phy.to(last_hidden.device, dtype=last_hidden.dtype)
            z_proj = self.phy_projector(z_phy)                      # [B, N, H]
            last_hidden = torch.cat([last_hidden, z_proj], dim=1)   # [B, L+N, H]
            if attn_mask is not None:
                phy_mask = torch.ones(
                    B, z_phy.shape[1],
                    dtype=torch.bool, device=attn_mask.device
                )
                attn_mask = torch.cat([attn_mask, phy_mask], dim=1)

        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=torch.bool)

        with torch.autocast("cuda", dtype=torch.float32):
            pred = self.vla.action_model.predict_action(
                last_hidden, state,
                encoder_attention_mask=attn_mask,
            )   # [B, action_horizon, 7]

        return pred.detach().cpu().numpy()

    # ── 训练前向 ─────────────────────────────────────────────────────────────

    def training_forward(
        self,
        last_hidden_cached: torch.Tensor,     # [B, L, H] 预缓存的 VLM 隐状态
        attn_mask_cached:   Optional[torch.Tensor],
        z_phy:              torch.Tensor,      # [B, N_phy, 256]
        a_gt:               torch.Tensor,      # [B, action_horizon, 7]
        state:              Optional[torch.Tensor] = None,
    ) -> dict:
        """
        使用预缓存的 VLM 隐状态训练（避免重复运行 VLM encoder）。

        Returns:
            {"action_loss": scalar Tensor, "pred_actions": [B, T, 7]}
        """
        B = last_hidden_cached.shape[0]
        dev = last_hidden_cached.device

        z_phy = z_phy.to(dev, dtype=last_hidden_cached.dtype)
        z_proj = self.phy_projector(z_phy)                                    # [B, N, H]
        last_hidden_aug = torch.cat([last_hidden_cached, z_proj], dim=1)     # [B, L+N, H]

        attn_mask_aug = attn_mask_cached
        if attn_mask_cached is not None:
            phy_mask = torch.ones(B, z_phy.shape[1],
                                  dtype=torch.bool, device=dev)
            attn_mask_aug = torch.cat([attn_mask_cached, phy_mask], dim=1)

        # flow-matching 训练前向（带 noise diffusion loss）
        with torch.autocast("cuda", dtype=torch.float32):
            action_loss = self.vla.action_model(
                last_hidden_aug, a_gt, state,
                encoder_attention_mask=attn_mask_aug,
            )

        return {"action_loss": action_loss}

    # ── 预缓存 VLM 隐状态 ────────────────────────────────────────────────────

    @torch.inference_mode()
    def cache_vlm_hidden(
        self, examples: List[dict]
    ) -> tuple:
        """一次性提取并缓存 VLM hidden states（VLM 冻结时只需运行一次）。

        Returns:
            last_hidden: [B, L, H] 移到 CPU 存储
            attn_mask:   [B, L]    移到 CPU | None
        """
        last_hidden, attn_mask = _encode_vlm(self.vla, examples)
        return last_hidden.cpu(), (attn_mask.cpu() if attn_mask is not None else None)

    def trainable_parameters(self):
        """只返回需要优化的参数（projector，以及未冻结的 action head）。"""
        for n, p in self.named_parameters():
            if p.requires_grad:
                yield p
