#!/usr/bin/env python3
"""PhyVLA Policy Server Wrapper
在StarVLA官方server架构上注入物理token
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "starVLA"))
sys.path.insert(0, "/home/jiangnan-luxitech/workspace/starVLA")

import os
os.chdir("/home/jiangnan-luxitech/workspace/starVLA")

import argparse
import logging
import socket
import torch
import numpy as np
from typing import Optional

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from deployment.model_server.seed_utils import set_seed_everywhere

# PhyVLA imports
from p5.starvla_physics import StarVLAPhysicsWrapper
from models.physics_vla_projector import PhysicsTokenProjector


class PhyVLAPolicyWrapper(PolicyServerWrapper):
    """PhyVLA增强的Policy Wrapper

    在原始PolicyServerWrapper基础上：
    1. 加载physics projector
    2. 接收client发送的physics_tokens
    3. 注入到VLM hidden states
    """

    def __init__(
        self,
        ckpt_path: str,
        projector_ckpt: str,
        device: str = "cuda",
        use_bf16: bool = False,
        config_overrides: list = None,
    ):
        # 调用父类初始化（加载StarVLA）
        super().__init__(
            ckpt_path=ckpt_path,
            device=device,
            use_bf16=use_bf16,
            config_overrides=config_overrides or [],
        )

        # 加载physics projector
        self.physics_projector = PhysicsTokenProjector(
            phy_token_dim=256,
            vla_hidden_dim=2560,
            num_phy_tokens=23,
        ).to(device)

        if Path(projector_ckpt).exists():
            ckpt = torch.load(projector_ckpt, map_location=device)
            self.physics_projector.load_state_dict(ckpt["model_state_dict"])
            logging.info(f"[PhyVLA] Loaded projector from {projector_ckpt} (epoch {ckpt['epoch']}, loss={ckpt['loss']:.4f})")
        else:
            logging.warning(f"[PhyVLA] Projector not found: {projector_ckpt}, using random weights")

        self.physics_projector.eval()

        # 更新metadata（告诉client需要发送physics_tokens）
        self.metadata["phyvla_enabled"] = True
        self.metadata["physics_token_dim"] = 256
        self.metadata["num_physics_tokens"] = 23

        logging.info("[PhyVLA] Wrapper initialized with physics augmentation")

    @torch.inference_mode()
    def predict(self, examples: list, unnorm_key: Optional[str] = None, **kwargs) -> dict:
        """预测动作（支持physics_tokens注入）

        Args:
            examples: list of dicts with keys:
                - image: list of PIL/np images
                - lang: str
                - physics_tokens: Optional[np.ndarray] shape [23, 256]
        """
        # 检查是否有physics_tokens
        physics_tokens = None
        if examples and "physics_tokens" in examples[0]:
            physics_tokens_list = [ex.get("physics_tokens") for ex in examples]
            if physics_tokens_list[0] is not None:
                physics_tokens = torch.from_numpy(
                    np.stack(physics_tokens_list)
                ).float().to(self.device)  # [B, 23, 256]

        # 如果有physics_tokens，使用增强推理
        if physics_tokens is not None:
            return self._predict_with_physics(examples, physics_tokens, unnorm_key, **kwargs)
        else:
            # 降级为baseline（无physics）
            logging.warning("[PhyVLA] No physics_tokens provided, falling back to baseline")
            return super().predict(examples, unnorm_key, **kwargs)

    def _predict_with_physics(
        self,
        examples: list,
        physics_tokens: torch.Tensor,
        unnorm_key: Optional[str] = None,
        **kwargs
    ) -> dict:
        """物理增强推理"""
        # 1. 编码VLM
        from deployment.model_server.tools.image_tools import to_pil_preserve
        from starVLA.training.trainer_utils.trainer_tools import resize_images

        batch_images = [to_pil_preserve(ex["image"]) for ex in examples]
        instructions = [ex["lang"] for ex in examples]

        train_size = getattr(
            getattr(getattr(self.vla.config, "datasets", None), "vla_data", None),
            "obs_image_size", None,
        )
        if train_size:
            batch_images = resize_images(batch_images, target_size=train_size)

        qwen_inputs = self.vla.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions
        )
        attn_mask = qwen_inputs.get("attention_mask", None)

        with torch.autocast("cuda", dtype=torch.bfloat16 if self.use_bf16 else torch.float32):
            out = self.vla.qwen_vl_interface(
                **qwen_inputs,
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True,
            )

        last_hidden = out.hidden_states[-1]  # [B, L, H]

        # 2. 投影physics tokens
        z_proj = self.physics_projector(physics_tokens)  # [B, 23, H]

        # 3. 拼接
        B = last_hidden.shape[0]
        last_hidden_aug = torch.cat([last_hidden, z_proj], dim=1)  # [B, L+23, H]

        if attn_mask is not None:
            phy_mask = torch.ones(B, physics_tokens.shape[1], dtype=torch.bool, device=attn_mask.device)
            attn_mask_aug = torch.cat([attn_mask, phy_mask], dim=1)
        else:
            attn_mask_aug = None

        # 4. DiT action head
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.vla.action_model.predict_action(
                last_hidden_aug,
                state=None,
                encoder_attention_mask=attn_mask_aug,
            )  # [B, horizon, 7]

        # 5. 反归一化
        pred_actions_np = pred_actions.detach().cpu().numpy()

        if unnorm_key:
            pred_actions_np = self._unnormalize_actions(pred_actions_np, unnorm_key)

        return {"actions": pred_actions_np}


def main(args):
    seed = getattr(args, "seed", None)
    if seed is not None:
        set_seed_everywhere(seed)

    config_overrides = getattr(args, "config_override", [])
    if config_overrides:
        override_keys = [item.split("=", 1)[0] for item in config_overrides]
        logging.info("Applying config override keys: %s", override_keys)

    # 使用PhyVLA wrapper
    wrapper = PhyVLAPolicyWrapper(
        ckpt_path=args.ckpt_path,
        projector_ckpt=args.projector_ckpt,
        device="cuda",
        use_bf16=args.use_bf16,
        config_overrides=config_overrides,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating PhyVLA server (host: %s, ip: %s)", hostname, local_ip)
    logging.info("PhyVLA metadata: %s", wrapper.metadata)

    # 启动websocket server
    metadata = dict(wrapper.metadata)
    metadata["seed"] = seed

    server = WebsocketPolicyServer(
        policy=wrapper,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=metadata,
    )

    logging.info("PhyVLA server running on port %d ... metadata=%s", args.port, metadata)
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PhyVLA Policy Server")
    parser.add_argument("--ckpt_path", type=str, required=True, help="StarVLA checkpoint")
    parser.add_argument("--projector_ckpt", type=str, required=True, help="Physics projector checkpoint")
    parser.add_argument("--port", type=int, default=6694)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--idle_timeout", type=int, default=1800)
    parser.add_argument(
        "--config_override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="OmegaConf dotlist override",
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parser.parse_args()
    main(args)
