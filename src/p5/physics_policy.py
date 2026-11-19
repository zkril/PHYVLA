"""Serve the legacy P5 projector with the baseline OFT action head."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from deployment.model_server.policy_wrapper import PolicyServerWrapper
from p5.starvla_physics import StarVLAPhysicsWrapper


class PhysicsPolicy(PolicyServerWrapper):
    def __init__(self, ckpt_path, projector_path, tokenizer_path=None, device="cuda", allow_unpaired_tokenizer=False):
        projector_path = Path(projector_path).resolve(strict=True)
        super().__init__(ckpt_path=ckpt_path, device=device, use_bf16=False)
        if type(self._framework.action_model).__name__ != "L1RegressionActionHead":
            raise TypeError("This legacy P5 checkpoint requires L1RegressionActionHead")
        self.physics = StarVLAPhysicsWrapper(
            self._framework, freeze_action_head=True
        ).to(device).eval()
        self.physics.load_projector(projector_path)
        self.projector_path = str(projector_path)
        self.projector_sha256 = hashlib.sha256(projector_path.read_bytes()).hexdigest()
        self.encoder = None
        self.unpaired_experiment = False
        if tokenizer_path:
            from p5.online_physics import OnlinePhysicsEncoder
            self.encoder = OnlinePhysicsEncoder(tokenizer_path, device=device)
            pairing_path = projector_path.with_suffix(".json")
            if not pairing_path.is_file() and allow_unpaired_tokenizer:
                self.unpaired_experiment = True
                self.encoder.metadata["pairing_note"] = "Experimental: original tokenizer missing; using a different saved encoder"
            else:
                if not pairing_path.is_file():
                    raise ValueError("Online inference requires the projector/tokenizer pairing manifest")
                pairing = json.loads(pairing_path.read_text())
                if pairing.get("tokenizer_sha256") != self.encoder.metadata["sha256"]:
                    raise ValueError("Tokenizer does not match the projector training checkpoint")
                if pairing.get("projector_sha256") != self.projector_sha256:
                    raise ValueError("Projector checksum differs from pairing manifest")
                if Path(pairing["action_head_source"]).resolve() != Path(ckpt_path).resolve():
                    raise ValueError("Baseline action head differs from paired training")
                self.encoder.metadata["paired_projector_verified"] = True
                self.encoder.metadata["training_validation"] = pairing.get("validation")

    @property
    def metadata(self):
        return {
            **super().metadata,
            "policy_kind": ("phyvla_unpaired_legacy_experiment" if self.unpaired_experiment else
                            "phyvla_causal_pair_baseline_head" if self.encoder else "phyvla_legacy_projector_baseline_head"),
            "experimental_unpaired_tokenizer": self.unpaired_experiment,
            "projector_path": self.projector_path,
            "projector_sha256": self.projector_sha256,
            "physics_token_shape": [23, 256],
            "action_head_source": "baseline_checkpoint",
            "fusion_mode": "legacy_physics_queries",
            "image_conditioned": False,
            "online_physics_ready": self.encoder is not None,
            "tokenizer": self.encoder.metadata if self.encoder else None,
        }

    @torch.inference_mode()
    def predict_action(self, examples, unnorm_key=None, z_phy=None,
                       physics_state=None, **kwargs):
        if (z_phy is None) == (physics_state is None):
            raise ValueError("Provide exactly one of z_phy or physics_state")
        if physics_state is not None:
            if self.encoder is None:
                raise RuntimeError("Online physics requires --tokenizer-checkpoint; no random fallback")
            if len(examples) != 1:
                raise ValueError("Online physics accepts one environment per request")
            z_phy = self.encoder.encode(physics_state, examples[0]["lang"])
        else:
            z_phy = torch.as_tensor(np.asarray(z_phy), dtype=torch.float32)
        expected = (len(examples), 23, 256)
        if tuple(z_phy.shape) != expected or not torch.isfinite(z_phy).all():
            raise ValueError(f"Expected finite physics tokens {expected}, got {tuple(z_phy.shape)}")
        normalized = self.physics.predict(examples, z_phy=z_phy)
        if normalized.shape != (len(examples), self._action_chunk_size, 7):
            raise ValueError(f"Unexpected action shape: {normalized.shape}")
        if not np.isfinite(normalized).all():
            raise FloatingPointError("Nonfinite model actions")
        proc = self.get_norm_processor(unnorm_key)
        actions = np.stack([proc.unapply_actions(x) for x in normalized])
        if not np.isfinite(actions).all():
            raise FloatingPointError("Nonfinite unnormalized actions")
        return {"actions": actions, "physics_used": True,
                "projector_sha256": self.projector_sha256}
