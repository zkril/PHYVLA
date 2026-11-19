"""Encode live snapshots; require a saved tokenizer, never random initialization."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from datasets.physics_dataset import PhysicsDataset, MAX_N_OBJECTS
from models.physics_tokenizer import PhysicalTokenizer


class OnlinePhysicsEncoder:
    def __init__(self, checkpoint, device="cuda"):
        path = Path(checkpoint).resolve(strict=True)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        # Live inference has no teacher-forced future frame. Only accept an
        # explicitly compatible checkpoint to avoid an unreported feature shift.
        if saved.get("feature_schema") != "causal_hold_v1":
            raise ValueError("Tokenizer checkpoint must declare feature_schema=causal_hold_v1")
        self.model = PhysicalTokenizer(token_dim=256, max_n_objects=MAX_N_OBJECTS)
        self.model.load_state_dict(saved["tokenizer_state_dict"], strict=True)
        self.model.to(device).eval()
        self.device = device
        self.metadata = {"checkpoint": str(path),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "feature_schema": "causal_hold_v1",
                         "paired_projector_verified": False}

    @torch.inference_mode()
    def encode(self, state, language):
        if state.get("feature_schema") != "causal_hold_v1":
            raise ValueError("Unsupported live physics schema")
        arrays = {k: np.asarray(v) for k, v in state["arrays"].items()}
        if arrays["contacts"].shape != (1, 16, 28) or arrays["contact_mask"].shape != (1, 16):
            raise ValueError("Live encoding expects one snapshot with 16 contacts of width 28")
        if any(not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError("Nonfinite live physics state")
        events = {k: np.asarray(v) for k, v in state["events"].items()}
        dataset = PhysicsDataset.from_arrays(arrays, events, language)
        if not 0 < len(dataset.obj_keys) <= MAX_N_OBJECTS:
            raise ValueError(f"Unsupported object count: {len(dataset.obj_keys)}")
        batch = {k: v.unsqueeze(0).to(self.device) for k, v in dataset[0].items()}
        # T=1 uses current state for t_fut, hence zero dynamic deltas, no future leakage.
        tokens = self.model(batch)
        if tokens.shape != (1, 23, 256) or not torch.isfinite(tokens).all():
            raise ValueError("Invalid encoded physics tokens")
        return tokens
