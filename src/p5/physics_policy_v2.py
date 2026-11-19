"""Baseline visual policy with checkpointed causal-physics residual fusion."""
from pathlib import Path
from threading import RLock
import numpy as np
import torch
from datasets.physics_v2 import SCHEMA,sha
from models.physics_v2 import PhysicsTokenizerV2,PhysicsResidualFusion
from deployment.model_server.policy_wrapper import PolicyServerWrapper

class PhysicsPolicyV2(PolicyServerWrapper):
    def __init__(self,checkpoint,physics_checkpoint,mode="full",device="cuda",residual_scale=1.0):
        if not 0 <= residual_scale <= 1:raise ValueError("residual_scale must be within [0, 1]")
        self.residual_scale=float(residual_scale)
        path=Path(physics_checkpoint).resolve(strict=True)
        saved=torch.load(path,map_location="cpu",weights_only=False)
        if saved["schema"]!=SCHEMA or saved["fusion_mode"]!="visual_query_residual":
            raise ValueError("Incompatible physics checkpoint")
        if Path(saved["baseline_path"]).resolve()!=Path(checkpoint).resolve():
            raise ValueError("Baseline path differs from training")
        if saved["feature_code_sha256"]!=sha(Path(__file__).parents[1]/"datasets/physics_v2.py"):
            raise ValueError("Feature implementation differs from training")
        if sha(checkpoint)!=saved["baseline_sha256"]:
            raise ValueError("Baseline weight checksum differs from training")
        for name,asset in saved["baseline_assets"].items():
            if sha(Path(checkpoint).parents[1]/name)!=asset["sha256"]:
                raise ValueError("Baseline configuration/normalization changed: "+name)
        super().__init__(ckpt_path=str(checkpoint),device=device,use_bf16=False)
        if saved["horizon"]!=self._action_chunk_size:
            raise ValueError("Action horizon differs from training")
        self.tokenizer=PhysicsTokenizerV2(**saved["tokenizer_config"]).to(device).eval()
        self.tokenizer.load_state_dict(saved["tokenizer_state_dict"])
        self.fusion=PhysicsResidualFusion(**saved["fusion_config"]).to(device).eval()
        self.fusion.load_state_dict(saved["fusion_state_dict"])
        # Check the frozen head instead of silently replacing baseline weights.
        for k,v in self._framework.action_model.state_dict().items():
            torch.testing.assert_close(v.cpu(),saved["action_head_state_dict"][k],rtol=0,atol=0)
        self.norm=saved["normalization"];self.device=device;self.mode=mode
        self.baseline_sha=saved["baseline_sha256"]
        self.policy_sha=sha(path);self.physics_path=str(path);self.lock=RLock()

    @property
    def metadata(self):
        return {**super().metadata,"policy_kind":"phyvla_v2","phyvla_enabled":True,
            "online_physics_ready":True,"feature_schema":SCHEMA,"image_conditioned":True,
            "fusion_mode":"visual_query_residual","physics_mode":self.mode,
            "residual_scale":self.residual_scale,"physics_checkpoint":self.physics_path,"projector_sha256":self.policy_sha,
            "paired_tokenizer_verified":True,"use_bf16":False,"seed":7,
            "baseline_sha256":self.baseline_sha,"vlm_autocast_dtype":"bfloat16",
            "action_head_dtype":str(next(self._framework.action_model.parameters()).dtype)}

    @torch.inference_mode()
    def predict_action(self,examples,physics_state=None,unnorm_key=None,**kwargs):
        if len(examples)!=1:raise ValueError("One environment per request required")
        if physics_state is None:raise ValueError("Missing causal physical features")
        tensors={}
        for key in ("robot","objects"):
            value=np.asarray(physics_state[key],np.float32)
            if not np.isfinite(value).all():raise ValueError("Nonfinite physics")
            cfg=self.norm[key]
            value=(value-np.array(cfg["mean"],np.float32))/np.array(cfg["std"],np.float32)
            tensors[key]=torch.from_numpy(value).to(self.device)
        mask=torch.ones(tensors["objects"].shape[:2],device=self.device)
        z,valid=self.tokenizer(tensors["robot"],tensors["objects"],mask)
        with self.lock:
            head=self._framework.action_model;original=head.predict_action
            def fused(queries):
                if self.mode=="no_physics":return original(queries)
                return original(self.fusion(queries.float(),z,valid,scale=self.residual_scale))
            head.predict_action=fused
            try:
                result=super().predict_action(examples,unnorm_key=unnorm_key,**kwargs)
            finally:
                head.predict_action=original
        return {**result,"physics_used":self.mode=="full","physics_mode":self.mode,
                "projector_sha256":self.policy_sha}
