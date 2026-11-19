"""Train a reproducible tokenizer/projector pair on original LIBERO physics.

Demonstration-level train/validation split; frozen baseline OFT action head.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "thirdparty/starVLA")]
from models.physics_tokenizer import PhysicalTokenizer
from models.physics_vla_projector import PhysicsTokenProjector

KEYS = ["robot_vec", "contacts", "contact_mask", "dyn_target", "dyn_rel_cont",
        "dyn_rel_mask", "risk_vec", "afford_vec", "obj_roles", "obj_cont", "obj_mask", "phase"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_torch(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def cpu_state(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def auxiliary(tokenizer, batch, z):
    valid = batch["obj_mask"] > .5
    objects = z[:, 7:][valid]
    truth = batch["obj_cont"][valid]
    contacts = batch["contacts"][:, :, 21:28] * batch["contact_mask"].unsqueeze(-1)
    targets = (contacts.amax(1) > .5).float()
    losses = {
        "robot_mse": F.mse_loss(tokenizer.decode_robot(z), batch["robot_vec"]),
        "phase_ce": F.cross_entropy(tokenizer.classify_phase(z), batch["phase"]),
        "contact_bce": F.binary_cross_entropy_with_logits(tokenizer.decode_contact_relations(z), targets),
        "object_pose_mse": F.mse_loss(tokenizer.obj_encoder.decode_pose(objects), truth[:, :9]),
        # pos(3), rot6d(6), linear/angular velocity(6), then material params(7).
        "object_param_mse": F.mse_loss(tokenizer.obj_encoder.decode_params(objects), truth[:, 15:22]),
        "object_role_ce": F.cross_entropy(tokenizer.obj_encoder.classify_role(objects), batch["obj_roles"][valid]),
        "risk_mse": F.mse_loss(tokenizer.decode_risk(z), batch["risk_vec"]),
    }
    return sum(losses.values())


def normalize_actions(proc, actions):
    # Invert the shared eval client's gripper mapping: env +/-1 -> server 0/1.
    values = actions.copy()
    values[:, 6] = (1.0-values[:, 6]) / 2.0
    parts, offset = {}, 0
    for key in proc.action_keys:
        width = proc.action_key_dims[key]
        parts[key] = values[:, offset:offset+width]
        offset += width
    transformed = proc.transform.apply(parts)
    normalized = np.concatenate([np.asarray(transformed[k]) for k in proc.action_keys], axis=-1)
    restored = proc.unapply_actions(normalized)
    np.testing.assert_allclose(restored, values, atol=2e-5, rtol=2e-5)
    np.testing.assert_array_equal(1-2*(restored[:, 6] > .5), np.where(actions[:, 6] > 0, 1, -1))
    return normalized.astype(np.float32)


def load_split(manifest, demo_indices, proc, device):
    buffers = {k: [] for k in KEYS + ["actions"]}
    episodes = [e for e in manifest["episodes"] if e["demo_index"] in demo_indices]
    for entry in episodes:
        path = Path(entry["path"])
        if sha(path) != entry["sha256"]:
            raise ValueError(f"Feature cache changed: {path}")
        with np.load(path, allow_pickle=False) as arrays:
            for key in KEYS:
                buffers[key].append(arrays[key])
            normalized = normalize_actions(proc, arrays["expert_actions"])
            # Hold the final expert action when a chunk crosses an episode boundary.
            indices = np.minimum(np.arange(len(normalized))[:, None]+np.arange(8), len(normalized)-1)
            buffers["actions"].append(normalized[indices])
    if not episodes:
        raise ValueError("Empty data split")
    tensors = {k: torch.as_tensor(np.concatenate(v), dtype=torch.long if k in ("phase", "obj_roles") else torch.float32).to(device) for k, v in buffers.items()}
    return tensors, episodes


def batches(data, batch_size, shuffle):
    count = len(data["phase"])
    indices = torch.randperm(count, device=data["phase"].device) if shuffle else torch.arange(count, device=data["phase"].device)
    for start in range(0, count, batch_size):
        ids = indices[start:start+batch_size]
        yield {k: v[ids] for k, v in data.items()}


@torch.no_grad()
def evaluate(tokenizer, data, batch_size, projector=None, head=None):
    tokenizer.eval()
    if projector is not None:
        projector.eval()
    sums = {"aux_loss": 0.0, "action_l1": 0.0, "robot_mse": 0.0,
            "phase_accuracy": 0.0, "object_param_mse": 0.0}
    count = 0
    for batch in batches(data, batch_size, False):
        z = tokenizer(batch)
        n = len(z)
        valid = batch["obj_mask"] > .5
        sums["aux_loss"] += float(auxiliary(tokenizer, batch, z))*n
        sums["robot_mse"] += float(F.mse_loss(tokenizer.decode_robot(z), batch["robot_vec"]))*n
        sums["phase_accuracy"] += float((tokenizer.classify_phase(z).argmax(-1)==batch["phase"]).float().mean())*n
        sums["object_param_mse"] += float(F.mse_loss(
            tokenizer.obj_encoder.decode_params(z[:, 7:][valid]), batch["obj_cont"][valid][:, 15:22]))*n
        if head is not None:
            predicted = head.predict_action(projector(z)[:, :8])
            sums["action_l1"] += float(F.l1_loss(predicted, batch["actions"]))*n
        count += n
    return {k: v/count for k, v in sums.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initial-projector", type=Path)
    parser.add_argument("--train-demos", type=int, nargs="+", default=[0])
    parser.add_argument("--val-demos", type=int, nargs="+", default=[1])
    parser.add_argument("--pretrain-epochs", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if set(args.train_demos) & set(args.val_demos):
        parser.error("Training and validation demonstrations must be disjoint")
    for attr in ("data", "output", "checkpoint", "initial_projector"):
        if getattr(args, attr) is not None:
            setattr(args, attr, getattr(args, attr).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/"summary.json").exists():
        parser.error("Completed output already exists; choose a new run directory")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    os.chdir(ROOT/"thirdparty/starVLA")
    from deployment.model_server.policy_norm_processor import PolicyNormProcessor
    from starVLA.model.framework.base_framework import baseframework
    proc = PolicyNormProcessor(str(args.checkpoint))
    manifest = json.loads((args.data/"manifest.json").read_text())
    expected = {(s, t, d) for s in ("libero_spatial","libero_object","libero_goal","libero_10")
                for t in range(10) for d in args.train_demos+args.val_demos}
    actual = {(e["suite"],e["task_id"],e["demo_index"]) for e in manifest["episodes"]}
    if manifest["feature_schema"] != "causal_hold_v1" or not expected <= actual:
        raise ValueError(f"Incomplete 40-task feature manifest; missing {sorted(expected-actual)}")
    train, training_episodes = load_split(manifest, args.train_demos, proc, args.device)
    val, validation_episodes = load_split(manifest, args.val_demos, proc, args.device)
    print(json.dumps({"train_frames": len(train["phase"]), "val_frames": len(val["phase"]),
                      "train_episodes": len(training_episodes), "val_episodes": len(validation_episodes)}), flush=True)
    tokenizer = PhysicalTokenizer().to(args.device)
    optimizer = torch.optim.AdamW(tokenizer.parameters(), lr=3e-4)
    history = []
    best_aux = float("inf")
    for epoch in range(1, args.pretrain_epochs+1):
        tokenizer.train()
        for batch in batches(train, args.batch_size, True):
            optimizer.zero_grad(set_to_none=True)
            loss = auxiliary(tokenizer, batch, tokenizer(batch))
            if not torch.isfinite(loss): raise FloatingPointError("Tokenizer loss is nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), 1)
            optimizer.step()
        metrics = evaluate(tokenizer, val, args.batch_size)
        record = {"stage":"tokenizer","epoch":epoch, **metrics}
        history.append(record); print(json.dumps(record), flush=True)
        if metrics["aux_loss"] < best_aux:
            best_aux = metrics["aux_loss"]
            save_torch({"tokenizer_state_dict":cpu_state(tokenizer), "feature_schema":"causal_hold_v1",
                        "seed":args.seed, "validation":metrics}, args.output/"tokenizer_pretrained.pt")
    tokenizer.load_state_dict(torch.load(args.output/"tokenizer_pretrained.pt", weights_only=True)["tokenizer_state_dict"])
    vla = baseframework.from_pretrained(str(args.checkpoint)).to(args.device).eval()
    for parameter in vla.parameters(): parameter.requires_grad_(False)
    head = vla.action_model
    if type(head).__name__ != "L1RegressionActionHead":
        raise ValueError("This pipeline requires the baseline OFT L1 head")
    projector = PhysicsTokenProjector(256, 2560, 23).to(args.device)
    if args.initial_projector:
        projector.load_state_dict(torch.load(args.initial_projector, map_location="cpu", weights_only=True))
    optimizer = torch.optim.AdamW([
        {"params": tokenizer.parameters(), "lr": 3e-5},
        {"params": projector.parameters(), "lr": 1e-4}])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    initial = evaluate(tokenizer, val, args.batch_size, projector, head)
    print(json.dumps({"stage":"initial_pair", **initial}), flush=True)
    best = float("inf"); best_epoch = 0
    for epoch in range(1, args.epochs+1):
        tokenizer.train(); projector.train()
        total = 0.0; count = 0
        for batch in batches(train, args.batch_size, True):
            optimizer.zero_grad(set_to_none=True)
            z = tokenizer(batch)
            predicted = head.predict_action(projector(z)[:, :8])
            action_loss = F.l1_loss(predicted, batch["actions"])
            loss = action_loss + .1*auxiliary(tokenizer, batch, z)
            if not torch.isfinite(loss): raise FloatingPointError("Pair loss is nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(tokenizer.parameters())+list(projector.parameters()), 1)
            optimizer.step()
            total += float(action_loss)*len(z); count += len(z)
        scheduler.step()
        metrics = evaluate(tokenizer, val, args.batch_size, projector, head)
        record = {"stage":"paired", "epoch":epoch, "train_action_l1":total/count, **metrics}
        history.append(record); print(json.dumps(record), flush=True)
        (args.output/"history.json").write_text(json.dumps(history, indent=2))
        if metrics["action_l1"] < best:
            best = metrics["action_l1"]; best_epoch = epoch
            save_torch({"tokenizer_state_dict":cpu_state(tokenizer), "feature_schema":"causal_hold_v1",
                        "seed":args.seed, "epoch":epoch, "validation":metrics}, args.output/"tokenizer_best.pt")
            save_torch(cpu_state(projector), args.output/"projector_best.pt")
            pairing = {"tokenizer_sha256":sha(args.output/"tokenizer_best.pt"),
                       "projector_sha256":sha(args.output/"projector_best.pt"),
                       "feature_schema":"causal_hold_v1", "action_head_source":str(args.checkpoint),
                       "fusion_mode":"legacy_physics_queries", "epoch":epoch,
                       "validation":metrics, "supervision":"normalized_expert_action_chunks"}
            (args.output/"projector_best.json").write_text(json.dumps(pairing,indent=2))
        if epoch % 10 == 0:
            save_torch({"tokenizer_state_dict":cpu_state(tokenizer),"projector_state_dict":cpu_state(projector),
                        "optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"epoch":epoch,
                        "feature_schema":"causal_hold_v1","torch_rng_state":torch.get_rng_state()},
                       args.output/"training_last.pt")
    tokenizer.load_state_dict(torch.load(args.output/"tokenizer_best.pt",weights_only=True)["tokenizer_state_dict"])
    tokenizer.eval()
    # Export all train/validation episode tokens from this exact saved encoder.
    for entry in training_episodes+validation_episodes:
        with np.load(entry["path"],allow_pickle=False) as arrays:
            data={k:torch.as_tensor(arrays[k], dtype=torch.long if k in ("phase", "obj_roles") else torch.float32).to(args.device) for k in KEYS}
        result=[]
        with torch.no_grad():
            for batch in batches(data,args.batch_size,False): result.append(tokenizer(batch).cpu().numpy())
        path=args.output/"tokens"/entry["suite"]/f'task_{entry["task_id"]}'/f'demo_{entry["demo_index"]}.npz'
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,tokens=np.concatenate(result),tokenizer_sha256=sha(args.output/"tokenizer_best.pt"))
    summary={"complete":True, "best_epoch":best_epoch, "best_val_action_l1":best,
             "initial_val_action_l1":initial["action_l1"], "train_demo_indices":args.train_demos,
             "validation_demo_indices":args.val_demos, "tasks":40, "feature_schema":"causal_hold_v1",
             "training_episodes":len(training_episodes),"validation_episodes":len(validation_episodes),
             "baseline_checkpoint":str(args.checkpoint), "initial_projector":str(args.initial_projector),
             "limitation":"Legacy physics-only action queries; 1 train and 1 validation demo per task by default; not a full benchmark training run",
             "pair":json.loads((args.output/"projector_best.json").read_text())}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2))
    print("PAIRED_TRAINING_COMPLETE",json.dumps(summary),flush=True)

if __name__ == "__main__":
    main()
