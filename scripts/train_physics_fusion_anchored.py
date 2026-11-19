"""Experimental baseline-anchored fusion; preserves v2 checkpoint compatibility."""
from pathlib import Path
import sys,argparse,json,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"thirdparty/starVLA")]
from models.physics_v2 import PhysicsTokenizerV2,PhysicsResidualFusion
from datasets.physics_v2 import sha,SCHEMA
from starVLA.model.modules.action_model.MLP_ActionHeader import L1RegressionActionHead

def load_data(manifest,norm,device="cuda"):
    maximum=0;buffers={}
    for split in ("train","validation"):
        rows=[]
        for e in manifest["episodes"]:
            if e["split"]!=split:continue
            if sha(e["path"])!=e["sha256"]:raise ValueError("Query cache checksum mismatch")
            with np.load(e["path"]) as a,np.load(e["feature_path"]) as f:
                ids=a["feature_index"]
                row={k:a[k] for k in ("queries","targets","action_mask","baseline_actions")}
                for k in ("robot","objects"):
                    row[k]=(f[k][ids]-np.array(norm[k]["mean"],np.float32))/np.array(norm[k]["std"],np.float32)
                row["suite"]=np.full(len(ids),("libero_spatial","libero_object","libero_goal","libero_10").index(e["suite"]))
                maximum=max(maximum,row["objects"].shape[1]);rows.append(row)
        buffers[split]=rows
    result={}
    for split,rows in buffers.items():
        for r in rows:
            obj=r["objects"];pad=np.zeros((len(obj),maximum,obj.shape[-1]),np.float32)
            pad[:,:obj.shape[1]]=obj
            mask=np.zeros(pad.shape[:2],np.float32);mask[:,:obj.shape[1]]=1
            r["objects"]=pad;r["mask"]=mask
        result[split]={k:torch.from_numpy(np.concatenate([r[k] for r in rows])).to(device) for k in rows[0]}
    return result

def iterate(data,size,shuffle=False):
    ids=torch.randperm(len(data["robot"]),device=data["robot"].device) if shuffle else torch.arange(len(data["robot"]),device=data["robot"].device)
    for ix in ids.split(size):yield {k:v[ix] for k,v in data.items()}

def action_loss(pred,b):
    return ((pred-b["targets"]).abs()*b["action_mask"]).sum()/b["action_mask"].sum()

@torch.no_grad()
def evaluate(tokenizer,fusion,head,data,size,mode="full"):
    tokenizer.eval();fusion.eval();head.eval()
    source=data
    if mode=="shuffled":
        generator=torch.Generator(device=data["robot"].device).manual_seed(101)
        perm=torch.randperm(len(data["robot"]),device=data["robot"].device,generator=generator)
        source={**data,**{k:data[k][perm] for k in ("robot","objects","mask")}}
    errors=torch.zeros(4,7,device=data["robot"].device);counts=torch.zeros_like(errors)
    action_delta=0.0;action_cells=0
    for b in iterate(source,size):
        if mode=="no_physics":
            q=b["queries"]
        else:
            z,valid=tokenizer(b["robot"],b["objects"],b["mask"])
            q=fusion(b["queries"],z,valid)
        pred=head.predict_action(q)
        action_delta+=float((pred-b["baseline_actions"]).abs().sum())
        action_cells+=pred.numel()
        for suite in range(4):
            selected=b["suite"]==suite
            errors[suite]+=((pred[selected]-b["targets"][selected]).abs()*b["action_mask"][selected]).sum((0,1))
            counts[suite]+=b["action_mask"][selected].sum((0,1))
    return {"l1":float(errors.sum()/counts.sum()),"mean_abs_action_delta_from_baseline":action_delta/action_cells,"suite_l1":(errors.sum(-1)/counts.sum(-1)).cpu().tolist(),
            "per_dimension_l1":(errors.sum(0)/counts.sum(0)).cpu().tolist()}

def main():
    p=argparse.ArgumentParser();p.add_argument("--queries",type=Path,required=True)
    p.add_argument("--tokenizer",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=20);p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--device",default="cuda")
    p.add_argument("--anchor-weight",type=float,default=0.25)
    p.add_argument("--fusion-lr",type=float,default=3e-5)
    args=p.parse_args()
    if args.anchor_weight < 0:raise ValueError("anchor-weight must be nonnegative")
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/"policy_best.pt").exists():raise ValueError("Use a fresh output directory")
    torch.manual_seed(7);np.random.seed(7);torch.set_num_threads(4)
    manifest=json.loads((args.queries/"manifest.json").read_text())
    assert manifest["status"]=="completed" and len(manifest["episodes"])==2000
    provenance=json.loads((args.queries/"baseline_provenance.json").read_text())
    assert provenance["path"]==manifest["baseline_path"] and sha(provenance["path"])==provenance["sha256"]
    for name,asset in provenance["assets"].items():
        assert sha(Path(provenance["path"]).parents[1]/name)==asset["sha256"]
    saved=torch.load(args.tokenizer,map_location="cpu",weights_only=False)
    assert saved["schema"]==SCHEMA and saved["manifest_sha256"]==manifest["feature_manifest_sha256"]
    feature_manifest_path=Path(manifest["episodes"][0]["feature_path"]).parents[2]/"manifest.json"
    if sha(feature_manifest_path)!=manifest["feature_manifest_sha256"]:
        raise ValueError("Feature manifest changed after query caching")
    feature_manifest=json.loads(feature_manifest_path.read_text())
    for entry in feature_manifest["episodes"]:
        if sha(entry["path"])!=entry["sha256"]:
            raise ValueError("Feature file checksum mismatch: "+entry["path"])
    norm=saved["normalization"];data=load_data(manifest,norm,args.device)
    tokenizer=PhysicsTokenizerV2(**saved["config"]).to(args.device)
    tokenizer.load_state_dict(saved["tokenizer_state_dict"])
    hc=torch.load(args.queries/"baseline_head.pt",map_location="cpu",weights_only=False)
    head=L1RegressionActionHead(input_dim=hc["hidden_dim"],hidden_dim=hc["hidden_dim"]*2,
                               action_dim=7,NUM_ACTIONS_CHUNK=hc["horizon"]).to(args.device).eval()
    head.load_state_dict(hc["state_dict"])
    for param in head.parameters():param.requires_grad_(False)
    fusion=PhysicsResidualFusion(hc["hidden_dim"],saved["config"]["width"]).to(args.device)
    with torch.no_grad():
        b=next(iterate(data["validation"],8));tokenizer.eval()
        z,valid=tokenizer(b["robot"],b["objects"],b["mask"])
        q=fusion(b["queries"],z,valid)
        torch.testing.assert_close(q,b["queries"],rtol=0,atol=0)
        torch.testing.assert_close(head.predict_action(q),b["baseline_actions"],rtol=1e-5,atol=1e-6)
    baseline=evaluate(tokenizer,fusion,head,data["validation"],args.batch_size,"no_physics")
    print("BASELINE",json.dumps(baseline),flush=True)
    optimizer=torch.optim.AdamW([{"params":fusion.parameters(),"lr":args.fusion_lr},
                                 {"params":tokenizer.parameters(),"lr":1e-5}],weight_decay=1e-4)
    history=[];best=float("inf");start=time.time()
    for epoch in range(1,args.epochs+1):
        tokenizer.train();fusion.train();total=0;count=0
        for b in iterate(data["train"],args.batch_size,True):
            optimizer.zero_grad(set_to_none=True)
            z,valid=tokenizer(b["robot"],b["objects"],b["mask"])
            prediction=head.predict_action(fusion(b["queries"],z,valid))
            bc=action_loss(prediction,b)
            robot_error=(tokenizer.robot_decoder(z[:,0])-b["robot"]).square().mean()
            object_error=(tokenizer.object_decoder(z[:,1:])-b["objects"]).square().mean(-1)
            auxiliary=robot_error+(object_error*b["mask"]).sum()/b["mask"].sum()
            # Penalize changes from the frozen baseline on the same observations.
            anchor=((prediction-b["baseline_actions"]).square()*b["action_mask"]).sum()/b["action_mask"].sum()
            loss=bc+args.anchor_weight*anchor+.01*auxiliary
            if not torch.isfinite(loss):raise ValueError("Nonfinite training loss")
            loss.backward();torch.nn.utils.clip_grad_norm_(list(tokenizer.parameters())+list(fusion.parameters()),1)
            optimizer.step();total+=float(bc)*len(z);count+=len(z)
        val=evaluate(tokenizer,fusion,head,data["validation"],args.batch_size)
        record={"epoch":epoch,"train_action_l1":total/count,"validation":val,"seconds":time.time()-start}
        print(json.dumps(record),flush=True);history.append(record)
        (args.output/"history.json").write_text(json.dumps(history,indent=2))
        payload={"schema":SCHEMA,"tokenizer_config":tokenizer.config,"fusion_config":fusion.config,
            "tokenizer_state_dict":tokenizer.state_dict(),"fusion_state_dict":fusion.state_dict(),
            "action_head_state_dict":head.state_dict(),"normalization":norm,"horizon":hc["horizon"],
            "baseline_path":manifest["baseline_path"],"baseline_sha256":provenance["sha256"],"baseline_assets":provenance["assets"],"query_manifest_sha256":sha(args.queries/"manifest.json"),
            "feature_code_sha256":saved["feature_code_sha256"],"epoch":epoch,"validation":val,
            "baseline_validation":baseline,"seed":7,
            "training_variant":"baseline_anchored_v1","anchor_weight":args.anchor_weight,"fusion_lr":args.fusion_lr,"fusion_mode":"visual_query_residual",
            "trainable_modules":["tokenizer","fusion"],"frozen_modules":["visual_language_backbone","action_head"]}
        if val["l1"]<best:
            best=val["l1"];torch.save(payload,args.output/"policy_best.pt")
        torch.save({**payload,"optimizer":optimizer.state_dict()},args.output/"training_last.pt")
    checkpoint=torch.load(args.output/"policy_best.pt",weights_only=False)
    tokenizer.load_state_dict(checkpoint["tokenizer_state_dict"])
    fusion.load_state_dict(checkpoint["fusion_state_dict"])
    metrics={mode:evaluate(tokenizer,fusion,head,data["validation"],args.batch_size,mode)
             for mode in ("full","no_physics","shuffled")}
    summary={"completed":True,"best_epoch":checkpoint["epoch"],"validation":metrics,
             "policy_sha256":sha(args.output/"policy_best.pt"),"zero_residual_baseline_parity":True,
             "train_samples":len(data["train"]["robot"]),"validation_samples":len(data["validation"]["robot"]),
             "tasks":40,"training_episodes":1800,"validation_episodes":200,
             "note":"Held-out action error does not establish closed-loop success improvement."}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2))
    print("FUSION_COMPLETE",json.dumps(summary),flush=True)
if __name__=="__main__":main()
