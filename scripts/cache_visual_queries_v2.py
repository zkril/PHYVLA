"""Cache exact frozen-baseline visual action queries on aligned demonstrations."""
from pathlib import Path
import sys,os,argparse,json,time
import numpy as np
import torch,h5py
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"thirdparty/starVLA"),str(ROOT/"scripts")]
from datasets.physics_v2 import sha
from train_online_physics import normalize_actions

def main():
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--batch-size",type=int,default=8);p.add_argument("--stride",type=int,default=8)
    p.add_argument("--limit-episodes",type=int)
    args=p.parse_args()
    args.data=args.data.resolve();args.output=args.output.resolve();args.checkpoint=args.checkpoint.resolve()
    args.output.mkdir(parents=True,exist_ok=True);os.chdir(ROOT/"thirdparty/starVLA")
    torch.set_num_threads(4);torch.manual_seed(7)
    from deployment.model_server.policy_wrapper import PolicyServerWrapper
    policy=PolicyServerWrapper(str(args.checkpoint));vla=policy._framework
    for param in vla.parameters():param.requires_grad_(False)
    proc=policy.get_norm_processor();horizon=vla.action_horizon
    if args.stride>horizon:raise ValueError("Stride may not leave gaps in action supervision")
    head=vla.action_model
    torch.save({"state_dict":head.state_dict(),"hidden_dim":head.model.fc1.in_features,
                "horizon":horizon},args.output/"baseline_head.pt")
    manifest=json.loads((args.data/"manifest.json").read_text())
    entries=manifest["episodes"]
    if args.limit_episodes:entries=entries[:args.limit_episodes]
    cached=[];start=time.time();capture={}
    original=head.predict_action
    def recording(queries):
        capture["queries"]=queries.detach().float().cpu().numpy()
        return original(queries)
    head.predict_action=recording
    provenance={"baseline_path":str(args.checkpoint),"feature_manifest_sha256":sha(args.data/"manifest.json"),
                "stride":args.stride,"horizon":horizon,"image_transform":"raw HDF5 rotate180; PIL bilinear224; baseline predict_action"}
    for i,e in enumerate(entries):
        dest=args.output/e["suite"]/("task_"+str(e["task_id"]))/(e["demo_key"]+".npz")
        dest.parent.mkdir(parents=True,exist_ok=True)
        side=dest.with_suffix(".json")
        if dest.exists() and side.exists():
            info=json.loads(side.read_text())
            if info["provenance"]!=provenance or sha(dest)!=info["sha256"]:raise ValueError("Stale query cache")
            cached.append(info);continue
        with np.load(e["path"]) as a:
            ids=np.arange(0,len(a["state_index"]),args.stride)
            obs=a["observation_index"][ids];state=a["state_index"][ids]
            truth=normalize_actions(proc,a["expert_actions"])
            target_idx=ids[:,None]+np.arange(horizon)
            mask=np.broadcast_to((target_idx<len(truth))[...,None],(*target_idx.shape,7)).copy()
            target=truth[np.minimum(target_idx,len(truth)-1)]
        queries=[];predictions=[]
        with h5py.File(e["source"],"r") as h:
            g=h["data"][e["demo_key"]]["obs"]
            for j in range(0,len(ids),args.batch_size):
                examples=[]
                for oi in obs[j:j+args.batch_size]:
                    images=[np.asarray(Image.fromarray(np.ascontiguousarray(g[key][oi][::-1,::-1])).resize((224,224),Image.BILINEAR))
                            for key in ("agentview_rgb","eye_in_hand_rgb")]
                    examples.append({"image":images,"lang":e["language"]})
                pred=vla.predict_action(examples)["normalized_actions"]
                q=capture["queries"]
                # Verify cache reproduces exact baseline action-head output.
                with torch.no_grad():
                    replay=head.predict_action(torch.from_numpy(q).cuda()).cpu().numpy()
                np.testing.assert_allclose(pred,replay,rtol=1e-5,atol=1e-6)
                queries.append(q);predictions.append(pred)
        np.savez_compressed(dest,queries=np.concatenate(queries),baseline_actions=np.concatenate(predictions),
            targets=target,action_mask=mask,feature_index=ids,state_index=state,observation_index=obs)
        info={k:e[k] for k in ("suite","task_id","demo_key","split")}
        info.update(path=str(dest),feature_path=e["path"],sha256=sha(dest),samples=len(ids),provenance=provenance)
        side.write_text(json.dumps(info,indent=2));cached.append(info)
        report={"status":"running","episodes":cached,**provenance}
        (args.output/"manifest.json").write_text(json.dumps(report,indent=2))
        if (i+1)%10==0:print(json.dumps({"episodes":i+1,"samples":sum(x["samples"] for x in cached),
                                        "seconds":time.time()-start}),flush=True)
    # Both training and held-out measurements are computed without updating baseline.
    metrics={}
    for split in ("train","validation"):
        error=np.zeros(7);count=np.zeros(7)
        for e in cached:
            if e["split"]!=split:continue
            with np.load(e["path"]) as a:
                error+=(np.abs(a["baseline_actions"]-a["targets"])*a["action_mask"]).sum((0,1))
                count+=a["action_mask"].sum((0,1))
        if count.min()>0:metrics[split]={"l1":float(error.sum()/count.sum()),"per_dimension_l1":(error/count).tolist()}
    report.update(status="completed",metrics=metrics,episodes=cached)
    (args.output/"manifest.json").write_text(json.dumps(report,indent=2))
    print("QUERY_CACHE_COMPLETE",json.dumps(metrics),flush=True)
if __name__=="__main__":main()
