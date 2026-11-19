"""Build all-demo causal features and train-only normalization."""
import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
from pathlib import Path
import sys, json, argparse, concurrent.futures
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from datasets.physics_v2 import features, read_episode, sha, SCHEMA

def worker(job):
    e, output = job
    if sha(e["path"]) != e["sha256"]: raise ValueError("Raw physics checksum changed")
    a=read_episode(e); f=features(a)
    ids=np.flatnonzero(a["observation_index_for_state"]>=0)
    assert np.all(a["observation_index_for_state"][ids]<ids)
    dest=Path(output)/e["suite"]/("task_"+str(e["task_id"]))/(e["demo_key"]+".npz")
    dest.parent.mkdir(parents=True,exist_ok=True)
    values={k:v[ids] for k,v in f.items()}
    np.savez_compressed(dest,**values, state_index=ids,
        observation_index=a["observation_index_for_state"][ids],
        expert_actions=a["expert_actions"][ids].astype(np.float32))
    result={k:e[k] for k in ("suite","task_id","demo_key","source","language","split")}
    result.update(path=str(dest),sha256=sha(dest),frames=len(ids),objects=f["objects"].shape[1])
    moments={}
    if e["split"]=="train":
        for k,v in values.items():
            v=v.reshape(-1,v.shape[-1]).astype(np.float64)
            moments[k]=(len(v),v.sum(0),np.square(v).sum(0))
    return result,moments

def main():
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--workers",type=int,default=8)
    args=p.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/"manifest.json").exists():raise ValueError("Use a fresh output directory")
    raw=json.loads((args.data/"manifest.json").read_text())
    validation=json.loads((args.data/"validation.json").read_text())
    assert raw["status"]=="completed" and validation["passed"] and len(raw["episodes"])==2000
    entries=[];stats={}
    with concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
        for e,mom in pool.map(worker,[(e,str(args.output)) for e in raw["episodes"]],chunksize=1):
            entries.append(e)
            for k,(n,s,ss) in mom.items():
                old=stats.get(k,(0,np.zeros_like(s),np.zeros_like(ss)))
                stats[k]=(old[0]+n,old[1]+s,old[2]+ss)
            if len(entries)%100==0:print("PREPARED",len(entries),flush=True)
    norm={}
    for k,(n,s,ss) in stats.items():
        mean=s/n;std=np.sqrt(np.maximum(ss/n-mean*mean,0))
        norm[k]={"count":n,"mean":mean.tolist(),"std":np.maximum(std,.01).tolist()}
    (args.output/"normalization.json").write_text(json.dumps(norm,indent=2))
    manifest={"schema":SCHEMA,"episodes":entries,"raw_manifest_sha256":sha(args.data/"manifest.json"),
        "feature_code_sha256":sha(ROOT/"src/datasets/physics_v2.py"),
        "normalization_sha256":sha(args.output/"normalization.json"),"normalization_split":"train",
        "frames":sum(e["frames"] for e in entries),"status":"completed",
        "max_objects":max(e["objects"] for e in entries)}
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print("PREPARATION_COMPLETE",manifest["frames"],flush=True)
if __name__=="__main__":main()
