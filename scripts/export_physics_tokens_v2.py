"""Export all episode tokens from the final paired physics policy checkpoint."""
from pathlib import Path
import sys,argparse,json
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from models.physics_v2 import PhysicsTokenizerV2
from datasets.physics_v2 import sha
def main():
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/"manifest.json").exists():raise ValueError("Use a fresh output directory")
    torch.set_num_threads(4)
    s=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    model=PhysicsTokenizerV2(**s["tokenizer_config"]).cuda().eval()
    model.load_state_dict(s["tokenizer_state_dict"]);norm=s["normalization"]
    source=json.loads((args.data/"manifest.json").read_text());result=[]
    checkpoint_hash=sha(args.checkpoint)
    for e in source["episodes"]:
        if sha(e["path"])!=e["sha256"]:raise ValueError("Changed feature data")
        with np.load(e["path"]) as a:
            b={k:torch.from_numpy((a[k]-np.array(norm[k]["mean"],np.float32))/np.array(norm[k]["std"],np.float32)).cuda()
               for k in ("robot","objects")}
            tokens=[]
            with torch.no_grad():
                for start in range(0,len(b["robot"]),512):
                    r=b["robot"][start:start+512];o=b["objects"][start:start+512]
                    z,_=model(r,o,torch.ones(o.shape[:2],device="cuda"))
                    tokens.append(z.cpu().numpy())
            dest=args.output/e["suite"]/("task_"+str(e["task_id"]))/(e["demo_key"]+".npz")
            dest.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest,tokens=np.concatenate(tokens),state_index=a["state_index"],
                                observation_index=a["observation_index"])
        result.append({"path":str(dest),"sha256":sha(dest),"suite":e["suite"],"task_id":e["task_id"],
                       "demo_key":e["demo_key"],"split":e["split"],"frames":e["frames"]})
        if len(result)%100==0:print("EXPORTED",len(result),flush=True)
    report={"status":"completed","schema":s["schema"],"checkpoint_sha256":checkpoint_hash,
            "feature_manifest_sha256":sha(args.data/"manifest.json"),"episodes":result}
    (args.output/"manifest.json").write_text(json.dumps(report,indent=2));print("TOKEN_EXPORT_COMPLETE",len(result),flush=True)
if __name__=="__main__":main()
