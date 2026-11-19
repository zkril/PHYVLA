"""Train a shared tokenizer on all 40 tasks; hold out complete demos."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from models.physics_v2 import PhysicsTokenizerV2
from datasets.physics_v2 import sha,SCHEMA

def load_data(data,device):
    m=json.loads((data/"manifest.json").read_text())
    norm=json.loads((data/"normalization.json").read_text())
    assert m["status"]=="completed" and m["schema"]==SCHEMA
    assert sha(data/"normalization.json")==m["normalization_sha256"]
    out={}
    for split in ("train","validation"):
        buffers={k:[] for k in ("robot","objects","mask")}
        for e in m["episodes"]:
            if e["split"]!=split:continue
            if sha(e["path"])!=e["sha256"]:raise ValueError("Feature checksum mismatch")
            with np.load(e["path"]) as a:
                r=(a["robot"]-np.array(norm["robot"]["mean"],np.float32))/np.array(norm["robot"]["std"],np.float32)
                o=(a["objects"]-np.array(norm["objects"]["mean"],np.float32))/np.array(norm["objects"]["std"],np.float32)
                padded=np.zeros((len(r),m["max_objects"],o.shape[-1]),np.float32)
                padded[:,:o.shape[1]]=o
                mask=np.zeros(padded.shape[:2],np.float32);mask[:,:o.shape[1]]=1
                for k,v in zip(buffers,(r,padded,mask)):buffers[k].append(v)
        out[split]={k:torch.from_numpy(np.concatenate(v)).to(device) for k,v in buffers.items()}
    return out,norm,m

def batches(data,size,shuffle=False):
    n=len(data["robot"]);dev=data["robot"].device
    ids=torch.randperm(n,device=dev) if shuffle else torch.arange(n,device=dev)
    for chunk in ids.split(size):yield {k:v[chunk] for k,v in data.items()}

@torch.no_grad()
def evaluate(model,data,size):
    model.eval();totals={};count=0
    for b in batches(data,size):
        _,metrics=model.loss(**b);n=len(b["robot"]);count+=n
        for k,v in metrics.items():totals[k]=totals.get(k,0)+float(v)*n
    out={k:v/count for k,v in totals.items()};out["loss"]=sum(out.values())
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--epochs",type=int,default=12)
    p.add_argument("--batch-size",type=int,default=1024);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/"tokenizer_best.pt").exists():raise ValueError("Use a fresh training directory")
    torch.manual_seed(7);np.random.seed(7);torch.set_num_threads(4)
    data,norm,manifest=load_data(args.data,"cuda")
    model=PhysicsTokenizerV2(data["train"]["robot"].shape[-1],data["train"]["objects"].shape[-1]).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    history=[];best=float("inf");start=time.time()
    initial=evaluate(model,data["validation"],args.batch_size)
    print("INITIAL",json.dumps(initial),flush=True)
    for epoch in range(1,args.epochs+1):
        model.train();total=0;count=0
        for b in batches(data["train"],args.batch_size,True):
            optimizer.zero_grad(set_to_none=True);loss,_=model.loss(**b)
            if not torch.isfinite(loss):raise ValueError("Nonfinite loss")
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
            total+=float(loss)*len(b["robot"]);count+=len(b["robot"])
        val=evaluate(model,data["validation"],args.batch_size)
        record={"epoch":epoch,"train_loss":total/count,"validation":val,"seconds":time.time()-start}
        history.append(record);print(json.dumps(record),flush=True)
        (args.output/"history.json").write_text(json.dumps(history,indent=2))
        payload={"schema":SCHEMA,"config":model.config,"tokenizer_state_dict":model.state_dict(),
            "normalization":norm,"manifest_sha256":sha(args.data/"manifest.json"),
            "feature_code_sha256":manifest["feature_code_sha256"],"epoch":epoch,"validation":val,
            "seed":7,"initial_validation":initial}
        if val["loss"]<best:
            best=val["loss"];torch.save(payload,args.output/"tokenizer_best.pt")
        torch.save({**payload,"optimizer":optimizer.state_dict()},args.output/"training_last.pt")
    saved=torch.load(args.output/"tokenizer_best.pt",weights_only=False)
    model.load_state_dict(saved["tokenizer_state_dict"]);model.eval()
    with torch.no_grad():
        b=next(batches(data["validation"],8));z,v=model(**b)
        reload=PhysicsTokenizerV2(**saved["config"]).cuda().eval()
        reload.load_state_dict(saved["tokenizer_state_dict"])
        torch.testing.assert_close(z,reload(**b)[0],rtol=0,atol=0)
        perm=torch.randperm(b["objects"].shape[1],device="cuda")
        zp,_=model(b["robot"],b["objects"][:,perm],b["mask"][:,perm])
        torch.testing.assert_close(z[:,0],zp[:,0],atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(z[:,1:][:,perm],zp[:,1:],atol=2e-5,rtol=2e-5)
    summary={"completed":True,"initial_validation":initial,"best_validation":saved["validation"],
        "best_epoch":saved["epoch"],"train_frames":len(data["train"]["robot"]),
        "validation_frames":len(data["validation"]["robot"]),"train_episodes":1800,
        "validation_episodes":200,"tasks":40,"save_reload_exact":True,"permutation_equivariance":True,
        "tokenizer_sha256":sha(args.output/"tokenizer_best.pt"),
        "note":"Reconstruction metrics are not task success rates."}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2))
    print("TOKENIZER_COMPLETE",json.dumps(summary),flush=True)
if __name__=="__main__":main()
