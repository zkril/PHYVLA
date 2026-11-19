"""CPU diagnostic of held-out residuals; never selects on plus test outcomes."""
from pathlib import Path
import sys,json,collections,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'thirdparty/starVLA')]
from models.physics_v2 import PhysicsTokenizerV2,PhysicsResidualFusion
from starVLA.model.modules.action_model.MLP_ActionHeader import L1RegressionActionHead

def main():
    torch.set_num_threads(4)
    saved=torch.load(ROOT/'outputs/physics_fusion_v2/policy_best.pt',map_location='cpu',weights_only=False)
    tok=PhysicsTokenizerV2(**saved['tokenizer_config']).eval();tok.load_state_dict(saved['tokenizer_state_dict'])
    fusion=PhysicsResidualFusion(**saved['fusion_config']).eval();fusion.load_state_dict(saved['fusion_state_dict'])
    h=saved['fusion_config']['hidden_dim'];head=L1RegressionActionHead(input_dim=h,hidden_dim=2*h,action_dim=7,NUM_ACTIONS_CHUNK=saved['horizon']).eval();head.load_state_dict(saved['action_head_state_dict'])
    manifest=json.loads((ROOT/'outputs/visual_queries_v2/manifest.json').read_text());feature=json.loads((ROOT/'outputs/physics_features_v2/manifest.json').read_text())
    sets={s:{(e['suite'],e['task_id'],e['demo_key']) for e in manifest['episodes'] if e['split']==s} for s in ['train','validation']}
    assert not sets['train']&sets['validation']
    scales=[0.,.1,.25,.5,1.];stats={str(s):collections.defaultdict(float) for s in scales};rows=[];start=time.time()
    with torch.inference_mode():
      for e in manifest['episodes']:
        if e['split']!='validation':continue
        with np.load(e['path']) as a,np.load(e['feature_path']) as f:
          ids=a['feature_index'];assert np.array_equal(a['state_index'],f['state_index'][ids]);assert np.array_equal(a['observation_index'],f['observation_index'][ids])
          values={k:torch.from_numpy((f[k][ids]-np.array(saved['normalization'][k]['mean'],np.float32))/np.array(saved['normalization'][k]['std'],np.float32)) for k in ['robot','objects']}
          q=torch.from_numpy(a['queries']);target=torch.from_numpy(a['targets']);mask=torch.from_numpy(a['action_mask']);base=torch.from_numpy(a['baseline_actions'])
          for j in range(0,len(q),16):
            ix=slice(j,j+16);r,o=values['robot'][ix],values['objects'][ix];z,valid=tok(r,o,torch.ones(o.shape[:2]));truth=target[ix];m=mask[ix];b=head.predict_action(q[ix]);cacheerr=float((b-base[ix]).abs().max())
            for scale in scales:
              pred=head.predict_action(fusion(q[ix],z,valid,scale));s=stats[str(scale)];s['error_sum']+=float(((pred-truth).abs()*m).sum());s['cells']+=float(m.sum());s['delta_sum']+=float(((pred-b).abs()*m).sum());s['gripper_flips']+=float((((pred[...,6]>.5)!=(b[...,6]>.5))*m[...,6]).sum());s['gripper_cells']+=float(m[...,6].sum());s['worse_chunks']+=float(((((pred-truth).abs()*m).sum((1,2)))>((b-truth).abs()*m).sum((1,2))+1e-7).sum());s['chunks']+=len(b)
            rows.append({'suite':e['suite'],'task_id':e['task_id'],'robot_abs_max':float(r.abs().max()),'objects_abs_max':float(o.abs().max()),'cache_replay_max_error':cacheerr})
    report={'scope':'All 200 held-out ordinary LIBERO demos; no plus outcomes used','gripper_threshold':0.5,'gripper_normalization_mask':False,'splits':{k:len(v) for k,v in sets.items()},'seconds':time.time()-start,'scales':{k:{'l1':v['error_sum']/v['cells'],'mean_abs_action_change':v['delta_sum']/v['cells'],'gripper_flip_fraction':v['gripper_flips']/v['gripper_cells'],'fraction_chunks_worse_than_baseline':v['worse_chunks']/v['chunks'],'chunks':v['chunks']} for k,v in stats.items()},'sample_diagnostics':rows}
    out=ROOT/'outputs/physics_improvement_20260928/residual_audit.json';out.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='sample_diagnostics'},indent=2),flush=True)
if __name__=='__main__':main()
