"""Validate source identity, all arrays and exact raw state/action alignment."""
from pathlib import Path
import argparse, hashlib, json
import h5py,numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((a.root/'manifest.json').read_text());errors=[];frames=0;contacts=0;aligned_frames=0;splits={};seen=set();tasks=set();source_cache={}
    for m in manifest['episodes']:
        key=(m['suite'],m['task_id'],m['demo_key'])
        if key in seen:errors.append([key,'duplicate episode'])
        seen.add(key);tasks.add(key[:2]);f=Path(m['path'])
        try:
            assert m['schema']=='libero-raw-physics-v2'
            assert Path(m['source']).parent.name==m['suite']
            assert Path(m['source']).stem==m['task_name']+'_demo'
            assert hashlib.sha256(f.read_bytes()).hexdigest()==m['sha256']
            with np.load(f,allow_pickle=False) as z,h5py.File(m['source'],'r') as h:
                g=h['data'][m['demo_key']];t=m['frames'];ids=z['object_body_ids'];off=z['contact_offsets']
                assert np.array_equal(z['recorded_states'],g['states'][:])
                assert np.array_equal(z['expert_actions'],g['actions'][:])
                assert np.array_equal(z['timesteps'],np.arange(t))
                index=z['observation_index_for_state'];mask=index>=0
                assert index[0]==-1 and np.all(index[mask]<z['timesteps'][mask])
                assert np.all(index[mask]==z['timesteps'][mask]-1)
                if np.any(mask):
                    assert np.max(np.abs(z['robot_qpos'][mask]-g['obs/joint_states'][:][index[mask]]))<1e-5
                aligned_frames+=int(mask.sum())
                assert len(off)==t+1 and off[0]==0 and off[-1]==len(z['contacts']) and np.all(np.diff(off)>=0)
                assert len(m['object_names'])==len(set(m['object_names']))==len(ids)
                cs=z['contacts'];geom=z['geom_body_ids']
                if len(cs):
                    pair=cs[:,17:19].astype(np.int64)
                    assert np.all(pair>=0) and np.all(pair<len(geom))
                    assert np.array_equal(geom[pair],cs[:,19:21])
                    assert np.max(np.abs(np.linalg.norm(cs[:,3:6],axis=1)-1))<1e-5
                assert np.all(z['object_mass']>=0) and np.all(z['object_inertia']>=0)
                assert z['object_goal_roles'][:,0].any()
                assert not any(k in z.files for k in ['success','phases','future','tokens'])
                assert np.max(np.abs(np.linalg.norm(z['object_quat_xyzw'],axis=-1)-1))<1e-5
                for k in z.files:assert np.isfinite(z[k]).all(),k
                frames+=t;contacts+=len(z['contacts'])
                splits[m['split']]=splits.get(m['split'],0)+1
                for k in ['agentview_rgb','eye_in_hand_rgb']:
                    if k in g['obs']:assert len(g['obs'][k])==t
        except Exception as exc:errors.append([key,repr(exc)])
    if manifest['status']!='completed' or manifest['errors']:errors.append(['manifest','export not complete'])
    result={'episodes':len(seen),'tasks':len(tasks),'frames':frames,'contacts':contacts,'verified_image_frames':aligned_frames,'splits':splits,'errors':errors,'passed':not errors}
    (a.root/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    if errors:raise SystemExit(1)
if __name__=='__main__':main()
