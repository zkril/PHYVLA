"""Export original LIBERO states without replay drift, token training or future labels.

Each NPZ uses recorded state index t and expert action t. Contacts are ragged;
object instance names and all generalized joint coordinates are retained.
"""
from pathlib import Path
import argparse, concurrent.futures, hashlib, json, os, sys
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ['LIBERO_CONFIG_PATH'] = str(ROOT / 'thirdparty/LIBERO/libero')
os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('PYOPENGL_PLATFORM', os.environ['MUJOCO_GL'])
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'thirdparty/LIBERO')]
SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
SCHEMA = 'libero-raw-physics-v2'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def write_json(path, value):
    path = Path(path); tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, indent=2)); tmp.replace(path)

def worker(job):
    import h5py, mujoco
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    from libero_replay_common import restore_recorded_model
    from adapters.libero_physics_adapter import LiberoPhysicsAdapter
    suite_name, tid, output, indices, code_hash = job
    suite = benchmark.get_benchmark_dict()[suite_name]()
    assert suite.n_tasks == 10
    task = suite.get_task(tid)
    source = ROOT/'data'/suite.get_task_demonstration(tid)
    assert source.parent.name == suite_name and source.stem == task.name + '_demo'
    bddl = Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    source_hash = sha(source)
    outdir = Path(output)/suite_name/f'task_{tid}'; outdir.mkdir(parents=True, exist_ok=True)
    env = ControlEnv(bddl_file_name=str(bddl), use_camera_obs=False, has_offscreen_renderer=False, has_renderer=False)
    results = []
    try:
        env.seed(7)
        with h5py.File(source) as h:
            keys = sorted((k for k in h['data'] if k.startswith('demo_')), key=lambda k:int(k.split('_')[-1]))
            selected = keys if indices is None else [keys[i] for i in indices]
            for key in selected:
                dest = outdir/key; dest.mkdir(exist_ok=True)
                meta_path = dest/'metadata.json'; npz_path = dest/'physics.npz'
                if meta_path.exists() and npz_path.exists():
                    meta = json.loads(meta_path.read_text())
                    if meta['source_sha256'] != source_hash or meta['exporter_sha256'] != code_hash or meta['sha256'] != sha(npz_path):
                        raise ValueError('Stale cache: '+str(dest))
                    results.append(meta); continue
                g = h['data'][key]; states=g['states'][:]; actions=g['actions'][:]
                if len(states)!=len(actions): raise ValueError('State/action count mismatch')
                xml=g.attrs.get('model_file',h['data'].attrs.get('model_file'))
                if isinstance(xml,bytes): xml=xml.decode()
                if not xml: raise ValueError('Missing recorded XML')
                restore_recorded_model(env,xml)
                model=getattr(env.sim.model,'_model',env.sim.model); data=getattr(env.sim.data,'_data',env.sim.data)
                if states.shape[1]!=1+model.nq+model.nv: raise ValueError('Recorded state dimension mismatch')
                names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) or '' for i in range(model.nbody)]
                # Preserve complete instance names. Do not merge _1 and _2 or drop articulated descendants.
                ids=[i for i,n in enumerate(names) if i and n and not n.startswith(('robot','gripper','mount'))]
                patterns={names[i]:(names[i],) for i in ids}
                adapter=LiberoPhysicsAdapter(env,object_patterns=patterns,max_contacts=1)
                parsed=env.env.parsed_problem
                goals=parsed['goal_state']
                def atoms(x):
                    if isinstance(x,(list,tuple)):
                        if x and isinstance(x[0],str) and x[0].lower() not in ('and','or','not'): yield x
                        else:
                            for y in x: yield from atoms(y)
                declarations={}
                for section in ['objects','fixtures']:
                    for typ,instances in parsed[section].items():
                        for instance in instances: declarations[instance]=typ
                endpoints={}; role=np.zeros((len(ids),2),dtype=np.uint8)
                for atom in atoms(goals):
                    for slot,ref in enumerate(atom[1:]):
                        if not isinstance(ref,str):continue
                        matches=[v for v in declarations if ref==v or ref.startswith(v+'_')]
                        if not matches:raise ValueError('Unresolved goal entity '+ref)
                        instance=max(matches,key=len)
                        selected_ids=[j for j,i in enumerate(ids) if names[i]==instance or names[i].startswith(instance+'_')]
                        if not selected_ids and instance == 'main_table' and declarations[instance] == 'table':
                            selected_ids=[j for j,i in enumerate(ids) if names[i] == 'table']
                        if not selected_ids:raise ValueError('No body for goal entity '+ref)
                        for j in selected_ids:role[j,0 if slot==0 else 1]=1
                        endpoints[ref]={'instance':instance,'body_names':[names[ids[j]] for j in selected_ids]}
                if not role[:,0].any():raise ValueError('No manipulation target')
                goal_site_ids=[]; goal_site_names=[]
                for ref in endpoints:
                    site_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,ref)
                    if site_id>=0:
                        goal_site_ids.append(site_id);goal_site_names.append(ref)
                rows=[]; contacts=[]; offsets=[0]; maxerr=0.; success=[]
                for t,state in enumerate(states):
                    env.set_state(state); data.qacc_warmstart[:] = 0; env.sim.forward()
                    actual=np.asarray(env.get_sim_state()); maxerr=max(maxerr,float(np.max(np.abs(actual-state))))
                    if maxerr>1e-9:raise ValueError('State restore changed qpos/qvel/time')
                    robot=adapter.extract_robot_state()
                    velocity=[]
                    for i in ids:
                        v=np.empty(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,i,v,0);velocity.append(v)
                    row={'robot_'+k:v for k,v in robot.items()}
                    row.update(object_position=data.xpos[ids].copy(),object_quat_xyzw=data.xquat[ids][:,[1,2,3,0]].copy(),
                               object_angular_velocity=np.asarray(velocity)[:,:3],object_linear_velocity=np.asarray(velocity)[:,3:])
                    row['goal_site_position']=data.site_xpos[goal_site_ids].copy()
                    row['goal_site_rotation']=data.site_xmat[goal_site_ids].copy()
                    rows.append(row)
                    # Preserve all contacts. No bowl/plate flags or fixed top-K truncation.
                    n=int(data.ncon);frame=np.zeros((n,21),dtype=np.float64)
                    if n:
                        cs=data.contact;rotation=np.asarray(cs.frame).reshape(n,3,3)
                        wrench=np.zeros((n,6))
                        for c in range(n):
                            if cs[c].efc_address>=0:mujoco.mj_contactForce(model,data,c,wrench[c])
                        frame[:,:3]=cs.pos;frame[:,3:6]=rotation[:,0,:]
                        frame[:,6:9]=np.einsum('nji,nj->ni',rotation,wrench[:,:3])
                        frame[:,9:12]=np.einsum('nji,nj->ni',rotation,wrench[:,3:])
                        frame[:,12]=cs.dist;frame[:,13]=wrench[:,0]
                        frame[:,14]=np.linalg.norm(wrench[:,1:3],axis=1)
                        frame[:,15]=np.linalg.norm(wrench[:,:3],axis=1)
                        frame[:,16]=np.linalg.norm(wrench[:,3:],axis=1)
                        frame[:,17]=cs.geom1;frame[:,18]=cs.geom2
                        frame[:,19]=model.geom_bodyid[cs.geom1];frame[:,20]=model.geom_bodyid[cs.geom2]
                    contacts.extend(frame);offsets.append(len(contacts));success.append(bool(env.check_success()))
                arrays={k:np.stack([r[k] for r in rows]).astype(np.float32) for k in rows[0]}
                # Original create_dataset.py stores post-action observations, pre-action states.
                # For causal BC use observation[t-1] with state/action[t], only where numerically verified.
                image_index=np.full(len(states),-1,dtype=np.int32)
                alignment_error=np.full(len(states),-1.,dtype=np.float64)
                if 'joint_states' in g['obs'] and len(states)>1:
                    recorded_obs=g['obs/joint_states'][:]
                    alignment_error[1:]=np.max(np.abs(arrays['robot_qpos'][1:]-recorded_obs[:-1]),axis=1)
                    matched=np.where(alignment_error[1:]<1e-5)[0]+1
                    image_index[matched]=matched-1
                arrays.update(observation_index_for_state=image_index,observation_joint_alignment_error=alignment_error,
                              timesteps=np.arange(len(states),dtype=np.int32),recorded_states=states,expert_actions=actions,
                              contacts=np.asarray(contacts,dtype=np.float64).reshape(-1,21),contact_offsets=np.asarray(offsets,dtype=np.int64),
                              object_body_ids=np.asarray(ids,dtype=np.int32),object_mass=model.body_mass[ids].copy(),
                              object_inertia=model.body_inertia[ids].copy(),object_goal_roles=role,
                              geom_body_ids=model.geom_bodyid.copy(),geom_friction=model.geom_friction.copy(),
                              joint_body_ids=model.jnt_bodyid.copy(),joint_qpos_address=model.jnt_qposadr.copy(),
                              joint_dof_address=model.jnt_dofadr.copy(),joint_types=model.jnt_type.copy(),body_parent_ids=model.body_parentid.copy(),
                              geom_size=model.geom_size.copy(),geom_types=model.geom_type.copy())
                for k,v in arrays.items():
                    if not np.isfinite(v).all():raise ValueError('Nonfinite '+k)
                norms=np.linalg.norm(arrays['object_quat_xyzw'],axis=-1)
                if np.max(np.abs(norms-1))>1e-5:raise ValueError('Non-unit quaternion')
                tmp=dest/'physics.tmp.npz';np.savez_compressed(tmp,**arrays);tmp.replace(npz_path)
                (dest/'model_recorded.xml').write_text(xml)
                meta={'schema':SCHEMA,'suite':suite_name,'task_id':tid,'task_name':task.name,'language':task.language,
                      'demo_key':key,'split':'validation' if int(key.split('_')[-1])%10==9 else 'train',
                      'source':str(source),'source_sha256':source_hash,'exporter_sha256':code_hash,'frames':len(states),
                      'sha256':sha(npz_path),'path':str(npz_path),'object_names':[names[i] for i in ids],
                      'body_names':names,'goal_site_names':goal_site_names,'solver_warmstart':'zeroed before each restored-state forward','geom_names':[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) for i in range(model.ngeom)],
                      'joint_names':[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(model.njnt)],
                      'bddl_goals':goals,'goal_entities':endpoints,'object_instances':declarations,
                      'state_restore_max_abs_error':maxerr,'alignment':'recorded_states[t], expert_actions[t], physics frame t; no env.step',
                      'image_alignment':'Source obs is post-action; use verified observation_index_for_state (-1 means unavailable), never assume obs[t] matches states[t]',
                      'verified_image_frames':int(np.count_nonzero(image_index>=0)),
                      'contact_fields':['px','py','pz','nx','ny','nz','fx','fy','fz','tx','ty','tz','distance','normal_force','tangent_force','force_norm','torque_norm','geom1','geom2','body1','body2'],
                      'arrays':{k:list(v.shape) for k,v in arrays.items()},'mujoco_version':mujoco.__version__,
                      'labels_separate':'labels.json; never include terminal success in tokenizer input'}
                write_json(dest/'labels.json',{'success':success});write_json(meta_path,meta)
                results.append(meta)
                print(json.dumps({'suite':suite_name,'task':tid,'demo':key,'frames':len(states)}),flush=True)
    finally:env.close()
    return results

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--demo-indices',nargs='+',type=int)
    p.add_argument('--suites',nargs='+',choices=SUITES,default=SUITES);p.add_argument('--task-ids',nargs='+',type=int,default=list(range(10)))
    args=p.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    if any(t<0 or t>=10 for t in args.task_ids):raise ValueError('Task IDs must be 0..9')
    (args.output/'exporter_snapshot.py').write_text(Path(__file__).read_text())
    jobs=[(s,t,str(args.output),args.demo_indices,sha(__file__)) for s in args.suites for t in args.task_ids]
    report={'schema':SCHEMA,'status':'running','episodes':[],'errors':[],'jobs':len(jobs),'source_root':str(ROOT/'data')}
    write_json(args.output/'manifest.json',report)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(worker,j):j[:2] for j in jobs}
        for f in concurrent.futures.as_completed(pending):
            try:report['episodes'].extend(f.result())
            except Exception as exc:report['errors'].append({'task':pending[f],'error':repr(exc)});print('ERROR',pending[f],repr(exc),flush=True)
            write_json(args.output/'manifest.json',report)
    report['status']='failed' if report['errors'] else 'completed';write_json(args.output/'manifest.json',report)
    if report['errors']:raise SystemExit(1)
    print('EXPORT_COMPLETE',len(report['episodes']),flush=True)

if __name__=='__main__':main()
