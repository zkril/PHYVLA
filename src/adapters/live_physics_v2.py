"""Read live physics with the raw-v2 contract, without modifying the simulator."""
import numpy as np
from types import SimpleNamespace
from datasets.physics_v2 import features
from adapters.libero_physics_adapter import LiberoPhysicsAdapter

class LivePhysicsV2:
    def __init__(self,env,task_description=None):
        import mujoco
        from robosuite.utils.binding_utils import MjSim
        self.env=env
        self.model=getattr(env.sim.model,"_model",env.sim.model)
        self.shadow=MjSim(self.model)
        self.data=self.shadow.data._data
        m=self.model
        names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,i) or "" for i in range(m.nbody)]
        self.ids=[i for i,n in enumerate(names) if i and n and not n.startswith(("robot","gripper","mount"))]
        self.adapter=LiberoPhysicsAdapter(SimpleNamespace(sim=self.shadow),
            object_patterns={names[i]:(names[i],) for i in self.ids},max_contacts=1)
        parsed=env.env.parsed_problem
        declarations={instance:typ for section in ("objects","fixtures")
                      for typ,instances in parsed[section].items() for instance in instances}
        def atoms(x):
            if isinstance(x,(list,tuple)):
                if x and isinstance(x[0],str) and x[0].lower() not in ("and","or","not"):
                    yield x
                else:
                    for y in x:yield from atoms(y)
        roles=np.zeros((len(self.ids),2),np.uint8);refs=[]
        for atom in atoms(parsed["goal_state"]):
            for slot,ref in enumerate(atom[1:]):
                if not isinstance(ref,str):continue
                matches=[v for v in declarations if ref==v or ref.startswith(v+"_")]
                if not matches:raise ValueError("Unresolved goal "+ref)
                instance=max(matches,key=len)
                selected=[j for j,i in enumerate(self.ids) if names[i]==instance or names[i].startswith(instance+"_")]
                if not selected and instance=="main_table" and declarations[instance]=="table":
                    selected=[j for j,i in enumerate(self.ids) if names[i]=="table"]
                if not selected:raise ValueError("Missing goal body "+ref)
                roles[selected,0 if slot==0 else 1]=1
                if ref not in refs:refs.append(ref)
        self.site_ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,ref) for ref in refs]
        self.site_ids=[i for i in self.site_ids if i>=0]
        self.static=dict(object_body_ids=np.array(self.ids),object_mass=m.body_mass[self.ids].copy(),
            object_inertia=m.body_inertia[self.ids].copy(),object_goal_roles=roles,
            geom_body_ids=m.geom_bodyid.copy(),geom_friction=m.geom_friction.copy(),
            joint_body_ids=m.jnt_bodyid.copy(),joint_qpos_address=m.jnt_qposadr.copy(),
            joint_dof_address=m.jnt_dofadr.copy(),joint_types=m.jnt_type.copy(),
            robot_body_ids=np.array([i for i,n in enumerate(names) if n.startswith(("robot","gripper"))]))

    def raw(self):
        import mujoco
        m=self.model;d=self.data
        source=getattr(self.env.sim.data,"_data",self.env.sim.data)
        # State restoration uses fresh control/solver history, matching raw-v2.
        d.qpos[:]=source.qpos;d.qvel[:]=source.qvel;d.time=source.time
        d.qacc_warmstart[:]=0
        mujoco.mj_forward(m,d)
        r={("robot_"+k):np.asarray(v,dtype=np.float32)[None] for k,v in self.adapter.extract_robot_state().items()}
        velocity=np.zeros((len(self.ids),6))
        for j,i in enumerate(self.ids):
            mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,i,velocity[j],0)
        r.update(object_position=d.xpos[self.ids].astype(np.float32)[None],
            object_quat_xyzw=d.xquat[self.ids][:,[1,2,3,0]].astype(np.float32)[None],
            object_angular_velocity=velocity[:,:3].astype(np.float32)[None],
            object_linear_velocity=velocity[:,3:].astype(np.float32)[None],
            goal_site_position=d.site_xpos[self.site_ids].astype(np.float32)[None],
            recorded_states=np.concatenate([[d.time],d.qpos,d.qvel])[None])
        n=int(d.ncon);cs=d.contact;contacts=np.zeros((n,21))
        if n:
            rotation=np.asarray(cs.frame).reshape(n,3,3);wrench=np.zeros((n,6))
            for i in range(n):
                if cs[i].efc_address>=0:mujoco.mj_contactForce(m,d,i,wrench[i])
            contacts[:,:3]=cs.pos;contacts[:,3:6]=rotation[:,0]
            contacts[:,6:9]=np.einsum("nji,nj->ni",rotation,wrench[:,:3])
            contacts[:,9:12]=np.einsum("nji,nj->ni",rotation,wrench[:,3:])
            contacts[:,12]=cs.dist;contacts[:,13]=wrench[:,0]
            contacts[:,14]=np.linalg.norm(wrench[:,1:3],axis=1)
            contacts[:,15]=np.linalg.norm(wrench[:,:3],axis=1)
            contacts[:,16]=np.linalg.norm(wrench[:,3:],axis=1)
            contacts[:,17]=cs.geom1;contacts[:,18]=cs.geom2
            contacts[:,19]=m.geom_bodyid[cs.geom1];contacts[:,20]=m.geom_bodyid[cs.geom2]
        r.update(self.static,contacts=contacts,contact_offsets=np.array([0,n]))
        return r

    def capture(self):
        return features(self.raw())
