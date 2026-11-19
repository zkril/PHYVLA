"""Check raw-v2 live extraction against held-out recorded states on all tasks."""
from pathlib import Path
import sys,os,json,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"thirdparty/LIBERO"),str(ROOT/"scripts")]
os.environ["LIBERO_CONFIG_PATH"]=str(ROOT/"thirdparty/LIBERO/libero")
os.environ["LIBERO_HOME"]=str(ROOT/"thirdparty/LIBERO")
os.environ["MUJOCO_GL"]="osmesa"
os.environ["PYOPENGL_PLATFORM"]="osmesa"
from libero.libero import benchmark,get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv
from libero_replay_common import restore_recorded_model
from datasets.physics_v2 import features,read_episode
from adapters.live_physics_v2 import LivePhysicsV2

def main():
    manifest=json.loads((ROOT/"outputs/physics_reexport_v2/manifest.json").read_text())
    results=[]
    for e in manifest["episodes"]:
        if e["demo_key"]!="demo_9":continue
        suite=benchmark.get_benchmark_dict()[e["suite"]]()
        task=suite.get_task(e["task_id"])
        env=ControlEnv(bddl_file_name=str(Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file),
                       use_camera_obs=False,has_offscreen_renderer=False,has_renderer=False)
        try:
            restore_recorded_model(env,Path(e["path"]).with_name("model_recorded.xml").read_text())
            live=LivePhysicsV2(env);a=read_episode(e);offline=features(a)
            errors={}
            for index in (1,len(a["recorded_states"])//2,len(a["recorded_states"])-1):
                env.set_state(a["recorded_states"][index]);env.sim.forward()
                before=env.get_sim_state().copy();current=live.capture()
                np.testing.assert_array_equal(before,env.get_sim_state())
                for k in offline:
                    err=float(np.max(np.abs(offline[k][index:index+1]-current[k])))
                    errors[k]=max(errors.get(k,0),err)
                    np.testing.assert_allclose(offline[k][index:index+1],current[k],rtol=2e-5,atol=2e-5)
            results.append({"suite":e["suite"],"task_id":e["task_id"],"max_abs_errors":errors})
            print(json.dumps(results[-1]),flush=True)
        finally:env.close()
    assert len(results)==40
    out=ROOT/"outputs/physics_tokenizer_v2/live_parity.json"
    out.write_text(json.dumps({"passed":True,"tasks":40,"frames":120,"simulator_unmodified":True,
                              "results":results},indent=2))
    print("LIVE_PARITY_PASSED",flush=True)
if __name__=="__main__":main()
