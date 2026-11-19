"""Run all four original LIBERO suites under one fixed evaluation protocol."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument("--port",type=int,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--phyvla",action="store_true")
    p.add_argument("--trials",type=int,default=1);p.add_argument("--seed",type=int,default=7)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.rglob("result.json")):
        raise ValueError("Evaluation output already contains results; choose a fresh directory")
    env=dict(os.environ,LIBERO_HOME=str(ROOT/"thirdparty/LIBERO"),
        LIBERO_CONFIG_PATH=str(ROOT/"thirdparty/LIBERO/libero"),
        PYTHONPATH=":".join([str(ROOT/"src"),str(ROOT/"thirdparty/LIBERO"),str(ROOT/"thirdparty/starVLA")]),
        MUJOCO_GL="osmesa",PYOPENGL_PLATFORM="osmesa",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
    report={"status":"running","seed":args.seed,"trials_per_task":args.trials,"suite_reports":[]}
    for suite in ("libero_spatial","libero_object","libero_goal","libero_10"):
        cmd=[sys.executable,"examples/simBenchmarks/LIBERO/eval_files/eval_libero_phyvla.py",
            "--args.port",str(args.port),"--args.task-suite-name",suite,"--args.num-trials-per-task",str(args.trials),
            "--args.seed",str(args.seed),"--args.video-out-path",str(out/suite),"--args.job-name","physics_v2"]
        if args.phyvla:cmd.append("--args.use-phyvla")
        with (out/(suite+".log")).open("w") as f:
            subprocess.run(cmd,cwd=ROOT/"thirdparty/starVLA",env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        r=json.loads((out/suite/"result.json").read_text());assert r["status"]=="completed"
        report["suite_reports"].append({"suite":suite,"successes":r["successes"],"total":r["total"]})
        (out/"summary.json").write_text(json.dumps(report,indent=2))
    report.update(status="completed",successes=sum(r["successes"] for r in report["suite_reports"]),
                  total=sum(r["total"] for r in report["suite_reports"]))
    (out/"summary.json").write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=="__main__":main()
