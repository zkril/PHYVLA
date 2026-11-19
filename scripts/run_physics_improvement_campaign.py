"""Finite experiment batch on allocated GPUs; no outcome-driven plus selection."""
import argparse,json,os,socket,subprocess,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/physics_improvement_20260928'
STAR=ROOT/'thirdparty/starVLA'
SIM='/0604-10T-test/Eamon/miniconda3/envs/libero-plus/bin/python'
POL='/0604-10T-test/Eamon/miniconda3/envs/starVLA/bin/python'
CK=STAR/'playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt'
SUITES=['libero_spatial','libero_object','libero_goal','libero_10']

def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)

def alive(pid):
    p=Path('/proc')/str(pid)/'stat'
    try:return p.read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False

def ready(port,pid,timeout=2400):
    start=time.time()
    while time.time()-start<timeout:
        if not alive(pid):raise RuntimeError(f'Server {pid} exited before readiness')
        try:
            with socket.create_connection(('127.0.0.1',port),timeout=2):return
        except OSError:time.sleep(5)
    raise TimeoutError(f'Server {port} readiness timeout')

def environment(gpu,plus):
    home=ROOT/'thirdparty'/('LIBERO-plus' if plus else 'LIBERO')
    return {**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'MUJOCO_GL':'egl','PYOPENGL_PLATFORM':'egl','MUJOCO_EGL_DEVICE_ID':str(gpu),'LIBERO_HOME':str(home),'LIBERO_CONFIG_PATH':str(home/'libero'),'PYTHONPATH':':'.join(map(str,[ROOT/'src',home,STAR])),'REQUIRE_V2_PRECISION':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1'}

def evaluate(name,gpu,port,physics,plus=False):
    dest=OUT/('plus' if plus else 'ordinary')/name
    plan=json.loads((OUT/'plus_protocol.json').read_text());summary=[]
    for suite in SUITES:
        output=dest/suite;output.mkdir(parents=True,exist_ok=True)
        if plus:
            ids=[str(e['task_id']) for e in plan['episodes'] if e['suite']==suite]
            cmd=[SIM,'examples/simBenchmarks/LIBERO-plus/eval_files/eval_libero.py','--args.port',str(port),'--args.policy-mode','phyvla' if physics else 'baseline','--args.pretrained-path',str(CK),'--args.task-suite-name',suite,'--args.num-trials-per-task','1','--args.seed','7','--args.video-out-path',str(output/'videos'),'--args.log-path',str(output),'--args.task-ids',*ids]
            result=output/(suite+'_result.json')
        else:
            cmd=[SIM,'examples/simBenchmarks/LIBERO/eval_files/eval_libero_phyvla.py','--args.port',str(port),'--args.task-suite-name',suite,'--args.num-trials-per-task','2','--args.initial-state-offset','1','--args.seed','7','--args.job-name','physics_v2','--args.video-out-path',str(output)]
            if physics:cmd.append('--args.use-phyvla')
            result=output/'result.json'
        if result.exists():raise RuntimeError('Use fresh campaign outputs: '+str(result))
        print('EVALUATE',name,suite,'plus' if plus else 'ordinary',flush=True)
        with (output/'evaluation.log').open('w') as log:
            subprocess.run(cmd,cwd=STAR,env=environment(gpu,plus),stdout=log,stderr=subprocess.STDOUT,check=True)
        r=json.loads(result.read_text());assert r['status']=='completed';assert len(r['episodes'])==(28 if plus else 20)
        summary.append({'suite':suite,'successes':sum(e['success'] for e in r['episodes']),'total':len(r['episodes']),'path':str(result)})
        write(dest/'summary.json',{'status':'running','suites':summary})
    report={'status':'completed','suites':summary,'successes':sum(x['successes'] for x in summary),'total':sum(x['total'] for x in summary)}
    write(dest/'summary.json',report);return report

def physics_eval(name,ckpt,scale,plus=False):
    log=(OUT/(name+'_server.log')).open('a');cmd=[POL,'scripts/serve_phyvla_v2.py','--checkpoint',str(CK),'--physics-checkpoint',str(ckpt),'--port','6703','--residual-scale',str(scale)]
    p=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'3','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT)
    write(OUT/'active_physics_server.json',{'pid':p.pid,'name':name,'gpu':3,'port':6703,'scale':scale,'plus':plus,'checkpoint':str(ckpt)})
    try:
        ready(6703,p.pid);return evaluate(name,3,6703,True,plus)
    finally:
        if p.poll() is None:
            p.terminate()
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        log.close()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',choices=['baseline','physics'],required=True);args=parser.parse_args();status=OUT/(args.worker+'_campaign_status.json')
    write(status,{'status':'running','started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    try:
        if args.worker=='baseline':
            pid=json.loads((OUT/'baseline_server_launch.json').read_text())['pid'];ready(6700,pid)
            evaluate('baseline',0,6700,False);evaluate('baseline',0,6700,False,True)
        else:
            pid=json.loads((OUT/'anchored_train_launch.json').read_text())['pid'];start=time.time()
            while not (OUT/'anchored/summary.json').exists():
                if not alive(pid):raise RuntimeError('Training exited without completed summary')
                if time.time()-start>43200:raise TimeoutError('Training wait exceeded 12 hours')
                time.sleep(10)
            train=json.loads((OUT/'anchored/summary.json').read_text());assert train['completed']
            old=ROOT/'outputs/physics_fusion_v2/policy_best.pt';candidates=[('old_full',old,1.),('old_quarter',old,.25),('anchored',OUT/'anchored/policy_best.pt',1.)];scores={}
            for name,ckpt,scale in candidates:scores[name]=physics_eval(name,ckpt,scale)
            # Ordinary development only. Prefer quarter residual on an exact tie.
            preference={'old_quarter':2,'anchored':1,'old_full':0};winner=max(scores,key=lambda k:(scores[k]['successes'],preference[k]))
            write(OUT/'candidate_selection.json',{'winner':winner,'ordinary_scores':scores,'criterion':'ordinary success count; fixed tie preference quarter > anchored > old full','plus_results_used':False})
            physics_eval('old_full',old,1.,True)
            if winner!='old_full':
                _,ckpt,scale=next(c for c in candidates if c[0]==winner);physics_eval(winner,ckpt,scale,True)
        write(status,{'status':'completed','finished_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    except Exception as exc:
        write(status,{'status':'failed','error':repr(exc),'at':time.strftime('%Y-%m-%dT%H:%M:%S%z')});raise
if __name__=='__main__':main()
