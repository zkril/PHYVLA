"""Serve the paired physical tokenizer and visual-query residual policy."""
from pathlib import Path
import os,sys,argparse,logging
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"thirdparty/starVLA")]
def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--physics-checkpoint",type=Path,required=True);p.add_argument("--port",type=int,default=6695)
    p.add_argument("--mode",choices=("full","no_physics"),default="full")
    p.add_argument("--residual-scale",type=float,default=1.0)
    args=p.parse_args();args.checkpoint=args.checkpoint.resolve();args.physics_checkpoint=args.physics_checkpoint.resolve()
    os.chdir(ROOT/"thirdparty/starVLA")
    import torch,numpy as np,random
    torch.manual_seed(7);np.random.seed(7);random.seed(7);torch.set_num_threads(4)
    from p5.physics_policy_v2 import PhysicsPolicyV2
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
    policy=PhysicsPolicyV2(args.checkpoint,args.physics_checkpoint,args.mode,residual_scale=args.residual_scale)
    logging.info("PHYVLA v2 metadata: %s",policy.metadata)
    WebsocketPolicyServer(policy,host="127.0.0.1",port=args.port,metadata=policy.metadata,idle_timeout=-1).serve_forever()
if __name__=="__main__":
    logging.basicConfig(level=logging.INFO,force=True);main()
