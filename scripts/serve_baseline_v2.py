"""Serve the precision-matched, unmodified visual baseline for v2 comparisons."""
from pathlib import Path
import sys,os,argparse,logging,random
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"thirdparty/starVLA")]
from datasets.physics_v2 import sha
def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--port",type=int,default=6696);args=p.parse_args()
    checkpoint=args.checkpoint.resolve();os.chdir(ROOT/"thirdparty/starVLA")
    torch.manual_seed(7);np.random.seed(7);random.seed(7);torch.set_num_threads(4)
    from deployment.model_server.policy_wrapper import PolicyServerWrapper
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
    policy=PolicyServerWrapper(str(checkpoint),use_bf16=False)
    meta={**policy.metadata,"use_bf16":False,"seed":7,"vlm_autocast_dtype":"bfloat16",
          "action_head_dtype":str(next(policy._framework.action_model.parameters()).dtype),
          "baseline_sha256":sha(checkpoint),"comparison_kind":"physics_v2_matched_baseline"}
    WebsocketPolicyServer(policy,host="127.0.0.1",port=args.port,metadata=meta,idle_timeout=-1).serve_forever()
if __name__=="__main__":
    logging.basicConfig(level=logging.INFO,force=True);main()
