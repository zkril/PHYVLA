"""Compare matched ordinary LIBERO rollouts and report sampling uncertainty."""
from pathlib import Path
import argparse,json,math
import numpy as np

SUITES=("libero_spatial","libero_object","libero_goal","libero_10")
def wilson(success,n):
    if not n:return None
    z=1.959963984540054;p=success/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-radius,center+radius]
def main():
    p=argparse.ArgumentParser();p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--phyvla",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();pairs=[];suites=[];paired_steps=[]
    for suite in SUITES:
        a=json.loads((args.baseline/suite/"result.json").read_text())
        b=json.loads((args.phyvla/suite/"result.json").read_text())
        assert a["status"]==b["status"]=="completed"
        for key in ("seed","num_trials_per_task","num_steps_wait","task_suite_name","max_tasks"):
            if a["args"][key]!=b["args"][key]:raise ValueError("Unmatched protocol: "+key)
        for key in ("baseline_sha256","use_bf16","action_head_dtype","vlm_autocast_dtype"):
            if key not in a["server"] or a["server"][key]!=b["server"].get(key):
                raise ValueError("Unmatched model precision/checkpoint: "+key)
        if a["libero_package"]!=b["libero_package"]:raise ValueError("Different benchmark packages")
        old={(e["task_id"],e["initial_state_index"]):e["success"] for e in a["episodes"]}
        new={(e["task_id"],e["initial_state_index"]):e["success"] for e in b["episodes"]}
        if old.keys()!=new.keys() or len(old)!=len(a["episodes"]):raise ValueError("Unmatched or repeated trials")
        samples=[(bool(old[k]),bool(new[k])) for k in sorted(old)]
        pairs.extend(samples)
        a_steps={(e["task_id"],e["initial_state_index"]):e["policy_steps"] for e in a["episodes"]}
        b_steps={(e["task_id"],e["initial_state_index"]):e["policy_steps"] for e in b["episodes"]}
        paired_steps.extend((a_steps[k],b_steps[k]) for k in old if old[k] and new[k])
        suites.append({"suite":suite,"episodes":len(samples),"baseline_successes":sum(x for x,y in samples),
                       "phyvla_successes":sum(y for x,y in samples)})
    base=sum(x for x,y in pairs);full=sum(y for x,y in pairs);n=len(pairs)
    wins=sum(y and not x for x,y in pairs);losses=sum(x and not y for x,y in pairs);discord=wins+losses
    p_value=min(1.,2*sum(math.comb(discord,k) for k in range(min(wins,losses)+1))/(2**discord)) if discord else 1.
    delta=np.array([int(y)-int(x) for x,y in pairs])
    rng=np.random.default_rng(7)
    bootstrap=delta[rng.integers(0,n,size=(10000,n))].mean(1)
    report={"complete":True,"episodes_per_model":n,"suite_results":suites,
        "baseline":{"successes":base,"rate":base/n,"wilson_95_interval":wilson(base,n)},
        "phyvla":{"successes":full,"rate":full/n,"wilson_95_interval":wilson(full,n)},
        "steps_on_joint_successes":{"count":len(paired_steps),
            "baseline_mean":float(np.mean([x for x,y in paired_steps])) if paired_steps else None,
            "phyvla_mean":float(np.mean([y for x,y in paired_steps])) if paired_steps else None,
            "mean_difference":float(np.mean([y-x for x,y in paired_steps])) if paired_steps else None},
        "paired_wins":wins,"paired_losses":losses,"success_rate_difference":(full-base)/n,
        "paired_bootstrap_95_interval":np.quantile(bootstrap,[.025,.975]).tolist(),
        "exact_mcnemar_two_sided_p":p_value,
        "interpretation":"Initial-state-limited exploratory comparison. More trials and independent seeds are required for a stable improvement claim.",
        "observation_difference":"PHYVLA additionally receives simulator physical state and BDDL-derived goal roles."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=="__main__":main()
