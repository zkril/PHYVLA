"""MJX compatibility audit for a LIBERO task XML.

Checks which features of the LIBERO MJCF are compatible with MJX,
reports incompatible elements, and optionally generates a reduced MJCF.

Run in phyvla-p2 environment (Python 3.11, mujoco-mjx installed):
    python scripts/audit_mjx_compatibility.py \
        --suite libero_spatial --task-id 2 --demo-index 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--xml",         type=Path, default=None,
                   help="Pre-extracted fixed XML path (preferred, no LIBERO needed)")
    p.add_argument("--suite",       default="libero_spatial")
    p.add_argument("--task-id",     type=int, default=2)
    p.add_argument("--demo-index",  type=int, default=0)
    p.add_argument("--output-xml",  type=Path,
                   default=Path("outputs/task02/model_mjx_v1.xml"))
    p.add_argument("--output-json", type=Path,
                   default=Path("outputs/task02/mjx_audit.json"))
    return p.parse_args()


def _check_actuators(model) -> dict:
    """Check if all actuators are MJX-compatible (motor/position/velocity)."""
    import mujoco
    SUPPORTED = {
        mujoco.mjtTrn.mjTRN_JOINT,
        mujoco.mjtTrn.mjTRN_JOINTINPARENT,
    }
    issues = []
    for i in range(model.nu):
        trntype = model.actuator_trntype[i]
        if trntype not in SUPPORTED:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or str(i)
            issues.append(f"actuator '{name}': trntype={trntype} unsupported")
    return {"total": model.nu, "issues": issues, "ok": len(issues) == 0}


def _check_contacts(model) -> dict:
    """Check contact parameters for MJX compatibility."""
    issues = []
    # MJX does not support contact pairs with condim != 3 or 6
    for i in range(model.npair):
        condim = int(model.pair_dim[i]) if hasattr(model, "pair_dim") else 3
        if condim not in (1, 3, 4, 6):
            issues.append(f"pair {i}: condim={condim}")
    return {"npair": model.npair, "ngeom": model.ngeom, "issues": issues,
            "ok": len(issues) == 0}


def _check_sensors(model) -> dict:
    """List sensors — needed later for MJX contact sensing."""
    import mujoco
    sensors = []
    for i in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) or str(i)
        sensors.append({"id": i, "name": name,
                        "type": int(model.sensor_type[i])})
    return {"total": model.nsensor, "sensors": sensors}


def _try_mjx_put(model) -> dict:
    """Attempt mjx.put_model — the definitive compatibility test."""
    try:
        from mujoco import mjx
        mx = mjx.put_model(model)
        return {"success": True, "error": None}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _extract_xml_from_demo(suite_name: str, task_id: int, demo_index: int) -> str:
    """Load model XML from the HDF5 demo file."""
    import h5py
    import re
    from libero.libero import benchmark, get_libero_path
    bd    = benchmark.get_benchmark_dict()
    suite = bd[suite_name]()
    task  = suite.get_task(task_id)
    rel   = suite.get_task_demonstration(task_id)
    demo_path = Path(get_libero_path("datasets")) / rel

    with h5py.File(str(demo_path), "r") as f:
        data   = f["data"]
        keys   = sorted(data.keys(),
                        key=lambda k: int(re.search(r"\d+$", k).group()))
        ep     = data[keys[demo_index]]
        raw    = ep.attrs.get("model_file")
        if isinstance(raw, bytes):
            return raw.decode("utf-8")
        return str(raw)


def _rewrite_xml_paths(xml_str: str) -> str:
    """Rewrite asset paths to current installation paths."""
    import xml.etree.ElementTree as ET
    import robosuite
    from libero.libero import get_libero_path

    root = ET.fromstring(xml_str)
    asset = root.find("asset")
    if asset is None:
        return xml_str

    robosuite_root = Path(robosuite.__file__).resolve().parent
    try:
        libero_assets = Path(get_libero_path("assets")).resolve()
    except Exception:
        libero_assets = None

    rewritten = 0
    for el in list(asset.findall("mesh")) + list(asset.findall("texture")) + \
              list(asset.findall("hfield")):
        old = el.get("file")
        if not old:
            continue
        parts = old.replace("\\", "/").split("/")

        new_path = None
        for idx, p in enumerate(parts):
            if p == "robosuite":
                cand = robosuite_root.joinpath(*parts[idx + 1:])
                if cand.exists():
                    new_path = cand
                    break
        if new_path is None and libero_assets and "assets" in parts:
            ai = max(i for i, p in enumerate(parts) if p == "assets")
            cand = libero_assets.joinpath(*parts[ai + 1:])
            if cand.exists():
                new_path = cand
        if new_path:
            el.set("file", str(new_path.resolve()))
            rewritten += 1

    print(f"[audit] rewritten asset paths: {rewritten}")
    return ET.tostring(root, encoding="unicode")


def main() -> None:
    args = parse_args()

    # 优先使用已提供的 XML，否则从 LIBERO HDF5 提取
    if args.xml and args.xml.exists():
        print(f"使用已有 XML: {args.xml}")
        xml_fixed = args.xml.read_text()
        args.output_xml.parent.mkdir(parents=True, exist_ok=True)
        if args.output_xml != args.xml:
            args.output_xml.write_text(xml_fixed)
    else:
        print("从演示文件提取 XML...")
        xml_str   = _extract_xml_from_demo(args.suite, args.task_id, args.demo_index)
        xml_fixed = _rewrite_xml_paths(xml_str)
        args.output_xml.parent.mkdir(parents=True, exist_ok=True)
        args.output_xml.write_text(xml_fixed)
        print(f"已保存 XML: {args.output_xml}")

    import mujoco
    model = mujoco.MjModel.from_xml_string(xml_fixed)
    data  = mujoco.MjData(model)

    print(f"\nModel stats:")
    print(f"  nbody={model.nbody}  ngeom={model.ngeom}  njnt={model.njnt}")
    print(f"  nu={model.nu}  nq={model.nq}  nv={model.nv}")
    print(f"  nmocap={model.nmocap}  nsensor={model.nsensor}")

    report = {
        "model_stats": {
            "nbody": model.nbody, "ngeom": model.ngeom,
            "njnt": model.njnt, "nu": model.nu,
            "nq": model.nq, "nv": model.nv,
            "nmocap": model.nmocap, "nsensor": model.nsensor,
        },
        "actuator_check": _check_actuators(model),
        "contact_check":  _check_contacts(model),
        "sensor_check":   _check_sensors(model),
        "mjx_put_model":  _try_mjx_put(model),
    }

    print(f"\n{'='*50}")
    print("MJX COMPATIBILITY AUDIT")
    print(f"{'='*50}")
    for section, result in report.items():
        if section == "model_stats":
            continue
        ok = result.get("ok", result.get("success", "?"))
        status = "✅" if ok else "❌"
        print(f"{status} {section}")
        issues = result.get("issues", []) or result.get("error") or []
        if isinstance(issues, str):
            issues = [issues]
        for issue in issues[:5]:
            print(f"     → {issue}")

    mjx_ok = report["mjx_put_model"]["success"]
    print(f"\nMJX VERDICT: {'✅ COMPATIBLE' if mjx_ok else '❌ NEEDS REDUCED MJCF'}")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {args.output_json}")


if __name__ == "__main__":
    main()
