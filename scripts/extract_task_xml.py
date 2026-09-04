"""从任意 LIBERO 任务 HDF5 中提取并修复模型 XML。

运行环境：libero_replay（Python 3.8，有 LIBERO + robosuite）

用法：
    python scripts/extract_task_xml.py \
        --suite libero_spatial --task-id 3 --demo-index 0 \
        --output outputs/task03/model_raw.xml
"""

from __future__ import annotations

import argparse
import h5py
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT / "scripts", PROJECT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--suite",       default="libero_spatial")
    p.add_argument("--task-id",     type=int, required=True)
    p.add_argument("--demo-index",  type=int, default=0)
    p.add_argument("--output",      type=Path, required=True)
    return p.parse_args()


def _fix_asset_paths(xml_str: str) -> str:
    import robosuite
    try:
        from libero.libero import get_libero_path
        libero_assets = Path(get_libero_path("assets")).resolve()
    except Exception:
        libero_assets = None

    robosuite_root = Path(robosuite.__file__).resolve().parent
    root  = ET.fromstring(xml_str)
    asset = root.find("asset")
    if asset is None:
        return xml_str

    rewritten = 0
    for el in (list(asset.findall("mesh")) + list(asset.findall("texture"))
               + list(asset.findall("hfield"))):
        old = el.get("file", "")
        if not old:
            continue
        parts = old.replace("\\", "/").split("/")
        new_path = None
        for i, p in enumerate(parts):
            if p == "robosuite":
                cand = robosuite_root.joinpath(*parts[i + 1:])
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

    print(f"[extract_xml] 路径修复: {rewritten} 个 asset")
    return ET.tostring(root, encoding="unicode")


def main() -> None:
    args = parse_args()

    from libero.libero import benchmark, get_libero_path
    bd    = benchmark.get_benchmark_dict()
    suite = bd[args.suite]()
    task  = suite.get_task(args.task_id)
    rel   = suite.get_task_demonstration(args.task_id)
    demo_path = Path(get_libero_path("datasets")) / rel

    print(f"[extract_xml] task {args.task_id}: {task.language}")
    print(f"[extract_xml] demo: {demo_path}")

    with h5py.File(str(demo_path), "r") as f:
        keys = sorted(
            f["data"].keys(),
            key=lambda k: int(re.search(r"\d+$", k).group()),
        )
        raw = f["data"][keys[args.demo_index]].attrs.get("model_file")
        xml_str = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)

    xml_fixed = _fix_asset_paths(xml_str)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(xml_fixed)
    print(f"[extract_xml] 已保存: {args.output}  ({len(xml_fixed)} chars)")


if __name__ == "__main__":
    main()
