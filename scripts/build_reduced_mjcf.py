"""构建 MJX 兼容 MJCF（v1 + nconmax）。

处理步骤：
  1. MESH → CAPSULE + 移除 asset mesh 定义（消除~18GB GPU 显存占用）
  2. CYLINDER → CAPSULE（MJX 不支持 CYLINDER↔MESH 碰撞）
  3. 非关键 geom 禁用碰撞（降低约束矩阵规模）
  4. <size nconmax="N"/> 硬限制接触点数（官方推荐，进一步限制 efc_J）

运行（phyvla-p2 环境）：
    python scripts/build_reduced_mjcf.py \
        --xml    outputs/task02/model_raw.xml \
        --output outputs/task02/model_mjx_reduced_v1.xml \
        --nconmax 32
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

KEEP_COLLISION_KEYWORDS = (
    "finger_joint1_tip",
    "finger_joint2_tip",
    "bowl",
    "plate",
    "table_top",
    "table_collision",
)

_MESH_FALLBACK_SIZE = "0.05 0.05"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--xml",     type=Path,
                   default=Path("outputs/task02/model_raw.xml"))
    p.add_argument("--output",  type=Path,
                   default=Path("outputs/task02/model_mjx_reduced_v1.xml"))
    p.add_argument("--nconmax", type=int, default=32)
    return p.parse_args()


def _should_keep_collision(name: str) -> bool:
    low = name.lower()
    return any(k in low for k in KEEP_COLLISION_KEYWORDS)


def set_compiler_discardvisual(root: ET.Element) -> bool:
    """在 <compiler> 元素中设置 discardvisual="true"。"""
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("discardvisual", "true")
    return True


def patch_xml(root: ET.Element, nconmax: int) -> dict:
    stats = {"mesh_replaced": 0, "cylinder_replaced": 0,
             "collision_disabled": 0, "kept": 0, "discardvisual": False}

    # 1. compiler discardvisual="true"（官方方式：编译时丢弃视觉 geom）
    stats["discardvisual"] = set_compiler_discardvisual(root)

    # 2. 移除 asset 中的 mesh 定义（进一步确保无 mesh 数据）
    asset = root.find("asset")
    if asset is not None:
        for el in list(asset.findall("mesh")):
            asset.remove(el)

    for geom in root.iter("geom"):
        name  = geom.get("name", "")
        gtype = geom.get("type", "sphere").lower()

        # 2. MESH → CAPSULE
        if gtype == "mesh":
            geom.set("type", "capsule")
            geom.attrib.pop("mesh", None)
            if not geom.get("size"):
                geom.set("size", _MESH_FALLBACK_SIZE)
            stats["mesh_replaced"] += 1

        # 3. CYLINDER → CAPSULE
        elif gtype == "cylinder":
            geom.set("type", "capsule")
            stats["cylinder_replaced"] += 1

        # 4. 非关键 geom 禁用碰撞
        if _should_keep_collision(name):
            stats["kept"] += 1
        else:
            geom.set("contype",     "0")
            geom.set("conaffinity", "0")
            stats["collision_disabled"] += 1

    # 5. 添加 nconmax（官方推荐，限制约束矩阵大小）
    size_el = root.find("size")
    if size_el is None:
        size_el = ET.SubElement(root, "size")
    size_el.set("nconmax", str(nconmax))
    if "njmax" not in size_el.attrib:
        size_el.set("njmax", str(nconmax * 6))

    return stats


def estimate_constraints(root: ET.Element) -> int:
    active = []
    for geom in root.iter("geom"):
        ct = int(geom.get("contype",    "1"))
        ca = int(geom.get("conaffinity","1"))
        if ct != 0 or ca != 0:
            active.append(geom.get("name", ""))
    n = len(active)
    return n * (n + 1) // 2


def main() -> None:
    args = parse_args()
    if not args.xml.exists():
        raise FileNotFoundError(f"XML 不存在: {args.xml}")

    print(f"输入: {args.xml}")
    tree = ET.parse(str(args.xml))
    root = tree.getroot()

    stats = patch_xml(root, args.nconmax)

    print(f"\n修改结果：")
    print(f"  MESH → CAPSULE     : {stats['mesh_replaced']} 个  ← 消除 ~18GB 显存")
    print(f"  CYLINDER → CAPSULE : {stats['cylinder_replaced']} 个")
    print(f"  禁用碰撞           : {stats['collision_disabled']} 个 geom")
    print(f"  保留碰撞           : {stats['kept']} 个 geom")
    print(f"  nconmax            : {args.nconmax}（官方推荐）")

    n_pairs = estimate_constraints(root)
    nv = 43
    mem_mb = min(n_pairs, args.nconmax) * 3 * nv * 4 / 1024 / 1024
    print(f"\n  估算 efc_J 显存    : ~{mem_mb*1024:.1f} KB（原 ~17 GB）")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(args.output), encoding="unicode", xml_declaration=False)
    print(f"\n已保存: {args.output}")

    print("\nMJX 兼容性验证...")
    try:
        import mujoco
        from mujoco import mjx
        model = mujoco.MjModel.from_xml_path(str(args.output))
        mx    = mjx.put_model(model)
        data  = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        dx = mjx.put_data(model, data)
        print(f"✅ mjx.put_model + put_data 成功")
        print(f"   nbody={model.nbody}  ngeom={model.ngeom}  nu={model.nu}  nconmax={args.nconmax}")
    except Exception as e:
        print(f"❌ 错误: {str(e)[:200]}")


if __name__ == "__main__":
    main()
