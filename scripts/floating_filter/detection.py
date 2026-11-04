#!/usr/bin/env python3
"""
检测 3D 场景中漂浮的物体并生成报告
"""
import sys
import json

try:
    import bpy
    import mathutils
except ImportError:
    # 允许在非 Blender 环境中作为主程序被调用时，自动通过 Blender 命令行执行自己
    pass


def _require_mathutils():
    import mathutils
    return mathutils


def _world_bbox_bounds(obj) -> tuple[list[float], list[float]]:
    corners = [obj.matrix_world @ _require_mathutils().Vector(corner) for corner in obj.bound_box]
    mins = [min(float(corner[i]) for corner in corners) for i in range(3)]
    maxs = [max(float(corner[i]) for corner in corners) for i in range(3)]
    return mins, maxs


def _child_support_sample_points(obj) -> list[tuple[float, float]]:
    mins, maxs = _world_bbox_bounds(obj)
    center_x = (mins[0] + maxs[0]) / 2.0
    center_y = (mins[1] + maxs[1]) / 2.0
    inset = 0.2
    x_low = mins[0] * (1.0 - inset) + center_x * inset
    x_high = maxs[0] * (1.0 - inset) + center_x * inset
    y_low = mins[1] * (1.0 - inset) + center_y * inset
    y_high = maxs[1] * (1.0 - inset) + center_y * inset
    return [
        (center_x, center_y),
        (x_low, y_low),
        (x_low, y_high),
        (x_high, y_low),
        (x_high, y_high),
    ]


def _inverted_matrix_or_none(matrix):
    try:
        return matrix.inverted()
    except ValueError:
        return None


def raycast_downwards(origin_world, ignore_obj=None):
    """
    自 origin_world 向下做射线检测，寻找下方碰撞的第一个网格体
    返回: (hit_obj, hit_z)
    """
    mathutils = _require_mathutils()
    direction_world = mathutils.Vector((0.0, 0.0, -1.0))
    
    best_hit_obj = None
    best_hit_z = -9999.0
    
    # 遍历场景中所有的网格体
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH' or obj == ignore_obj:
            continue
            
        inv_world = _inverted_matrix_or_none(obj.matrix_world)
        if inv_world is None:
            continue
            
        origin_local = inv_world @ origin_world
        direction_local = (inv_world.to_3x3() @ direction_world).normalized()
        
        hit, location, normal, _face_index = obj.ray_cast(
            origin_local,
            direction_local,
            distance=1000.0
        )
        
        if hit:
            hit_world = obj.matrix_world @ location
            # 我们只需要朝下检测，所以 z 坐标必须低于 origin_world.z
            if hit_world.z < origin_world.z - 1e-4:
                # 寻找最高（最近）的支撑面
                if hit_world.z > best_hit_z:
                    best_hit_z = hit_world.z
                    best_hit_obj = obj
                    
    if best_hit_obj is not None:
        return best_hit_obj, best_hit_z
    return None, None


def classify_floating_state(
    bottom_z: float,
    support_z: float,
    ground_z: float,
    tolerance: float = 0.015,
) -> dict | None:
    """Return floating issue metadata for one object, or None when supported."""
    below_floor_gap = round(float(ground_z) - float(bottom_z), 4)
    if below_floor_gap > tolerance:
        return {
            "kind": "below_floor",
            "gap": below_floor_gap,
            "severity": "BELOW_FLOOR (低于地板)",
        }

    support_gap = round(float(bottom_z) - float(support_z), 4)
    if support_gap <= tolerance:
        return None

    if support_gap < 0.04:
        severity = "SLIGHT (轻微漂浮)"
    elif support_gap < 0.15:
        severity = "MODERATE (明显悬空)"
    else:
        severity = "SEVERE (严重漂浮)"

    return {
        "kind": "floating",
        "gap": support_gap,
        "severity": severity,
    }


def should_remove_floating_issue(issue: dict) -> bool:
    """Return whether a detected issue is severe enough to remove from data."""
    kind = issue.get("kind")
    if kind in {"below_floor", "no_support"}:
        return True
    severity = str(issue.get("severity", ""))
    return not severity.startswith("SLIGHT")


def partition_floating_issues(issues: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split detected issues into removable issues and ignored slight issues."""
    removable = []
    ignored = []
    for issue in issues:
        if should_remove_floating_issue(issue):
            removable.append(issue)
        else:
            ignored.append(issue)
    return removable, ignored


def detect_floating_objects_in_current_scene(
    layout_data: dict,
    tolerance: float = 0.015,
) -> list[dict]:
    """Detect floating or below-floor asset objects in the current Blender scene."""
    json_objects = {obj["id"]: obj for obj in layout_data.get("objects", [])}
    
    # 找出所有带 object_id 的资产物体
    asset_objects = []
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and "object_id" in obj:
            asset_objects.append(obj)
            
    # 获取地板的 Z 轴高度作为 ground_z 基准
    floor_obj = bpy.data.objects.get("Floor")
    ground_z = 0.0
    if floor_obj is not None:
        _, floor_maxs = _world_bbox_bounds(floor_obj)
        ground_z = floor_maxs[2]
    
    floating_results = []
    
    for obj in asset_objects:
        obj_id = int(obj["object_id"])
        short_desc = obj.get("short_desc", "Unknown")
        jid = obj.get("jid", "Unknown")
        
        # 获取 JSON 中的配置
        json_obj = json_objects.get(obj_id, {})
        category = json_obj.get("category", "")
        
        # 排除墙面挂饰、顶灯等本来就不应该落地或有下方支撑的物体
        is_attachment = False
        expected_relations = json_obj.get("expected_relations", {})
        for rel in expected_relations.get("relative_relations", []):
            if rel.get("type") in ["inside", "in", "within"]:
                # 在容器内的物体由 snap_contained 阶段处理，也可以检测
                pass
                
        # 如果品类为 挂画, 窗帘, 壁灯, 吊灯, 镜子, 门 等，不需要检测其离地/支撑距离
        skip_categories = {"picture", "wall_art", "curtain", "painting", "wall_lamp", "ceiling_lamp", "chandelier", "mirror", "door", "window"}
        if category in skip_categories or any(kw in short_desc.lower() for kw in ["picture", "painting", "curtain", "wall lamp", "ceiling lamp", "chandelier"]):
            continue
            
        mins, maxs = _world_bbox_bounds(obj)
        bottom_z = mins[2]

        floor_issue = classify_floating_state(
            bottom_z=bottom_z,
            support_z=ground_z,
            ground_z=ground_z,
            tolerance=tolerance,
        )
        if floor_issue is not None and floor_issue["kind"] == "below_floor":
            floating_results.append({
                "id": obj_id,
                "name": short_desc,
                "category": category,
                "bottom_z": bottom_z,
                "support_type": "FLOOR",
                "support_name": "地板以下 (Below Floor)",
                "gap": floor_issue["gap"],
                "severity": floor_issue["severity"],
                "kind": floor_issue["kind"],
            })
            continue
        
        # 修正射线起点 Z：防止落地家具的 mesh 底部（比如餐桌桌腿等）因为穿入/低于地板而被误判为无支撑
        origin_z = max(bottom_z + 0.05, ground_z + 0.05)
        
        # 采集底部支撑点进行射线向下检测
        sample_points = _child_support_sample_points(obj)
        hits = []
        for x, y in sample_points:
            origin_world = _require_mathutils().Vector((x, y, origin_z))
            hit_obj, hit_z = raycast_downwards(origin_world, ignore_obj=obj)
            if hit_obj is not None:
                hits.append((hit_obj, hit_z))
                
        if not hits:
            # 没有检测到任何支撑，说明彻底悬空在虚无中
            floating_results.append({
                "id": obj_id,
                "name": short_desc,
                "category": category,
                "bottom_z": bottom_z,
                "support_type": "None",
                "support_name": "虚空 (无下方支撑)",
                "gap": 999.0,
                "severity": "CRITICAL (彻底悬空)",
                "kind": "no_support",
            })
            continue
            
        # 选择最高的支撑面作为物体的实际支撑面
        best_hit_obj, best_hit_z = max(hits, key=lambda x: x[1])
        floating_issue = classify_floating_state(
            bottom_z=bottom_z,
            support_z=best_hit_z,
            ground_z=ground_z,
            tolerance=tolerance,
        )
        
        # 容忍 1.5 cm 以下的微小空隙 (由于网格 AABB 近似或浮点数微调导致)
        if floating_issue is not None:
            support_desc = best_hit_obj.name
            if "Floor" in best_hit_obj.name:
                support_desc = "地板 (Floor)"
            elif "short_desc" in best_hit_obj:
                support_desc = f"{best_hit_obj['short_desc']} (ID: {best_hit_obj['object_id']})"
                
            floating_results.append({
                "id": obj_id,
                "name": short_desc,
                "category": category,
                "bottom_z": bottom_z,
                "support_type": "MESH",
                "support_name": support_desc,
                "gap": floating_issue["gap"],
                "severity": floating_issue["severity"],
                "kind": floating_issue["kind"],
            })

    return floating_results


def detect_floating(blend_path: str, json_path: str):
    """检测漂浮物体的主函数"""
    # 打开 blend 文件
    bpy.ops.wm.open_mainfile(filepath=blend_path)
    bpy.context.view_layer.update()
    
    # 读取原始 JSON 用于辅助分析类型和父子关系
    with open(json_path, "r", encoding="utf-8") as f:
        layout_data = json.load(f)

    floating_results = detect_floating_objects_in_current_scene(layout_data)

    asset_count = sum(
        1
        for obj in bpy.context.scene.objects
        if obj.type == 'MESH' and "object_id" in obj
    )
    floor_obj = bpy.data.objects.get("Floor")
    ground_z = 0.0
    if floor_obj is not None:
        _, floor_maxs = _world_bbox_bounds(floor_obj)
        ground_z = floor_maxs[2]

    print(f"\n============================================================")
    print(f"正在分析场景: {blend_path}")
    print(f"检测到的资产物体总数: {asset_count}")
    print(f"提取的地板表面 Z 轴高度 (ground_z): {ground_z:.4f} m")
    print(f"============================================================")

    # 输出报告
    print(f"\n物理悬空检测报告:")
    print("-" * 100)
    if not floating_results:
        print("🎉 恭喜！未检测到任何漂浮/悬空物体，所有物体均完美贴合支撑面。")
    else:
        print(f"发现共 {len(floating_results)} 个物体存在悬空/漂浮问题：\n")
        # 按照 gap 严重程度从大到小排序
        floating_results.sort(key=lambda x: x["gap"], reverse=True)
        
        print(f"{'物体ID':<6} | {'物体描述':<30} | {'品类':<12} | {'下方支撑物':<35} | {'悬空高度(cm)':<12} | {'严重程度'}")
        print("-" * 100)
        for r in floating_results:
            gap_cm = r["gap"] * 100.0
            gap_str = f"{gap_cm:.2f} cm" if r["gap"] < 99.0 else "∞"
            print(f"{r['id']:<8} | {r['name'][:30]:<30} | {r['category']:<12} | {r['support_name'][:35]:<35} | {gap_str:<12} | {r['severity']}")
    print("-" * 100)
    print("报告分析完毕。")


if __name__ == "__main__":
    # Blender -P 执行时，脚本参数通过 `--` 之后的 sys.argv 传入
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(args) < 2:
        print(
            "用法: blender -b -P scripts/floating_filter/detection.py -- "
            "<scene.blend> <scene.json>"
        )
        sys.exit(1)
    detect_floating(args[0], args[1])
