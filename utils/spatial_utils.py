"""
空间关系计算的公共函数
用于在数据生成和评估中复用
"""
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
from shapely.geometry import LineString, Point as ShapelyPoint, Polygon as ShapelyPolygon


# ==================== 常量定义 ====================

# 接触阈值（米），用于判断 on/in 等接触关系
CONTACT_THRESHOLD = 0.05

# under 关系的阈值（米）
UNDER_GROUND_THRESHOLD = 0.1

# 水平关系（left/right/in_front/behind/next_to）的底面高度差阈值比例：
# |bottom_A - bottom_B| > max(height_A, height_B) * 该比例 时，排除水平关系。
# 直觉：两个物体底面高度差超过较高物体高度的 80%，就不应该说"旁边/前面/后面"。
HORIZONTAL_RELATION_Z_RATIO = 0.8

# ==================== 距离阈值 ====================

# 距离模式开关：True 使用相对距离（边缘距离/投影长度），False 使用绝对边缘距离
USE_RELATIVE_DISTANCE = False

# --- 相对距离阈值（边缘距离 / 平均方向投影长度） ---
NEXT_TO_RELATIVE_THRESHOLD = 0.5
RELATION_MAX_RELATIVE_THRESHOLD = 1.0

# --- 绝对距离阈值（米） ---
NEXT_TO_ABSOLUTE_THRESHOLD = 0.5  # 边缘距离 < 0.5m 认为是 next_to
RELATION_MAX_ABSOLUTE_THRESHOLD = 1.5  # 边缘距离 < 1.5m 才计算方向关系

# 靠墙相对阈值：物体到墙的距离 < 对角线 * 0.15
WALL_RELATIVE_THRESHOLD = 0.15

# 场景区域划分阈值：物体中心偏离场景中心超过场景尺寸的 25% 才算在边缘区域
REGION_THRESHOLD_RATIO = 0.25

# face_to 关系的角度阈值（度）：A 的 yaw 指向 B，夹角小于此值
FACE_TO_ANGLE_THRESHOLD = 15.0


# ==================== 旋转变换函数 ====================

def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    将欧拉角(度)转换为旋转矩阵。

    约定说明：
    - 本函数使用 SciPy 的小写 ``"xyz"`` 欧拉角约定（intrinsic，绕物体自身坐标轴依次旋转）。
    - 输入顺序固定为 ``[roll, pitch, yaw]``，分别对应绕 X/Y/Z。
    - 该约定与手写矩阵形式 ``R = Rz @ Ry @ Rx``（列向量右乘）等价。
    
    Args:
        roll: X轴旋转角度（度）
        pitch: Y轴旋转角度（度）
        yaw: Z轴旋转角度（度）
    
    Returns:
        3x3 旋转矩阵
    """
    rot = R.from_euler("xyz", [roll, pitch, yaw], degrees=True)
    return rot.as_matrix()


def quat_to_rotation_matrix(quat: List[float]) -> np.ndarray:
    """
    将四元数转换为旋转矩阵
    
    Args:
        quat: 四元数 [w, x, y, z]
    
    Returns:
        3x3 旋转矩阵
    """
    # scipy 的四元数格式是 [x, y, z, w]
    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    return rot.as_matrix()


# ==================== 旋转 OBB / AABB 辅助函数 ====================

def get_obb_world_vertices(
    center: Union[List[float], np.ndarray],
    size: Union[List[float], np.ndarray],
    rot_matrix: np.ndarray,
) -> np.ndarray:
    """
    获取 OBB 的 8 个顶点在世界坐标系下的位置。

    说明：
    - 支持任意姿态（roll/pitch/yaw），因此也能覆盖“物体倒下/放平”的情况。
    - 返回顶点顺序不保证稳定；调用方若需要投影多边形，请自行做凸包。
    """
    center_arr = np.asarray(center, dtype=float)
    size_arr = np.asarray(size, dtype=float)
    half = size_arr / 2.0

    local_vertices = np.array(
        [
            [-half[0], -half[1], -half[2]],
            [+half[0], -half[1], -half[2]],
            [-half[0], +half[1], -half[2]],
            [+half[0], +half[1], -half[2]],
            [-half[0], -half[1], +half[2]],
            [+half[0], -half[1], +half[2]],
            [-half[0], +half[1], +half[2]],
            [+half[0], +half[1], +half[2]],
        ],
        dtype=float,
    )
    world_vertices = (rot_matrix @ local_vertices.T).T + center_arr
    return world_vertices


def get_obb_world_vertices_from_euler(
    center: List[float],
    size: List[float],
    euler: List[float],
) -> np.ndarray:
    rot_matrix = euler_to_rotation_matrix(*euler)
    return get_obb_world_vertices(center=center, size=size, rot_matrix=rot_matrix)


def _get_rotation_matrix_from_obj(obj: Dict) -> np.ndarray:
    rotation = obj.get("rotation")
    if rotation is None:
        raise ValueError("Object rotation is missing.")
    return euler_to_rotation_matrix(*rotation)


def get_obb_world_vertices_from_obj(obj: Dict) -> np.ndarray:
    center = obj.get("center")
    if center is None:
        raise ValueError("Object center is missing.")
    size = obj["size"]
    rot_matrix = _get_rotation_matrix_from_obj(obj)
    return get_obb_world_vertices(center=center, size=size, rot_matrix=rot_matrix)


def get_aabb_from_vertices(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    从顶点计算 3D AABB 的 min/max。
    """
    v = np.asarray(vertices, dtype=float)
    v_min = v.min(axis=0)
    v_max = v.max(axis=0)
    return v_min, v_max


def get_aabb_from_obj(obj: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    获取物体在世界坐标系下的 3D AABB（支持倾斜/翻倒）。
    """
    vertices = get_obb_world_vertices_from_obj(obj)
    return get_aabb_from_vertices(vertices)


def get_object_z_bounds(obj: Dict[str, Any]) -> Tuple[float, float]:
    """获取物体的 z 方向边界 (z_bottom, z_top)（考虑旋转）。"""
    v_min, v_max = get_aabb_from_obj(obj)
    return float(v_min[2]), float(v_max[2])


def check_obb_contains_obj(inner_obj: Dict, outer_obj: Dict, eps: float = CONTACT_THRESHOLD) -> bool:
    """
    检查 inner_obj 的 OBB 是否完全落在 outer_obj 的 OBB 内（含边界）。

    实现方式：
    - 取 inner_obj 的 8 个世界坐标顶点；
    - 变换到 outer_obj 的局部坐标系；
    - 检查每个顶点是否都在 outer_obj 的 half-size 范围内。
    """
    inner_vertices = get_obb_world_vertices_from_obj(inner_obj)
    outer_center = np.asarray(outer_obj["center"], dtype=float)
    outer_half_size = np.asarray(outer_obj["size"], dtype=float) / 2.0
    outer_rotation = _get_rotation_matrix_from_obj(outer_obj)

    outer_local_vertices = (outer_rotation.T @ (inner_vertices - outer_center).T).T
    return bool(np.all(np.abs(outer_local_vertices) <= outer_half_size + eps))


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """
    2D 点集凸包。返回凸包顶点（逆时针排列）。
    基于 scipy.spatial.ConvexHull。
    """
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=float)

    unique = np.unique(pts, axis=0)
    if unique.shape[0] <= 2:
        return unique

    try:
        hull = ConvexHull(unique)
        return unique[hull.vertices]
    except Exception:
        return unique


def point_in_polygon_2d(point: np.ndarray, polygon: np.ndarray, eps: float = 1e-9) -> bool:
    """
    点在多边形内（含边界）。polygon 需要是 (N,2) 且 N>=3。
    基于 Shapely。
    """
    poly = np.asarray(polygon, dtype=float)
    if poly.shape[0] < 3:
        return False
    p = ShapelyPoint(float(point[0]), float(point[1]))
    sp = ShapelyPolygon(poly)
    return sp.contains(p) or sp.boundary.distance(p) <= eps


# ==================== 边界信息函数 ====================

def _prepare_boundary_array(boundary: List[List[float]]) -> np.ndarray:
    """
    将 boundary 规范化为 (N, 2) 的 numpy 数组。
    """
    boundary_array = np.asarray(boundary, dtype=float)

    if boundary_array.size == 0:
        raise ValueError("Boundary is empty")

    if boundary_array.ndim == 1:
        if len(boundary_array) % 2 == 0:
            boundary_array = boundary_array.reshape(-1, 2)
        else:
            raise ValueError(f"Invalid boundary format: 1D array with odd length {len(boundary_array)}")

    if boundary_array.ndim != 2 or boundary_array.shape[1] != 2:
        raise ValueError(f"Invalid boundary shape: {boundary_array.shape}, expected (N, 2)")

    if boundary_array.shape[0] < 3:
        raise ValueError(f"Invalid boundary: need at least 3 points, got {boundary_array.shape[0]}")

    return boundary_array


def _polygon_signed_area_2d(poly: np.ndarray) -> float:
    """
    多边形有向面积。基于 Shapely；CCW 为正、CW 为负。
    """
    p = np.asarray(poly, dtype=float)
    sp = ShapelyPolygon(p)
    area = sp.area
    ring = sp.exterior
    if ring.is_ccw:
        return float(area)
    return float(-area)


def _polygon_centroid_2d(poly: np.ndarray) -> np.ndarray:
    """
    多边形重心。基于 Shapely。面积退化时返回点集均值。
    """
    p = np.asarray(poly, dtype=float)
    sp = ShapelyPolygon(p)
    if sp.area < 1e-12:
        return p.mean(axis=0)
    c = sp.centroid
    return np.array([c.x, c.y], dtype=float)


def _boundary_edges(boundary_array: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    edges: List[Tuple[np.ndarray, np.ndarray]] = []
    n = int(boundary_array.shape[0])
    for i in range(n):
        a = boundary_array[i]
        b = boundary_array[(i + 1) % n]
        edges.append((a, b))
    return edges


def _classify_outward_normal_to_direction(n: np.ndarray) -> str:
    """
    将二维法向量映射到 4 个主方向之一（east/west/north/south）。
    """
    nx, ny = float(n[0]), float(n[1])
    if abs(nx) >= abs(ny):
        return "east" if nx > 0 else "west"
    return "north" if ny > 0 else "south"


def _edge_outward_normal(a: np.ndarray, b: np.ndarray, is_ccw: bool) -> np.ndarray:
    """
    给定多边形边 (a->b)，返回指向“多边形外侧”的单位法向量。
    约定：若顶点为 CCW，则外侧在边的右侧；若为 CW，则外侧在边的左侧。
    """
    e = np.asarray(b - a, dtype=float)
    if np.linalg.norm(e) < 1e-12:
        return np.array([0.0, 0.0], dtype=float)

    # CCW: outward = right normal; CW: outward = left normal
    if is_ccw:
        n = np.array([e[1], -e[0]], dtype=float)
    else:
        n = np.array([-e[1], e[0]], dtype=float)

    norm = float(np.linalg.norm(n))
    if norm < 1e-12:
        return np.array([0.0, 0.0], dtype=float)
    return n / norm


def _vertex_turn_cross_2d(boundary_array: np.ndarray, i: int) -> float:
    """
    计算多边形在顶点 i 处的转向叉积：
    cross( (v_i - v_{i-1}), (v_{i+1} - v_i) )
    """
    n = int(boundary_array.shape[0])
    prev = boundary_array[(i - 1) % n]
    cur = boundary_array[i]
    nxt = boundary_array[(i + 1) % n]
    e1 = cur - prev
    e2 = nxt - cur
    return float(e1[0] * e2[1] - e1[1] * e2[0])


def _is_convex_vertex(boundary_array: np.ndarray, i: int, is_ccw: bool, eps: float = 1e-12) -> bool:
    """
    判断顶点 i 是否为凸角。
    - 对 CCW 多边形：turn_cross > 0 为凸角
    - 对 CW  多边形：turn_cross < 0 为凸角
    """
    c = _vertex_turn_cross_2d(boundary_array, i)
    if abs(c) <= eps:
        return False
    return (c > 0) if is_ccw else (c < 0)


def _corner_name_from_edge_directions(dir1: str, dir2: str) -> Optional[str]:
    """
    由两条相邻边的外法线主方向（east/west/north/south）推断角落名称。
    """
    s = {dir1, dir2}
    if s == {"north", "east"}:
        return "at northeast corner"
    if s == {"north", "west"}:
        return "at northwest corner"
    if s == {"south", "east"}:
        return "at southeast corner"
    if s == {"south", "west"}:
        return "at southwest corner"
    return None


def _polygon_to_segment_distance_2d(polygon: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """
    多边形到线段的最小距离（2D）。基于 Shapely。
    """
    poly = np.asarray(polygon, dtype=float)
    if poly.shape[0] < 3:
        return float("inf")
    sp = ShapelyPolygon(poly)
    seg = LineString([a, b])
    return float(sp.distance(seg))


def get_object_footprint_polygon_xy(obj: Dict) -> np.ndarray:
    """
    获取物体在 XY 平面的“脚印”多边形：OBB 8 顶点投影 -> 凸包。
    """
    vertices = get_obb_world_vertices_from_obj(obj)
    proj_xy = vertices[:, :2]
    hull = convex_hull_2d(proj_xy)

    # 退化情况：用投影 AABB 兜底成矩形
    if hull.shape[0] < 3:
        min_x = float(np.min(proj_xy[:, 0]))
        max_x = float(np.max(proj_xy[:, 0]))
        min_y = float(np.min(proj_xy[:, 1]))
        max_y = float(np.max(proj_xy[:, 1]))
        return np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]], dtype=float)

    return hull


def get_object_footprint_area_xy(obj: Dict) -> float:
    """
    获取物体在 XY 平面的“脚印面积”（单位：m^2）。

    说明：
    - 使用 OBB 8 顶点投影到 XY 后的凸包面积。
    - 对 yaw 旋转不敏感（仅在倾斜/翻倒时面积会随投影变化），比 AABB 面积更稳定。
    """
    poly = get_object_footprint_polygon_xy(obj)
    return float(ShapelyPolygon(poly).area)


def get_object_boundary_overlap_metrics(
    obj: Dict,
    boundary: List[List[float]],
    eps: float = 1e-9,
) -> Dict[str, Any]:
    """
    计算物体 footprint 与房间 boundary 的重叠指标。

    返回:
        - footprint_area: 物体 footprint 面积
        - inside_area: footprint 落在 boundary 内的面积
        - outside_area: footprint 落在 boundary 外的面积
        - outside_ratio: outside_area / footprint_area
        - center_inside_boundary: 物体中心点是否在 boundary 内（含边界）
        - footprint_inside_boundary: footprint 是否几乎完全落在 boundary 内
        - degenerate_footprint: 是否发生了退化 footprint 回退
        - degenerate_reason: 退化原因（若无则为 None）
    """
    boundary_array = _prepare_boundary_array(boundary)
    footprint = get_object_footprint_polygon_xy(obj)
    center_xy = np.asarray(obj["center"][:2], dtype=float)
    center_inside_boundary = point_in_polygon_2d(center_xy, boundary_array, eps=eps)

    boundary_polygon = ShapelyPolygon(boundary_array)
    if not boundary_polygon.is_valid:
        boundary_polygon = boundary_polygon.buffer(0)

    footprint_polygon = ShapelyPolygon(footprint)
    if not footprint_polygon.is_valid:
        footprint_polygon = footprint_polygon.buffer(0)

    if boundary_polygon.is_empty:
        raise ValueError("Boundary polygon is empty after validation.")
    if footprint_polygon.is_empty:
        # 少量退化物体（如 footprint 面积为 0 的线/点）不应打断整批过滤。
        # 此时回退到中心点判定：中心在边界外则视为完全出界，否则视为在界内。
        return {
            "footprint_area": 0.0,
            "inside_area": 0.0,
            "outside_area": 0.0,
            "outside_ratio": 0.0 if center_inside_boundary else 1.0,
            "center_inside_boundary": center_inside_boundary,
            "footprint_inside_boundary": center_inside_boundary,
            "degenerate_footprint": True,
            "degenerate_reason": "empty_polygon",
        }

    footprint_area = float(footprint_polygon.area)
    if footprint_area <= eps:
        # 零面积 footprint 同样退化为中心点判定。
        return {
            "footprint_area": footprint_area,
            "inside_area": footprint_area if center_inside_boundary else 0.0,
            "outside_area": 0.0,
            "outside_ratio": 0.0 if center_inside_boundary else 1.0,
            "center_inside_boundary": center_inside_boundary,
            "footprint_inside_boundary": center_inside_boundary,
            "degenerate_footprint": True,
            "degenerate_reason": "zero_area",
        }

    inside_area = float(boundary_polygon.intersection(footprint_polygon).area)
    inside_area = min(max(inside_area, 0.0), footprint_area)
    outside_area = max(0.0, footprint_area - inside_area)
    outside_ratio = outside_area / footprint_area

    return {
        "footprint_area": footprint_area,
        "inside_area": inside_area,
        "outside_area": outside_area,
        "outside_ratio": outside_ratio,
        "center_inside_boundary": center_inside_boundary,
        "footprint_inside_boundary": outside_area <= eps,
        "degenerate_footprint": False,
        "degenerate_reason": None,
    }


def get_object_diagonal_xy(obj: Dict) -> float:
    """
    计算物体在 XY 平面的对角线长度（基于 size）。
    
    用于计算相对距离阈值，使距离判断能够适应不同大小的物体。
    
    Args:
        obj: 物体信息，包含 size [width, depth, height]
    
    Returns:
        XY 平面对角线长度（米）
    """
    size = obj["size"]
    return math.sqrt(size[0] ** 2 + size[1] ** 2)


def get_object_extent_in_direction(obj: Dict, axis: str) -> float:
    """
    获取物体在指定轴方向上的投影长度（考虑旋转后的 AABB）。
    
    Args:
        obj: 物体信息，包含 center, size, rotation
        axis: 'x' 或 'y'
    
    Returns:
        物体在该轴方向的投影长度（米）
    """
    vertices = get_obb_world_vertices_from_obj(obj)
    if axis == 'x':
        return float(np.max(vertices[:, 0]) - np.min(vertices[:, 0]))
    else:  # 'y'
        return float(np.max(vertices[:, 1]) - np.min(vertices[:, 1]))


def _check_horizontal_z_compatible(obj_a: Dict, obj_b: Dict) -> bool:
    """
    检查两个物体的底面高度差是否兼容水平关系（left/right/front/behind/next_to）。
 
    当两个物体的底面 z 坐标差距过大时（如一个在地面、一个在高处），
    说它们"在旁边"或"在前面"不符合日常直觉，应排除水平关系。

    判定标准：|bottom_A - bottom_B| <= max(height_A, height_B) * HORIZONTAL_RELATION_Z_RATIO
    """
    a_min, a_max = get_aabb_from_obj(obj_a)
    b_min, b_max = get_aabb_from_obj(obj_b)
    bottom_a, top_a = float(a_min[2]), float(a_max[2])
    bottom_b, top_b = float(b_min[2]), float(b_max[2])

    height_a = top_a - bottom_a
    height_b = top_b - bottom_b
    max_height = max(height_a, height_b)

    bottom_diff = abs(bottom_a - bottom_b)
    return bottom_diff <= max_height * HORIZONTAL_RELATION_Z_RATIO


def get_object_edge_distance_xy(obj_a: Dict, obj_b: Dict) -> float:
    """
    计算两个物体在 XY 平面的边缘距离（footprint 之间的最短距离）。
    
    Args:
        obj_a: 物体 A
        obj_b: 物体 B
    
    Returns:
        两个物体 footprint 之间的最短距离（米）
    """
    poly_a = ShapelyPolygon(get_object_footprint_polygon_xy(obj_a))
    poly_b = ShapelyPolygon(get_object_footprint_polygon_xy(obj_b))
    return float(poly_a.distance(poly_b))


def normalize_angle(angle_rad: float) -> float:
    """
    将角度（弧度）规范化到 [-π, π] 范围。
    
    Args:
        angle_rad: 输入角度（弧度）
    
    Returns:
        规范化后的角度（弧度），范围 [-π, π]
    """
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def check_face_to(
    obj_a: Dict,
    obj_b: Dict,
    threshold_deg: float = FACE_TO_ANGLE_THRESHOLD,
    threshold_relative_dist: float = RELATION_MAX_RELATIVE_THRESHOLD,
) -> bool:
    """
    检查物体 A 是否朝向物体 B（A 的正面指向 B）。
    
    基于物体 A 的 yaw 角度（rotation[2]）判断是否指向物体 B。
    约定：yaw = 0 时物体正面朝向 -Y 方向（即 -90° 方向）。
    
    同时考虑距离限制，太远的物体不认为存在 face_to 关系。
    使用边缘距离和方向投影长度来判断相对距离。
    
    Args:
        obj_a: 物体 A，包含 center, size, rotation
        obj_b: 物体 B，包含 center, size
        threshold_deg: 角度阈值（度），夹角小于此值认为朝向
        threshold_relative_dist: 相对距离阈值，小于此值才认为 face_to
    
    Returns:
        A 的正面是否指向 B
    """
    pos_a = obj_a["center"]
    pos_b = obj_b["center"]
    
    # A 到 B 的方向角（弧度）
    dx = pos_b[0] - pos_a[0]
    dy = pos_b[1] - pos_a[1]
    
    # 如果 A 和 B 重合，无法判断朝向
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return False
    
    # 计算边缘距离
    edge_distance = get_object_edge_distance_xy(obj_a, obj_b)
    
    # 根据 A→B 方向选择投影轴：主要在 X 方向用 X 投影，主要在 Y 方向用 Y 投影
    if abs(dx) >= abs(dy):
        extent_a = get_object_extent_in_direction(obj_a, 'x')
        extent_b = get_object_extent_in_direction(obj_b, 'x')
    else:
        extent_a = get_object_extent_in_direction(obj_a, 'y')
        extent_b = get_object_extent_in_direction(obj_b, 'y')
    
    avg_extent = (extent_a + extent_b) / 2.0
    relative_distance = edge_distance / avg_extent if avg_extent > 1e-9 else float("inf")
    
    # 距离太远，不认为存在 face_to 关系
    # 根据 USE_RELATIVE_DISTANCE 开关选择使用相对距离还是绝对边缘距离
    if USE_RELATIVE_DISTANCE:
        if relative_distance > threshold_relative_dist:
            return False
    else:
        if edge_distance > RELATION_MAX_ABSOLUTE_THRESHOLD:
            return False
    
    direction_to_b = math.atan2(dy, dx)
    
    # A 的 yaw 角度（rotation[2]，单位：度）转为弧度
    # 初始朝向为 -Y（-90°），所以实际朝向 = yaw + (-90°)
    yaw_a = math.radians(obj_a["rotation"][2] - 90)
    
    # 计算夹角（规范化到 [-π, π]）
    angle_diff = abs(normalize_angle(direction_to_b - yaw_a))
    
    return angle_diff < math.radians(threshold_deg)


def get_boundary_info(boundary: List[List[float]]) -> Dict[str, Any]:
    """
    从边界点计算场景的边界信息

    Args:
        boundary: 场景边界点列表 [[x1, y1], [x2, y2], ...]

    Returns:
        边界信息字典，包含 min_x, max_x, min_y, max_y, center, centroid, width, depth。
        - center: AABB 几何中心 ((min+max)/2)，用于区域划分
        - centroid: 多边形重心，用于角落象限判定等需要反映不规则形状内部中心的场景
        - boundary: 规范化后的边界点（list[list[float]]）
    """
    boundary_array = _prepare_boundary_array(boundary)

    min_x = float(np.min(boundary_array[:, 0]))
    max_x = float(np.max(boundary_array[:, 0]))
    min_y = float(np.min(boundary_array[:, 1]))
    max_y = float(np.max(boundary_array[:, 1]))
    
    width = max_x - min_x
    depth = max_y - min_y

    centroid = _polygon_centroid_2d(boundary_array)
    
    return {
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "center": [(min_x + max_x) / 2.0, (min_y + max_y) / 2.0],
        "centroid": [float(centroid[0]), float(centroid[1])],
        "width": width,
        "depth": depth,
        "boundary": boundary_array.tolist(),
    }


# ==================== 包围盒函数 ====================

def get_obb_aabb_xy_from_euler(
    pos: List[float], 
    size: List[float], 
    euler: List[float]
) -> Tuple[float, float, float, float]:
    """
    计算物体在XY平面上的轴对齐包围盒 (AABB)
    使用欧拉角表示旋转
    
    Args:
        pos: 物体中心位置 [x, y, z]
        size: 物体尺寸 [width, depth, height]
        euler: 物体旋转（欧拉角）[roll, pitch, yaw]
    
    Returns:
        (min_x, max_x, min_y, max_y): AABB的边界
    """
    world_vertices = get_obb_world_vertices_from_euler(pos, size, euler)
    v_min, v_max = get_aabb_from_vertices(world_vertices)
    min_x, max_x = float(v_min[0]), float(v_max[0])
    min_y, max_y = float(v_min[1]), float(v_max[1])
    return min_x, max_x, min_y, max_y


# ==================== 空间关系计算函数 ====================

def calculate_relative_relation(obj_a: Dict, obj_b: Dict) -> Dict[str, Any]:
    """
    计算物体A相对于物体B的空间关系
    
    使用边缘距离和方向投影长度来判断关系：
    - edge_distance: 两个物体 footprint 之间的最短距离
    - relative_distance: edge_distance / avg_extent（基于方向的投影长度）
    
    水平关系（基于边缘相对距离和角度）：
    - Left of: θ_AB in [135, 180] or in (-180, -135), relative_distance < RELATION_MAX_RELATIVE_THRESHOLD
    - Right of: θ_AB in [-45, 45), relative_distance < RELATION_MAX_RELATIVE_THRESHOLD
    - In front of: θ_AB in [45, 135), relative_distance < RELATION_MAX_RELATIVE_THRESHOLD
    - Behind: θ_AB in [-135, -45), relative_distance < RELATION_MAX_RELATIVE_THRESHOLD
    - Next to: relative_distance < NEXT_TO_RELATIVE_THRESHOLD
    - Face to: A 的 yaw 指向 B
    
    垂直关系：
    - On: A的底部接近B的顶部，且 A 的中心在 B 的 XY 投影内
    - Under: A 的中心在 B 的 XY 投影内，且 A 的底部接近 B 的底部，且 A 的中心 z 低于 B 的顶部
    - Within: A 的 3D OBB 完全在 B 的 3D OBB 内
    
    - None: relative_distance >= RELATION_MAX_RELATIVE_THRESHOLD 且无其他关系
    
    Args:
        obj_a: 物体A，包含 center, size, rotation (欧拉角)
        obj_b: 物体B，同上
    
    Returns:
        包含 edge_distance, relative_distance, angle, relations 等信息的字典
    """
    pos_a = obj_a["center"]
    pos_b = obj_b["center"]

    X_A, Y_A, Z_A = pos_a[0], pos_a[1], pos_a[2]
    X_B, Y_B, Z_B = pos_b[0], pos_b[1], pos_b[2]

    # 计算边缘距离
    edge_distance = get_object_edge_distance_xy(obj_a, obj_b)

    # 计算相对角度（度数）- 用于判断方向
    dx = X_A - X_B
    dy = Y_A - Y_B
    theta_AB = math.degrees(math.atan2(dy, dx))

    # 根据方向选择投影轴：
    # - left/right (主要 X 方向): 用 X 方向投影长度
    # - front/behind (主要 Y 方向): 用 Y 方向投影长度
    # 使用角度判断主方向
    abs_theta = abs(theta_AB)
    if 45 <= abs_theta <= 135:
        # 主要在 Y 方向（front/behind）
        extent_a = get_object_extent_in_direction(obj_a, 'y')
        extent_b = get_object_extent_in_direction(obj_b, 'y')
        direction_axis = 'y'
    else:
        # 主要在 X 方向（left/right）
        extent_a = get_object_extent_in_direction(obj_a, 'x')
        extent_b = get_object_extent_in_direction(obj_b, 'x')
        direction_axis = 'x'
    
    avg_extent = (extent_a + extent_b) / 2.0
    
    # 防止除零
    if avg_extent < 1e-6:
        avg_extent = 1e-6
    
    relative_distance = edge_distance / avg_extent

    # 检查 Inside 条件（使用 xy 平面投影，中心点判定）
    inside_A_in_B = check_inside_xy_projection(obj_a, obj_b)

    relations = []
    
    # 计算垂直方向的边界（支持倾斜/翻倒：使用旋转后 3D AABB 的 Z 范围）
    a_min, a_max = get_aabb_from_obj(obj_a)
    b_min, b_max = get_aabb_from_obj(obj_b)
    bottom_A, top_A = float(a_min[2]), float(a_max[2])
    bottom_B, top_B = float(b_min[2]), float(b_max[2])

    # 检查 On 关系（A on B，如书在桌上）
    # 放宽条件：只要 A 的中心点落在 B 的 XY 投影内（投影后），且 A 的底部接近 B 的顶部
    if inside_A_in_B:
        bottom_to_top_dist = abs(bottom_A - top_B)
        if bottom_to_top_dist < CONTACT_THRESHOLD:
            relations.append("on")

    # 检查 Within 关系（A within B：A 的 3D OBB 完全在 B 的 3D OBB 内）
    within_3d = check_obb_contains_obj(obj_a, obj_b, eps=CONTACT_THRESHOLD)
    if within_3d:
        # 排除 on 的情况（on 语义优先）
        if "on" not in relations:
            relations.append("within")

    # 检查 Under 关系（A under B，如盒子在桌下）
    # 仅使用 bbox 时，“under”容易与“沙发上抱枕”等情况混淆，因此加上“底部接近 B 底部”的约束。
    # 注意：under 与 within 可能同时成立（例如桌子 AABB 覆盖了桌下空间）。
    # 会根据物体 B 的类别选择更合适的一个写进指令。
    if inside_A_in_B:
        is_bottom_close_to_B = abs(bottom_A - bottom_B) <= UNDER_GROUND_THRESHOLD
        is_center_below_top = Z_A < top_B
        if is_bottom_close_to_B and is_center_below_top:
            # 排除 on 的情况（避免边界抖动导致同时出现）
            if not (abs(bottom_A - top_B) < CONTACT_THRESHOLD):
                relations.append("under")

    # 检查水平关系
    # 前提：两个物体底面高度差不能过大（否则不符合"旁边/前面/后面"的日常语义）
    z_compatible = _check_horizontal_z_compatible(obj_a, obj_b)

    # 根据 USE_RELATIVE_DISTANCE 开关选择使用相对距离还是绝对边缘距离
    if USE_RELATIVE_DISTANCE:
        is_within_range = relative_distance < RELATION_MAX_RELATIVE_THRESHOLD
        is_next_to = relative_distance < NEXT_TO_RELATIVE_THRESHOLD
    else:
        is_within_range = edge_distance < RELATION_MAX_ABSOLUTE_THRESHOLD
        is_next_to = edge_distance < NEXT_TO_ABSOLUTE_THRESHOLD
    
    # 排除垂直/包含关系：on/under/within 表示叠放或包含，不应该再有水平方向关系
    has_vertical_relation = any(r in relations for r in ("on", "under", "within"))
    if is_within_range and z_compatible and not has_vertical_relation:
        if is_next_to:
            relations.append("next_to")
        
        if 135 <= theta_AB <= 180 or -180 < theta_AB <= -135:
            relations.append("left")
        elif -45 <= theta_AB < 45:
            relations.append("right")
        elif 45 <= theta_AB < 135:
            relations.append("in_front")
        elif -135 <= theta_AB < -45:
            relations.append("behind")
    
    # 检查 face_to 关系（A 的朝向指向 B）
    if check_face_to(obj_a, obj_b):
        relations.append("face_to")

    return {
        "edge_distance": edge_distance,
        "relative_distance": relative_distance,
        "direction_axis": direction_axis,
        "extent_a": extent_a,
        "extent_b": extent_b,
        "angle": theta_AB,
        "relations": relations if relations else ["none"],
    }


def _evaluate_absolute_relations(
    obj: Dict,
    boundary_info: Dict,
) -> Optional[Dict[str, Any]]:
    """计算物体相对于场景边界的各方向靠墙、角落与区域状态；若物体悬空则返回 None。"""
    pos = obj["center"]
    size = obj["size"]
    rotation = obj["rotation"]

    obj_min, _obj_max = get_aabb_from_obj(obj)
    bottom_z = float(obj_min[2])
    if bottom_z > UNDER_GROUND_THRESHOLD:
        return None

    obj_diagonal = get_object_diagonal_xy(obj)
    wall_threshold = obj_diagonal * WALL_RELATIVE_THRESHOLD

    scene_center_x = float(boundary_info["center"][0])
    scene_center_y = float(boundary_info["center"][1])
    scene_width = float(boundary_info["width"])
    scene_depth = float(boundary_info["depth"])

    boundary = boundary_info["boundary"]
    if boundary is None:
        scene_min_x = boundary_info["min_x"]
        scene_max_x = boundary_info["max_x"]
        scene_min_y = boundary_info["min_y"]
        scene_max_y = boundary_info["max_y"]

        obj_min_x, obj_max_x, obj_min_y, obj_max_y = get_obb_aabb_xy_from_euler(pos, size, rotation)

        dist_to_east = abs(scene_max_x - obj_max_x)
        dist_to_west = abs(obj_min_x - scene_min_x)
        dist_to_north = abs(scene_max_y - obj_max_y)
        dist_to_south = abs(obj_min_y - scene_min_y)

        against_east = dist_to_east < wall_threshold
        against_west = dist_to_west < wall_threshold
        against_north = dist_to_north < wall_threshold
        against_south = dist_to_south < wall_threshold

        corner_name: Optional[str] = None
        if against_north and against_east:
            corner_name = "at northeast corner"
        elif against_north and against_west:
            corner_name = "at northwest corner"
        elif against_south and against_east:
            corner_name = "at southeast corner"
        elif against_south and against_west:
            corner_name = "at southwest corner"
        is_corner = corner_name is not None
    else:
        boundary_array = _prepare_boundary_array(boundary)
        poly = get_object_footprint_polygon_xy(obj)
        edges = _boundary_edges(boundary_array)

        edge_distances: List[float] = [
            _polygon_to_segment_distance_2d(poly, a, b) for a, b in edges
        ]

        is_ccw = _polygon_signed_area_2d(boundary_array) > 0
        dir_best: Dict[str, float] = {
            "east": float("inf"),
            "west": float("inf"),
            "north": float("inf"),
            "south": float("inf"),
        }
        for idx, (a, b) in enumerate(edges):
            n = _edge_outward_normal(a, b, is_ccw=is_ccw)
            if float(np.linalg.norm(n)) < 1e-12:
                continue
            dkey = _classify_outward_normal_to_direction(n)
            dir_best[dkey] = min(dir_best[dkey], float(edge_distances[idx]))

        against_east = dir_best["east"] < wall_threshold
        against_west = dir_best["west"] < wall_threshold
        against_north = dir_best["north"] < wall_threshold
        against_south = dir_best["south"] < wall_threshold

        n = int(boundary_array.shape[0])
        best_corner: Optional[Tuple[int, float]] = None
        for i in range(n):
            if not _is_convex_vertex(boundary_array, i, is_ccw=is_ccw):
                continue
            d1 = float(edge_distances[(i - 1) % n])
            d2 = float(edge_distances[i])
            if d1 < wall_threshold and d2 < wall_threshold:
                score = d1 + d2
                if best_corner is None or score < best_corner[1]:
                    best_corner = (i, score)

        corner_name = None
        if best_corner is not None:
            vidx = int(best_corner[0])
            e_prev = edges[(vidx - 1) % n]
            e_next = edges[vidx]
            n1 = _edge_outward_normal(e_prev[0], e_prev[1], is_ccw=is_ccw)
            n2 = _edge_outward_normal(e_next[0], e_next[1], is_ccw=is_ccw)
            d1 = _classify_outward_normal_to_direction(n1) if float(np.linalg.norm(n1)) > 1e-12 else ""
            d2 = _classify_outward_normal_to_direction(n2) if float(np.linalg.norm(n2)) > 1e-12 else ""
            corner_name = _corner_name_from_edge_directions(d1, d2) if d1 and d2 else None

            if corner_name is None:
                centroid = np.asarray(boundary_info["centroid"], dtype=float)
                vx, vy = boundary_array[vidx]
                dx = float(vx - centroid[0])
                dy = float(vy - centroid[1])
                if dy >= 0 and dx >= 0:
                    corner_name = "at northeast corner"
                elif dy >= 0 and dx < 0:
                    corner_name = "at northwest corner"
                elif dy < 0 and dx >= 0:
                    corner_name = "at southeast corner"
                else:
                    corner_name = "at southwest corner"

        is_corner = corner_name is not None

    against_any_wall = bool(against_east or against_west or against_north or against_south)

    obj_x = float(pos[0])
    obj_y = float(pos[1])
    offset_x = obj_x - scene_center_x
    offset_y = obj_y - scene_center_y
    x_threshold = scene_width * REGION_THRESHOLD_RATIO
    y_threshold = scene_depth * REGION_THRESHOLD_RATIO

    in_front = offset_y > y_threshold
    in_back = offset_y < -y_threshold
    in_right = offset_x > x_threshold
    in_left = offset_x < -x_threshold
    in_center = not (in_front or in_back or in_right or in_left)

    return {
        "against_east": against_east,
        "against_west": against_west,
        "against_north": against_north,
        "against_south": against_south,
        "against_any_wall": against_any_wall,
        "is_corner": is_corner,
        "corner_name": corner_name,
        "in_front": in_front,
        "in_back": in_back,
        "in_right": in_right,
        "in_left": in_left,
        "in_center": in_center,
    }


def calculate_absolute_relation(
    obj: Dict,
    boundary_info: Dict,
    directional: bool = False,
) -> List[str]:
    """
    计算物体相对于场景的绝对位置关系
    """
    state = _evaluate_absolute_relations(obj, boundary_info)
    if state is None:
        return []

    is_corner = state["is_corner"]
    corner_name = state["corner_name"]
    against_any_wall = state["against_any_wall"]
    against_east = state["against_east"]
    against_west = state["against_west"]
    against_north = state["against_north"]
    against_south = state["against_south"]
    in_front = state["in_front"]
    in_back = state["in_back"]
    in_right = state["in_right"]
    in_left = state["in_left"]
    in_center = state["in_center"]

    if not directional:
        if is_corner:
            return ["at corner"]
        if against_any_wall:
            return ["against wall"]
        if in_front:
            return ["in front region"]
        if in_back:
            return ["in back region"]
        if in_right:
            return ["in right region"]
        if in_left:
            return ["in left region"]
        if in_center:
            return ["in center region"]
        return []

    absolute_relations: List[str] = []
    if is_corner and corner_name is not None:
        absolute_relations.append(corner_name)

    if not is_corner:
        if against_east and not (against_north or against_south or against_west):
            absolute_relations.append("against east wall")
        if against_west and not (against_north or against_south or against_east):
            absolute_relations.append("against west wall")
        if against_north and not (against_east or against_west or against_south):
            absolute_relations.append("against north wall")
        if against_south and not (against_east or against_west or against_north):
            absolute_relations.append("against south wall")

    if not absolute_relations:
        if in_front:
            absolute_relations.append("in front region")
        if in_back:
            absolute_relations.append("in back region")
        if in_right:
            absolute_relations.append("in right region")
        if in_left:
            absolute_relations.append("in left region")
        if in_center:
            absolute_relations.append("in center region")

    return absolute_relations

# ==================== 空间关系判断函数 ====================

def check_inside_xy_projection(obj_a: Dict, obj_b: Dict) -> bool:
    """
    检查物体 A 的中心在 xy 平面的投影是否在物体 B 的 OBB 在 xy 平面的投影内
    
    用于 on/under/within 等关系的辅助判定：只检查 xy 平面的投影，忽略 z 坐标。
    
    Args:
        obj_a: 物体A信息，包含 center, size, rotation
        obj_b: 物体B信息，包含 center, size, rotation
    
    Returns:
        A的中心在XY平面的投影是否在B的OBB在XY平面的投影内
    """
    pos_a = obj_a.get("center")
    if pos_a is None:
        return False

    # 只考虑 xy 平面，忽略 z 坐标
    p_xy = np.array([pos_a[0], pos_a[1]], dtype=float)

    # 支持倾斜/翻倒：用 B 的 8 个顶点投影到 XY，并对投影做凸包
    try:
        vertices_b = get_obb_world_vertices_from_obj(obj_b)
    except (KeyError, TypeError, ValueError):
        return False

    proj_xy = vertices_b[:, :2]
    hull = convex_hull_2d(proj_xy)

    # 退化情况（投影成线/点）：回退到投影 AABB
    if hull.shape[0] < 3:
        min_x = float(np.min(proj_xy[:, 0]))
        max_x = float(np.max(proj_xy[:, 0]))
        min_y = float(np.min(proj_xy[:, 1]))
        max_y = float(np.max(proj_xy[:, 1]))
        eps = 1e-9
        return bool(
            (min_x - eps <= p_xy[0] <= max_x + eps) and (min_y - eps <= p_xy[1] <= max_y + eps)
        )

    return bool(point_in_polygon_2d(p_xy, hull, eps=1e-9))


def check_relative_relation(new_obj: Dict, target_obj: Dict, relation_type: str) -> bool:
    """
    检查两个物体之间是否满足指定的空间关系
    
    使用边缘距离和方向投影长度来判断水平关系。
    
    Args:
        new_obj: 新物体（物体A），包含 center, size, rotation
        target_obj: 目标物体（物体B），包含 center, size, rotation
        relation_type: 关系类型，必须为 calculate_relative_relation 返回的标准名称：
            "left", "right", "in_front", "behind", "next_to", "face_to", "on", "under", "within"
    
    Returns:
        是否满足指定关系
    """
    relation_type = relation_type.lower()
    center_new = np.array(new_obj["center"])
    center_target = np.array(target_obj["center"])

    # 计算相对角度 θ_AB
    dx = center_new[0] - center_target[0]
    dy = center_new[1] - center_target[1]
    theta = math.degrees(math.atan2(dy, dx))

    # 计算边缘距离
    edge_distance = get_object_edge_distance_xy(new_obj, target_obj)
    
    # 根据方向选择投影轴
    abs_theta = abs(theta)
    if 45 <= abs_theta <= 135:
        # 主要在 Y 方向（front/behind）
        extent_new = get_object_extent_in_direction(new_obj, 'y')
        extent_target = get_object_extent_in_direction(target_obj, 'y')
    else:
        # 主要在 X 方向（left/right）
        extent_new = get_object_extent_in_direction(new_obj, 'x')
        extent_target = get_object_extent_in_direction(target_obj, 'x')
    
    avg_extent = (extent_new + extent_target) / 2.0
    if avg_extent < 1e-6:
        avg_extent = 1e-6
    relative_distance = edge_distance / avg_extent

    # 垂直边界（支持倾斜/翻倒：使用旋转后 3D AABB 的 Z 范围）
    new_min, new_max = get_aabb_from_obj(new_obj)
    target_min, target_max = get_aabb_from_obj(target_obj)
    bottom_new, top_new = float(new_min[2]), float(new_max[2])
    bottom_target, top_target = float(target_min[2]), float(target_max[2])

    # 根据 USE_RELATIVE_DISTANCE 开关选择使用相对距离还是绝对边缘距离
    if USE_RELATIVE_DISTANCE:
        max_threshold = RELATION_MAX_RELATIVE_THRESHOLD
        next_to_threshold = NEXT_TO_RELATIVE_THRESHOLD
        distance_value = relative_distance
    else:
        max_threshold = RELATION_MAX_ABSOLUTE_THRESHOLD
        next_to_threshold = NEXT_TO_ABSOLUTE_THRESHOLD
        distance_value = edge_distance

    # 水平关系的前置检查：
    # 1. 底面高度差不能过大
    # 2. 不能同时处于 on/under/within 关系（叠放/包含时不应有水平关系）
    z_compatible = _check_horizontal_z_compatible(new_obj, target_obj)

    def _has_vertical_relation() -> bool:
        xy_inside = check_inside_xy_projection(new_obj, target_obj)
        is_on = xy_inside and abs(bottom_new - top_target) < CONTACT_THRESHOLD
        is_under = (
            xy_inside
            and abs(bottom_new - bottom_target) <= UNDER_GROUND_THRESHOLD
            and float(center_new[2]) < top_target
            and not is_on
        )
        is_within = check_obb_contains_obj(new_obj, target_obj, eps=CONTACT_THRESHOLD)
        return is_on or is_under or is_within

    is_horizontal = relation_type in [
        "left", "right", "in_front", "behind", "next_to",
    ]
    if is_horizontal:
        if not z_compatible or _has_vertical_relation():
            return False

    if relation_type == "left":
        return (
            (135 <= theta <= 180 or -180 < theta <= -135)
            and distance_value < max_threshold
        )

    elif relation_type == "right":
        return (
            -45 <= theta < 45
            and distance_value < max_threshold
        )

    elif relation_type == "in_front":
        return (
            45 <= theta < 135
            and distance_value < max_threshold
        )

    elif relation_type == "behind":
        return (
            -135 <= theta < -45
            and distance_value < max_threshold
        )

    elif relation_type == "next_to":
        return distance_value < next_to_threshold
    
    elif relation_type == "face_to":
        return check_face_to(new_obj, target_obj)

    elif relation_type == "on":
        # A on B: A 的中心点在 B 的 XY 投影内，且 A 的底部接近 B 的顶部
        xy_within = check_inside_xy_projection(new_obj, target_obj)
        bottom_to_top_dist = abs(bottom_new - top_target)
        return bool(xy_within and bottom_to_top_dist < CONTACT_THRESHOLD)

    elif relation_type == "under":
        # A under B: A 的中心在 B 的 XY 投影内；A 的底部接近 B 的底部；A 的中心 z 低于 B 的顶部
        inside_A_in_B = check_inside_xy_projection(new_obj, target_obj)
        if not inside_A_in_B:
            return False
        is_bottom_close_to_B = abs(bottom_new - bottom_target) <= UNDER_GROUND_THRESHOLD
        is_center_below_top = float(center_new[2]) < top_target
        is_on_top = abs(bottom_new - top_target) < CONTACT_THRESHOLD
        return bool(is_bottom_close_to_B and is_center_below_top and not is_on_top)

    elif relation_type == "within":
        # A within B: A 的 3D OBB 完全在 B 的 3D OBB 内
        return check_obb_contains_obj(new_obj, target_obj, eps=CONTACT_THRESHOLD)

    else:
        return False


def check_absolute_relation(
    obj: Dict, boundary: List[List[float]], relation_type: str
) -> bool:
    """
    检查绝对关系
    
    relation_type 必须为 calculate_absolute_relation 返回的标准名称（带空格）：
    
    非方向性：
    - "against wall": 物体靠墙（任意墙）
    - "at corner": 物体在角落（任意角落）
    - "in front region": 物体在场景前部
    - "in back region": 物体在场景后部
    - "in left region": 物体在场景左部
    - "in right region": 物体在场景右部
    - "in center region": 物体在场景中心
    
    方向性：
    - "against east wall": 靠东墙 (+X)
    - "against west wall": 靠西墙 (-X)
    - "against north wall": 靠北墙 (+Y)
    - "against south wall": 靠南墙 (-Y)
    - "at northeast corner": 东北角
    - "at northwest corner": 西北角
    - "at southeast corner": 东南角
    - "at southwest corner": 西南角
    
    Args:
        obj: 物体信息，包含 center, size, rotation
        boundary: 场景边界点列表
        relation_type: 关系类型（calculate_absolute_relation 返回的标准名称）
    
    Returns:
        是否满足指定的绝对关系
    """
    boundary_info = get_boundary_info(boundary)
    rel = " ".join((relation_type or "").strip().lower().split())
    state = _evaluate_absolute_relations(obj, boundary_info)
    if state is None:
        return False

    non_directional = {
        "against wall": state["against_any_wall"] or state["is_corner"],
        "at corner": state["is_corner"],
        "in front region": state["in_front"],
        "in back region": state["in_back"],
        "in left region": state["in_left"],
        "in right region": state["in_right"],
        "in center region": state["in_center"],
    }
    if rel in non_directional:
        return bool(non_directional[rel])

    directional = {
        "against east wall": state["against_east"],
        "against west wall": state["against_west"],
        "against north wall": state["against_north"],
        "against south wall": state["against_south"],
        "at northeast corner": state["corner_name"] == "at northeast corner",
        "at northwest corner": state["corner_name"] == "at northwest corner",
        "at southeast corner": state["corner_name"] == "at southeast corner",
        "at southwest corner": state["corner_name"] == "at southwest corner",
    }
    if rel in directional:
        return bool(directional[rel])

    return False

