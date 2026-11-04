"""
预处理 3D 资产，生成点云数据

此脚本支持从以下格式采样点云，并保存为 .ply 格式：
  - OBJ/FBX 文件（单文件或整个目录）

点云密度根据物体的表面积和复杂度自适应调整。
同时保存包含采样信息（期望点数、实际点数等）的元数据 JSON 文件。

颜色采样：使用 trimesh 从纹理采样 RGB 颜色（支持 vertex_colors、face_colors、UV 纹理）

Usage:
  # 处理单个 OBJ/FBX 文件
  python dataset_pipeline/preprocessing/00_preprocess_asset_pointclouds.py --file /path/to/raw_model.obj

  # 处理整个 OBJ/FBX 资产目录（每个子目录是一个资产）
  python dataset_pipeline/preprocessing/00_preprocess_asset_pointclouds.py --dir path/to/3D-FUTURE-model
  python dataset_pipeline/preprocessing/00_preprocess_asset_pointclouds.py --dir path/to/assets --output ./asset_pointclouds
"""

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import trimesh
from plyfile import PlyData
from tqdm import tqdm
import os


def determine_pointcloud_size(
    mesh: trimesh.Trimesh,
    min_points: int = 512,
    max_points: int = 16384,
) -> int:
    """
    根据表面积自适应确定点云采样数量

    使用平滑衰减的点密度策略：
        density = base_density / (1 + area * decay_rate)

    参数设定 (base_density=5000, decay_rate=0.15)：
    - 0.1 m² (小物体): ~4926 点/m² → ~493 点
    - 0.5 m² (小物体): ~4651 点/m² → ~2326 点
    - 2.0 m² (中等):   ~3846 点/m² → ~7692 点
    - 5.0 m² (中等):   ~2857 点/m² → ~14286 点
    - 10 m² (大物体):  ~2000 点/m² → ~20000 点 (受 max_points 限制)
    - 20 m² (大物体):  ~1250 点/m² → ~25000 点 (受 max_points 限制)

    优点：
    - 密度随表面积平滑衰减，无边界跳变
    - 小物体获得足够的细节
    - 大物体自然趋向较低密度

    Args:
        mesh: trimesh 对象
        min_points: 最小点数，默认 512
        max_points: 最大点数，默认 16384

    Returns:
        num_points: 采样点数
    """
    # 计算表面积
    area = mesh.area

    # 平滑衰减密度策略
    # density(area) = base_density / (1 + area * decay_rate)
    # - base_density: 极小物体的理论最大密度
    # - decay_rate: 控制衰减速度，越大衰减越快
    base_density = 5000.0
    decay_rate = 0.15
    density = base_density / (1.0 + area * decay_rate)

    # 根据表面积和密度计算点数
    calculated_points = int(area * density)

    # 限制在 [min_points, max_points] 范围内
    num_points = max(min_points, min(calculated_points, max_points))

    # 量化到常用的档位（便于批处理和内存管理）
    valid_sizes = [512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384]

    # 找到最接近的档位
    num_points = min(valid_sizes, key=lambda x: abs(x - num_points))

    return num_points


def save_pointcloud_ply(
    path: str,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
):
    """
    保存点云为 PLY 格式（支持 xyz + rgb + normals）

    Args:
        path: 输出文件路径
        points: (N, 3) 点云坐标
        colors: (N, 3) RGB 颜色 (uint8)，可选
        normals: (N, 3) 法向量，可选
    """
    n = len(points)
    has_color = colors is not None
    has_normal = normals is not None

    # 构建 header (顺序: xyz rgb normals)
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_color:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    if has_normal:
        header += ["property float nx", "property float ny", "property float nz"]
    header.append("end_header\n")

    # 构建数据 (顺序: xyz rgb normals)
    dtype_list = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if has_color:
        dtype_list += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    if has_normal:
        dtype_list += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]

    vertex_data = np.zeros(n, dtype=dtype_list)
    vertex_data["x"], vertex_data["y"], vertex_data["z"] = (
        points[:, 0],
        points[:, 1],
        points[:, 2],
    )
    if has_color:
        vertex_data["red"], vertex_data["green"], vertex_data["blue"] = (
            colors[:, 0],
            colors[:, 1],
            colors[:, 2],
        )
    if has_normal:
        vertex_data["nx"], vertex_data["ny"], vertex_data["nz"] = (
            normals[:, 0],
            normals[:, 1],
            normals[:, 2],
        )

    with open(path, "wb") as f:
        f.write("\n".join(header).encode("ascii"))
        vertex_data.tofile(f)


def transform_coordinates(points: np.ndarray) -> np.ndarray:
    """
    坐标系转换：从 Y-up 转换到 Z-up

    3D-FUTURE 原始坐标系：Y轴朝上，原点在底面中心
    目标坐标系：Z轴朝上，原点在几何中心

    转换关系：(x, y, z) -> (x, -z, y)
    - x' = x (保持不变)
    - y' = -z (z取反后变成新的y)
    - z' = y (原来的y变成新的z)

    Args:
        points: (N, 3) 原始点云坐标

    Returns:
        transformed_points: (N, 3) 转换后的点云坐标
    """
    # 坐标变换矩阵：(x, y, z) -> (x, -z, y)
    transform_matrix = np.array(
        [
            [1, 0, 0],  # x' = x
            [0, 0, -1],  # y' = -z
            [0, 1, 0],  # z' = y
        ],
        dtype=np.float32,
    )

    return points @ transform_matrix.T


def sample_pointcloud_from_mesh(
    mesh: trimesh.Trimesh,
    num_points: Optional[int] = None,
    source_path: Optional[Path] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], dict]:
    """
    从 mesh 表面采样点云（使用 uniform 采样）

    Args:
        mesh: trimesh 对象
        num_points: 目标采样点数（None 时自动确定）
        source_path: 用于纹理回退解析的源文件路径（OBJ/FBX->OBJ）

    Returns:
        points: (N, 3) 点云坐标（在物体 LCS 中）
        colors: (N, 3) 点云 RGB 颜色（uint8, 0-255），无颜色时为 None
        normals: (N, 3) 点云法向量（float32，单位向量），无法向量时为 None
        metadata: 点云元信息（原始尺度、中心等）
    """
    # 自适应确定点云数量
    if num_points is None:
        num_points = determine_pointcloud_size(mesh)

    # 记录期望采样的点数
    requested_points = num_points

    # 使用 trimesh 的标准均匀采样（保证精确点数）
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)

    # 为采样点附加 RGB 颜色:
    # 1) 优先使用已存在的 vertex_colors
    # 2) 其次使用 face_colors
    # 3) 最后尝试使用 UV + 纹理贴图, 按重心坐标从 texture 采样颜色
    colors: Optional[np.ndarray] = None
    has_visual = False
    has_vertex_colors = False
    has_face_colors = False
    has_uv = False
    has_texture_image = False
    has_kd_color = False
    try:
        visual = getattr(mesh, "visual", None)
        if visual is not None:
            has_visual = True
            # 计算采样点的重心坐标（用于插值顶点颜色）
            triangles = mesh.triangles[face_indices]  # (N, 3, 3)
            barycentric = trimesh.triangles.points_to_barycentric(
                triangles=triangles,
                points=points,
            )  # (N, 3)
            barycentric = np.nan_to_num(barycentric, nan=0.0)
            face_vertices = mesh.faces[face_indices]  # (N, 3)

            # 1) vertex_colors 路径（顶点颜色）
            vertex_colors = getattr(visual, "vertex_colors", None)
            has_vertex_colors = (
                vertex_colors is not None and len(vertex_colors) == len(mesh.vertices)
            )
            if vertex_colors is not None and len(vertex_colors) == len(mesh.vertices):
                # 插值顶点颜色到采样点
                tri_colors = vertex_colors[face_vertices, :3]  # (N, 3, 3)
                colors = (
                    (tri_colors * barycentric[:, :, None]).sum(axis=1).astype(np.uint8)
                )

            # 2) face_colors 路径 (例如已经烤好的面片颜色)
            if colors is None:
                face_colors = getattr(visual, "face_colors", None)
                has_face_colors = face_colors is not None and len(face_colors) == len(mesh.faces)
                if face_colors is not None and len(face_colors) == len(mesh.faces):
                    colors = face_colors[face_indices, :3].astype(np.uint8)
                    # 确保是 3 通道
                    if colors.shape[1] != 3:
                        if colors.shape[1] == 4:
                            colors = colors[:, :3]
                        elif colors.shape[1] == 1:
                            colors = np.tile(colors, (1, 3))
                        else:
                            colors = None

            # 3) 纹理贴图路径: 使用 UV + material.image 采样颜色
            if colors is None:
                uv = getattr(visual, "uv", None)
                material = getattr(visual, "material", None)
                has_uv = uv is not None

                # 尝试多种获取纹理图像的方式
                image = None
                if material is not None:
                    # 方式1: material.image (OBJ 常用)
                    image = getattr(material, "image", None)
                    # 方式2: 从 material 的其他属性获取
                    if image is None and hasattr(material, "to_simple"):
                        try:
                            simple_mat = material.to_simple()
                            image = getattr(simple_mat, "image", None)
                        except Exception as exc:
                            print(f"  [DEBUG] material.to_simple() 失败: {exc}")

                # 如果材质里拿不到 image，则尝试从 OBJ/MTL 解析
                if image is None and source_path is not None:
                    image = _load_basecolor_image_from_mtl(source_path)
                has_texture_image = image is not None

                if uv is not None and image is not None:
                    # 强制转换为 RGB 模式，处理 Gray+Alpha (LA) 或 CMYK 等特殊格式
                    if hasattr(image, "convert"):
                        try:
                            image = image.convert("RGB")
                        except Exception as exc:
                            print(f"  [DEBUG] 纹理转 RGB 失败，将使用原图: {exc}")

                    texture = np.asarray(image)
                    # 灰度/单通道纹理: 扩展为 3 通道
                    if texture.ndim == 2:
                        texture = np.stack([texture] * 3, axis=-1)
                    h, w = texture.shape[:2]

                    # 根据重心坐标插值出采样点的 UV
                    tri_uvs = uv[face_vertices]  # (N, 3, 2)
                    point_uvs = (tri_uvs * barycentric[:, :, None]).sum(
                        axis=1
                    )  # (N, 2)
                    point_uvs = np.clip(point_uvs, 0.0, 1.0)

                    # UV -> 纹理像素坐标
                    # 注意：UV 坐标是 OpenGL 约定（Y 轴向上），需要翻转 Y 轴
                    # 以匹配图像坐标系（Y 轴向下）
                    x = (point_uvs[:, 0] * (w - 1)).astype(np.int32)
                    y = ((1 - point_uvs[:, 1]) * (h - 1)).astype(np.int32)
                    x = np.clip(x, 0, w - 1)
                    y = np.clip(y, 0, h - 1)

                    # 从纹理中读取 RGB
                    colors = texture[y, x, :3].astype(np.uint8)

                    # 再次确保是 3 通道 (处理 potential edge cases)
                    if colors.shape[1] != 3:
                        if colors.shape[1] == 4:
                            colors = colors[:, :3]
                        elif colors.shape[1] == 1:
                            colors = np.tile(colors, (1, 3))
                        else:
                            # 如果形状依然异常(如 2 通道)，放弃颜色
                            colors = None
                # 无纹理但有 Kd 颜色时，回退为常量颜色
                if colors is None and source_path is not None:
                    kd_color = _load_kd_color_from_mtl(source_path)
                    if kd_color is not None:
                        has_kd_color = True
                        colors = np.tile(kd_color, (len(points), 1))
    except Exception as e:
        # 颜色采样失败时, 静默退化为无颜色点云, 不中断整个流程
        print(f"  [DEBUG] 颜色采样失败: {e}")
        colors = None

    if colors is None:
        src = str(source_path) if source_path is not None else "<unknown>"
        print(
            "  [WARN] 无法采样颜色，输出无 RGB 点云: "
            f"{src} | visual={has_visual} vertex_colors={has_vertex_colors} "
            f"face_colors={has_face_colors} uv={has_uv} texture={has_texture_image} "
            f"kd_fallback={has_kd_color}"
        )

    # 采样法向量（使用重心坐标从顶点法向量插值）
    normals: Optional[np.ndarray] = None
    try:
        triangles = mesh.triangles[face_indices]  # (N, 3, 3)
        barycentric = trimesh.triangles.points_to_barycentric(triangles, points)
        barycentric = np.nan_to_num(barycentric, nan=0.0)
        face_vertices = mesh.faces[face_indices]  # (N, 3)

        if mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(
            mesh.vertices
        ):
            # 顶点法向量插值（更平滑）
            tri_normals = mesh.vertex_normals[face_vertices]  # (N, 3, 3)
            normals = (tri_normals * barycentric[:, :, None]).sum(axis=1)
        else:
            # 退化为面法向量
            print(f"  顶点法向量不存在，退化为面法向量")
            normals = mesh.face_normals[face_indices]
        # 归一化
        norms = np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-8, None)
        normals = (normals / norms).astype(np.float32)
    except Exception as e:
        print(f"  [DEBUG] 法向量采样失败: {e}")
        normals = None

    # 记录原始尺度信息（在坐标转换之前）
    bounds = mesh.bounds
    bbox_center = (bounds[0] + bounds[1]) / 2  # 边界框中心
    size = bounds[1] - bounds[0]

    # 计算点密度（用于统计分析）
    point_density = len(points) / max(mesh.area, 0.001)  # 点/m²

    metadata = {
        "requested_points": requested_points,  # 期望采样的点数
        "actual_points": len(points),  # 实际采样的点数
        "point_density": float(point_density),  # 点密度（点/m²）
        "area": float(mesh.area),  # 表面面积, 单位: m²
        "volume": float(abs(mesh.volume)) if mesh.is_volume else 0.0,  # 体积, 单位: m³
    }

    # 坐标系转换（在居中之前）
    # 转换点云坐标：(x, y, z) -> (x, -z, y)
    # Y up, Z forward -> Z up, -Y forward
    points = transform_coordinates(points)
    # 转换边界框中心和质心
    bbox_center = transform_coordinates(bbox_center.reshape(1, 3)).flatten()
    # 转换尺寸：size = [width, height, depth] -> [width, depth, height]
    size = np.array([size[0], size[2], size[1]])
    # 转换法向量（旋转变换）
    if normals is not None:
        normals = transform_coordinates(normals)

    metadata["transformed_bbox_center"] = bbox_center.tolist()
    metadata["transformed_size"] = size.tolist()

    # 居中操作：使用边界框中心
    points = points - bbox_center

    return points.astype(np.float32), colors, normals, metadata


def _resolve_mtl_path_from_obj(source_path: Path) -> Optional[Path]:
    """
    为 OBJ 解析实际的 MTL 路径。

    优先级：
    读取 OBJ 内的 mtllib 指令
    目录下常见命名 model.mtl（3D-FUTURE）
    目录下唯一的 .mtl
    """
    try:
        source_path = Path(source_path)
        if source_path.suffix.lower() != ".obj" or not source_path.exists():
            return None

        # 1) 优先按 OBJ 的 mtllib 指向解析
        for line in source_path.read_text(errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("mtllib"):
                remainder = line[len("mtllib") :].strip()
                if not remainder:
                    continue
                for mtl_name in remainder.split():
                    candidate = source_path.parent / mtl_name
                    if candidate.exists():
                        return candidate

        # 3D-FUTURE 常见命名
        candidate = source_path.parent / "model.mtl"
        if candidate.exists():
            return candidate

        # 目录下唯一 mtl
        mtl_files = sorted(source_path.parent.glob("*.mtl"))
        if len(mtl_files) == 1:
            return mtl_files[0]
    except Exception:
        return None
    return None


def _load_basecolor_image_from_mtl(source_path: Path):
    """
    从 OBJ 对应的 MTL 中解析 base color 纹理（map_Kd），
    兼容带有 -o/-s 等参数的导出格式。
    """
    try:
        mtl_path = _resolve_mtl_path_from_obj(source_path)
        if mtl_path is None:
            return None

        for line in mtl_path.read_text(errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("map_kd"):
                tokens = line.split()
                if len(tokens) < 2:
                    continue
                # Blender 的导出可能包含 -o/-s 参数，取最后一个 token 作为贴图文件名
                texture_name = tokens[-1]
                texture_path = mtl_path.parent / texture_name
                if texture_path.exists():
                    from PIL import Image

                    return Image.open(texture_path)
    except Exception:
        return None
    return None


def _load_kd_color_from_mtl(source_path: Path) -> Optional[np.ndarray]:
    """从 MTL 解析 Kd 颜色并转换为 uint8 RGB。"""
    try:
        mtl_path = _resolve_mtl_path_from_obj(source_path)
        if mtl_path is None:
            return None

        for line in mtl_path.read_text(errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("kd "):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        r, g, b = float(parts[1]), float(parts[2]), float(parts[3])
                    except ValueError:
                        return None
                    rgb = np.clip([r, g, b], 0.0, 1.0) * 255.0
                    return rgb.astype(np.uint8)
    except Exception:
        return None
    return None


def fbx_to_obj_blender(fbx_file: Path, obj_file: Path, timeout: int = 300) -> bool:
    """用 Blender 将单个 FBX 转为 OBJ (带纹理提取)"""
    import subprocess
    import shutil

    # 自动检测 Blender 路径
    blender_path = shutil.which("blender")  # 先从 PATH 找
    if blender_path is None:
        # 尝试常见路径
        for path in [
            "/Applications/Blender.app/Contents/MacOS/Blender",  # macOS
            "/usr/local/bin/blender",  # Linux
        ]:
            if Path(path).exists():
                blender_path = path
                break
    if blender_path is None:
        return False

    output_dir = obj_file.parent

    script = f"""
import bpy
from pathlib import Path

# 清空场景
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# 导入 FBX
bpy.ops.import_scene.fbx(filepath="{fbx_file}")

# 提取嵌入的纹理（确保文件名唯一，避免覆盖）
saved_texture_paths = set()
for img in bpy.data.images:
    if img.packed_file:
        try:
            # 使用纹理的内部名称作为基础文件名
            base_name = img.name
            texture_path = Path("{output_dir}") / base_name
            
            # 根据扩展名设置格式
            if img.name.lower().endswith('.jpg') or img.name.lower().endswith('.jpeg'):
                img.file_format = 'JPEG'
            elif img.name.lower().endswith('.png'):
                img.file_format = 'PNG'
            else:
                # EXR 等其他格式转为 PNG
                img.file_format = 'PNG'
                texture_path = texture_path.with_suffix('.png')
            
            # 确保文件名唯一，避免同名纹理互相覆盖
            counter = 0
            final_path = texture_path
            while str(final_path) in saved_texture_paths:
                counter += 1
                stem = texture_path.stem
                suffix = texture_path.suffix
                final_path = texture_path.parent / f"{{stem}}_{{counter}}{{suffix}}"
            
            img.filepath_raw = str(final_path)
            img.save()
            saved_texture_paths.add(str(final_path))
        except RuntimeError as e:
            # 跳过损坏的纹理，继续处理其他纹理
            print(f"Warning: Failed to save texture {{img.name}}: {{e}}")

# 导出 OBJ
bpy.ops.wm.obj_export(filepath="{obj_file}", export_selected_objects=False, export_materials=True)
"""

    script_file = output_dir / f"_temp_{fbx_file.stem}.py"
    script_file.write_text(script)

    try:
        result = subprocess.run(
            [blender_path, "--background", "--python", str(script_file)],
            capture_output=True,
            timeout=timeout,
        )
        script_file.unlink()

        obj_exists = obj_file.exists() and obj_file.stat().st_size > 0

        # Blender 可能在导出完成后异常退出（如 segfault），此时 OBJ 仍可用
        if result.returncode != 0:
            if obj_exists:
                print(f"  [WARN] Blender 异常退出但 OBJ 已生成，继续处理:")
            else:
                print(f"  [ERROR] Blender FBX 转换失败:")

            print(f"    - FBX 文件: {fbx_file}")
            print(f"    - 目标 OBJ: {obj_file}")
            print(f"    - 返回码: {result.returncode}")
            print(f"    - OBJ 文件存在: {obj_exists}")
            if result.stdout:
                stdout_str = result.stdout.decode("utf-8", errors="ignore")[
                    -2000:
                ]  # 最后 2000 字符
                print(f"    - Blender stdout:\n{stdout_str}")
            if result.stderr:
                stderr_str = result.stderr.decode("utf-8", errors="ignore")[-2000:]
                print(f"    - Blender stderr:\n{stderr_str}")

            return obj_exists

        if not obj_exists:
            print(f"  [ERROR] Blender FBX 转换失败: OBJ 未生成")
            print(f"    - FBX 文件: {fbx_file}")
            print(f"    - 目标 OBJ: {obj_file}")
            return False

        return True
    except Exception as e:
        print(f"  [ERROR] Blender 执行异常: {e}")
        if script_file.exists():
            script_file.unlink()
        return False


def preprocess_single_asset(
    jid: str,
    obj_file: Path,
    output_dir: Optional[Path] = None,  # 改为可选参数
    num_points: Optional[int] = None,
) -> Tuple[bool, Optional[str]]:
    """
    预处理单个资产

    Args:
        jid: 资产 ID
        obj_file: .obj 或 .fbx 文件路径
        output_dir: 输出目录（None 时保存到资产原目录）
        num_points: 采样点数（None 时自动）

    Returns:
        (success, error_message): 是否成功和失败原因（成功时为 None）
    """
    try:
        # 使用 PIL 的宽松模式来处理可能损坏的纹理图片
        from PIL import ImageFile

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        # 确定输出目录：如果未指定，使用资产原目录
        if output_dir is None:
            output_dir = obj_file.parent

        import shutil

        # 准备加载的模型文件路径
        model_to_load = obj_file

        # 如果是 FBX 文件，转换为 OBJ 并持久化到资产原目录
        if obj_file.suffix.lower() == ".fbx":
            persistent_obj = obj_file.parent / f"{jid}.obj"

            if persistent_obj.exists() and persistent_obj.stat().st_size > 0:
                model_to_load = persistent_obj
            else:
                with tempfile.TemporaryDirectory() as temp_dir_str:
                    temp_dir = Path(temp_dir_str)
                    temp_obj_file = temp_dir / f"{jid}.obj"

                    if not fbx_to_obj_blender(obj_file, temp_obj_file):
                        return False, "Blender 转换 FBX 失败"

                    # 持久化 OBJ + MTL + 纹理到资产原目录
                    for f in temp_dir.iterdir():
                        if f.is_file():
                            shutil.copy2(f, obj_file.parent / f.name)

                model_to_load = persistent_obj

        # 加载 mesh
        mesh = trimesh.load(str(model_to_load), process=False, force="mesh")

        # 如果加载的是 Scene，尝试合并为单个 mesh
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        # 检查是否有效
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
            error_msg = "无效的 mesh（类型错误或无顶点）"
            print(f"  [WARN] {jid}: {error_msg}")
            return False, error_msg

        # 采样点云 (包含可选的 RGB 颜色和法向量)
        points, colors, normals, metadata = sample_pointcloud_from_mesh(
            mesh,
            num_points=num_points,
            source_path=model_to_load,
        )

        # 元数据
        metadata["has_color"] = colors is not None
        metadata["has_normal"] = normals is not None

        # 保存点云为 .ply 格式（包含 xyz + rgb + normals）
        os.makedirs(str(output_dir), exist_ok=True)
        ply_file = output_dir / f"{jid}.ply"
        save_pointcloud_ply(str(ply_file), points, colors, normals)

        # 保存元数据
        metadata_file = output_dir / f"{jid}_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return True, None

    except Exception as e:
        error_msg = str(e)
        print(f"  [ERROR] {jid}: {error_msg}")
        return False, error_msg


def _find_model_file(jid_dir: Path) -> Tuple[Optional[Path], Optional[str]]:
    """
    查找资产目录中的模型文件

    Returns:
        (model_file, error_msg): 模型文件路径和错误信息（如果未找到）
    """
    jid = jid_dir.name

    # 查找模型文件: 优先 3D-FUTURE 的 raw_model.obj，其次 Imaginarium 的 {jid}.fbx
    obj_file = jid_dir / "raw_model.obj"
    fbx_file = jid_dir / f"{jid}.fbx"

    if obj_file.exists():
        return obj_file, None
    elif fbx_file.exists():
        return fbx_file, None
    else:
        return None, "No .obj or .fbx file found"


def _process_single_obj_worker(
    jid_dir: Path,
    output_path: Optional[Path],
    num_points: Optional[int],
    overwrite: bool,
) -> Tuple[Optional[bool], Optional[str], str, str]:
    """
    OBJ/FBX 资产处理的 worker 函数（用于多进程）

    Returns:
        (success, error_msg, jid, model_file_path)
        - success=None 表示跳过
    """
    jid = jid_dir.name

    # 查找模型文件
    model_file, error_msg = _find_model_file(jid_dir)
    if model_file is None:
        return False, error_msg, jid, str(jid_dir)

    # 确定输出文件位置
    if output_path:
        output_file = output_path / f"{jid}.ply"
    else:
        output_file = jid_dir / f"{jid}.ply"

    # 检查是否已存在
    if output_file.exists() and not overwrite:
        return None, "skipped", jid, str(model_file)

    # 处理
    success, error_msg = preprocess_single_asset(
        jid=jid,
        obj_file=model_file,
        output_dir=output_path,
        num_points=num_points,
    )

    return success, error_msg, jid, str(model_file)


def preprocess_assets(
    asset_dir: str,
    output_dir: Optional[str] = None,
    num_points: Optional[int] = None,
    overwrite: bool = True,
    max_assets: Optional[int] = None,
    num_workers: Optional[int] = None,
):
    """
    批量预处理所有资产（OBJ/FBX）（支持多进程并行）

    Args:
        asset_dir: 3D-FUTURE 资产目录
        output_dir: 点云输出目录（None 时保存到各资产原目录）
        num_points: 固定采样点数（None 时自适应）
        overwrite: 是否覆盖已有文件
        max_assets: 最大处理资产数（None 时处理全部）
        num_workers: 并行工作进程数（默认为 CPU 核心数）
    """
    asset_path = Path(asset_dir)

    # 如果指定了输出目录，创建它
    output_path = None
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    # 遍历所有 jid 目录
    jid_dirs = sorted([d for d in asset_path.iterdir() if d.is_dir()])

    # 限制最大处理数
    if max_assets is not None:
        jid_dirs = jid_dirs[:max_assets]

    total_assets = len(jid_dirs)
    print(f"找到 {total_assets} 个资产目录")
    if output_path:
        print(f"输出目录: {output_path}")
    else:
        print(f"输出策略: 保存到各资产原目录")
    print(f"采样策略: {'自适应' if num_points is None else f'固定 {num_points} 点'}")
    print(f"采样方法: uniform (trimesh 从纹理采样 RGB)")

    if total_assets == 0:
        return

    # 确定工作进程数（未指定时默认上限为 10，显式传入则按用户指定值）
    if num_workers is None:
        num_workers = min(cpu_count(), 10, total_assets)
    num_workers = max(1, min(num_workers, total_assets))
    print(f"并行进程数: {num_workers}")
    print("-" * 60)

    # 创建 worker 函数
    worker_func = partial(
        _process_single_obj_worker,
        output_path=output_path,
        num_points=num_points,
        overwrite=overwrite,
    )

    # 使用多进程池并行处理
    with Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(worker_func, jid_dirs),
                total=total_assets,
                desc="Processing assets",
            )
        )

    # 统计结果
    processed = 0
    skipped = 0
    failed = 0
    pointcloud_size_stats = {}
    complexity_list = []
    failed_assets = []

    for success, error_msg, jid, model_file_path in results:
        if success is None:
            # 跳过的文件
            skipped += 1
        elif success:
            processed += 1
            # 统计点云数量和记录 complexity
            if output_path:
                metadata_file = output_path / f"{jid}_metadata.json"
            else:
                metadata_file = Path(model_file_path).parent / f"{jid}_metadata.json"
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    size = metadata.get("actual_points", 0)
                    pointcloud_size_stats[size] = pointcloud_size_stats.get(size, 0) + 1
                    # 记录 complexity 值（如果存在）
                    if "complexity" in metadata:
                        complexity_list.append(
                            {
                                "jid": jid,
                                "complexity": metadata["complexity"],
                                "actual_points": metadata["actual_points"],
                            }
                        )
            except Exception as exc:
                print(f"  [WARN] 读取 metadata 失败，跳过统计: {metadata_file} | {exc}")
        else:
            failed += 1
            failed_assets.append(
                {"jid": jid, "reason": error_msg, "obj_file_path": model_file_path}
            )

    # 保存统计信息到 JSON 文件
    stats_output = {
        "summary": {
            "total_assets": total_assets,
            "processed": processed,
            "skipped": skipped,
            "failed": failed,
        },
        "pointcloud_size_distribution": {
            str(size): count for size, count in sorted(pointcloud_size_stats.items())
        },
        "complexity_distribution": complexity_list,
        "failed_assets": failed_assets,
        "config": {
            "sampling_strategy": (
                "adaptive" if num_points is None else f"fixed_{num_points}"
            ),
            "num_workers": num_workers,
        },
    }

    # 保存到资产目录下
    stats_file = asset_path / "pointcloud_preprocessing_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats_output, f, indent=2, ensure_ascii=False)

    # 输出统计信息
    print("\n" + "=" * 60)
    print(f"处理完成:")
    print(f"  - 成功处理: {processed}")
    print(f"  - 跳过（已存在）: {skipped}")
    print(f"  - 失败: {failed}")
    print(f"  - 总计: {total_assets}")

    # 输出失败的资产详情
    if failed_assets:
        print(f"\n失败的资产列表:")
        for idx, asset_info in enumerate(failed_assets[:20], 1):
            print(f"  {idx}. JID: {asset_info['jid']}")
            print(f"     原因: {asset_info['reason']}")
            print(f"     路径: {asset_info['obj_file_path']}")

    if output_path:
        print(f"\n点云文件保存在: {output_path}")
    else:
        print(f"\n点云文件保存在: 各资产原目录")

    print(f"\n统计信息已保存到: {stats_file}")
    print("=" * 60)


def _collect_ply_files(root: Path, recursive: bool = True) -> List[Path]:
    if root.is_file() and root.suffix.lower() == ".ply":
        return [root]
    if recursive:
        return sorted(root.rglob("*.ply"))
    return sorted(root.glob("*.ply"))


def _inspect_ply(ply_path: Path) -> Dict[str, object]:
    result: Dict[str, object] = {
        "path": str(ply_path),
        "status": "ok",
        "point_count": 0,
        "has_xyz": False,
        "has_rgb": False,
        "has_normals": False,
        "properties": "",
        "error": "",
    }
    try:
        plydata = PlyData.read(str(ply_path))
        if "vertex" not in plydata:
            result["status"] = "no_vertex"
            return result
        vertex = plydata["vertex"]
        result["point_count"] = int(getattr(vertex, "count", 0))
        prop_names = tuple(vertex.data.dtype.names or ())
        result["properties"] = ",".join(prop_names)
        result["has_xyz"] = all(p in prop_names for p in ("x", "y", "z"))
        result["has_rgb"] = all(p in prop_names for p in ("red", "green", "blue"))
        result["has_normals"] = all(p in prop_names for p in ("nx", "ny", "nz"))
        return result
    except Exception as e:
        result["status"] = "read_error"
        result["error"] = str(e)
        return result


def check_ply_fields(
    target_path: Path,
    recursive: bool = True,
    report_path: Optional[Path] = None,
) -> bool:
    """检查已有 PLY 文件是否包含完整的 xyz / rgb / normals 字段。"""
    if not target_path.exists():
        raise FileNotFoundError(f"路径不存在: {target_path}")

    ply_files = _collect_ply_files(target_path, recursive=recursive)
    reports = [_inspect_ply(p) for p in ply_files]

    total = len(reports)
    missing_xyz = sum(1 for r in reports if not r["has_xyz"])
    missing_rgb = sum(1 for r in reports if not r["has_rgb"])
    missing_normals = sum(1 for r in reports if not r["has_normals"])
    read_errors = sum(1 for r in reports if r["status"] == "read_error")
    no_vertex = sum(1 for r in reports if r["status"] == "no_vertex")

    print(f"检查 PLY 文件数: {total}")
    print(f"读取失败: {read_errors} | 无 vertex: {no_vertex}")
    print(f"缺失 xyz: {missing_xyz} | 缺失 rgb: {missing_rgb} | 缺失 normals: {missing_normals}")

    has_any_issue = any([missing_xyz, missing_rgb, missing_normals, read_errors, no_vertex])
    if has_any_issue:
        print("\n问题文件（前 20 个）：")
        shown = 0
        for r in reports:
            has_issue = (
                r["status"] != "ok"
                or not r["has_xyz"]
                or not r["has_rgb"]
                or not r["has_normals"]
            )
            if not has_issue:
                continue
            issues = []
            if r["status"] == "read_error":
                issues.append("read_error")
            if r["status"] == "no_vertex":
                issues.append("no_vertex")
            if not r["has_xyz"]:
                issues.append("missing_xyz")
            if not r["has_rgb"]:
                issues.append("missing_rgb")
            if not r["has_normals"]:
                issues.append("missing_normals")
            print(f"- {r['path']} | {','.join(issues)}")
            shown += 1
            if shown >= 20:
                break

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "path",
            "status",
            "point_count",
            "has_xyz",
            "has_rgb",
            "has_normals",
            "properties",
            "error",
        ]
        with report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(reports)
        print(f"\n已保存检查报告: {report_path}")

    return not has_any_issue


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="预处理 3D 资产生成点云数据 (.ply 格式)，或检查已有 PLY 字段完整性",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理单个 OBJ/FBX 文件
  python dataset_pipeline/preprocessing/00_preprocess_asset_pointclouds.py --file path/to/model.obj

  # 处理整个资产目录
  python dataset_pipeline/preprocessing/00_preprocess_asset_pointclouds.py --dir path/to/3D-FUTURE-model

  # 指定输出目录和采样点数
  python dataset_pipeline/preprocessing/00_preprocess_asset_pointclouds.py --dir path/to/assets --output ./pointclouds --num-points 4096

  # 仅检查已有 PLY 文件是否包含完整的 xyz / rgb / normals 字段
  python dataset_pipeline/preprocessing/00_preprocess_asset_pointclouds.py --dir datasets/Imaginarium/imaginarium_assets --check-only
        """,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--file", "-f", type=str, help="Path to a single OBJ/FBX/PLY file"
    )
    input_group.add_argument(
        "--dir",
        "-d",
        type=str,
        help="Asset directory path (each subdirectory is an asset)",
    )
    # 输出配置
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: save to asset original directory)",
    )
    # 采样配置
    parser.add_argument(
        "--num-points",
        "-n",
        type=int,
        default=None,
        help="固定采样点数（默认自适应: 512-16384）",
    )
    # 处理控制
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的点云文件")
    parser.add_argument(
        "--max-assets",
        type=int,
        default=None,
        help="最大处理资产数量（仅用于 --dir 模式）",
    )
    parser.add_argument(
        "--jid",
        type=str,
        default=None,
        help="指定资产 ID（仅用于 --file 模式，默认从文件路径推断）",
    )
    parser.add_argument(
        "--num-workers",
        "-j",
        type=int,
        default=None,
        help="并行工作进程数（默认为 CPU 核心数）",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="仅检查已有 PLY 文件是否包含 xyz/rgb/normals 字段（不重新采样）",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="在 --check-only 模式下可选输出 CSV 检查报告路径",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.check_only:
        target = Path(args.file or args.dir)
        report = Path(args.report) if args.report else None
        ok = check_ply_fields(target, recursive=True, report_path=report)
        if not ok:
            sys.exit(1)
        return

    output_dir = args.output

    if args.file:
        # === 单文件模式 ===
        obj_file = Path(args.file)
        if not obj_file.exists():
            print(f"❌ 文件不存在: {obj_file}")
            sys.exit(1)

        # 推断 JID（资产 ID）
        if args.jid:
            jid = args.jid
        else:
            # 从父目录名推断
            jid = obj_file.parent.name

        print("=" * 60)
        print(f"处理单个资产:")
        print(f"  - 文件: {obj_file}")
        print(f"  - JID: {jid}")
        print(f"  - 输出目录: {output_dir or obj_file.parent}")
        print(
            f"  - 采样策略: {'自适应' if args.num_points is None else f'固定 {args.num_points} 点'}"
        )
        print("=" * 60)

        success, error_msg = preprocess_single_asset(
            jid=jid,
            obj_file=obj_file,
            output_dir=Path(output_dir) if output_dir else None,
            num_points=args.num_points,
        )

        if success:
            out_path = Path(output_dir) if output_dir else obj_file.parent
            print(f"\n✅ 处理成功！")
            print(f"   点云文件: {out_path / f'{jid}.ply'}")
            print(f"   元数据: {out_path / f'{jid}_metadata.json'}")
        else:
            print(f"\n❌ 处理失败: {error_msg}")
            sys.exit(1)

    else:
        # === 目录模式（OBJ/FBX） ===
        asset_dir = args.dir
        if not Path(asset_dir).exists():
            print(f"❌ 目录不存在: {asset_dir}")
            sys.exit(1)

        preprocess_assets(
            asset_dir=asset_dir,
            output_dir=output_dir,
            num_points=args.num_points,
            overwrite=args.overwrite,
            max_assets=args.max_assets,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
