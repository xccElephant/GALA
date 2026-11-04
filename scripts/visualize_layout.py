"""
使用 Blender 可视化生成的 Layout JSON

这个脚本需要在 Blender 中运行，使用真实 3D 模型（3D-FUTURE 或 Imaginarium）替换包围盒渲染场景。

用法 (在命令行中运行 Blender):
    # 渲染单张场景图 - 自动识别 3D-FUTURE/Imaginarium 资产（同一场景可混用）
    blender -b -P scripts/visualize_layout.py -- \
        --layout output/qwen2_5-0_5b-text/extra_trained_on_imaginarium_bedroom_23_layout.json \
        --render-image output/visualizations/scene_auto.png \
        --dataset auto

    # 渲染单张场景图 - 强制使用 Imaginarium 资产（FBX）
    blender -b -P scripts/visualize_layout.py -- \
        --layout output/qwen2_5-0_5b-text/extra_trained_on_imaginarium_bedroom_23_layout.json \
        --render-image output/visualizations/scene_imaginarium.png \
        --dataset imaginarium

    # 渲染并做后处理（按 alpha 内容裁剪、等比缩放居中贴回固定大小方图）
    blender -b -P scripts/visualize_layout.py -- \
        --layout output/qwen2_5-0_5b-text/extra_trained_on_imaginarium_bedroom_23_layout.json \
        --render-image output/visualizations/scene_auto.png \
        --postprocess

用法 (在 Blender 中作为脚本运行):
    1. 打开 Blender
    2. 切换到 Scripting 工作区
    3. 打开此脚本
    4. 修改底部的参数后运行
"""

import argparse
import json
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, NamedTuple

import bpy
import bmesh

# ==================== 配置 ====================

script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from blender_scene_setup import setup_indoor_lighting_and_render  # type: ignore

# 尝试导入项目配置
try:
    # 添加项目根目录到 Python 路径
    project_root = script_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from config import ProjectPaths
    from utils.scene_types import scene_from_external_dict

    THREED_FUTURE_MODEL_DIR = ProjectPaths.THREED_FUTURE_MODEL_DIR
    IMAGINARIUM_ASSETS_DIR = ProjectPaths.IMAGINARIUM_ASSETS_DIR
except ImportError:
    # 如果无法导入配置，使用默认路径（需要手动修改）
    THREED_FUTURE_MODEL_DIR = Path(
        "/path/to/datasets/3D-FRONT/3D-FUTURE-model"
    )
    IMAGINARIUM_ASSETS_DIR = Path(
        "/path/to/datasets/Imaginarium/imaginarium_assets"
    )
    scene_from_external_dict = None

DEFAULT_FLOOR_TEXTURE = "assets/floor_texture.jpg"
DONE_PREFIX = "__GALA_RENDER_DONE__ "
PROFILE_PREFIX = "__GALA_RENDER_PROFILE__ "


# ==================== 工具函数 ====================


def _finish_profile_stage(timings: dict[str, float], stage: str, started_at: float) -> float:
    now = time.perf_counter()
    timings[stage] = now - started_at
    return now


def _emit_profile(
    *,
    scene_id: str,
    layout_path: Path,
    render_image_path: str | None,
    timings: dict[str, float],
) -> None:
    payload = {
        "scene_id": scene_id,
        "layout": str(layout_path),
        "render_image": render_image_path,
        "stages": {key: round(value, 4) for key, value in timings.items()},
    }
    print(f"{PROFILE_PREFIX}{json.dumps(payload, ensure_ascii=False, sort_keys=True)}", flush=True)


def clear_scene():
    """清空场景中的所有物体"""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    cached_meshes = set(globals().get("_ASSET_MESH_CACHE", {}).values())
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0 and mesh not in cached_meshes:
            bpy.data.meshes.remove(mesh)
    for material in list(bpy.data.materials):
        if material.users == 0:
            bpy.data.materials.remove(material)
    for image in list(bpy.data.images):
        if image.users == 0:
            bpy.data.images.remove(image)


def save_blend_file(output_path: str | Path, pack_assets: bool = True):
    """保存 .blend 文件；默认把外部贴图打包进文件，避免移动后丢纹理。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if pack_assets:
        print("打包外部贴图和资产到 .blend 文件...")
        try:
            bpy.ops.file.pack_all()
        except RuntimeError as exc:
            print(f"警告: 部分外部贴图无法打包，将继续保存 .blend: {exc}")

    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
    return output_path


def render_image_file(output_path: str | Path) -> Path:
    """将当前场景渲染为单张静态图片。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)
    print(f"已渲染图片: {output_path}")
    return output_path


def _bpy_image_to_array(img) -> "Any":
    """把 bpy.data.images 的像素读成 (H, W, C) 的 float32 numpy 数组（已翻转为常规的从上到下顺序）。

    Blender 的 Image.pixels 是从左下角开始按行存储的（Y 轴向上），
    这里翻转为图像处理里更常见的从左上角开始（Y 轴向下）的顺序，方便与 PIL 等工具的裁剪坐标系保持一致。
    """
    import numpy as np

    w, h = img.size
    channels = img.channels
    buf = np.empty(w * h * channels, dtype=np.float32)
    img.pixels.foreach_get(buf)
    arr = buf.reshape(h, w, channels)
    return arr[::-1, :, :]


def _array_to_bpy_image(arr: "Any", name: str, *, float_buffer: bool = False, non_color: bool = False):
    """把从上到下顺序的 (H, W, C) numpy 数组写回一张新的 bpy.data.images（内部翻转回 Blender 的存储顺序）。

    Args:
        float_buffer: 数值范围超出 [0, 1]（例如深度、任意物理量）时必须传 True，
            否则 Blender 内部按 8-bit 存储，超出范围的值会被错误地截断/归零。
        non_color: 数据不是颜色（例如深度）时传 True，避免保存/读取时被当成颜色
            套用场景的 view transform（Filmic/AgX 等）做 tone mapping，把原始数值改写掉。
    """
    import numpy as np

    h, w, channels = arr.shape
    img = bpy.data.images.new(
        name, width=w, height=h, alpha=(channels == 4), float_buffer=float_buffer
    )
    if non_color:
        img.colorspace_settings.name = "Non-Color"
    img.pixels.foreach_set(np.ascontiguousarray(arr[::-1, :, :]).ravel())
    return img


def postprocess_render_image(
    image_path: str | Path,
    *,
    final_size: int = 1920,
    padding: int = 32,
    quiet: bool = False,
) -> dict | None:
    """按 alpha 通道内容裁剪渲染图，再等比缩放居中贴回固定大小的透明方图。

    用于把「场景占画面比例差异很大」的渲染图统一成同一尺寸、内容居中的方图，
    避免下游（例如喂给 VLM）看到大片无意义的透明/黑色背景。

    纯 numpy + bpy.data.images 实现，不依赖 PIL（Blender 自带 Python 默认没有装 Pillow），
    因此可以直接在 Blender 脚本内部调用，不需要额外的宿主进程后处理步骤。

    Args:
        image_path: 待处理的 PNG 路径（会原地覆盖）
        final_size: 输出方图边长（像素）
        padding: 内容缩放后与画布边缘保留的像素边距
        quiet: True 时不打印提示信息

    Returns:
        None: 图片全透明（没有任何 alpha>0 的像素），已跳过处理
        dict: {
            "bbox": (left, upper, right, lower),  # 原图上被裁剪保留的区域（像素，左上角原点，右/下开区间）
            "scale": float,                        # 裁剪区域 -> 最终画布的缩放系数
            "paste_xy": (x, y),                    # 缩放后的内容在最终画布中的左上角像素坐标
            "orig_size": (width, height),          # 原始渲染图尺寸
            "final_size": final_size,
        }
        可用于把原图上的像素坐标（例如相机内参 cx/cy）换算到后处理后的图上。
    """
    import numpy as np

    if padding < 0:
        raise ValueError("padding 不能小于 0")
    if final_size <= padding * 2:
        raise ValueError("final_size 必须大于 2 倍 padding")

    image_path = Path(image_path)
    src_img = bpy.data.images.load(str(image_path))
    try:
        orig_w, orig_h = src_img.size
        arr = _bpy_image_to_array(src_img)
    finally:
        bpy.data.images.remove(src_img)

    if arr.shape[2] < 4:
        if not quiet:
            print(f"警告: 图片没有 alpha 通道，跳过后处理: {image_path}")
        return None

    alpha_mask = arr[:, :, 3] > (1.0 / 255.0)
    rows = np.any(alpha_mask, axis=1)
    cols = np.any(alpha_mask, axis=0)
    if not rows.any():
        if not quiet:
            print(f"警告: 图片没有非透明内容，跳过后处理: {image_path}")
        return None

    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]
    upper, lower = int(row_indices[0]), int(row_indices[-1]) + 1
    left, right = int(col_indices[0]), int(col_indices[-1]) + 1

    cropped = arr[upper:lower, left:right, :]
    crop_h, crop_w = cropped.shape[:2]

    target_extent = final_size - 2 * padding
    scale = min(target_extent / crop_w, target_extent / crop_h)
    resized_w = max(1, round(crop_w * scale))
    resized_h = max(1, round(crop_h * scale))

    if (resized_w, resized_h) != (crop_w, crop_h):
        temp_img = _array_to_bpy_image(cropped, name="__postprocess_crop_temp__")
        try:
            temp_img.scale(resized_w, resized_h)
            resized = _bpy_image_to_array(temp_img)
        finally:
            bpy.data.images.remove(temp_img)
    else:
        resized = cropped

    canvas = np.zeros((final_size, final_size, arr.shape[2]), dtype=np.float32)
    paste_x = (final_size - resized_w) // 2
    paste_y = (final_size - resized_h) // 2
    canvas[paste_y : paste_y + resized_h, paste_x : paste_x + resized_w, :] = resized

    out_img = _array_to_bpy_image(canvas, name="__postprocess_output_temp__")
    try:
        out_img.filepath_raw = str(image_path)
        out_img.file_format = "PNG"
        out_img.save()
    finally:
        bpy.data.images.remove(out_img)

    if not quiet:
        print(
            f"已后处理图片: {image_path} "
            f"(裁剪 bbox=({left}, {upper}, {right}, {lower}), "
            f"输出={final_size}x{final_size}, padding={padding})"
        )

    return {
        "bbox": (left, upper, right, lower),
        "scale": scale,
        "paste_xy": (paste_x, paste_y),
        "orig_size": (orig_w, orig_h),
        "final_size": final_size,
    }


def create_floor_from_boundary(
    boundary: list, height: float = 0.0, texture_path: str = None, texture_scale: float = 1.0
):
    """
    根据边界多边形创建地板

    Args:
        boundary: 边界顶点列表 [[x1, y1], [x2, y2], ...]
        height: 地板高度（Z坐标）
        texture_path: 纹理贴图路径（可选）
        texture_scale: 纹理缩放比例，值越大纹理越密集

    Returns:
        floor_object: 地板物体
    """
    # 创建网格
    mesh = bpy.data.meshes.new("Floor_Mesh")
    obj = bpy.data.objects.new("Floor", mesh)

    # 链接到场景
    bpy.context.collection.objects.link(obj)

    # 创建几何体
    bm = bmesh.new()

    # 添加顶点
    verts = []
    for point in boundary:
        v = bm.verts.new((point[0], point[1], height))
        verts.append(v)

    # 创建面
    face = bm.faces.new(verts)

    # 创建 UV 层
    uv_layer = bm.loops.layers.uv.new("UVMap")

    # 计算边界的范围，用于 UV 映射
    min_x = min(p[0] for p in boundary)
    max_x = max(p[0] for p in boundary)
    min_y = min(p[1] for p in boundary)
    max_y = max(p[1] for p in boundary)
    width = max_x - min_x
    depth = max_y - min_y

    # 设置 UV 坐标（基于世界坐标，支持纹理平铺）
    for loop in face.loops:
        uv = loop[uv_layer]
        x, y, z = loop.vert.co
        # UV 坐标基于实际尺寸，texture_scale 控制平铺密度
        uv.uv = ((x - min_x) * texture_scale, (y - min_y) * texture_scale)

    # 更新网格
    bm.to_mesh(mesh)
    bm.free()

    # 设置材质
    mat = bpy.data.materials.new(name="Floor_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    bsdf = nodes.get("Principled BSDF")

    # 处理纹理路径（支持相对路径和绝对路径）
    resolved_texture_path = None
    if texture_path:
        texture_path_obj = Path(texture_path)
        if texture_path_obj.is_absolute():
            resolved_texture_path = texture_path_obj
        else:
            # 相对路径：基于项目根目录解析
            resolved_texture_path = project_root / texture_path
        
        if not resolved_texture_path.exists():
            print(f"警告: 纹理文件不存在: {resolved_texture_path}")
            resolved_texture_path = None

    if resolved_texture_path:
        # 使用纹理贴图
        # 创建图像纹理节点
        tex_image = nodes.new("ShaderNodeTexImage")
        tex_image.image = bpy.data.images.load(str(resolved_texture_path))
        tex_image.image.colorspace_settings.name = "sRGB"
        tex_image.location = (-500, 300)

        # 创建纹理坐标节点
        tex_coord = nodes.new("ShaderNodeTexCoord")
        tex_coord.location = (-900, 300)

        # 创建 Mapping 节点用于控制纹理平铺
        mapping = nodes.new("ShaderNodeMapping")
        mapping.location = (-700, 300)

        # 连接节点
        links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], tex_image.inputs["Vector"])
        links.new(tex_image.outputs["Color"], bsdf.inputs["Base Color"])

        # 设置纹理重复模式
        tex_image.extension = "REPEAT"

        print(f"地板纹理已加载: {resolved_texture_path}")
    else:
        # 无外部贴图时，退化为程序化木地板材质，避免地板纯灰无纹理。
        tex_coord = nodes.new("ShaderNodeTexCoord")
        tex_coord.location = (-1200, 0)

        mapping = nodes.new("ShaderNodeMapping")
        mapping.location = (-1000, 0)
        mapping.inputs["Scale"].default_value[0] = max(texture_scale * 1.2, 0.2)
        mapping.inputs["Scale"].default_value[1] = max(texture_scale * 6.0, 1.0)
        mapping.inputs["Scale"].default_value[2] = 1.0

        wave = nodes.new("ShaderNodeTexWave")
        wave.location = (-780, 100)
        wave.wave_type = "BANDS"
        wave.bands_direction = "Y"
        wave.inputs["Scale"].default_value = 18.0
        wave.inputs["Distortion"].default_value = 2.2
        wave.inputs["Detail"].default_value = 4.0

        noise = nodes.new("ShaderNodeTexNoise")
        noise.location = (-780, -140)
        noise.inputs["Scale"].default_value = 14.0
        noise.inputs["Detail"].default_value = 8.0
        noise.inputs["Roughness"].default_value = 0.55

        mix_rgb = nodes.new("ShaderNodeMixRGB")
        mix_rgb.location = (-520, 20)
        mix_rgb.blend_type = "MULTIPLY"
        mix_rgb.inputs["Fac"].default_value = 0.35

        color_ramp = nodes.new("ShaderNodeValToRGB")
        color_ramp.location = (-260, 20)
        color_ramp.color_ramp.elements[0].position = 0.35
        color_ramp.color_ramp.elements[0].color = (0.34, 0.23, 0.12, 1.0)
        color_ramp.color_ramp.elements[1].position = 0.78
        color_ramp.color_ramp.elements[1].color = (0.68, 0.53, 0.32, 1.0)

        bump = nodes.new("ShaderNodeBump")
        bump.location = (-260, -180)
        bump.inputs["Strength"].default_value = 0.08
        bump.inputs["Distance"].default_value = 0.02

        links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], wave.inputs["Vector"])
        links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
        links.new(wave.outputs["Color"], mix_rgb.inputs["Color1"])
        links.new(noise.outputs["Color"], mix_rgb.inputs["Color2"])
        links.new(mix_rgb.outputs["Color"], color_ramp.inputs["Fac"])
        links.new(mix_rgb.outputs["Color"], bump.inputs["Height"])
        if bsdf:
            links.new(color_ramp.outputs["Color"], bsdf.inputs["Base Color"])
            links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

        if bsdf:
            bsdf.inputs["Roughness"].default_value = 0.62
            specular_input = bsdf.inputs.get("Specular IOR Level") or bsdf.inputs.get("Specular")
            if specular_input:
                specular_input.default_value = 0.35

        print("地板未提供外部贴图，已使用默认程序化木地板材质")

    obj.data.materials.append(mat)

    return obj


def create_box(name: str, center: list, size: list, rotation: list):
    """
    创建一个 Box 物体

    Args:
        name: 物体名称
        center: 中心坐标 [x, y, z]
        size: 尺寸 [width, depth, height]
        rotation: 旋转角度 [rx, ry, rz]（度数）

    Returns:
        box_object: Box 物体
    """
    bpy.ops.mesh.primitive_cube_add()
    box = bpy.context.active_object
    box.name = name

    # 设置尺寸
    box.dimensions = size

    # 设置位置
    box.location = center

    # 设置旋转（转换为弧度）
    box.rotation_euler = (
        math.radians(rotation[0]),
        math.radians(rotation[1]),
        math.radians(rotation[2]),
    )

    # 应用变换
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    return box


def _new_imported_objects(before_object_names: set[str]):
    return [obj for obj in bpy.data.objects if obj.name not in before_object_names]


_ASSET_MESH_CACHE: OrderedDict[tuple[str, str], object] = OrderedDict()


def clear_asset_mesh_cache() -> None:
    """Drop cached template meshes created for geometry-only batch jobs."""
    for mesh in _ASSET_MESH_CACHE.values():
        if getattr(mesh, "users", 0) == 0:
            bpy.data.meshes.remove(mesh)
    _ASSET_MESH_CACHE.clear()


def _normalize_imported_asset_object(asset):
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = asset
    asset.select_set(True)
    bpy.ops.object.transform_apply(scale=True, rotation=True)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")
    asset.select_set(False)


def _clear_object_runtime_state(asset) -> None:
    for key in list(asset.keys()):
        del asset[key]
    if asset.animation_data is not None:
        asset.animation_data_clear()
    asset.parent = None
    for constraint in list(asset.constraints):
        asset.constraints.remove(constraint)


def _apply_asset_transform(
    asset,
    name: str,
    center: list,
    size: list,
    rotation: list,
    jid: str,
    rename_mesh: bool = True,
):
    _clear_object_runtime_state(asset)
    asset.name = name
    if rename_mesh and getattr(asset, "data", None) is not None:
        asset.data.name = f"{name}_mesh"
    asset.scale = (1.0, 1.0, 1.0)
    asset.rotation_euler = (0.0, 0.0, 0.0)
    asset.location = center
    asset.dimensions = size
    asset.rotation_euler = (
        math.radians(rotation[0]),
        math.radians(rotation[1]),
        math.radians(rotation[2]),
    )
    asset["jid"] = jid
    return asset


def _cache_key(dataset: str, jid: str) -> tuple[str, str]:
    return ((dataset or "auto").lower(), str(jid))


def _clone_cached_asset(cache_key: tuple[str, str], name: str, center: list, size: list, rotation: list, jid: str):
    template_mesh = _ASSET_MESH_CACHE.get(cache_key)
    if template_mesh is None:
        return None
    _ASSET_MESH_CACHE.move_to_end(cache_key)
    asset = bpy.data.objects.new(name, template_mesh)
    bpy.context.collection.objects.link(asset)
    return _apply_asset_transform(
        asset,
        name,
        center,
        size,
        rotation,
        jid,
        rename_mesh=False,
    )


def _remember_asset_mesh(
    cache_key: tuple[str, str],
    asset,
    asset_cache_size: int,
    geometry_only_asset_cache: bool,
) -> None:
    if asset_cache_size <= 0 or getattr(asset, "data", None) is None:
        return
    if cache_key in _ASSET_MESH_CACHE:
        _ASSET_MESH_CACHE.move_to_end(cache_key)
        return
    template_mesh = asset.data
    if geometry_only_asset_cache:
        template_mesh.materials.clear()
    template_mesh.name = f"__asset_template__{cache_key[0]}__{cache_key[1]}"
    _ASSET_MESH_CACHE[cache_key] = template_mesh
    while len(_ASSET_MESH_CACHE) > asset_cache_size:
        _old_key, old_mesh = _ASSET_MESH_CACHE.popitem(last=False)
        if getattr(old_mesh, "users", 0) == 0:
            bpy.data.meshes.remove(old_mesh)


def _import_asset_with_mesh_cache(
    cache_key: tuple[str, str],
    import_func,
    jid: str,
    name: str,
    center: list,
    size: list,
    rotation: list,
    asset_cache_size: int,
    geometry_only_asset_cache: bool,
):
    if asset_cache_size > 0:
        cached_asset = _clone_cached_asset(cache_key, name, center, size, rotation, jid)
        if cached_asset is not None:
            return cached_asset
    asset = import_func(jid, name, center, size, rotation)
    if asset is not None:
        _remember_asset_mesh(cache_key, asset, asset_cache_size, geometry_only_asset_cache)
    return asset


def _join_imported_mesh_objects(imported_objects, imported_mesh_objects, asset_path: Path):
    if not imported_mesh_objects:
        print(f"警告: 导入后无 MESH 物体: {asset_path}")
        for obj in imported_objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        return None

    imported_object_names = [obj.name for obj in imported_objects]
    bpy.ops.object.select_all(action="DESELECT")
    for obj in imported_mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = imported_mesh_objects[0]
    if len(imported_mesh_objects) > 1:
        bpy.ops.object.join()
    asset = bpy.context.active_object

    # FBX/GLB 导入常带 Armature/Empty 等非 mesh 根节点；若留下会绕过 object_id 物理约束。
    # join 会删除被合并的 mesh Object，不能再访问原 Object 引用，只能按名字查仍存在的对象。
    asset_name = asset.name if asset is not None else None
    for obj_name in imported_object_names:
        if obj_name == asset_name:
            continue
        obj = bpy.data.objects.get(obj_name)
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)
    return asset


def import_3d_future_asset(
    jid: str, name: str, center: list, size: list, rotation: list
):
    """
    导入 3D-FUTURE 模型并设置变换

    Args:
        jid: 3D-FUTURE 资产 ID
        name: 物体名称
        center: 中心坐标 [x, y, z]
        size: 尺寸 [width, depth, height]
        rotation: 旋转角度 [rx, ry, rz]（度数）

    Returns:
        asset_object: 导入的资产物体，如果失败返回 None
    """
    asset_path = THREED_FUTURE_MODEL_DIR / jid / "raw_model.obj"

    if not asset_path.exists():
        print(f"警告: 资产不存在: {asset_path}")
        return None

    before_object_names = {obj.name for obj in bpy.data.objects}

    # 导入 OBJ 模型
    bpy.ops.wm.obj_import(filepath=str(asset_path))

    imported_objects = _new_imported_objects(before_object_names)
    imported_mesh_objects = [obj for obj in imported_objects if obj.type == "MESH"]
    asset = _join_imported_mesh_objects(imported_objects, imported_mesh_objects, asset_path)
    if asset is None:
        return None

    _normalize_imported_asset_object(asset)
    
    # 设置材质透射属性 (Transmission)
    if asset.data.materials:
        for mat in asset.data.materials:
            if mat and mat.use_nodes:
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if bsdf:
                    # 尝试设置 Transmission (兼容不同 Blender 版本)
                    # Blender 4.0+ 使用 "Transmission Weight"，旧版本使用 "Transmission"
                    transmission_input = bsdf.inputs.get("Transmission Weight") or bsdf.inputs.get("Transmission")
                    if transmission_input:
                        transmission_input.default_value = 0.0

    return _apply_asset_transform(asset, name, center, size, rotation, jid)


def import_imaginarium_asset(
    jid: str, name: str, center: list, size: list, rotation: list
):
    """
    导入 Imaginarium 模型（FBX 格式）并设置变换

    Args:
        jid: Imaginarium 资产 ID（通常是物体名称）
        name: 物体名称
        center: 中心坐标 [x, y, z]
        size: 尺寸 [width, depth, height]
        rotation: 旋转角度 [rx, ry, rz]（度数）

    Returns:
        asset_object: 导入的资产物体，如果失败返回 None
    """
    # Imaginarium 资产路径格式: jid/jid.fbx
    asset_path = IMAGINARIUM_ASSETS_DIR / jid / f"{jid}.fbx"

    if not asset_path.exists():
        print(f"警告: Imaginarium 资产不存在: {asset_path}")
        return None

    before_object_names = {obj.name for obj in bpy.data.objects}

    # 导入 FBX 模型
    bpy.ops.import_scene.fbx(filepath=str(asset_path))

    imported_objects = _new_imported_objects(before_object_names)
    imported_mesh_objects = [obj for obj in imported_objects if obj.type == "MESH"]
    asset = _join_imported_mesh_objects(imported_objects, imported_mesh_objects, asset_path)
    if asset is None:
        return None

    _normalize_imported_asset_object(asset)
    return _apply_asset_transform(asset, name, center, size, rotation, jid)


def import_asset(
    jid: str,
    name: str,
    center: list,
    size: list,
    rotation: list,
    dataset: str = "auto",
    asset_cache_size: int = 0,
    geometry_only_asset_cache: bool = False,
):
    """
    根据 dataset 类型导入资产并设置变换。

    Args:
        jid: 资产 ID
        name: 物体名称
        center/size/rotation: 目标变换
        dataset:
            - "auto": 按文件存在性依次尝试 3D-FUTURE(OBJ) -> Imaginarium(FBX)
            - "3dfront": 仅尝试 3D-FUTURE(OBJ)，对应 ReSpace
            - "imaginarium": 仅尝试 Imaginarium(FBX)

    Returns:
        asset_object: 导入的资产物体，如果失败返回 None
    """
    ds = (dataset or "auto").lower()
    if ds not in {"auto", "3dfront", "imaginarium"}:
        ds = "auto"

    if ds == "3dfront":
        return _import_asset_with_mesh_cache(
            _cache_key(ds, jid),
            import_3d_future_asset,
            jid,
            name,
            center,
            size,
            rotation,
            asset_cache_size,
            geometry_only_asset_cache,
        )
    if ds == "imaginarium":
        return _import_asset_with_mesh_cache(
            _cache_key(ds, jid),
            import_imaginarium_asset,
            jid,
            name,
            center,
            size,
            rotation,
            asset_cache_size,
            geometry_only_asset_cache,
        )
    # auto: 按文件存在性依次尝试
    future_obj = THREED_FUTURE_MODEL_DIR / jid / "raw_model.obj"
    imag_fbx = IMAGINARIUM_ASSETS_DIR / jid / f"{jid}.fbx"

    if future_obj.exists():
        asset = _import_asset_with_mesh_cache(
            _cache_key("3dfront", jid),
            import_3d_future_asset,
            jid,
            name,
            center,
            size,
            rotation,
            asset_cache_size,
            geometry_only_asset_cache,
        )
        if asset is not None:
            return asset
    if imag_fbx.exists():
        asset = _import_asset_with_mesh_cache(
            _cache_key("imaginarium", jid),
            import_imaginarium_asset,
            jid,
            name,
            center,
            size,
            rotation,
            asset_cache_size,
            geometry_only_asset_cache,
        )
        if asset is not None:
            return asset

    return None


def build_instruction_map(layout_data: dict) -> dict:
    """
    从 layout_data 中提取每个物体的 instruction。

    目前 instruction 通常位于 layout_data["successful_placements"][*]["instruction"]，
    并通过 object_id 与最终场景中的 object.id 对齐。
    """
    instruction_map = {}
    for placement in layout_data.get("successful_placements", []) or []:
        obj_id = placement.get("object_id")
        instruction = placement.get("instruction")
        if obj_id is None or instruction is None:
            continue
        instruction_map[int(obj_id)] = str(instruction)
    return instruction_map


def create_world_axes(
    length: float = 1.0,
    origin: tuple = (0.0, 0.0, 0.0),
    shaft_radius: float = 0.015,
    head_radius_ratio: float = 3.0,
    head_length_ratio: float = 0.12,
    with_labels: bool = True,
):
    """
    创建一组 RGB 坐标轴指示器 (X 红 / Y 绿 / Z 蓝)，每根轴一个 cylinder + cone。

    坐标轴的方向始终对齐世界 X/Y/Z；origin 仅决定整组 axes 在世界中的悬挂位置，
    用于把 axes 抬高到家具上方避免被遮挡（默认仍在世界原点 (0,0,0)）。

    Args:
        length: 每根轴的长度（米），从原点延伸到 (length, 0, 0) / (0, length, 0) / (0, 0, length)
        origin: 整组坐标轴的世界平移量 (x, y, z)，默认 (0, 0, 0)
        shaft_radius: 轴杆半径
        head_radius_ratio: 箭头底圆半径 = shaft_radius * head_radius_ratio
        head_length_ratio: 箭头长度 = length * head_length_ratio
        with_labels: 是否在每根轴尖端附近放置 "+X / +Y / +Z" 文本标签

    返回值: 创建的所有 axis 物体名称列表。
    """
    head_length = length * head_length_ratio
    head_radius = shaft_radius * head_radius_ratio
    shaft_length = length - head_length

    # axis_name -> (color RGBA, 让 +Z 朝向目标方向所需的 euler 旋转)
    axis_specs = {
        "X": ((1.0, 0.1, 0.1, 1.0), (0.0, math.radians(90.0), 0.0)),
        "Y": ((0.1, 0.85, 0.1, 1.0), (math.radians(-90.0), 0.0, 0.0)),
        "Z": ((0.1, 0.3, 1.0, 1.0), (0.0, 0.0, 0.0)),
    }

    created_names = []

    for axis_name, (color, rot_euler) in axis_specs.items():
        # 1. shaft: 默认沿 +Z, 中心在 (0,0,shaft_length/2), 然后整体旋转到目标轴
        bpy.ops.mesh.primitive_cylinder_add(
            radius=shaft_radius,
            depth=shaft_length,
            location=(0.0, 0.0, shaft_length / 2.0),
        )
        shaft = bpy.context.active_object
        shaft.name = f"WorldAxis_{axis_name}_shaft"

        # 2. head: cone, 默认 +Z, 中心在 shaft 顶端 + head_length/2
        bpy.ops.mesh.primitive_cone_add(
            radius1=head_radius,
            radius2=0.0,
            depth=head_length,
            location=(0.0, 0.0, shaft_length + head_length / 2.0),
        )
        head = bpy.context.active_object
        head.name = f"WorldAxis_{axis_name}_head"

        # 3. 合并 shaft + head 为单个物体
        bpy.ops.object.select_all(action="DESELECT")
        shaft.select_set(True)
        head.select_set(True)
        bpy.context.view_layer.objects.active = shaft
        bpy.ops.object.join()
        axis_obj = bpy.context.active_object
        axis_obj.name = f"WorldAxis_{axis_name}"

        # 4. 把整体（默认沿 +Z）旋转到目标轴方向，原点保持在 (0,0,0)
        axis_obj.rotation_euler = rot_euler

        # 5. emission 材质，颜色亮，避免被场景灯光弱化
        mat = bpy.data.materials.new(name=f"WorldAxis_{axis_name}_mat")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        emission = nodes.new(type="ShaderNodeEmission")
        emission.inputs["Color"].default_value = color
        emission.inputs["Strength"].default_value = 4.0
        output = nodes.new(type="ShaderNodeOutputMaterial")
        links.new(emission.outputs["Emission"], output.inputs["Surface"])
        axis_obj.data.materials.clear()
        axis_obj.data.materials.append(mat)

        created_names.append(axis_obj.name)

        # 6. 可选：在轴尖端外侧放一个 "+X / +Y / +Z" 文本
        if with_labels:
            # +X / +Y 让标签中心与箭头尖端共点（offset=length），并把字号缩小，
            # 这样 CENTER 对齐时整段文字都覆盖在箭头尖端附近，视觉上紧贴箭头；
            # +Z 仍然留较大间距，因为它指向天花板，太近容易和顶灯/坐标轴自身重合。
            if axis_name == "X":
                label_loc = (length, 0.0, 0.0)
            elif axis_name == "Y":
                label_loc = (0.0, length, 0.0)
            else:
                label_loc = (0.0, 0.0, length + head_length * 1.5)

            bpy.ops.object.text_add(location=label_loc)
            label = bpy.context.active_object
            label.name = f"WorldAxis_{axis_name}_label"
            label.data.body = f"+{axis_name}"
            label.data.align_x = "CENTER"
            # X/Y 用 BOTTOM 对齐：字底部对齐箭头水平线，整段字浮在箭头上方，
            # 视觉上字与箭头"平齐"而不再半边低于箭头；+Z 仍用 CENTER 保持对称。
            label.data.align_y = "BOTTOM" if axis_name in ("X", "Y") else "CENTER"
            # X/Y 字号缩小到 0.12*length，避免大字本身把视觉中心拉远
            xy_size = max(length * 0.12, 0.06)
            z_size = max(length * 0.18, 0.08)
            label.data.size = xy_size if axis_name in ("X", "Y") else z_size
            label.data.extrude = 0.005

            # 让标签始终朝向相机（如果存在）
            camera = bpy.context.scene.camera
            if camera is not None:
                constraint = label.constraints.new(type="TRACK_TO")
                constraint.target = camera
                constraint.track_axis = "TRACK_Z"
                constraint.up_axis = "UP_Y"

            label.data.materials.clear()
            label.data.materials.append(mat)

            created_names.append(label.name)

    ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
    if (ox, oy, oz) != (0.0, 0.0, 0.0):
        for name in created_names:
            obj = bpy.data.objects.get(name)
            if obj is None:
                continue
            obj.location.x += ox
            obj.location.y += oy
            obj.location.z += oz
        bpy.context.view_layer.update()

    print(
        f"[Axes] 已创建世界坐标轴 (length={length}m, origin=({ox:.2f}, {oy:.2f}, {oz:.2f}), "
        f"labels={'on' if with_labels else 'off'}): {created_names}"
    )
    return created_names


class SceneBuildResult(NamedTuple):
    """build_scene_from_layout 的返回结果，供可视化与物理修正复用。"""

    layout_data: dict
    layout_path: Path
    scene_data: dict
    scene_id: str
    floor: Any
    objects: list
    boundary: list
    asset_objects: list
    stage_started_at: float


def build_scene_from_layout(
    layout_path: str,
    *,
    dataset: str = "auto",
    floor_texture: str = DEFAULT_FLOOR_TEXTURE,
    texture_scale: float = 0.5,
    asset_cache_size: int = 0,
    geometry_only_asset_cache: bool = False,
    timings: dict[str, float] | None = None,
    stage_started_at: float | None = None,
) -> SceneBuildResult:
    """从 layout JSON 构建 Blender 场景：清空场景、按 boundary 建地板、导入资产。

    这是可视化与物理修正共用的"建场景"步骤，不包含灯光/相机/渲染，
    因此既可被 visualize_layout 调用，也可被仅做物理修正的入口复用。
    """
    if timings is None:
        timings = {}
    if stage_started_at is None:
        stage_started_at = time.perf_counter()

    layout_path = Path(layout_path)
    if not layout_path.exists():
        raise FileNotFoundError(f"Layout 文件不存在: {layout_path}")

    with open(layout_path, "r", encoding="utf-8") as f:
        layout_data = json.load(f)

    instruction_map = build_instruction_map(layout_data)
    stage_started_at = _finish_profile_stage(timings, "load_json", stage_started_at)

    # 获取最终场景数据
    if scene_from_external_dict is not None:
        scene_data = scene_from_external_dict(layout_data).to_dict(object_field="objects")
    elif "final_scene" in layout_data:
        # generate 模式产生的格式（带有 placement_history 等额外信息）
        scene_data = layout_data["final_scene"]
    elif "objects" in layout_data:
        # 统一格式：顶层直接包含场景数据
        scene_data = {
            "scene_id": layout_data.get("scene_id", "Unknown"),
            "scene_type": layout_data.get("scene_type", ""),
            "boundary": layout_data.get("boundary", []),
            "objects": layout_data.get("objects", []),
        }
    else:
        raise ValueError("无法识别的 Layout 文件格式（缺少 'final_scene' 或 'objects' 字段）")
    scene_id = scene_data.get("scene_id", "Unknown")
    stage_started_at = _finish_profile_stage(timings, "normalize_scene_data", stage_started_at)

    print(f"\n{'='*60}")
    print(f"构建场景: {scene_id}")
    print(f"{'='*60}")

    # 清空场景
    clear_scene()
    stage_started_at = _finish_profile_stage(timings, "clear_scene", stage_started_at)

    # 设置变换轴心点为包围盒中心
    bpy.context.scene.tool_settings.transform_pivot_point = "BOUNDING_BOX_CENTER"

    # 创建地板
    boundary = scene_data.get("boundary", [])
    floor = None
    if boundary:
        floor = create_floor_from_boundary(
            boundary,
            texture_path=floor_texture,
            texture_scale=texture_scale,
        )
        print(f"创建地板: {len(boundary)} 个顶点")
    stage_started_at = _finish_profile_stage(timings, "create_floor", stage_started_at)

    # 创建物体
    objects = scene_data.get("objects", [])

    print(f"物体数量: {len(objects)}")
    print("-" * 60)

    success_count = 0
    asset_objects = []
    for obj_data in objects:
        obj_id = obj_data["id"]
        short_desc = obj_data.get("short_desc", f"Object_{obj_id}")
        instruction = obj_data.get("instruction")
        if instruction is None:
            # 生成场景的 objects 往往不带 instruction，尝试从 successful_placements 补齐
            try:
                instruction = instruction_map.get(int(obj_id), "")
            except Exception:
                instruction = ""
        center = obj_data["center"]
        size = obj_data["size"]
        rotation = obj_data.get("rotation", [0, 0, 0])
        jid = obj_data.get("jid", "")

        # 生成物体名称
        obj_name = f"{obj_id:02d}_{short_desc.replace(' ', '_')}"
        if not jid:
            raise ValueError(f"Object {obj_name} 缺少 jid，无法在仅资产模式下可视化")

        # 按 dataset 导入资产（auto 支持同一场景混用 3D-FUTURE/Imaginarium）
        obj = import_asset(
            jid,
            obj_name,
            center,
            size,
            rotation,
            dataset=dataset,
            asset_cache_size=asset_cache_size,
            geometry_only_asset_cache=geometry_only_asset_cache,
        )
        if obj is None:
            raise RuntimeError(
                f"Object {obj_name} 的资产导入失败 (jid={jid}, dataset={dataset})"
            )

        # 存储描述信息
        obj["short_desc"] = short_desc
        obj["object_id"] = int(obj_id)
        if jid:
            obj["jid"] = jid
        # 存储 instruction（用于回溯每个物体的放置指令）
        if instruction:
            obj["instruction"] = str(instruction)

        asset_objects.append(obj)
        success_count += 1

    print("-" * 60)
    print(f"成功创建 {success_count}/{len(objects)} 个物体")
    stage_started_at = _finish_profile_stage(timings, "import_assets", stage_started_at)

    return SceneBuildResult(
        layout_data=layout_data,
        layout_path=layout_path,
        scene_data=scene_data,
        scene_id=scene_id,
        floor=floor,
        objects=objects,
        boundary=boundary,
        asset_objects=asset_objects,
        stage_started_at=stage_started_at,
    )


# ==================== 包围盒/高亮工具 ====================


def make_principled_material(name, color, alpha=1.0, roughness=0.55):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    if alpha < 1.0:
        mat.blend_method = "BLEND"
        mat.show_transparent_back = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        if "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = color[:3] + (alpha,)
        if "Alpha" in bsdf.inputs:
            bsdf.inputs["Alpha"].default_value = alpha
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = roughness
    return mat


def make_emission_material(name, color, strength=1.5):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = strength
    output = nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def add_cylinder_edge(start, end, material, radius=0.012, name="NewObject_BBox_Edge"):
    from mathutils import Vector

    start = Vector(start)
    end = Vector(end)
    direction = end - start
    length = direction.length
    if length <= 1e-6:
        return None
    midpoint = (start + end) * 0.5
    bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=radius, depth=length, location=midpoint)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
    obj.data.materials.append(material)
    return obj


def get_bbox_corners(center, rotation_deg, size):
    from mathutils import Euler, Vector

    half = [s * 0.5 for s in size]
    local = [
        Vector((-half[0], -half[1], -half[2])),
        Vector((half[0], -half[1], -half[2])),
        Vector((half[0], half[1], -half[2])),
        Vector((-half[0], half[1], -half[2])),
        Vector((-half[0], -half[1], half[2])),
        Vector((half[0], -half[1], half[2])),
        Vector((half[0], half[1], half[2])),
        Vector((-half[0], half[1], half[2])),
    ]
    rot = Euler(tuple(math.radians(v) for v in rotation_deg), "XYZ").to_matrix()
    ctr = Vector(center)
    return [ctr + rot @ p for p in local]


_BBOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def _join_mesh_objects(objs, name):
    objs = [o for o in objs if o is not None]
    if not objs:
        return None
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objs:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1:
        bpy.ops.object.join()
    merged = bpy.context.view_layer.objects.active
    merged.name = name
    if merged.data:
        merged.data.name = f"{name}_Mesh"
    return merged


def add_solid_bbox(center, rotation_deg, size):
    """在新物体周围画一圈实心发光蓝色包围盒边框（醒目，适合远景/缩略图）。"""
    corners = get_bbox_corners(center, rotation_deg, size)
    mat = make_emission_material("NewObject_BlueBox_Material", (0.10, 0.55, 1.0, 1.0), strength=2.4)
    edges = [
        add_cylinder_edge(corners[a], corners[b], mat, radius=0.012, name="NewObject_BBox_Edge")
        for a, b in _BBOX_EDGES
    ]
    return _join_mesh_objects(edges, name="NewObject_BBox")


def add_transparent_bbox(center, rotation_deg, size):
    """在新物体周围叠加一层半透明蓝色包围盒（细边框 + 透明面合并为一个物体），适合近景细节图。"""
    from mathutils import Euler

    edge_mat = make_emission_material(
        "NewObject_TransparentBBox_Edge_Material", (0.35, 0.95, 1.0, 1.0), strength=1.2
    )
    corners = get_bbox_corners(center, rotation_deg, size)
    edges = [
        add_cylinder_edge(corners[a], corners[b], edge_mat, radius=0.006, name="NewObject_TransparentBBox_Edge")
        for a, b in _BBOX_EDGES
    ]

    face_mat = make_principled_material(
        "NewObject_TransparentBBox_Material", (0.05, 0.55, 1.0, 0.18), alpha=0.18, roughness=0.35
    )
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=center)
    face_obj = bpy.context.object
    face_obj.name = "NewObject_TransparentBBox_Face"
    face_obj.data.name = "NewObject_TransparentBBox_Face_Mesh"
    face_obj.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    face_obj.rotation_euler = Euler(tuple(math.radians(v) for v in rotation_deg), "XYZ")
    face_obj.display_type = "TEXTURED"
    face_obj.show_transparent = True
    face_obj.data.materials.append(face_mat)

    return _join_mesh_objects(edges + [face_obj], name="NewObject_TransparentBBox")


def highlight_new_object(asset_objects, new_object_id, bbox_style="transparent"):
    """在新物体周围加发光包围盒，其余物体保留各自原始材质/纹理。返回被高亮的物体。"""
    new_obj = next((obj for obj in asset_objects if int(obj["object_id"]) == int(new_object_id)), None)
    if new_obj is None:
        available = [int(obj["object_id"]) for obj in asset_objects]
        raise ValueError(f"未找到 new_object_id={new_object_id} 对应的物体，可选 id: {sorted(available)}")

    center = tuple(new_obj.location)
    rotation_deg = tuple(math.degrees(v) for v in new_obj.rotation_euler)
    size = tuple(new_obj.dimensions)

    if bbox_style == "transparent":
        add_transparent_bbox(center, rotation_deg, size)
    else:
        add_solid_bbox(center, rotation_deg, size)

    return new_obj


def visualize_layout(
    layout_path: str,
    dataset: str = "auto",
    floor_texture: str = DEFAULT_FLOOR_TEXTURE,
    texture_scale: float = 0.5,
    show_axes: bool = True,
    axis_length: float = 1.0,
    axes_origin: tuple = (0.0, 0.0, 2.8),
    asset_cache_size: int = 0,
    geometry_only_asset_cache: bool = False,
    render_image_path: str | None = None,
    output_blend: str | None = None,
    pack_assets: bool = True,
    highlight_object_id: int | None = None,
    highlight_last: bool = False,
    bbox_style: str = "transparent",
    render_resolution: tuple[int, int] = (1920, 1920),
    render_engine: str = "BLENDER_EEVEE_NEXT",
    cycles_samples: int = 512,
    cycles_gpu_backend: str = "AUTO",
    use_gpu: bool = True,
    eevee_samples: int = 64,
    use_raytracing: bool = True,
    use_eevee_ao: bool = True,
    profile_render: bool = False,
    postprocess_render: bool = False,
    postprocess_final_size: int = 1920,
    postprocess_padding: int = 32,
):
    """
    可视化 Layout / Converted Scene JSON 文件（使用真实资产渲染场景）

    Args:
        layout_path: Layout 或 Converted Scene JSON 文件路径
        dataset: 资产数据集类型: "auto" / "3dfront" / "imaginarium"
        floor_texture: 地板纹理贴图路径（可选）
        texture_scale: 纹理缩放比例，值越大纹理越密集
        show_axes: 是否放置 RGB 坐标轴指示 (X 红 / Y 绿 / Z 蓝)
        axis_length: 坐标轴长度（米）
        axes_origin: 整组坐标轴的世界平移量；默认 (0,0,2.8) 抬到天花板高度避免被家具遮挡
        geometry_only_asset_cache: True 时缓存 mesh 会清空材质；默认保留源材质和贴图
        render_image_path: 若指定，在可视化结束后渲染单张静态图片
        output_blend: 若指定，将构建好的 Blender 场景另存为 .blend 文件
        pack_assets: 保存 .blend 时是否打包外部贴图和资产
        highlight_object_id: 若指定，高亮凸显该物体 ID 的包围盒
        highlight_last: 若为 True 且未指定 highlight_object_id，自动高亮最后一个物体
        bbox_style: 高亮包围盒样式 ("transparent" 或 "blue")
        render_resolution: 渲染输出分辨率 (width, height)
        postprocess_render: 是否在渲染后按 alpha 内容裁剪并等比缩放居中贴回固定大小方图
        postprocess_final_size: 后处理输出方图边长（像素）
        postprocess_padding: 后处理内容与画布边缘保留的像素边距
    """
    timings: dict[str, float] = {}
    total_started_at = time.perf_counter()
    stage_started_at = total_started_at

    # 构建场景（清空、建地板、导入资产）——与物理修正入口共用
    build = build_scene_from_layout(
        layout_path,
        dataset=dataset,
        floor_texture=floor_texture,
        texture_scale=texture_scale,
        asset_cache_size=asset_cache_size,
        geometry_only_asset_cache=geometry_only_asset_cache,
        timings=timings,
        stage_started_at=stage_started_at,
    )
    layout_path = build.layout_path
    scene_id = build.scene_id
    stage_started_at = build.stage_started_at

    target_highlight_id = highlight_object_id
    if target_highlight_id is None and highlight_last and build.scene_data.get("objects"):
        target_highlight_id = max(int(obj_data["id"]) for obj_data in build.scene_data["objects"])
    if target_highlight_id is not None:
        highlighted = highlight_new_object(build.asset_objects, target_highlight_id, bbox_style=bbox_style)
        print(f"高亮新物体: id={highlighted['object_id']} short_desc={highlighted.get('short_desc')}")

    print("应用自动灯光、相机和渲染设置...")
    setup_indoor_lighting_and_render(
        cycles_samples=cycles_samples,
        render_engine=render_engine,
        resolution_x=render_resolution[0],
        resolution_y=render_resolution[1],
        use_gpu=use_gpu,
        cycles_gpu_backend=cycles_gpu_backend,
        eevee_samples=eevee_samples,
        use_raytracing=use_raytracing,
        use_eevee_ao=use_eevee_ao,
    )
    stage_started_at = _finish_profile_stage(timings, "setup_render", stage_started_at)

    # 可选：放一组坐标轴指示器 (X 红 / Y 绿 / Z 蓝)，默认抬到天花板高度避免被家具挡住
    if show_axes:
        create_world_axes(length=axis_length, origin=tuple(axes_origin))
    stage_started_at = _finish_profile_stage(timings, "create_axes", stage_started_at)

    if output_blend:
        save_blend_file(output_blend, pack_assets=pack_assets)

    if render_image_path:
        render_image_file(render_image_path)
        if postprocess_render:
            postprocess_render_image(
                render_image_path,
                final_size=postprocess_final_size,
                padding=postprocess_padding,
            )
    stage_started_at = _finish_profile_stage(timings, "render_image", stage_started_at)

    print(f"\n{'='*60}")
    print("可视化完成!")
    print(f"{'='*60}")
    timings["total"] = time.perf_counter() - total_started_at
    if profile_render:
        _emit_profile(
            scene_id=str(scene_id),
            layout_path=layout_path,
            render_image_path=render_image_path,
            timings=timings,
        )


# ==================== 主函数 ====================


def main():
    """主函数：解析命令行参数并执行可视化"""
    # 获取 -- 之后的参数（Blender 命令行参数）
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="使用 Blender 可视化 Layout JSON")
    parser.add_argument("--layout", "-l", type=str, default=None, help="Layout JSON 文件路径")
    parser.add_argument(
        "--batch-manifest",
        type=str,
        default=None,
        help="批量渲染 manifest JSON，包含 entries: [{layout, render_image}]",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="auto",
        choices=["auto", "3dfront", "imaginarium"],
        help=(
            "资产数据集类型: auto(自动识别/可混用) / 3dfront(OBJ) / "
            "imaginarium(FBX)"
        ),
    )
    parser.add_argument(
        "--floor-texture",
        "-f",
        type=str,
        default=DEFAULT_FLOOR_TEXTURE,
        help=f"地板纹理贴图路径（默认: {DEFAULT_FLOOR_TEXTURE}）",
    )
    parser.add_argument(
        "--no-floor-external-texture",
        action="store_true",
        help="不使用外部地板贴图（与未找到贴图时相同，使用程序化木地板）",
    )
    parser.add_argument(
        "--texture-scale", "-s", type=float, default=0.5, help="纹理缩放比例"
    )
    parser.add_argument(
        "--no-axes",
        action="store_true",
        help="不在世界原点显示 RGB 坐标轴 (默认会显示, X 红 / Y 绿 / Z 蓝)",
    )
    parser.add_argument(
        "--axis-length", type=float, default=1.0, help="世界坐标轴长度（米）"
    )
    parser.add_argument(
        "--axes-origin",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=[0.0, 0.0, 2.8],
        help="整组坐标轴在世界中的平移量，默认 (0 0 2.8) 抬到天花板高度避免被家具遮挡",
    )
    parser.add_argument(
        "--render-image",
        type=str,
        default=None,
        help="渲染单张静态图片到指定路径",
    )
    parser.add_argument(
        "--output-blend",
        "-o",
        type=str,
        default=None,
        help="另存 .blend 文件到指定路径",
    )
    parser.add_argument(
        "--no-pack-assets",
        action="store_true",
        help="保存 .blend 时不打包外部贴图和资产（默认会打包）",
    )
    parser.add_argument(
        "--highlight-object-id",
        "--new-object-id",
        dest="highlight_object_id",
        type=int,
        default=None,
        help="要高亮凸显的物体 id（周围加发光包围盒）",
    )
    parser.add_argument(
        "--highlight-last",
        action="store_true",
        help="自动高亮凸显自回归顺序中最后一个摆放的物体",
    )
    parser.add_argument(
        "--bbox-style",
        choices=["blue", "transparent"],
        default="transparent",
        help="高亮包围盒样式：transparent(默认，半透明罩+细边框) 或 blue(实心发光线框)",
    )
    parser.add_argument(
        "--render-resolution",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=[1920, 1920],
        help="静态图渲染分辨率，默认 1920 1920",
    )
    parser.add_argument(
        "--render-engine",
        type=str,
        default="BLENDER_EEVEE_NEXT",
        choices=["BLENDER_EEVEE_NEXT", "EEVEE", "CYCLES"],
        help="渲染引擎；CYCLES 会尝试使用 GPU",
    )
    parser.add_argument("--cycles-samples", type=int, default=512, help="Cycles 采样数")
    parser.add_argument(
        "--cycles-gpu-backend",
        type=str,
        default="AUTO",
        choices=["AUTO", "CUDA", "OPTIX", "HIP", "ONEAPI", "METAL"],
        help="Cycles GPU 后端，默认 AUTO；H20 上可尝试 CUDA",
    )
    parser.add_argument("--no-gpu", action="store_true", help="关闭 Cycles GPU 探测，强制 CPU")
    parser.add_argument(
        "--asset-cache-size",
        type=int,
        default=0,
        help="缓存导入资产 mesh 的数量；批量渲染时建议开启，如 512",
    )
    parser.add_argument(
        "--geometry-only-asset-cache",
        action="store_true",
        help="资产缓存只保留 geometry 并清空材质；默认保留源材质和贴图",
    )
    parser.add_argument("--eevee-samples", type=int, default=64, help="EEVEE 静态图采样数")
    parser.add_argument("--no-raytracing", action="store_true", help="关闭 EEVEE raytracing")
    parser.add_argument("--no-eevee-ao", action="store_true", help="关闭 EEVEE 环境光遮蔽")
    parser.add_argument(
        "--profile-render",
        action="store_true",
        help="输出当前 scene 的阶段耗时 JSON，便于定位导入/设置/渲染瓶颈",
    )
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="渲染后按 alpha 内容裁剪并等比缩放居中贴回固定大小方图（仅在 --render-image 时生效）",
    )
    parser.add_argument(
        "--postprocess-final-size",
        type=int,
        default=1920,
        help="后处理输出方图边长（像素），默认 1920",
    )
    parser.add_argument(
        "--postprocess-padding",
        type=int,
        default=32,
        help="后处理内容与画布边缘保留的像素边距，默认 32",
    )

    args = parser.parse_args(argv)

    floor_texture = None if args.no_floor_external_texture else args.floor_texture

    common_kwargs = {
        "dataset": args.dataset,
        "floor_texture": floor_texture,
        "texture_scale": args.texture_scale,
        "show_axes": not args.no_axes,
        "axis_length": args.axis_length,
        "axes_origin": tuple(args.axes_origin),
        "asset_cache_size": args.asset_cache_size,
        "geometry_only_asset_cache": args.geometry_only_asset_cache,
        "output_blend": args.output_blend,
        "pack_assets": not args.no_pack_assets,
        "highlight_object_id": args.highlight_object_id,
        "highlight_last": args.highlight_last,
        "bbox_style": args.bbox_style,
        "render_resolution": tuple(args.render_resolution),
        "render_engine": args.render_engine,
        "cycles_samples": args.cycles_samples,
        "cycles_gpu_backend": args.cycles_gpu_backend,
        "use_gpu": not args.no_gpu,
        "eevee_samples": args.eevee_samples,
        "use_raytracing": not args.no_raytracing,
        "use_eevee_ao": not args.no_eevee_ao,
        "profile_render": args.profile_render,
        "postprocess_render": args.postprocess,
        "postprocess_final_size": args.postprocess_final_size,
        "postprocess_padding": args.postprocess_padding,
    }

    if args.batch_manifest:
        with open(args.batch_manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        entries = manifest["entries"]
        for entry in entries:
            render_image = entry["render_image"]
            visualize_layout(
                layout_path=entry["layout"],
                render_image_path=render_image,
                **common_kwargs,
            )
            print(f"{DONE_PREFIX}{render_image}", flush=True)
        return

    if args.layout is None:
        raise ValueError("必须指定 --layout 或 --batch-manifest")

    visualize_layout(
        layout_path=args.layout,
        render_image_path=args.render_image,
        **common_kwargs,
    )



if __name__ == "__main__":
    # 检查是否从命令行运行
    if "--" in sys.argv:
        main()
    else:
        # 在 Blender 中作为脚本直接运行时，使用下面的参数
        LAYOUT_PATH = "output/agent/generated_scene/generated_scene_layout_repaired.json"
        RENDER_IMAGE_PATH = "output/visualizations/generated_scene.png"
        DATASET = "auto"  # auto / 3dfront / imaginarium
        FLOOR_TEXTURE = DEFAULT_FLOOR_TEXTURE  # 地板纹理贴图路径；None: 使用程序化木地板
        TEXTURE_SCALE = 0.5  # 纹理缩放比例，值越大纹理越密集

        visualize_layout(
            layout_path=LAYOUT_PATH,
            render_image_path=RENDER_IMAGE_PATH,
            dataset=DATASET,
            floor_texture=FLOOR_TEXTURE,
            texture_scale=TEXTURE_SCALE,
        )

# run
# blender -P scripts/visualize_layout.py