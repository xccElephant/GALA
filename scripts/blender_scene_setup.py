"""
Blender 场景自动设置工具：
- 根据场景包围盒自动创建室内灯光
- 自动放置可全览场景的相机
- 统一应用渲染参数
"""

import math

import bpy  # type: ignore
from mathutils import Vector


def get_scene_bounds():
    """计算所有可见网格对象的整体包围盒。"""
    min_co = Vector((float("inf"),) * 3)
    max_co = Vector((float("-inf"),) * 3)
    found = False

    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or not obj.visible_get():
            continue
        for corner in obj.bound_box:
            world_co = obj.matrix_world @ Vector(corner)
            min_co = Vector(map(min, zip(min_co, world_co)))
            max_co = Vector(map(max, zip(max_co, world_co)))
            found = True

    if not found:
        return Vector((-3, -3, 0)), Vector((3, 3, 2.8))

    return min_co, max_co


def remove_existing_setup_objects():
    """清除场景中所有的相机和灯光物体及其数据块。"""
    to_delete = [obj for obj in bpy.data.objects if obj.type in ("CAMERA", "LIGHT")]
    for obj in to_delete:
        bpy.data.objects.remove(obj, do_unlink=True)

    for cam_data in list(bpy.data.cameras):
        bpy.data.cameras.remove(cam_data)

    for light_data in list(bpy.data.lights):
        bpy.data.lights.remove(light_data)


def get_floor_center(default_min, default_max):
    """优先使用地板 mesh 的包围盒中心作为相机视角中心点。"""
    floor = bpy.data.objects.get("Floor")
    if floor is not None and floor.type == "MESH" and floor.visible_get():
        corners = [floor.matrix_world @ Vector(corner) for corner in floor.bound_box]
        min_co = Vector(
            (
                min(c.x for c in corners),
                min(c.y for c in corners),
                min(c.z for c in corners),
            )
        )
        max_co = Vector(
            (
                max(c.x for c in corners),
                max(c.y for c in corners),
                max(c.z for c in corners),
            )
        )
        return (min_co + max_co) / 2
    return Vector(
        (
            (default_min.x + default_max.x) / 2,
            (default_min.y + default_max.y) / 2,
            default_min.z,
        )
    )


def add_area_light(name, location, rotation_euler, size, energy, color=(1, 1, 1)):
    bpy.ops.object.light_add(type="AREA", location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.rotation_euler = rotation_euler
    light = obj.data
    light.name = name + "_data"
    light.size = size
    light.energy = energy
    light.color = color
    light.use_shadow = True
    light.shadow_soft_size = size * 0.5
    return obj


def add_spot_light(name, location, rotation_euler, energy, spot_size_deg, color=(1, 1, 1)):
    bpy.ops.object.light_add(type="SPOT", location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.rotation_euler = rotation_euler
    light = obj.data
    light.name = name + "_data"
    light.energy = energy
    light.spot_size = math.radians(spot_size_deg)
    light.spot_blend = 0.4
    light.color = color
    light.use_shadow = True
    return obj


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalize(v):
    length = math.sqrt(_dot(v, v))
    if length == 0:
        raise ValueError(f"无法归一化零向量: {v}")
    return tuple(x / length for x in v)


def _camera_basis(azimuth_deg, elevation_deg):
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    view_from_target = _normalize(
        (
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        )
    )
    forward = tuple(-x for x in view_from_target)
    world_up = (0.0, 0.0, 1.0)
    right = _cross(forward, world_up)
    if _dot(right, right) < 1e-12:
        right = (1.0, 0.0, 0.0)
    else:
        right = _normalize(right)
    up = _normalize(_cross(right, forward))
    return view_from_target, right, up


def _room_corners(room_w, room_d, room_h):
    half_w, half_d, half_h = room_w / 2, room_d / 2, room_h / 2
    return [
        (x, y, z)
        for x in (-half_w, half_w)
        for y in (-half_d, half_d)
        for z in (-half_h, half_h)
    ]


def _bounds_corners(bounds_min, bounds_max, target):
    return [
        (x - target[0], y - target[1], z - target[2])
        for x in (bounds_min[0], bounds_max[0])
        for y in (bounds_min[1], bounds_max[1])
        for z in (bounds_min[2], bounds_max[2])
    ]


def _required_camera_distance(corners, azimuth_deg, elevation_deg, fov_h, fov_v):
    view_from_target, right, up = _camera_basis(azimuth_deg, elevation_deg)
    tan_h = math.tan(fov_h / 2)
    tan_v = math.tan(fov_v / 2)

    required_distance = 0.0
    for corner in corners:
        along_camera = _dot(corner, view_from_target)
        required_distance = max(
            required_distance,
            along_camera + abs(_dot(corner, right)) / tan_h,
            along_camera + abs(_dot(corner, up)) / tan_v,
        )
    return required_distance


def calculate_camera_distance(
    room_w,
    room_d,
    room_h,
    azimuth_deg,
    elevation_deg,
    fov_h,
    fov_v,
    framing_margin=1.0,
):
    """按相机视锥投影计算刚好包住场景包围盒所需的距离。"""
    required_distance = _required_camera_distance(
        corners=_room_corners(room_w, room_d, room_h),
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        fov_h=fov_h,
        fov_v=fov_v,
    )

    return max(required_distance * framing_margin, max(room_w, room_d, room_h) * 0.5)


def calculate_camera_distance_for_bounds(
    bounds_min,
    bounds_max,
    target,
    azimuth_deg,
    elevation_deg,
    fov_h,
    fov_v,
    framing_margin=1.0,
):
    """按真实 bounds 与目标点计算刚好包住场景所需的相机距离。"""
    corners = _bounds_corners(bounds_min, bounds_max, target)
    required_distance = _required_camera_distance(
        corners=corners,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        fov_h=fov_h,
        fov_v=fov_v,
    )
    bounds_size = (
        bounds_max[0] - bounds_min[0],
        bounds_max[1] - bounds_min[1],
        bounds_max[2] - bounds_min[2],
    )
    return max(required_distance * framing_margin, max(bounds_size) * 0.5)


def setup_camera(
    center,
    room_w,
    room_d,
    room_h,
    azimuth_deg=270.0,
    elevation_deg=24.0,
    focal_length_mm=35.0,
    target_center=None,
    bounds_min=None,
    bounds_max=None,
):
    """
    在场景包围盒外放置一台俯视相机，自动保证整体场景尽量进入画面。
    """
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)

    scene = bpy.context.scene
    render = scene.render
    aspect = render.resolution_x / render.resolution_y if render.resolution_y else 1.0
    sensor_w = 36.0
    fov_h = 2 * math.atan(sensor_w / (2 * focal_length_mm))
    fov_v = 2 * math.atan(sensor_w / aspect / (2 * focal_length_mm))

    target = Vector(target_center) if target_center is not None else center
    if bounds_min is not None and bounds_max is not None:
        dist = calculate_camera_distance_for_bounds(
            bounds_min=tuple(bounds_min),
            bounds_max=tuple(bounds_max),
            target=tuple(target),
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
            fov_h=fov_h,
            fov_v=fov_v,
        )
    else:
        dist = calculate_camera_distance(
            room_w=room_w,
            room_d=room_d,
            room_h=room_h,
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
            fov_h=fov_h,
            fov_v=fov_v,
        )
    view_from_target, _, _ = _camera_basis(azimuth_deg, elevation_deg)

    cam_x = target.x + dist * view_from_target[0]
    cam_y = target.y + dist * view_from_target[1]
    cam_z = target.z + dist * view_from_target[2]

    cam_data = bpy.data.cameras.new("Camera_data")
    cam_data.lens = focal_length_mm
    cam_data.clip_start = 0.1
    cam_data.clip_end = max(room_w, room_d, room_h) * 20

    cam_obj = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam_obj)
    cam_obj.location = (cam_x, cam_y, cam_z)

    direction = target - Vector((cam_x, cam_y, cam_z))
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    scene.camera = cam_obj

    print(f"[Camera] ✓ 相机已设置：位置 ({cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f})")
    print(
        f"[Camera]   方位角 {azimuth_deg}°  俯角 {elevation_deg}°"
        f"  焦距 {focal_length_mm}mm  距离 {dist:.2f}m"
    )
    print(f"[Camera]   视角中心目标点: ({target.x:.2f}, {target.y:.2f}, {target.z:.2f})")
    return cam_obj


def setup_indoor_lighting_and_render(
    brightness=1.0,
    azimuth_deg=270.0,
    elevation_deg=24.0,
    focal_length_mm=35.0,
    cycles_samples=512,
    transparent_bg=True,
    render_engine="BLENDER_EEVEE_NEXT",
    output_dir="/path/to/output/",
    output_format="png",
    resolution_x=1920,
    resolution_y=1920,
    world_color=(1.0, 1.0, 1.0, 1.0),
    world_strength=2.0,
    use_gpu=True,
    cycles_gpu_backend="AUTO",
    eevee_samples=64,
    use_raytracing=True,
    use_eevee_ao=True,
):
    """为当前 Blender 场景自动应用灯光、相机和渲染参数。

    Args:
        transparent_bg: 是否启用透明背景。默认 True。
        render_engine: "EEVEE" / "CYCLES"，默认 EEVEE。EEVEE 在 Blender 4.2+
            会自动落到 BLENDER_EEVEE_NEXT，旧版本退化到 BLENDER_EEVEE。
        output_dir: 渲染输出目录，默认 ``/path/to/output/``。
        output_format: "mp4" / "mkv" / "png" / 等。"mp4"/"mkv" 走 FFMPEG H.264，
            其它情况会保留 Blender 默认 PNG 序列。
    """
    remove_existing_setup_objects()

    min_co, max_co = get_scene_bounds()
    center = (min_co + max_co) / 2
    camera_target = get_floor_center(min_co, max_co)
    room_size = max_co - min_co
    room_w, room_d, room_h = room_size.x, room_size.y, room_size.z
    ceil_z = max_co.z + 0.3
    scene = bpy.context.scene
    scene.render.resolution_x = int(resolution_x)
    scene.render.resolution_y = int(resolution_y)
    scene.render.resolution_percentage = 100

    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node is None:
        bg_node = world.node_tree.nodes.new("ShaderNodeBackground")
    bg_node.inputs["Color"].default_value = world_color
    bg_node.inputs["Strength"].default_value = world_strength * brightness

    # 启用 World 级别的环境光遮蔽（主要服务 Cycles；EEVEE 走下面的 GTAO）
    try:
        world.light_settings.use_ambient_occlusion = True
        world.light_settings.ao_factor = 0.6
        world.light_settings.distance = 1.0
    except AttributeError:
        pass

    print(f"[Lighting] 场景包围盒: {min_co} → {max_co}")
    print(f"[Lighting] 房间尺寸: W={room_w:.2f}m  D={room_d:.2f}m  H={room_h:.2f}m")
    print(f"[Camera] 地板中心目标点: {camera_target}")

    main_size = min(room_w, room_d) * 0.7
    main_energy = 200 * brightness
    add_area_light(
        name="MainCeiling",
        location=(center.x, center.y, ceil_z),
        rotation_euler=(0, 0, 0),
        size=main_size,
        energy=main_energy,
        color=(1.0, 0.97, 0.88),
    )

    fill_z = center.z + room_h * 0.5
    fill_dist = max(room_w, room_d) * 0.65
    fill_energy = main_energy * 0.35
    fill_size = min(room_w, room_d) * 0.5
    add_area_light(
        name="FillA",
        location=(center.x + fill_dist, center.y, fill_z),
        rotation_euler=(math.radians(-45), 0, math.radians(90)),
        size=fill_size,
        energy=fill_energy,
        color=(0.95, 0.98, 1.0),
    )
    add_area_light(
        name="FillB",
        location=(center.x, center.y - fill_dist, fill_z),
        rotation_euler=(math.radians(-45), 0, 0),
        size=fill_size,
        energy=fill_energy * 0.7,
        color=(0.95, 0.98, 1.0),
    )
    add_spot_light(
        name="Rim",
        location=(
            center.x - room_w * 0.5,
            center.y + room_d * 0.5,
            ceil_z + room_h * 0.4,
        ),
        rotation_euler=(math.radians(-120), 0, math.radians(-45)),
        energy=main_energy * 0.15,
        spot_size_deg=60,
        color=(1.0, 0.95, 0.85),
    )

    setup_camera(
        center=center,
        room_w=room_w,
        room_d=room_d,
        room_h=room_h,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        focal_length_mm=focal_length_mm,
        target_center=camera_target,
        bounds_min=min_co,
        bounds_max=max_co,
    )

    _set_render_engine(scene, render_engine)
    if use_gpu and scene.render.engine == "CYCLES":
        _configure_cycles_gpu(scene, preferred_backend=cycles_gpu_backend)

    scene.render.film_transparent = transparent_bg
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    try:
        scene.cycles.use_denoising = True
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
        scene.cycles.samples = cycles_samples
    except Exception:
        pass

    try:
        scene.eevee.taa_render_samples = int(eevee_samples)
    except Exception:
        pass

    try:
        scene.eevee.use_ssr = use_raytracing
        scene.eevee.use_ssr_refraction = use_raytracing
        # EEVEE legacy: gtao；EEVEE Next 仍兼容这两个属性
        scene.eevee.use_gtao = use_eevee_ao
        scene.eevee.gtao_distance = 0.5
    except Exception:
        pass
    # EEVEE Next 的环境光遮蔽走 raytracing
    try:
        scene.eevee.use_raytracing = use_raytracing
    except Exception:
        pass

    _set_render_output(scene, output_dir, output_format)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    print("[Lighting] ✓ 灯光设置完成！共创建 4 盏灯光 + 世界环境光 + AO")
    print(
        f"[Render] ✓ 引擎={scene.render.engine}, "
        f"背景={'透明/白色世界' if transparent_bg else '白色世界'}, "
        f"输出={scene.render.filepath} ({scene.render.image_settings.file_format})"
    )


def _configure_cycles_gpu(scene, preferred_backend="AUTO"):
    """尽量启用 Cycles GPU 渲染设备；不可用时保留 CPU。"""
    try:
        cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
    except Exception as exc:
        print(f"[Render] 警告: 无法访问 Cycles 设备偏好，使用 CPU: {exc}")
        scene.cycles.device = "CPU"
        return "CPU"

    requested_backend = (preferred_backend or "AUTO").upper()
    default_backends = ("CUDA", "OPTIX", "HIP", "ONEAPI", "METAL")
    if requested_backend == "AUTO":
        backends = default_backends
    else:
        backends = (requested_backend,)

    for backend in backends:
        try:
            cycles_prefs.compute_device_type = backend
        except Exception:
            continue

        try:
            if hasattr(cycles_prefs, "refresh_devices"):
                cycles_prefs.refresh_devices()
            elif hasattr(cycles_prefs, "get_devices"):
                cycles_prefs.get_devices()
        except Exception:
            pass

        enabled = []
        for device in getattr(cycles_prefs, "devices", []):
            device_type = str(getattr(device, "type", "")).upper()
            is_gpu = device_type not in {"", "NONE", "CPU"}
            try:
                device.use = is_gpu
            except Exception:
                pass
            if is_gpu:
                enabled.append(f"{device_type}:{getattr(device, 'name', 'unknown')}")

        if enabled:
            scene.cycles.device = "GPU"
            print(f"[Render] Cycles GPU 后端: {backend}")
            print(f"[Render] 启用设备: {', '.join(enabled)}")
            return backend

    scene.cycles.device = "CPU"
    print("[Render] 警告: 未找到可用 GPU 设备，使用 CPU")
    return "CPU"


def _set_render_engine(scene, render_engine: str):
    """切换渲染引擎，支持 EEVEE / EEVEE_NEXT / CYCLES，自动兼容 Blender 版本差异。"""
    name = (render_engine or "EEVEE").upper()
    if name == "CYCLES":
        candidates = ("CYCLES",)
    else:
        # Blender 4.2+ 是 BLENDER_EEVEE_NEXT；4.1 及之前是 BLENDER_EEVEE
        candidates = ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE")

    for engine_id in candidates:
        try:
            if scene.render.engine != engine_id:
                scene.render.engine = engine_id
                print(f"[Render] 渲染引擎切换为 {engine_id}")
            return
        except (TypeError, ValueError):
            continue
    print(
        f"[Render] 警告: 当前 Blender 不支持引擎 {render_engine}，"
        f"沿用 {scene.render.engine}"
    )


def _set_render_output(scene, output_dir: str, output_format: str):
    """配置默认输出路径与编码格式。

    - mp4 / mkv 走 FFMPEG H.264（mp4 用 MPEG4 容器，mkv 用 MATROSKA 容器）；
    - 其它格式视为静态图序列，直接交给 Blender 默认 PNG。
    """
    if output_dir:
        # Blender 的 filepath 以 "/" 结尾时表示目录，会自动用 "<start>-<end>" 命名
        scene.render.filepath = output_dir if output_dir.endswith("/") else output_dir + "/"

    fmt = (output_format or "").lower()
    if fmt in ("mp4", "mkv"):
        scene.render.image_settings.file_format = "FFMPEG"
        try:
            scene.render.ffmpeg.format = "MPEG4" if fmt == "mp4" else "MATROSKA"
            scene.render.ffmpeg.codec = "H264"
            scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
            scene.render.ffmpeg.audio_codec = "NONE"
            # mp4 加 faststart 让网络播放器能立刻起播
            if fmt == "mp4":
                scene.render.ffmpeg.use_max_b_frames = False
        except AttributeError:
            print("[Render] 警告: 当前 Blender 构建未启用 FFmpeg，编码参数已跳过")
    elif fmt in ("png", "jpg", "jpeg", "exr", "tiff"):
        scene.render.image_settings.file_format = {
            "png": "PNG",
            "jpg": "JPEG",
            "jpeg": "JPEG",
            "exr": "OPEN_EXR",
            "tiff": "TIFF",
        }[fmt]
