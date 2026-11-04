import logging
from typing import Optional

import numpy as np
import open3d as o3d

log = logging.getLogger(__name__)


def load_o3d_pcd(file_path: str):
    """Load a point cloud from file using Open3D."""
    return o3d.io.read_point_cloud(file_path)


def get_points_and_colors(pcd: o3d.geometry.PointCloud, file_path: Optional[str] = None):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    path_info = f" (path: {file_path})" if file_path else ""
    if colors.size == 0:
        raise ValueError(f"Point cloud has no colors{path_info}")
    if colors.ndim != 2:
        raise ValueError(f"Point cloud colors has invalid shape {colors.shape}{path_info}")
    if colors.shape[1] not in (3, 4):
        raise ValueError(f"Point cloud colors must have 3 or 4 channels, got {colors.shape[1]}{path_info}")
    if colors.shape[0] != points.shape[0]:
        raise ValueError(
            f"Point cloud colors count {colors.shape[0]} does not match points {points.shape[0]}{path_info}"
        )
    if colors.shape[1] == 4:
        colors = colors[:, :3]  # Drop alpha channel if present.
    if colors.max() < 1.1:
        colors = (colors * 255).astype(np.uint8)
    return points, colors


def load_pcd_with_object_id(file_path: str, return_normals: bool = False):
    """Load point cloud fields (xyz/rgb/object_id) from a .ply file."""
    import plyfile

    plydata = plyfile.PlyData.read(file_path)
    vertex = plydata["vertex"]
    field_names = vertex.data.dtype.names

    points = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T

    missing_color_fields = [name for name in ("red", "green", "blue") if name not in field_names]
    if missing_color_fields:
        raise ValueError(f"Point cloud is missing color fields {missing_color_fields} (path: {file_path})")

    colors = np.vstack([vertex["red"], vertex["green"], vertex["blue"]]).T
    if colors.size == 0:
        raise ValueError(f"Point cloud has no colors (path: {file_path})")
    if colors.shape[0] != points.shape[0]:
        raise ValueError(
            f"Point cloud colors count {colors.shape[0]} does not match points {points.shape[0]} (path: {file_path})"
        )
    if colors.max() <= 1.1:
        colors = (colors * 255).astype(np.uint8)
    else:
        colors = colors.astype(np.uint8)

    normals = None
    has_normals = all(name in field_names for name in ("nx", "ny", "nz"))
    if has_normals:
        normals = np.vstack([vertex["nx"], vertex["ny"], vertex["nz"]]).T.astype(np.float32)

    object_ids = None
    if "object_id" in field_names:
        object_ids = np.asarray(vertex["object_id"], dtype=np.int64)
    if object_ids is None:
        raise ValueError(f"Point cloud is missing required field 'object_id' (path: {file_path})")

    if return_normals:
        return points, colors, object_ids, normals
    return points, colors, object_ids
