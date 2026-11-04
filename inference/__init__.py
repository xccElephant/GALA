"""
Spatial language model inference module
"""

__all__ = [
    "SceneLayoutInference",
    "ModelService",
    "PointCloudSceneManager",
]


def __getattr__(name):
    if name == "SceneLayoutInference":
        from inference.scene_layout_inference import SceneLayoutInference

        return SceneLayoutInference
    if name == "ModelService":
        from inference.model_service import ModelService

        return ModelService
    if name == "PointCloudSceneManager":
        from inference.pointcloud_scene_manager import PointCloudSceneManager

        return PointCloudSceneManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
