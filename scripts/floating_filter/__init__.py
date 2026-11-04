"""Blender 悬浮/穿地过滤相关模块。

子模块按依赖边界拆分：
- ``detection``: 悬浮判定（纯函数 + Blender 射线检测，可脱离 Blender 单测纯函数）
- ``checkpoint``: 断点续跑 / 分片 / 聚合（无 Blender 依赖）
- ``runner``: Blender 编排入口（强依赖 bpy，需在 Blender 内运行）

注意：本 ``__init__`` 故意不导入 ``runner``，避免在无 Blender 的环境下触发 ``import bpy``。
"""
