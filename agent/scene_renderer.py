"""
场景渲染器

为 Agent 提供视觉反馈，生成场景俯视图：
1. 显示房间边界
2. 显示已放置物体（带标签）
3. 标注坐标系
4. 标注墙壁方向

使用 matplotlib 实现轻量级渲染
"""

import io
import base64
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
from matplotlib.transforms import Affine2D

from config.constants import DEFAULT_SCENE_HEIGHT
from utils.scene_types import SceneObject, scene_object_list_to_dicts


class SceneRenderer:
    """
    场景渲染器
    
    生成面向 VLM 的极简场景俯视图，用于 Agent 视觉反馈。
    """
    
    def __init__(
        self,
        figsize: Tuple[float, float] = (10, 10),
        dpi: int = 150,
        output_dir: Optional[str] = None,
    ):
        """
        初始化渲染器
        
        Args:
            figsize: 图像尺寸（英寸）
            dpi: 图像 DPI
            output_dir: 输出目录（可选，用于保存渲染图）
        """
        self.figsize = figsize
        self.dpi = dpi
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 颜色方案
        self.colors = {
            'boundary': '#2C3E50',      # 深蓝灰 - 边界
            'floor': '#ECF0F1',         # 浅灰 - 地板
            'object': '#3498DB',        # 蓝色 - 物体填充
            'object_edge': '#2980B9',   # 深蓝 - 物体边框
            'text': '#2C3E50',          # 深蓝灰 - 文字
            'axis_x': '#E74C3C',        # 红色 - X 轴
            'axis_y': '#27AE60',        # 绿色 - Y 轴
            'grid': '#BDC3C7',          # 浅灰 - 网格
            'wall_label': '#7F8C8D',    # 灰色 - 墙壁标签
        }
        
        # 物体颜色列表（用于区分不同物体）
        self.object_colors = [
            '#3498DB',  # 蓝
            '#E74C3C',  # 红
            '#2ECC71',  # 绿
            '#F39C12',  # 橙
            '#9B59B6',  # 紫
            '#1ABC9C',  # 青
            '#E91E63',  # 粉
            '#00BCD4',  # 青蓝
            '#FF5722',  # 深橙
            '#795548',  # 棕
        ]
    
    def render_scene(
        self,
        boundary: List[List[float]],
        existing_objects: Sequence[Union[SceneObject, Dict[str, Any]]],
        scene_height: float = DEFAULT_SCENE_HEIGHT,
        new_object_info: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, str]:
        """
        渲染场景俯视图

        Args:
            boundary: 边界点列表 [[x1, y1], [x2, y2], ...]
            existing_objects: 已放置物体列表
            scene_height: 房间高度
            new_object_info: 待放置物体信息（可选）
            save_path: 保存路径（可选）

        Returns:
            (fig, base64_image): matplotlib Figure 对象和 base64 编码的图像
        """
        existing_objects = scene_object_list_to_dicts(existing_objects)
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        
        # 计算边界范围
        if boundary:
            xs = [p[0] for p in boundary]
            ys = [p[1] for p in boundary]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            width = x_max - x_min
            depth = y_max - y_min
        else:
            x_min, x_max = 0, 6
            y_min, y_max = 0, 6
            width = depth = 6
        
        # 极简渲染：留出更大外边距，用于放置坐标轴、外侧刻度和方向标签
        margin = max(width, depth) * 0.14
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_aspect('equal')
        
        # 绘制房间边界
        if boundary:
            floor_polygon = plt.Polygon(
                boundary,
                facecolor="#FBFBFA",
                edgecolor="#111111",
                linewidth=2.8,
                zorder=1
            )
            ax.add_patch(floor_polygon)
        
        # 沿房间边界绘制简洁刻度，给 VLM 提供尺度感但避免整图网格噪声
        self._draw_axis_ticks(ax, x_min, x_max, y_min, y_max, max(width, depth))
        
        # 将坐标轴移到房间外侧，避免与角落物体和标签重叠
        axis_gap = max(width, depth) * 0.10
        self._draw_coordinate_axes(
            ax,
            x_min - axis_gap,
            y_min - axis_gap,
            max(width, depth) * 0.13,
        )
        
        # 仅保留 FRONT / RIGHT 两个方向标签，减少视觉负担
        self._draw_wall_labels(ax, x_min, x_max, y_min, y_max)
        
        # 绘制已放置物体
        for idx, obj in enumerate(existing_objects):
            self._draw_object(ax, obj, idx)
        
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()
        
        # 转换为 base64
        base64_image = self._fig_to_base64(fig)
        
        # 保存图像
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(save_path), dpi=self.dpi, bbox_inches='tight')
            print(f"[SceneRenderer] Saved to {save_path}")
        elif self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            default_path = self.output_dir / f"scene_render_{len(existing_objects)}_objects.png"
            fig.savefig(str(default_path), dpi=self.dpi, bbox_inches='tight')
            print(f"[SceneRenderer] Saved to {default_path}")
        
        return fig, base64_image
    
    def _draw_coordinate_axes(
        self,
        ax: plt.Axes,
        x_origin: float,
        y_origin: float,
        arrow_length: float,
    ):
        """绘制坐标轴箭头"""
        # X 轴（红色）
        ax.annotate(
            '', 
            xy=(x_origin + arrow_length, y_origin),
            xytext=(x_origin, y_origin),
            arrowprops=dict(
                arrowstyle='->', 
                color=self.colors['axis_x'],
                lw=2.2,
                mutation_scale=12
            ),
            zorder=100
        )
        ax.text(
            x_origin + arrow_length + 0.06, y_origin,
            '+X', fontsize=16, fontweight='bold',
            color=self.colors['axis_x'], ha='left', va='center',
            zorder=100
        )
        
        # Y 轴（绿色）
        ax.annotate(
            '',
            xy=(x_origin, y_origin + arrow_length),
            xytext=(x_origin, y_origin),
            arrowprops=dict(
                arrowstyle='->',
                color=self.colors['axis_y'],
                lw=2.2,
                mutation_scale=12
            ),
            zorder=100
        )
        ax.text(
            x_origin, y_origin + arrow_length + 0.06,
            '+Y', fontsize=16, fontweight='bold',
            color=self.colors['axis_y'], ha='center', va='bottom',
            zorder=100
        )
        
        # 原点标记
        ax.plot(x_origin, y_origin, 'o', color='black', markersize=4.5, zorder=100)
        ax.text(
            x_origin - 0.06, y_origin - 0.06,
            'O', fontsize=12, fontweight='bold',
            color='black', ha='right', va='top',
            zorder=100
        )
    
    def _draw_wall_labels(
        self,
        ax: plt.Axes,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ):
        """绘制墙壁方向标签"""
        mid_x = (x_min + x_max) / 2
        mid_y = (y_min + y_max) / 2
        offset = 0.3
        
        # 极简模式只保留与坐标轴互补的两个标签
        labels = [
            (x_max + offset, mid_y, 'RIGHT', 'left', 'center'),
            (mid_x, y_max + offset, 'FRONT', 'center', 'bottom'),
        ]
        
        for x, y, text, ha, va in labels:
            ax.text(
                x, y, text,
                fontsize=16, fontweight='bold',
                color=self.colors['wall_label'],
                ha=ha, va=va,
                alpha=0.8,
                zorder=50
            )
    
    def _draw_axis_ticks(
        self,
        ax: plt.Axes,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        scale: float,
    ):
        """沿房间下边和左边外侧绘制整数刻度。"""
        tick_len = scale * 0.012
        label_offset = scale * 0.035
        
        x_start = int(math.ceil(x_min))
        x_end = int(math.floor(x_max))
        y_start = int(math.ceil(y_min))
        y_end = int(math.floor(y_max))
        
        for x in range(x_start, x_end + 1):
            ax.plot([x, x], [y_min, y_min - tick_len], color='#666666', lw=0.9, zorder=3)
            ax.text(
                x, y_min - label_offset, str(x),
                ha='center', va='top', fontsize=14, color='#555555', zorder=3
            )
        
        for y in range(y_start, y_end + 1):
            ax.plot([x_min - tick_len, x_min], [y, y], color='#666666', lw=0.9, zorder=3)
            ax.text(
                x_min - label_offset, y, str(y),
                ha='right', va='center', fontsize=14, color='#555555', zorder=3
            )
    
    def _draw_object(
        self,
        ax: plt.Axes,
        obj: Dict[str, Any],
        idx: int = 0,
    ):
        """绘制物体（带旋转的矩形），每个物体使用不同颜色"""
        center = obj.get('center')
        size = obj.get('size')
        rotation = obj.get('rotation')
        obj_id = obj.get('id')
        
        # 提取 XY 平面信息
        cx, cy = center[0], center[1]
        w, d = size[0], size[1]  # width (X), depth (Y)
        rz = rotation[2] if len(rotation) > 2 else 0  # Z 轴旋转角度
        
        fill_color = self.object_colors[idx % len(self.object_colors)]
        
        # 创建矩形（中心在原点）
        rect = patches.Rectangle(
            (-w / 2, -d / 2), w, d,
            facecolor=fill_color,
            edgecolor='#222222',
            linewidth=1.8,
            alpha=0.35,
            zorder=10,
        )
        
        # 应用旋转和平移
        transform = Affine2D().rotate_deg(rz).translate(cx, cy) + ax.transData
        rect.set_transform(transform)
        ax.add_patch(rect)
        
        label = ax.text(
            cx, cy,
            str(obj_id),
            fontsize=12, fontweight='bold',
            color=fill_color,
            ha='center', va='center',
            zorder=20,
            bbox=dict(
                boxstyle='round,pad=0.18',
                facecolor='white',
                edgecolor=fill_color,
                linewidth=1.4,
                alpha=0.96,
            ),
        )
        label.set_path_effects([
            path_effects.withStroke(linewidth=1.0, foreground='white')
        ])

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """将 matplotlib Figure 转换为 base64 编码的 PNG"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        buf.seek(0)
        base64_image = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return base64_image


# ==================== 测试代码 ====================

if __name__ == "__main__":
    # 测试渲染器
    renderer = SceneRenderer(output_dir="output/agent/renders")
    
    # 测试场景数据
    boundary = [[0, 0], [6, 0], [6, 4.5], [0, 4.5]]
    existing_objects = [
        {
            "id": 0,
            "short_desc": "grey upholstered bed",
            "center": [3.07, 3.42, 0.56],
            "size": [2.08, 2.19, 1.11],
            "rotation": [0, 0, 0]
        },
        {
            "id": 1,
            "short_desc": "gray bronze wardrobe",
            "center": [0.32, 3.5, 1.05],
            "size": [1.0, 0.62, 2.1],
            "rotation": [0, 0, 90]
        },
        {
            "id": 2,
            "short_desc": "mahogany dresser",
            "center": [5.4, 3.8, 0.46],
            "size": [1.23, 0.54, 0.92],
            "rotation": [0, 0, 0]
        },
    ]
    
    new_object = {
        "id": 3,
        "short_desc": "nightstand",
        "size": [0.46, 0.42, 0.5]
    }
    
    fig, base64_img = renderer.render_scene(
        boundary=boundary,
        existing_objects=existing_objects,
        new_object_info=new_object,
        save_path="output/agent/renders/test_scene.png"
    )
    
    print(f"Base64 image length: {len(base64_img)}")
    plt.show()
