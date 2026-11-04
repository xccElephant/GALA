"""
场景规划 Agent

调用大模型进行场景规划和指令生成，支持：
1. 场景规划：解析用户需求，生成物体清单和摆放顺序
2. 指令生成：为每个物体生成摆放指令
3. 编辑模式：在现有场景基础上添加新物体
"""

import json
import re
from typing import Any, Dict, List, Optional

from agent.llm import call_llm, call_vlm
from agent.prompts import (
    EDIT_INSTRUCTION_SYSTEM_PROMPT,
    EDIT_INSTRUCTION_USER_TEMPLATE,
    INSTRUCTION_GENERATION_SYSTEM_PROMPT,
    INSTRUCTION_GENERATION_USER_TEMPLATE,
    SCENE_PLANNING_SYSTEM_PROMPT,
    SCENE_PLANNING_USER_TEMPLATE,
    VALIDATION_FEEDBACK_SYSTEM_PROMPT,
    VALIDATION_FEEDBACK_USER_TEMPLATE,
    VISUAL_INSTRUCTION_GENERATION_SYSTEM_PROMPT,
    VISUAL_INSTRUCTION_GENERATION_USER_TEMPLATE,
    VISUAL_VALIDATION_FEEDBACK_SYSTEM_PROMPT,
    VISUAL_VALIDATION_FEEDBACK_USER_TEMPLATE,
    format_boundary_info,
    format_existing_objects_info,
)
from config.constants import DEFAULT_SCENE_HEIGHT, RELATIVE_RELATION_TEXT
from utils.scene_types import Scene, SceneObject, scene_object_list_to_dicts

# 场景规划要一次性输出整屋物体清单，开启原生 think 后推理较长，单独放宽 token 上限
SCENE_PLANNING_MAX_TOKENS = 16384


def build_instruction_from_relations(
    short_desc: str,
    expected_relations: Dict[str, Any],
    existing_objects: List[SceneObject],
    parent_id: Optional[int] = None,
    include_object_descriptions: bool = True,
    category: str = "",
) -> str:
    """
    根据 expected_relations 用代码拼接摆放指令，格式与训练集完全一致：
      place the {short_desc} {part1}, and {part2}, and ...

    Args:
        short_desc: 物体简短描述
        expected_relations: LLM 返回的空间关系，格式：
            {
                "absolute_relations": ["against south wall", ...],
                "relative_relations": [{"type": "next_to", "target_id": 0}, ...]
            }
        existing_objects: 已放置的物体列表，用于查找 target 的 short_desc
        parent_id: 父物体 ID。非 None 时为子物体，强制跳过绝对关系（子物体位置由父物体决定）
    """
    obj_by_id = {obj.id: obj for obj in existing_objects}
    parts: List[str] = []
    if include_object_descriptions:
        subject_desc = short_desc
    else:
        subject_desc = (category or "object") if category else "object"

    if parent_id is None:
        for abs_rel in expected_relations.get("absolute_relations", []):
            parts.append(str(abs_rel))

    for rel in expected_relations.get("relative_relations", []):
        rel_type = rel.get("type", "")
        target_id = rel.get("target_id")
        rel_text = RELATIVE_RELATION_TEXT.get(rel_type)
        if rel_text is None or target_id is None:
            continue
        target_obj = obj_by_id.get(target_id)
        if include_object_descriptions and target_obj:
            target_desc = target_obj.short_desc
        else:
            target_desc = (
                (target_obj.category or "object") if target_obj else "object"
            )
        parts.append(f"{rel_text} the {target_desc} (id {target_id})")

    if not parts:
        return f"place the {subject_desc} in a suitable position."
    return f"place the {subject_desc} " + ", and ".join(parts) + "."


class ScenePlannerAgent:
    """场景规划 Agent"""

    def __init__(
        self,
        temperature: float = 0.7,
        max_retries: int = 3,
        verbose: bool = False,
        include_object_descriptions: bool = False,
    ):
        self.temperature = temperature
        self.max_retries = max_retries
        self.verbose = verbose
        self.include_object_descriptions = include_object_descriptions

    def plan_scene(
        self,
        user_input: str,
        boundary: List[List[float]],
        scene_height: float = DEFAULT_SCENE_HEIGHT,
        stream: bool = True,
    ) -> Dict[str, Any]:
        """规划场景：生成物体清单和摆放顺序"""
        response = call_llm(
            system_prompt=SCENE_PLANNING_SYSTEM_PROMPT,
            user_prompt=SCENE_PLANNING_USER_TEMPLATE.format(
                user_input=user_input,
                boundary=format_boundary_info(boundary),
                scene_height=scene_height,
            ),
            task="scene_planner",
            temperature=self.temperature,
            stream=stream,
            max_retries=self.max_retries,
            verbose=self.verbose,
            max_tokens=SCENE_PLANNING_MAX_TOKENS,
        )

        result = self._extract_json(response)
        if result is None:
            raise ValueError(
                f"Failed to parse scene plan JSON, raw response:\n{response}"
            )

        return result

    def generate_instruction(
        self,
        scene_object: SceneObject,
        scene: Scene,
        existing_objects: List[SceneObject],
        object_description: str = "",
        scene_image_base64: Optional[str] = None,
    ) -> Dict[str, str]:
        """为物体生成摆放指令"""
        short_desc = scene_object.short_desc
        prompt_kwargs = self._build_prompt_kwargs(
            scene_object, object_description, scene, existing_objects
        )

        if scene_image_base64:
            response = call_vlm(
                system_prompt=VISUAL_INSTRUCTION_GENERATION_SYSTEM_PROMPT,
                user_prompt=VISUAL_INSTRUCTION_GENERATION_USER_TEMPLATE.format(
                    **prompt_kwargs
                ),
                image_base64=scene_image_base64,
                task="instruction_generator",
                temperature=0.5,
                max_retries=self.max_retries,
                verbose=self.verbose,
            )
        else:
            response = call_llm(
                system_prompt=INSTRUCTION_GENERATION_SYSTEM_PROMPT,
                user_prompt=INSTRUCTION_GENERATION_USER_TEMPLATE.format(
                    **prompt_kwargs
                ),
                task="instruction_generator",
                temperature=0.5,
                max_retries=self.max_retries,
                verbose=self.verbose,
            )

        return self._process_instruction_response(
            response,
            short_desc,
            existing_objects,
            object_id=scene_object.id,
            category=scene_object.category,
        )

    def generate_edit_instruction(
        self,
        user_input: str,
        scene: Scene,
        existing_objects: List[SceneObject],
    ) -> Dict[str, Any]:
        """编辑模式：解析用户编辑请求"""
        response = call_llm(
            system_prompt=EDIT_INSTRUCTION_SYSTEM_PROMPT,
            user_prompt=EDIT_INSTRUCTION_USER_TEMPLATE.format(
                user_input=user_input,
                scene_type=scene.scene_type,
                scene_description=scene.scene_description,
                boundary=format_boundary_info(scene.boundary),
                scene_height=scene.scene_height,
                existing_objects_info=format_existing_objects_info(
                    scene_object_list_to_dicts(existing_objects)
                ),
            ),
            task="edit_parser",
            temperature=self.temperature,
            max_retries=self.max_retries,
            verbose=self.verbose,
        )

        result = self._extract_json(response)
        if result is None:
            raise ValueError(
                f"Failed to parse edit instruction JSON, raw response:\n{response}"
            )
        return result

    def regenerate_instruction_with_feedback(
        self,
        scene_object: SceneObject,
        scene: Scene,
        existing_objects: List[SceneObject],
        previous_instruction: str,
        validation_result: Dict[str, Any],
        compliance_result: Optional[Dict[str, Any]] = None,
        object_description: str = "",
        scene_image_base64: Optional[str] = None,
    ) -> Dict[str, str]:
        """基于验证失败反馈，重新生成摆放指令"""
        short_desc = scene_object.short_desc
        prompt_kwargs = self._build_prompt_kwargs(
            scene_object, object_description, scene, existing_objects
        )
        prompt_kwargs["previous_instruction"] = previous_instruction
        prompt_kwargs["failure_details"] = self._format_failure_details(
            validation_result, compliance_result, existing_objects
        )

        if scene_image_base64:
            response = call_vlm(
                system_prompt=VISUAL_VALIDATION_FEEDBACK_SYSTEM_PROMPT,
                user_prompt=VISUAL_VALIDATION_FEEDBACK_USER_TEMPLATE.format(
                    **prompt_kwargs
                ),
                image_base64=scene_image_base64,
                task="instruction_generator",
                temperature=0.6,
                max_retries=self.max_retries,
                verbose=self.verbose,
            )
        else:
            response = call_llm(
                system_prompt=VALIDATION_FEEDBACK_SYSTEM_PROMPT,
                user_prompt=VALIDATION_FEEDBACK_USER_TEMPLATE.format(**prompt_kwargs),
                task="instruction_generator",
                temperature=0.6,
                max_retries=self.max_retries,
                verbose=self.verbose,
            )

        return self._process_instruction_response(
            response,
            short_desc,
            existing_objects,
            object_id=scene_object.id,
            category=scene_object.category,
        )

    # ========== 辅助方法 ==========

    def _build_prompt_kwargs(
        self,
        scene_object: SceneObject,
        object_description: str,
        scene: Scene,
        existing_objects: List[SceneObject],
    ) -> Dict:
        """构建 prompt 参数"""
        return dict(
            scene_type=scene.scene_type,
            scene_description=scene.scene_description,
            boundary=format_boundary_info(scene.boundary),
            scene_height=scene.scene_height,
            object_id=scene_object.id,
            object_description=object_description or scene_object.short_desc,
            object_size=[round(s, 2) for s in scene_object.size],
            object_category=scene_object.category,
            existing_objects_info=format_existing_objects_info(
                scene_object_list_to_dicts(existing_objects)
            ),
            short_desc=scene_object.short_desc,
        )

    @staticmethod
    def _split_thinking(text: str) -> tuple[str, str]:
        """分离 Qwen3 原生 <think>...</think> 思考块与正式输出。

        返回 (thinking, body)。没有思考块时 thinking 为空、body 为原文。
        """
        if "</think>" not in text:
            return "", text
        head, _, body = text.rpartition("</think>")
        thinking = head.split("<think>", 1)[-1].strip()
        return thinking, body

    def _extract_json(self, text: str) -> Optional[Dict]:
        """从响应文本中提取 JSON（先剥离原生 think，再把推理回填到 thinking 字段）"""

        def ensure_dict(result):
            if isinstance(result, list) and result and isinstance(result[0], dict):
                return result[0]
            return result if isinstance(result, dict) else None

        thinking, body = self._split_thinking(text)

        def finalize(result):
            if isinstance(result, dict) and thinking and not result.get("thinking"):
                result["thinking"] = thinking
            return result

        try:
            return finalize(ensure_dict(json.loads(body)))
        except json.JSONDecodeError:
            pass

        for pattern in [
            r"```json\s*([\s\S]*?)\s*```",
            r"```\s*([\s\S]*?)\s*```",
            r"\{[\s\S]*\}",
        ]:
            match = re.search(pattern, body)
            if match:
                try:
                    json_str = match.group(1) if "```" in pattern else match.group(0)
                    return finalize(ensure_dict(json.loads(json_str)))
                except (json.JSONDecodeError, IndexError):
                    continue
        return None

    @staticmethod
    def _parse_parent_id(result: Dict[str, Any]) -> Optional[int]:
        """从 LLM 响应中提取 parent_id。

        返回 None 表示主物体，int 表示父物体 ID。
        """
        raw = result.get("parent_id")
        if raw is not None:
            return int(raw)
        return None

    def _process_instruction_response(
        self,
        response: str,
        short_desc: str,
        existing_objects: List[SceneObject],
        object_id: Optional[int] = None,
        category: str = "",
    ) -> Dict[str, Any]:
        """处理指令生成的 LLM 响应，用代码拼接指令保证训推格式一致"""
        result = self._extract_json(response)

        thinking = ""
        expected_relations: Dict[str, Any] = {}
        parent_id: Optional[int] = None

        if result and isinstance(result, dict):
            thinking = result.get("thinking", "")
            expected_relations = result.get("expected_relations", {})
            parent_id = self._parse_parent_id(result)
        else:
            print(
                f"[WARNING] Failed to parse relations JSON for '{short_desc}' (id={object_id}), raw:\n{response}"
            )

        normalized = self.normalize_relations(expected_relations)
        instruction = build_instruction_from_relations(
            short_desc,
            normalized,
            existing_objects,
            parent_id=parent_id,
            include_object_descriptions=self.include_object_descriptions,
            category=category,
        )
        return {
            "thinking": thinking,
            "instruction": instruction,
            "expected_relations": normalized,
            "parent_id": parent_id,
        }

    def normalize_relations(self, relations: Dict[str, Any]) -> Dict[str, Any]:
        """校验并提取 expected_relations 结构"""
        if not relations or not isinstance(relations, dict):
            return {"relative_relations": [], "absolute_relations": []}

        relative = []
        for rel in relations.get("relative_relations", []):
            if isinstance(rel, dict) and "type" in rel:
                target_id = rel.get("target_id")
                if target_id is not None:
                    relative.append({"type": rel["type"], "target_id": int(target_id)})

        absolute = []
        for rel in relations.get("absolute_relations", []):
            if isinstance(rel, str) and rel.strip():
                absolute.append(rel.strip())

        return {"relative_relations": relative, "absolute_relations": absolute}

    def _format_failure_details(
        self,
        validation_result: Dict[str, Any],
        compliance_result: Optional[Dict[str, Any]],
        existing_objects: List[SceneObject],
    ) -> str:
        """格式化验证失败信息（包括物理验证和指令符合度）"""
        parts = []

        # 1. 格式化 compliance 失败信息（空间关系不满足）
        if compliance_result and not compliance_result.get("compliant", True):
            compliance_parts = []

            # 相对关系失败
            for rel in compliance_result.get("relative_results", []):
                if not rel.get("satisfied"):
                    rel_type = rel.get("type", "unknown")
                    target_id = rel.get("target_id")
                    target_desc = self._get_object_desc_by_id(
                        existing_objects, target_id
                    )
                    compliance_parts.append(
                        f"  - Relation '{rel_type}' to {target_desc} (id {target_id}): NOT SATISFIED"
                    )

            # 绝对关系失败
            for rel in compliance_result.get("absolute_results", []):
                if not rel.get("satisfied"):
                    rel_type = rel.get("type", "unknown")
                    compliance_parts.append(f"  - Relation '{rel_type}': NOT SATISFIED")

            if compliance_parts:
                ratio = compliance_result.get("compliance_ratio", 0)
                parts.append(f"- INSTRUCTION NOT FOLLOWED (compliance: {ratio:.0%}):")
                parts.extend(compliance_parts)

        # 2. 格式化物理验证失败信息
        if validation_result:
            errors = validation_result.get("errors", [])
            details = validation_result.get("details", {})
            warnings = validation_result.get("warnings", [])

            if "collision_detected" in errors:
                info = details.get("collision", {})
                objs = [
                    f"{o.get('object_desc', 'unknown')} (id {o.get('object_id', '?')})"
                    for o in info.get("colliding_objects", [])
                ]
                parts.append(
                    f"- COLLISION with: {', '.join(objs) or 'existing objects'}. Ratio: {info.get('collision_ratio', 0):.1%}"
                )

            if "out_of_boundary" in errors:
                parts.append(
                    f"- OUT OF BOUNDARY: {len(details.get('boundary', {}).get('out_of_bounds_points', []))} vertices outside"
                )

            if "exceeds_ceiling" in errors:
                h = details.get("height", {})
                parts.append(
                    f"- EXCEEDS CEILING: top={h.get('object_top', 0):.2f}m, ceiling={h.get('scene_height', 2.6):.2f}m"
                )

            if "object_floating" in warnings:
                parts.append(
                    f"- FLOATING: bottom at z={details.get('floating', {}).get('bottom_z', 0):.2f}m"
                )

        return "\n".join(parts) or "Validation failed."

    def _get_object_desc_by_id(
        self, existing_objects: List[SceneObject], object_id: int
    ) -> str:
        """根据 ID 获取物体描述"""
        for obj in existing_objects:
            if obj.id == object_id:
                return obj.short_desc
        return "unknown object"
