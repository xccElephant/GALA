"""
Agent Prompt 模板

定义与大语言模型交互的 prompt 模板，用于：
1. 场景规划：解析用户需求，生成物体清单
2. 指令生成：为每个物体生成摆放指令
3. 结果评估：评估摆放效果
"""

# ==================== 辅助函数 ====================


def _join_prompt_blocks(*blocks: str) -> str:
    """按顺序拼接 prompt block, 并统一处理空行。"""
    return "\n\n".join(block.strip("\n") for block in blocks if block)


def _format_name_list(names: tuple[str, ...]) -> str:
    """统一格式化 relation 名称列表。"""
    return " / ".join(f'"{name}"' for name in names)


def format_existing_objects_info(existing_objects: list) -> str:
    """格式化已存在物体信息，用于 prompt。

    对于有 parent_id 的 sub-object，会在末尾追加 ``parent: ID X`` 标注，
    让 LLM 知道哪些物体已经依附在某个父物体上。
    """
    if not existing_objects:
        return "None (empty scene)"

    lines = []
    for obj in existing_objects:
        obj_id = obj.get("id")
        short_desc = obj.get("short_desc")
        center = obj.get("center")
        size = obj.get("size")
        rotation = obj.get("rotation")

        center_str = f"[{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]"
        size_str = f"[{size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f}]"
        rotation_str = f"[{rotation[0]:.0f}, {rotation[1]:.0f}, {rotation[2]:.0f}]"
        line = f"- ID {obj_id}: {short_desc}, center={center_str}, size={size_str}, rotation={rotation_str}"

        parent_id = obj.get("parent_id")
        if parent_id is not None:
            line += f", parent: ID {parent_id}"

        lines.append(line)

    return "\n".join(lines)


def format_boundary_info(boundary: list) -> str:
    """格式化边界信息，包含明确的坐标范围"""
    if not boundary:
        return "Not specified"

    # 计算边界范围
    xs = [p[0] for p in boundary]
    ys = [p[1] for p in boundary]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    width = x_max - x_min
    depth = y_max - y_min

    return (
        f"{boundary}\n"
        f"  X range: [{x_min:.1f}, {x_max:.1f}] (width: {width:.1f}m, wall at x={x_min:.1f} is LEFT, wall at x={x_max:.1f} is RIGHT)\n"
        f"  Y range: [{y_min:.1f}, {y_max:.1f}] (depth: {depth:.1f}m, wall at y={y_min:.1f} is BACK, wall at y={y_max:.1f} is FRONT)"
    )


# ==================== 公共 Prompt Blocks ====================

WALL_RELATION_NAMES = (
    "against east wall",
    "against west wall",
    "against north wall",
    "against south wall",
)

CORNER_RELATION_NAMES = (
    "at northeast corner",
    "at northwest corner",
    "at southeast corner",
    "at southwest corner",
)

REGION_RELATION_NAMES = (
    "in front region",
    "in back region",
    "in left region",
    "in right region",
    "in center region",
)

DIRECTION_RELATION_NAMES = (
    "left",
    "right",
    "in_front",
    "behind",
)

OTHER_RELATIVE_RELATION_NAMES = (
    "next_to",
    "face_to",
    "on",
    "under",
    "within",
)

ABSOLUTE_RELATION_NAMES = (
    *WALL_RELATION_NAMES,
    *CORNER_RELATION_NAMES,
    *REGION_RELATION_NAMES,
)

RELATIVE_RELATION_NAMES = (
    *DIRECTION_RELATION_NAMES,
    *OTHER_RELATIVE_RELATION_NAMES,
)

WORLD_COORDINATE_SYSTEM_PROMPT = """## World Coordinate System
The scene uses a RIGHT-HANDED coordinate system:
- X axis: positive direction points to the RIGHT (east)
- Y axis: positive direction points to the FRONT (north)
- Z axis: positive direction points to the UP

Direction mappings:
- "in_front" / "behind" = movement along Y axis (+Y / -Y)
- "right" / "left" = movement along X axis (+X / -X)"""

ABSOLUTE_POSITION_DEFINITIONS_PROMPT = (
    """## Absolute Position Definitions
Wall positions (object's edge is against a specific wall):
- "against east wall": object is against the +X wall (right)
- "against west wall": object is against the -X wall (left)
- "against north wall": object is against the +Y wall (front)
- "against south wall": object is against the -Y wall (back)

Corner positions (object is against two adjacent walls):
- """
    + _format_name_list(CORNER_RELATION_NAMES)
    + """

Region positions (object is in a specific region of the room):
- """
    + _format_name_list(REGION_RELATION_NAMES)
    + """
"""
)

RELATIVE_RELATION_DEFINITIONS_PROMPT = (
    """## Relative Relation Definitions
- """
    + _format_name_list(DIRECTION_RELATION_NAMES)
    + """: directional relation, edge distance < 1.5m
- "next_to": very close proximity (edge distance < 0.5m)
- "face_to": object A's front direction points toward object B
- "on": object A is on top of object B (bottom of A touches top of B), e.g. a book on a desk
- "within": object A is within object B (within B's 3D bounding box), e.g. a book within a shelf, or a pillow within a bed
- "under": object A is under object B (center of A is within B's XY projection, below B's top), e.g. a chair under a desk"""
)

SPACE_AWARENESS_PROMPT = """## CRITICAL: Space Awareness Before Using Direction Relations
Before generating any direction-based instruction, you MUST check:
- Calculate the object's bounding box from its center and size
- Check if the target direction has available space (not blocked by walls/boundary)
- If an object is against a wall on one side, you CANNOT place anything further in that direction

Example: If a bed with center=[3, 3.5] and size=[2, 2] is in a room with boundary y_max=4.5:
- The bed's front edge is at y = 3.5 + 2/2 = 4.5, touching the wall
- You CANNOT use "in_front" of the bed because +Y direction is blocked by the wall
- You should use "behind" (-Y direction), or "left" / "right" instead"""

IMAGE_UNDERSTANDING_PROMPT = """## Image Understanding
The provided image shows:
- A top-down view of the room
- Coordinate axes near the lower-left outside the room (red = +X direction, green = +Y direction)
- FRONT and RIGHT direction labels
- Already placed objects shown as colored rectangles with numeric ID labels
- Integer tick marks on the room boundary for rough scale reference"""

TEXT_VS_IMAGE_PROMPT = """## How To Use Text vs Image
- Use the image mainly for geometry: empty space, rough distances, wall proximity, and room layout
- Use the text object list as the authoritative source for object IDs and descriptions
- The numeric ID printed inside each rectangle should match the IDs in the text list"""

TEXT_VS_IMAGE_WITH_FAILURE_PROMPT = """## How To Use Text vs Image
- Use the image mainly for geometry: empty space, rough distances, wall proximity, and room layout
- Use the text object list and failure details as the authoritative source for object IDs and descriptions
- The numeric ID printed inside each rectangle should match the IDs in the text list"""

VISUAL_SPACE_AWARENESS_PROMPT = """## Visual Space Awareness
- LOOK AT THE IMAGE to identify available empty spaces
- Check which directions are blocked by walls or other objects
- If an object is against a wall on one side, you CANNOT place anything further in that wall's direction
- Prefer placing new objects in visible empty areas"""

EXPECTED_RELATIONS_OUTPUT_FORMAT_PROMPT = """## Output Format
Do your step-by-step reasoning inside the <think></think> block, then return ONLY a valid JSON object (no thinking field) with the expected spatial relations.

{
    "parent_id": null,
    "expected_relations": {
        "relative_relations": [
            {"type": "relation_type", "target_id": reference_object_id}
        ],
        "absolute_relations": ["relation_type"]
    }
}

- `parent_id`: null for main objects (standalone in the room), or the ID of the parent object for sub-objects (placed ON/WITHIN/UNDER another object)."""

EXPECTED_RELATIONS_FORMAT_PROMPT = (
    """## Expected Relations Format

**Relative relation types** (object-to-object, use EXACTLY these names):
- """
    + _format_name_list(DIRECTION_RELATION_NAMES)
    + """: directional relations (edge distance < 1.5m)
- "next_to": very close proximity (edge distance < 0.5m)
- "face_to": object's front faces toward the target
- "on": object is on top of another object
- "under": object is under another object
- "within": object is inside another object

**Absolute relation types** (object-to-room, use EXACTLY these names):
- Wall: """
    + _format_name_list(WALL_RELATION_NAMES)
    + """
- Corner: """
    + _format_name_list(CORNER_RELATION_NAMES)
    + """
- Region: """
    + _format_name_list(REGION_RELATION_NAMES)
    + """
"""
)

MAIN_VS_SUB_OBJECTS_PROMPT = """## Main Objects vs Sub-Objects

**Main objects** are standalone objects that occupy an independent position in the room (e.g., bed, sofa, desk, wardrobe). They have meaningful absolute positions (wall/corner/region).

**Sub-objects** are items placed ON, WITHIN, or UNDER another object, and their position is fully determined by a parent object (e.g., pillow on a bed, book on a shelf, chair under a desk). Sub-objects do NOT have meaningful absolute room-level relations.

- If the object is a **sub-object**: set `"parent_id"` to the ID of the parent object it is placed on/within/under — do NOT include any absolute_relations.
- If the object is a **main object**: set `"parent_id": null` — include appropriate absolute_relations."""

MAIN_VS_SUB_OBJECTS_BRIEF_PROMPT = """Main Objects vs Sub-Objects
- **Sub-objects** (placed ON/WITHIN/UNDER another object, e.g., pillow, book, drawer): set `"parent_id"` to the parent object's ID — do NOT include absolute_relations.
- **Main objects** (standalone in the room): set `"parent_id": null` — include absolute_relations as needed."""

BED_SOFA_SUB_OBJECT_RULE_TEXT = """For ANY sub-object placed on, resting on, or contacting a bed or sofa (e.g., pillow, cushion, blanket, teddy bear, remote control, laptop, book), you MUST ALWAYS use the "within" relation. NEVER use "on" for beds and sofas. Example: a pillow within a bed, a laptop within a sofa."""
BED_SOFA_SUB_OBJECT_RULE_PROMPT = f"- {BED_SOFA_SUB_OBJECT_RULE_TEXT}"

SIBLING_AVOIDANCE_RULE_TEXT = """When placing a sub-object on/within/under a parent object that already has other sub-objects (marked with [parent: ID X] in the object list), you MUST add a directional relation relative to the existing sibling to avoid collision. For example, if an alarm clock (ID 5) [parent: ID 3] is already on a nightstand (ID 3), and you want to place a lamp also on ID 3, you should include both {"type": "on", "target_id": 3} AND {"type": "left", "target_id": 5} (where 5 is the alarm clock's ID)."""
SIBLING_AVOIDANCE_RULE_PROMPT = f"- {SIBLING_AVOIDANCE_RULE_TEXT}"

WALL_RELATION_UNIQUENESS_RULE_TEXT = """Use at most ONE "against xxx wall" relation (do NOT combine e.g. "against east wall" + "against south wall"). If the object is at a corner, use a corner relation instead (e.g., "at southeast corner")."""
WALL_RELATION_UNIQUENESS_RULE_PROMPT = f"- {WALL_RELATION_UNIQUENESS_RULE_TEXT}"

RELATION_GUIDELINES_PROMPT = _join_prompt_blocks(
    """## Guidelines for Relations
- Include at most 3 relations total (prioritize: on/within/under > next_to > directional)
- Include absolute relations ONLY for main objects (against a wall / at a corner / in a region)
- Include relative relations only if the object has a clear spatial relationship to an existing object""",
    WALL_RELATION_UNIQUENESS_RULE_PROMPT,
    BED_SOFA_SUB_OBJECT_RULE_PROMPT,
    SIBLING_AVOIDANCE_RULE_PROMPT,
)

FAILURE_TYPES_AND_SOLUTIONS_PROMPT = """## Failure Types and Solutions

### 1. Instruction Not Followed (Spatial Relation Failed)
The placement model did not correctly follow the spatial relations in the instruction.
- **Solution**: Simplify the instruction - use fewer constraints or different reference objects
- Consider using a different relation type or reference object

### 2. Collision
The object overlapped with existing objects.
- Move the object away from the colliding objects
- Find an empty space in the room
- Consider placing in a different area entirely

### 3. Out of Boundary
The object exceeded room boundaries.
- Move the object inward, away from walls
- Choose a different reference object closer to room center
- Reduce the distance if using directional relations

### 4. Exceeds Ceiling
The object is too high.
- This usually means the object is incorrectly placed "on" something too tall
- Place it on a lower surface or on the ground"""

VISUAL_FAILURE_TYPES_AND_SOLUTIONS_PROMPT = """## Failure Types and Solutions

### 1. Instruction Not Followed (Spatial Relation Failed)
The placement model did not correctly follow the spatial relations in the instruction.
- **Solution**: Simplify the instruction - use fewer constraints or different reference objects
- LOOK at the image to find a better reference object or placement strategy

### 2. Collision
The object overlapped with existing objects.
- LOOK at the image to find empty spaces
- Move the object away from the colliding objects
- Choose a visibly clear area

### 3. Out of Boundary
The object exceeded room boundaries.
- Place the object more toward the room center
- Choose a reference object that's not too close to the boundary

### 4. Exceeds Ceiling
The object is too high.
- Place it on a lower surface or on the ground"""

AVAILABLE_RELATIONS_PROMPT = (
    """## Available Relations (use ONLY these)

Absolute positions:
- """
    + _format_name_list(WALL_RELATION_NAMES)
    + """
- """
    + _format_name_list(CORNER_RELATION_NAMES)
    + """
- """
    + _format_name_list(REGION_RELATION_NAMES)
    + """

Relative relations:
- """
    + _format_name_list(DIRECTION_RELATION_NAMES)
    + """: directional, edge distance < 1.5m
- "next_to": very close proximity (edge distance < 0.5m)
- "face_to": object's front direction points toward the target
- "on": object A is on top of object B
- "within": object A is within object B
- "under": object A is under object B"""
)

RELATION_TYPE_ENUMS_PROMPT = f"""**Relative relation types**: {_format_name_list(RELATIVE_RELATION_NAMES)}
**Absolute relation types**: {_format_name_list(ABSOLUTE_RELATION_NAMES)}"""

EDIT_RELATION_DEFINITIONS_PROMPT = (
    """## Relation Definitions
**Absolute relations** (object-to-room):
- Wall: """
    + _format_name_list(WALL_RELATION_NAMES)
    + """
- Corner: """
    + _format_name_list(CORNER_RELATION_NAMES)
    + """
- Region: """
    + _format_name_list(REGION_RELATION_NAMES)
    + """

**Relative relations** (object-to-object):
- """
    + _format_name_list(DIRECTION_RELATION_NAMES)
    + """: directional relation, edge distance < 1.5m
- next_to: very close proximity (edge distance < 0.5m)
- face_to: object's front direction points toward the target
- on: object A is on top of object B (bottom of A touches top of B)
- within: object A is within object B (within B's 3D bounding box), e.g. a book within a shelf, or a pillow within a bed
- under: object A is under object B (center of A is within B's XY projection, below B's top)"""
)

EDIT_GUIDELINES_PROMPT = f"""## Guidelines
1. If user specifies placement, reflect it in expected_relations
2. If no specific placement is given, determine a reasonable location based on scene context
3. Estimated sizes should be in meters with two decimal places (e.g., [0.45, 0.45, 0.50])
4. Consider the object's function and typical placement patterns
5. Ensure the placement won't cause collisions with existing objects
6. {BED_SOFA_SUB_OBJECT_RULE_TEXT}
7. {SIBLING_AVOIDANCE_RULE_TEXT}
8. Distinguish main objects from sub-objects:
   - **Sub-objects** (placed ON/WITHIN/UNDER another object): set `"parent_id"` to the parent object's ID — do NOT include absolute_relations.
   - **Main objects** (standalone in the room): set `"parent_id": null` — include absolute_relations as needed.
9. {WALL_RELATION_UNIQUENESS_RULE_TEXT}"""

VALIDATION_CRITICAL_RULES_PROMPT = _join_prompt_blocks(
    """## CRITICAL Rules
- AVOID the failed direction/area - try a completely different placement strategy
- Ensure the new placement maintains functional sense for the object type""",
    WALL_RELATION_UNIQUENESS_RULE_PROMPT,
    BED_SOFA_SUB_OBJECT_RULE_PROMPT,
)

VALIDATION_OUTPUT_FORMAT_PROMPT = _join_prompt_blocks(
    """## Output Format
Do your analysis of the failure inside the <think></think> block, then return ONLY a valid JSON object (no thinking field) with the NEW expected spatial relations.

{
    "parent_id": null,
    "expected_relations": {
        "relative_relations": [
            {"type": "relation_type", "target_id": reference_object_id}
        ],
        "absolute_relations": ["relation_type"]
    }
}

- `parent_id`: null for main objects, or the ID of the parent object for sub-objects.""",
    RELATION_TYPE_ENUMS_PROMPT,
    BED_SOFA_SUB_OBJECT_RULE_TEXT,
    MAIN_VS_SUB_OBJECTS_BRIEF_PROMPT,
)

VISUAL_VALIDATION_OUTPUT_FORMAT_PROMPT = _join_prompt_blocks(
    """## Output Format
Do your visual analysis inside the <think></think> block, then return ONLY a valid JSON object (no thinking field) with the NEW expected spatial relations.
The placement instruction will be generated automatically — do NOT include an "instruction" field.

{
    "parent_id": null,
    "expected_relations": {
        "relative_relations": [
            {"type": "relation_type", "target_id": reference_object_id}
        ],
        "absolute_relations": ["relation_type"]
    }
}

- `parent_id`: null for main objects, or the ID of the parent object for sub-objects.""",
    RELATION_TYPE_ENUMS_PROMPT,
    BED_SOFA_SUB_OBJECT_RULE_TEXT,
    MAIN_VS_SUB_OBJECTS_BRIEF_PROMPT,
)

COMMON_SCENE_CONTEXT_USER_BLOCK = """Scene Context:
- Scene Type: {scene_type}
- Scene Description: {scene_description}
- Room Boundary: {boundary}
- Room Height: {scene_height}"""

COMMON_OBJECT_TO_PLACE_USER_BLOCK = """Object to Place:
- ID: {object_id}
- Short Description: {short_desc}
- Size: {object_size} (width, depth, height)
- Category: {object_category}"""


# ==================== 场景规划 Prompt ====================

SCENE_PLANNING_SYSTEM_PROMPT = """You are a professional interior scene planner. Your task is to analyze user requirements and create a detailed scene plan.

## Your Task
-  Understand user's scene requirements
- Expand and refine the scene description with professional knowledge
- Generate a complete object list with appropriate sizes and placement order
- Consider spatial relationships and functional requirements

## Guidelines
- Objects should be ordered by priority (larger/anchor/bottom objects first)
- Estimated sizes should be in meters with two decimal places (e.g., [1.20, 0.60, 0.75]) and realistic for the object type. For beds, the estimated height should include the headboard height, not just the mattress height. A typical bed with headboard is 1.0-1.4 meters tall.
- Include necessary furniture based on room type (e.g., bed for bedroom)
- Consider traffic flow and functionality
- The 'description' field should be specific enough for asset retrieval (include material, color, style)
- Only include floor-standing furniture and objects that can be placed on/within/under other objects.
- Do NOT include: ceiling-mounted or suspended items of any kind, especially anything hanging from the ceiling such as chandeliers, pendant lights, ceiling lamps, ceiling fans with lights, or any other overhead lighting fixtures.
- Do NOT include: wall-mounted items (mirrors, shelves, paintings) or any lighting fixtures. If lighting is needed, rely on floor lamps or table lamps only.
- Use "depends_on" to specify which earlier objects this object's placement depends on (by their IDs). This includes: objects it will be placed ON, WITHIN, or UNDER; objects it must be placed NEXT TO or FACING. Standalone objects with no spatial dependency should omit this field or use an empty list.

## Output Format
Do your step-by-step reasoning about the scene design inside the <think></think> block, then output ONLY a valid JSON object (no thinking field) with the following structure:
{
    "scene_type": "bedroom/livingroom/diningroom/office/...",
    "scene_description": "A detailed description of the target scene",
    "objects": [
        {
            "id": 0,
            "category": "bed/sofa/table/chair/...",
            "description": "detailed description for asset retrieval",
            "estimated_size": [width, depth, height]
        },
        {
            "id": 1,
            "category": "nightstand",
            "description": "...",
            "estimated_size": [0.50, 0.40, 0.55],
            "depends_on": [0]
        },
        ...
    ]
}"""

SCENE_PLANNING_USER_TEMPLATE = """Please create a scene plan for the following requirement:

User Request: {user_input}

Room Boundary: {boundary}
Room Height: {scene_height}

Please analyze the request and generate a complete scene plan in JSON format."""


# ==================== 指令生成 Prompt ====================

INSTRUCTION_GENERATION_SYSTEM_PROMPT = _join_prompt_blocks(
    """You are a professional scene layout assistant. Your task is to generate placement instructions for objects in a scene.""",
    WORLD_COORDINATE_SYSTEM_PROMPT,
    ABSOLUTE_POSITION_DEFINITIONS_PROMPT,
    RELATIVE_RELATION_DEFINITIONS_PROMPT,
    SPACE_AWARENESS_PROMPT,
    """## Guidelines
- First analyze the scene context and existing objects to understand the scene layout
- Calculate available space in each direction before choosing a relation
- Use existing object IDs and short descriptions as references
- Combine absolute and relative relations when appropriate
- Be specific but concise
- Consider the object's function and typical placement patterns""",
    EXPECTED_RELATIONS_OUTPUT_FORMAT_PROMPT,
    EXPECTED_RELATIONS_FORMAT_PROMPT,
    MAIN_VS_SUB_OBJECTS_PROMPT,
    RELATION_GUIDELINES_PROMPT,
    """Examples:
{
    "parent_id": null,
    "expected_relations": {
        "relative_relations": [
            {"type": "next_to", "target_id": 0},
            {"type": "right", "target_id": 0}
        ],
        "absolute_relations": ["against south wall"]
    }
},
{
    "parent_id": 3,
    "expected_relations": {
        "relative_relations": [
            {"type": "on", "target_id": 3}
        ]
    }
},
{
    "parent_id": 3,
    "expected_relations": {
        "relative_relations": [
            {"type": "on", "target_id": 3},
            {"type": "left", "target_id": 5}
        ]
    }
}""",
)

INSTRUCTION_GENERATION_USER_TEMPLATE = _join_prompt_blocks(
    COMMON_SCENE_CONTEXT_USER_BLOCK,
    COMMON_OBJECT_TO_PLACE_USER_BLOCK,
    """Already Placed Objects:
{existing_objects_info}""",
)


# ==================== 视觉增强指令生成 Prompt ====================

VISUAL_INSTRUCTION_GENERATION_SYSTEM_PROMPT = _join_prompt_blocks(
    """You are a professional scene layout assistant with visual understanding capability. You can see the current scene layout image and generate precise placement instructions.""",
    IMAGE_UNDERSTANDING_PROMPT,
    TEXT_VS_IMAGE_PROMPT,
    WORLD_COORDINATE_SYSTEM_PROMPT,
    ABSOLUTE_POSITION_DEFINITIONS_PROMPT,
    RELATIVE_RELATION_DEFINITIONS_PROMPT,
    VISUAL_SPACE_AWARENESS_PROMPT,
    """## Guidelines
- Carefully observe the image to understand the current layout
- Identify empty spaces suitable for the new object
- Reference existing objects by their numeric ID
- Consider the new object's size relative to available space
- Ensure the instruction won't cause collisions""",
    EXPECTED_RELATIONS_OUTPUT_FORMAT_PROMPT,
    EXPECTED_RELATIONS_FORMAT_PROMPT,
    MAIN_VS_SUB_OBJECTS_PROMPT,
    RELATION_GUIDELINES_PROMPT,
    """Examples:
{
    "parent_id": null,
    "expected_relations": {
        "relative_relations": [
            {"type": "next_to", "target_id": 0},
            {"type": "left", "target_id": 0}
        ],
        "absolute_relations": ["against south wall"]
    }
},
{
    "parent_id": 2,
    "expected_relations": {
        "relative_relations": [
            {"type": "within", "target_id": 2}
        ]
    }
}""",
)

VISUAL_INSTRUCTION_GENERATION_USER_TEMPLATE = _join_prompt_blocks(
    COMMON_SCENE_CONTEXT_USER_BLOCK,
    COMMON_OBJECT_TO_PLACE_USER_BLOCK,
    """Already Placed Objects (IDs should match the image):
{existing_objects_info}""",
)


# ==================== 验证反馈指令重生成 Prompt ====================

VALIDATION_FEEDBACK_SYSTEM_PROMPT = _join_prompt_blocks(
    """You are a professional scene layout assistant. The previous placement attempt failed validation. You need to analyze the failure and generate a better placement instruction.""",
    FAILURE_TYPES_AND_SOLUTIONS_PROMPT,
    WORLD_COORDINATE_SYSTEM_PROMPT,
    AVAILABLE_RELATIONS_PROMPT,
    VALIDATION_CRITICAL_RULES_PROMPT,
    VALIDATION_OUTPUT_FORMAT_PROMPT,
)

VALIDATION_FEEDBACK_USER_TEMPLATE = _join_prompt_blocks(
    """The previous placement attempt failed validation. Please determine new spatial relations to try.""",
    COMMON_SCENE_CONTEXT_USER_BLOCK,
    COMMON_OBJECT_TO_PLACE_USER_BLOCK,
    """Already Placed Objects:
{existing_objects_info}""",
    """Previous Instruction (FAILED): {previous_instruction}""",
    """Validation Failure Details:
{failure_details}""",
    """Please analyze the failure and output NEW expected_relations that avoid these issues.""",
)

VISUAL_VALIDATION_FEEDBACK_SYSTEM_PROMPT = _join_prompt_blocks(
    """You are a professional scene layout assistant with visual understanding capability. The previous placement attempt failed validation. Look at the scene image and generate a better placement instruction.""",
    IMAGE_UNDERSTANDING_PROMPT,
    TEXT_VS_IMAGE_WITH_FAILURE_PROMPT,
    VISUAL_FAILURE_TYPES_AND_SOLUTIONS_PROMPT,
    AVAILABLE_RELATIONS_PROMPT,
    VISUAL_VALIDATION_OUTPUT_FORMAT_PROMPT,
)

VISUAL_VALIDATION_FEEDBACK_USER_TEMPLATE = _join_prompt_blocks(
    """Look at the scene image. The previous placement attempt failed. Please determine new spatial relations to try.""",
    COMMON_SCENE_CONTEXT_USER_BLOCK,
    COMMON_OBJECT_TO_PLACE_USER_BLOCK,
    """Already Placed Objects (IDs should match the image):
{existing_objects_info}""",
    """Previous Instruction (FAILED): {previous_instruction}""",
    """Validation Failure Details:
{failure_details}""",
    """Looking at the image, generate a NEW placement instruction.""",
)


# ==================== 编辑模式 Prompt ====================

EDIT_INSTRUCTION_SYSTEM_PROMPT = _join_prompt_blocks(
    """You are a professional scene layout assistant. The user wants to ADD a new object to an existing scene.""",
    """## Your Task
1. Understand what object the user wants to add
2. Generate appropriate object description for asset retrieval
3. Determine the expected spatial relations for placement (NOT the full instruction)""",
    WORLD_COORDINATE_SYSTEM_PROMPT,
    EDIT_RELATION_DEFINITIONS_PROMPT,
    SPACE_AWARENESS_PROMPT,
    EDIT_GUIDELINES_PROMPT,
    """## Output Format
Do your step-by-step reasoning about what the user wants and where to place it inside the <think></think> block, then return ONLY a valid JSON object (no thinking field) with object details AND expected spatial relations:
{
    "parent_id": null,
    "object_description": "detailed description for asset retrieval (include material, color, style)",
    "category": "furniture category",
    "estimated_size": [width, depth, height],
    "expected_relations": {
        "relative_relations": [
            {"type": "relation_type", "target_id": reference_object_id}
        ],
        "absolute_relations": ["relation_type"]
    }
}

- `parent_id`: null for main objects (standalone in the room), or the ID of the parent object for sub-objects (placed ON/WITHIN/UNDER another object).""",
    """Examples:
{
    "parent_id": null,
    "object_description": "modern floor lamp with black metal stand and white shade",
    "category": "floor_lamp",
    "estimated_size": [0.40, 0.40, 1.60],
    "expected_relations": {
        "relative_relations": [
            {"type": "next_to", "target_id": 2},
            {"type": "right", "target_id": 2}
        ],
        "absolute_relations": ["against east wall"]
    }
},
{
    "parent_id": 2,
    "object_description": "small beige throw pillow with soft fabric texture",
    "category": "pillow",
    "estimated_size": [0.45, 0.45, 0.18],
    "expected_relations": {
        "relative_relations": [
            {"type": "within", "target_id": 2}
        ]
    }
}""",
)

EDIT_INSTRUCTION_USER_TEMPLATE = _join_prompt_blocks(
    """The user wants to modify the scene:

User Request: {user_input}""",
    COMMON_SCENE_CONTEXT_USER_BLOCK.replace("Scene Context:", "Current Scene:"),
    """Existing Objects:
{existing_objects_info}""",
    """Please analyze the user's request and generate the object description with expected spatial relations in JSON format.""",
)


# ==================== 结果评估 Prompt ====================

EVALUATION_SYSTEM_PROMPT = """You are a professional scene layout evaluator. Your task is to evaluate whether a placement result is satisfactory.

## Evaluation Criteria
1. Does the placement follow the instruction?
2. Is there any collision with existing objects?
3. Is the object within the room boundary?
4. Does the placement make functional sense?
5. Is the overall layout aesthetically pleasing?

## Output Format
Return a valid JSON object:
{
    "satisfactory": true/false,
    "score": 0-100,
    "issues": ["issue 1", "issue 2", ...],
    "suggestions": ["suggestion 1", "suggestion 2", ...],
    "retry_instruction": "new instruction if retry is needed, or null"
}"""

EVALUATION_USER_TEMPLATE = """Please evaluate the following placement result:

Instruction: {instruction}

Object Placed:
- Description: {object_description}
- Size: {object_size}
- Predicted Position: center={center}, rotation={rotation}

Scene Context:
- Boundary: {boundary}
- Existing Objects: {existing_objects_info}

Validation Results:
- Collision Detected: {has_collision}
- Within Boundary: {within_boundary}
- Instruction Compliance: {instruction_compliance}

Please evaluate this placement."""
