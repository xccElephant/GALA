import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "baselines" / "gpt_oss_autoregressive_layout.py"
SPEC = importlib.util.spec_from_file_location("gpt_oss_autoregressive_layout", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules["gpt_oss_autoregressive_layout"] = module
SPEC.loader.exec_module(module)


def test_system_prompt_matches_test_data_coordinate_contract():
    assert "You are a scene layout assistant." in module.SYSTEM_PROMPT
    assert "X: right, Y: forward, Z: up. Ground plane: XY." in module.SYSTEM_PROMPT
    assert "Yaw: rotation around Z, 0\u00b0 points -Y, positive counterclockwise." in module.SYSTEM_PROMPT
    assert "The new object must rest on a supporting surface." in module.SYSTEM_PROMPT
    assert "Return one list: [x, y, z, roll, pitch, yaw]" in module.SYSTEM_PROMPT


def test_parse_pose_extracts_first_json_pose_from_text():
    assert module.parse_pose("final\n[1, 2, 0.5, 0, 0, 90]") == [1.0, 2.0, 0.5, 0.0, 0.0, 90.0]


def test_parse_pose_skips_non_pose_arrays_in_reasoning_text():
    response = (
        "analysis table center [1.575, 2.5, 0.275] and boundary [0, 0]. "
        "final answer [0.555, 1.315, 0.44, 0, 0, 90]"
    )

    assert module.parse_pose(response) == [0.555, 1.315, 0.44, 0.0, 0.0, 90.0]


def test_build_user_prompt_uses_current_autoregressive_scene_state():
    scene_info = {
        "scene_id": "scene",
        "scene_type": "LivingRoom",
        "boundary": [[0, 0], [4, 0], [4, 3], [0, 3]],
        "scene_height": 2.8,
    }
    placed_objects = [
        {
            "id": 0,
            "short_desc": "sofa",
            "category": "sofa",
            "center": [1.0, 1.0, 0.4],
            "rotation": [0.0, 0.0, 0.0],
            "size": [2.0, 0.8, 0.8],
            "jid": "sofa-jid",
        }
    ]
    task = {
        "id": 1,
        "new_object_id": 1,
        "short_desc": "coffee table",
        "category": "table",
        "size": [1.0, 0.6, 0.4],
        "instruction": "place the table in front of the sofa (id 0).",
        "jid": "table-jid",
    }

    prompt = module.build_user_prompt(scene_info, placed_objects, task)

    assert '"existing_objects":[{"id":0' in prompt
    assert '"center":[1.0,1.0,0.4]' in prompt
    assert '"new_object":{"id":1' in prompt
    assert "place the table in front of the sofa" in prompt


class FakeCompletion:
    def __init__(self, text):
        self.text = text
        self.token_ids = []


class FakeGeneration:
    def __init__(self, text):
        self.outputs = [FakeCompletion(text)]


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompts, sampling_params):
        self.prompts.append(prompts[0])
        return [FakeGeneration(self.responses.pop(0))]


class FakeTokenizer:
    def __init__(self):
        self.chat_kwargs = []

    def apply_chat_template(self, messages, **kwargs):
        self.chat_kwargs.append(kwargs)
        return messages[-1]["content"]

    def decode(self, token_ids, skip_special_tokens=False):
        return ""


def test_infer_one_pose_retries_with_stricter_prompt_after_parse_failure():
    llm = FakeLLM(["analysis no valid pose", "[1, 2, 0.2, 0, 0, 45]"])
    tokenizer = FakeTokenizer()

    pose, final_text, _ = module.infer_one_pose(
        llm,
        tokenizer,
        sampling_params=None,
        user_prompt='{"scene_id":"scene"}',
        enable_thinking=False,
        reasoning_effort="low",
        max_retries=1,
    )

    assert pose == [1.0, 2.0, 0.2, 0.0, 0.0, 45.0]
    assert final_text == "[1, 2, 0.2, 0, 0, 45]"
    assert len(llm.prompts) == 2
    assert "Previous response was invalid" in llm.prompts[1]
    assert "Return exactly one JSON list" in llm.prompts[1]


def test_object_from_task_and_pose_matches_visualizer_schema():
    task = {
        "short_desc": "coffee table",
        "category": "table",
        "size": [1.0, 0.6, 0.4],
        "instruction": "place the table in front of the sofa.",
        "jid": "table-jid",
    }

    obj = module.object_from_task_and_pose(task, [1, 2, 0.2, 0, 0, 45], object_id=3)

    assert obj == {
        "id": 3,
        "short_desc": "coffee table",
        "category": "table",
        "center": [1.0, 2.0, 0.2],
        "size": [1.0, 0.6, 0.4],
        "rotation": [0.0, 0.0, 45.0],
        "jid": "table-jid",
        "instruction": "place the table in front of the sofa.",
    }
