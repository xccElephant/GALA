"""
Utility helpers shared by PointLM modules.
"""

import torch

from ...extras.constants import POINT_CLOUD_PLACEHOLDERS, POINT_E_TOKEN, POINT_PAD_TOKEN, POINT_S_TOKEN



def is_rank0_process() -> bool:
    """Return True on rank 0 (or non-distributed runs)."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def replace_point_cloud_placeholders(prompt: str) -> str:
    """将所有点云占位符替换为 <|point_start|><|point_pad|><|point_end|>。"""
    for placeholder in POINT_CLOUD_PLACEHOLDERS:
        prompt = prompt.replace(
            placeholder,
            f"{POINT_S_TOKEN}{POINT_PAD_TOKEN}{POINT_E_TOKEN}",
        )
    return prompt


def build_chatml_prompt(
    user_content: str,
    system_prompt: str = "You are a helpful assistant.",
) -> str:
    """构建 Qwen3 ChatML 格式 prompt,不含 assistant 回复."""
    parts = []
    if system_prompt:
        parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{user_content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)

