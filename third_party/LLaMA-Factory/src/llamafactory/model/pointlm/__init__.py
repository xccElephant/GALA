from .inference import GenerationConfig, InferenceResult, PointLMInference
from .pointlm_qwen3 import PointLMQwen3Config, PointLMQwen3ForCausalLM

__all__ = [
    "GenerationConfig",
    "InferenceResult",
    "PointLMInference",
    "PointLMQwen3Config",
    "PointLMQwen3ForCausalLM",
    "register",
]


def register() -> None:
    """向 transformers Auto 系列注册 PointLM 自定义架构。"""
    from transformers import AutoConfig, AutoModelForCausalLM

    AutoConfig.register("pointlm_qwen3", PointLMQwen3Config)
    AutoModelForCausalLM.register(PointLMQwen3Config, PointLMQwen3ForCausalLM)
