# Copyright (c) Manycore Tech Inc. and affiliates.
# All rights reserved.

from torch import nn
from transformers import Qwen3ForCausalLM, Qwen3Model
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from .pointlm_mixin import PointLMMixin


class PointLMQwen3Config(Qwen3Config):
    """
    Configuration class for PointLM with Qwen3 backbone.

    Qwen3 improvements over Qwen2:
    - Uses QK-Norm (query-key normalization)
    - Removes attention bias (attention_bias=False)
    - Uses head_dim configuration
    """
    model_type = "pointlm_qwen3"


class PointLMQwen3ForCausalLM(PointLMMixin, Qwen3ForCausalLM):
    """
    PointLM model based on Qwen3 with point cloud encoding capability.

    Inherits point cloud processing logic from PointLMMixin.
    """
    config_class = PointLMQwen3Config

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize point cloud modules from mixin
        self._init_point_cloud_modules(config)

        # Initialize weights and apply final processing
        self.post_init()


