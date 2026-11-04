"""
PointLM 独立推理接口

不依赖 LLaMA-Factory 的 ChatModel / HuggingfaceEngine，
直接加载模型 + tokenizer，自行构建 prompt 和调用 generate。

Usage:
    from llamafactory.model.pointlm import PointLMInference, GenerationConfig

    model = PointLMInference(model_path="/path/to/model")
    model.load()

    result = model.generate(
        prompt="<floor_pointcloud>\\n描述这个场景",
        separate_point_cloud_paths=["floor.ply"],
    )
    print(result.response)
"""

import copy
import traceback
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig as HFGenerationConfig

from ...data.pcd_preprocess import make_random_noise_pcd_batch, preprocess_point_cloud_batch
from .utils import build_chatml_prompt, replace_point_cloud_placeholders


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass
class GenerationConfig:
    """生成配置"""

    max_new_tokens: int = 4096
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 10
    num_beams: int = 1
    do_sample: bool = True
    seed: int = -1


@dataclass
class InferenceResult:
    """推理结果"""

    response: str
    success: bool = True
    error: Optional[str] = None
    scene_point_count: Optional[int] = None
    object_point_count: Optional[int] = None
    pose: Optional[list] = None


class PointLMInference:
    """
    PointLM 独立推理服务

    直接使用 transformers 的 AutoModelForCausalLM / AutoTokenizer 完成推理，
    不经过 LLaMA-Factory 的 ChatModel / HuggingfaceEngine。

    Attributes:
        model_path: 模型路径
        inference_dtype: 推理数据类型 ("bfloat16", "float16", "float32")
        verbose: 是否打印详细日志
    """

    def __init__(
        self,
        model_path: str,
        inference_dtype: str = "bfloat16",
        verbose: bool = False,
    ) -> None:
        self.model_path = model_path
        self.inference_dtype = inference_dtype
        self.verbose = verbose

        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._stop_token_ids: list[int] = []

    def load(self) -> "PointLMInference":
        """加载模型（幂等）。

        Returns:
            self: 支持链式调用，如 model = PointLMInference(...).load()
        """
        if self._loaded:
            return self

        from . import register as register_pointlm

        register_pointlm()

        dtype = _DTYPE_MAP.get(self.inference_dtype.lower(), torch.bfloat16)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="right",
        )

        config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=config,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self._model.eval()
        self._model.requires_grad_(False)

        if torch.cuda.is_available():
            self._model = self._model.cuda()

        im_end_id = self._tokenizer.convert_tokens_to_ids("<|im_end|>")
        eos_id = self._tokenizer.eos_token_id
        self._stop_token_ids = list({eos_id, im_end_id} - {None})

        point_config = getattr(config, "point_config", None) or {}
        self._backbone_type = point_config.get("backbone_type")
        self._point_config = point_config

        self._loaded = True
        if self.verbose:
            total = sum(p.numel() for p in self._model.parameters())
            print(f"[PointLMInference] Loaded model ({total:,} params, dtype={dtype})")

        return self

    def _prepare_pcd_tensor(
        self,
        pcd_paths: list[str],
        pcd_random_noise: bool,
    ) -> Optional[torch.Tensor]:
        """Preprocess point clouds and return the tensor on model device."""
        if not pcd_paths:
            return None
        if not self._point_config or not self._backbone_type:
            raise ValueError(
                f"{self.model_path} config.json has no 'point_config', cannot encode point clouds."
            )
        if pcd_random_noise:
            pcd_tensor = torch.as_tensor(make_random_noise_pcd_batch(len(pcd_paths)))
        else:
            pc = self._point_config
            pcd_tensor = torch.as_tensor(preprocess_point_cloud_batch(
                pcd_paths,
                backbone_type=self._backbone_type,
                utonia_scale=pc.get("utonia_scale", 1.0),
                utonia_grid_size=pc.get("utonia_grid_size", 0.02),
                utonia_normalize_coord=pc.get("utonia_normalize_coord", False),
                utonia_center_shift=pc.get("utonia_center_shift", False),
            ))
        return pcd_tensor.to(self._model.device)

    @staticmethod
    def _apply_seed(seed: int) -> None:
        """Apply the same seed to Python / NumPy / PyTorch RNGs."""
        if seed < 0:
            return
        from transformers import set_seed
        set_seed(seed)

    def _run_generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        prompt_length: int,
        generation_config: "GenerationConfig",
        pcd_tensor: Optional[torch.Tensor],
        num_pcd: int,
    ) -> str:
        """Run a single model.generate() call and decode the response."""
        self._apply_seed(generation_config.seed)

        gen_kwargs: dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "generation_config": HFGenerationConfig(
                max_new_tokens=generation_config.max_new_tokens,
                do_sample=generation_config.do_sample,
                temperature=generation_config.temperature if generation_config.do_sample else None,
                top_p=generation_config.top_p if generation_config.do_sample else None,
                top_k=generation_config.top_k if generation_config.do_sample else None,
                num_beams=generation_config.num_beams,
                eos_token_id=self._stop_token_ids,
                pad_token_id=self._tokenizer.pad_token_id,
            ),
        }

        if pcd_tensor is not None:
            gen_kwargs["separate_point_clouds"] = pcd_tensor
            gen_kwargs["separate_point_clouds_counts"] = torch.tensor(
                [num_pcd], dtype=torch.long, device=self._model.device
            )

        with torch.inference_mode():
            output = self._model.generate(**gen_kwargs)

        response_ids = output[0, prompt_length:]
        return self._tokenizer.decode(response_ids, skip_special_tokens=True)

    def generate(
        self,
        prompt: str,
        scene_point_cloud_path: Optional[str] = None,  # noqa: ARG002  reserved for API compatibility
        object_point_cloud_path: Optional[str] = None,  # noqa: ARG002  reserved for API compatibility
        separate_point_cloud_paths: Optional[list[str]] = None,
        system_prompt: str = "You are a helpful assistant.",
        generation_config: Optional[GenerationConfig] = None,
        pcd_random_noise: bool = False,
    ) -> InferenceResult:
        """生成响应。

        Args:
            prompt: 用户输入，可含占位符（<point_cloud>、<scene_pointcloud> 等）
            scene_point_cloud_path: 保留参数，暂不使用，请改用 separate_point_cloud_paths
            object_point_cloud_path: 保留参数，暂不使用，请改用 separate_point_cloud_paths
            separate_point_cloud_paths: 按占位符顺序排列的点云路径列表
            system_prompt: 系统提示
            generation_config: 生成参数，None 使用默认值
            pcd_random_noise: 用随机噪声替代真实点云（消融实验用）

        Returns:
            InferenceResult
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if generation_config is None:
            generation_config = GenerationConfig()

        try:
            # Seed must be applied before point-cloud preprocessing because
            # preprocessing may perform random subsampling / voxel selection.
            self._apply_seed(generation_config.seed)
            pcd_paths = list(separate_point_cloud_paths) if separate_point_cloud_paths else []

            user_content = replace_point_cloud_placeholders(prompt)
            full_prompt = build_chatml_prompt(user_content, system_prompt)

            if self.verbose:
                pcd_info = f"{len(pcd_paths)} point cloud(s)" if pcd_paths else "no point clouds"
                if pcd_random_noise:
                    pcd_info += " (random noise)"
                print(f"[PointLMInference] generating with {pcd_info}")

            encoding = self._tokenizer(full_prompt, return_tensors="pt")
            input_ids = encoding["input_ids"].to(self._model.device)
            attention_mask = encoding["attention_mask"].to(self._model.device)
            prompt_length = input_ids.shape[1]

            pcd_tensor = self._prepare_pcd_tensor(pcd_paths, pcd_random_noise)
            response = self._run_generate(
                input_ids, attention_mask, prompt_length,
                generation_config, pcd_tensor, len(pcd_paths),
            )
            return InferenceResult(response=response, success=True)

        except Exception as e:
            traceback.print_exc()
            return InferenceResult(response="", success=False, error=str(e))

    def _prefill(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pcd_tensor: Optional[torch.Tensor],
        num_pcd: int,
    ):
        """Run a single prefill forward pass and return (past_key_values, last_logits, expanded_seq_len).

        The returned KV-cache contains all the context including injected
        point-cloud tokens. Callers can deep-copy it for multiple decode runs.
        """
        fwd_kwargs: dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": True,
        }
        if pcd_tensor is not None:
            fwd_kwargs["separate_point_clouds"] = pcd_tensor
            fwd_kwargs["separate_point_clouds_counts"] = torch.tensor(
                [num_pcd], dtype=torch.long, device=self._model.device,
            )

        with torch.inference_mode():
            outputs = self._model(**fwd_kwargs)

        return outputs.past_key_values, outputs.logits

    @staticmethod
    def _clone_cache(cache):
        """Deep-copy a KV-cache so that decode runs are independent."""
        return copy.deepcopy(cache)

    def _decode_from_cache(
        self,
        past_key_values,
        prefill_logits: torch.Tensor,
        cache_seq_len: int,
        generation_config: "GenerationConfig",
    ) -> str:
        """Autoregressively decode from a prefilled KV-cache.

        Manually samples tokens one at a time using the model's forward pass,
        avoiding model.generate() compatibility issues with pre-filled caches.
        """
        if generation_config.seed >= 0:
            from transformers import set_seed
            set_seed(generation_config.seed)

        device = prefill_logits.device
        max_new = generation_config.max_new_tokens
        generated_ids: list[int] = []

        logits = prefill_logits[:, -1, :]
        pos = cache_seq_len

        for _ in range(max_new):
            next_id = self._sample_token(logits, generation_config)

            if next_id in self._stop_token_ids:
                break

            generated_ids.append(next_id)
            next_input = torch.tensor([[next_id]], dtype=torch.long, device=device)
            cache_position = torch.tensor([pos], dtype=torch.long, device=device)

            with torch.inference_mode():
                out = self._model(
                    input_ids=next_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                    cache_position=cache_position,
                )

            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :]
            pos += 1

        return self._tokenizer.decode(generated_ids, skip_special_tokens=True)

    @staticmethod
    def _sample_token(logits: torch.Tensor, gc: "GenerationConfig") -> int:
        """Sample a single token id from logits."""
        if not gc.do_sample:
            return logits.argmax(dim=-1).item()

        if gc.temperature and gc.temperature > 0 and gc.temperature != 1.0:
            logits = logits / gc.temperature

        if gc.top_k and gc.top_k > 0:
            topk_vals, topk_ids = torch.topk(logits, gc.top_k)
            probs = torch.softmax(topk_vals, dim=-1)
            idx = torch.multinomial(probs, 1)
            return topk_ids.gather(-1, idx).item()

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).item()

    def generate_candidates(
        self,
        prompt: str,
        separate_point_cloud_paths: Optional[list[str]] = None,
        system_prompt: str = "You are a helpful assistant.",
        generation_configs: Optional[list[GenerationConfig]] = None,
        pcd_random_noise: bool = False,
    ) -> list[InferenceResult]:
        """为同一个 prompt 生成多个候选响应。

        只做一次 prefill（包含点云编码），然后复用 KV-cache 执行 K 次独立的
        decode，省掉 K-1 次 prefill 的开销。

        Args:
            prompt: 用户输入
            separate_point_cloud_paths: 点云路径列表
            system_prompt: 系统提示
            generation_configs: 每个候选的生成配置列表
            pcd_random_noise: 用随机噪声替代真实点云

        Returns:
            与 generation_configs 等长的 InferenceResult 列表
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if not generation_configs:
            generation_configs = [GenerationConfig()]

        pcd_paths = list(separate_point_cloud_paths) if separate_point_cloud_paths else []

        user_content = replace_point_cloud_placeholders(prompt)
        full_prompt = build_chatml_prompt(user_content, system_prompt)

        encoding = self._tokenizer(full_prompt, return_tensors="pt")
        input_ids = encoding["input_ids"].to(self._model.device)
        attention_mask = encoding["attention_mask"].to(self._model.device)

        pcd_tensor = self._prepare_pcd_tensor(pcd_paths, pcd_random_noise)

        try:
            base_cache, prefill_logits = self._prefill(
                input_ids, attention_mask, pcd_tensor, len(pcd_paths),
            )
            cache_seq_len = base_cache.get_seq_length()
        except Exception as e:
            traceback.print_exc()
            return [
                InferenceResult(response="", success=False, error=f"prefill failed: {e}")
                for _ in generation_configs
            ]

        results: list[InferenceResult] = []
        for i, gc in enumerate(generation_configs):
            try:
                cache = self._clone_cache(base_cache) if i < len(generation_configs) - 1 else base_cache
                response = self._decode_from_cache(
                    cache, prefill_logits, cache_seq_len, gc,
                )
                results.append(InferenceResult(response=response, success=True))
            except Exception as e:
                traceback.print_exc()
                results.append(InferenceResult(response="", success=False, error=str(e)))

        return results

    def generate_batch(
        self,
        prompts: list[str],
        separate_point_cloud_paths_list: Optional[list[Optional[list[str]]]] = None,
        system_prompt: str = "You are a helpful assistant.",
        generation_config: Optional[GenerationConfig] = None,
        pcd_random_noise: bool = False,
    ) -> list[InferenceResult]:
        """Batch inference: generate responses for multiple prompts in one call.

        Args:
            prompts: List of user prompts (may contain point cloud placeholders).
            separate_point_cloud_paths_list: Per-sample point cloud path lists,
                aligned with *prompts*. ``None`` entries mean no point clouds.
            system_prompt: Shared system prompt for all samples.
            generation_config: Generation parameters (shared across the batch).
            pcd_random_noise: Replace real point clouds with random noise.

        Returns:
            List of InferenceResult, one per prompt.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        batch_size = len(prompts)
        if batch_size == 0:
            return []

        if generation_config is None:
            generation_config = GenerationConfig()

        if separate_point_cloud_paths_list is None:
            separate_point_cloud_paths_list = [None] * batch_size
        if len(separate_point_cloud_paths_list) != batch_size:
            raise ValueError(
                f"prompts ({batch_size}) and separate_point_cloud_paths_list "
                f"({len(separate_point_cloud_paths_list)}) length mismatch."
            )

        try:
            full_prompts: list[str] = []
            all_pcd_paths: list[str] = []
            counts: list[int] = []

            for i, prompt in enumerate(prompts):
                user_content = replace_point_cloud_placeholders(prompt)
                full_prompts.append(build_chatml_prompt(user_content, system_prompt))

                pcd_paths = list(separate_point_cloud_paths_list[i]) if separate_point_cloud_paths_list[i] else []
                all_pcd_paths.extend(pcd_paths)
                counts.append(len(pcd_paths))

            if generation_config.seed >= 0:
                from transformers import set_seed
                set_seed(generation_config.seed)

            orig_padding_side = self._tokenizer.padding_side
            self._tokenizer.padding_side = "left"
            try:
                encoding = self._tokenizer(
                    full_prompts, return_tensors="pt", padding=True,
                )
            finally:
                self._tokenizer.padding_side = orig_padding_side

            input_ids = encoding["input_ids"].to(self._model.device)
            attention_mask = encoding["attention_mask"].to(self._model.device)
            padded_length = input_ids.shape[1]

            pcd_tensor = self._prepare_pcd_tensor(all_pcd_paths, pcd_random_noise)
            counts_tensor = torch.tensor(counts, dtype=torch.long, device=self._model.device)

            gen_kwargs: dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "generation_config": HFGenerationConfig(
                    max_new_tokens=generation_config.max_new_tokens,
                    do_sample=generation_config.do_sample,
                    temperature=generation_config.temperature if generation_config.do_sample else None,
                    top_p=generation_config.top_p if generation_config.do_sample else None,
                    top_k=generation_config.top_k if generation_config.do_sample else None,
                    num_beams=generation_config.num_beams,
                    eos_token_id=self._stop_token_ids,
                    pad_token_id=self._tokenizer.pad_token_id,
                ),
            }

            if pcd_tensor is not None:
                gen_kwargs["separate_point_clouds"] = pcd_tensor
                gen_kwargs["separate_point_clouds_counts"] = counts_tensor

            if self.verbose:
                total_pcds = sum(counts)
                print(
                    f"[PointLMInference] batch generate: {batch_size} prompts, "
                    f"{total_pcds} point clouds total"
                )

            with torch.inference_mode():
                outputs = self._model.generate(**gen_kwargs)

            results: list[InferenceResult] = []
            for i in range(batch_size):
                response_ids = outputs[i, padded_length:]
                response = self._tokenizer.decode(response_ids, skip_special_tokens=True)
                results.append(InferenceResult(response=response, success=True))

            return results

        except Exception as e:
            traceback.print_exc()
            return [
                InferenceResult(response="", success=False, error=str(e))
                for _ in range(batch_size)
            ]

    @property
    def is_loaded(self) -> bool:
        """模型是否已加载。"""
        return self._loaded

    @property
    def supports_scene_point_cloud(self) -> bool:
        return self._loaded

    @property
    def supports_object_point_cloud(self) -> bool:
        return self._loaded

    @property
    def object_backbone_type(self) -> Optional[str]:
        if not self._loaded:
            return None
        return self._backbone_type
