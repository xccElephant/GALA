"""
LLM 调用模块

统一管理 LLM 配置、客户端创建和调用，支持：
1. 多个 provider（google-genai, openai, anthropic, local transformers）
2. 每个任务使用不同的模型
3. 文本和视觉调用
4. 流式输出和重试机制
"""

import base64
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

# 配置文件路径
CONFIG_DIR = Path(__file__).parent.parent / "config"
LLM_KEY_FILE = CONFIG_DIR / "llm_key.json"
DEFAULT_MAX_TOKENS = 8192


class LLMManager:
    """LLM 管理器（单例），管理配置和客户端缓存"""
    
    _instance: Optional["LLMManager"] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._config: Dict[str, Any] = {}
        self._clients: Dict[str, Tuple[Any, str, str, Optional[str]]] = {}
        self._local_models: Dict[str, Tuple[Any, Any, bool]] = {}
        self._load_config()
        self._initialized = True
    
    def _load_config(self):
        if not LLM_KEY_FILE.exists():
            raise FileNotFoundError(f"LLM config not found: {LLM_KEY_FILE}")
        with open(LLM_KEY_FILE, "r", encoding="utf-8") as f:
            self._config = json.load(f)
    
    def get_client(self, task: str = "default") -> Tuple[Any, str, str, Optional[str]]:
        """获取客户端: (client, model, provider, sdk)"""
        if task in self._clients:
            return self._clients[task]
        
        model_spec = self._config["models"].get(task) or self._config["models"].get("default")
        if not model_spec:
            raise ValueError(f"No model for task '{task}'")
        
        provider, model = model_spec.split("/", 1) if "/" in model_spec else (list(self._config["providers"].keys())[0], model_spec)
        provider_config = self._config["providers"][provider]
        sdk = provider_config.get("sdk")
        
        client = self._create_client(provider_config, model)
        self._clients[task] = (client, model, provider, sdk)
        return self._clients[task]
    
    def _create_client(self, config: Dict[str, Any], model: str) -> Any:
        sdk = config.get("sdk")
        if sdk == "google-genai":
            from google import genai
            return genai.Client(api_key=config["api_key"])
        elif sdk == "azure-anthropic":
            return {
                "api_key": config["api_key"],
                "base_url": config["base_url"].rstrip("/"),
                "anthropic_version": config["anthropic_version"],
                "timeout": int(config["timeout"]) if "timeout" in config else 120,
                "max_tokens": int(config["max_tokens"]) if "max_tokens" in config else DEFAULT_MAX_TOKENS,
            }
        elif sdk == "azure-openai-chat":
            return {
                "api_key": config["api_key"],
                "base_url": config["base_url"].rstrip("/"),
                "timeout": int(config["timeout"]) if "timeout" in config else 120,
                "max_completion_tokens": int(config["max_completion_tokens"]) if "max_completion_tokens" in config else DEFAULT_MAX_TOKENS,
            }
        elif sdk == "azure-openai-responses":
            return {
                "api_key": config["api_key"],
                "url": config["base_url"],
                "timeout": int(config["timeout"]) if "timeout" in config else 120,
                "max_output_tokens": int(config["max_output_tokens"]) if "max_output_tokens" in config else DEFAULT_MAX_TOKENS,
                "supports_temperature": bool(config.get("supports_temperature", True)),
            }
        elif sdk == "transformers":
            return self._load_local_model(config.get("model_path") or model, config)
        else:
            from openai import OpenAI
            kwargs = {"api_key": config["api_key"]}
            if config.get("base_url"):
                kwargs["base_url"] = config["base_url"]
            return OpenAI(**kwargs)
    
    def _load_local_model(self, model_path: str, config: Dict[str, Any]) -> Tuple[Any, Any, bool]:
        if model_path in self._local_models:
            return self._local_models[model_path]
        
        print(f"[LLM] Loading model: {model_path}")
        import torch
        from transformers import AutoProcessor, AutoTokenizer
        
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(config.get("dtype", "bfloat16"), torch.bfloat16)
        load_kwargs = {"dtype": dtype, "device_map": config.get("device_map", "auto")}
        if config.get("use_flash_attention", True):
            load_kwargs["attn_implementation"] = "flash_attention_2"
        
        model_lower = model_path.lower()
        # 是否视觉模型：优先用配置里的显式 is_vl 标志（名字猜测不可靠），否则按名字回退判断
        if "is_vl" in config:
            is_vl = bool(config["is_vl"])
        else:
            is_vl = any(kw in model_lower for kw in ["qwen3-vl", "qwen3vl", "qwen2-vl", "-vl-", "vision"])
        
        if "qwen3-vl" in model_lower or "qwen3vl" in model_lower:
            from transformers import Qwen3VLForConditionalGeneration
            model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
        elif is_vl:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
        else:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        
        processor = AutoProcessor.from_pretrained(model_path) if is_vl else AutoTokenizer.from_pretrained(model_path)
        self._local_models[model_path] = (model, processor, is_vl)
        return self._local_models[model_path]


_manager: Optional[LLMManager] = None

def _get_manager() -> LLMManager:
    global _manager
    if _manager is None:
        _manager = LLMManager()
    return _manager


# ========== 调用实现 ==========

def _call_google(client, model: str, system_prompt: str, user_prompt: str, temperature: float, stream: bool) -> str:
    from google.genai import types
    config = types.GenerateContentConfig(system_instruction=system_prompt, temperature=temperature)
    
    if stream:
        return _collect_stream(
            client.models.generate_content_stream(model=model, contents=user_prompt, config=config),
            lambda c: c.text
        )
    return client.models.generate_content(model=model, contents=user_prompt, config=config).text


def _call_openai(client, model: str, system_prompt: str, user_prompt: str, temperature: float, stream: bool) -> str:
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    
    if stream:
        return _collect_stream(
            client.chat.completions.create(model=model, messages=messages, temperature=temperature, stream=True),
            lambda c: c.choices[0].delta.content if c.choices and c.choices[0].delta.content else ""
        )
    return client.chat.completions.create(model=model, messages=messages, temperature=temperature).choices[0].message.content

# Azure Anthropic 的完整响应是 {"content": [{"type": "text", "text": ...}, ...]}。
def _extract_azure_anthropic_text(response_json: Dict[str, Any]) -> str:
    content_blocks = response_json.get("content")
    if not isinstance(content_blocks, list):
        raise ValueError(f"Invalid Azure Anthropic response: {response_json}")
    texts: List[str] = []
    for block in content_blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if text:
                texts.append(text)
    if not texts:
        raise ValueError(f"No text content in Azure Anthropic response: {response_json}")
    return "".join(texts)

# OpenAI 兼容接口的 content 可能是纯字符串，也可能是 content block 列表。
# 这个函数用于解析完整响应里的 message.content，遇到异常结构时直接报错。
def _extract_openai_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError(f"Invalid OpenAI content: {content}")

    texts: List[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if text:
            texts.append(text)

    if not texts:
        raise ValueError(f"No text content in OpenAI response: {content}")
    return "".join(texts)

# 从完整的 OpenAI 风格响应里取出 choices[0].message.content，再交给通用 content 解析器。
def _extract_openai_text(response_json: Dict[str, Any]) -> str:
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"Invalid OpenAI response: {response_json}")

    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError(f"Invalid OpenAI message: {response_json}")
    return _extract_openai_content_text(message.get("content"))

# 流式响应的 delta.content 经常缺失或只带部分字段，因此这里宽松处理，拿不到文本时返回空串。
def _extract_openai_delta_text(delta: Dict[str, Any]) -> str:
    content = delta.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    texts: List[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if text:
            texts.append(text)
    return "".join(texts)


# Responses API 的输出在 output[].content[]，常见文本块类型是 output_text。
def _extract_responses_text(response_json: Dict[str, Any]) -> str:
    output_text = response_json.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text

    output = response_json.get("output")
    if not isinstance(output, list):
        raise ValueError(f"Invalid Responses API response: {response_json}")

    texts: List[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") not in {"output_text", "text"}:
                continue
            text = block.get("text")
            if text:
                texts.append(text)

    if not texts:
        raise ValueError(f"No text content in Responses API response: {response_json}")
    return "".join(texts)


def _raise_api_error(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}") from exc


def _call_azure_anthropic(
    client: Dict[str, Any],
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    stream: bool,
) -> str:
    url = f"{client['base_url']}/anthropic/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": client["api_key"],
        "anthropic-version": client["anthropic_version"],
    }
    payload = {
        "model": model,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": temperature,
        "max_tokens": client["max_tokens"],
    }

    if stream:
        payload["stream"] = True
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=client["timeout"]) as response:
            _raise_api_error(response)
            result = ""
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data:
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "content_block_delta":
                    delta = event.get("delta")
                    if isinstance(delta, dict):
                        text = delta.get("text")
                        if text:
                            print(text, end="", flush=True)
                            result += text
                elif event.get("type") == "message_stop":
                    break
            print()
            return result

    response = requests.post(url, headers=headers, json=payload, timeout=client["timeout"])
    _raise_api_error(response)
    return _extract_azure_anthropic_text(response.json())


def _call_azure_openai_chat(
    client: Dict[str, Any],
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    stream: bool,
) -> str:
    url = f"{client['base_url']}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "api-key": client["api_key"],
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_completion_tokens": client["max_completion_tokens"],
    }

    if stream:
        payload["stream"] = True
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=client["timeout"]) as response:
            _raise_api_error(response)
            result = ""
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choices = event.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                delta = choices[0].get("delta")
                if not isinstance(delta, dict):
                    continue
                text = _extract_openai_delta_text(delta)
                if text:
                    print(text, end="", flush=True)
                    result += text
            print()
            return result

    response = requests.post(url, headers=headers, json=payload, timeout=client["timeout"])
    _raise_api_error(response)
    return _extract_openai_text(response.json())


def _call_azure_openai_responses(
    client: Dict[str, Any],
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    stream: bool,
) -> str:
    headers = {
        "Content-Type": "application/json",
        "api-key": client["api_key"],
    }
    payload = {
        "model": model,
        "instructions": system_prompt,
        "input": user_prompt,
        "max_output_tokens": client["max_output_tokens"],
    }
    if client["supports_temperature"]:
        payload["temperature"] = temperature

    if stream:
        payload["stream"] = True
        with requests.post(client["url"], headers=headers, json=payload, stream=True, timeout=client["timeout"]) as response:
            _raise_api_error(response)
            result = ""
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "response.output_text.delta":
                    text = event.get("delta")
                    if text:
                        print(text, end="", flush=True)
                        result += text
                elif event.get("type") == "response.completed":
                    break
            print()
            return result

    response = requests.post(client["url"], headers=headers, json=payload, timeout=client["timeout"])
    _raise_api_error(response)
    return _extract_responses_text(response.json())


def _call_azure_openai_responses_vision(
    client: Dict[str, Any],
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_base64: str,
    temperature: float,
) -> str:
    headers = {
        "Content-Type": "application/json",
        "api-key": client["api_key"],
    }
    payload = {
        "model": model,
        "instructions": system_prompt,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{image_base64}"},
                ],
            }
        ],
        "max_output_tokens": client["max_output_tokens"],
    }
    if client["supports_temperature"]:
        payload["temperature"] = temperature

    response = requests.post(client["url"], headers=headers, json=payload, timeout=client["timeout"])
    _raise_api_error(response)
    return _extract_responses_text(response.json())

def _call_local(client: Tuple[Any, Any, bool], system_prompt: str, user_prompt: str, temperature: float, stream: bool, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    model, processor, is_vl = client
    
    if is_vl:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]
    else:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    
    try:
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", enable_thinking=True).to(model.device)
    except TypeError:
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
    
    if stream:
        from transformers import TextIteratorStreamer
        streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
        thread = threading.Thread(target=model.generate, kwargs={**inputs, "max_new_tokens": max_tokens, "temperature": temperature, "do_sample": temperature > 0, "streamer": streamer})
        thread.start()
        result = "".join(text for text in streamer if (print(text, end="", flush=True) or True))
        print()
        thread.join()
        return result
    
    ids = model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature, do_sample=temperature > 0)
    return processor.batch_decode([o[len(i):] for i, o in zip(inputs.input_ids, ids)], skip_special_tokens=True)[0]


def _call_google_vision(client, model: str, system_prompt: str, user_prompt: str, image_base64: str, temperature: float) -> str:
    from google.genai import types
    contents = [user_prompt, types.Part.from_bytes(data=base64.b64decode(image_base64), mime_type="image/png")]
    config = types.GenerateContentConfig(system_instruction=system_prompt, temperature=temperature)
    return client.models.generate_content(model=model, contents=contents, config=config).text


def _call_openai_vision(client, model: str, provider: str, system_prompt: str, user_prompt: str, image_base64: str, temperature: float) -> str:
    if provider == "anthropic":
        # Anthropic 风格
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_base64}},
            {"type": "text", "text": user_prompt},
        ]
    else:
        # OpenAI 风格
        content = [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}", "detail": "high"}},
        ]
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]
    return client.chat.completions.create(model=model, messages=messages, temperature=temperature).choices[0].message.content


def _call_azure_anthropic_vision(
    client: Dict[str, Any],
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_base64: str,
    temperature: float,
) -> str:
    url = f"{client['base_url']}/anthropic/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": client["api_key"],
        "anthropic-version": client["anthropic_version"],
    }
    payload = {
        "model": model,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64,
                        },
                    },
                    {"type": "text", "text": user_prompt},
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": client["max_tokens"],
    }
    response = requests.post(url, headers=headers, json=payload, timeout=client["timeout"])
    _raise_api_error(response)
    return _extract_azure_anthropic_text(response.json())


def _call_azure_openai_chat_vision(
    client: Dict[str, Any],
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_base64: str,
    temperature: float,
) -> str:
    url = f"{client['base_url']}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "api-key": client["api_key"],
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}", "detail": "high"}},
                ],
            },
        ],
        "temperature": temperature,
        "max_completion_tokens": client["max_completion_tokens"],
    }
    response = requests.post(url, headers=headers, json=payload, timeout=client["timeout"])
    _raise_api_error(response)
    return _extract_openai_text(response.json())


def _call_local_vision(client: Tuple[Any, Any, bool], system_prompt: str, user_prompt: str, image_base64: str, temperature: float) -> str:
    model, processor, is_vl = client
    if not is_vl:
        return _call_local(client, system_prompt, user_prompt, temperature, stream=False)
    
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [
            {"type": "image", "image": f"data:image/png;base64,{image_base64}"},
            {"type": "text", "text": user_prompt},
        ]},
    ]
    try:
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", enable_thinking=True).to(model.device)
    except TypeError:
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
    ids = model.generate(**inputs, max_new_tokens=DEFAULT_MAX_TOKENS, temperature=temperature, do_sample=temperature > 0)
    return processor.batch_decode([o[len(i):] for i, o in zip(inputs.input_ids, ids)], skip_special_tokens=True)[0]


def _collect_stream(stream, extract: Callable) -> str:
    result = ""
    for chunk in stream:
        text = extract(chunk)
        if text:
            print(text, end="", flush=True)
            result += text
    print()
    return result


def _retry_call(fn: Callable, max_retries: int, task: str) -> str:
    """带重试的调用"""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            print(f"[LLM] Call failed (attempt {attempt + 1}, task={task}): {e}")
            if attempt == max_retries - 1:
                raise
            error_str = str(e).lower()
            wait = 30 * (attempt + 1) if "429" in error_str or "rate" in error_str else 2 ** attempt
            print(f"[LLM] Waiting {wait}s...")
            time.sleep(wait)
    return ""


# ========== 公开接口 ==========

def call_llm(
    system_prompt: str,
    user_prompt: str,
    task: str = "default",
    temperature: float = 0.7,
    stream: bool = False,
    max_retries: int = 3,
    verbose: bool = False,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """
    调用 LLM
    
    Args:
        system_prompt: 系统提示
        user_prompt: 用户提示
        task: 任务名称，用于选择模型
        temperature: 采样温度
        stream: 是否流式输出
        max_retries: 最大重试次数
        verbose: 是否打印详细日志
        max_tokens: 生成的最大 token 数（目前仅对本地 transformers 模型生效）
    
    Returns:
        模型响应文本
    """
    client, model, provider, sdk = _get_manager().get_client(task)
    
    if verbose:
        print(f"\n[LLM: {task}] model={model}")
        # print(f"\n[SYSTEM] {system_prompt}")
        print(f"\n[USER] {user_prompt}")
    
    def do_call():
        if sdk == "google-genai":
            return _call_google(client, model, system_prompt, user_prompt, temperature, stream)
        elif sdk == "azure-anthropic":
            return _call_azure_anthropic(client, model, system_prompt, user_prompt, temperature, stream)
        elif sdk == "azure-openai-chat":
            return _call_azure_openai_chat(client, model, system_prompt, user_prompt, temperature, stream)
        elif sdk == "azure-openai-responses":
            return _call_azure_openai_responses(client, model, system_prompt, user_prompt, temperature, stream)
        elif sdk == "transformers":
            return _call_local(client, system_prompt, user_prompt, temperature, stream, max_tokens)
        else:
            return _call_openai(client, model, system_prompt, user_prompt, temperature, stream)
    
    result = _retry_call(do_call, max_retries, task)
    
    if verbose and not stream:
        print(f"\n[OUTPUT] {result}")
    
    return result


def call_vlm(
    system_prompt: str,
    user_prompt: str,
    image_base64: str,
    task: str = "default",
    temperature: float = 0.7,
    max_retries: int = 3,
    verbose: bool = False,
) -> str:
    """
    调用支持视觉的 LLM
    
    Args:
        system_prompt: 系统提示
        user_prompt: 用户提示
        image_base64: base64 编码的图像
        task: 任务名称
        temperature: 采样温度
        max_retries: 最大重试次数
        verbose: 是否打印详细日志
    
    Returns:
        模型响应文本
    """
    client, model, provider, sdk = _get_manager().get_client(task)
    
    if verbose:
        print(f"\n[LLM: {task}] model={model} (vision)")
        print(f"\n[IMAGE] {len(image_base64)} bytes")
        # print(f"\n[SYSTEM] {system_prompt}")
        print(f"\n[USER] {user_prompt}")
    def do_call():
        if sdk == "google-genai":
            return _call_google_vision(client, model, system_prompt, user_prompt, image_base64, temperature)
        elif sdk == "azure-anthropic":
            return _call_azure_anthropic_vision(client, model, system_prompt, user_prompt, image_base64, temperature)
        elif sdk == "azure-openai-chat":
            return _call_azure_openai_chat_vision(client, model, system_prompt, user_prompt, image_base64, temperature)
        elif sdk == "azure-openai-responses":
            return _call_azure_openai_responses_vision(client, model, system_prompt, user_prompt, image_base64, temperature)
        elif sdk == "transformers":
            return _call_local_vision(client, system_prompt, user_prompt, image_base64, temperature)
        else:
            return _call_openai_vision(client, model, provider, system_prompt, user_prompt, image_base64, temperature)
    
    try:
        result = _retry_call(do_call, max_retries, task)
    except Exception as e:
        logging.warning(f"[LLM] Vision failed ({type(e).__name__}: {e}), falling back to text-only")
        result = call_llm(system_prompt, user_prompt, task, temperature, stream=False, max_retries=max_retries, verbose=verbose)
    
    if verbose:
        print(f"\n[OUTPUT] {result}")
    
    return result
