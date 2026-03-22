import base64
import io
import json
import mimetypes
import os
import urllib.request
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI

try:  # pragma: no cover - optional dependency in some environments
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

try:  # pragma: no cover - import path depends on how the agent is launched
    from model_client import _get_model_mode, chat_completion_with_tools, create_openai_client
except ImportError:  # pragma: no cover - exercised in unit tests
    try:
        from agents.model_client import _get_model_mode, chat_completion_with_tools, create_openai_client
    except ImportError:  # pragma: no cover - exercised in local runner temp dirs
        import requests

        DEFAULT_PROXY_URL = "https://llm-chat-api.alibaba-inc.com/v1/api/chat"
        DEFAULT_PROXY_OPENAI_BASE_URL = "https://llm-chat-api.alibaba-inc.com/openai"
        DEFAULT_PROXY_TOKEN = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
            ".eyJ1c2VyIjoibW9kZWxfdHJhaW5fdmxtIn0"
            ".XYFz9JOEJ-f1xXPPqs9yi5cnEG4sjrR6cZayWz_NDqM"
        )
        DEFAULT_QUOTA_ID = "dd95187c-29dd-464d-9b96-8f62e6ab8eb5"
        DEFAULT_ACCESS_KEY = "9101ac974ab20f60f668dcf099bc6a10"
        DEFAULT_USER_ID = "506759"
        DEFAULT_TAG = "大模型团队_VLM_自动化评测"
        DEFAULT_APP = "model_train_vlm"

        def _get_model_mode(**kwargs):
            return kwargs.get("model_mode", os.getenv("MODEL_MODE", "local"))

        def _resolve_local_client(model_name: str, **kwargs):
            resolved_model = model_name

            if "gemini" in model_name:
                resolved_model = model_name.replace("gemini/", "openai/")
                api_key = os.getenv("GEMINI_API_KEY")
                base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
                return OpenAI(api_key=api_key, base_url=base_url), resolved_model

            if "anthropic" in model_name:
                resolved_model = model_name.replace("anthropic/", "openai/")
                api_key = os.getenv("ANTHROPIC_API_KEY")
                base_url = "https://api.anthropic.com/v1"
                return OpenAI(api_key=api_key, base_url=base_url), resolved_model

            if "together_ai" in model_name:
                resolved_model = model_name.replace("together_ai/", "openai/")
                api_key = os.environ.get("TOGETHERAI_API_KEY")
                base_url = "https://api.together.xyz/v1"
                return OpenAI(api_key=api_key, base_url=base_url), resolved_model

            return OpenAI(), resolved_model

        def create_openai_client(model_name: str = "", **kwargs):
            mode = _get_model_mode(**kwargs)
            if mode == "proxy":
                proxy_base_url = kwargs.get(
                    "proxy_openai_base_url",
                    os.getenv("PROXY_OPENAI_BASE_URL", DEFAULT_PROXY_OPENAI_BASE_URL),
                )
                proxy_api_key = kwargs.get(
                    "proxy_openai_api_key",
                    os.getenv("PROXY_OPENAI_API_KEY", os.getenv("PROXY_TOKEN", DEFAULT_PROXY_TOKEN)),
                )
                if proxy_base_url and proxy_api_key:
                    return OpenAI(base_url=proxy_base_url, api_key=proxy_api_key), model_name
            return _resolve_local_client(model_name, **kwargs)

        def _build_proxy_request(messages, model, tools=None, **kwargs):
            proxy_url = kwargs.get("proxy_url", os.getenv("PROXY_URL", DEFAULT_PROXY_URL))
            proxy_token = kwargs.get("proxy_token", os.getenv("PROXY_TOKEN", DEFAULT_PROXY_TOKEN))
            quota_id = kwargs.get("quota_id", os.getenv("QUOTA_ID", DEFAULT_QUOTA_ID))
            access_key = kwargs.get("access_key", os.getenv("ACCESS_KEY", DEFAULT_ACCESS_KEY))
            user_id = kwargs.get("user_id", os.getenv("PROXY_USER_ID", DEFAULT_USER_ID))
            tag = kwargs.get("proxy_tag", DEFAULT_TAG)
            app = kwargs.get("proxy_app", DEFAULT_APP)

            params = {
                "max_new_tokens": int(kwargs.get("max_tokens", 1024)),
                "temperature": float(kwargs.get("temperature", 1.0)),
            }
            if tools:
                params["tools"] = tools

            headers = {
                "Content-Type": "application/json",
                "token": proxy_token,
            }
            payload = json.dumps(
                {
                    "model": model,
                    "prompt": messages,
                    "tag": tag,
                    "quota_id": quota_id,
                    "app": app,
                    "params": params,
                    "user_id": user_id,
                    "access_key": access_key,
                }
            )
            return proxy_url, headers, payload

        def _build_proxy_openai_extra_body(**kwargs):
            quota_id = kwargs.get("quota_id", os.getenv("QUOTA_ID", DEFAULT_QUOTA_ID))
            access_key = kwargs.get("access_key", os.getenv("ACCESS_KEY", DEFAULT_ACCESS_KEY))
            user_id = kwargs.get("user_id", os.getenv("PROXY_USER_ID", DEFAULT_USER_ID))
            app = kwargs.get("proxy_app", DEFAULT_APP)
            return {
                "app": app,
                "quota_id": quota_id,
                "user_id": user_id,
                "access_key": access_key,
            }

        def _proxy_openai_chat_completion(messages, model, tools=None, raw_response=False, **kwargs):
            proxy_base_url = kwargs.get(
                "proxy_openai_base_url",
                os.getenv("PROXY_OPENAI_BASE_URL", DEFAULT_PROXY_OPENAI_BASE_URL),
            )
            proxy_api_key = kwargs.get(
                "proxy_openai_api_key",
                os.getenv("PROXY_OPENAI_API_KEY", os.getenv("PROXY_TOKEN", DEFAULT_PROXY_TOKEN)),
            )
            timeout = float(kwargs.get("timeout", 400))

            client = OpenAI(base_url=proxy_base_url, api_key=proxy_api_key)
            extra_params = {
                "extra_body": _build_proxy_openai_extra_body(**kwargs),
                "timeout": timeout,
            }
            if tools:
                extra_params["tools"] = tools
            if "max_tokens" in kwargs:
                extra_params["max_tokens"] = int(kwargs["max_tokens"])
            if "reasoning_effort" in kwargs:
                extra_params["reasoning_effort"] = kwargs["reasoning_effort"]
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=float(kwargs.get("temperature", 1.0)),
                **extra_params,
            )
            if raw_response:
                return response
            return response.choices[0].message.content

        def chat_completion_with_tools(messages, model, tools=None, **kwargs):
            mode = _get_model_mode(**kwargs)
            if mode == "proxy":
                try:
                    return _proxy_openai_chat_completion(
                        messages,
                        model,
                        tools=tools,
                        raw_response=True,
                        **kwargs,
                    )
                except Exception:
                    pass
                proxy_url, headers, payload = _build_proxy_request(
                    messages, model, tools=tools, **kwargs
                )
                response = requests.request("POST", proxy_url, headers=headers, data=payload)
                response_dict = json.loads(response.text)
                return response_dict.get("data")

            client, resolved_model = create_openai_client(model, **kwargs)
            extra_params = {}
            if tools:
                extra_params["tools"] = tools
            if "max_tokens" in kwargs:
                extra_params["max_tokens"] = int(kwargs["max_tokens"])
            if "reasoning_effort" in kwargs:
                extra_params["reasoning_effort"] = kwargs["reasoning_effort"]
            return client.chat.completions.create(
                model=resolved_model,
                messages=messages,
                temperature=float(kwargs.get("temperature", 1.0)),
                **extra_params,
            )

try:  # pragma: no cover - import path depends on how the agent is launched
    from common.sandbox_executor import LocalPythonExecutor, SandboxManager
except ImportError:  # pragma: no cover - exercised in unit tests
    from agents.common.sandbox_executor import LocalPythonExecutor, SandboxManager

try:  # pragma: no cover - import path depends on how the agent is launched
    from common.vqa_mcp_tools import VQAImageSession, VQAToolRegistry
except ImportError:  # pragma: no cover - exercised in unit tests
    from agents.common.vqa_mcp_tools import VQAImageSession, VQAToolRegistry


SYSTEM_PROMPT = (
    "You are solving multimodal benchmark questions. "
    "Use the execute_python tool when calculations, counting, geometry, "
    "unit conversion, or numerical verification would help. "
    "You can also use visual tools to list images, inspect image metadata, crop, zoom, "
    "or resize images when local details, small text, charts, or spatial relationships matter. "
    "You may reason step by step when helpful, but the final line of your response must begin with "
    "'Final answer:' and follow the user prompt's answer format."
)


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_debug_enabled(kwargs: Dict[str, Any]) -> bool:
    return _as_bool(kwargs.get("debug"), default=False)


def _debug_log(task_id: str, message: str, debug: bool) -> None:
    if not debug:
        return
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    print(f"[vqa_agent][{timestamp}][task={task_id}] {message}", flush=True)


def _extract_question(task_data: Dict[str, Any]) -> str:
    for key in ("question", "problem", "prompt", "query", "text", "instruction"):
        value = task_data.get(key)
        if _normalize_text(value):
            return _normalize_text(value)

    metadata = task_data.get("metadata")
    if isinstance(metadata, dict):
        for key in ("question", "problem", "prompt", "query", "text", "instruction"):
            value = metadata.get(key)
            if _normalize_text(value):
                return _normalize_text(value)

    return ""


def _extract_choices(task_data: Dict[str, Any]) -> Dict[str, str]:
    raw_choices = task_data.get("choices")
    if not isinstance(raw_choices, dict):
        return {}
    return {
        str(label).strip().upper(): _normalize_text(text)
        for label, text in raw_choices.items()
        if _normalize_text(text)
    }


def _question_embeds_choices(question: str, choices: Dict[str, str]) -> bool:
    if len(choices) < 2:
        return False

    embedded_markers = 0
    for label in choices:
        if f"({label})" in question or f"\n{label}." in question or f"\n{label})" in question:
            embedded_markers += 1
    return embedded_markers >= 2


def _collect_image_references(task_data: Dict[str, Any]) -> List[str]:
    refs: List[str] = []
    files = task_data.get("files") if isinstance(task_data.get("files"), dict) else {}

    def add_ref(candidate: Any) -> None:
        if not candidate:
            return
        if not isinstance(candidate, str):
            return
        if candidate in refs:
            return
        if candidate.startswith(("http://", "https://")):
            refs.append(candidate)
            return
        if os.path.exists(candidate):
            refs.append(os.path.abspath(candidate))
            return
        if candidate in files and os.path.exists(files[candidate]):
            refs.append(os.path.abspath(files[candidate]))
            return

    for key in ("image_path", "file_name", "image_url"):
        add_ref(task_data.get(key))

    for key in ("image_paths", "image_urls"):
        values = task_data.get(key)
        if isinstance(values, (list, tuple)):
            for value in values:
                add_ref(value)

    if not refs and files:
        for abs_path in files.values():
            add_ref(abs_path)

    return refs


def _guess_mime_type(path: str) -> str:
    mime_type, _ = mimetypes.guess_type(path)
    return mime_type or "image/png"


def _should_resize_local_image(benchmark_name: str) -> bool:
    return benchmark_name in {"hrbench4k", "hrbench8k"}


def _read_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return default


def _read_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return default


def _serialize_image_bytes(
    image: "Image.Image",
    *,
    prefer_png: bool,
    jpeg_quality: int,
) -> Tuple[str, bytes]:
    buffer = io.BytesIO()
    if prefer_png:
        image.save(buffer, format="PNG", optimize=True)
        return "image/png", buffer.getvalue()

    rgb_image = image.convert("RGB")
    rgb_image.save(
        buffer,
        format="JPEG",
        quality=jpeg_quality,
        optimize=True,
    )
    return "image/jpeg", buffer.getvalue()


def _remote_image_ref_to_content_part(image_ref: str) -> Dict[str, Any]:
    with urllib.request.urlopen(image_ref, timeout=30) as response:
        image_bytes = response.read()
        mime_type = response.headers.get_content_type()

    encoded = base64.b64encode(image_bytes).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type or _guess_mime_type(image_ref)};base64,{encoded}"},
    }


def _prepare_local_image_payload(
    image_path: Path,
    benchmark_name: str,
    model_mode: Optional[str] = None,
) -> Tuple[str, bytes]:
    if not _should_resize_local_image(benchmark_name):
        mime_type = _guess_mime_type(str(image_path))
        return mime_type, image_path.read_bytes()

    if Image is None:
        mime_type = _guess_mime_type(str(image_path))
        return mime_type, image_path.read_bytes()

    default_target_size = -1
    if model_mode == "local":
        default_target_size = _read_int_env(
            "VQA_AGENT_HRBENCH_LOCAL_VLLM_TARGET_SIZE",
            1024,
        )

    initial_target_size = _read_int_env(
        "VQA_AGENT_HRBENCH_LOCAL_IMAGE_TARGET_SIZE",
        _read_int_env("VQA_AGENT_HRBENCH_LOCAL_IMAGE_MAX_EDGE", default_target_size),
    )
    max_payload_bytes = _read_int_env(
        "VQA_AGENT_HRBENCH_LOCAL_IMAGE_MAX_BYTES",
        _read_int_env("VLMEVAL_MAX_IMAGE_SIZE", 3_000_000),
    )
    min_edge = _read_int_env(
        "VQA_AGENT_HRBENCH_LOCAL_IMAGE_MIN_EDGE",
        _read_int_env("VLMEVAL_MIN_IMAGE_EDGE", 100),
    )
    resize_factor = _read_float_env("VQA_AGENT_HRBENCH_LOCAL_IMAGE_RESIZE_FACTOR", 0.7)
    jpeg_quality = _read_int_env(
        "VQA_AGENT_HRBENCH_LOCAL_IMAGE_JPEG_QUALITY",
        _read_int_env("VQA_AGENT_LOCAL_IMAGE_JPEG_QUALITY", 85),
    )

    try:
        with Image.open(image_path) as image:
            source_image = image.copy()
            has_alpha = source_image.mode in ("RGBA", "LA") or (
                source_image.mode == "P" and "transparency" in source_image.info
            )

            if initial_target_size > 0:
                source_image.thumbnail((initial_target_size, initial_target_size))

            mime_type, image_bytes = _serialize_image_bytes(
                source_image,
                prefer_png=has_alpha,
                jpeg_quality=jpeg_quality,
            )
            if len(image_bytes) <= max_payload_bytes:
                return mime_type, image_bytes

            factor = 1.0
            while min(source_image.size) > min_edge:
                factor *= resize_factor
                if factor <= 0:
                    break

                resized = source_image.copy()
                target_width = max(int(source_image.width * factor), min_edge)
                target_height = max(int(source_image.height * factor), min_edge)
                resized.thumbnail((target_width, target_height))
                mime_type, image_bytes = _serialize_image_bytes(
                    resized,
                    prefer_png=has_alpha,
                    jpeg_quality=jpeg_quality,
                )
                if len(image_bytes) <= max_payload_bytes or min(resized.size) <= min_edge:
                    return mime_type, image_bytes

            return mime_type, image_bytes
    except Exception:
        mime_type = _guess_mime_type(str(image_path))
        return mime_type, image_path.read_bytes()


def _image_ref_to_content_part(
    image_ref: str,
    benchmark_name: str,
    model_mode: Optional[str] = None,
) -> Dict[str, Any]:
    if image_ref.startswith(("http://", "https://")):
        if model_mode == "proxy":
            try:
                return _remote_image_ref_to_content_part(image_ref)
            except Exception:
                pass
        return {"type": "image_url", "image_url": {"url": image_ref}}

    image_path = Path(image_ref)
    mime_type, image_bytes = _prepare_local_image_payload(
        image_path,
        benchmark_name,
        model_mode=model_mode,
    )
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
    }


def _build_task_prompt(question: str, choices: Dict[str, str], include_image: bool) -> str:
    lines = [
        "Answer the following benchmark question as accurately as possible.",
    ]
    if include_image:
        lines.append("Use the provided image(s) when relevant.")

    lines.extend(["", f"Question: {question}"])

    if choices and not _question_embeds_choices(question, choices):
        lines.extend(["", "Choices:"])
        for label, choice_text in choices.items():
            lines.append(f"({label}) {choice_text}")

    lines.extend(
        [
            "",
            "You may think step by step if helpful.",
            "End with a final line in the format: Final answer: <answer>",
        ]
    )
    return "\n".join(lines)


def _build_user_content(
    task_data: Dict[str, Any],
    include_image: bool,
    benchmark_name: str,
    model_mode: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str, Dict[str, str], List[str]]:
    question = _extract_question(task_data)
    choices = _extract_choices(task_data)
    image_refs = _collect_image_references(task_data) if include_image else []
    prompt = _build_task_prompt(question, choices, include_image=include_image)

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image_ref in image_refs:
        content.append(_image_ref_to_content_part(image_ref, benchmark_name, model_mode=model_mode))
    return content, question, choices, image_refs


def _summarize_task_media(task_data: Dict[str, Any]) -> Dict[str, Any]:
    files = task_data.get("files") if isinstance(task_data.get("files"), dict) else {}
    choices = _extract_choices(task_data)
    return {
        "image_path": task_data.get("image_path"),
        "file_name": task_data.get("file_name"),
        "image_url": task_data.get("image_url"),
        "image_paths_len": len(task_data.get("image_paths") or []),
        "image_urls_len": len(task_data.get("image_urls") or []),
        "files_len": len(files),
        "choices_len": len(choices),
    }


def _extract_proxy_error_message(raw_response: Dict[str, Any]) -> str:
    completion = raw_response.get("completion")
    if isinstance(completion, dict):
        error = completion.get("error")
        if isinstance(error, dict):
            message = _normalize_text(error.get("message"))
            if message:
                return message

    error = raw_response.get("error")
    if isinstance(error, dict):
        message = _normalize_text(error.get("message"))
        if message:
            return message

    return ""


def _normalize_model_response(raw_response: Any, mode: str, iteration: int) -> Tuple[str, Any, Optional[str]]:
    if mode == "local" or hasattr(raw_response, "choices"):
        choice = raw_response.choices[0]
        assistant_message = choice.message
        finish_reason = choice.finish_reason or ("tool_calls" if getattr(assistant_message, "tool_calls", None) else "stop")
        thinking_content = (
            getattr(assistant_message, "reasoning_content", None)
            or getattr(assistant_message, "reasoning", None)
        )
        return finish_reason, assistant_message, thinking_content

    if not isinstance(raw_response, dict):
        return "stop", SimpleNamespace(content="", tool_calls=None, role="assistant"), None

    message = raw_response.get("message", "")
    if isinstance(message, dict) and message.get("is_function_call"):
        tool_calls = [
            SimpleNamespace(
                id=f"proxy_call_{iteration}",
                function=SimpleNamespace(
                    name=message.get("function_call_name", ""),
                    arguments=message.get("function_call_args", "{}"),
                ),
            )
        ]
        assistant_message = SimpleNamespace(content=None, tool_calls=tool_calls, role="assistant")
        return "tool_calls", assistant_message, None

    error_message = _extract_proxy_error_message(raw_response)
    if not message and error_message:
        assistant_message = SimpleNamespace(
            content=f"ERROR: {error_message}",
            tool_calls=None,
            role="assistant",
        )
        return "stop", assistant_message, None

    content = message if isinstance(message, str) else str(message)
    assistant_message = SimpleNamespace(content=content, tool_calls=None, role="assistant")
    return "stop", assistant_message, None


def _assistant_message_to_dict(assistant_message: Any) -> Dict[str, Any]:
    if isinstance(assistant_message, dict):
        return assistant_message

    message = {"role": "assistant", "content": assistant_message.content or ""}
    tool_calls = getattr(assistant_message, "tool_calls", None)
    if tool_calls:
        message["tool_calls"] = []
        for tool_call in tool_calls:
            message["tool_calls"].append(
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
            )
    return message


def _invoke_completion(
    *,
    messages: List[Dict[str, Any]],
    model_name: str,
    tools: Optional[List[Dict[str, Any]]] = None,
    api_base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    mode = _get_model_mode(**kwargs)
    temperature = float(kwargs.get("temperature", 0.0))
    extra_params: Dict[str, Any] = {}
    if mode == "proxy":
        pass
    else:
        extra_params["max_tokens"] = int(kwargs.get("max_tokens", 1024))
    if "reasoning_effort" in kwargs:
        extra_params["reasoning_effort"] = kwargs["reasoning_effort"]

    if api_base_url or api_key:
        client = OpenAI(
            base_url=api_base_url or os.getenv("OPENAI_BASE_URL"),
            api_key=api_key or os.getenv("OPENAI_API_KEY", "empty"),
        )
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            tools=tools,
            **extra_params,
        )

    if mode == "proxy":
        return chat_completion_with_tools(
            messages=messages,
            model=model_name,
            tools=tools,
            temperature=temperature,
            **extra_params,
            **{k: v for k, v in kwargs.items() if k not in {"temperature", "max_tokens", "reasoning_effort"}},
        )

    client, resolved_model = create_openai_client(model_name, **kwargs)
    return client.chat.completions.create(
        model=resolved_model,
        messages=messages,
        temperature=temperature,
        tools=tools,
        **extra_params,
    )


def _create_code_executor(code_executor: str, sandbox_url: Optional[str]) -> Any:
    if code_executor == "sandbox":
        return SandboxManager(base_url=sandbox_url)
    return LocalPythonExecutor()


def _solve_single_channel(
    *,
    task_id: str,
    task_data: Dict[str, Any],
    benchmark_name: str,
    model_name: str,
    include_image: bool,
    debug: bool,
    enable_tools: bool,
    code_executor: str,
    sandbox_url: Optional[str],
    api_base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[str, Dict[str, Any]]:
    mode = _get_model_mode(**kwargs)
    user_content, question, _, image_refs = _build_user_content(
        task_data,
        include_image=include_image,
        benchmark_name=benchmark_name,
        model_mode=mode,
    )
    _debug_log(
        task_id,
        (
            f"starting request include_image={include_image} "
            f"question_len={len(question)} image_count={len(image_refs)} "
            f"max_tokens={kwargs.get('max_tokens')}"
        ),
        debug,
    )
    if image_refs:
        _debug_log(task_id, f"image_refs={','.join(image_refs)}", debug)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    max_iterations = int(kwargs.get("max_iterations", 8))

    conversation_history: List[Dict[str, Any]] = []
    tool_call_count = 0
    sandbox_error_types: List[str] = []
    executor = _create_code_executor(code_executor, sandbox_url) if enable_tools else None
    image_session = VQAImageSession(task_id)
    final_answer = ""
    exhausted_tool_iterations = False

    try:
        image_session.register_initial_images(image_refs)
        tool_registry = VQAToolRegistry(executor, image_session) if enable_tools else None
        for iteration in range(max_iterations):
            raw_response = _invoke_completion(
                messages=messages,
                model_name=model_name,
                tools=tool_registry.get_tool_schemas() if enable_tools and tool_registry else None,
                api_base_url=api_base_url,
                api_key=api_key,
                **kwargs,
            )
            finish_reason, assistant_message, thinking_content = _normalize_model_response(
                raw_response, mode, iteration
            )
            current_content = assistant_message.content or ""
            tool_calls = getattr(assistant_message, "tool_calls", None)
            if enable_tools and tool_calls:
                final_answer = ""
            else:
                final_answer = current_content
            response_preview = current_content.replace("\n", " ")[:160]
            _debug_log(
                task_id,
                (
                    f"request finished include_image={include_image} "
                    f"response_len={len(current_content)} response_preview={response_preview!r}"
                ),
                debug,
            )

            assistant_record: Dict[str, Any] = {
                "iteration": iteration,
                "role": "assistant",
                "content": current_content,
            }
            if thinking_content:
                assistant_record["thinking_content"] = thinking_content

            if tool_calls:
                assistant_record["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "function_name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    }
                    for tool_call in tool_calls
                ]
            conversation_history.append(assistant_record)
            messages.append(_assistant_message_to_dict(assistant_message))

            if not enable_tools or not tool_calls:
                break

            for tool_call in tool_calls:
                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}

                execution_result = tool_registry.execute_tool(
                    tool_call.function.name,
                    arguments,
                )
                tool_call_count += 1
                error_type = execution_result.get("error_type")
                if error_type:
                    sandbox_error_types.append(error_type)

                tool_output = str(execution_result.get("output", "(No output)"))
                tool_record: Dict[str, Any] = {
                    "iteration": iteration,
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "tool_name": tool_call.function.name,
                    "arguments": arguments,
                    "result": tool_output,
                    "error_type": error_type,
                }
                tool_payload = execution_result.get("tool_payload")
                if isinstance(tool_payload, dict):
                    tool_record["tool_payload"] = tool_payload
                conversation_history.append(
                    tool_record
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_output,
                    }
                )
                generated_image_path = execution_result.get("generated_image_path")
                generated_image_id = execution_result.get("generated_image_id")
                if generated_image_path and generated_image_id:
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "A derived image was created by a visual tool. "
                                        f"It is available as image_id={generated_image_id}. "
                                        "You may inspect it in subsequent reasoning."
                                    ),
                                },
                                _image_ref_to_content_part(
                                    str(generated_image_path),
                                    benchmark_name,
                                    model_mode=mode,
                                ),
                            ],
                        }
                    )
        else:
            exhausted_tool_iterations = True

        if enable_tools and not final_answer:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Stop using tools now and provide the final answer directly. "
                        "Return only the final answer with no explanation."
                    ),
                }
            )
            raw_response = _invoke_completion(
                messages=messages,
                model_name=model_name,
                tools=None,
                api_base_url=api_base_url,
                api_key=api_key,
                **kwargs,
            )
            finish_reason, assistant_message, thinking_content = _normalize_model_response(
                raw_response, mode, max_iterations
            )
            final_answer = assistant_message.content or ""
            response_preview = (final_answer or "").replace("\n", " ")[:160]
            _debug_log(
                task_id,
                (
                    f"fallback request finished include_image={include_image} "
                    f"response_len={len(final_answer)} response_preview={response_preview!r} "
                    f"after_exhausted={exhausted_tool_iterations}"
                ),
                debug,
            )
            assistant_record = {
                "iteration": max_iterations,
                "role": "assistant",
                "content": final_answer,
            }
            if thinking_content:
                assistant_record["thinking_content"] = thinking_content
            conversation_history.append(assistant_record)
        return final_answer, {
            "tool_call_count": tool_call_count,
            "conversation_history": conversation_history,
            "sandbox_error_types": sandbox_error_types,
            "benchmark_name": benchmark_name,
        }
    finally:
        if executor is not None:
            executor.destroy()
        image_session.destroy()


def _merge_channel_metrics(channel_metrics: Iterable[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
    merged_history: List[Dict[str, Any]] = []
    merged_error_types: List[str] = []
    tool_call_count = 0

    for channel_name, metrics in channel_metrics:
        if not metrics:
            continue
        tool_call_count += int(metrics.get("tool_call_count", 0))
        merged_error_types.extend(metrics.get("sandbox_error_types", []))
        for turn in metrics.get("conversation_history", []):
            if isinstance(turn, dict):
                merged_history.append({**turn, "channel": channel_name})

    return {
        "tool_call_count": tool_call_count,
        "conversation_history": merged_history,
        "sandbox_error_types": merged_error_types,
    }


def _solve_standard_vqa_task(
    task_id: str,
    task_data: Dict[str, Any],
    benchmark_name: str,
    debug: bool,
    enable_tools: bool,
    code_executor: str,
    sandbox_url: Optional[str],
    **kwargs: Any,
) -> Any:
    channel_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in {"model_name", "benchmark_name", "debug", "enable_tools", "code_executor", "sandbox_url"}
    }
    answer, metrics = _solve_single_channel(
        task_id=task_id,
        task_data=task_data,
        benchmark_name=benchmark_name,
        model_name=kwargs["model_name"],
        include_image=True,
        debug=debug,
        enable_tools=enable_tools,
        code_executor=code_executor,
        sandbox_url=sandbox_url,
        **channel_kwargs,
    )
    if enable_tools:
        return {"answer": answer, "metrics": metrics}
    return answer


def _solve_mmstar_task(
    task_id: str,
    task_data: Dict[str, Any],
    benchmark_name: str,
    debug: bool,
    enable_tools: bool,
    code_executor: str,
    sandbox_url: Optional[str],
    **kwargs: Any,
) -> Dict[str, Any]:
    channel_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in {"model_name", "benchmark_name", "debug", "enable_tools", "code_executor", "sandbox_url"}
    }
    vision_answer, vision_metrics = _solve_single_channel(
        task_id=task_id,
        task_data=task_data,
        benchmark_name=benchmark_name,
        model_name=kwargs["model_name"],
        include_image=True,
        debug=debug,
        enable_tools=enable_tools,
        code_executor=code_executor,
        sandbox_url=sandbox_url,
        **channel_kwargs,
    )
    no_image_answer, no_image_metrics = _solve_single_channel(
        task_id=task_id,
        task_data=task_data,
        benchmark_name=benchmark_name,
        model_name=kwargs["model_name"],
        include_image=False,
        debug=debug,
        enable_tools=enable_tools,
        code_executor=code_executor,
        sandbox_url=sandbox_url,
        **channel_kwargs,
    )

    base_model_name = kwargs.get("base_model_name")
    base_openai_base_url = kwargs.get("base_openai_base_url")
    base_openai_api_key = kwargs.get("base_openai_api_key")

    if base_model_name or base_openai_base_url or base_openai_api_key:
        base_llm_answer, base_llm_metrics = _solve_single_channel(
            task_id=task_id,
            task_data=task_data,
            benchmark_name=benchmark_name,
            model_name=base_model_name or kwargs["model_name"],
            include_image=False,
            debug=debug,
            enable_tools=enable_tools,
            code_executor=code_executor,
            sandbox_url=sandbox_url,
            api_base_url=base_openai_base_url,
            api_key=base_openai_api_key,
            **channel_kwargs,
        )
    else:
        base_llm_answer = no_image_answer
        base_llm_metrics = {}

    result = {
        "vision_answer": vision_answer,
        "no_image_answer": no_image_answer,
        "base_llm_answer": base_llm_answer,
    }
    if enable_tools:
        result["metrics"] = _merge_channel_metrics(
            [
                ("vision", vision_metrics),
                ("no_image", no_image_metrics),
                ("base_llm", base_llm_metrics),
            ]
        )
    return result


def run_vqa_agent(input: Dict[str, Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
    assert "model_name" in kwargs, "model_name is required"

    benchmark_name = str(kwargs.get("benchmark_name", ""))
    debug = _is_debug_enabled(kwargs)
    enable_tools = _as_bool(kwargs.get("enable_tools"), default=True)
    code_executor = str(kwargs.get("code_executor", "local")).strip().lower()
    sandbox_url = kwargs.get("sandbox_url")
    solver_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in {"benchmark_name", "debug", "enable_tools", "code_executor", "sandbox_url"}
    }

    outputs: Dict[str, Any] = {}
    for task_id, task_data in input.items():
        task_payload = task_data if isinstance(task_data, dict) else {}
        _debug_log(
            str(task_id),
            (
                f"task received benchmark={benchmark_name or 'unknown'} "
                f"timeout={kwargs.get('timeout')} has_question={bool(_extract_question(task_payload))} "
                f"media={_summarize_task_media(task_payload)}"
            ),
            debug,
        )
        try:
            if benchmark_name == "mmstar":
                outputs[task_id] = _solve_mmstar_task(
                    str(task_id),
                    task_payload,
                    benchmark_name,
                    debug,
                    enable_tools,
                    code_executor,
                    sandbox_url,
                    **solver_kwargs,
                )
            else:
                outputs[task_id] = _solve_standard_vqa_task(
                    str(task_id),
                    task_payload,
                    benchmark_name,
                    debug,
                    enable_tools,
                    code_executor,
                    sandbox_url,
                    **solver_kwargs,
                )
        except Exception as exc:  # pragma: no cover - exercised in integration runtime
            error_text = f"ERROR: {exc}"
            _debug_log(str(task_id), f"request failed error={error_text}", debug)
            if benchmark_name == "mmstar":
                outputs[task_id] = {
                    "vision_answer": error_text,
                    "no_image_answer": error_text,
                    "base_llm_answer": error_text,
                }
            else:
                outputs[task_id] = error_text

    return outputs
