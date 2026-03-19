import base64
import mimetypes
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from openai import OpenAI


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _is_debug_enabled(kwargs: Dict[str, Any]) -> bool:
    raw_value = kwargs.get("debug")
    if raw_value is None:
        raw_value = os.getenv("VQA_AGENT_DEBUG", "")
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def _debug_log(enabled: bool, task_id: str, message: str) -> None:
    if not enabled:
        return
    timestamp = datetime.now().isoformat(timespec="seconds")
    print(f"[vqa_agent][{timestamp}][task={task_id}] {message}", flush=True)


def _summarize_task_media(task_data: Dict[str, Any]) -> str:
    summary = {
        "image_path": task_data.get("image_path"),
        "file_name": task_data.get("file_name"),
        "image_url": task_data.get("image_url"),
        "image_paths_len": len(task_data.get("image_paths") or []),
        "image_urls_len": len(task_data.get("image_urls") or []),
        "files_len": len(task_data.get("files") or {}),
        "choices_len": len(_extract_choices(task_data)),
    }
    return str(summary)


def _extract_question(task_data: Dict[str, Any]) -> str:
    for key in ("question", "problem", "prompt", "query", "instruction", "text"):
        value = _normalize_text(task_data.get(key))
        if value:
            return value

    metadata = task_data.get("metadata") or {}
    if isinstance(metadata, dict):
        for key in ("question", "problem", "prompt", "query", "instruction", "text"):
            value = _normalize_text(metadata.get(key))
            if value:
                return value
    return ""


def _extract_choices(task_data: Dict[str, Any]) -> Dict[str, str]:
    raw_choices = task_data.get("choices") or {}
    if isinstance(raw_choices, dict):
        return {
            str(label).strip().upper(): _normalize_text(choice)
            for label, choice in raw_choices.items()
            if _normalize_text(choice)
        }
    if isinstance(raw_choices, (list, tuple)):
        return {
            chr(ord("A") + idx): _normalize_text(choice)
            for idx, choice in enumerate(raw_choices)
            if idx < 26 and _normalize_text(choice)
        }
    return {}


def _collect_image_references(task_data: Dict[str, Any]) -> List[str]:
    image_refs: List[str] = []

    def _append_if_present(value: Any) -> None:
        if isinstance(value, str) and value and value not in image_refs:
            image_refs.append(value)

    _append_if_present(task_data.get("image_path"))
    _append_if_present(task_data.get("file_name"))
    _append_if_present(task_data.get("image_url"))

    for key in ("image_paths", "image_urls"):
        values = task_data.get(key) or []
        if isinstance(values, (list, tuple)):
            for value in values:
                _append_if_present(value)

    files = task_data.get("files") or {}
    if isinstance(files, dict):
        for relative_path in files.keys():
            if os.path.splitext(str(relative_path))[1].lower() in IMAGE_EXTENSIONS:
                _append_if_present(str(relative_path))

    return image_refs


def _guess_mime_type(path: str) -> str:
    mime_type, _ = mimetypes.guess_type(path)
    return mime_type or "image/png"


def _image_ref_to_content_part(image_ref: str) -> Optional[Dict[str, Any]]:
    if image_ref.startswith(("http://", "https://", "data:")):
        return {"type": "image_url", "image_url": {"url": image_ref}}

    if not os.path.exists(image_ref):
        return None

    with open(image_ref, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")

    return {
        "type": "image_url",
        "image_url": {"url": f"data:{_guess_mime_type(image_ref)};base64,{encoded}"},
    }


def _build_task_prompt(task_data: Dict[str, Any], include_image: bool) -> str:
    benchmark_name = str(task_data.get("benchmark_name", ""))
    question = _extract_question(task_data)
    choices = _extract_choices(task_data)

    prompt_parts = []
    if benchmark_name == "mmstar":
        prompt_parts.append(
            "You are answering a multimodal benchmark question. "
            "Return only the final answer with no explanation."
        )
    else:
        prompt_parts.append(
            "Answer the following benchmark question as accurately as possible. "
            "Return only the final answer with no explanation."
        )

    if include_image:
        prompt_parts.append("Use the provided image(s) when relevant.")
    else:
        prompt_parts.append("Do not assume access to any image.")

    if question:
        prompt_parts.append(f"Question: {question}")

    if choices:
        choice_lines = [f"{label}. {text}" for label, text in choices.items()]
        prompt_parts.append("Choices:\n" + "\n".join(choice_lines))
        prompt_parts.append(
            "Respond with only the single best choice letter (for example: A)."
        )
    else:
        prompt_parts.append(
            "Respond with only the final short answer. Do not include reasoning."
        )

    return "\n\n".join(prompt_parts)


def _build_user_content(task_data: Dict[str, Any], include_image: bool) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": _build_task_prompt(task_data, include_image=include_image)}
    ]
    if include_image:
        for image_ref in _collect_image_references(task_data):
            image_part = _image_ref_to_content_part(image_ref)
            if image_part is not None:
                content.append(image_part)
    return content


def _create_client(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: Optional[float] = None,
) -> OpenAI:
    return OpenAI(
        base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        api_key=api_key or os.getenv("OPENAI_API_KEY", "EMPTY"),
        timeout=timeout,
    )


def _chat_once(
    client: OpenAI,
    model_name: str,
    task_data: Dict[str, Any],
    include_image: bool,
    temperature: float,
    max_tokens: Optional[int],
    task_id: str,
    debug_enabled: bool,
) -> str:
    image_refs = _collect_image_references(task_data) if include_image else []
    _debug_log(
        debug_enabled,
        task_id,
        f"starting request include_image={include_image} question_len={len(_extract_question(task_data))} image_count={len(image_refs)} max_tokens={max_tokens}",
    )
    if image_refs:
        preview = ", ".join(image_refs[:2])
        if len(image_refs) > 2:
            preview += ", ..."
        _debug_log(debug_enabled, task_id, f"image_refs={preview}")

    completion_kwargs: Dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": _build_user_content(task_data, include_image)}],
        "temperature": temperature,
    }
    if max_tokens is not None:
        completion_kwargs["max_tokens"] = max_tokens

    response = client.chat.completions.create(**completion_kwargs)
    text = _normalize_text(response.choices[0].message.content or "")
    _debug_log(
        debug_enabled,
        task_id,
        f"request finished include_image={include_image} response_len={len(text)} response_preview={text[:120]!r}",
    )
    return text


def _solve_standard_vqa_task(
    client: OpenAI,
    model_name: str,
    task_data: Dict[str, Any],
    temperature: float,
    max_tokens: Optional[int],
    task_id: str,
    debug_enabled: bool,
) -> str:
    return _chat_once(
        client=client,
        model_name=model_name,
        task_data=task_data,
        include_image=True,
        temperature=temperature,
        max_tokens=max_tokens,
        task_id=task_id,
        debug_enabled=debug_enabled,
    )


def _solve_mmstar_task(
    client: OpenAI,
    model_name: str,
    task_data: Dict[str, Any],
    temperature: float,
    max_tokens: Optional[int],
    base_model_name: Optional[str],
    base_client: Optional[OpenAI],
    task_id: str,
    debug_enabled: bool,
) -> Dict[str, str]:
    vision_answer = _chat_once(
        client=client,
        model_name=model_name,
        task_data=task_data,
        include_image=True,
        temperature=temperature,
        max_tokens=max_tokens,
        task_id=task_id,
        debug_enabled=debug_enabled,
    )
    no_image_answer = _chat_once(
        client=client,
        model_name=model_name,
        task_data=task_data,
        include_image=False,
        temperature=temperature,
        max_tokens=max_tokens,
        task_id=task_id,
        debug_enabled=debug_enabled,
    )

    if base_model_name and base_client is not None:
        base_llm_answer = _chat_once(
            client=base_client,
            model_name=base_model_name,
            task_data=task_data,
            include_image=False,
            temperature=temperature,
            max_tokens=max_tokens,
            task_id=task_id,
            debug_enabled=debug_enabled,
        )
    else:
        base_llm_answer = no_image_answer

    return {
        "vision_answer": vision_answer,
        "no_image_answer": no_image_answer,
        "base_llm_answer": base_llm_answer,
    }


def run_vqa_agent(input: Dict[str, Dict[str, Any]], **kwargs) -> Dict[str, Any]:
    assert "model_name" in kwargs, "model_name is required"

    model_name = kwargs["model_name"]
    temperature = float(kwargs.get("temperature", 0.0))
    max_tokens = (
        int(kwargs["max_tokens"])
        if "max_tokens" in kwargs and kwargs["max_tokens"] is not None
        else None
    )
    timeout = float(kwargs.get("timeout", 300))
    debug_enabled = _is_debug_enabled(kwargs)

    client = _create_client(timeout=timeout)

    base_model_name = kwargs.get("base_model_name") or os.getenv("BASE_MODEL_NAME")
    base_client = None
    if base_model_name:
        base_client = _create_client(
            base_url=kwargs.get("base_openai_base_url") or os.getenv("BASE_OPENAI_BASE_URL"),
            api_key=kwargs.get("base_openai_api_key") or os.getenv("BASE_OPENAI_API_KEY", "EMPTY"),
            timeout=timeout,
        )

    results: Dict[str, Any] = {}
    benchmark_name = str(kwargs.get("benchmark_name", ""))

    for task_id, original_task_data in input.items():
        task_data = dict(original_task_data)
        task_data["benchmark_name"] = benchmark_name
        _debug_log(
            debug_enabled,
            task_id,
            f"task received benchmark={benchmark_name} timeout={timeout} has_question={bool(_extract_question(task_data))} media={_summarize_task_media(task_data)}",
        )

        try:
            if benchmark_name == "mmstar":
                results[task_id] = _solve_mmstar_task(
                    client=client,
                    model_name=model_name,
                    task_data=task_data,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    base_model_name=base_model_name,
                    base_client=base_client,
                    task_id=task_id,
                    debug_enabled=debug_enabled,
                )
            else:
                results[task_id] = _solve_standard_vqa_task(
                    client=client,
                    model_name=model_name,
                    task_data=task_data,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    task_id=task_id,
                    debug_enabled=debug_enabled,
                )
        except Exception as exc:
            _debug_log(debug_enabled, task_id, f"request failed error={exc}")
            if benchmark_name == "mmstar":
                error_text = f"ERROR: {exc}"
                results[task_id] = {
                    "vision_answer": error_text,
                    "no_image_answer": error_text,
                    "base_llm_answer": error_text,
                }
            else:
                results[task_id] = f"ERROR: {exc}"

    return results
