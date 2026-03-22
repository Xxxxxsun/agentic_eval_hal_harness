"""
Unified model client that supports two modes:
  - "local": Uses OpenAI SDK to call a local inference server (via OPENAI_BASE_URL)
  - "proxy": Uses the internal proxy gateway (llm-chat-api.alibaba-inc.com) to call
              external models like gpt-5, gemini, etc.

Usage:
    from model_client import create_client, chat_completion

    # In agent's run() function:
    client = create_client(**kwargs)
    result = chat_completion(client, messages=[...], model="gpt-5", **kwargs)
"""

import os
import json
import traceback
from typing import List, Dict, Optional, Any, Tuple

import requests
from openai import OpenAI


# ---------------------------------------------------------------------------
# Default proxy configuration
# ---------------------------------------------------------------------------
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


def _get_model_mode(**kwargs) -> str:
    """Determine model mode from kwargs or environment variable."""
    return kwargs.get("model_mode", os.getenv("MODEL_MODE", "local"))


# ---------------------------------------------------------------------------
# Proxy mode: call external models via internal gateway
# ---------------------------------------------------------------------------
def _build_proxy_request(
    messages: List[Dict],
    model: str,
    tools: Optional[List[Dict]] = None,
    **kwargs,
) -> Tuple[str, Dict, str]:
    """
    Build the proxy gateway HTTP request components.
    Returns (url, headers, payload_json).
    """
    proxy_url = kwargs.get("proxy_url", os.getenv("PROXY_URL", DEFAULT_PROXY_URL))
    proxy_token = kwargs.get("proxy_token", os.getenv("PROXY_TOKEN", DEFAULT_PROXY_TOKEN))
    quota_id = kwargs.get("quota_id", os.getenv("QUOTA_ID", DEFAULT_QUOTA_ID))
    access_key = kwargs.get("access_key", os.getenv("ACCESS_KEY", DEFAULT_ACCESS_KEY))
    user_id = kwargs.get("user_id", os.getenv("PROXY_USER_ID", DEFAULT_USER_ID))
    tag = kwargs.get("proxy_tag", DEFAULT_TAG)
    app = kwargs.get("proxy_app", DEFAULT_APP)

    params = {}
    params["max_new_tokens"] = int(kwargs.get("max_tokens", 1024))
    params["temperature"] = float(kwargs.get("temperature", 1.0))
    if tools:
        params["tools"] = tools

    headers = {
        "Content-Type": "application/json",
        "token": proxy_token,
    }

    payload = json.dumps({
        "model": model,
        "prompt": messages,
        "tag": tag,
        "quota_id": quota_id,
        "app": app,
        "params": params,
        "user_id": user_id,
        "access_key": access_key,
    })

    return proxy_url, headers, payload


def _build_proxy_openai_extra_body(**kwargs) -> Dict[str, Any]:
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


def _proxy_openai_chat_completion(
    messages: List[Dict],
    model: str,
    tools: Optional[List[Dict]] = None,
    raw_response: bool = False,
    **kwargs,
) -> Any:
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

    extra_params: Dict[str, Any] = {
        "extra_body": _build_proxy_openai_extra_body(**kwargs),
        "timeout": timeout,
    }
    if "reasoning_effort" in kwargs:
        extra_params["reasoning_effort"] = kwargs["reasoning_effort"]
    if tools:
        extra_params["tools"] = tools
    if "max_tokens" in kwargs:
        extra_params["max_tokens"] = int(kwargs["max_tokens"])
    if "n" in kwargs:
        extra_params["n"] = int(kwargs["n"])

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=float(kwargs.get("temperature", 1.0)),
        **extra_params,
    )

    if raw_response:
        return response
    return response.choices[0].message.content


def _proxy_chat_completion(
    messages: List[Dict],
    model: str,
    **kwargs,
) -> Optional[str]:
    """
    Call external model through the internal proxy gateway.
    Returns the model's text response, or None on failure.
    """
    try:
        return _proxy_openai_chat_completion(messages, model, **kwargs)
    except Exception:
        print(traceback.format_exc(), flush=True)

    proxy_url, headers, payload = _build_proxy_request(messages, model, **kwargs)

    try:
        response = requests.request("POST", proxy_url, headers=headers, data=payload)
        response_dict = json.loads(response.text)
    except Exception:
        print(traceback.format_exc(), flush=True)
        return None

    if "data" in response_dict:
        return response_dict["data"]["message"]
    else:
        print(response_dict, flush=True)
        return None


def _proxy_chat_completion_raw(
    messages: List[Dict],
    model: str,
    tools: Optional[List[Dict]] = None,
    **kwargs,
) -> Optional[Dict]:
    """
    Call external model through the proxy gateway and return the full
    response dict (including tool_calls if present).

    Used by agents that need tool calling / function calling support.

    Returns:
        The full response dict from the proxy gateway, or None on failure.
        Typically contains: {"data": {"message": ..., "tool_calls": [...], ...}}
    """
    try:
        return _proxy_openai_chat_completion(
            messages,
            model,
            tools=tools,
            raw_response=True,
            **kwargs,
        )
    except Exception:
        print(traceback.format_exc(), flush=True)

    proxy_url, headers, payload = _build_proxy_request(
        messages, model, tools=tools, **kwargs
    )

    try:
        response = requests.request("POST", proxy_url, headers=headers, data=payload)
        response_dict = json.loads(response.text)
    except Exception:
        print(traceback.format_exc(), flush=True)
        return None

    if "data" in response_dict:
        return response_dict["data"]
    else:
        print(response_dict, flush=True)
        return None


# ---------------------------------------------------------------------------
# Local mode: call local inference server via OpenAI SDK
# ---------------------------------------------------------------------------
def _resolve_local_client(model_name: str, **kwargs) -> Tuple[OpenAI, str]:
    """
    Build an OpenAI client for local mode, handling provider-specific
    base_url / api_key overrides (gemini, anthropic, together_ai, etc.).
    Returns (client, resolved_model_name).
    """
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

    # Default: use OPENAI_BASE_URL / OPENAI_API_KEY from environment
    return OpenAI(), resolved_model


def _local_chat_completion(
    messages: List[Dict],
    model: str,
    tools: Optional[List[Dict]] = None,
    raw_response: bool = False,
    **kwargs,
) -> Any:
    """
    Call model via OpenAI-compatible API (local inference server or
    third-party provider).

    Args:
        tools:        Optional list of tool definitions for function calling.
        raw_response: If True, return the full OpenAI response object
                      instead of just the text content.
    """
    client, resolved_model = _resolve_local_client(model, **kwargs)

    extra_params = {}
    if "reasoning_effort" in kwargs:
        extra_params["reasoning_effort"] = kwargs["reasoning_effort"]
    if tools:
        extra_params["tools"] = tools
    if "max_tokens" in kwargs:
        extra_params["max_tokens"] = int(kwargs["max_tokens"])
    if "n" in kwargs:
        extra_params["n"] = int(kwargs["n"])

    temperature = float(kwargs.get("temperature", 1.0))

    response = client.chat.completions.create(
        model=resolved_model,
        messages=messages,
        temperature=temperature,
        **extra_params,
    )

    if raw_response:
        return response
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def create_openai_client(model_name: str = "", **kwargs) -> Tuple[OpenAI, str]:
    """
    Create an OpenAI client configured for the appropriate backend.

    For agents that need the raw OpenAI client (e.g. for tool calling /
    function calling), use this instead of chat_completion().

    In "local" mode, handles provider-specific base_url / api_key overrides.
    In "proxy" mode, falls back to local mode since the proxy gateway does
    not support OpenAI-compatible tool calling protocol; prints a warning
    if proxy mode was explicitly requested.

    Args:
        model_name: Model name, possibly with provider prefix.
        **kwargs:   Agent args (may contain model_mode, etc.).

    Returns:
        (client, resolved_model_name) tuple.
    """
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

    # Local mode (or proxy fallback)
    return _resolve_local_client(model_name, **kwargs)


def chat_completion(
    messages: List[Dict],
    model: str,
    **kwargs,
) -> Optional[str]:
    """
    Unified chat completion interface.

    Dispatches to either proxy or local backend based on:
      - kwargs["model_mode"]  (takes precedence)
      - env var MODEL_MODE
      - defaults to "local"

    Args:
        messages: OpenAI-format message list.
        model:    Model name (e.g. "gpt-5", "qwen2-vl").
        **kwargs: Additional config passed from agent_args.

    Returns:
        The model's response text, or None on failure.
    """
    mode = _get_model_mode(**kwargs)

    if mode == "proxy":
        return _proxy_chat_completion(messages, model, **kwargs)
    elif mode == "local":
        return _local_chat_completion(messages, model, **kwargs)
    else:
        raise ValueError(
            f"Unknown model_mode '{mode}'. Expected 'local' or 'proxy'."
        )


def chat_completion_with_tools(
    messages: List[Dict],
    model: str,
    tools: Optional[List[Dict]] = None,
    **kwargs,
) -> Any:
    """
    Unified chat completion interface with tool calling support.

    Unlike chat_completion() which returns plain text, this function
    returns the full response object so callers can inspect tool_calls,
    finish_reason, reasoning_content, etc.

    In local mode:
        Returns the raw OpenAI SDK response object.
        Access via: response.choices[0].message

    In proxy mode:
        Returns the 'data' dict from the proxy gateway response.
        Contains 'message' (text) and optionally 'tool_calls', etc.

    Args:
        messages: OpenAI-format message list.
        model:    Model name.
        tools:    Optional list of tool definitions for function calling.
        **kwargs: Additional config passed from agent_args.

    Returns:
        Full response object (format depends on mode), or None on failure.
    """
    mode = _get_model_mode(**kwargs)

    if mode == "proxy":
        return _proxy_chat_completion_raw(messages, model, tools=tools, **kwargs)
    elif mode == "local":
        return _local_chat_completion(
            messages, model, tools=tools, raw_response=True, **kwargs
        )
    else:
        raise ValueError(
            f"Unknown model_mode '{mode}'. Expected 'local' or 'proxy'."
        )
