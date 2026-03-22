#!/usr/bin/env python3
import argparse
import base64
import json
import mimetypes
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI


DEFAULT_PROXY_OPENAI_BASE_URL = "https://llm-chat-api.alibaba-inc.com/openai"
DEFAULT_PROXY_URL = "https://llm-chat-api.alibaba-inc.com/v1/api/chat"
DEFAULT_MODEL = "claude-opus-4-6"
DEFAULT_APP = "model_train_vlm"
DEFAULT_QUOTA_ID = "dd95187c-29dd-464d-9b96-8f62e6ab8eb5"
DEFAULT_ACCESS_KEY = "9101ac974ab20f60f668dcf099bc6a10"
DEFAULT_USER_ID = "506759"


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _mime_type(path: Path) -> str:
    return mimetypes.guess_type(str(path))[0] or "image/png"


def _data_url(path: Path) -> str:
    payload = base64.b64encode(_read_bytes(path)).decode("ascii")
    return f"data:{_mime_type(path)};base64,{payload}"


def _plain_base64(path: Path) -> str:
    return base64.b64encode(_read_bytes(path)).decode("ascii")


def _extra_body() -> Dict[str, Any]:
    return {
        "app": os.getenv("PROXY_APP", DEFAULT_APP),
        "quota_id": os.getenv("QUOTA_ID", DEFAULT_QUOTA_ID),
        "user_id": os.getenv("PROXY_USER_ID", DEFAULT_USER_ID),
        "access_key": os.getenv("ACCESS_KEY", DEFAULT_ACCESS_KEY),
    }


def _openai_client() -> OpenAI:
    return OpenAI(
        base_url=os.getenv("PROXY_OPENAI_BASE_URL", DEFAULT_PROXY_OPENAI_BASE_URL),
        api_key=os.getenv("PROXY_OPENAI_API_KEY", os.getenv("PROXY_TOKEN", "")),
    )


def _build_openai_messages(image_block: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Describe the dominant object or scene in one short sentence. "
                        "If you cannot read the image, say exactly IMAGE_UNREADABLE."
                    ),
                },
                image_block,
            ],
        }
    ]


def _run_openai_probe(name: str, image_block: Dict[str, Any], model: str, timeout: float) -> Dict[str, Any]:
    result: Dict[str, Any] = {"probe": name, "mode": "openai"}
    try:
        response = _openai_client().chat.completions.create(
            model=model,
            messages=_build_openai_messages(image_block),
            max_tokens=128,
            temperature=0.0,
            timeout=timeout,
            extra_body=_extra_body(),
        )
        choice = response.choices[0]
        result.update(
            {
                "ok": True,
                "finish_reason": choice.finish_reason,
                "content": choice.message.content,
                "tool_calls": [tc.function.name for tc in (choice.message.tool_calls or [])],
            }
        )
    except Exception as exc:
        result.update(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    return result


def _run_raw_openai_probe(
    name: str,
    image_block: Dict[str, Any],
    model: str,
    timeout: float,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"probe": name, "mode": "raw_openai_http"}
    url = os.getenv("PROXY_OPENAI_BASE_URL", DEFAULT_PROXY_OPENAI_BASE_URL).rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('PROXY_OPENAI_API_KEY', os.getenv('PROXY_TOKEN', ''))}",
    }
    payload = {
        "model": model,
        "messages": _build_openai_messages(image_block),
        "max_tokens": 128,
        "temperature": 0.0,
        **{"extra_body": _extra_body()},
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        result["status_code"] = response.status_code
        try:
            result["response_json"] = response.json()
        except Exception:
            result["response_text"] = response.text
        result["ok"] = response.ok
    except Exception as exc:
        result.update(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    return result


def _run_legacy_proxy_probe(name: str, image_block: Dict[str, Any], model: str, timeout: float) -> Dict[str, Any]:
    result: Dict[str, Any] = {"probe": name, "mode": "legacy_proxy"}
    url = os.getenv("PROXY_URL", DEFAULT_PROXY_URL)
    token = os.getenv("PROXY_TOKEN", "")
    payload = {
        "model": model,
        "prompt": _build_openai_messages(image_block),
        "tag": os.getenv("PROXY_TAG", "image_format_probe"),
        "quota_id": os.getenv("QUOTA_ID", DEFAULT_QUOTA_ID),
        "app": os.getenv("PROXY_APP", DEFAULT_APP),
        "params": {
            "max_new_tokens": 128,
            "temperature": 0.0,
        },
        "user_id": os.getenv("PROXY_USER_ID", DEFAULT_USER_ID),
        "access_key": os.getenv("ACCESS_KEY", DEFAULT_ACCESS_KEY),
    }
    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json", "token": token},
            data=json.dumps(payload),
            timeout=timeout,
        )
        result["status_code"] = response.status_code
        try:
            result["response_json"] = response.json()
        except Exception:
            result["response_text"] = response.text
        result["ok"] = response.ok
    except Exception as exc:
        result.update(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe which Claude image formats the proxy accepts.")
    parser.add_argument("--image-path", required=True, help="Local image file to probe with.")
    parser.add_argument("--remote-url", help="Optional remote image URL to probe.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    image_path = Path(args.image_path).expanduser().resolve()
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    probes: List[Dict[str, Any]] = []

    if args.remote_url:
        probes.append(
            _run_openai_probe(
                "openai_remote_url",
                {"type": "image_url", "image_url": {"url": args.remote_url}},
                args.model,
                args.timeout,
            )
        )
        probes.append(
            _run_raw_openai_probe(
                "raw_openai_remote_url",
                {"type": "image_url", "image_url": {"url": args.remote_url}},
                args.model,
                args.timeout,
            )
        )
        probes.append(
            _run_legacy_proxy_probe(
                "legacy_remote_url",
                {"type": "image_url", "image_url": {"url": args.remote_url}},
                args.model,
                args.timeout,
            )
        )

    probes.append(
        _run_openai_probe(
            "openai_data_url",
            {"type": "image_url", "image_url": {"url": _data_url(image_path)}},
            args.model,
            args.timeout,
        )
    )
    probes.append(
        _run_raw_openai_probe(
            "raw_openai_data_url",
            {"type": "image_url", "image_url": {"url": _data_url(image_path)}},
            args.model,
            args.timeout,
        )
    )
    probes.append(
        _run_legacy_proxy_probe(
            "legacy_data_url",
            {"type": "image_url", "image_url": {"url": _data_url(image_path)}},
            args.model,
            args.timeout,
        )
    )

    probes.append(
        _run_openai_probe(
            "openai_plain_base64_in_url_field",
            {"type": "image_url", "image_url": {"url": _plain_base64(image_path)}},
            args.model,
            args.timeout,
        )
    )
    probes.append(
        _run_raw_openai_probe(
            "raw_openai_plain_base64_in_url_field",
            {"type": "image_url", "image_url": {"url": _plain_base64(image_path)}},
            args.model,
            args.timeout,
        )
    )

    print(json.dumps({"image_path": str(image_path), "probes": probes}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
