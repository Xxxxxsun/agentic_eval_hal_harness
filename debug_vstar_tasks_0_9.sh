#!/usr/bin/env bash

set -euo pipefail

: "${PROXY_OPENAI_API_KEY:?PROXY_OPENAI_API_KEY is required}"

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${ROOT_DIR}/vstar_task_debug_logs"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WEAVE_DISABLED="${WEAVE_DISABLED:-true}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export WANDB_MODE="${WANDB_MODE:-offline}"

export MODEL_MODE="${MODEL_MODE:-proxy}"
export PROXY_OPENAI_BASE_URL="${PROXY_OPENAI_BASE_URL:-https://llm-chat-api.alibaba-inc.com/openai}"
export PROXY_TOKEN="${PROXY_TOKEN:-$PROXY_OPENAI_API_KEY}"
export QUOTA_ID="${QUOTA_ID:-dd95187c-29dd-464d-9b96-8f62e6ab8eb5}"
export ACCESS_KEY="${ACCESS_KEY:-9101ac974ab20f60f668dcf099bc6a10}"
export PROXY_USER_ID="${PROXY_USER_ID:-506759}"

mkdir -p "${LOG_DIR}"

for TASK_ID in $(seq 0 9); do
  echo "==== RUN TASK ${TASK_ID} ===="
  TASK_ID="${TASK_ID}" python - <<'PY' > "${LOG_DIR}/task_${TASK_ID}.log" 2>&1
import json
import os

from hal.benchmarks.vstar_bench import VStarBenchBenchmark
from agents.vstar_bench_agent.main import run
import agents.common.vqa_runtime as vqa_runtime
from agents.common.vqa_mcp_tools import VQAToolRegistry

task_id = os.environ["TASK_ID"]

benchmark = VStarBenchBenchmark("agents", {})
task = benchmark.get_dataset()[task_id]

print("=== TASK ===")
print(json.dumps(task, ensure_ascii=False, indent=2))

orig_invoke = vqa_runtime._invoke_completion
orig_execute_tool = VQAToolRegistry.execute_tool

call_idx = {"n": 0}

def summarize_messages(messages):
    out = []
    for i, msg in enumerate(messages):
        item = {"idx": i, "role": msg.get("role")}
        content = msg.get("content")
        if isinstance(content, list):
            blocks = []
            for j, block in enumerate(content):
                if isinstance(block, dict) and block.get("type") == "image_url":
                    url = block.get("image_url", {}).get("url", "")
                    blocks.append(
                        {
                            "block_idx": j,
                            "type": "image_url",
                            "url_prefix": url[:120],
                            "url_len": len(url),
                        }
                    )
                else:
                    blocks.append(
                        {
                            "block_idx": j,
                            "type": block.get("type") if isinstance(block, dict) else type(block).__name__,
                            "text_preview": (
                                block.get("text", "")[:200] if isinstance(block, dict) else str(block)[:200]
                            ),
                        }
                    )
            item["content_summary"] = blocks
        else:
            item["content_preview"] = str(content)[:300]
        if "tool_calls" in msg:
            item["tool_calls"] = msg["tool_calls"]
        out.append(item)
    return out

def debug_invoke_completion(**kwargs):
    idx = call_idx["n"]
    call_idx["n"] += 1
    print(f"\n=== COMPLETION CALL {idx} ===")
    print(json.dumps(summarize_messages(kwargs["messages"]), ensure_ascii=False, indent=2))

    resp = orig_invoke(**kwargs)

    print(f"\n=== RAW RESPONSE {idx} TYPE ===")
    print(type(resp))
    print(f"=== RAW RESPONSE {idx} REPR ===")
    print(repr(resp))

    try:
        if isinstance(resp, dict):
            print(f"=== RAW RESPONSE {idx} JSON ===")
            print(json.dumps(resp, ensure_ascii=False, indent=2))
    except Exception as exc:
        print("RAW_JSON_ERROR:", repr(exc))

    try:
        finish_reason, assistant_message, thinking_content = vqa_runtime._normalize_model_response(
            resp,
            kwargs.get("model_mode", os.getenv("MODEL_MODE", "local")),
            idx,
        )
        print(f"=== NORMALIZED {idx} ===")
        print("FINISH_REASON:", repr(finish_reason))
        print("CONTENT:", repr(getattr(assistant_message, "content", None)))
        print("THINKING:", repr(thinking_content))
        tool_calls = getattr(assistant_message, "tool_calls", None)
        print("TOOL_CALLS:", repr(tool_calls))
        if tool_calls:
            for t_i, tc in enumerate(tool_calls):
                print(f"TOOL_CALL_{t_i}_NAME:", tc.function.name)
                print(f"TOOL_CALL_{t_i}_ARGS:", tc.function.arguments)
    except Exception as exc:
        print("NORMALIZE_ERROR:", repr(exc))

    return resp

def debug_execute_tool(self, tool_name, arguments):
    print("\n=== TOOL EXECUTION ===")
    print("TOOL_NAME:", tool_name)
    print("ARGUMENTS:", json.dumps(arguments, ensure_ascii=False))
    result = orig_execute_tool(self, tool_name, arguments)
    print("TOOL_RESULT:", json.dumps(result, ensure_ascii=False))
    return result

vqa_runtime._invoke_completion = debug_invoke_completion
VQAToolRegistry.execute_tool = debug_execute_tool

result = run(
    {task_id: task},
    model_name="claude-opus-4-6",
    model_mode="proxy",
    benchmark_name="vstar_bench",
    enable_tools=True,
    code_executor="local",
    debug=True,
    timeout=400,
    max_tokens=1024,
    max_iterations=8,
    empty_answer_retries=3,
)

print("\n=== FINAL RESULT ===")
print(json.dumps(result, ensure_ascii=False, indent=2))
PY
  echo "saved -> ${LOG_DIR}/task_${TASK_ID}.log"
done

echo "all logs saved under ${LOG_DIR}"
