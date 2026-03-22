#!/usr/bin/env python3
import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_hal_command(
    *,
    benchmark: str,
    agent_name: str,
    agent_dir: str,
    agent_function: str,
    model_name: str,
    model_mode: str,
    enable_tools: bool,
    code_executor: str,
    timeout: int,
    max_tokens: int,
    max_iterations: int,
    empty_answer_retries: int,
    max_tasks: Optional[int],
    max_concurrent: int,
    task_ids: Optional[List[str]],
    debug: bool,
    extra_agent_args: List[str],
) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "hal.cli",
        "--agent_name",
        agent_name,
        "--agent_dir",
        agent_dir,
        "--agent_function",
        agent_function,
        "--benchmark",
        benchmark,
        "-A",
        f"model_name={model_name}",
        "-A",
        f"model_mode={model_mode}",
        "-A",
        f"enable_tools={'true' if enable_tools else 'false'}",
        "-A",
        f"code_executor={code_executor}",
        "-A",
        f"timeout={timeout}",
        "-A",
        f"max_tokens={max_tokens}",
        "-A",
        f"max_iterations={max_iterations}",
        "-A",
        f"empty_answer_retries={empty_answer_retries}",
        "--max_concurrent",
        str(max_concurrent),
    ]
    if debug:
        command.extend(["-A", "debug=true"])
    if max_tasks is not None and task_ids is None:
        command.extend(["--max_tasks", str(max_tasks)])
    if task_ids:
        command.extend(["--task_ids", ",".join(task_ids)])
    for agent_arg in extra_agent_args:
        command.extend(["-A", agent_arg])
    return command


def _find_latest_upload_json(benchmark: str, *, since_ts: Optional[float]) -> Path:
    pattern = str(REPO_ROOT / "results" / benchmark / "**" / "*_UPLOAD.json")
    candidates = [Path(path) for path in glob.glob(pattern, recursive=True)]
    if since_ts is not None:
        candidates = [path for path in candidates if path.stat().st_mtime >= since_ts]
    if not candidates:
        raise FileNotFoundError(f"No _UPLOAD.json found for benchmark={benchmark}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _extract_empty_task_ids(upload_json: Path) -> List[str]:
    with upload_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    empty_ids: List[str] = []
    for task_id, record in payload.items():
        if not isinstance(record, dict):
            continue
        candidate_fields = (
            "answer",
            "predicted",
            "raw_response",
            "vision_predicted",
            "no_image_predicted",
            "base_llm_predicted",
        )
        if any(field in record and str(record.get(field, "")) == "" for field in candidate_fields):
            empty_ids.append(str(task_id))
    return sorted(empty_ids)


def _run_command(command: List[str]) -> None:
    print("=== RUNNING ===")
    print(" ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a VQA benchmark once, then extract empty-answer tasks from the latest "
            "_UPLOAD.json and rerun only those tasks."
        )
    )
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--agent-name")
    parser.add_argument("--agent-dir")
    parser.add_argument("--agent-function", default="main.run")
    parser.add_argument("--model-mode", default="proxy")
    parser.add_argument("--enable-tools", default="true")
    parser.add_argument("--code-executor", default="local")
    parser.add_argument("--timeout", type=int, default=400)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--empty-answer-retries", type=int, default=3)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--upload-json",
        help="If provided, skip the first run and rerun empty-answer tasks from this _UPLOAD.json file.",
    )
    parser.add_argument(
        "--agent-arg",
        action="append",
        default=[],
        help="Additional -A key=value arguments forwarded to hal.cli. Can be repeated.",
    )
    args = parser.parse_args()

    benchmark = args.benchmark
    model_name = args.model_name
    agent_name = args.agent_name or model_name
    agent_dir = args.agent_dir or f"agents/{benchmark}_agent"
    enable_tools = str(args.enable_tools).strip().lower() in {"1", "true", "yes", "y", "on"}

    upload_json_path: Path
    if args.upload_json:
        upload_json_path = Path(args.upload_json).expanduser().resolve()
        if not upload_json_path.exists():
            raise FileNotFoundError(f"Upload json not found: {upload_json_path}")
    else:
        started_at = time.time()
        first_run_cmd = _build_hal_command(
            benchmark=benchmark,
            agent_name=agent_name,
            agent_dir=agent_dir,
            agent_function=args.agent_function,
            model_name=model_name,
            model_mode=args.model_mode,
            enable_tools=enable_tools,
            code_executor=args.code_executor,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            max_iterations=args.max_iterations,
            empty_answer_retries=args.empty_answer_retries,
            max_tasks=args.max_tasks,
            max_concurrent=args.max_concurrent,
            task_ids=None,
            debug=args.debug,
            extra_agent_args=args.agent_arg,
        )
        _run_command(first_run_cmd)
        upload_json_path = _find_latest_upload_json(benchmark, since_ts=started_at)

    print(f"=== USING UPLOAD JSON ===\n{upload_json_path}")
    empty_task_ids = _extract_empty_task_ids(upload_json_path)
    print(f"=== EMPTY TASK IDS ({len(empty_task_ids)}) ===")
    print(",".join(empty_task_ids) if empty_task_ids else "(none)")

    if not empty_task_ids:
        return

    rerun_cmd = _build_hal_command(
        benchmark=benchmark,
        agent_name=agent_name,
        agent_dir=agent_dir,
        agent_function=args.agent_function,
        model_name=model_name,
        model_mode=args.model_mode,
        enable_tools=enable_tools,
        code_executor=args.code_executor,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        max_iterations=args.max_iterations,
        empty_answer_retries=args.empty_answer_retries,
        max_tasks=None,
        max_concurrent=args.max_concurrent,
        task_ids=empty_task_ids,
        debug=args.debug,
        extra_agent_args=args.agent_arg,
    )
    _run_command(rerun_cmd)


if __name__ == "__main__":
    main()
