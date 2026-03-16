import re
import json
import os
import logging
from typing import Dict, Any

from .base_benchmark import BaseBenchmark

logger = logging.getLogger(__name__)


def extract_numerical_answer(response: str) -> int:
    """
    Extract the final numerical answer from the agent's response.

    Supports multiple formats:
      - ANSWER: 42
      - \\boxed{42}
      - The answer is 42
      - Final answer: 42

    Returns -1 if no valid answer is found.
    """
    if not isinstance(response, str):
        return -1

    # Priority 1: "ANSWER: <number>"
    match = re.search(r"ANSWER:\s*(\d+)", response, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Priority 2: \boxed{<number>}
    match = re.search(r"\\boxed\{(\d+)\}", response)
    if match:
        return int(match.group(1))

    # Priority 3: "the answer is <number>"
    match = re.search(r"the\s+answer\s+is\s+(\d+)", response, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Priority 4: "final answer: <number>" or "final answer is <number>"
    match = re.search(
        r"final\s+answer[:\s]+(?:is\s+)?(\d+)", response, re.IGNORECASE
    )
    if match:
        return int(match.group(1))

    # Fallback: last standalone integer in the response
    all_numbers = re.findall(r"\b(\d+)\b", response)
    if all_numbers:
        return int(all_numbers[-1])

    return -1


class AIME2025Benchmark(BaseBenchmark):
    """AIME 2025 benchmark implementation.

    Loads the 30 problems (AIME I + AIME II) from the HuggingFace dataset
    `math-ai/AIME_2025` and evaluates agent answers via exact integer match.
    """

    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        self.benchmark_name = "aime2025"
        self.requires_sandbox = False
        super().__init__(agent_dir, config, requires_sandbox=self.requires_sandbox)

        # Load dataset from HuggingFace
        self.benchmark = self._load_dataset()

    def _load_dataset(self) -> Dict[str, Any]:
        """Load AIME 2025 dataset from HuggingFace."""
        try:
            from datasets import load_dataset

            dataset = load_dataset("math-ai/AIME_2025", split="train")
        except Exception as first_error:
            logger.warning(
                f"Failed to load math-ai/AIME_2025: {first_error}. "
                "Trying fallback dataset math-ai/aime25..."
            )
            try:
                from datasets import load_dataset

                dataset = load_dataset("math-ai/aime25", split="test")
            except Exception as second_error:
                raise RuntimeError(
                    f"Failed to load AIME 2025 dataset from HuggingFace. "
                    f"Please ensure the `datasets` library is installed "
                    f"(`pip install datasets`) and you have internet access. "
                    f"Errors: {first_error}; {second_error}"
                )

        benchmark = {}
        for row in dataset:
            problem_id = str(row.get("id", row.get("ID", "")))
            task_id = f"aime2025_{problem_id}"
            benchmark[task_id] = {
                "problem": row["problem"],
                "answer": str(row["answer"]).strip(),
            }

        logger.info(f"Loaded {len(benchmark)} AIME 2025 problems from HuggingFace")
        return benchmark

    def evaluate_output(
        self, agent_output: Dict[str, Any], run_id: str
    ) -> Dict[str, Any]:
        """Evaluate agent outputs by exact integer match against ground truth."""
        normalized_output = self._normalize_agent_output(agent_output)
        results = {}

        for task_id, response in normalized_output.items():
            if task_id not in self.benchmark:
                logger.warning(f"Task {task_id} not found in benchmark dataset")
                results[task_id] = {
                    "correct": False,
                    "error": "Task not found in dataset",
                }
                continue

            expected_answer = int(self.benchmark[task_id]["answer"])
            predicted_answer = extract_numerical_answer(str(response))

            result_entry = {
                "correct": predicted_answer == expected_answer,
                "expected": expected_answer,
                "predicted": predicted_answer,
                "raw_response": str(response),
            }

            # Extract conversation history and tool call info from raw agent output
            raw_task_data = agent_output.get(task_id, {})
            if isinstance(raw_task_data, dict) and "metrics" in raw_task_data:
                metrics = raw_task_data["metrics"]
                result_entry["tool_call_count"] = metrics.get("tool_call_count", 0)
                result_entry["has_thinking"] = metrics.get("has_thinking", False)
                conversation_history = metrics.get("conversation_history", [])
                result_entry["conversation_history"] = conversation_history

                # Count successful tool calls by checking if result contains "error"
                successful_tool_calls = 0
                failed_tool_calls = 0
                for turn in conversation_history:
                    if turn.get("role") == "tool" and "result" in turn:
                        result_text = str(turn.get("result", ""))
                        if "error" in result_text.lower():
                            failed_tool_calls += 1
                        else:
                            successful_tool_calls += 1
                result_entry["successful_tool_calls"] = successful_tool_calls
                result_entry["failed_tool_calls"] = failed_tool_calls

            results[task_id] = result_entry

        return results

    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate accuracy, tool usage stats, and identify successful/failed tasks."""
        correct_count = sum(
            1 for result in eval_results.values() if result.get("correct", False)
        )
        total_count = len(eval_results)

        # Tool usage statistics
        tasks_with_tools = sum(
            1 for result in eval_results.values()
            if result.get("tool_call_count", 0) > 0
        )
        total_tool_calls = sum(
            result.get("tool_call_count", 0) for result in eval_results.values()
        )
        total_successful_tool_calls = sum(
            result.get("successful_tool_calls", 0) for result in eval_results.values()
        )
        total_failed_tool_calls = sum(
            result.get("failed_tool_calls", 0) for result in eval_results.values()
        )

        return {
            "accuracy": correct_count / total_count if total_count > 0 else 0.0,
            "tasks_with_tool_calls": tasks_with_tools,
            "total_tool_calls": total_tool_calls,
            "successful_tool_calls": total_successful_tool_calls,
            "failed_tool_calls": total_failed_tool_calls,
            "successful_tasks": [
                task_id
                for task_id, result in eval_results.items()
                if result.get("correct", False)
            ],
            "failed_tasks": [
                task_id
                for task_id, result in eval_results.items()
                if not result.get("correct", False)
            ],
        }
