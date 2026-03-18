import re
import random
import logging
from typing import Dict, Any, List

from .base_benchmark import BaseBenchmark

logger = logging.getLogger(__name__)

CHOICE_LABELS = ["A", "B", "C", "D"]


def extract_choice_answer(response: str) -> str:
    """
    Extract the chosen option letter (A/B/C/D) from the agent's response.

    Supports multiple formats:
      - ANSWER: A
      - \\boxed{A}
      - The answer is A
      - Final answer: B
      - (C)
      - Just a standalone letter at the end

    Returns empty string if no valid choice is found.
    """
    if not isinstance(response, str):
        return ""

    response_upper = response.upper()

    # Priority 1: "ANSWER: X"
    match = re.search(r"ANSWER:\s*([A-D])\b", response_upper)
    if match:
        return match.group(1)

    # Priority 2: \boxed{X}
    match = re.search(r"\\boxed\{([A-Da-d])\}", response)
    if match:
        return match.group(1).upper()

    # Priority 3: "the answer is X"
    match = re.search(r"the\s+answer\s+is\s+([A-D])\b", response_upper)
    if match:
        return match.group(1)

    # Priority 4: "final answer: X" or "final answer is X"
    match = re.search(r"final\s+answer[:\s]+(?:is\s+)?([A-D])\b", response_upper)
    if match:
        return match.group(1)

    # Priority 5: Parenthesized choice like "(A)" or "(B)"
    matches = re.findall(r"\(([A-D])\)", response_upper)
    if matches:
        return matches[-1]

    # Fallback: last standalone A/B/C/D in the response
    matches = re.findall(r"\b([A-D])\b", response_upper)
    if matches:
        return matches[-1]

    return ""


def _shuffle_choices(
    correct_answer: str,
    incorrect_answers: List[str],
    seed: int,
) -> Dict[str, Any]:
    """
    Shuffle the four answer choices with a deterministic seed.

    Returns a dict with:
      - choices: {"A": "...", "B": "...", "C": "...", "D": "..."}
      - correct_label: the letter corresponding to the correct answer
    """
    all_answers = [correct_answer] + incorrect_answers
    rng = random.Random(seed)
    rng.shuffle(all_answers)

    choices = {}
    correct_label = ""
    for idx, answer_text in enumerate(all_answers):
        label = CHOICE_LABELS[idx]
        choices[label] = answer_text
        if answer_text == correct_answer:
            correct_label = label

    return {"choices": choices, "correct_label": correct_label}


class GPQADiamondBenchmark(BaseBenchmark):
    """GPQA-Diamond benchmark implementation.

    Loads the 198 graduate-level multiple-choice science questions from the
    HuggingFace dataset `Idavidrein/gpqa` (gpqa_diamond subset) and evaluates
    agent answers via exact letter match (A/B/C/D).
    """

    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        self.benchmark_name = "gpqa_diamond"
        self.requires_sandbox = False
        super().__init__(agent_dir, config, requires_sandbox=self.requires_sandbox)

        self.benchmark = self._load_dataset()

    def _load_dataset(self) -> Dict[str, Any]:
        """Load GPQA-Diamond dataset from HuggingFace.

        Tries the official gated dataset first (requires HF_TOKEN).
        Falls back to public community datasets if authentication fails.

        Handles two dataset formats:
          - Official (Idavidrein/gpqa): separate columns for question, correct
            answer, incorrect answers, and subdomain. Choices are shuffled
            deterministically.
          - Pre-processed (fingertap/GPQA-Diamond): only 'question' (with
            choices embedded in text) and 'answer' (a letter A-D).
        """
        from datasets import load_dataset

        dataset = None
        dataset_source = None

        # Attempt 1: Official gated dataset (requires HuggingFace authentication)
        try:
            dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
            dataset_source = "Idavidrein/gpqa"
        except Exception as first_error:
            logger.warning(
                f"Failed to load official GPQA dataset (may require HF_TOKEN): {first_error}. "
                "Trying public fallback datasets..."
            )

        # Attempt 2: Public community dataset (fingertap/GPQA-Diamond)
        if dataset is None:
            try:
                dataset = load_dataset("fingertap/GPQA-Diamond", split="test")
                dataset_source = "fingertap/GPQA-Diamond"
            except Exception as second_error:
                logger.warning(
                    f"Failed to load fingertap/GPQA-Diamond: {second_error}. "
                    "Trying another fallback..."
                )

        # Attempt 3: Another public dataset (nikhilchandak/GPQA-diamond-free)
        if dataset is None:
            try:
                dataset = load_dataset("nikhilchandak/GPQA-diamond-free", split="test")
                dataset_source = "nikhilchandak/GPQA-diamond-free"
            except Exception as third_error:
                raise RuntimeError(
                    f"Failed to load GPQA-Diamond from any source. "
                    f"Options: 1) Set HF_TOKEN env var for official dataset access, "
                    f"2) Run `huggingface-cli login`, "
                    f"3) Ensure internet access. "
                    f"Last error: {third_error}"
                )

        logger.info(f"Loading GPQA-Diamond from: {dataset_source}")

        column_names = set(dataset.column_names)
        is_preprocessed = (
            "question" in column_names
            and "answer" in column_names
            and "Correct Answer" not in column_names
            and "correct_answer" not in column_names
        )

        if is_preprocessed:
            return self._parse_preprocessed_dataset(dataset, dataset_source)
        return self._parse_official_dataset(dataset, dataset_source, column_names)

    def _parse_preprocessed_dataset(
        self, dataset, dataset_source: str
    ) -> Dict[str, Any]:
        """Parse a pre-processed dataset where choices are already embedded
        in the question text (as A./B./C./D.) and the answer is a letter.

        Example question text:
            "What is ...?\n\nA. Option one.\nB. Option two.\nC. ...\nD. ..."
        """
        benchmark = {}
        for row_idx, row in enumerate(dataset):
            task_id = f"gpqa_diamond_{row_idx}"
            question_text = row["question"]
            correct_label = row["answer"].strip().upper()

            # Choices are already embedded in question_text, so we pass an
            # empty dict. The agent will receive the full question_text as-is.
            benchmark[task_id] = {
                "question": question_text,
                "choices": {},
                "answer": correct_label,
                "subdomain": "Unknown",
            }

        logger.info(
            f"Loaded {len(benchmark)} GPQA-Diamond problems from {dataset_source} "
            "(pre-processed format)"
        )
        return benchmark

    def _parse_official_dataset(
        self, dataset, dataset_source: str, column_names: set
    ) -> Dict[str, Any]:
        """Parse the official dataset with separate columns for question,
        correct answer, incorrect answers, and subdomain."""
        question_col = "Question" if "Question" in column_names else "question"
        correct_col = (
            "Correct Answer" if "Correct Answer" in column_names
            else "correct_answer" if "correct_answer" in column_names
            else "Correct_Answer"
        )
        incorrect_cols = []
        for variant in [
            ["Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"],
            ["incorrect_answer_1", "incorrect_answer_2", "incorrect_answer_3"],
            ["Incorrect_Answer_1", "Incorrect_Answer_2", "Incorrect_Answer_3"],
        ]:
            if variant[0] in column_names:
                incorrect_cols = variant
                break

        if not incorrect_cols:
            raise RuntimeError(
                f"Cannot find incorrect answer columns in dataset {dataset_source}. "
                f"Available columns: {column_names}"
            )

        subdomain_col = (
            "Subdomain" if "Subdomain" in column_names
            else "subdomain" if "subdomain" in column_names
            else None
        )

        benchmark = {}
        for row_idx, row in enumerate(dataset):
            task_id = f"gpqa_diamond_{row_idx}"

            correct_answer = row[correct_col]
            incorrect_answers = [row[col] for col in incorrect_cols]

            shuffled = _shuffle_choices(correct_answer, incorrect_answers, seed=row_idx)

            benchmark[task_id] = {
                "question": row[question_col],
                "choices": shuffled["choices"],
                "answer": shuffled["correct_label"],
                "subdomain": row.get(subdomain_col, "Unknown") if subdomain_col else "Unknown",
            }

        logger.info(
            f"Loaded {len(benchmark)} GPQA-Diamond problems from HuggingFace"
        )
        return benchmark

    def evaluate_output(
        self, agent_output: Dict[str, Any], run_id: str
    ) -> Dict[str, Any]:
        """Evaluate agent outputs by exact letter match against ground truth."""
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

            expected_label = self.benchmark[task_id]["answer"]
            predicted_label = extract_choice_answer(str(response))

            result_entry = {
                "correct": predicted_label == expected_label,
                "expected": expected_label,
                "predicted": predicted_label,
                "subdomain": self.benchmark[task_id]["subdomain"],
                "raw_response": str(response),
            }

            # Extract conversation history and tool call info from raw agent output
            raw_task_data = agent_output.get(task_id, {})
            if isinstance(raw_task_data, dict) and "metrics" in raw_task_data:
                metrics = raw_task_data["metrics"]
                result_entry["tool_call_count"] = metrics.get("tool_call_count", 0)
                conversation_history = metrics.get("conversation_history", [])
                result_entry["conversation_history"] = conversation_history

                # Count failed/successful tool calls based on sandbox_error_types
                sandbox_error_types = metrics.get("sandbox_error_types", [])
                failed_tool_calls = len(sandbox_error_types)
                total_calls = metrics.get("tool_call_count", 0)
                successful_tool_calls = total_calls - failed_tool_calls
                result_entry["successful_tool_calls"] = successful_tool_calls
                result_entry["failed_tool_calls"] = failed_tool_calls
                result_entry["sandbox_error_types"] = sandbox_error_types

            results[task_id] = result_entry

        return results

    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall and per-subdomain accuracy, plus tool usage stats."""
        correct_count = sum(
            1 for result in eval_results.values() if result.get("correct", False)
        )
        total_count = len(eval_results)

        # Per-subdomain breakdown
        subdomain_stats: Dict[str, Dict[str, int]] = {}
        for result in eval_results.values():
            subdomain = result.get("subdomain", "Unknown")
            if subdomain not in subdomain_stats:
                subdomain_stats[subdomain] = {"correct": 0, "total": 0}
            subdomain_stats[subdomain]["total"] += 1
            if result.get("correct", False):
                subdomain_stats[subdomain]["correct"] += 1

        subdomain_accuracy = {
            subdomain: stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            for subdomain, stats in subdomain_stats.items()
        }

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

        # Aggregate sandbox error type counts across all tasks
        sandbox_error_type_counts: Dict[str, int] = {}
        for result in eval_results.values():
            for error_type in result.get("sandbox_error_types", []):
                sandbox_error_type_counts[error_type] = (
                    sandbox_error_type_counts.get(error_type, 0) + 1
                )

        return {
            "accuracy": correct_count / total_count if total_count > 0 else 0.0,
            "tasks_with_tool_calls": tasks_with_tools,
            "total_tool_calls": total_tool_calls,
            "successful_tool_calls": total_successful_tool_calls,
            "failed_tool_calls": total_failed_tool_calls,
            "sandbox_error_type_counts": sandbox_error_type_counts,
            "subdomain_accuracy": subdomain_accuracy,
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
