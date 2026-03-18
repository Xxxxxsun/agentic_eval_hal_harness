import os
import logging
from typing import Dict, Any

from openai import OpenAI

from .base_benchmark import BaseBenchmark

logger = logging.getLogger(__name__)

ANSWER_AUTOGRADER_PROMPT = """\
# System Role: Deterministic Mathematical Autograder

You are a precise, automated grading system. Your sole function is to determine \
if the final answer provided in the Model Solution is mathematically equivalent \
to the Golden Answer. You must NOT grade the reasoning or steps, only the final result.

# 1. Grading Guidelines (Equivalence Rules)

Equivalence is mandatory for a correct grade. You must rigorously verify if the \
answers represent the exact same mathematical value or expression, even if the \
format differs.

- **Algebraic Equivalence:** e.g., 'n(n+1)/2' is equivalent to 'n^2/2 + n/2'. \
You must verify the algebra.
- **Numerical Equivalence:** e.g., '1/2' is equivalent to '0.5'; 'sqrt(2)/2' is \
equivalent to '1/sqrt(2)'.
- **Set/List Equivalence:** Unless specified as an ordered tuple/vector, the order \
of elements does not matter (e.g., {{1, 2}} is equivalent to {{2, 1}}).
- **Partial Credit:** No partial credit is allowed. If the answer is incomplete or \
partially incorrect, it is incorrect.
- **No Answers:** If no clear, unambiguous final answer can be extracted, the \
solution must be graded as incorrect.

# 2. Output Protocol (Strict Compliance Required)

You must execute the task using a two-part structure.

**Part 1: Analysis (Chain-of-Thought)**
You MUST perform your analysis within <thinking></thinking> tags. This section \
details your reasoning process and must follow these steps sequentially:
1. **Golden Answer:** State the Golden Answer.
2. **Extracted Model Answer:** State the extracted answer from the model solution. \
If none found, state "No clear final answer found."
3. **Equivalence Analysis:** Compare the two answers using the Grading Guidelines. \
Detail the steps taken to verify mathematical equivalence. You must actively try \
to prove they are the same before concluding they are different.
4. **Conclusion:** State the final determination ("Correct" or "Incorrect").

**Part 2: Final Grade**
Immediately following the closing </thinking> tag, output ONLY the final grade.
- If Correct: \\boxed{{Correct}}
- If Incorrect: \\boxed{{Incorrect}}

CRITICAL CONSTRAINT: Do not add any text outside the <thinking> tags or the \
final \\boxed{{}} output.

# 3. Input Data

Problem:
{problem}

Model Solution:
{model_solution}

Golden Answer:
{golden_answer}
"""


def grade_answer_with_llm(
    client: OpenAI,
    grader_model: str,
    problem: str,
    model_solution: str,
    golden_answer: str,
) -> bool:
    """Use an LLM as AnswerAutoGrader to judge semantic equivalence.

    Returns True if the model's answer is judged correct, False otherwise.
    """
    prompt = ANSWER_AUTOGRADER_PROMPT.format(
        problem=problem,
        model_solution=model_solution,
        golden_answer=golden_answer,
    )

    try:
        response = client.chat.completions.create(
            model=grader_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        grader_output = response.choices[0].message.content or ""

        # Extract verdict from \boxed{Correct} or \boxed{Incorrect}
        grader_output_lower = grader_output.lower()
        if "\\boxed{correct}" in grader_output_lower:
            return True
        if "\\boxed{incorrect}" in grader_output_lower:
            return False

        # Fallback: look for the word in the last line
        last_line = grader_output.strip().split("\n")[-1].lower()
        return "correct" in last_line and "incorrect" not in last_line

    except Exception as grading_error:
        logger.error(f"LLM grading failed: {grading_error}")
        return False


class IMOAnswerBenchBenchmark(BaseBenchmark):
    """IMO-AnswerBench benchmark implementation.

    Loads 400 Olympiad-level short-answer problems from the HuggingFace dataset
    `OpenEvals/IMO-AnswerBench`. Answers can be integers, LaTeX expressions,
    polynomials, intervals, etc.

    Evaluation uses an LLM-based AnswerAutoGrader (following the original paper)
    to judge semantic equivalence between the model's answer and the ground truth.
    """

    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        self.benchmark_name = "imo_answerbench"
        self.requires_sandbox = False
        super().__init__(agent_dir, config, requires_sandbox=self.requires_sandbox)

        self.benchmark = self._load_dataset()

        # Grader model: configurable via env or config (no default, must be set)
        if isinstance(config, dict):
            self.grader_model = config.get(
                "grader_model", os.getenv("GRADER_MODEL")
            )
        else:
            self.grader_model = os.getenv("GRADER_MODEL")

        if not self.grader_model:
            raise ValueError(
                "GRADER_MODEL must be set via environment variable or config. "
                "Set GRADER_MODEL env var or pass grader_model in config."
            )

    def _load_dataset(self) -> Dict[str, Any]:
        """Load IMO-AnswerBench dataset from HuggingFace."""
        try:
            from datasets import load_dataset

            dataset = load_dataset("OpenEvals/IMO-AnswerBench", split="train")
        except Exception as load_error:
            raise RuntimeError(
                f"Failed to load IMO-AnswerBench from HuggingFace. "
                f"Ensure the `datasets` library is installed "
                f"(`pip install datasets`) and you have internet access. "
                f"Error: {load_error}"
            )

        benchmark = {}
        for row in dataset:
            problem_id = row["Problem ID"]
            task_id = f"imo_answerbench_{problem_id}"
            benchmark[task_id] = {
                "problem": row["Problem"],
                "answer": row["Short Answer"],
                "category": row["Category"],
                "subcategory": row["Subcategory"],
                "source": row["Source"],
            }

        logger.info(
            f"Loaded {len(benchmark)} IMO-AnswerBench problems from HuggingFace"
        )
        return benchmark

    def evaluate_output(
        self, agent_output: Dict[str, Any], run_id: str
    ) -> Dict[str, Any]:
        """Evaluate agent outputs using LLM-based AnswerAutoGrader."""
        normalized_output = self._normalize_agent_output(agent_output)

        client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
        )

        results = {}
        for task_id, response in normalized_output.items():
            if task_id not in self.benchmark:
                logger.warning(f"Task {task_id} not found in benchmark dataset")
                results[task_id] = {
                    "correct": False,
                    "error": "Task not found in dataset",
                }
                continue

            task_data = self.benchmark[task_id]
            golden_answer = task_data["answer"]
            problem_text = task_data["problem"]

            is_correct = grade_answer_with_llm(
                client=client,
                grader_model=self.grader_model,
                problem=problem_text,
                model_solution=str(response),
                golden_answer=golden_answer,
            )

            result_entry = {
                "correct": is_correct,
                "expected": golden_answer,
                "category": task_data["category"],
                "subcategory": task_data["subcategory"],
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
        """Calculate overall and per-category accuracy, plus tool usage stats."""
        correct_count = sum(
            1 for result in eval_results.values() if result.get("correct", False)
        )
        total_count = len(eval_results)

        # Per-category breakdown
        category_stats: Dict[str, Dict[str, int]] = {}
        for result in eval_results.values():
            category = result.get("category", "Unknown")
            if category not in category_stats:
                category_stats[category] = {"correct": 0, "total": 0}
            category_stats[category]["total"] += 1
            if result.get("correct", False):
                category_stats[category]["correct"] += 1

        category_accuracy = {
            category: stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            for category, stats in category_stats.items()
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
            "category_accuracy": category_accuracy,
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
