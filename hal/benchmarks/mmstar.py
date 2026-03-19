import logging
from typing import Any, Dict, Iterable, List

from ._benchmark_utils import (
    attach_agent_metrics,
    compute_group_accuracy,
    evaluate_choice_or_text_response,
    extract_choices_from_row,
    extract_serializable_metadata,
    pick_first,
    prepare_task_media,
    summarize_tool_metrics,
)
from .base_benchmark import BaseBenchmark

logger = logging.getLogger(__name__)

BREAKDOWN_FIELDS = ("category", "l2_category", "domain", "source")
REQUIRED_OUTPUT_FIELDS = ("vision_answer", "no_image_answer", "base_llm_answer")


def parse_mmstar_row(row: Dict[str, Any], row_idx: int) -> Dict[str, Any]:
    task_id = str(pick_first(row, ("index", "id", "question_id", "qid"), default=f"mmstar_{row_idx}"))
    question = pick_first(
        row,
        ("question", "text", "prompt", "query", "instruction"),
        default="",
    )
    answer = pick_first(
        row,
        ("answer", "gt_answer", "label", "correct_answer", "solution"),
        default="",
    )
    choices = extract_choices_from_row(row)

    parsed = {
        "question": question,
        "answer": answer,
        "choices": choices,
        "category": pick_first(row, ("category", "coarse_category")),
        "l2_category": pick_first(row, ("l2_category", "fine_category")),
        "domain": pick_first(row, ("domain",)),
        "source": pick_first(row, ("source",)),
        "metadata": extract_serializable_metadata(
            row,
            excluded_keys={
                "index",
                "id",
                "question_id",
                "qid",
                "question",
                "text",
                "prompt",
                "query",
                "instruction",
                "answer",
                "gt_answer",
                "label",
                "correct_answer",
                "solution",
                "choices",
                "options",
                "candidates",
                "image",
                "images",
                "img",
            },
        ),
    }
    parsed.update(prepare_task_media(row, task_id, "mmstar"))
    return {"task_id": task_id, "task": parsed}


class MMStarBenchmark(BaseBenchmark):
    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        self.benchmark_name = "mmstar"
        self.requires_sandbox = False
        super().__init__(agent_dir, config, requires_sandbox=self.requires_sandbox)
        self.benchmark = self._build_benchmark(self._load_dataset_rows())

    def _load_dataset_rows(self) -> List[Dict[str, Any]]:
        from datasets import load_dataset

        load_attempts = []
        for split_name in ("val", "validation", "test"):
            try:
                return list(load_dataset("Lin-Chen/MMStar", split=split_name))
            except Exception as exc:
                load_attempts.append(f"split={split_name}: {exc}")

        raise RuntimeError(
            "Failed to load Lin-Chen/MMStar from Hugging Face. "
            + " | ".join(load_attempts)
        )

    def _build_benchmark(self, rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        benchmark = {}
        for row_idx, row in enumerate(rows):
            parsed = parse_mmstar_row(dict(row), row_idx)
            benchmark[parsed["task_id"]] = parsed["task"]
        logger.info("Loaded %s MMStar tasks", len(benchmark))
        return benchmark

    def evaluate_output(
        self, agent_output: Dict[str, Any], run_id: str
    ) -> Dict[str, Any]:
        del run_id
        results = {}

        for task_id, raw_task_output in agent_output.items():
            if task_id not in self.benchmark:
                results[task_id] = {"correct": False, "error": "Task not found in dataset"}
                continue

            task = self.benchmark[task_id]
            if not isinstance(raw_task_output, dict):
                results[task_id] = {
                    "correct": False,
                    "error": "MMStar requires a dict with vision_answer, no_image_answer, and base_llm_answer",
                }
                continue

            missing_fields = [
                field_name
                for field_name in REQUIRED_OUTPUT_FIELDS
                if field_name not in raw_task_output
            ]

            if missing_fields:
                result_entry = {
                    "correct": False,
                    "vision_correct": False,
                    "no_image_correct": False,
                    "base_llm_correct": False,
                    "error": f"Missing required MMStar answer channels: {', '.join(missing_fields)}",
                }
                attach_agent_metrics(result_entry, raw_task_output)
                results[task_id] = result_entry
                continue

            vision_correct, vision_predicted = evaluate_choice_or_text_response(
                raw_task_output["vision_answer"],
                task["answer"],
                choices=task.get("choices"),
            )
            no_image_correct, no_image_predicted = evaluate_choice_or_text_response(
                raw_task_output["no_image_answer"],
                task["answer"],
                choices=task.get("choices"),
            )
            base_llm_correct, base_llm_predicted = evaluate_choice_or_text_response(
                raw_task_output["base_llm_answer"],
                task["answer"],
                choices=task.get("choices"),
            )

            result_entry = {
                "correct": vision_correct,
                "vision_correct": vision_correct,
                "no_image_correct": no_image_correct,
                "base_llm_correct": base_llm_correct,
                "expected": task["answer"],
                "vision_predicted": vision_predicted,
                "no_image_predicted": no_image_predicted,
                "base_llm_predicted": base_llm_predicted,
            }

            for field_name in BREAKDOWN_FIELDS:
                if task.get(field_name) not in (None, ""):
                    result_entry[field_name] = task[field_name]

            attach_agent_metrics(result_entry, raw_task_output)
            results[task_id] = result_entry

        return results

    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        total_count = len(eval_results)
        vision_correct = sum(
            1 for result in eval_results.values() if result.get("vision_correct", False)
        )
        no_image_correct = sum(
            1 for result in eval_results.values() if result.get("no_image_correct", False)
        )
        base_llm_correct = sum(
            1 for result in eval_results.values() if result.get("base_llm_correct", False)
        )

        accuracy = vision_correct / total_count if total_count > 0 else 0.0
        no_image_accuracy = no_image_correct / total_count if total_count > 0 else 0.0
        base_llm_accuracy = base_llm_correct / total_count if total_count > 0 else 0.0

        metrics = {
            "accuracy": accuracy,
            "no_image_accuracy": no_image_accuracy,
            "base_llm_accuracy": base_llm_accuracy,
            "MG": accuracy - no_image_accuracy,
            "ML": max(0.0, no_image_accuracy - base_llm_accuracy),
            **summarize_tool_metrics(eval_results),
            "successful_tasks": [
                task_id
                for task_id, result in eval_results.items()
                if result.get("vision_correct", False)
            ],
            "failed_tasks": [
                task_id
                for task_id, result in eval_results.items()
                if not result.get("vision_correct", False)
            ],
        }

        for field_name in BREAKDOWN_FIELDS:
            grouped = compute_group_accuracy(eval_results, field_name, success_key="vision_correct")
            if grouped:
                metrics[f"{field_name}_accuracy"] = grouped

        return metrics
