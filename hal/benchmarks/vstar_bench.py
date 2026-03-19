import logging
from typing import Any, Dict, Iterable, List

from ._benchmark_utils import (
    attach_agent_metrics,
    build_boolean_metrics,
    evaluate_choice_or_text_response,
    extract_choices_from_row,
    extract_serializable_metadata,
    pick_first,
    prepare_task_media,
)
from .base_benchmark import BaseBenchmark

logger = logging.getLogger(__name__)

BREAKDOWN_FIELDS = ("task", "category", "source", "split", "domain")


def parse_vstar_row(row: Dict[str, Any], row_idx: int) -> Dict[str, Any]:
    task_id = str(
        pick_first(row, ("id", "question_id", "qid", "pid"), default=f"vstar_bench_{row_idx}")
    )
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
        "metadata": extract_serializable_metadata(
            row,
            excluded_keys={
                "id",
                "question_id",
                "qid",
                "pid",
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
                "image",
                "images",
                "img",
                "choices",
                "options",
                "candidates",
            },
        ),
    }
    parsed.update(prepare_task_media(row, task_id, "vstar_bench"))
    return {"task_id": task_id, "task": parsed}


class VStarBenchBenchmark(BaseBenchmark):
    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        self.benchmark_name = "vstar_bench"
        self.requires_sandbox = False
        super().__init__(agent_dir, config, requires_sandbox=self.requires_sandbox)
        self.benchmark = self._build_benchmark(self._load_dataset_rows())

    def _load_dataset_rows(self) -> List[Dict[str, Any]]:
        from datasets import load_dataset

        load_attempts: List[str] = []
        for split_name in ("test", "validation", "train"):
            try:
                return list(load_dataset("craigwu/vstar_bench", split=split_name))
            except Exception as exc:
                load_attempts.append(f"split={split_name}: {exc}")

        raise RuntimeError(
            "Failed to load craigwu/vstar_bench from Hugging Face. "
            + " | ".join(load_attempts)
        )

    def _build_benchmark(self, rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        benchmark = {}
        for row_idx, row in enumerate(rows):
            parsed = parse_vstar_row(dict(row), row_idx)
            benchmark[parsed["task_id"]] = parsed["task"]
        logger.info("Loaded %s V* benchmark tasks", len(benchmark))
        return benchmark

    def evaluate_output(
        self, agent_output: Dict[str, Any], run_id: str
    ) -> Dict[str, Any]:
        del run_id
        normalized_output = self._normalize_agent_output(agent_output)
        results = {}

        for task_id, response in normalized_output.items():
            if task_id not in self.benchmark:
                results[task_id] = {"correct": False, "error": "Task not found in dataset"}
                continue

            task = self.benchmark[task_id]
            correct, predicted = evaluate_choice_or_text_response(
                response,
                task["answer"],
                choices=task.get("choices"),
            )
            result_entry = {
                "correct": correct,
                "expected": task["answer"],
                "predicted": predicted,
                "raw_response": str(response),
            }

            for field_name in BREAKDOWN_FIELDS:
                if field_name in task.get("metadata", {}):
                    result_entry[field_name] = task["metadata"][field_name]

            attach_agent_metrics(result_entry, agent_output.get(task_id))
            results[task_id] = result_entry

        return results

    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        return build_boolean_metrics(
            eval_results,
            success_key="correct",
            breakdown_fields=BREAKDOWN_FIELDS,
        )
