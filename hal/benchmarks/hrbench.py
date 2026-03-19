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

BREAKDOWN_FIELDS = ("category", "subcategory", "domain", "source", "task")


def parse_hrbench_row(
    row: Dict[str, Any], row_idx: int, benchmark_name: str
) -> Dict[str, Any]:
    task_id = str(
        pick_first(
            row,
            ("id", "question_id", "qid", "pid"),
            default=f"{benchmark_name}_{row_idx}",
        )
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
    parsed.update(prepare_task_media(row, task_id, benchmark_name))
    return {"task_id": task_id, "task": parsed}


class HRBenchBenchmark(BaseBenchmark):
    def __init__(self, agent_dir: str, config: Dict[str, Any], benchmark_name: str):
        if benchmark_name not in {"hrbench4k", "hrbench8k"}:
            raise ValueError(f"Unsupported HR-Bench variant: {benchmark_name}")

        self.benchmark_name = benchmark_name
        self.requires_sandbox = False
        super().__init__(agent_dir, config, requires_sandbox=self.requires_sandbox)
        self.benchmark = self._build_benchmark(self._load_dataset_rows())

    def _load_dataset_rows(self) -> List[Dict[str, Any]]:
        from datasets import load_dataset

        dataset_attempts = []
        config_candidates = {
            "hrbench4k": ("4k", "4K", "hrbench4k", "HRBench4K"),
            "hrbench8k": ("8k", "8K", "hrbench8k", "HRBench8K"),
        }[self.benchmark_name]

        split_candidates = ("test", "validation", "train")
        for config_name in config_candidates:
            for split_name in split_candidates:
                try:
                    return list(load_dataset("DreamMr/HR-Bench", config_name, split=split_name))
                except Exception as exc:
                    dataset_attempts.append(
                        f"config={config_name}, split={split_name}: {exc}"
                    )

        for split_name in split_candidates:
            try:
                rows = list(load_dataset("DreamMr/HR-Bench", split=split_name))
                filtered = [
                    row
                    for row in rows
                    if self._matches_benchmark_variant(dict(row))
                ]
                if filtered:
                    return filtered
            except Exception as exc:
                dataset_attempts.append(f"split={split_name}: {exc}")

        raise RuntimeError(
            f"Failed to load {self.benchmark_name} from DreamMr/HR-Bench. "
            + " | ".join(dataset_attempts)
        )

    def _matches_benchmark_variant(self, row: Dict[str, Any]) -> bool:
        marker = f"{self.benchmark_name[-2:]}".casefold()
        for key in ("split", "subset", "version", "source", "task"):
            value = pick_first(row, (key,))
            if value and marker in str(value).casefold():
                return True
        return False

    def _build_benchmark(self, rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        benchmark = {}
        for row_idx, row in enumerate(rows):
            parsed = parse_hrbench_row(dict(row), row_idx, self.benchmark_name)
            benchmark[parsed["task_id"]] = parsed["task"]
        logger.info("Loaded %s HR-Bench tasks for %s", len(benchmark), self.benchmark_name)
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
