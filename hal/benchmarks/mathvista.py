import logging
from typing import Any, Dict, Iterable, List, Optional

from ._benchmark_utils import (
    attach_agent_metrics,
    build_boolean_metrics,
    canonical_text,
    extract_choices_from_row,
    extract_final_answer_text,
    extract_numeric_value,
    extract_predicted_choice,
    extract_serializable_metadata,
    format_numeric_value,
    normalize_choice_label,
    pick_first,
    prepare_task_media,
)
from .base_benchmark import BaseBenchmark

logger = logging.getLogger(__name__)

BREAKDOWN_FIELDS = ("question_type", "answer_type", "task", "source", "category")


def parse_mathvista_row(row: Dict[str, Any], row_idx: int) -> Dict[str, Any]:
    task_id = str(pick_first(row, ("pid", "id", "question_id"), default=f"mathvista_{row_idx}"))
    question = pick_first(
        row,
        ("question", "text", "prompt", "query", "instruction"),
        default="",
    )
    answer = pick_first(
        row,
        ("answer", "decoded_answer", "gt_answer", "label", "correct_answer"),
        default="",
    )
    choices = extract_choices_from_row(row)

    parsed = {
        "question": question,
        "answer": answer,
        "choices": choices,
        "question_type": pick_first(row, ("question_type", "problem_type")),
        "answer_type": pick_first(row, ("answer_type",)),
        "precision": pick_first(row, ("precision", "answer_precision")),
        "task": pick_first(row, ("task", "task_type")),
        "source": pick_first(row, ("source",)),
        "category": pick_first(row, ("category", "subfield")),
        "metadata": extract_serializable_metadata(
            row,
            excluded_keys={
                "pid",
                "id",
                "question_id",
                "question",
                "text",
                "prompt",
                "query",
                "instruction",
                "answer",
                "decoded_answer",
                "gt_answer",
                "label",
                "correct_answer",
                "choices",
                "options",
                "candidates",
                "image",
                "images",
                "img",
            },
        ),
    }
    parsed.update(prepare_task_media(row, task_id, "mathvista"))
    return {"task_id": task_id, "task": parsed}


def _normalize_mathvista_numeric(value: Any, precision: Optional[int]) -> Optional[str]:
    numeric_value = extract_numeric_value(value)
    if numeric_value is None:
        return None

    try:
        precision_value = int(precision) if precision is not None else None
    except (TypeError, ValueError):
        precision_value = None
    return format_numeric_value(numeric_value, precision=precision_value)


def evaluate_mathvista_answer(response: Any, task: Dict[str, Any]) -> tuple[bool, str]:
    choices = task.get("choices") or {}
    if choices:
        gold_label = normalize_choice_label(task["answer"], choices)
        predicted_label = extract_predicted_choice(response, choices)
        if gold_label and predicted_label:
            return predicted_label == gold_label, predicted_label

    answer_type = canonical_text(task.get("answer_type", ""))
    if any(token in answer_type for token in ("integer", "float", "number", "numeric", "decimal")):
        predicted_numeric = _normalize_mathvista_numeric(
            extract_final_answer_text(response),
            task.get("precision"),
        )
        gold_numeric = _normalize_mathvista_numeric(task["answer"], task.get("precision"))
        return predicted_numeric == gold_numeric and predicted_numeric is not None, predicted_numeric or ""

    predicted_text = extract_final_answer_text(response)
    return canonical_text(predicted_text) == canonical_text(task["answer"]), predicted_text


class MathVistaBenchmark(BaseBenchmark):
    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        self.benchmark_name = "mathvista"
        self.requires_sandbox = False
        super().__init__(agent_dir, config, requires_sandbox=self.requires_sandbox)
        self.benchmark = self._build_benchmark(self._load_dataset_rows())

    def _load_dataset_rows(self) -> List[Dict[str, Any]]:
        from datasets import load_dataset

        load_attempts = []
        for dataset_name in ("AI4Math/MathVista", "TIGER-Lab/MathVista"):
            try:
                return list(load_dataset(dataset_name, split="testmini"))
            except Exception as exc:
                load_attempts.append(f"{dataset_name}: {exc}")

        raise RuntimeError(
            "Failed to load MathVista testmini from Hugging Face. "
            + " | ".join(load_attempts)
        )

    def _build_benchmark(self, rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        benchmark = {}
        for row_idx, row in enumerate(rows):
            parsed = parse_mathvista_row(dict(row), row_idx)
            benchmark[parsed["task_id"]] = parsed["task"]
        logger.info("Loaded %s MathVista testmini tasks", len(benchmark))
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
            correct, predicted = evaluate_mathvista_answer(response, task)
            result_entry = {
                "correct": correct,
                "expected": task["answer"],
                "predicted": predicted,
                "raw_response": str(response),
            }

            for field_name in BREAKDOWN_FIELDS:
                if task.get(field_name) not in (None, ""):
                    result_entry[field_name] = task[field_name]

            attach_agent_metrics(result_entry, agent_output.get(task_id))
            results[task_id] = result_entry

        return results

    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        return build_boolean_metrics(
            eval_results,
            success_key="correct",
            breakdown_fields=BREAKDOWN_FIELDS,
        )
