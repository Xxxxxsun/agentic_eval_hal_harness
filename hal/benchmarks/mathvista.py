import json
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

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

if TYPE_CHECKING:
    from openai import OpenAI

DEFAULT_PROXY_OPENAI_BASE_URL = "https://llm-chat-api.alibaba-inc.com/openai"
DEFAULT_PROXY_TOKEN = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    ".eyJ1c2VyIjoibW9kZWxfdHJhaW5fdmxtIn0"
    ".XYFz9JOEJ-f1xXPPqs9yi5cnEG4sjrR6cZayWz_NDqM"
)
DEFAULT_QUOTA_ID = "dd95187c-29dd-464d-9b96-8f62e6ab8eb5"
DEFAULT_ACCESS_KEY = "9101ac974ab20f60f668dcf099bc6a10"
DEFAULT_USER_ID = "506759"
DEFAULT_APP = "model_train_vlm"

BREAKDOWN_FIELDS = ("question_type", "answer_type", "task", "source", "category")
DEFAULT_MATHVISTA_JUDGE_MODEL = "gpt-4o"
MATHVISTA_JUDGE_PROMPT = """You are grading a MathVista answer.

Decide whether the model response should be accepted as correct for the given question.
Judge only the final answer meaning. Ignore extra explanation unless it changes the final answer.

Return strict JSON only:
{"correct": true, "predicted_answer": "<short answer>"}
or
{"correct": false, "predicted_answer": "<short answer>"}

Question:
{question}

Choices:
{choices}

Answer type:
{answer_type}

Precision:
{precision}

Gold answer:
{gold_answer}

Model response:
{model_response}
"""


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
                "decoded_image",
                "decoded_images",
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


def _is_truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _build_mathvista_choices_text(choices: Dict[str, str]) -> str:
    if not choices:
        return "N/A"
    return "\n".join(f"{label}. {text}" for label, text in choices.items())


def _create_mathvista_judge_client() -> Any:
    from openai import OpenAI

    return OpenAI(
        base_url=os.getenv("PROXY_OPENAI_BASE_URL", DEFAULT_PROXY_OPENAI_BASE_URL),
        api_key=os.getenv(
            "PROXY_OPENAI_API_KEY",
            os.getenv("PROXY_TOKEN", DEFAULT_PROXY_TOKEN),
        ),
    )


def _build_mathvista_judge_extra_body() -> Dict[str, Any]:
    return {
        "app": DEFAULT_APP,
        "quota_id": os.getenv("QUOTA_ID", DEFAULT_QUOTA_ID),
        "user_id": os.getenv("PROXY_USER_ID", DEFAULT_USER_ID),
        "access_key": os.getenv("ACCESS_KEY", DEFAULT_ACCESS_KEY),
    }


def _parse_mathvista_judge_output(content: str) -> Optional[Tuple[bool, str]]:
    normalized = (content or "").strip()
    if not normalized:
        return None

    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError:
        match = None
        for candidate in reversed([line.strip() for line in normalized.splitlines() if line.strip()]):
            try:
                payload = json.loads(candidate)
                match = payload
                break
            except json.JSONDecodeError:
                continue
        if match is None:
            return None
        payload = match

    correct = payload.get("correct")
    if not isinstance(correct, bool):
        return None
    predicted_answer = extract_final_answer_text(payload.get("predicted_answer", ""))
    return correct, predicted_answer


def judge_mathvista_answer_with_llm(
    client: Any,
    grader_model: str,
    task: Dict[str, Any],
    response: Any,
) -> Optional[Tuple[bool, str]]:
    prompt = MATHVISTA_JUDGE_PROMPT.format(
        question=task.get("question", ""),
        choices=_build_mathvista_choices_text(task.get("choices") or {}),
        answer_type=task.get("answer_type", ""),
        precision=task.get("precision", ""),
        gold_answer=task.get("answer", ""),
        model_response=str(response),
    )

    try:
        judge_response = client.chat.completions.create(
            model=grader_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
            extra_body=_build_mathvista_judge_extra_body(),
        )
    except Exception as exc:
        logger.warning("MathVista LLM judge request failed: %s", exc)
        return None

    return _parse_mathvista_judge_output(judge_response.choices[0].message.content or "")


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

    def _should_use_llm_judge(self) -> bool:
        agent_value = (self.agent_args or {}).get("mathvista_use_llm_judge")
        if agent_value is not None:
            return _is_truthy(agent_value)

        config_value = self.config.get("mathvista_use_llm_judge") if isinstance(self.config, dict) else None
        if config_value is not None:
            return _is_truthy(config_value)

        return _is_truthy(os.getenv("MATHVISTA_USE_LLM_JUDGE", "false"))

    def _get_llm_judge_model(self) -> str:
        agent_value = (self.agent_args or {}).get("mathvista_judge_model")
        if agent_value:
            return str(agent_value)

        config_value = self.config.get("mathvista_judge_model") if isinstance(self.config, dict) else None
        if config_value:
            return str(config_value)

        return os.getenv("MATHVISTA_JUDGE_MODEL", DEFAULT_MATHVISTA_JUDGE_MODEL)

    def evaluate_output(
        self, agent_output: Dict[str, Any], run_id: str
    ) -> Dict[str, Any]:
        del run_id
        normalized_output = self._normalize_agent_output(agent_output)
        results = {}
        use_llm_judge = self._should_use_llm_judge()
        grader_model = self._get_llm_judge_model()
        judge_client = _create_mathvista_judge_client() if use_llm_judge else None

        for task_id, response in normalized_output.items():
            if task_id not in self.benchmark:
                results[task_id] = {"correct": False, "error": "Task not found in dataset"}
                continue

            task = self.benchmark[task_id]
            correct, predicted = evaluate_mathvista_answer(response, task)
            llm_judge_used = False
            if not correct and judge_client is not None:
                llm_judge_result = judge_mathvista_answer_with_llm(
                    judge_client,
                    grader_model,
                    task,
                    response,
                )
                if llm_judge_result is not None:
                    correct, judged_predicted = llm_judge_result
                    if judged_predicted:
                        predicted = judged_predicted
                    llm_judge_used = True
            result_entry = {
                "correct": correct,
                "expected": task["answer"],
                "predicted": predicted,
                "raw_response": str(response),
            }
            if llm_judge_used:
                result_entry["llm_judge_model"] = grader_model
                result_entry["llm_judge_used"] = True

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
