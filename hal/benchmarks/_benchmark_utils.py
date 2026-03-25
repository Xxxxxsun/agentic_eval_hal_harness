import base64
import binascii
from collections import Counter
import json
import os
import re
import string
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib.parse import quote


CHOICE_LABELS = tuple(string.ascii_uppercase)
ASSET_CACHE_DIR = os.path.join(tempfile.gettempdir(), "hal_benchmark_assets")
BENCHMARK_DATASET_REPOS = {
    "vstar_bench": "craigwu/vstar_bench",
    "hrbench4k": "DreamMr/HR-Bench",
    "hrbench8k": "DreamMr/HR-Bench",
    "mathvista": "AI4Math/MathVista",
    "mmstar": "Lin-Chen/MMStar",
}
_LOCAL_ASSET_PATH_CACHE: Dict[str, Optional[str]] = {}


def _dataset_asset_base_url() -> str:
    return os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def pick_first(mapping: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return default


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def canonical_text(value: Any) -> str:
    text = normalize_text(value).casefold()
    text = text.strip(" \t\r\n`'\".,;:!?()[]{}")
    return text


def extract_final_answer_text(response: Any) -> str:
    if not isinstance(response, str):
        return normalize_text(response)

    text = response.strip()
    if not text:
        return ""

    boxed_matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed_matches:
        return normalize_text(boxed_matches[-1])

    patterns = [
        r"ANSWER\s*:\s*(.+)",
        r"final\s+answer\s*[:\-]?\s*(.+)",
        r"(?:the\s+)?answer\s+is\s+(.+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return normalize_text(matches[-1])

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return normalize_text(lines[-1])
    return normalize_text(text)


def extract_choice_answer(
    response: Any, valid_labels: Optional[Iterable[str]] = None
) -> str:
    if not isinstance(response, str):
        return ""

    label_set = {
        str(label).strip().upper()
        for label in (valid_labels or CHOICE_LABELS)
        if str(label).strip()
    }
    if not label_set:
        return ""

    patterns = [
        r"ANSWER\s*:\s*\(?([A-Z])\)?\b",
        r"final\s+answer\s*[:\-]?\s*\(?([A-Z])\)?\b",
        r"(?:the\s+)?answer\s+is\s+\(?([A-Z])\)?\b",
        r"\\boxed\{([A-Z])\}",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, response.upper(), flags=re.IGNORECASE)
        for match in reversed(matches):
            if match in label_set:
                return match

    for match in reversed(re.findall(r"\b([A-Z])\b", response.upper())):
        if match in label_set:
            return match
    return ""


def build_choice_map(raw_choices: Any) -> Dict[str, str]:
    if raw_choices is None:
        return {}

    if isinstance(raw_choices, dict):
        if all(canonical_text(key) in {canonical_text(label) for label in CHOICE_LABELS} for key in raw_choices):
            return {
                str(key).strip().upper(): normalize_text(value)
                for key, value in raw_choices.items()
                if normalize_text(value)
            }
        return {
            CHOICE_LABELS[idx]: normalize_text(value)
            for idx, value in enumerate(raw_choices.values())
            if idx < len(CHOICE_LABELS) and normalize_text(value)
        }

    if isinstance(raw_choices, (list, tuple)):
        return {
            CHOICE_LABELS[idx]: normalize_text(value)
            for idx, value in enumerate(raw_choices)
            if idx < len(CHOICE_LABELS) and normalize_text(value)
        }

    return {}


def extract_inline_choices_from_text(text: Any) -> Dict[str, str]:
    normalized_text = str(text or "")
    if not normalized_text:
        return {}

    patterns = [
        re.compile(r"\(([A-Z])\)\s*(.*?)(?=\s*\([A-Z]\)\s*|\Z)", flags=re.DOTALL),
        re.compile(
            r"(?:^|\n)\s*([A-Z])[\.\):]\s*(.*?)(?=(?:\n\s*[A-Z][\.\):]\s)|\Z)",
            flags=re.DOTALL,
        ),
    ]
    trailing_instruction = re.compile(
        r"\s+(?:answer|respond|choose|select)\b.*$",
        flags=re.IGNORECASE | re.DOTALL,
    )

    for pattern in patterns:
        matches = pattern.findall(normalized_text)
        if len(matches) < 2:
            continue

        extracted: Dict[str, str] = {}
        for label, option_text in matches:
            normalized_option = normalize_text(option_text)
            normalized_option = trailing_instruction.sub("", normalized_option).strip()
            if label in CHOICE_LABELS and normalized_option:
                extracted[label] = normalized_option

        if len(extracted) >= 2:
            return extracted

    return {}


def extract_choices_from_row(row: Dict[str, Any]) -> Dict[str, str]:
    for key in ("choices", "options", "candidates"):
        choices = build_choice_map(row.get(key))
        if choices:
            return choices

    direct_letter_choices = {
        key.upper(): normalize_text(value)
        for key, value in row.items()
        if isinstance(key, str)
        and len(key) == 1
        and key.upper() in CHOICE_LABELS
        and normalize_text(value)
    }
    if direct_letter_choices:
        return dict(sorted(direct_letter_choices.items()))

    prefix_choices = {}
    for label in CHOICE_LABELS:
        for key in (f"option_{label.lower()}", f"choice_{label.lower()}"):
            if normalize_text(row.get(key)):
                prefix_choices[label] = normalize_text(row[key])
                break
    if prefix_choices:
        return prefix_choices

    for text_key in ("question", "text", "prompt", "query", "instruction"):
        inline_choices = extract_inline_choices_from_text(row.get(text_key))
        if inline_choices:
            return inline_choices

    return {}


def normalize_choice_label(answer: Any, choices: Dict[str, str]) -> str:
    if answer is None:
        return ""

    labels = list(choices.keys())
    text = normalize_text(answer)
    upper = text.upper().strip("()[]{}")
    if upper in choices:
        return upper

    canonical_answer = canonical_text(text)
    for label, choice_text in choices.items():
        if canonical_answer == canonical_text(choice_text):
            return label

    if text.isdigit():
        index = int(text)
        if 0 <= index < len(labels):
            return labels[index]
        if 1 <= index <= len(labels):
            return labels[index - 1]
    return ""


def extract_predicted_choice(response: Any, choices: Dict[str, str]) -> str:
    predicted = extract_choice_answer(response, valid_labels=choices.keys())
    if predicted:
        return predicted
    return normalize_choice_label(extract_final_answer_text(response), choices)


def extract_numeric_value(text: Any) -> Optional[float]:
    candidate = normalize_text(text)
    if not candidate:
        return None

    candidate = candidate.strip("`'\"")
    if candidate.endswith("%"):
        try:
            return float(candidate[:-1].replace(",", "")) / 100.0
        except ValueError:
            return None

    try:
        return float(Fraction(candidate.replace(",", "")))
    except (ValueError, ZeroDivisionError):
        pass

    matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?", candidate)
    if not matches:
        return None

    raw = matches[-1].replace(",", "")
    try:
        return float(Fraction(raw))
    except (ValueError, ZeroDivisionError):
        try:
            return float(raw)
        except ValueError:
            return None


def format_numeric_value(value: float, precision: Optional[int] = None) -> str:
    if precision is not None:
        value = round(value, precision)
        return f"{value:.{precision}f}".rstrip("0").rstrip(".") or "0"

    if float(value).is_integer():
        return str(int(value))
    return f"{value:.15g}"


def evaluate_choice_or_text_response(
    response: Any, gold_answer: Any, choices: Optional[Dict[str, str]] = None
) -> tuple[bool, str]:
    choice_map = choices or {}
    if choice_map:
        gold_label = normalize_choice_label(gold_answer, choice_map)
        predicted_label = extract_predicted_choice(response, choice_map)
        if gold_label and predicted_label:
            return predicted_label == gold_label, predicted_label

    predicted_text = extract_final_answer_text(response)
    return canonical_text(predicted_text) == canonical_text(gold_answer), predicted_text


def extract_serializable_metadata(
    row: Dict[str, Any], excluded_keys: Iterable[str]
) -> Dict[str, Any]:
    metadata = {}
    excluded = set(excluded_keys)
    for key, value in row.items():
        if key in excluded:
            continue
        try:
            json.dumps(value)
        except TypeError:
            continue
        metadata[key] = value
    return metadata


def _candidate_local_asset_roots() -> Iterable[Path]:
    env_roots = [
        os.getenv("HF_DATASETS_CACHE"),
        os.getenv("HUGGINGFACE_HUB_CACHE"),
        os.getenv("HF_HOME"),
        os.getenv("HAL_BENCHMARK_ASSET_CACHE"),
    ]

    roots = []
    for root in env_roots:
        if not root:
            continue
        root_path = Path(root).expanduser()
        roots.append(root_path)
        if root_path.name != "datasets":
            roots.append(root_path / "datasets")
            roots.append(root_path / "hub")

    default_hf_root = Path.home() / ".cache" / "huggingface"
    roots.extend([default_hf_root, default_hf_root / "datasets", default_hf_root / "hub"])
    roots.append(Path(ASSET_CACHE_DIR))
    roots.append(Path.cwd())

    seen = set()
    for root in roots:
        resolved = str(root)
        if resolved in seen or not root.exists():
            continue
        seen.add(resolved)
        yield root


def _resolve_local_asset_path(asset_path: str) -> Optional[str]:
    normalized = asset_path.lstrip("/")
    if not normalized:
        return None

    cache_key = normalized
    if cache_key in _LOCAL_ASSET_PATH_CACHE:
        return _LOCAL_ASSET_PATH_CACHE[cache_key]

    target = Path(normalized)
    basename = target.name
    suffix = str(target).replace("\\", "/")

    for root in _candidate_local_asset_roots():
        direct_match = root / target
        try:
            if direct_match.exists():
                resolved = str(direct_match.resolve())
                _LOCAL_ASSET_PATH_CACHE[cache_key] = resolved
                return resolved
        except OSError:
            _LOCAL_ASSET_PATH_CACHE[cache_key] = None
            return None

        try:
            for candidate in root.rglob(basename):
                candidate_suffix = str(candidate).replace("\\", "/")
                if candidate_suffix.endswith(suffix):
                    resolved = str(candidate.resolve())
                    _LOCAL_ASSET_PATH_CACHE[cache_key] = resolved
                    return resolved
        except OSError:
            continue

    _LOCAL_ASSET_PATH_CACHE[cache_key] = None
    return None


def _decode_inline_image_string(asset: str) -> Optional[bytes]:
    candidate = asset.strip()
    if not candidate:
        return None

    if candidate.startswith("data:image/") and "," in candidate:
        candidate = candidate.split(",", 1)[1]

    compact = "".join(candidate.split())
    if len(compact) < 128:
        return None
    if not re.fullmatch(r"[A-Za-z0-9+/=]+", compact):
        return None

    try:
        return base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError):
        return None


def _materialize_asset(asset: Any, benchmark_name: str, file_stub: str) -> Optional[str]:
    asset_dir = os.path.join(ASSET_CACHE_DIR, benchmark_name)
    os.makedirs(asset_dir, exist_ok=True)

    if isinstance(asset, str):
        if asset.startswith(("http://", "https://")):
            return asset
        inline_image_bytes = _decode_inline_image_string(asset)
        if inline_image_bytes is not None:
            output_path = os.path.join(asset_dir, file_stub)
            with open(output_path, "wb") as handle:
                handle.write(inline_image_bytes)
            return output_path
        if os.path.exists(asset):
            return os.path.abspath(asset)
        resolved_local_path = _resolve_local_asset_path(asset)
        if resolved_local_path:
            return resolved_local_path
        dataset_repo = BENCHMARK_DATASET_REPOS.get(benchmark_name)
        asset_extension = os.path.splitext(asset)[1].lower()
        if dataset_repo and asset_extension in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}:
            return (
                f"{_dataset_asset_base_url()}/datasets/{dataset_repo}/resolve/main/"
                f"{quote(asset.lstrip('/'))}"
            )
        return None

    if isinstance(asset, dict):
        path = asset.get("path")
        if path and os.path.exists(path):
            return os.path.abspath(path)
        asset = asset.get("bytes")

    if isinstance(asset, (bytes, bytearray)):
        output_path = os.path.join(asset_dir, file_stub)
        with open(output_path, "wb") as handle:
            handle.write(asset)
        return output_path

    if hasattr(asset, "save"):
        output_path = os.path.join(asset_dir, file_stub)
        asset.save(output_path)
        return output_path

    return None


def prepare_task_media(
    row: Dict[str, Any], task_id: str, benchmark_name: str
) -> Dict[str, Any]:
    task_updates: Dict[str, Any] = {}
    raw_assets = pick_first(
        row,
        ("decoded_images", "decoded_image", "images", "image", "img"),
    )
    if raw_assets in (None, "", []):
        return task_updates

    if not isinstance(raw_assets, (list, tuple)):
        raw_assets = [raw_assets]

    file_mappings: Dict[str, str] = {}
    image_paths = []
    image_urls = []

    for idx, asset in enumerate(raw_assets):
        abs_path = _materialize_asset(asset, benchmark_name, f"{task_id}_{idx}.png")
        if not abs_path:
            continue

        if abs_path.startswith(("http://", "https://")):
            image_urls.append(abs_path)
            continue

        extension = os.path.splitext(abs_path)[1] or ".png"
        relative_path = os.path.join("images", f"{task_id}_{idx}{extension}")
        file_mappings[relative_path] = abs_path
        image_paths.append(relative_path)

    if file_mappings:
        task_updates["files"] = file_mappings
        task_updates["image_paths"] = image_paths
        task_updates["image_path"] = image_paths[0]
        task_updates["file_name"] = image_paths[0]

    if image_urls:
        task_updates["image_urls"] = image_urls
        task_updates["image_url"] = image_urls[0]

    return task_updates


def attach_agent_metrics(result_entry: Dict[str, Any], raw_task_data: Any) -> None:
    if not isinstance(raw_task_data, dict) or "metrics" not in raw_task_data:
        return

    metrics = raw_task_data["metrics"]
    result_entry["tool_call_count"] = metrics.get("tool_call_count", 0)
    conversation_history = metrics.get("conversation_history", [])
    result_entry["conversation_history"] = conversation_history
    result_entry["sandbox_error_types"] = metrics.get("sandbox_error_types", [])

    successful_tool_calls = 0
    failed_tool_calls = 0
    tool_usage_by_name: Counter[str] = Counter()
    for turn in conversation_history:
        if turn.get("role") == "tool" and "result" in turn:
            tool_name = turn.get("tool_name")
            if tool_name:
                tool_usage_by_name[str(tool_name)] += 1
            result_text = str(turn.get("result", ""))
            if "error" in result_text.lower():
                failed_tool_calls += 1
            else:
                successful_tool_calls += 1

    result_entry["successful_tool_calls"] = successful_tool_calls
    result_entry["failed_tool_calls"] = failed_tool_calls
    result_entry["tool_usage_by_name"] = dict(tool_usage_by_name)


def summarize_tool_metrics(eval_results: Dict[str, Any]) -> Dict[str, Any]:
    tool_usage_by_name: Counter[str] = Counter()
    for result in eval_results.values():
        if not isinstance(result, dict):
            continue
        for tool_name, count in (result.get("tool_usage_by_name") or {}).items():
            try:
                tool_usage_by_name[str(tool_name)] += int(count)
            except (TypeError, ValueError):
                continue

    return {
        "tasks_with_tool_calls": sum(
            1 for result in eval_results.values() if result.get("tool_call_count", 0) > 0
        ),
        "total_tool_calls": sum(
            result.get("tool_call_count", 0) for result in eval_results.values()
        ),
        "successful_tool_calls": sum(
            result.get("successful_tool_calls", 0) for result in eval_results.values()
        ),
        "failed_tool_calls": sum(
            result.get("failed_tool_calls", 0) for result in eval_results.values()
        ),
        "tool_usage_by_name": dict(tool_usage_by_name),
    }


def compute_group_accuracy(
    eval_results: Dict[str, Any], field_name: str, success_key: str = "correct"
) -> Dict[str, float]:
    grouped: Dict[str, Dict[str, int]] = {}
    for result in eval_results.values():
        field_value = result.get(field_name)
        if field_value in (None, "", "Unknown"):
            continue

        key = str(field_value)
        grouped.setdefault(key, {"correct": 0, "total": 0})
        grouped[key]["total"] += 1
        if result.get(success_key, False):
            grouped[key]["correct"] += 1

    return {
        key: values["correct"] / values["total"]
        for key, values in grouped.items()
        if values["total"] > 0
    }


def build_boolean_metrics(
    eval_results: Dict[str, Any],
    success_key: str = "correct",
    breakdown_fields: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    total_count = len(eval_results)
    correct_count = sum(1 for result in eval_results.values() if result.get(success_key, False))

    metrics = {
        "accuracy": correct_count / total_count if total_count > 0 else 0.0,
        **summarize_tool_metrics(eval_results),
        "successful_tasks": [
            task_id for task_id, result in eval_results.items() if result.get(success_key, False)
        ],
        "failed_tasks": [
            task_id for task_id, result in eval_results.items() if not result.get(success_key, False)
        ],
    }

    for field_name in breakdown_fields or ():
        group_accuracy = compute_group_accuracy(eval_results, field_name, success_key=success_key)
        if group_accuracy:
            metrics[f"{field_name}_accuracy"] = group_accuracy

    return metrics
