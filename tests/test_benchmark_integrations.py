import os
import sys
import tempfile
import types
import unittest
import base64
from pathlib import Path
from unittest.mock import patch


def _install_weave_stub() -> None:
    if "weave" in sys.modules:
        return

    weave_module = types.ModuleType("weave")
    weave_module.finish = lambda: None
    weave_module.op = lambda *args, **kwargs: (lambda func: func)

    trace_server_module = types.ModuleType("weave.trace_server")
    trace_interface_module = types.ModuleType(
        "weave.trace_server.trace_server_interface"
    )

    class CallsFilter:  # pragma: no cover - simple import stub
        pass

    class CallsQueryReq:  # pragma: no cover - simple import stub
        pass

    trace_interface_module.CallsFilter = CallsFilter
    trace_interface_module.CallsQueryReq = CallsQueryReq

    sys.modules["weave"] = weave_module
    sys.modules["weave.trace_server"] = trace_server_module
    sys.modules[
        "weave.trace_server.trace_server_interface"
    ] = trace_interface_module


def _install_rich_stub() -> None:
    if "rich" in sys.modules:
        return

    rich_module = types.ModuleType("rich")
    rich_console_module = types.ModuleType("rich.console")
    rich_progress_module = types.ModuleType("rich.progress")

    class Console:  # pragma: no cover - simple import stub
        def __init__(self, *args, **kwargs):
            pass

    class Progress:  # pragma: no cover - simple import stub
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def add_task(self, *args, **kwargs):
            return 0

        def update(self, *args, **kwargs):
            return None

    rich_console_module.Console = Console
    rich_progress_module.Progress = Progress
    rich_progress_module.SpinnerColumn = lambda *args, **kwargs: object()
    rich_progress_module.TextColumn = lambda *args, **kwargs: object()
    rich_progress_module.BarColumn = lambda *args, **kwargs: object()
    rich_progress_module.TaskProgressColumn = lambda *args, **kwargs: object()
    rich_progress_module.TimeRemainingColumn = lambda *args, **kwargs: object()
    rich_progress_module.TaskID = int

    sys.modules["rich"] = rich_module
    sys.modules["rich.console"] = rich_console_module
    sys.modules["rich.progress"] = rich_progress_module


_install_weave_stub()
_install_rich_stub()

from hal.benchmark_manager import BenchmarkManager
from hal.benchmarks.hrbench import HRBenchBenchmark, parse_hrbench_row
from hal.benchmarks.mathvista import MathVistaBenchmark, parse_mathvista_row
from hal.benchmarks.mmstar import MMStarBenchmark, parse_mmstar_row
from hal.benchmarks._benchmark_utils import (
    _LOCAL_ASSET_PATH_CACHE,
    extract_inline_choices_from_text,
)
from hal.benchmarks.vstar_bench import VStarBenchBenchmark, parse_vstar_row
from agents.common.vqa_runtime import run_vqa_agent


class _FakeToolFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.function = _FakeToolFunction(name, arguments)


class _FakeMessage:
    def __init__(self, content=None, tool_calls=None, reasoning_content=None):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content
        self.reasoning = None


class _FakeChoice:
    def __init__(self, finish_reason: str, message: _FakeMessage):
        self.finish_reason = finish_reason
        self.message = message


class _FakeResponse:
    def __init__(self, finish_reason: str, message: _FakeMessage):
        self.choices = [_FakeChoice(finish_reason, message)]


class BenchmarkIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.image_path = os.path.join(self.temp_dir.name, "sample.png")
        with open(self.image_path, "wb") as handle:
            handle.write(b"not-a-real-png")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    class _FakeSavableImage:
        def __init__(self, payload: bytes):
            self.payload = payload

        def save(self, path: str) -> None:
            with open(path, "wb") as handle:
                handle.write(self.payload)

    def test_vstar_row_parsing_and_metrics(self) -> None:
        _LOCAL_ASSET_PATH_CACHE.clear()
        local_cache_root = Path(self.temp_dir.name) / "hf_cache" / "datasets"
        local_cache_image = local_cache_root / "direct_attributes" / "sa_4690.jpg"
        local_cache_image.parent.mkdir(parents=True, exist_ok=True)
        local_cache_image.write_bytes(b"fake-jpg")
        with patch.dict(os.environ, {"HF_DATASETS_CACHE": str(local_cache_root)}, clear=False):
            parsed = parse_vstar_row(
                {
                    "id": "1",
                    "text": "Which option is correct?",
                    "choices": ["Alpha", "Beta", "Gamma", "Delta"],
                    "answer": "B",
                    "task": "reasoning",
                    "category": "multi_choice",
                    "source": "official",
                    "image": "direct_attributes/sa_4690.jpg",
                },
                0,
            )

        self.assertEqual(parsed["task_id"], "1")
        self.assertEqual(parsed["task"]["question"], "Which option is correct?")
        self.assertEqual(parsed["task"]["file_name"], "images/1_0.jpg")
        self.assertIn("images/1_0.jpg", parsed["task"]["files"])

        with patch.object(
            VStarBenchBenchmark,
            "_load_dataset_rows",
            return_value=[
                {
                    "id": "1",
                    "text": "Which option is correct?",
                    "choices": ["Alpha", "Beta", "Gamma", "Delta"],
                    "answer": "B",
                    "task": "reasoning",
                    "category": "multi_choice",
                    "source": "official",
                    "image": "direct_attributes/sa_4690.jpg",
                }
            ],
        ), patch.dict(os.environ, {"HF_DATASETS_CACHE": str(local_cache_root)}, clear=False):
            benchmark = VStarBenchBenchmark("agents", {})

        eval_results = benchmark.evaluate_output({"1": "ANSWER: B"}, "run")
        metrics = benchmark.get_metrics(eval_results)

        self.assertTrue(eval_results["1"]["correct"])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["category_accuracy"]["multi_choice"], 1.0)

    def test_vstar_row_uses_hf_endpoint_for_url_fallback(self) -> None:
        _LOCAL_ASSET_PATH_CACHE.clear()
        with patch.dict(os.environ, {"HF_ENDPOINT": "https://hf-mirror.com"}, clear=False):
            parsed = parse_vstar_row(
                {
                    "id": "1",
                    "text": "Which option is correct?",
                    "answer": "B",
                    "image": "direct_attributes/sa_4690.jpg",
                },
                0,
            )

        self.assertEqual(
            parsed["task"]["image_url"],
            "https://hf-mirror.com/datasets/craigwu/vstar_bench/resolve/main/direct_attributes/sa_4690.jpg",
        )

    def test_extract_inline_choices_from_question_text(self) -> None:
        choices = extract_inline_choices_from_text(
            "What is the material of the glove?\n"
            "(A) rubber\n(B) cotton\n(C) kevlar\n(D) leather\n"
            "Answer with the option's letter from the given choices directly."
        )

        self.assertEqual(
            choices,
            {
                "A": "rubber",
                "B": "cotton",
                "C": "kevlar",
                "D": "leather",
            },
        )

    def test_hrbench_row_parsing_and_metrics(self) -> None:
        parsed = parse_hrbench_row(
            {
                "id": "2",
                "question": "Pick the best answer.",
                "A": "North",
                "B": "South",
                "C": "East",
                "D": "West",
                "answer": "C",
                "category": "geography",
                "domain": "knowledge",
                "source": "hrbench",
                "image": self.image_path,
            },
            0,
            "hrbench4k",
        )

        self.assertEqual(parsed["task_id"], "2")
        self.assertEqual(parsed["task"]["choices"]["C"], "East")

        with patch.object(
            HRBenchBenchmark,
            "_load_dataset_rows",
            return_value=[
                {
                    "id": "2",
                    "question": "Pick the best answer.",
                    "A": "North",
                    "B": "South",
                    "C": "East",
                    "D": "West",
                    "answer": "C",
                    "category": "geography",
                    "domain": "knowledge",
                    "source": "hrbench",
                    "image": self.image_path,
                }
            ],
        ):
            benchmark = HRBenchBenchmark("agents", {}, "hrbench4k")

        eval_results = benchmark.evaluate_output({"2": "final answer: C"}, "run")
        metrics = benchmark.get_metrics(eval_results)

        self.assertTrue(eval_results["2"]["correct"])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["domain_accuracy"]["knowledge"], 1.0)

    def test_hrbench_base64_image_is_materialized(self) -> None:
        encoded_image = base64.b64encode(b"fake-jpeg-bytes" * 20).decode("ascii")
        parsed = parse_hrbench_row(
            {
                "id": "2b",
                "question": "Pick the best answer.",
                "A": "North",
                "B": "South",
                "answer": "B",
                "image": encoded_image,
            },
            0,
            "hrbench4k",
        )

        self.assertEqual(parsed["task"]["file_name"], "images/2b_0.png")
        self.assertIn("images/2b_0.png", parsed["task"]["files"])

    def test_hrbench_load_uses_version_split_config(self) -> None:
        fake_rows = [{"id": "2", "question": "Q", "answer": "A"}]

        datasets_module = types.ModuleType("datasets")
        mock_load_dataset = unittest.mock.Mock(return_value=fake_rows)
        datasets_module.load_dataset = mock_load_dataset

        with patch.dict(sys.modules, {"datasets": datasets_module}):
            benchmark = HRBenchBenchmark("agents", {}, "hrbench4k")

        self.assertIn("2", benchmark.benchmark)
        mock_load_dataset.assert_called_with(
            "DreamMr/HR-Bench",
            "hrbench_version_split",
            split="hrbench_4k",
        )

    def test_mathvista_parsing_evaluation_and_metrics(self) -> None:
        parsed = parse_mathvista_row(
            {
                "pid": "mv_1",
                "question": "What is 2 + 2?",
                "choices": ["1", "4", "6", "8"],
                "answer": "4",
                "question_type": "multi_choice",
                "answer_type": "integer",
                "image": self.image_path,
            },
            0,
        )
        self.assertEqual(parsed["task_id"], "mv_1")
        self.assertEqual(parsed["task"]["choices"]["B"], "4")

        rows = [
            {
                "pid": "mv_1",
                "question": "What is 2 + 2?",
                "choices": ["1", "4", "6", "8"],
                "answer": "4",
                "question_type": "multi_choice",
                "answer_type": "integer",
                "task": "arithmetic",
                "source": "mathvista",
                "image": self.image_path,
            },
            {
                "pid": "mv_2",
                "question": "Approximate pi to two decimals.",
                "answer": "3.14",
                "question_type": "free_form",
                "answer_type": "float",
                "precision": 2,
                "task": "estimation",
                "source": "mathvista",
                "image": self.image_path,
            },
            {
                "pid": "mv_3",
                "question": "Name the shape.",
                "answer": "circle",
                "question_type": "free_form",
                "answer_type": "text",
                "task": "recognition",
                "source": "mathvista",
                "image": self.image_path,
            },
        ]

        with patch.object(MathVistaBenchmark, "_load_dataset_rows", return_value=rows):
            benchmark = MathVistaBenchmark("agents", {})

        eval_results = benchmark.evaluate_output(
            {
                "mv_1": "ANSWER: B",
                "mv_2": "The answer is 3.14159",
                "mv_3": "Final answer: square",
            },
            "run",
        )
        metrics = benchmark.get_metrics(eval_results)

        self.assertTrue(eval_results["mv_1"]["correct"])
        self.assertTrue(eval_results["mv_2"]["correct"])
        self.assertFalse(eval_results["mv_3"]["correct"])
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertEqual(metrics["question_type_accuracy"]["multi_choice"], 1.0)
        self.assertEqual(metrics["question_type_accuracy"]["free_form"], 0.5)

    def test_mathvista_prefers_decoded_image_asset(self) -> None:
        parsed = parse_mathvista_row(
            {
                "pid": "mv_img",
                "question": "Read the chart.",
                "answer": "1.2",
                "answer_type": "float",
                "image": "images/1.jpg",
                "decoded_image": self._FakeSavableImage(b"fake-mathvista-image"),
            },
            0,
        )

        self.assertEqual(parsed["task"]["file_name"], "images/mv_img_0.png")
        self.assertIn("images/mv_img_0.png", parsed["task"]["files"])

    def test_mmstar_metrics_and_missing_channels(self) -> None:
        parsed = parse_mmstar_row(
            {
                "index": "mm_1",
                "question": "Which option matches the image?",
                "choices": ["Cat", "Dog", "Bird", "Fish"],
                "answer": "A",
                "category": "perception",
                "l2_category": "object",
                "image": self.image_path,
            },
            0,
        )
        self.assertEqual(parsed["task_id"], "mm_1")
        self.assertEqual(parsed["task"]["choices"]["A"], "Cat")

        rows = [
            {
                "index": "mm_1",
                "question": "Which option matches the image?",
                "choices": ["Cat", "Dog", "Bird", "Fish"],
                "answer": "A",
                "category": "perception",
                "l2_category": "object",
                "image": self.image_path,
            },
            {
                "index": "mm_2",
                "question": "Which direction is shown?",
                "choices": ["Left", "Right", "Up", "Down"],
                "answer": "B",
                "category": "reasoning",
                "l2_category": "spatial",
                "image": self.image_path,
            },
        ]

        with patch.object(MMStarBenchmark, "_load_dataset_rows", return_value=rows):
            benchmark = MMStarBenchmark("agents", {})

        eval_results = benchmark.evaluate_output(
            {
                "mm_1": {
                    "vision_answer": "ANSWER: A",
                    "no_image_answer": "ANSWER: B",
                    "base_llm_answer": "ANSWER: B",
                },
                "mm_2": {
                    "vision_answer": "ANSWER: B",
                    "no_image_answer": "ANSWER: B",
                    "base_llm_answer": "ANSWER: A",
                },
            },
            "run",
        )
        metrics = benchmark.get_metrics(eval_results)

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["no_image_accuracy"], 0.5)
        self.assertEqual(metrics["base_llm_accuracy"], 0.0)
        self.assertEqual(metrics["MG"], 0.5)
        self.assertEqual(metrics["ML"], 0.5)

        missing_channel = benchmark.evaluate_output(
            {
                "mm_1": {
                    "vision_answer": "ANSWER: A",
                    "no_image_answer": "ANSWER: B",
                }
            },
            "run",
        )
        self.assertIn("Missing required MMStar answer channels", missing_channel["mm_1"]["error"])

    def test_vqa_agent_without_tools_matches_plain_behavior(self) -> None:
        with patch(
            "agents.common.vqa_runtime._invoke_completion",
            return_value=_FakeResponse("stop", _FakeMessage(content="B")),
        ):
            result = run_vqa_agent(
                {
                    "task_1": {
                        "question": "Which option is correct?",
                        "choices": {"A": "Alpha", "B": "Beta"},
                    }
                },
                model_name="test-model",
                benchmark_name="vstar_bench",
                enable_tools=False,
                max_tokens=64,
            )

        self.assertEqual(result["task_1"], "B")

    def test_vqa_agent_with_tools_executes_python_and_preserves_context(self) -> None:
        responses = [
            _FakeResponse(
                "tool_calls",
                _FakeMessage(
                    content="",
                    tool_calls=[
                        _FakeToolCall("call_1", "execute_python", '{"code": "x = 6"}')
                    ],
                ),
            ),
            _FakeResponse(
                "tool_calls",
                _FakeMessage(
                    content="",
                    tool_calls=[
                        _FakeToolCall(
                            "call_2", "execute_python", '{"code": "print(x * 7)"}'
                        )
                    ],
                ),
            ),
            _FakeResponse("stop", _FakeMessage(content="Final answer: 42")),
        ]

        with patch(
            "agents.common.vqa_runtime._invoke_completion",
            side_effect=responses,
        ):
            result = run_vqa_agent(
                {"task_2": {"question": "Compute the value."}},
                model_name="test-model",
                benchmark_name="mathvista",
                enable_tools=True,
                code_executor="local",
                max_tokens=64,
            )

        self.assertEqual(result["task_2"]["answer"], "Final answer: 42")
        self.assertEqual(result["task_2"]["metrics"]["tool_call_count"], 2)
        tool_turns = [
            turn
            for turn in result["task_2"]["metrics"]["conversation_history"]
            if turn.get("role") == "tool"
        ]
        self.assertEqual(tool_turns[-1]["result"], "42")
        self.assertEqual(result["task_2"]["metrics"]["sandbox_error_types"], [])

    def test_vqa_agent_with_tools_records_sandbox_error_types(self) -> None:
        responses = [
            _FakeResponse(
                "tool_calls",
                _FakeMessage(
                    content="",
                    tool_calls=[
                        _FakeToolCall(
                            "call_err",
                            "execute_python",
                            '{"code": "raise ValueError(\\"boom\\")"}',
                        )
                    ],
                ),
            ),
            _FakeResponse("stop", _FakeMessage(content="Final answer: fallback")),
        ]

        with patch(
            "agents.common.vqa_runtime._invoke_completion",
            side_effect=responses,
        ):
            result = run_vqa_agent(
                {"task_3": {"question": "Use python and recover."}},
                model_name="test-model",
                benchmark_name="vstar_bench",
                enable_tools=True,
                code_executor="local",
                max_tokens=64,
            )

        self.assertEqual(
            result["task_3"]["metrics"]["sandbox_error_types"],
            ["ValueError"],
        )

    def test_mmstar_with_tools_preserves_three_channel_contract(self) -> None:
        responses = [
            _FakeResponse(
                "tool_calls",
                _FakeMessage(
                    content="",
                    tool_calls=[
                        _FakeToolCall(
                            "vision_call",
                            "execute_python",
                            '{"code": "print(1 + 1)"}',
                        )
                    ],
                ),
            ),
            _FakeResponse("stop", _FakeMessage(content="A")),
            _FakeResponse("stop", _FakeMessage(content="B")),
            _FakeResponse("stop", _FakeMessage(content="C")),
        ]

        with patch(
            "agents.common.vqa_runtime._invoke_completion",
            side_effect=responses,
        ):
            result = run_vqa_agent(
                {
                    "task_4": {
                        "question": "Which option matches?",
                        "choices": {"A": "Cat", "B": "Dog", "C": "Bird", "D": "Fish"},
                    }
                },
                model_name="test-model",
                benchmark_name="mmstar",
                enable_tools=True,
                code_executor="local",
                max_tokens=64,
            )

        self.assertEqual(result["task_4"]["vision_answer"], "A")
        self.assertEqual(result["task_4"]["no_image_answer"], "B")
        self.assertEqual(result["task_4"]["base_llm_answer"], "B")
        self.assertEqual(result["task_4"]["metrics"]["tool_call_count"], 1)

    def test_attach_agent_metrics_includes_sandbox_error_types(self) -> None:
        with patch.object(
            VStarBenchBenchmark,
            "_load_dataset_rows",
            return_value=[
                {"id": "1", "question": "Q", "answer": "A", "image": self.image_path}
            ],
        ):
            benchmark = VStarBenchBenchmark("agents", {})

        eval_results = benchmark.evaluate_output(
            {
                "1": {
                    "answer": "A",
                    "metrics": {
                        "tool_call_count": 1,
                        "conversation_history": [],
                        "sandbox_error_types": ["ValueError"],
                    },
                }
            },
            "run",
        )

        self.assertEqual(eval_results["1"]["sandbox_error_types"], ["ValueError"])

    def test_benchmark_manager_registration(self) -> None:
        manager = BenchmarkManager(agent_dir="agents", config={})
        for benchmark_name in (
            "vstar_bench",
            "hrbench4k",
            "hrbench8k",
            "mathvista",
            "mmstar",
        ):
            self.assertIn(benchmark_name, manager.list_benchmarks())

        with (
            patch.object(
                VStarBenchBenchmark,
                "_load_dataset_rows",
                return_value=[{"id": "1", "question": "Q", "answer": "A"}],
            ),
            patch.object(
                HRBenchBenchmark,
                "_load_dataset_rows",
                return_value=[{"id": "2", "question": "Q", "answer": "A"}],
            ),
            patch.object(
                MathVistaBenchmark,
                "_load_dataset_rows",
                return_value=[{"pid": "3", "question": "Q", "answer": "1"}],
            ),
            patch.object(
                MMStarBenchmark,
                "_load_dataset_rows",
                return_value=[{"index": "4", "question": "Q", "answer": "A"}],
            ),
        ):
            self.assertIsInstance(manager.get_benchmark("vstar_bench"), VStarBenchBenchmark)
            self.assertIsInstance(manager.get_benchmark("hrbench4k"), HRBenchBenchmark)
            self.assertIsInstance(manager.get_benchmark("hrbench8k"), HRBenchBenchmark)
            self.assertIsInstance(manager.get_benchmark("mathvista"), MathVistaBenchmark)
            self.assertIsInstance(manager.get_benchmark("mmstar"), MMStarBenchmark)


if __name__ == "__main__":
    unittest.main()
