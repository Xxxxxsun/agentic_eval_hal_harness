"""
Code executor module for running Python code.

This module provides two executor classes with the same interface:
  - SandboxManager: Executes code in a remote iagent sandbox (isolated, resource-controlled).
  - LocalPythonExecutor: Executes code locally via exec() (no network dependency, faster).

Both preserve context across multiple execute_code() calls.
"""

import io
import os
import sys
import traceback

try:
    from iagent.adk.sandbox.iagent_sandbox import IAgentSandbox, CodeLanguage, HttpConfig
    from iagent.adk.sandbox.sandbox_type import SandboxSpecConfig
except ImportError:  # pragma: no cover - only exercised when sandbox deps are absent
    IAgentSandbox = None
    CodeLanguage = None
    HttpConfig = None
    SandboxSpecConfig = None


class LocalPythonExecutor:
    """Executes Python code locally with context preservation across calls."""

    def __init__(self, **kwargs):
        del kwargs
        self._globals = {"__builtins__": __builtins__}

    def execute_code(self, code: str) -> dict:
        captured_output = io.StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = captured_output
            exec(code, self._globals)
            sys.stdout = old_stdout

            output = captured_output.getvalue().strip() or "(No output)"
            if len(output) > 10000:
                output = output[:10000] + "\n... [output truncated]"
            return {"output": output, "error_type": None}
        except Exception as execution_error:
            sys.stdout = old_stdout
            error_type = type(execution_error).__name__
            return {
                "output": f"Error: {traceback.format_exc()}",
                "error_type": error_type,
            }

    def destroy(self):
        self._globals = {"__builtins__": __builtins__}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()
        return False


class SandboxManager:
    """Manages iagent sandbox for code execution with context preservation."""

    def __init__(self, base_url: str = None):
        if IAgentSandbox is None:
            raise ImportError(
                "SandboxManager requires iagent sandbox dependencies. "
                "Use code_executor=local or install the iagent package."
            )

        self.base_url = base_url or os.getenv(
            "IAGENT_SANDBOX_URL", "http://pre-iagent-sandbox-2.alibaba-inc.com"
        )
        self.http_config = HttpConfig(
            timeout=300,
            max_retries=3,
            retry_delay=1,
            retry_on_status=(500, 502, 503, 504),
            verify_ssl=True,
            proxies=None,
            default_headers={"User-Agent": "PythonHttpClient/1.0"},
        )
        self.sandbox_spec = SandboxSpecConfig(
            timeout_seconds=60 * 60,
            cpu=4,
            memory_gb=8,
            resource="iagent-test",
            envs={},
            init_commands=[],
            template="iagent-sandbox-server",
            allow_domains=[],
        )
        self.sandbox = None
        self.context = None
        self.initialized = False

    def initialize(self):
        if self.initialized:
            return

        self.sandbox = IAgentSandbox(
            base_url=self.base_url,
            sandbox_id="",
            http_config=self.http_config,
            sandbox_spec=self.sandbox_spec,
        )
        self.context = self.sandbox.create_code_context(
            code_language=CodeLanguage.PYTHON_3_12
        )
        self.initialized = True

    def execute_code(self, code: str) -> dict:
        if not self.initialized:
            self.initialize()

        try:
            result = self.sandbox.run_code_with_context(code=code, context=self.context)
            result_json = result.to_json()
            if result_json.get("success", False):
                raw_outputs = result_json.get("outputs", "")
                output = self._extract_text_output(raw_outputs)
                if len(output) > 10000:
                    output = output[:10000] + "\n... [output truncated]"
                return {"output": output, "error_type": None}

            error_type = self._extract_error_type(result_json)
            error_msg = result_json.get("error_message") or "Code execution failed"
            return {"output": f"Error: {error_msg}", "error_type": error_type}
        except Exception as execution_error:
            return {
                "output": f"Error executing code: {str(execution_error)}",
                "error_type": "ExecutionException",
            }

    @staticmethod
    def _extract_text_output(raw_outputs) -> str:
        if isinstance(raw_outputs, str):
            return raw_outputs.strip() if raw_outputs else "(No output)"

        if not isinstance(raw_outputs, list) or not raw_outputs:
            return "(No output)"

        text_parts = []
        for entry in raw_outputs:
            if not isinstance(entry, dict):
                text_parts.append(str(entry))
                continue
            text_value = entry.get("text")
            if text_value:
                if isinstance(text_value, list):
                    text_parts.append("".join(text_value))
                else:
                    text_parts.append(str(text_value))
                continue
            data_value = entry.get("data")
            if data_value:
                if isinstance(data_value, dict):
                    plain = data_value.get("text/plain", "")
                    if plain:
                        text_parts.append(str(plain))
                else:
                    text_parts.append(str(data_value))

        combined = "\n".join(text_parts).strip()
        return combined if combined else "(No output)"

    @staticmethod
    def _extract_error_type(result_json: dict) -> str | None:
        outputs = result_json.get("outputs", [])
        if not isinstance(outputs, list):
            return None

        for entry in outputs:
            if not isinstance(entry, dict):
                continue
            error_info = entry.get("error")
            if isinstance(error_info, dict) and error_info.get("ename"):
                return str(error_info["ename"])
        return None

    def destroy(self):
        if self.sandbox is not None:
            try:
                self.sandbox.destroy()
            except Exception:
                pass
            self.sandbox = None
            self.context = None
            self.initialized = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()
        return False
