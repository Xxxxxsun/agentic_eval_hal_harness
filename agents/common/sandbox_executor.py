"""
Sandbox executor module for running Python code in iagent sandbox.

This module provides a SandboxManager class that manages iagent sandbox instances
for code execution with context preservation across multiple calls.
"""

import os
from iagent.adk.sandbox.iagent_sandbox import IAgentSandbox, CodeLanguage, HttpConfig
from iagent.adk.sandbox.sandbox_type import SandboxSpecConfig


class SandboxManager:
    """Manages iagent sandbox for code execution with context preservation.

    Each SandboxManager instance creates its own sandbox, allowing parallel
    execution of different tasks without interference.

    The sandbox maintains context across multiple execute_code() calls,
    meaning variables and imports from previous executions are preserved.

    Usage:
        sandbox = SandboxManager()
        result1 = sandbox.execute_code("x = 10")
        result2 = sandbox.execute_code("print(x)")  # x is still available
        sandbox.destroy()  # Clean up when done
    """

    def __init__(self, base_url: str = None):
        """Initialize SandboxManager.

        Args:
            base_url: URL of the iagent sandbox service.
                      Defaults to IAGENT_SANDBOX_URL env var or
                      http://pre-iagent-sandbox-2.alibaba-inc.com
        """
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
            resource="iagent-offline",
            # resource="iagent-test",
            envs={},
            init_commands=[],
            template="docker-in-docker-swe",
            # template="iagent-sandbox-server",
            allow_domains=[],
        )
        self.sandbox = None
        self.context = None
        self.initialized = False

    def initialize(self):
        """Initialize sandbox and install dependencies.

        This method is called automatically on first execute_code() call.
        It creates the sandbox, installs required packages (numpy, scipy, sympy),
        and creates an execution context for code execution.
        """
        if self.initialized:
            return

        self.sandbox = IAgentSandbox(
            base_url=self.base_url,
            sandbox_id="",
            http_config=self.http_config,
            sandbox_spec=self.sandbox_spec,
        )

        # Install dependencies (only once per sandbox)
#         install_ctx = self.sandbox.create_code_context(
#             code_language=CodeLanguage.PYTHON_3_11
#         )
#         install_cmds = """
# %pip install numpy -i https://mirrors.aliyun.com/pypi/simple/
# %pip install scipy -i https://mirrors.aliyun.com/pypi/simple/
# %pip install sympy -i https://mirrors.aliyun.com/pypi/simple/
# """
#         self.sandbox.run_code_with_context(install_cmds, context=install_ctx)

        # Create execution context (preserves state across calls)
        self.context = self.sandbox.create_code_context(
            code_language=CodeLanguage.PYTHON_3_12
        )
        self.initialized = True

    def execute_code(self, code: str) -> dict:
        """Execute Python code in the sandbox and return structured output.

        Args:
            code: Python code to execute.

        Returns:
            A dict with:
              - output (str): The text output or error message.
              - error_type (str or None): The Python exception class name
                (e.g. "IndexError") extracted from the sandbox ``ename`` field
                when execution fails, or None on success.
        """
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

            # Extract error type from sandbox outputs
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
        """Extract readable text from sandbox outputs.

        The sandbox returns ``outputs`` as a list of dicts, each with fields
        like ``outputType``, ``text``, ``data``, etc.  For successful runs the
        ``text`` field typically holds the printed output.  This method
        concatenates all available text fragments into a single string.

        If ``raw_outputs`` is already a plain string (legacy format), it is
        returned as-is after stripping whitespace.
        """
        if isinstance(raw_outputs, str):
            return raw_outputs.strip() if raw_outputs else "(No output)"

        if not isinstance(raw_outputs, list) or not raw_outputs:
            return "(No output)"

        text_parts = []
        for entry in raw_outputs:
            if not isinstance(entry, dict):
                text_parts.append(str(entry))
                continue
            # Prefer 'text' field (printed output)
            text_value = entry.get("text")
            if text_value:
                if isinstance(text_value, list):
                    text_parts.append("".join(text_value))
                else:
                    text_parts.append(str(text_value))
                continue
            # Fall back to 'data' field
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
        """Extract the error type name from sandbox result JSON.

        Looks for the ``ename`` field in the first error output entry.
        Returns None if no error type can be determined.
        """
        outputs = result_json.get("outputs", [])
        if not isinstance(outputs, list):
            return None
        for entry in outputs:
            if isinstance(entry, dict) and entry.get("outputType") == "error":
                ename = entry.get("ename")
                if ename:
                    return str(ename)
        return None

    def destroy(self):
        """Destroy the sandbox to release resources.

        Should be called when the task is complete to clean up the sandbox.
        """
        if self.sandbox is not None:
            try:
                self.sandbox.destroy()
            except Exception:
                pass
            self.sandbox = None
            self.context = None
            self.initialized = False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures sandbox is destroyed."""
        self.destroy()
        return False
