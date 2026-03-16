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
            timeout_seconds=60 * 60 * 24,
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
        install_ctx = self.sandbox.create_code_context(
            code_language=CodeLanguage.PYTHON_3_11
        )
        install_cmds = """
%pip install numpy -i https://mirrors.aliyun.com/pypi/simple/
%pip install scipy -i https://mirrors.aliyun.com/pypi/simple/
%pip install sympy -i https://mirrors.aliyun.com/pypi/simple/
"""
        self.sandbox.run_code_with_context(install_cmds, context=install_ctx)

        # Create execution context (preserves state across calls)
        self.context = self.sandbox.create_code_context(
            code_language=CodeLanguage.PYTHON_3_11
        )
        self.initialized = True

    def execute_code(self, code: str) -> str:
        """Execute Python code in the sandbox and return output.

        Args:
            code: Python code to execute.

        Returns:
            The output from code execution, or an error message if execution failed.
            Error messages start with "Error:" prefix.
        """
        if not self.initialized:
            self.initialize()

        try:
            result = self.sandbox.run_code_with_context(code=code, context=self.context)
            result_json = result.to_json()

            if result_json.get("success", False):
                output = result_json.get("outputs", "")
                output = output.strip() if output else "(No output)"
                # Truncate very long outputs
                if len(output) > 10000:
                    output = output[:10000] + "\n... [output truncated]"
                return output
            else:
                error_msg = result_json.get("error_message") or "Code execution failed"
                return f"Error: {error_msg}"
        except Exception as execution_error:
            return f"Error executing code: {str(execution_error)}"

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
