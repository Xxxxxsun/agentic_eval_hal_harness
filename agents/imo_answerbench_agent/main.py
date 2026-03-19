import json
import sys
import os
from types import SimpleNamespace

# Add agents directory to path for common module import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_client import chat_completion_with_tools, _get_model_mode
from common.sandbox_executor import SandboxManager, LocalPythonExecutor

SYSTEM_PROMPT = """\
You are an expert mathematician solving challenging Olympiad-level problems \
from the International Mathematical Olympiad (IMO) and similar competitions.

You have access to a Python code interpreter tool that you can use to:
- Perform complex calculations and symbolic manipulations
- Verify your reasoning with brute-force enumeration
- Test conjectures numerically
- Solve equations symbolically (using sympy)
- Compute geometric quantities, check inequalities, etc.

IMPORTANT RULES:
1. Think step by step and show your reasoning clearly.
2. Use the Python code interpreter liberally to verify calculations and explore.
3. The answer may be a number, a mathematical expression, a set of values, \
a polynomial, an interval, a fraction, or any other mathematical object.
4. When you have determined the final answer, output it in EXACTLY this format:
   ANSWER: <your answer>
   For example:
     ANSWER: 42
     ANSWER: $\\frac{1}{2}$
     ANSWER: $P(x) = 2x^2 + c$
     ANSWER: $(-\\infty, 0) \\cup \\{\\frac{1}{2}\\}$
     ANSWER: 2026, 2030
5. Use LaTeX notation for non-trivial mathematical expressions.
6. Make sure your ANSWER line is the LAST thing you output, after all reasoning \
and verification.
7. If the problem asks to "find all" values/functions, list ALL of them in your answer.
8. Simplify your answer as much as possible.
9. IMPORTANT: The code interpreter maintains context across multiple calls. Variables \
and imports from previous code executions are preserved and can be reused in subsequent calls."""

SYSTEM_PROMPT_NO_TOOLS = """\
You are an expert mathematician solving challenging Olympiad-level problems \
from the International Mathematical Olympiad (IMO) and similar competitions.

IMPORTANT RULES:
1. Think step by step and show your reasoning clearly.
2. The answer may be a number, a mathematical expression, a set of values, \
a polynomial, an interval, a fraction, or any other mathematical object.
3. When you have determined the final answer, output it in EXACTLY this format:
   ANSWER: <your answer>
   For example:
     ANSWER: 42
     ANSWER: $\\frac{1}{2}$
     ANSWER: $P(x) = 2x^2 + c$
     ANSWER: $(-\\infty, 0) \\cup \\{\\frac{1}{2}\\}$
     ANSWER: 2026, 2030
4. Use LaTeX notation for non-trivial mathematical expressions.
5. Make sure your ANSWER line is the LAST thing you output, after all reasoning \
and verification.
6. If the problem asks to "find all" values/functions, list ALL of them in your answer.
7. Simplify your answer as much as possible."""

PYTHON_EXECUTION_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": (
            "Execute Python code in a sandboxed environment. "
            "Use this to perform calculations, verify results, solve equations, "
            "enumerate cases, or run any computation that helps solve the math problem. "
            "Common libraries available: math, itertools, functools, fractions, "
            "decimal, sympy, numpy."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute. Print the results you want to see.",
                }
            },
            "required": ["code"],
        },
    },
}


def _normalize_response(raw_response, mode: str, iteration: int = 0):
    """
    Normalize the response from chat_completion_with_tools into a
    uniform structure regardless of backend mode.

    Returns (finish_reason, assistant_message, thinking_content) where
    assistant_message is a namespace-like object with:
      - .content (str or None)
      - .tool_calls (list or None), each with .id, .function.name, .function.arguments
    """
    if mode == "local":
        choice = raw_response.choices[0]
        thinking_content = (
            getattr(choice.message, "reasoning_content", None)
            or getattr(choice.message, "reasoning", None)
        )
        return choice.finish_reason, choice.message, thinking_content
    else:
        # raw_response is a dict from the proxy gateway.
        if not isinstance(raw_response, dict):
            return "stop", SimpleNamespace(content="", tool_calls=None, role="assistant"), None

        message = raw_response.get("message", "")

        if isinstance(message, dict) and message.get("is_function_call"):
            tool_calls = [SimpleNamespace(
                id=f"proxy_call_{iteration}",
                function=SimpleNamespace(
                    name=message.get("function_call_name", ""),
                    arguments=message.get("function_call_args", "{}"),
                ),
            )]
            assistant_message = SimpleNamespace(
                content=None,
                tool_calls=tool_calls,
                role="assistant",
            )
            return "tool_calls", assistant_message, None
        else:
            content = message if isinstance(message, str) else str(message)
            assistant_message = SimpleNamespace(
                content=content,
                tool_calls=None,
                role="assistant",
            )
            return "stop", assistant_message, None


def _message_to_dict(assistant_message) -> dict:
    """Convert an assistant message (SDK object or SimpleNamespace) to a dict
    suitable for appending to the messages list."""
    if isinstance(assistant_message, dict):
        return assistant_message
    msg = {"role": "assistant", "content": assistant_message.content or ""}
    if assistant_message.tool_calls:
        msg["tool_calls"] = []
        for tc in assistant_message.tool_calls:
            msg["tool_calls"].append({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            })
    return msg


def solve_problem(
    model_name: str, problem: str, max_iterations: int = 20, **kwargs
) -> dict:
    """Run the agent loop to solve a single IMO-AnswerBench problem.

    Returns a dict with:
      - answer: the final text response (str)
      - conversation_history: list of conversation turns with thinking and tool calls
      - tool_call_count: total number of tool calls made
      - has_thinking: whether any thinking/reasoning content was present
    """
    mode = _get_model_mode(**kwargs)
    enable_tools = str(kwargs.get("enable_tools", "true")).lower() == "true"
    system_prompt = SYSTEM_PROMPT if enable_tools else SYSTEM_PROMPT_NO_TOOLS
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
    ]

    conversation_history = []
    tool_call_count = 0
    sandbox_error_types = []
    has_thinking = "unknown" if mode == "proxy" else False

    tools_list = [PYTHON_EXECUTION_TOOL] if enable_tools else None

    # Create code executor only when tools are enabled
    code_executor = None
    if enable_tools:
        code_executor_type = str(kwargs.get("code_executor", "sandbox")).lower()
        if code_executor_type == "local":
            code_executor = LocalPythonExecutor()
        else:
            code_executor = SandboxManager(base_url=kwargs.get("sandbox_url"))

    try:
        for iteration in range(max_iterations):
            raw_response = chat_completion_with_tools(
                messages=messages,
                model=model_name,
                tools=tools_list,
                temperature=0.0,
                **{k: v for k, v in kwargs.items() if k not in ("model_name", "sandbox_url", "enable_tools", "code_executor")},
            )

            if raw_response is None:
                break

            finish_reason, assistant_message, thinking_content = _normalize_response(
                raw_response, mode, iteration=iteration
            )

            if thinking_content and has_thinking != "unknown":
                has_thinking = True

            # Build a record for this turn
            turn_record = {
                "iteration": iteration,
                "role": "assistant",
                "content": assistant_message.content or "",
            }
            if thinking_content:
                turn_record["thinking_content"] = thinking_content

            # Record tool calls in this turn
            if assistant_message.tool_calls:
                turn_tool_calls = []
                for tool_call in assistant_message.tool_calls:
                    turn_tool_calls.append({
                        "id": tool_call.id,
                        "function_name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    })
                turn_record["tool_calls"] = turn_tool_calls

            conversation_history.append(turn_record)

            # Append assistant message to conversation as dict
            messages.append(_message_to_dict(assistant_message))

            # If the model is done (no tool calls), return the structured result
            if finish_reason == "stop" or not assistant_message.tool_calls:
                return {
                    "answer": assistant_message.content or "",
                    "conversation_history": conversation_history,
                    "tool_call_count": tool_call_count,
                    "has_thinking": has_thinking,
                    "sandbox_error_types": sandbox_error_types,
                }

            # Process each tool call
            for tool_call in assistant_message.tool_calls:
                if tool_call.function.name == "execute_python":
                    arguments = json.loads(tool_call.function.arguments)
                    code = arguments.get("code", "")
                    execution_result = code_executor.execute_code(code)
                    tool_call_count += 1

                    result_output = execution_result["output"]
                    error_type = execution_result.get("error_type")
                    if error_type:
                        sandbox_error_types.append(error_type)

                    # Record tool execution result
                    tool_record = {
                        "iteration": iteration,
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "code": code,
                        "result": result_output,
                    }
                    if error_type:
                        tool_record["error_type"] = error_type
                    conversation_history.append(tool_record)

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_output,
                        }
                    )

        # If we exhausted iterations, return whatever we have
        last_content = (
            messages[-1].get("content", "")
            if isinstance(messages[-1], dict)
            else getattr(messages[-1], "content", "")
        )
        return {
            "answer": last_content or "",
            "conversation_history": conversation_history,
            "tool_call_count": tool_call_count,
            "has_thinking": has_thinking,
            "sandbox_error_types": sandbox_error_types,
        }
    finally:
        # Always destroy code executor after task completes
        if code_executor is not None:
            code_executor.destroy()


def run(input: dict[str, dict], **kwargs) -> dict[str, dict]:
    """
    Agent entry point for the HAL harness.

    Args:
        input: Dictionary mapping task IDs to task data.
               Each task has {"problem": "...", "answer": "...", ...}.
        **kwargs: Must include 'model_name'. Optional: 'max_iterations'.

    Returns:
        Dictionary mapping task IDs to structured results with 'answer' and 'metrics'.
    """
    assert "model_name" in kwargs, "model_name is required"

    model_name = kwargs["model_name"]
    max_iterations = int(kwargs.get("max_iterations", "20"))

    results = {}
    for task_id, task_data in input.items():
        problem_text = task_data.get("problem", "")
        solve_result = solve_problem(
            model_name, problem_text, max_iterations,
            **{k: v for k, v in kwargs.items() if k not in ("model_name", "max_iterations")},
        )
        results[task_id] = {
            "answer": solve_result["answer"],
            "metrics": {
                "conversation_history": solve_result["conversation_history"],
                "tool_call_count": solve_result["tool_call_count"],
                "has_thinking": solve_result["has_thinking"],
                "sandbox_error_types": solve_result["sandbox_error_types"],
            },
        }

    return results
