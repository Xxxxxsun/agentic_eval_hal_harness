import json
import subprocess
import tempfile
import os
from types import SimpleNamespace

from model_client import chat_completion_with_tools, _get_model_mode

SYSTEM_PROMPT = """You are an expert mathematician solving problems from the American Invitational Mathematics Examination (AIME).

You have access to a Python code interpreter tool that you can use to:
- Perform complex calculations
- Verify your reasoning with brute-force enumeration
- Test conjectures numerically
- Solve equations symbolically (using sympy)

IMPORTANT RULES:
1. Think step by step and show your reasoning.
2. Use the Python code interpreter liberally to verify calculations and explore the problem.
3. AIME answers are always integers from 000 to 999 inclusive.
4. When you have determined the final answer, output it in EXACTLY this format: ANSWER: <number>
   For example: ANSWER: 42
5. Do NOT include leading zeros in your final answer. For example, use ANSWER: 7 not ANSWER: 007.
6. Make sure your final answer is the LAST thing you output, after all reasoning and verification."""

PYTHON_EXECUTION_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": (
            "Execute Python code in a sandboxed environment. "
            "Use this to perform calculations, verify results, solve equations, "
            "enumerate cases, or run any computation that helps solve the math problem. "
            "Common libraries available: math, itertools, functools, fractions, decimal, sympy, numpy."
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

def execute_python_code(code: str, timeout_seconds: int = 30) -> str:
    """Execute Python code in a subprocess and return the output."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as temp_file:
        temp_file.write(code)
        temp_file_path = temp_file.name

    try:
        result = subprocess.run(
            ["python3", temp_file_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )

        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout.strip())
        if result.returncode != 0 and result.stderr:
            output_parts.append(f"Error:\n{result.stderr.strip()}")

        output = "\n".join(output_parts) if output_parts else "(No output)"
        # Truncate very long outputs
        if len(output) > 10000:
            output = output[:10000] + "\n... [output truncated]"
        return output

    except subprocess.TimeoutExpired:
        return f"Error: Code execution timed out after {timeout_seconds} seconds."
    except Exception as execution_error:
        return f"Error executing code: {str(execution_error)}"
    finally:
        os.unlink(temp_file_path)

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
        # raw_response is an OpenAI SDK ChatCompletion object
        choice = raw_response.choices[0]
        thinking_content = (
            getattr(choice.message, "reasoning_content", None)
            or getattr(choice.message, "reasoning", None)
        )
        return choice.finish_reason, choice.message, thinking_content
    else:
        # raw_response is a dict from the proxy gateway.
        # The gateway returns tool calls inside the "message" field:
        #   {"message": {"function_call_args": "...", "function_call_name": "...", "is_function_call": true}}
        # For plain text replies:
        #   {"message": "some text"}
        if not isinstance(raw_response, dict):
            return "stop", SimpleNamespace(content="", tool_calls=None, role="assistant"), None

        message = raw_response.get("message", "")

        if isinstance(message, dict) and message.get("is_function_call"):
            # Gateway returned a tool call
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
            # Plain text reply
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


def solve_problem(model_name: str, problem: str, max_iterations: int = 15, **kwargs) -> dict:
    """Run the agent loop to solve a single AIME problem.

    Returns a dict with:
      - answer: the final text response (str)
      - conversation_history: list of conversation turns with thinking and tool calls
      - tool_call_count: total number of tool calls made
      - has_thinking: whether any thinking/reasoning content was present
    """
    mode = _get_model_mode(**kwargs)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]

    conversation_history = []
    tool_call_count = 0
    has_thinking = "unknown" if mode == "proxy" else False

    for iteration in range(max_iterations):
        raw_response = chat_completion_with_tools(
            messages=messages,
            model=model_name,
            tools=[PYTHON_EXECUTION_TOOL],
            temperature=0.0,
            **{k: v for k, v in kwargs.items() if k not in ("model_name",)},
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
            }

        # Process each tool call
        for tool_call in assistant_message.tool_calls:
            if tool_call.function.name == "execute_python":
                arguments = json.loads(tool_call.function.arguments)
                code = arguments.get("code", "")
                execution_result = execute_python_code(code)
                tool_call_count += 1

                # Record tool execution result
                conversation_history.append({
                    "iteration": iteration,
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "code": code,
                    "result": execution_result,
                })

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": execution_result,
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
    }

def run(input: dict[str, dict], **kwargs) -> dict[str, dict]:
    """
    Agent entry point for the HAL harness.

    Args:
        input: Dictionary mapping task IDs to task data.
               Each task has {"problem": "...", "answer": "..."}.
        **kwargs: Must include 'model_name'. Optional: 'max_iterations'.

    Returns:
        Dictionary mapping task IDs to structured results with 'answer' and 'metrics'.
    """
    assert "model_name" in kwargs, "model_name is required"

    model_name = kwargs["model_name"]
    max_iterations = int(kwargs.get("max_iterations", "15"))

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
            },
        }

    return results
