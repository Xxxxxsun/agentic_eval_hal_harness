import json
import sys
import os
from types import SimpleNamespace

# Add agents directory to path for common module import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_client import chat_completion_with_tools, _get_model_mode
from common.sandbox_executor import SandboxManager

SYSTEM_PROMPT = """You are an expert scientist solving graduate-level multiple-choice questions from the GPQA (Graduate-Level Google-Proof Q&A) benchmark.

These questions span biology, physics, and chemistry, and are designed to be very challenging — even domain experts only achieve ~65% accuracy.

You have access to a Python code interpreter tool that you can use to:
- Perform complex scientific calculations
- Verify your reasoning numerically
- Solve equations symbolically (using sympy)
- Run simulations or numerical experiments

IMPORTANT RULES:
1. Read the question and ALL four answer choices carefully before reasoning.
2. Think step by step and show your reasoning.
3. Use the Python code interpreter to verify calculations when helpful.
4. You must choose exactly ONE answer from A, B, C, or D.
5. When you have determined the final answer, output it in EXACTLY this format: ANSWER: X
   where X is one of A, B, C, or D.
   For example: ANSWER: B
6. Make sure your final answer is the LAST thing you output, after all reasoning and verification.
7. IMPORTANT: The code interpreter maintains context across multiple calls. Variables and imports from previous code executions are preserved and can be reused in subsequent calls."""

PYTHON_EXECUTION_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": (
            "Execute Python code in a sandboxed environment. "
            "Use this to perform calculations, verify results, solve equations, "
            "run simulations, or any computation that helps answer the question. "
            "Common libraries available: math, itertools, functools, fractions, decimal, sympy, numpy, scipy."
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


def _format_question_prompt(question: str, choices: dict) -> str:
    """Format the question and choices into a clear prompt for the model.

    If choices dict is non-empty, appends formatted A/B/C/D options.
    If choices dict is empty (pre-processed dataset where choices are already
    embedded in the question text), uses the question text as-is.
    """
    if choices:
        lines = [question, "", "Answer choices:"]
        for label in ["A", "B", "C", "D"]:
            if label in choices:
                lines.append(f"  {label}. {choices[label]}")
        lines.append("")
        lines.append("Please analyze the question carefully and select the correct answer (A, B, C, or D).")
        return "\n".join(lines)

    # Pre-processed format: choices are already in the question text
    return question + "\n\nPlease analyze the question carefully and select the correct answer (A, B, C, or D)."


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


def solve_problem(model_name: str, question: str, choices: dict, max_iterations: int = 15, **kwargs) -> dict:
    """Run the agent loop to solve a single GPQA-Diamond problem.

    Returns a dict with:
      - answer: the final text response (str)
      - conversation_history: list of conversation turns with thinking and tool calls
      - tool_call_count: total number of tool calls made
    """
    mode = _get_model_mode(**kwargs)
    formatted_prompt = _format_question_prompt(question, choices)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": formatted_prompt},
    ]

    conversation_history = []
    tool_call_count = 0
    sandbox_error_types = []

    sandbox_manager = SandboxManager(
        base_url=kwargs.get("sandbox_url")
    )

    try:
        for iteration in range(max_iterations):
            raw_response = chat_completion_with_tools(
                messages=messages,
                model=model_name,
                tools=[PYTHON_EXECUTION_TOOL],
                temperature=0.0,
                **{k: v for k, v in kwargs.items() if k not in ("model_name", "sandbox_url")},
            )

            if raw_response is None:
                break

            finish_reason, assistant_message, thinking_content = _normalize_response(
                raw_response, mode, iteration=iteration
            )

            turn_record = {
                "iteration": iteration,
                "role": "assistant",
                "content": assistant_message.content or "",
            }
            if thinking_content:
                turn_record["thinking_content"] = thinking_content

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
            messages.append(_message_to_dict(assistant_message))

            if finish_reason == "stop" or not assistant_message.tool_calls:
                return {
                    "answer": assistant_message.content or "",
                    "conversation_history": conversation_history,
                    "tool_call_count": tool_call_count,
                    "sandbox_error_types": sandbox_error_types,
                }

            for tool_call in assistant_message.tool_calls:
                if tool_call.function.name == "execute_python":
                    arguments = json.loads(tool_call.function.arguments)
                    code = arguments.get("code", "")
                    execution_result = sandbox_manager.execute_code(code)
                    tool_call_count += 1

                    result_output = execution_result["output"]
                    error_type = execution_result.get("error_type")
                    if error_type:
                        sandbox_error_types.append(error_type)

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

        last_content = (
            messages[-1].get("content", "")
            if isinstance(messages[-1], dict)
            else getattr(messages[-1], "content", "")
        )
        return {
            "answer": last_content or "",
            "conversation_history": conversation_history,
            "tool_call_count": tool_call_count,
            "sandbox_error_types": sandbox_error_types,
        }
    finally:
        sandbox_manager.destroy()


def run(input: dict[str, dict], **kwargs) -> dict[str, dict]:
    """
    Agent entry point for the HAL harness.

    Args:
        input: Dictionary mapping task IDs to task data.
               Each task has {"question": "...", "choices": {"A": ..., ...}, "answer": "A"}.
        **kwargs: Must include 'model_name'. Optional: 'max_iterations'.

    Returns:
        Dictionary mapping task IDs to structured results with 'answer' and 'metrics'.
    """
    assert "model_name" in kwargs, "model_name is required"

    model_name = kwargs["model_name"]
    max_iterations = int(kwargs.get("max_iterations", "15"))

    results = {}
    for task_id, task_data in input.items():
        question_text = task_data.get("question", "")
        choices = task_data.get("choices", {})
        solve_result = solve_problem(
            model_name, question_text, choices, max_iterations,
            **{k: v for k, v in kwargs.items() if k not in ("model_name", "max_iterations")},
        )
        results[task_id] = {
            "answer": solve_result["answer"],
            "metrics": {
                "conversation_history": solve_result["conversation_history"],
                "tool_call_count": solve_result["tool_call_count"],
                "sandbox_error_types": solve_result["sandbox_error_types"],
            },
        }

    return results
