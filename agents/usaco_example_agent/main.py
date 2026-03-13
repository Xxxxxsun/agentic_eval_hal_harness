# This is an example agent that solves a USACO problem
# Disclaimer: this is not a functional agent and is only for demonstration purposes. This implementation is just a single model call.

from model_client import chat_completion


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    assert "model_name" in kwargs, "model_name is required"
    assert len(input) == 1, "input must contain only one task"

    task_id, task = list(input.items())[0]

    results = {}

    result = chat_completion(
        messages=[
            {
                "role": "user",
                "content": "Solve the following problem: " + task["description"],
            },
        ],
        model=kwargs["model_name"],
        max_tokens=2000,
        **kwargs,
    )

    results[task_id] = result

    return results