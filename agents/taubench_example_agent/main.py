from tau_bench.envs import get_env
from tau_bench.types import Action

from model_client import chat_completion


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    assert "model_name" in kwargs, "model_name is required"
    task_id = list(input.keys())[0]

    ### ENV SETUP (usually this should be untouched) ###
    isolated_env = get_env(
        input[task_id]["env"],
        input[task_id]["user_strategy"],
        input[task_id]["user_model"],
        input[task_id]["task_split"],
        input[task_id]["user_provider"],
        input[task_id]["task_index"],
    )
    # get instruction from environment
    instruction = isolated_env.reset(input[task_id]["task_index"]).observation

    ### YOUR AGENT CODE HERE ###
    result = chat_completion(
        messages=[
            {"role": "user", "content": instruction},
        ],
        model=kwargs["model_name"],
        max_tokens=2000,
        **kwargs,
    )

    ### ACTION ###
    action = Action(name=result, kwargs={})
    response = isolated_env.step(action)