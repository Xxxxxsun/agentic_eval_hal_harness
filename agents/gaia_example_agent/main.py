# Disclaimer: this is not a functional agent and is only for demonstration purposes. This implementation is just a single model call.
from model_client import chat_completion


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    assert "model_name" in kwargs, "model_name is required"
    assert len(input) == 1, "input must contain only one task"

    task_id, task = list(input.items())[0]

    results = {}

    prompt = f"""Please answer the question below. You should:                                                                                                                   
                                                                                                                                                                 
- Return only your answer, which should be a number, or a short phrase with as few words as possible, or a comma separated list of numbers and/or strings.      
- If the answer is a number, return only the number without any units unless specified otherwise.                                                               
- If the answer is a string, don't include articles, and don't use abbreviations (e.g. for states).                                                             
- If the answer is a comma separated list, apply the above rules to each element in the list.                                                                                                                                                                                                                    
                                                                                                                                                                 
Here is the question:

{task["Question"]}"""

    result = chat_completion(
        messages=[
            {"role": "user", "content": prompt},
        ],
        model=kwargs["model_name"],
        max_tokens=2000,
        **kwargs,
    )

    results[task_id] = result

    return results