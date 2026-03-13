# Amazon Bedrock models for USACO

AI Agents for [`USACO`](https://github.com/princeton-nlp/USACO). We can use Amazon Bedrock to create GenAI apps for different models such as Amazon Nova, Anthropic Claude and Meta Llama3 family of models. The [`The Holistic Agent Leaderboard for Reproducible Agent Evaluation`](https://github.com/benediktstroebl/hal-harness) a.k.a. HAL is used for evaluating the performance of these models on the USACO benchmark.

See code in [`main.py`](main.py) and the prompt templates [`here`](prompt_templates) directory.

## Instructions to run

This code is run as part of the `HAL` platform over the tasks in the `USACO` benchmark. To run this code for an individual task for testing purposes you can follow these instructions:

1. Create a new conda envronment.

    ```{.bashrc}
    conda create --name usaco_bedrock_models -y python=3.11 ipykernel
    source activate usaco_bedrock_models;
    pip install -r requirements.txt
    ```

1. The [`input.txt`](./input.txt) included in this repo contains a random task from the USACO benchmark. You can modify this as needed but this should be good as-is for testing purposes.

1. Run the agent code.

    ```{.bashrc}
    python main.py
    ```

1. You should see the generated Python code for the task in `input.txt` as part of the traces printed out on the console.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](./LICENSE) file.
