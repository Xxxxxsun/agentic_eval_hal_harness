#!/bin/bash
set -x
export NCCL_WORK_FIFO_DEPTH=4194304

MODEL_NAME=""
BENCHMARK_NAME=""
MAX_TASKS=""
NUM_SAMPLES=""
MAX_CONCURRENT=""
ENABLE_TOOLS="true"
CODE_EXECUTOR="local"
SANDBOX_URL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --benchmark)
            BENCHMARK_NAME="$2"
            shift 2
            ;;
        --max-tasks)
            MAX_TASKS="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --max-concurrent)
            MAX_CONCURRENT="$2"
            shift 2
            ;;
        --enable-tools)
            ENABLE_TOOLS="$2"
            shift 2
            ;;
        --code-executor)
            CODE_EXECUTOR="$2"
            shift 2
            ;;
        --sandbox-url)
            SANDBOX_URL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --model-name <model_name> --benchmark <benchmark_name> [--max-tasks <num>] [--num-samples <num>] [--max-concurrent <num>] [--enable-tools <true|false>] [--code-executor <local|sandbox>] [--sandbox-url <url>]"
            exit 1
            ;;
    esac
done

if [[ -z "${MODEL_NAME}" ]]; then
    echo "Error: --model-name is required"
    exit 1
fi

if [[ -z "${BENCHMARK_NAME}" ]]; then
    echo "Error: --benchmark is required"
    exit 1
fi

if [[ "${CODE_EXECUTOR}" != "local" && "${CODE_EXECUTOR}" != "sandbox" ]]; then
    echo "Error: --code-executor must be either 'local' or 'sandbox'"
    exit 1
fi

echo "Using MODEL_NAME: ${MODEL_NAME}"
echo "Using BENCHMARK_NAME: ${BENCHMARK_NAME}"
echo "Using ENABLE_TOOLS: ${ENABLE_TOOLS}"
echo "Using CODE_EXECUTOR: ${CODE_EXECUTOR}"
if [[ -n "${SANDBOX_URL}" ]]; then
    echo "Using SANDBOX_URL: ${SANDBOX_URL}"
fi
if [[ -n "${MAX_TASKS}" ]]; then
    echo "Using MAX_TASKS: ${MAX_TASKS}"
fi
if [[ -n "${NUM_SAMPLES}" ]]; then
    echo "Using NUM_SAMPLES: ${NUM_SAMPLES}"
fi
if [[ -n "${MAX_CONCURRENT}" ]]; then
    echo "Using MAX_CONCURRENT: ${MAX_CONCURRENT}"
fi

apt-get update && apt-get install -y rclone
pip install qwen_vl_utils ijson
wget "http://yum.tbsite.net/aliyun-pypi/packages/iagent-adk/iagent_adk-1.1.4-py3-none-any.whl#sha256=dbd7b32311129c2ea10cb87629dade85a45afcc46a9399c5b481986decdecf54" -O iagent_adk-1.1.4-py3-none-any.whl
pip install iagent_adk-1.1.4-py3-none-any.whl

if [[ "${NNODES}" != "2" ]]; then
  echo "Assertion failed: NNODES must be 2, but got '${NNODES}'"
  exit 1
fi

if [[ "${NUM_ACCELERATORS}" != "8" ]]; then
  echo "Assertion failed: NUM_ACCELERATORS must be 8, but got '${NUM_ACCELERATORS}'"
  exit 1
fi

tensor_parallel_size=16
gpu_memory_utilization=0.8
max_num_seqs=256
dtype=auto
seed=1234

rclone copy ${PRIMUS_SOURCE_ACTOR_CHECKPOINT_DIR}/${PRIMUS_SOURCE_ACTOR_CHECKPOINT_ITERATION_DIRNAME} /tmp/model_actor --transfers=10

if [[ "${RANK}" == "0" ]]; then
    ray start --head --port=6379

    echo "Waiting for Ray workers to join..."
    sleep 30

    vllm serve /tmp/model_actor \
        --port 8234 \
        --served-model-name ${MODEL_NAME} \
        --max-model-len 32768 \
        --distributed-executor-backend ray \
        --enable-expert-parallel \
        --tensor-parallel-size ${tensor_parallel_size} \
        --gpu-memory-utilization ${gpu_memory_utilization} \
        --max-num-seqs ${max_num_seqs} \
        --trust-remote-code \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_xml \
        --reasoning-parser qwen3 \
        --dtype ${dtype} \
        --seed ${seed} &

    VLLM_PID=$!

    echo "Waiting for vLLM server to be ready..."
    max_retries=60
    retry_count=0
    while ! curl -s http://localhost:8234/health > /dev/null 2>&1; do
        sleep 60
        retry_count=$((retry_count + 1))
        if [[ ${retry_count} -ge ${max_retries} ]]; then
            echo "Error: vLLM server failed to start within timeout"
            exit 1
        fi
        echo "Waiting for vLLM server... (${retry_count}/${max_retries})"
    done
    echo "vLLM server is ready!"

    cd /root/code/agentic_eval_hal_harness
    export HF_ENDPOINT=https://hf-mirror.com
    export OPENAI_BASE_URL=http://localhost:8234/v1
    export OPENAI_API_KEY=empty
    export GRADER_MODEL=${MODEL_NAME}
    pip install -e .

    HAL_EVAL_CMD="hal-eval \
        --agent_name ${MODEL_NAME} \
        --agent_dir agents/${BENCHMARK_NAME}_agent \
        --agent_function main.run \
        --benchmark ${BENCHMARK_NAME} \
        -A model_name=${MODEL_NAME} \
        -A enable_tools=${ENABLE_TOOLS} \
        -A code_executor=${CODE_EXECUTOR}"

    if [[ -n "${SANDBOX_URL}" ]]; then
        HAL_EVAL_CMD="${HAL_EVAL_CMD} -A sandbox_url=${SANDBOX_URL}"
    fi

    if [[ -n "${MAX_TASKS}" ]]; then
        HAL_EVAL_CMD="${HAL_EVAL_CMD} --max_tasks ${MAX_TASKS}"
    fi

    if [[ -n "${NUM_SAMPLES}" ]]; then
        HAL_EVAL_CMD="${HAL_EVAL_CMD} --num_samples ${NUM_SAMPLES}"
    fi

    if [[ -n "${MAX_CONCURRENT}" ]]; then
        HAL_EVAL_CMD="${HAL_EVAL_CMD} --max_concurrent ${MAX_CONCURRENT}"
    fi

    eval ${HAL_EVAL_CMD}

    echo "Evaluation completed. Keeping container alive for debugging..."
    wait ${VLLM_PID}
else
    ray start --address=${MASTER_ADDR}:6379 --block
fi
