#!/bin/bash
set -x
export NCCL_WORK_FIFO_DEPTH=4194304

# 解析命令行参数
MODEL_NAME=""
BENCHMARK_NAME=""
MAX_TASKS=""
NUM_SAMPLES=""
MAX_CONCURRENT=""

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
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --model-name <model_name> --benchmark <benchmark_name> [--max-tasks <num>] [--num-samples <num>] [--max-concurrent <num>]"
            exit 1
            ;;
    esac
done

# 参数校验
if [[ -z "${MODEL_NAME}" ]]; then
    echo "Error: --model-name is required"
    echo "Usage: $0 --model-name <model_name> --benchmark <benchmark_name> [--max-tasks <num>] [--num-samples <num>] [--max-concurrent <num>]"
    exit 1
fi

if [[ -z "${BENCHMARK_NAME}" ]]; then
    echo "Error: --benchmark is required"
    echo "Usage: $0 --model-name <model_name> --benchmark <benchmark_name> [--max-tasks <num>] [--num-samples <num>] [--max-concurrent <num>]"
    exit 1
fi

echo "Using MODEL_NAME: ${MODEL_NAME}"
echo "Using BENCHMARK_NAME: ${BENCHMARK_NAME}"
if [[ -n "${MAX_TASKS}" ]]; then
    echo "Using MAX_TASKS: ${MAX_TASKS}"
fi
if [[ -n "${NUM_SAMPLES}" ]]; then
    echo "Using NUM_SAMPLES: ${NUM_SAMPLES}"
fi
if [[ -n "${MAX_CONCURRENT}" ]]; then
    echo "Using MAX_CONCURRENT: ${MAX_CONCURRENT}"
fi

# 安装依赖
apt-get update && apt-get install -y rclone
pip install qwen_vl_utils ijson
wget "http://yum.tbsite.net/aliyun-pypi/packages/py-vipserver/py_vipserver-0.0.3-py3-none-any.whl#sha256=bfc2fd439b6b64e874bd45eeb0a525f7ca337f7406e3ceb23ece834c7ca049b9" -O py_vipserver-0.0.3-py3-none-any.whl
pip install py_vipserver-0.0.3-py3-none-any.whl
wget "http://yum.tbsite.net/aliyun-pypi/packages/iagent-adk/iagent_adk-1.1.7-py3-none-any.whl#sha256=dbd7b32311129c2ea10cb87629dade85a45afcc46a9399c5b481986decdecf54" -O iagent_adk-1.1.7-py3-none-any.whl
pip install iagent_adk-1.1.7-py3-none-any.whl

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 环境检查
if [[ "${NNODES}" != "2" ]]; then
  echo "Assertion failed: NNODES must be 2, but got '${NNODES}'"
  exit 1
fi

if [[ "${NUM_ACCELERATORS}" != "8" ]]; then
  echo "Assertion failed: NUM_ACCELERATORS must be 8, but got '${NUM_ACCELERATORS}'"
  exit 1
fi

# vllm 配置参数
tensor_parallel_size=16
gpu_memory_utilization=0.8
max_num_seqs=256
dtype=auto
seed=1234

# 下载模型
rclone copy ${PRIMUS_SOURCE_ACTOR_CHECKPOINT_DIR}/${PRIMUS_SOURCE_ACTOR_CHECKPOINT_ITERATION_DIRNAME} /tmp/model_actor --transfers=10

# 启动 Ray 集群
if [[ "${RANK}" == "0" ]]; then
    # Master 节点：启动 Ray head
    ray start --head --port=6379
    
    # 等待所有 worker 节点加入
    echo "Waiting for Ray workers to join..."
    sleep 30
    
    # 只在 master 节点启动 vllm server
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
    
    # 等待 vllm server 就绪
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
    
    # 运行评测
    cd /root/code/agentic_eval_hal_harness
    export HF_ENDPOINT=https://hf-mirror.com
    export OPENAI_BASE_URL=http://localhost:8234/v1
    export OPENAI_API_KEY=empty
    export GRADER_MODEL=${MODEL_NAME}
    pip install -e .
    
    # 构建 hal-eval 命令
    HAL_EVAL_CMD="hal-eval \
        --agent_name ${MODEL_NAME} \
        --agent_dir agents/${BENCHMARK_NAME}_agent \
        --agent_function main.run \
        --benchmark ${BENCHMARK_NAME} \
        -A model_name=${MODEL_NAME}"
    
    # 如果指定了 max-tasks，添加该参数
    if [[ -n "${MAX_TASKS}" ]]; then
        HAL_EVAL_CMD="${HAL_EVAL_CMD} --max_tasks ${MAX_TASKS}"
    fi
    
    # 如果指定了 num-samples，添加该参数
    if [[ -n "${NUM_SAMPLES}" ]]; then
        HAL_EVAL_CMD="${HAL_EVAL_CMD} --num_samples ${NUM_SAMPLES}"
    fi
    
    # 如果指定了 max-concurrent，添加该参数
    if [[ -n "${MAX_CONCURRENT}" ]]; then
        HAL_EVAL_CMD="${HAL_EVAL_CMD} --max_concurrent ${MAX_CONCURRENT}"
    fi
    
    # 执行评测
    eval ${HAL_EVAL_CMD}
    
    # 评测完成后，保持容器运行便于调试
    echo "Evaluation completed. Keeping container alive for debugging..."
    wait ${VLLM_PID}
    
else
    # Worker 节点：加入 Ray 集群并保持运行
    ray start --address=${MASTER_ADDR}:6379 --block
fi
