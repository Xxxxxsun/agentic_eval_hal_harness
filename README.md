# Agentic Eval HAL Harness

基于 HAL (Holistic Agent Leaderboard) 框架的智能体评测工具，支持在分布式 vLLM 推理服务上运行Agentic Benchmarks评测。

## 快速开始

### 使用 `local_vllm_server_eval.sh` 脚本

该脚本用于在双节点 Ray 集群上启动 vLLM 分布式推理服务，并运行智能体评测任务。

#### 基本用法

```bash
./scripts/local_vllm_server_eval.sh --model-name <模型名称> --benchmark <评测基准>
```

#### 参数说明

| 参数 | 必填 | 说明 | 示例 |
|------|------|------|------|
| `--model-name` | ✅ | 模型名称，用于 vLLM 服务和评测标识 | `qwen3-32b` |
| `--benchmark` | ✅ | 评测基准名称，需对应 `agents/` 目录下的 agent | `aime2025`, `gpqa_diamond`, `imo_answerbench` |
| `--max-tasks` | ❌ | 限制评测的最大任务数量，用于快速测试 | `10` |
| `--num-samples` | ❌ | 每个任务的采样次数，用于计算 pass@N 和 avg@N | `5` |
| `--max-concurrent` | ❌ | 单次遍历中同时运行的最大任务数 | `4` |
| `--enable-tools` | ❌ | 是否允许模型使用代码工具（默认 `true`），设为 `false` 时模型仅纯文本推理 | `true`, `false` |
| `--code-executor` | ❌ | 代码执行器类型（默认 `sandbox`）：`sandbox` 使用远程 iagent 沙盒，`local` 使用本地 Python 执行（更稳定，无网络依赖） | `sandbox`, `local` |
| `--debug` | ❌ | 开启 debug 模式，评测完成后保持容器运行便于调试；不加此参数时评测完成后自动退出 | （flag，无需传值） |

#### 示例

**基础评测**（运行所有任务，单次采样）：
```bash
./scripts/local_vllm_server_eval.sh \
    --model-name qwen3-32b \
    --benchmark aime2025
```

**快速测试**（仅运行 10 个任务）：
```bash
./scripts/local_vllm_server_eval.sh \
    --model-name qwen3-32b \
    --benchmark aime2025 \
    --max-tasks 10
```

**多次采样评测**（计算 pass@5 和 avg@5）：
```bash
./scripts/local_vllm_server_eval.sh \
    --model-name qwen3-32b \
    --benchmark aime2025 \
    --num-samples 5
```

**完整配置示例**：
```bash
./scripts/local_vllm_server_eval.sh \
    --model-name qwen3-32b \
    --benchmark imo_answerbench \
    --max-tasks 50 \
    --num-samples 3 \
    --max-concurrent 8
```

**禁用工具调用**（模型仅纯文本推理，不使用代码沙盒）：
```bash
./scripts/local_vllm_server_eval.sh \
    --model-name qwen3-32b \
    --benchmark aime2025 \
    --enable-tools false
```

**使用本地 Python 执行器**（替代远程沙盒，更稳定）：
```bash
./scripts/local_vllm_server_eval.sh \
    --model-name qwen3-32b \
    --benchmark aime2025 \
    --code-executor local
```

**开启 debug 模式**（评测完成后保持容器运行，便于登录调试）：
```bash
./scripts/local_vllm_server_eval.sh \
    --model-name qwen3-32b \
    --benchmark aime2025 \
    --debug
```

## 支持的评测基准

| 基准名称 | 说明 | Agent 目录 |
|----------|------|------------|
| `aime2025` | AIME 2025 数学竞赛题目（30 题） | `agents/aime2025_agent` |
| `gpqa_diamond` | GPQA Diamond 研究生级别科学问答（198 题，涵盖生物、物理、化学） | `agents/gpqa_diamond_agent` |
| `imo_answerbench` | IMO 级别奥数简答题（400 题） | `agents/imo_answerbench_agent` |

## 评测指标

### 基础指标

- **accuracy**: 正确率
- **total_tool_calls**: 工具调用总次数
- **successful_tool_calls**: 成功的工具调用次数
- **failed_tool_calls**: 失败的工具调用次数（基于 `sandbox_error_types` 精确统计）
- **tasks_with_tool_calls**: 使用了工具的任务数量
- **sandbox_error_type_counts**: 沙盒执行错误类型的细粒度分布（按 `ename` 统计，如 `SyntaxError`、`NameError`、`TimeoutError` 等）

### 多次采样指标（`--num-samples > 1` 时）

- **avg@N**: 所有任务的平均正确率（每个任务 N 次采样的平均值再取平均）
- **pass@N**: 至少有一次正确的任务比例
- **avg_tool_calls_per_sample**: 每个样本（task × round）的平均工具调用次数
- **avg_successful_tool_calls_per_sample**: 每个样本的平均成功工具调用次数
- **avg_failed_tool_calls_per_sample**: 每个样本的平均失败工具调用次数
- **tool_call_usage_rate_per_sample**: 使用了工具调用的样本占总样本的比例（0~1）
- **sandbox_error_type_counts**: 所有轮次汇总的沙盒错误类型分布
- **per_round_sandbox_error_type_counts**: 按轮次分别统计的沙盒错误类型分布

## 环境要求

- **节点数**: 2 个节点（`NNODES=2`）
- **GPU 数量**: 每节点 8 张 GPU（`NUM_ACCELERATORS=8`）
- **依赖**: rclone, Ray, vLLM, qwen_vl_utils, ijson

## vLLM 服务配置

脚本内置的 vLLM 配置：

| 配置项 | 值 |
|--------|-----|
| tensor_parallel_size | 16 |
| gpu_memory_utilization | 0.8 |
| max_num_seqs | 256 |
| max_model_len | 32768 |
| tool_call_parser | qwen3_xml |
| reasoning_parser | qwen3 |

## 输出结果

评测结果保存在 `results/<benchmark_name>/<run_id>/` 目录下：

- `<run_id>.json`: 原始评测结果
- `<run_id>_UPLOAD.json`: 完整的评测报告，包含指标汇总
- `<run_id>_RAW_SUBMISSIONS.jsonl`: 每个任务的原始输出

## 注意事项

1. 脚本会自动从 `PRIMUS_SOURCE_ACTOR_CHECKPOINT_DIR` 下载模型到 `/tmp/model_actor`
2. Master 节点（`RANK=0`）负责启动 vLLM 服务和运行评测
3. Worker 节点自动加入 Ray 集群并保持运行
4. 评测完成后容器会保持运行，便于调试
