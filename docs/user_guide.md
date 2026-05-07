# ModelPerf 用户指南

> **版本**：v1.0  
> **适用项目**：`/mnt/d/ubuntu/opencode/training_framework/ModelPerf`  
> **目标读者**：首次接触 ModelPerf 的用户  
> **环境要求**：纯 CPU 运行，无需 GPU  

---

## 目录

1. [项目概述](#1-项目概述)
2. [快速开始](#2-快速开始)
3. [核心概念](#3-核心概念)
4. [独立使用 ModelPerf](#4-独立使用-modelperf)
5. [在 Pai-Megatron-Patch 中使用](#5-在-pai-megatron-patch-中使用)
6. [在 Megatron-LM 中使用](#6-在-megatron-lm-中使用)
7. [性能仿真与 What-if 分析](#7-性能仿真与-what-if-分析)
8. [Auto-tuning 自动策略推荐](#8-auto-tuning-自动策略推荐)
9. [可视化报告](#9-可视化报告)
10. [API 速查](#10-api-速查)
11. [常见问题](#11-常见问题)

---

## 1. 项目概述

### 1.1 什么是 ModelPerf

ModelPerf 是一个面向大模型（LLM）分布式训练的性能仿真与分析框架。它位于 `/mnt/d/ubuntu/opencode/training_framework/ModelPerf`，与同级目录下的 `Pai-Megatron-Patch-12.0` 和 `Megatron-LM-20250707` 配合使用，也可以在任意 PyTorch 模型上独立运行。

ModelPerf 采用 **Capture-Parameterize-Execute** 三层架构：

- **Capture（捕获层）**：通过 5 层 Hook 架构，在真实训练的前向/反向过程中捕获完整的计算图，包括计算算子和通信算子。
- **Parameterize（参数化层）**：使用 `sympy` 将张量形状、通信量、FLOPs 等转换为符号公式（如 `B*S*H/TP`）。
- **Execute（执行层）**：通过 Roofline 模型估算计算时间，带宽模型估算通信时间，内存跟踪器估算峰值显存，最终在纯 CPU 上完成性能仿真。

### 1.2 为什么选择 ModelPerf

| 核心价值 | 说明 |
|---------|------|
| **一次捕获，多次仿真** | 捕获一次计算图后，可在秒级内评估数百种并行策略配置，无需重新启动训练 |
| **What-if 分析** | 修改 TP、PP、DP、Batch Size 等参数后，立即获得新的迭代时间、吞吐量和内存预测 |
| **OOM 预判** | 在真实训练启动前，预测目标配置是否会超出 GPU 显存上限 |
| **自动策略推荐** | 基于搜索空间自动生成并筛选最优并行策略组合，输出帕累托前沿 |
| **纯 CPU 运行** | 所有操作无需 GPU，适合在开发机、CI/CD 环境中快速验证配置 |

### 1.3 典型工作流

```
真实训练脚本（CPU） -> 捕获计算图 -> 符号化参数 -> 虚拟执行仿真
                              |
                    What-if 修改配置 -> 重新仿真 -> 可视化报告
                              |
                    Auto-tuning 搜索最优策略 -> 导出推荐配置
```

在 Qwen3 0.6B 模型的真实训练验证中，ModelPerf 成功捕获了 **3161 个节点的计算图**（含 261 个通信节点和 3154 条边），并通过 142 个单元测试验证框架正确性。

---

## 2. 快速开始

### 2.1 环境要求

- **Python**：3.10+
- **PyTorch**：CPU 版本即可（无需 CUDA）
- **依赖包**：`sympy`（符号计算）
- **可选**：`matplotlib`（可视化报告，无则自动回退到 HTML）

> **重要提示**：ModelPerf 的所有操作均为 CPU-only，不需要 NVIDIA GPU 或 CUDA Toolkit。项目中涉及的 A100/H100 算力、带宽等数值仅作为仿真输入的理论参数。

### 2.2 安装步骤

```bash
# 1. 进入项目目录
cd /mnt/d/ubuntu/opencode/training_framework/ModelPerf

# 2. 安装依赖
pip install sympy matplotlib

# 3. 如果使用 Megatron-LM / Pai-Patch 集成，需设置 PYTHONPATH
export PYTHONPATH="/mnt/d/ubuntu/opencode/training_framework/ModelPerf:\
/mnt/d/ubuntu/opencode/training_framework/Pai-Megatron-Patch-12.0:\
/mnt/d/ubuntu/opencode/training_framework/Pai-Megatron-Patch-12.0/backends/megatron/Megatron-LM-20250707:\
$PYTHONPATH"
```

### 2.3 运行单元测试

```bash
cd /mnt/d/ubuntu/opencode/training_framework/ModelPerf
python -m pytest tests/ -v
```

当前测试情况：**142 tests passed**（含 `test_graph.py`、`test_roofline.py`、`test_bandwidth.py`、`test_memory_tracker.py`、`test_symbolic.py`、`test_coordinator.py`、`test_auto_tuner.py`、`test_visualization.py` 等）。

### 2.4 运行端到端示例

```bash
cd /mnt/d/ubuntu/opencode/training_framework/ModelPerf
python examples/end_to_end_pipeline.py
```

该示例会：
1. 加载已捕获的配置（`examples/output/captured_configs/`）
2. 构建合成计算图
3. 运行虚拟执行仿真
4. 执行 What-if 分析（对比 1F1B 与 GPipe）
5. 生成可视化报告到 `examples/output/reports/`

---

## 3. 核心概念

### 3.1 ComputationalGraph（计算图）

`ComputationalGraph` 是 ModelPerf 的核心数据结构，定义于 `modelperf/capture/graph.py`。

```python
from modelperf.capture import ComputationalGraph, GraphNode, OpType, CommType

graph = ComputationalGraph()
graph.model_config = {"hidden_size": 1024, "num_layers": 4}
graph.strategy_config = {"tensor_parallel_size": 1, "pipeline_parallel_size": 1}

node = graph.add_node(
    node_id="linear_0",
    op_type="Linear",
    node_type=OpType.COMPUTE,
    input_shapes=[(2, 1024)],
    output_shapes=[(2, 512)],
    params={"weight": (1024, 512)},
)
graph.add_edge("prev_node_id", "linear_0")
```

**关键字段**：
- `nodes`: 字典，键为 `node_id`，值为 `GraphNode`
- `edges`: 有向边列表 `(from_id, to_id)`
- `forward_nodes` / `backward_nodes`: 前向/反向节点 ID 列表
- `model_config` / `strategy_config` / `system_config`: 三类配置字典

`GraphNode` 支持以下 `OpType`：
- `COMPUTE` — 计算算子（Linear、Attention、Norm 等）
- `COMMUNICATION` — 通信算子（All-Reduce、All-Gather 等）
- `AUTOGRAD_FWD` / `AUTOGRAD_BWD` — 自定义 Autograd Function
- `BACKWARD` — 反向传播钩子节点
- `OPTIMIZER` — 优化器步骤

### 3.2 五层 Hook 架构

ModelPerf 通过五层 Hook 捕获完整的训练计算图：

| 层级 | 类名 | 文件 | 技术方案 | 捕获目标 |
|------|------|------|---------|---------|
| Layer 1 | `ModuleCapture` | `capture/module_hook.py` | `register_forward_hook` | 所有 `nn.Module` 的前向调用、输入/输出 shape、参数 shape |
| Layer 2 | `CommunicationCapture` | `capture/comm_hook.py` | Monkey-patch `torch.distributed` | All-Reduce、All-Gather、Reduce-Scatter、All-to-All、Broadcast |
| Layer 3 | `AutogradCapture` | `capture/autograd_hook.py` | Patch `forward`/`backward` | 自定义 Autograd Function（如 `bias_gelu_impl`、`linear_async`） |
| Layer 4 | `BackwardCapture` | `capture/backward_hook.py` | `Tensor.register_hook` | 反向传播梯度流，建立 forward->backward 边 |
| Layer 5 | `ProfilerCapture` | `capture/profiler_hook.py` | `torch.profiler.profile` | 运行时 Kernel 时间校准、FLOPs 统计 |

**Layer 1 -> Layer 2 联动**：`ModuleCapture` 在创建节点时通过回调通知 `CommunicationCapture`，使通信节点自动关联到当前计算节点，建立 `compute -> comm` 的边。

### 3.3 Symbolic Parameters（符号化参数）

使用 `sympy` 将张量形状和通信量符号化，支持在修改配置后自动重求值。

```python
from modelperf.symbolic.shape_inferer import SymbolicShapeInferer

inferer = SymbolicShapeInferer()
B = inferer.get_symbol('B')      # batch_size
S = inferer.get_symbol('S')      # sequence_length
H = inferer.get_symbol('H')      # hidden_size
TP = inferer.get_symbol('TP')    # tensor_parallel_size

# ColumnParallelLinear 的输出 shape
output_shape = (S, B, H / TP)

# 代入具体数值求值
shape = inferer.evaluate(output_shape, {"S": 128, "B": 2, "H": 1024, "TP": 2})
# 结果: (128, 2, 512.0)
```

**预定义符号**：`B`、`S`、`H`、`V`、`F`、`NH`、`NG`、`KV`、`TP`、`PP`、`DP`、`CP`、`EP`。

### 3.4 VirtualExecutor（虚拟执行引擎）

`VirtualExecutor` 接收 `ComputationalGraph`，遍历 forward 和 backward 节点，分别用 `RooflineModel` 和 `BandwidthModel` 估算计算时间和通信时间，最终输出 `ExecutionResult`。

### 3.5 CaptureCoordinator（捕获协调器）

`CaptureCoordinator` 是五层 Hook 的统一入口，负责生命周期管理：

```python
from modelperf.capture import CaptureCoordinator

coordinator = CaptureCoordinator()
coordinator.attach(model)     # 注册所有 Hook
coordinator.start()           # 开始捕获
# ... 运行训练前向/反向 ...
coordinator.stop()            # 停止捕获
graph = coordinator.get_graph()
```

支持上下文管理器：

```python
with coordinator:
    output = model(input)
    loss = output.sum()
    loss.backward()
```

---

## 4. 独立使用 ModelPerf

### 4.1 构建合成计算图并仿真

不依赖任何外部框架，从零构建计算图并运行仿真：



### 4.2 使用 CaptureCoordinator 捕获 PyTorch 模型



### 4.3 导出与导入计算图


---

## 4. 独立使用 ModelPerf

### 4.1 构建合成计算图并仿真

不依赖任何外部框架，从零构建计算图并运行仿真：

```python
from modelperf.capture import ComputationalGraph, OpType
from modelperf.simulation import VirtualExecutor, ExecutionConfig

graph = ComputationalGraph()
graph.model_config = {
    "hidden_size": 256,
    "num_layers": 4,
    "num_attention_heads": 8,
    "ffn_hidden_size": 768,
    "vocab_size": 1000,
    "seq_length": 64,
}
graph.strategy_config = {
    "tensor_parallel_size": 1,
    "pipeline_parallel_size": 1,
    "data_parallel_size": 1,
    "micro_batch_size": 1,
}

emb = graph.add_node(
    node_id="emb",
    op_type="Embedding",
    node_type=OpType.COMPUTE,
    output_shapes=[(64, 1, 256)],
)
graph.forward_nodes.append(emb.node_id)

for i in range(4):
    ln = graph.add_node(
        node_id=f"ln_{i}",
        op_type="LayerNorm",
        node_type=OpType.COMPUTE,
        output_shapes=[(64, 1, 256)],
    )
    graph.forward_nodes.append(ln.node_id)

config = ExecutionConfig(
    peak_compute_tflops=312.0,
    peak_bandwidth_gbs=2039.0,
    network_bandwidth_gbs=600.0,
    memory_per_device_gb=80.0,
)
executor = VirtualExecutor(config=config)
result = executor.execute(graph)

print(f"Iteration time: {result.iteration_time_ms:.2f} ms")
print(f"Peak memory: {result.peak_memory_mb:.2f} MB")
print(f"Bottleneck: {result.bottleneck}")
```

### 4.2 使用 CaptureCoordinator 捕获 PyTorch 模型

```python
import torch
import torch.nn as nn
from modelperf.capture import CaptureCoordinator

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = SimpleModel()
coordinator = CaptureCoordinator()
coordinator.attach(model)

with coordinator:
    x = torch.randn(2, 10, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()

graph = coordinator.get_graph()
print(f"Captured {len(graph.nodes)} nodes, {len(graph.edges)} edges")
```

### 4.3 导出与导入计算图

```python
import json

graph_dict = graph.to_dict()
with open("/mnt/d/ubuntu/opencode/training_framework/ModelPerf/examples/output/my_graph.json", "w") as f:
    json.dump(graph_dict, f, indent=2)

with open("/mnt/d/ubuntu/opencode/training_framework/ModelPerf/examples/output/my_graph.json", "r") as f:
    loaded = json.load(f)
```
---

## 5. 在 Pai-Megatron-Patch 中使用

### 5.1 前置条件

确保 `PYTHONPATH` 包含以下路径：

```bash
export PYTHONPATH="/mnt/d/ubuntu/opencode/training_framework/ModelPerf:/mnt/d/ubuntu/opencode/training_framework/Pai-Megatron-Patch-12.0:/mnt/d/ubuntu/opencode/training_framework/Pai-Megatron-Patch-12.0/backends/megatron/Megatron-LM-20250707:$PYTHONPATH"
```

同时设置 CPU 环境：

```bash
export CUDA_VISIBLE_DEVICES=-1
```

### 5.2 与 Qwen3 训练脚本的自动集成

Pai-Megatron-Patch 的 Qwen3 训练入口位于：

```
/mnt/d/ubuntu/opencode/training_framework/Pai-Megatron-Patch-12.0/examples/qwen3/train_qwen3_0.6B.sh
```

通过 `framework_adapter` 模块，ModelPerf 可以零侵入地集成到训练流程中：

```python
import sys
sys.path.insert(0, '/mnt/d/ubuntu/opencode/training_framework/Pai-Megatron-Patch-12.0')
sys.path.insert(0, '/mnt/d/ubuntu/opencode/training_framework/Pai-Megatron-Patch-12.0/backends/megatron/Megatron-LM-20250707')
sys.path.insert(0, '/mnt/d/ubuntu/opencode/training_framework/ModelPerf')

from megatron.training.arguments import parse_args
from megatron_patch.arguments import get_patch_args
from modelperf.framework_adapter.config_extractor import ConfigExtractor
from modelperf.capture.coordinator import CaptureCoordinator

# 1. 解析参数
args = parse_args(extra_args_provider=get_patch_args)

# 2. 显式提取配置
extractor = ConfigExtractor()
extractor.extract_from_args(args)
print(f"Model: {len(extractor.model_config)} fields")
print(f"Strategy: {len(extractor.strategy_config)} fields")

# 3. 创建模型
from megatron.training.arguments import core_transformer_config_from_args
config = core_transformer_config_from_args(args)

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
spec = get_gpt_layer_local_spec(
    args.num_experts, args.moe_grouped_gemm,
    args.qk_layernorm, args.multi_latent_attention,
    normalization=args.normalization,
)

from megatron.core.models.gpt import GPTModel
model = GPTModel(
    config=config,
    transformer_layer_spec=spec,
    vocab_size=args.padded_vocab_size,
    max_sequence_length=args.max_position_embeddings,
    pre_process=True,
    post_process=True,
    parallel_output=True,
    position_embedding_type=args.position_embedding_type,
)

# 4. 附加 CaptureCoordinator
coordinator = CaptureCoordinator()
coordinator.attach(model)

# 5. 运行前向+反向并捕获
with coordinator:
    x = torch.randint(0, args.padded_vocab_size, (1, args.seq_length))
    y = model(x)
    loss = y.sum()
    loss.backward()

graph = coordinator.get_graph()
graph.model_config = extractor.model_config
graph.strategy_config = extractor.strategy_config

print(f"Captured {len(graph.nodes)} nodes")
```

### 5.3 端到端流水线（捕获后）

捕获完成后，可直接衔接仿真与报告生成：

```python
from modelperf.simulation import VirtualExecutor, ExecutionConfig
from modelperf.analysis.what_if import WhatIfAnalyzer
from modelperf.visualization.report_generator import generate_performance_report

exec_config = ExecutionConfig(
    peak_compute_tflops=312.0,
    peak_bandwidth_gbs=2039.0,
    network_bandwidth_gbs=600.0,
    memory_per_device_gb=80.0,
    enable_overlap=True,
)
executor = VirtualExecutor(config=exec_config)
result = executor.execute(graph)

analyzer = WhatIfAnalyzer(graph)
new_config = analyzer.modify_config({"micro_batch_size": 2})
new_result = analyzer.re_evaluate(new_config)

report_path = generate_performance_report(
    result,
    "/mnt/d/ubuntu/opencode/training_framework/ModelPerf/examples/output/reports"
)
print(f"Report: {report_path}")
```

### 5.4 手动集成其他模型

对于 Pai-Patch 支持的其他模型（LLaMA3、DeepSeek-V3 等），集成方式与 Qwen3 相同：

1. 导入对应模型的 `pretrain_*.py`
2. 使用 `ConfigExtractor` 提取参数
3. 创建 `CaptureCoordinator` 并 `attach(model)`
4. 在 `with coordinator:` 上下文中运行一次前向+反向

### 5.5 train_qwen3_0.6B.sh 关键配置参数

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `TP` | Tensor Parallelism（张量并行度） | 1 |
| `PP` | Pipeline Parallelism（流水并行度） | 1 |
| `CP` | Context Parallelism（上下文并行度） | 1 |
| `BATCH_SIZE` | Micro Batch Size（单卡微批次） | 1 |
| `GLOBAL_BATCH_SIZE` | Global Batch Size（全局批次） | 1 |
| `SEQ_LEN` | 序列长度 | 128 |
| `AC` | Activation Checkpointing（激活检查点） | sel |
| `DO` | Distributed Optimizer（分布式优化器） | false |
| `SP` | Sequence Parallelism（序列并行） | true |

这些参数会被 `ConfigExtractor` 自动提取到 `strategy_config` 中，供后续仿真使用。

### 5.6 多进程计算图捕获与合并（DP/TP/PP）

当使用混合并行（如 DP=2, TP=2, PP=2）时，每个 rank 捕获的计算图不同。ModelPerf 通过 `parallel_identity` 机制管理多 rank 图：

**保存阶段**：每个 rank 独立保存，文件名包含并行身份

```python
# 在 pretrain_qwen.py 中自动完成
graph_obj.parallel_identity = {
    'world_rank': rank,
    'dp_rank': dp_rank,
    'tp_rank': tp_rank,
    'pp_rank': pp_rank,
    'dp_size': dp_size,
    'tp_size': tp_size,
    'pp_size': pp_size,
}

# 保存为：computational_graph_pp{pp}_tp{tp}_dp{dp}.json
```

**输出目录结构**（以 DP=2, TP=2, PP=2 为例）：
```
captured_graph/
├── computational_graph_pp0_tp0_dp0.json   # Stage 0, TP rank 0
├── computational_graph_pp0_tp1_dp0.json   # Stage 0, TP rank 1
├── computational_graph_pp1_tp0_dp0.json   # Stage 1, TP rank 0
└── computational_graph_pp1_tp1_dp0.json   # Stage 1, TP rank 1
```

**三种并行维度的图特征**：

| 维度 | 各 rank 图关系 | 合并策略 |
|------|---------------|---------|
| **DP** | 基本相同（数据输入不同） | 去重：保留一个副本 |
| **TP** | 互补（权重切片不同） | 保留所有 unique 节点 |
| **PP** | 层间互补（不同 stage） | 按层号拼接 |

### 5.7 使用 modelperf_eval.py 进行仿真

`modelperf_eval.py` 是端到端的性能评估脚本，支持自动加载并合并多 rank 计算图：

```bash
cd /mnt/d/ubuntu/opencode/training_framework/ModelPerf
python examples/modelperf_eval.py \
    --config-dir examples/output/captured_configs \
    --graph-dir examples/output/captured_graph \
    --output-dir examples/output/reports \
    --label "qwen3_dp2_tp2_pp2"
```

**三级合并流水线**：

```
输入: N 个 rank 的图文件
  ↓
[Step 1/3] 去重 DP replicas
  按 (pp_rank, tp_rank) 分组，每组内 node_id 去重
  输出: M 个 unique (PP, TP) 组合图
  ↓
[Step 2/3] 合并 TP ranks
  按 pp_rank 分组，组内保留所有 unique 节点（TP 切分互补）
  输出: P 个 PP stage 图（P = pp_size）
  ↓
[Step 3/3] 拼接 PP stages
  按 pp_rank 排序，合并为全局逻辑图
  输出: 1 个最终图
```

**回退兼容**：支持旧格式 `computational_graph_rank*.json` 和 `computational_graph.json`。

### 5.8 混合并行仿真实例

以下是在 Qwen3 0.6B 上运行 DP=2, TP=2, PP=2 并仿真的完整流程：

```bash
# 1. 运行训练（自动捕获计算图）
cd /mnt/d/ubuntu/opencode/training_framework/Pai-Megatron-Patch-12.0/examples/qwen3
bash train_qwen3_0.6B_dp2.sh   # DP=2
bash train_qwen3_0.6B_tp2.sh   # TP=2
bash train_qwen3_0.6B_pp2.sh   # PP=2

# 2. 仿真评估
cd /mnt/d/ubuntu/opencode/training_framework/ModelPerf
python examples/modelperf_eval.py \
    --config-dir examples/output/captured_configs \
    --graph-dir examples/output/captured_graph \
    --output-dir examples/output/reports_dp2 \
    --label "DP2"

python examples/modelperf_eval.py \
    --config-dir examples/output/captured_configs \
    --graph-dir examples/output/captured_graph \
    --output-dir examples/output/reports_tp2 \
    --label "TP2"

python examples/modelperf_eval.py \
    --config-dir examples/output/captured_configs \
    --graph-dir examples/output/captured_graph \
    --output-dir examples/output/reports_pp2 \
    --label "PP2"
```

**仿真结果示例**：

| 策略 | Iteration Time | Compute Time | Comm Time | Peak Memory | Bottleneck |
|------|---------------|--------------|-----------|-------------|------------|
| **DP2** | 0.43 ms | 0.43 ms | 0.00 ms | 42.34 MB | compute |
| **TP2** | 0.43 ms | 0.43 ms | 0.00 ms | 42.34 MB | compute |
| **PP2** | 0.43 ms | 0.43 ms | 0.00 ms | 42.34 MB | compute |

> **注意**：当前 Roofline 模型对小模型（0.09B）的评估结果趋于一致。在真实大模型（>1B）上，不同并行策略的差异会更显著。

---

## 6. 在 Megatron-LM 中使用

### 6.1 手动集成方法

对于独立使用 `Megatron-LM-20250707`（不经过 Pai-Patch）的场景：

```python
import sys
sys.path.insert(0, '/mnt/d/ubuntu/opencode/training_framework/Megatron-LM-20250707')
sys.path.insert(0, '/mnt/d/ubuntu/opencode/training_framework/ModelPerf')

from modelperf.framework_adapter.megatron_hooks import register_megatron_hooks
from modelperf.framework_adapter.config_extractor import ConfigExtractor
from modelperf.capture.coordinator import CaptureCoordinator

# 1. 注册 Megatron Hook
hook_manager = register_megatron_hooks(
    export_dir="/mnt/d/ubuntu/opencode/training_framework/ModelPerf/examples/output/captured_configs"
)

# 2. 正常调用 Megatron 训练入口
from megatron.training.arguments import parse_args
args = parse_args()

# 3. 提取配置
extractor = ConfigExtractor()
extractor.extract_from_args(args)

# 4. 创建模型后附加 CaptureCoordinator
coordinator = CaptureCoordinator()
coordinator.attach(model)

with coordinator:
    output = model(input_batch)
    loss = compute_loss(output)
    loss.backward()

# 5. 清理 Hook
from modelperf.framework_adapter.megatron_hooks import unregister_megatron_hooks
unregister_megatron_hooks(hook_manager)
```

### 6.2 使用 ConfigExtractor 从 args 提取配置

`ConfigExtractor` 将 Megatron 的 `args` 对象解析为三个结构化字典：

```python
from modelperf.framework_adapter.config_extractor import ConfigExtractor

extractor = ConfigExtractor()
extractor.extract_from_args(args)

print(extractor.model_config)      # 模型结构参数
print(extractor.strategy_config)   # 并行策略参数
print(extractor.system_config)     # 系统/训练参数

extractor.export_json("/mnt/d/ubuntu/opencode/training_framework/ModelPerf/examples/output/captured_configs")
```

**model_config 包含**：`num_layers`、`hidden_size`、`ffn_hidden_size`、`num_attention_heads`、`vocab_size`、`params_dtype` 等。

**strategy_config 包含**：`tensor_parallel_size`、`pipeline_parallel_size`、`data_parallel_size`、`micro_batch_size`、`global_batch_size`、`sequence_parallel` 等。

**system_config 包含**：`train_iters`、`lr`、`world_size`、`distributed_backend` 等。

### 6.3 在 model_provider 中添加 CaptureCoordinator

如果你自定义了 Megatron 的 `model_provider` 函数，可在返回模型前附加 Coordinator：

```python
def model_provider(pre_process=True, post_process=True):
    from megatron.core.models.gpt import GPTModel
    from modelperf.capture.coordinator import CaptureCoordinator

    model = GPTModel(...)

    if hasattr(model, '_modelperf_coordinator'):
        return model

    coordinator = CaptureCoordinator()
    coordinator.attach(model)
    model._modelperf_coordinator = coordinator
    coordinator.start()

    return model
```

建议在训练循环的第一次迭代后调用 `coordinator.stop()` 并导出计算图。
---

## 7. 性能仿真与 What-if 分析

### 7.1 使用 VirtualExecutor

`VirtualExecutor` 是性能仿真的核心引擎：

```python
from modelperf.simulation import VirtualExecutor, ExecutionConfig

config = ExecutionConfig(
    gpu_type="a100",                  # GPU 类型（仅影响默认参数）
    peak_compute_tflops=312.0,        # 峰值算力（FP16 TFLOPS）
    peak_bandwidth_gbs=2039.0,        # HBM 带宽（GB/s）
    network_bandwidth_gbs=600.0,      # 网络带宽（GB/s）
    network_latency_us=2.0,           # 网络延迟（us）
    memory_per_device_gb=80.0,        # 单卡显存（GB）
    enable_overlap=True,              # 是否启用 compute/comm 重叠
    pipeline_schedule="1f1b",         # 流水线调度策略
)

executor = VirtualExecutor(config=config)
result = executor.execute(graph)

print(f"Iteration time: {result.iteration_time_ms:.2f} ms")
print(f"Forward: {result.forward_time_ms:.2f} ms")
print(f"Backward: {result.backward_time_ms:.2f} ms")
print(f"Optimizer: {result.optimizer_time_ms:.2f} ms")
print(f"Compute: {result.compute_time_ms:.2f} ms")
print(f"Comm: {result.comm_time_ms:.2f} ms")
print(f"Bubble: {result.bubble_time_ms:.2f} ms")
print(f"Peak memory: {result.peak_memory_mb:.2f} MB")
print(f"Throughput: {result.throughput_tokens_per_sec:.2f} tokens/sec")
print(f"Bottleneck: {result.bottleneck}")
```

### 7.2 ExecutionConfig 选项详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `gpu_type` | str | "a100" | GPU 类型标识 |
| `peak_compute_tflops` | float | 312.0 | 峰值计算吞吐（A100 FP16） |
| `peak_bandwidth_gbs` | float | 2039.0 | 显存带宽 |
| `network_bandwidth_gbs` | float | 600.0 | 网络带宽（NVLink） |
| `network_latency_us` | float | 2.0 | 通信延迟 |
| `memory_per_device_gb` | float | 80.0 | 单卡显存上限 |
| `enable_overlap` | bool | True | 计算与通信是否重叠 |
| `pipeline_schedule` | str | "1f1b" | 流水线调度：1f1b / gpipe / interleaved |
| `num_microbatches` | int | 1 | 微批次数量 |
| `enable_activation_checkpointing` | bool | False | 是否启用激活检查点 |

### 7.3 使用 WhatIfAnalyzer 修改配置并重新评估

```python
from modelperf.analysis.what_if import WhatIfAnalyzer, WhatIfConfig

analyzer = WhatIfAnalyzer(graph)

# 方法 1：基于当前配置修改特定字段
new_config = analyzer.modify_config({
    "tensor_parallel_size": 2,
    "micro_batch_size": 4,
})
result = analyzer.re_evaluate(new_config)

# 方法 2：从零创建 WhatIfConfig
config = WhatIfConfig(
    tensor_parallel_size=2,
    pipeline_parallel_size=2,
    data_parallel_size=2,
    hidden_size=1024,
    num_layers=8,
    micro_batch_size=2,
    gpu_type="A100",
)
result = analyzer.re_evaluate(config)
```

### 7.4 使用 grid_search 批量评估多种配置

```python
param_grid = {
    "tensor_parallel_size": [1, 2, 4],
    "pipeline_parallel_size": [1, 2],
    "micro_batch_size": [1, 2, 4],
}

results = analyzer.grid_search(
    param_grid,
    sort_by="iteration_time_ms",   # 按迭代时间排序
    ascending=True,                # 升序（时间越短越好）
    n_workers=4,                   # 4 线程并行
)

for changes, result in results[:5]:
    print(f"Config: {changes}")
    print(f"  Time: {result.iteration_time_ms:.2f} ms")
    print(f"  Memory: {result.peak_memory_mb:.2f} MB")
```

### 7.5 OOM 预测

```python
will_oom, memory_gb, utilization = analyzer.predict_oom(
    config=new_config,
    gpu_memory_gb=80.0,
)
print(f"OOM: {will_oom}, Memory: {memory_gb:.2f} GB, Util: {utilization*100:.1f}%")
```
---

## 8. Auto-tuning 自动策略推荐

### 8.1 使用 recommend() 便捷函数

```python
from modelperf.tuning.auto_tuner import recommend, ParallelStrategyPreset

report = recommend(
    graph,
    gpu_memory_gb=80.0,
    max_iteration_time_ms=1000.0,
    min_throughput=1000.0,
    preset=ParallelStrategyPreset.BALANCED,
)

print(f"Recommended config: {report.recommended_config}")
print(f"Expected iteration time: {report.expected_iteration_time_ms:.2f} ms")
print(f"Expected peak memory: {report.expected_peak_memory_mb:.2f} MB")
print(f"Expected throughput: {report.expected_throughput:.2f} tokens/sec")
```

### 8.2 ParallelStrategyPreset 说明

| 预设 | 目标 | 排序依据 |
|------|------|---------|
| `THROUGHPUT` | 最大化吞吐量 | `throughput_tokens_per_sec` 降序 |
| `MEMORY_EFFICIENT` | 最小化显存占用 | `peak_memory_mb` 升序 |
| `BALANCED` | 吞吐与内存的平衡 | `throughput / log(memory)` 降序 |

### 8.3 理解 TuningReport

```python
from modelperf.tuning.auto_tuner import TuningReport

# recommended_config: 字典，包含推荐配置（如 tp_size, pp_size, dp_size, micro_batch_size）
print(report.recommended_config)

# expected_iteration_time_ms: 推荐配置的预测迭代时间
print(report.expected_iteration_time_ms)

# expected_peak_memory_mb: 推荐配置的预测峰值显存
print(report.expected_peak_memory_mb)

# expected_throughput: 推荐配置的预测吞吐量（tokens/sec）
print(report.expected_throughput)

# all_results: 所有通过约束过滤的 (config_dict, ExecutionResult) 列表，按预设排序
for cfg, result in report.all_results[:10]:
    print(cfg, result.iteration_time_ms, result.peak_memory_mb)

# pareto_frontier: 帕累托前沿（throughput vs memory 的非支配解集）
for cfg, result in report.pareto_frontier:
    print(f"Pareto: {cfg}, throughput={result.throughput_tokens_per_sec}, memory={result.peak_memory_mb}")
```

### 8.4 搜索空间生成与约束过滤

`AutoTuner` 会自动生成合法的搜索空间：

```python
from modelperf.tuning.auto_tuner import AutoTuner

tuner = AutoTuner(graph)

# 自动生成搜索空间（基于 world_size 和模型结构）
space = tuner.generate_search_space(world_size=8)
print(space)
# 输出示例：
# {
#     "tensor_parallel_size": [1, 2, 4, 8],
#     "pipeline_parallel_size": [1, 2, 4],
#     "data_parallel_size": [1, 2, 4, 8],
#     "micro_batch_size": [1, 2, 4, 8],
# }
```

自动过滤规则：
- `TP` 必须整除 `hidden_size` 和 `num_attention_heads`
- `PP` 必须整除 `num_layers`
- `TP * PP * DP` 必须等于 `world_size`

自定义搜索空间：

```python
report = tuner.search(
    gpu_memory_gb=80.0,
    max_iteration_time_ms=500.0,   # 过滤掉迭代时间超过 500ms 的配置
    min_throughput=5000.0,          # 过滤掉吞吐量低于 5000 tok/s 的配置
    preset=ParallelStrategyPreset.THROUGHPUT,
    param_grid={                    # 自定义搜索空间
        "tensor_parallel_size": [1, 2, 4],
        "pipeline_parallel_size": [1, 2],
        "data_parallel_size": [1, 2, 4],
        "micro_batch_size": [1, 2, 4],
    },
    n_workers=4,
)
```
---

## 9. 可视化报告

### 9.1 使用 generate_performance_report()

```python
from modelperf.visualization.report_generator import generate_performance_report

report_path = generate_performance_report(
    result,                                              # ExecutionResult
    "/mnt/d/ubuntu/opencode/training_framework/ModelPerf/examples/output/reports"
)
print(f"Report generated: {report_path}")
```

### 9.2 输出文件说明

运行后会生成以下文件到指定输出目录：

| 文件名 | 类型 | 内容 |
|--------|------|------|
| `report.html` | HTML | 汇总报告，包含所有指标表格和图表嵌入 |
| `iteration_time_breakdown.png` / `.html` | 图片/HTML | 迭代时间堆叠条形图（Forward / Backward / Optimizer / Comm / Bubble） |
| `memory_breakdown.png` / `.html` | 图片/HTML | 内存饼图（Activation / Parameter / Gradient / Buffer / Temp） |
| `compute_comm_breakdown.png` / `.html` | 图片/HTML | Compute vs Communication 对比条形图 |

### 9.3 matplotlib 与 HTML 回退

`report_generator` 会自动检测 `matplotlib` 是否可用：

- **已安装 matplotlib**：生成 `.png` 图片，嵌入 `report.html`
- **未安装 matplotlib**：生成纯 HTML/SVG 图表，无需外部依赖

安装 matplotlib 以获得更高质量的矢量图表：

```bash
pip install matplotlib
```

### 9.4 报告示例

生成的 `report.html` 包含以下内容：

- **Summary Statistics**：迭代时间、前向/反向/优化器时间、计算时间、通信时间、Bubble 时间、峰值内存、瓶颈类型、吞吐量、内存效率、流水线阶段数、微批次数
- **Iteration Time Breakdown**：可视化各阶段耗时占比
- **Memory Breakdown**：可视化各类内存占用占比
- **Compute vs Communication Breakdown**：各节点的计算时间与通信时间对比
---

## 10. API 速查

### 10.1 计算图捕获

| 类/函数 | 导入路径 | 说明 |
|---------|---------|------|
| `ComputationalGraph` | `modelperf.capture.ComputationalGraph` | 计算图数据结构 |
| `GraphNode` | `modelperf.capture.GraphNode` | 计算图节点 |
| `OpType` | `modelperf.capture.OpType` | 节点类型枚举 |
| `CommType` | `modelperf.capture.CommType` | 通信类型枚举 |
| `CaptureCoordinator` | `modelperf.capture.CaptureCoordinator` | 五层 Hook 协调器 |
| `ModuleCapture` | `modelperf.capture.ModuleCapture` | Layer 1: Module Hook |
| `CommunicationCapture` | `modelperf.capture.CommunicationCapture` | Layer 2: Comm Hook |
| `AutogradCapture` | `modelperf.capture.autograd_hook.AutogradCapture` | Layer 3: Autograd Hook |
| `BackwardCapture` | `modelperf.capture.backward_hook.BackwardCapture` | Layer 4: Backward Hook |
| `ProfilerCapture` | `modelperf.capture.profiler_hook.ProfilerCapture` | Layer 5: Profiler Hook |

### 10.2 符号化参数

| 类/函数 | 导入路径 | 说明 |
|---------|---------|------|
| `SymbolicShapeInferer` | `modelperf.symbolic.shape_inferer.SymbolicShapeInferer` | 符号化 Shape 推断器 |

### 10.3 虚拟执行

| 类/函数 | 导入路径 | 说明 |
|---------|---------|------|
| `VirtualExecutor` | `modelperf.simulation.VirtualExecutor` | 虚拟执行引擎 |
| `ExecutionConfig` | `modelperf.simulation.ExecutionConfig` | 执行配置 |
| `ExecutionResult` | `modelperf.simulation.ExecutionResult` | 执行结果 |
| `RooflineModel` | `modelperf.simulation.roofline.RooflineModel` | Roofline 计算模型 |
| `BandwidthModel` | `modelperf.simulation.bandwidth.BandwidthModel` | 带宽通信模型 |
| `MemoryTracker` | `modelperf.simulation.memory_tracker.MemoryTracker` | 内存跟踪器 |

### 10.4 What-if 分析

| 类/函数 | 导入路径 | 说明 |
|---------|---------|------|
| `WhatIfAnalyzer` | `modelperf.analysis.what_if.WhatIfAnalyzer` | What-if 分析器 |
| `WhatIfConfig` | `modelperf.analysis.what_if.WhatIfConfig` | What-if 配置 |

### 10.5 Auto-tuning

| 类/函数 | 导入路径 | 说明 |
|---------|---------|------|
| `recommend` | `modelperf.tuning.auto_tuner.recommend` | 便捷推荐函数 |
| `AutoTuner` | `modelperf.tuning.auto_tuner.AutoTuner` | 自动调优器 |
| `TuningReport` | `modelperf.tuning.auto_tuner.TuningReport` | 调优报告 |
| `ParallelStrategyPreset` | `modelperf.tuning.auto_tuner.ParallelStrategyPreset` | 优化目标预设 |

### 10.6 可视化

| 类/函数 | 导入路径 | 说明 |
|---------|---------|------|
| `generate_performance_report` | `modelperf.visualization.report_generator.generate_performance_report` | 生成性能报告 |

### 10.7 框架适配

| 类/函数 | 导入路径 | 说明 |
|---------|---------|------|
| `ConfigExtractor` | `modelperf.framework_adapter.config_extractor.ConfigExtractor` | 配置提取器 |
| `register_megatron_hooks` | `modelperf.framework_adapter.megatron_hooks.register_megatron_hooks` | 注册 Megatron Hook |
| `unregister_megatron_hooks` | `modelperf.framework_adapter.megatron_hooks.unregister_megatron_hooks` | 注销 Megatron Hook |
| `register_pai_patch_hooks` | `modelperf.framework_adapter.pai_patch_hooks.register_pai_patch_hooks` | 注册 Pai-Patch Hook |

### 10.8 配置数据类

| 类 | 导入路径 | 说明 |
|----|---------|------|
| `ModelConfig` | `modelperf.config.ModelConfig` | 模型配置数据类 |
| `StrategyConfig` | `modelperf.config.StrategyConfig` | 策略配置数据类 |
| `SystemConfig` | `modelperf.config.SystemConfig` | 系统配置数据类 |

---

## 11. 常见问题

### Q1: ModelPerf 是否必须配合 Megatron 使用？

**答**：不需要。ModelPerf 可以独立运行。你可以用 `CaptureCoordinator` 捕获任意 PyTorch 模型的计算图，然后使用 `VirtualExecutor` 进行性能仿真。Megatron 和 Pai-Patch 的集成是可选的，主要用于自动提取分布式训练配置。

### Q2: 为什么所有测试都通过，但真实训练捕获的图节点数为 0？

**答**：最常见的原因是 `CaptureCoordinator` 的 `start()` 在 forward 调用之后才执行。确保 `with coordinator:` 上下文管理器包裹了模型的前向和反向调用：

```python
coordinator = CaptureCoordinator()
coordinator.attach(model)
with coordinator:  # start() 在这里调用
    y = model(x)
    loss = y.sum()
    loss.backward()
# stop() 在这里自动调用
```

### Q3: 仿真结果的迭代时间为什么和真实训练差异很大？

**答**：ModelPerf 的仿真是基于理论硬件参数（Roofline 模型和带宽模型）进行的理想化估算，未考虑以下因素：
- GPU kernel launch overhead
- 实际的 CUDA stream 调度效率
- 框架层的额外开销（如 Python GIL）
- 内存带宽争用

建议将 ModelPerf 用于**相对比较**（哪种配置更好），而非绝对时间预测。通过 PyTorch Profiler 采集的真实 trace 可用于校准 `efficiency_factor`。

### Q4: 如何在 Auto-tuning 中指定更大的搜索空间？

**答**：使用 `AutoTuner.generate_search_space()` 时传入自定义参数：

```python
tuner = AutoTuner(graph)
search_space = tuner.generate_search_space(
    world_size=8,
    tp_values=[1, 2, 4],
    pp_values=[1, 2],
    bs_values=[1, 2, 4, 8, 16]
)
report = tuner.search(
    gpu_memory_gb=80.0,
    param_grid=search_space
)
```

注意：搜索空间大小与评估时间成正比。TP=4 × PP=2 × DP=[1,2,4] × BS=5 = 120 种组合，在 8 核 CPU 上约需 10-30 秒。

### Q5: 可视化报告中的图表显示为空白？

**答**：如果运行环境没有安装 matplotlib，ModelPerf 会自动回退到纯 HTML/SVG 模式。检查 `report.html` 同级目录下是否有 `.html` 结尾的图表文件（如 `iteration_time_breakdown.html`）。如果安装了 matplotlib 但仍然空白，尝试设置后端：

```python
import matplotlib
matplotlib.use('Agg')  # 在写入文件前设置
```

### Q6: 捕获的计算图文件很大（几 MB），如何缩小？

**答**：计算图大小与模型层数正相关。对于 4-layer 的 Qwen3 0.6B，计算图约 2.5 MB（3161 节点）。这是正常的。如果只需要仿真而无需查看原始图，可以在导出时省略详细节点信息：

```python
# 只导出 summary，不导出完整 nodes
summary = graph.summary()
with open("graph_summary.json", "w") as f:
    json.dump(summary, f)
```

### Q7: 如何在 LLaMA3 或 DeepSeek-V3 训练中使用 ModelPerf？

**答**：参考 [5.4 手动集成到其他模型](#54-手动集成到其他模型)。核心步骤是在 `model_provider()` 函数中添加 `CaptureCoordinator.attach(model)`。不同模型的训练入口文件位置不同：
- LLaMA3: `Pai-Megatron-Patch-12.0/examples/llama3/pretrain_llama3.py`
- DeepSeek-V3: `Pai-Megatron-Patch-12.0/examples/deepseek_v3/pretrain_deepseek_v3.py`

如果模型训练脚本已内置 ModelPerf Hook（如 Qwen3），则无需手动修改。

### Q8: 单元测试中有 1 个失败（test_real_training_integration），是否正常？

**答**：`test_real_training_integration.py` 是一个可选的集成测试，用于在真实 Megatron 环境中验证 ModelPerf。该测试需要完整的 Megatron + Pai-Patch 环境才能运行。如果运行环境缺少这些依赖，测试会失败，但这不影响 ModelPerf 的核心功能。运行测试时可以使用：

```bash
python -m unittest discover -s tests -v  # 运行所有测试
# 或
python -m unittest discover -s tests -p "test_*.py" -v  # 排除集成测试
```

---

> **文档版本**：v1.1  
> **最后更新**：2026-05-07  
> **项目地址**：`ModelPerf/`  
> **相关问题请查阅**：`docs/development_log.md`（开发历程记录）
