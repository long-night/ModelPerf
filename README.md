# ModelPerf

基于 Megatron-LM 的大模型分布式训练性能仿真框架

## 项目简介

> **环境声明：本项目的大模型训练与性能仿真完全基于 CPU 环境运行，与 CUDA、GPU 无关。**
>
> 所有训练流程、计算图捕获、性能仿真和 What-if 分析均在纯 CPU 环境下完成，无需 NVIDIA GPU 或 CUDA  Toolkit。项目中涉及的硬件参数（如 A100/H100 的算力、带宽）仅作为仿真输入的理论值，用于估算性能，不依赖真实 GPU 硬件。

ModelPerf 是一个用于大模型（LLM）分布式训练性能仿真与分析的开源框架。它通过捕获真实训练过程中的计算图（含通信算子），使用符号化公式参数化所有张量形状和通信量，然后通过虚拟执行引擎估算不同配置下的训练性能，无需在真实 GPU 上重新运行训练。

**核心价值**：
- 一次捕获，多次仿真 —— 捕获一次计算图后，可快速评估数百种并行策略配置
- What-if 分析 —— 修改 TP/PP/DP/BS 等参数后，秒级获得新的性能估算
- 内存峰值预测 —— 提前预判配置是否会导致 OOM

## 技术架构

### Capture-Parameterize-Execute 三层架构

```
┌─────────────────────────────────────────────────────────┐
│                    Capture (捕获层)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ Module   │  │ Comm     │  │ Autograd │  │ Backward │ │
│  │ Hook     │  │ Hook     │  │ Hook     │  │ Hook     │ │
│  │ (Layer1) │  │ (Layer2) │  │ (Layer3) │  │ (Layer4) │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │
│                    ┌──────────┐                          │
│                    │ Profiler │ (Layer5)                 │
│                    │ Hook     │                          │
│                    └──────────┘                          │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│                 Parameterize (参数化层)                   │
│              SymbolicShapeInferer (sympy)                │
│         B, S, H, TP, PP, DP, ... → 符号公式             │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   Execute (执行层)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │ Roofline    │  │ Bandwidth   │  │ MemoryTracker   │ │
│  │ Model       │  │ Model       │  │                 │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
│                                                         │
│              VirtualExecutor → ExecutionResult          │
└─────────────────────────────────────────────────────────┘
```

## 项目结构

```
ModelPerf/
├── modelperf/                    # 核心框架代码
│   ├── capture/                  # 计算图捕获（5层Hook架构）
│   │   ├── graph.py              # 计算图数据结构
│   │   ├── module_hook.py        # Layer 1: Module Hook
│   │   ├── comm_hook.py          # Layer 2: Communication Monkey-patch
│   │   ├── autograd_hook.py      # Layer 3: Autograd Function Hook
│   │   ├── backward_hook.py      # Layer 4: Backward Hook
│   │   └── profiler_hook.py      # Layer 5: PyTorch Profiler Hook
│   ├── symbolic/                 # 符号化参数推断
│   │   └── shape_inferer.py      # SymbolicShapeInferer (sympy)
│   ├── simulation/               # 虚拟执行引擎
│   │   ├── roofline.py           # Roofline 计算模型
│   │   ├── bandwidth.py          # 带宽通信模型
│   │   ├── memory_tracker.py     # 内存峰值跟踪
│   │   ├── virtual_executor.py   # 虚拟执行引擎
│   │   └── execution_result.py   # 执行结果结构
│   ├── analysis/                 # What-if 分析
│   │   ├── what_if.py            # WhatIfAnalyzer
│   │   └── config_variants.py    # 配置变体生成
│   └── utils/                    # 工具函数
│       ├── validator.py          # 验证框架
│       └── metrics.py            # 误差指标计算
├── tests/                        # 单元测试（105 tests）
├── examples/                     # 使用示例
│   └── basic_usage.py            # Qwen3 0.6B 端到端示例
├── docs/                         # 设计文档
│   └── 基于Megatron-LM和Pai-Patch的大模型训练性能仿真方案设计报告.md
└── backends/                     # 参考的开源项目
    ├── SimAI/                    # 阿里巴巴 SimAI
    ├── SimuMax/                  # SimuMax 静态分析器
    ├── DistSim/                  # DistSim 事件驱动仿真
    └── lumos/                    # Lumos (参考项目)
```

## 核心功能

### 1. 计算图捕获（5层 Hook 架构）

| 层级 | 目标 | 技术方案 |
|------|------|---------|
| Layer 1 | nn.Module 前向调用 | `register_forward_hook` |
| Layer 2 | 分布式通信算子 | Monkey-patch `torch.distributed` |
| Layer 3 | 自定义 Autograd Function | Patch `forward`/`backward` |
| Layer 4 | 反向传播结构 | `Tensor.register_hook` |
| Layer 5 | 运行时校准 | PyTorch Profiler (Kineto) |

### 2. 符号化参数推断

使用 sympy 将张量形状和通信量转换为符号公式：

```python
from modelperf.symbolic import SymbolicShapeInferer

inferer = SymbolicShapeInferer()
# ColumnParallelLinear: [S, B, H] -> [S, B, F/TP]
output, input, weight = inferer.column_parallel_linear_shape(
    input_shape=(S, B, H), in_features=H, out_features=F
)
# 评估: 代入具体值
shape = inferer.evaluate(output, {"S": 4096, "B": 4, "H": 576, "F": 1536, "TP": 2})
# -> (4096, 4, 768.0)
```

### 3. 虚拟执行引擎

```python
from modelperf.simulation import VirtualExecutor, ExecutionConfig

config = ExecutionConfig(
    peak_compute_tflops=312.0,    # A100 FP16
    peak_bandwidth_gbs=2039.0,     # A100 HBM
    network_bandwidth_gbs=600.0,   # NVLink
)
executor = VirtualExecutor(config)
result = executor.execute(graph)

print(f"Iteration time: {result.iteration_time_ms:.2f} ms")
print(f"Peak memory: {result.peak_memory_mb:.2f} MB")
print(f"Bottleneck: {result.bottleneck}")
```

### 4. What-if 分析

```python
from modelperf.analysis import WhatIfAnalyzer

analyzer = WhatIfAnalyzer(graph)

# 修改 TP=2，自动重评估
new_config = WhatIfConfig(tensor_parallel_size=2)
result = analyzer.re_evaluate(new_config)

# 网格搜索最优配置
param_grid = {
    "tensor_parallel_size": [1, 2, 4],
    "pipeline_parallel_size": [1, 2],
    "micro_batch_size": [1, 2, 4],
}
results = analyzer.grid_search(param_grid)
# 返回按迭代时间排序的结果列表
```

## 快速开始

### 安装依赖

```bash
pip install sympy
# PyTorch CPU 版本（无需 CUDA）
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 运行示例

```bash
cd ModelPerf
PYTHONPATH=".:$PYTHONPATH" python examples/basic_usage.py
```

### 运行单元测试

```bash
cd ModelPerf
PYTHONPATH=".:$PYTHONPATH" python -m unittest discover -s tests -v
```

## 当前进展

### 已实现 ✅

1. **计算图数据结构** (`capture/graph.py`)
   - GraphNode 支持 compute/communication/autograd/backward/optimizer 类型
   - ComputationalGraph 支持节点管理、边依赖、序列化

2. **5层 Hook 捕获框架**
   - Module Hook: 捕获所有 nn.Module 前向调用
   - Communication Hook: Monkey-patch 6 种 `torch.distributed` 通信原语
   - Autograd Hook: 捕获自定义 Autograd Function 的 forward/backward
   - Backward Hook: 注册反向钩子捕获梯度流
   - Profiler Hook: PyTorch Profiler 运行时校准

3. **符号化 Shape 推断器** (`symbolic/shape_inferer.py`)
   - 支持 B, S, H, V, F, TP, PP, DP, CP, EP 等符号
   - 覆盖 ColumnParallelLinear, RowParallelLinear, SelfAttention, MLP, Embedding
   - 通信量公式: all_reduce, all_gather, reduce_scatter, all_to_all

4. **仿真模型**
   - Roofline Model: 计算时间估算（支持 A100/H100/V100）
   - Bandwidth Model: 通信时间估算（支持 ring/tree/direct 算法）
   - Memory Tracker: 多设备内存峰值跟踪（支持 activation checkpointing）

5. **虚拟执行引擎** (`simulation/virtual_executor.py`)
   - 自动识别 compute/comm 节点并估算时间
   - FLOPs 估算: Linear, Attention (QKV/scores/output), MLP
   - Pipeline Parallelism 气泡时间模拟
   - 瓶颈识别: compute/communication/bubble/memory

6. **What-if 分析器** (`analysis/what_if.py`)
   - 配置修改后自动符号替换和重评估
   - 网格搜索最优配置（多线程并行）
   - OOM 预测

7. **验证框架** (`utils/validator.py`)
   - MAPE/RMSE/MAE/R² 误差计算
   - 自动生成验证报告

8. **单元测试** (105 tests, 全部通过)
   - `test_graph.py`: 计算图数据结构
   - `test_roofline.py`: Roofline 模型
   - `test_bandwidth.py`: 带宽模型
   - `test_memory_tracker.py`: 内存跟踪
   - `test_symbolic.py`: 符号化推断

9. **设计文档** (1232 行)
   - 完整的技术方案设计报告，含附录 A.1-A.6

### 验证结果

使用 Qwen3 0.6B 合成数据端到端测试：
- 迭代时间: 2.07 ms（仿真）vs 2.24 ms（基准）, MAPE 7.41%
- 内存峰值: 2409.50 MB（仿真）vs 2289.03 MB（基准）, MAPE 5.26%
- 105 个单元测试全部通过

## 后续工作

### 短期目标（1-2 周）

1. **真实训练捕获验证**
   - 在 CPU 环境运行 `Pai-Megatron-Patch-12.0` + `Megatron-LM-20250707` 的 Qwen3 0.6B 训练
   - 启用 5 层 Hook 捕获真实计算图（基于 CPU 模拟的分布式通信）
   - 对比仿真结果与 CPU 训练的实际性能，校准 Roofline/Bandwidth 模型参数

2. **通信 Hook 完善**
   - 当前 CPU 环境使用 gloo 后端模拟分布式通信
   - 验证 `all_reduce`, `all_gather_into_tensor` 等 Hook 在 CPU 环境下能否正确捕获通信事件和数据量

### 中期目标（2-4 周）

1. **多模型支持**
   - 利用 Pai-Patch 的 40+ 模型生态
   - 为 LLaMA3、DeepSeek-V3 等模型运行捕获，建立模型库

2. **Pipeline Parallelism 完整支持**
   - 当前仅支持气泡时间估算
   - 需要完整模拟 1F1B / Interleaved 调度逻辑

3. **可视化报告**
   - 生成性能对比图表（迭代时间、内存、通信 breakdown）
   - 策略搜索结果的帕累托前沿展示

### 长期目标（1-2 个月）

1. **与 Phase 1 静态仿真集成**
   - Phase 1 静态仿真用于粗筛（秒级，误差 10-50%）
   - 参数化计算图用于精选（分钟级，误差 5-10%）
   - 统一输出格式和报告生成

2. **Auto-tuning**
   - 基于仿真结果自动推荐最优并行策略
   - 支持约束条件（内存上限、最大迭代时间等）

3. **生产环境部署**
   - 作为 Megatron-LM 训练前的配置预检工具
   - CI/CD 集成，训练前自动验证配置可行性

## 参考文献

本项目的设计参考了以下开源项目和学术工作：

- [SimAI](https://github.com/aliyun/SimAI) - 阿里巴巴全栈 AI 训练/推理仿真器
- [SimuMax](https://github.com/Optimization-AI/Resolving) - 静态分析性能预测器
- [Lumos](https://github.com/aiwaves-cn/Lumos) - 多模态智能体框架（参考非 MLSys 版本）
- [DistSim](https://github.com/Wenbin-Xiao/Distributed-Training-System-Simulator) - 事件驱动分布式训练仿真
- STAGE (arXiv:2410.07189) - 符号张量图表示
- Pytorch FX, torch.export, Compiled Autograd - PyTorch 计算图捕获技术

## License

MIT License

## 贡献者

本项目基于 Megatron-LM-20250707 和 Pai-Megatron-Patch-12.0 训练框架开发。
