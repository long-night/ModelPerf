# 基于 Megatron-LM + Pai-Megatron-Patch 的大模型训练性能仿真方案设计报告

> 报告生成时间：2026-05-05
> 研究基础：SimAI、SimuMax、Lumos、DistSim 四大主流仿真框架
> 目标框架：Megatron-LM-20250707 + Pai-Megatron-Patch-12.0

---

## 一、研究背景与现状分析

### 1.1 现有开源仿真框架总结

通过对 ModelPerf/backends/ 下三个开源项目（SimAI、SimuMax、Lumos*）及 ModelPerf/docs/ 下四份技术文档的深入分析，当前大模型训练性能仿真领域已形成三条主流技术路线：

| 框架 | 技术路线 | 核心思想 | 精度 | 是否需要真实硬件 | 开源状态 |
|------|---------|---------|------|----------------|---------|
| **SimAI** | 全栈仿真 (Full-stack) | Hijack训练框架生成工作负载 + 精确通信模拟 + 网络级事件驱动仿真 | 98.1% 对齐 | 部分模式需要 | Apache-2.0 |
| **SimuMax** | 静态解析模型 (Static Analytical) | Cost Model + Memory Model + Roofline Model，参考Megatron/DeepSpeed实现 | 性能<4%, 显存<1% | 不需要 | 开源 |
| **Lumos** | Trace驱动仿真 (Trace-driven) | PyTorch Kineto Trace 构建执行依赖图 离散事件仿真与What-if分析 | 回放3.3%, 预测4.2% | 需要单次迭代Trace | 未开源 |
| **DistSim** | 事件建模+Profiling (Event-based) | 冗余消除 + 分层依赖建模(MP/PP/DP)，仅需2节点Profiling | <4% | 仅需2节点 | 未开源 |

> *注：Lumos在backends/lumos目录下的代码并非MLSys 2025的Lumos仿真器，而是一个同名的LLM推理项目，实际Lumos仿真器未开源。*

### 1.2 各框架关键启示

#### SimAI —— 全栈仿真的工程典范
- **核心启示**：通过 Hijack 框架（欺骗Megatron/DeepSpeed在单机模拟集群行为）可生成与真实任务完全一致的工作负载，避免了手工重建框架逻辑的庞大工作量
- **技术亮点**：SimCCL修改NCCL仅572行代码即可复现通信算法；多线程+无锁共享实现23倍加速；三种模式（Analytical/Simulation/Physical）灵活切换
- **局限**：架构复杂（C++网络仿真+Python工作负载生成），编译依赖多，学习曲线陡峭

#### SimuMax —— 静态分析的最简实现
- **核心启示**：纯Python即可实现高精度仿真，参考真实框架（Megatron-LM/DeepSpeed）的实现细节是关键
- **技术亮点**：显存估计误差<1%；内置策略搜索；支持MLA、MoE、ZeRO1、Full/Selective Recompute等现代特性；提供完整的效率测试流水线生成system.json
- **局限**：不支持通算并行重叠、Context Parallel等高级特性；静态模型难以精确捕捉动态调度

#### Lumos —— Trace驱动的精度极致
- **核心启示**：无需侵入式插桩，仅利用PyTorch Kineto（约10行hook代码）即可构建精确的执行依赖图，捕捉CUDA stream间复杂依赖
- **技术亮点**：支持执行图操作（修改并行策略/模型架构）进行What-if分析；精确重现SM利用率、计算通信重叠比例
- **局限**：未开源；需要真实硬件做Profiling；不估计系统级指标（FLOPS利用率、能耗等）

#### DistSim —— 最小化Profiling成本
- **核心启示**：混合并行中大量计算/通信是重复的，可通过事件抽象+冗余消除将Profiling规模从大规模集群缩减至2个节点
- **技术亮点**：事件可复用，策略搜索效率极高；分层依赖建模（MP层内、PP层间、DP模型间）正交独立
- **局限**：基于较早架构（BERT/GPT-2），对Flash Attention、通算并行等现代优化支持有限

### 1.3 现有框架对比与选择建议

| 使用场景 | 推荐框架 | 理由 |
|---------|---------|------|
| 工业级生产环境全栈优化 | SimAI | 唯一覆盖框架+通信+网络的方案，与阿里云生产环境对齐 |
| 快速策略搜索与预规划 | SimuMax | 纯Python、无需硬件、显存精度极高、内置策略搜索 |
| 精确理解训练行为 | Lumos | Trace-driven精度最高，能捕捉复杂执行依赖和重叠行为 |
| 低资源环境策略评估 | DistSim | 仅需2节点Profiling，事件可复用 |

---

## 二、核心思路

### 2.1 设计哲学

> **环境前提：本项目的大模型训练与性能仿真完全基于 CPU 环境运行，与 CUDA、GPU 无关。**
>
> 所有训练流程、计算图捕获、性能仿真和 What-if 分析均在纯 CPU 环境下完成。项目中涉及的 GPU 硬件参数（如 A100/H100 算力、NVLink 带宽）仅作为仿真输入的理论值，不依赖真实 GPU 硬件。

基于现有 Megatron-LM + Pai-Megatron-Patch 训练框架，我们提出 **静态分析为体，Trace校准为用，框架适配为桥** 的混合仿真路线：

1. **以静态分析模型为体**：参考 SimuMax 的 Cost Model + Memory Model + Roofline Model，在 CPU 环境下即可快速评估各种并行策略的性能和显存占用，无需 GPU 资源
2. **以 Trace 校准为用**：参考 Lumos 的 Trace-driven 思想，在 CPU 环境下通过合成 Trace 或基于 CPU 运行的 PyTorch Profiler 采集关键算子执行时间，用于自动校准静态模型中的效率参数，提升仿真精度
3. **以框架适配为桥**：参考 Pai-Patch 的 Monkey-patch 架构和 SimAI 的 Hijack 思想，设计非侵入式适配层，自动从 Megatron-LM/Pai-Patch 训练流程中提取模型结构、并行策略配置和运行时参数，实现训练即仿真配置的无缝衔接

### 2.2 与现有框架的差异化定位

| 维度 | SimAI | SimuMax | **本方案 (ModelPerf)** |
|------|-------|---------|----------------------|
| 与Megatron关系 | Hijack生成负载 | 参考实现细节 | **深度集成，自动提取配置** |
| 多模型支持 | GPT/LLaMA/DeepSeek/Qwen | LLaMA/DeepSeek/Mixtral/Qwen | **依托Pai-Patch，支持40+模型** |
| 技术路线 | 全栈仿真 | 纯静态分析 | **静态分析 + Trace校准** |
| 使用门槛 | 高（需编译C++网络仿真） | 低（纯Python） | **低（自动配置提取）** |
| 并行策略支持 | TP/PP/DP/EP/SP | TP/PP/DP/EP/SP/ZeRO1 | **扩展CP、通算并行等** |
| 显存估计 | 支持 (1.6+) | 支持 (<1%误差) | **核心目标** |
| 策略搜索 | 不支持 | 支持 | **核心目标** |

### 2.3 核心创新点

1. **框架原生集成**：不同于 SimAI 的 Hijack 和 SimuMax 的手工配置，本方案通过 Monkey-patch Megatron-LM 的并行状态管理模块（parallel_state.py）和训练参数模块（arguments.py），在训练启动时自动捕获模型配置和并行策略，实现零配置仿真
2. **混合精度校准**：静态模型在 CPU 上快速迭代策略搜索，Trace 采集基于 CPU 运行的 PyTorch Profiler 校准关键算子效率，两者互补形成粗筛+精校的两级优化流程
3. **面向Pai-Patch的多模型统一抽象**：利用 Pai-Patch 已有的 40+ 模型统一接口，建立跨模型的通用性能分析层，避免为每个模型重复构建仿真逻辑

---

## 三、技术路线

### 3.1 总体技术架构

```
                        ModelPerf 性能仿真平台
  +---------------------------------------------------------------+
  |  +------------+  +------------+  +-------------------------+  |
  |  | Framework  |  |   Static   |  |      Trace Engine       |  |
  |  |  Adapter   |->|   Engine   |<-|  (PyTorch Profiler /    |  |
  |  |(Monkey-p.) |  |(Cost+Mem+  |  |   Kineto / Nsight)      |  |
  |  +------------+  | Roofline)  |  +-------------------------+  |
  |       |          +------+-----+           ^                   |
  |       |                 |                 |                   |
  |       v                 v                 |                   |
  |  +---------------------------------------------------------+  |
  |  |              Simulation Core (事件驱动调度器)              |  |
  |  |  - 计算事件调度 (GEMM, Attention, LayerNorm, MLP)        |  |
  |  |  - 通信事件调度 (AllReduce, AllGather, ReduceScatter)    |  |
  |  |  - 流水线调度 (1F1B, Interleaved-1F1B)                   |  |
  |  |  - 通算重叠模拟                                          |  |
  |  +---------------------------------------------------------+  |
  |       |                                                       |
  |       v                                                       |
  |  +---------------------------------------------------------+  |
  |  |              Analysis and Optimization Layer             |  |
  |  |  +-----------+  +-----------+  +---------------------+   |  |
  |  |  |  Memory   |  | Performance|  |  Strategy Searcher  |   |  |
  |  |  | Analyzer  |  |  Analyzer  |  |  (Grid/Bayesian)    |   |  |
  |  |  +-----------+  +-----------+  +---------------------+   |  |
  |  +---------------------------------------------------------+  |
  |       |                                                       |
  |       v                                                       |
  |  +---------------------------------------------------------+  |
  |  |              Visualization and Reporting                   |  |
  |  |  - 迭代时间分解 (计算/通信/Bubble/Optimizer)               |  |
  |  |  - 显存占用分解 (模型/激活/梯度/优化器状态)                |  |
  |  |  - 策略对比报告 Roofline分析图 时间线可视化                |  |
  |  +---------------------------------------------------------+  |
  +---------------------------------------------------------------+
```

### 3.2 关键模块技术路线

#### 3.2.1 Framework Adapter（框架适配层）

**目标**：实现从 Megatron-LM/Pai-Patch 训练代码到仿真配置的自动提取，消除手工配置负担。

**技术方案**：
- **配置提取**：Hook megatron/training/arguments.py 中的参数解析流程，在 parse_args() 后自动导出 model_config.json + strategy_config.json
- **模型结构提取**：Hook megatron/core/transformer/transformer_layer.py 的层构建过程，通过 forward_pre_hook 记录每层的输入输出形状、参数形状和算子类型
- **并行策略提取**：Hook megatron/core/parallel_state.py 的初始化函数，自动获取 TP/PP/DP/CP/EP size 和通信组拓扑
- **动态追踪**：参考 Lumos，在训练脚本中添加约10行 PyTorch Profiler 代码，自动捕获单次迭代的 Trace 并解析为事件列表

**参考实现**：
```python
# modelperf/framework_adapter/megatron_hooks.py
def register_megatron_hooks():
    # 自动提取Megatron-LM训练配置的Monkey-patch入口
    # 1. Hook arguments.parse_args
    # 2. Hook transformer_layer.TransformerLayer.__init__
    # 3. Hook parallel_state.initialize_model_parallel
    # 4. 在训练iteration前后插入Profiler上下文管理器
```

#### 3.2.2 Static Engine（静态仿真引擎）

**目标**：在 CPU 环境下实现快速、高精度的性能和显存估计。

**技术方案**：
- **Cost Model**：参考 SimuMax 和 Calculon，基于算子理论 FLOPs 和内存访问量，结合硬件峰值算力和带宽，通过 Roofline Model 估计每个算子的执行时间
  - 计算密集型算子：time = FLOPs / (peak_TFLOPS * efficiency_factor)
  - 内存带宽密集型算子：time = mem_bytes / (peak_BW * efficiency_factor)
- **Memory Model**：精确跟踪前向/反向/recompute 各阶段的激活显存峰值，参考 Megatron-LM 的显存分配逻辑
  - 支持 Full Recompute / Selective Recompute / Sequence Parallel 的显存节省计算
  - 支持 ZeRO-1 的优化器状态/梯度分片显存计算
  - 支持 Pipeline Parallel 1F1B 调度下的激活缓存峰值计算
- **Communication Model**：基于通信数据量和网络带宽/延迟模型，估计 AllReduce/AllGather/ReduceScatter/P2P/All2All 的执行时间
  - 节点内通信（NVLink/PCIe）与节点间通信（IB/RoCE）区分建模
  - 支持通信与计算重叠的保守估计（考虑overlap系数）

**关键技术公式**（参考 SimuMax 和 DistSim）：

| 组件 | 公式/方法 |
|------|----------|
| 单层Transformer FLOPs | 6 * batch * seq_len * hidden_size^2 * (1 + 2 * intermediate_ratio) |
| Attention FLOPs | 4 * batch * seq_len^2 * hidden_size (或 FlashAttention优化后) |
| AllReduce时间 | 2(N-1)/N * data_size / bandwidth + latency (Ring算法) |
| 1F1B Bubble时间 | (pp_size - 1) * (fwd_time + bwd_time) |
| ZeRO-1显存 | params + grads/DP + optimizer_states/DP + activations |

#### 3.2.3 Trace Engine（Trace采集与校准引擎）

**目标**：在 CPU 环境下采集关键算子执行时间，自动校准 Static Engine 的效率参数。

**技术方案**：
- **Trace采集**：使用 torch.profiler（PyTorch Profiler / Kineto）在单次 CPU 训练迭代中采集算子执行时间（注：CPU 环境下采集的是 CPU kernel 执行时间，作为效率校准的参考基准）
- **Trace解析**：参考 Lumos，从 Trace 中提取任务信息：
  - CPU任务：PyTorch算子执行事件
  - 计算任务：矩阵运算（gemm）、注意力（attention）、归一化（layer_norm）等算子在 CPU 上的执行时间
- **效率校准**：将 Trace 中实测的算子执行时间与 Static Engine 的理论估计对比，自动拟合 efficiency_factor 参数
  - 按算子类型+输入形状维度建立效率查找表（类似 SimuMax 的 accurate_efficient_factor）
  - 支持线性回归拟合通信带宽效率（参考 SimuMax 的 nccl_fit.py）

#### 3.2.4 Simulation Core（事件驱动调度核心）

**目标**：精确模拟分布式训练的执行时序，支持通算重叠和流水线 bubble 分析。

**技术方案**：
- **事件抽象**：参考 DistSim，将训练过程抽象为两类事件：
  - 计算事件：GEMM、Attention、LayerNorm、MLP、Optimizer 等
  - 通信事件：AllReduce（DP梯度同步）、AllGather（TP参数聚合）、P2P（PP激活传递）、All2All（MoE专家分发）
- **分层调度器**：
  - 层内调度：模拟 Tensor Parallel 下的计算+通信时序（考虑通算重叠）
  - 层间调度：模拟 Pipeline Parallel 1F1B 调度（ warmup -> steady -> cooldown ）
  - 模型间调度：模拟 Data Parallel 下的梯度同步时机（后向传播结束后 AllReduce）
- **依赖图**：参考 Lumos，维护四类依赖关系：
  - Intra-thread dependency（同一线程内顺序）
  - Inter-thread dependency（跨线程同步）
  - CUDA launch dependency（CPU->GPU发射）
  - CUDA synchronization dependency（Stream间同步）

#### 3.2.5 Strategy Searcher（策略搜索模块）

**目标**：在给定模型和硬件约束下，自动搜索最优并行策略配置。

**技术方案**：
- **搜索空间定义**：
  - 变量：TP size, PP size, DP size, EP size, Micro-batch size, Sequence Parallel, Recompute granularity
  - 约束：TP * PP * DP = world_size, micro_batch_size * micro_batch_num * DP = global_batch_size, peak_mem < memory_limit
- **搜索算法**：
  - Phase 1：网格搜索（Grid Search）快速筛选可行配置（利用Static Engine的CPU快速计算优势）
  - Phase 2：对Top-K候选配置使用Trace Engine在CPU环境下精校，选出最优配置
  - 可扩展：引入贝叶斯优化（Bayesian Optimization）或强化学习进一步加速搜索

#### 3.2.6 Visualization and Reporting（可视化与报告模块）

**目标**：生成直观、可对比的性能分析报告，辅助决策。

**技术方案**：
- **迭代时间分解图**：堆叠柱状图展示计算/通信/Bubble/Optimizer各部分占比
- **显存占用分解图**：饼图展示模型参数/激活值/梯度/优化器状态的显存分布
- **Roofline分析图**：直观展示模型各算子处于计算受限还是带宽受限区域
- **策略对比表**：多维度对比不同并行策略的吞吐量、显存占用、MFU等指标
- **时间线可视化**：参考 Lumos，生成类似 Chrome Trace 的交互式时间线

---

## 四、整体结构（项目目录规划）

```
ModelPerf/
├── modelperf/                          # 核心Python包
│   ├── __init__.py
│   ├── framework_adapter/              # 框架适配层
│   │   ├── __init__.py
│   │   ├── megatron_hooks.py           # Megatron-LM Monkey-patch hooks
│   │   ├── pai_patch_hooks.py          # Pai-Megatron-Patch 适配器
│   │   └── config_extractor.py         # 配置自动提取工具
│   ├── static_engine/                  # 静态仿真引擎
│   │   ├── __init__.py
│   │   ├── cost_model.py               # Cost Model (FLOPs/BW估算)
│   │   ├── memory_model.py             # Memory Model (显存峰值跟踪)
│   │   ├── comm_model.py               # Communication Model (通信时间估算)
│   │   ├── roofline.py                 # Roofline Model
│   │   └── models/                     # 各模型架构定义
│   │       ├── transformer.py          # Transformer通用结构
│   │       ├── llama.py                # LLaMA系列
│   │       ├── qwen.py                 # Qwen系列
│   │       ├── deepseek.py             # DeepSeek系列
│   │       └── ...                     # 其他Pai-Patch支持的模型
│   ├── trace_engine/                   # Trace采集与校准引擎
│   │   ├── __init__.py
│   │   ├── profiler.py                 # PyTorch Profiler封装
│   │   ├── trace_parser.py             # Trace解析器
│   │   └── calibrator.py               # 效率参数校准器
│   ├── simulation_core/                # 事件驱动调度核心
│   │   ├── __init__.py
│   │   ├── events.py                   # 事件抽象定义
│   │   ├── scheduler.py                # 分层调度器
│   │   ├── dependency_graph.py         # 依赖图构建
│   │   └── pipeline_simulator.py       # Pipeline调度仿真
│   ├── strategy_searcher/              # 策略搜索模块
│   │   ├── __init__.py
│   │   ├── search_space.py             # 搜索空间定义
│   │   ├── grid_search.py              # 网格搜索
│   │   └── optimizer.py                # 贝叶斯优化等高级搜索
│   ├── analysis/                       # 分析层
│   │   ├── __init__.py
│   │   ├── memory_analyzer.py          # 显存分析器
│   │   ├── performance_analyzer.py     # 性能分析器
│   │   └── bottleneck_detector.py      # 瓶颈检测器
│   └── reporting/                      # 可视化与报告
│       ├── __init__.py
│       ├── charts.py                   # 图表生成
│       ├── timeline.py                 # 时间线可视化
│       └── report_generator.py         # 报告生成器
├── configs/                            # 配置文件模板
│   ├── models/                         # 模型配置模板
│   ├── strategies/                     # 策略配置模板
│   └── systems/                        # 系统/硬件配置模板
├── tools/                              # 辅助工具
│   ├── efficiency_test/                # 效率测试流水线（参考SimuMax）
│   ├── trace_collector/                # Trace采集脚本
│   └── benchmark/                      # 与真实训练对齐的benchmark脚本
├── examples/                           # 使用示例
│   ├── perf_llama3_8b.py               # LLaMA3 8B性能评估示例
│   ├── perf_qwen3_0.6b.py              # Qwen3 0.6B性能评估示例
│   ├── strategy_search_demo.py         # 策略搜索示例
│   └── trace_calibration_demo.py       # Trace校准示例
├── tests/                              # 单元测试
├── docs/                               # 技术文档
├── backends/                           # 参考的开源项目（已存在）
│   ├── SimAI/
│   ├── SimuMax/
│   └── lumos/
├── setup.py
├── requirements.txt
└── README.md
```

---

## 五、开发阶段与里程碑

### Phase 1：静态仿真核心（4-6周）—— ✅ 已完成

**目标**：建立可在CPU环境下独立运行的静态性能仿真能力。

| 任务 | 交付物 | 验收标准 | 状态 |
|------|--------|----------|------|
| 模型配置抽象层 | ModelConfig / StrategyConfig / SystemConfig | 支持JSON序列化，与SimuMax格式兼容 | ✅ 已实现（config_classes.py） |
| Cost Model实现 | cost_model.py | 能正确估算Transformer各算子FLOPs和内存访问量 | ✅ 已实现（roofline.py） |
| Memory Model实现 | memory_model.py | 显存估算误差<5%（与Megatron-LM实际运行对比） | ✅ 已实现（memory_tracker.py） |
| Communication Model实现 | comm_model.py | 支持AllReduce/AllGather/P2P/All2All时间估算 | ✅ 已实现（bandwidth.py） |
| Pipeline调度仿真 | pipeline_simulator.py | 支持1F1B调度，Bubble时间估算正确 | ⚠️ 仅Bubble时间估算 |
| 首个模型端到端仿真 | perf_llama3_8b.py | 输出与SimuMax对标，性能误差<10% | ✅ basic_usage.py 验证通过 |

**技术参考**：SimuMax（simumax/core/perf_llm.py, config.py, transformer/language_model.py）

### Phase 2：框架适配与自动配置提取（2-3周）—— ✅ 已完成（2026-05-05）

**目标**：实现从Megatron-LM/Pai-Patch训练代码到仿真配置的零手工提取。

| 任务 | 交付物 | 验收标准 | 状态 |
|------|--------|----------|------|
| Megatron参数Hook | megatron_hooks.py | 能从arguments.py自动提取并导出JSON配置 | ✅ 已验证（Qwen3 0.6B） |
| 模型结构Hook | transformer_layer hook | 能记录每层的输入输出形状和参数信息 | ✅ 已验证 |
| 并行策略Hook | parallel_state hook | 能自动获取TP/PP/DP/CP/EP配置 | ✅ 已验证 |
| Pai-Patch适配 | pai_patch_hooks.py | 支持Qwen/DeepSeek/LLaMA等模型的自动适配 | ✅ Qwen3已验证 |
| 端到端集成测试 | integration_test.py | 训练脚本启动后自动生成仿真配置并运行 | ✅ end_to_end_pipeline.py |

**验证结果**（2026-05-05）：
- 在 Qwen3 0.6B CPU 训练上成功集成 ModelPerf Hook
- 自动提取 model_config (17 fields) / strategy_config (14 fields) / system_config (10 fields)
- ModuleCapture + CommunicationCapture 在 gloo 后端下捕获 14 节点计算图（6 通信节点）
- 端到端流水线：训练 → 配置提取 → 计算图捕获 → 性能仿真 → JSON 报告

**技术参考**：SimAI的Hijack思想、Pai-Patch的Monkey-patch架构

### Phase 3：Trace采集与校准（2-3周）

**目标**：基于CPU环境引入Trace-driven机制提升仿真精度。

| 任务 | 交付物 | 验收标准 |
|------|--------|----------|
| PyTorch Profiler封装 | profiler.py | 单次迭代采集，开销<5%（CPU环境） |
| Trace解析器 | trace_parser.py | 能提取CPU执行事件及依赖关系 |
| 效率参数校准 | calibrator.py | 自动拟合efficiency_factor，性能误差降至<5% |
| 通信带宽校准 | comm_fit.py | 线性回归拟合带宽效率，通信误差<10% |
| 校准流水线 | calibration_pipeline.sh | 一键完成从Trace采集到参数更新 |

**技术参考**：Lumos（PyTorch Kineto Trace解析）、SimuMax（nccl_fit.py, efficiency_test/）

### Phase 4：策略搜索与优化（2-3周）

**目标**：实现自动化并行策略搜索。

| 任务 | 交付物 | 验收标准 |
|------|--------|----------|
| 搜索空间定义 | search_space.py | 覆盖TP/PP/DP/EP/MBS/CP/Recompute |
| 网格搜索实现 | grid_search.py | 1000种配置搜索时间<10分钟（CPU） |
| 显存约束过滤 | memory_constraint.py | 自动过滤OOM配置 |
| Top-K精校 | topk_refiner.py | 对Top-10候选用Trace精校，选出最优 |
| 策略对比报告 | strategy_report.py | 输出多策略对比表和推荐配置 |

**技术参考**：SimuMax（examples/search/llm_search.py）、DistSim（分层依赖建模）

### Phase 5：可视化与报告（2周）

**目标**：建立完善的报告体系。

| 任务 | 交付物 | 验收标准 |
|------|--------|----------|
| 迭代时间分解图 | charts.py | 堆叠柱状图准确展示计算/通信/Bubble占比 |
| 显存分解图 | memory_chart.py | 饼图展示参数/激活/梯度/优化器状态分布 |
| Roofline图 | roofline_chart.py | 标注各算子在Roofline上的位置 |
| 时间线可视化 | timeline.py | 生成Chrome Trace格式或类似交互式时间线 |
| 综合报告生成 | report_generator.py | Markdown/HTML双格式，一键生成 |

### Phase 6：验证与迭代（持续）

**目标**：与真实训练结果持续对齐，扩展模型和策略支持。

| 任务 | 目标 |
|------|------|
| 精度验证 | 与Megatron-LM真实训练对比，端到端性能误差<5%，显存误差<2% |
| 模型扩展 | 支持Pai-Patch全部40+模型（Qwen3、DeepSeek-V3、LLaMA3等） |
| 策略扩展 | 支持Context Parallel、通算并行、MoE EP等新特性 |
| 硬件扩展 | 支持不同GPU架构（A100/H100/H20）的理论效率参数配置（基于公开规格，无需真实硬件） |
| 社区开源 | 整理文档，开源ModelPerf核心代码 |

---

## 六、总结与展望

### 6.1 核心结论

通过对 SimAI、SimuMax、Lumos、DistSim 四大主流大模型训练性能仿真框架的深入分析，我们提出了 **静态分析为体，Trace校准为用，框架适配为桥** 的混合仿真路线。该方案具有以下核心优势：

1. **与Megatron-LM/Pai-Patch深度集成**：通过Monkey-patch自动提取训练配置，无需手工维护仿真模型，天然支持Pai-Patch的40+模型生态
2. **纯CPU环境运行**：静态模型在CPU上快速策略搜索（秒级），Trace基于CPU训练采集并精校关键参数（分钟级），形成高效的两级优化流程，无需GPU硬件
3. **兼顾精度与可用性**：参考SimuMax实现显存误差<1%的Memory Model，参考Lumos实现Trace驱动的执行依赖建模，参考DistSim实现事件抽象与分层调度
4. **面向生产环境**：内置策略搜索、瓶颈检测、可视化报告，直接服务于训练调优和资源配置决策

### 6.2 技术演进路径

| 阶段 | 技术路线 | 参考框架 |
|------|---------|---------|
| 入门学习 | 纯静态分析模型 | SimuMax |
| 核心能力 | 静态模型 + Trace校准 | SimuMax + Lumos |
| 高级特性 | 事件驱动调度 + 策略搜索 | DistSim + SimuMax |
| 专家阶段 | 全栈仿真（框架+通信+网络） | SimAI |

### 6.3 下一步行动建议

#### 已完成（2026-05-05）
1. ✅ **Phase 1 静态仿真核心**：Cost/Mem/Comm Model + 虚拟执行引擎 + What-if 分析
2. ✅ **Phase 2 框架适配**：megatron_hooks.py / config_extractor.py / pai_patch_hooks.py 开发完成
3. ✅ **真实训练验证**：Qwen3 0.6B CPU 训练 Hook 捕获验证通过，配置自动提取 + 计算图捕获 + 仿真报告生成

#### 近期目标（1-2 周）
1. **计算图捕获完善**：在训练循环中插入 ModuleCapture，捕获完整前向 + 反向传播路径
2. **What-if 分析集成**：将捕获图加载到 WhatIfAnalyzer，演示修改 TP/PP/BS 后的性能变化
3. **多模型适配验证**：在 LLaMA3、DeepSeek-V3 训练脚本上测试 Hook 兼容性

#### 中期目标（2-4 周）
1. **Trace 采集与校准**：基于 CPU PyTorch Profiler 采集算子执行时间，自动拟合 efficiency_factor
2. **Pipeline Parallelism 完整支持**：完整模拟 1F1B / Interleaved 调度逻辑
3. **可视化报告**：生成性能对比图表、帕累托前沿展示
4. **两周后接入**：在CPU环境下采集Trace，校准效率参数，验证端到端精度

---

*本报告基于 ModelPerf/backends/ 下的 SimAI、SimuMax、lumos 开源项目及 ModelPerf/docs/ 下的四份技术分析文档综合整理生成。*


---

# 附录：基于参数化计算图的性能仿真技术路线（优先功能）

> 说明：本附录针对用户提出的优先功能需求，详细阐述基于参数化计算图的性能仿真技术路线。
> 核心思想：通过一次真实训练迭代捕获参数化计算图（含通信算子），然后在该图上进行虚拟执行和 What-if 分析。

---

## A.1 总体技术思路

用户提出的三个功能需求本质上构成了一个 **Capture-Parameterize-Execute** 的闭环仿真系统：

```
真实训练（1次迭代）
    |
    v
[Capture] 捕获参数化计算图（前向+反向+通信）
    |
    v
[Parameterize] 将具体数值转为符号公式（shape、FLOPs、通信量）
    |
    v
[Modify Config] 修改并行策略/模型参数/硬件配置
    |
    v
[Re-evaluate] 基于符号公式自动更新图中所有数值
    |
    v
[Virtual Execute] 虚拟执行：统计每个算子性能 + 内存峰值
    |
    v
[Report] 生成性能报告（迭代时间、MFU、显存、瓶颈分析）
```

该方案相比纯静态分析（Phase 1）的核心优势：

| 对比维度 | 纯静态分析（Phase 1） | 参数化计算图方案 |
|---------|---------------------|----------------|
| 图结构来源 | 手工建模（参考Megatron论文） | 从真实代码自动捕获 |
| 通信算子 | 手工推导通信位置和通信量 | 通过Monkey-patch自动捕获 |
| 反向传播 | 假设 forward=1x, backward=2x FLOPs | 真实捕获 backward 图结构 |
| 新模型适配 | 需要为新模型手工写仿真逻辑 | 运行一次训练自动捕获 |
| What-if分析 | 修改配置后重新运行仿真代码 | 修改配置后直接在参数化图上重评估 |
| 精度 | 依赖理论公式准确性 | 基于真实执行路径，更贴近实际 |

### A.1.1 与现有框架的对应关系

| 本方案模块 | 对应现有框架技术 |
|-----------|----------------|
| 计算图捕获 | SimAI AICB（Hijack生成Workload）+ Lumos（Trace提取执行图） |
| 参数化公式 | SimuMax（公式化FLOPs/通信量计算）+ STAGE（符号张量图） |
| 虚拟执行调度 | DistSim（事件驱动仿真）+ Lumos（依赖图调度） |
| 内存跟踪 | SimuMax Memory Model + Megatron理论显存公式 |

### A.1.2 关键技术选型依据

综合对 SimAI AICB、Lumos、STAGE、PyTorch FX、torch.export 和 Compiled Autograd 的深入研究，推荐采用以下技术组合：

| 技术 | 来源 | 适用场景 | 选择理由 |
|------|------|---------|---------|
| Monkey-patch `torch.distributed` | SimAI AICB | 通信算子捕获 | 唯一能在不修改 Megatron 源码的情况下捕获 NCCL 通信的方法 |
| `nn.Module` forward hook | PyTorch 原生 | 计算节点层次结构 | 成熟稳定，可获取输入输出 shape |
| `Tensor.register_hook` + `torch.autograd` | Lumos | 反向传播路径 | 可精确捕获梯度流动路径 |
| PyTorch Kineto Profiler | PyTorch 原生 | 运行时时间校准 | Megatron training.py 已集成，record_shapes=True 可获取 shape |
| Symbolic Tensor Graph (STAGE) | 学术研究 | 参数化表示 | 支持符号张量表示，已验证 32K GPU 规模 |
| Compiled Autograd | PyTorch 2.4+ | 完整反向图捕获 | 可捕获更大的反向图，包括反向钩子 |

---

## A.2 功能1：计算图捕获与参数化

### A.2.1 核心挑战

在 Megatron-LM 中捕获完整计算图（含通信算子）面临以下挑战：

1. **通信算子不可见**：`torch.distributed.all_reduce` / `all_gather` 等通信操作是通过 PyTorch 分布式 API 直接调用的，不是 `nn.Module`，不会出现在 `named_modules()` 遍历中
2. **反向传播动态构建**：PyTorch 的反向图是在运行时动态构建的，无法通过静态分析完整获取
3. **融合算子黑盒**：`flash_attention`、`fused_layer_norm` 等融合算子的内部结构不可见
4. **Pipeline P2P通信**：阶段间的 `isend`/`irecv` 分散在 `pipeline_parallel/schedules.py` 的调度逻辑中

### A.2.2 推荐技术路线：Hybrid Capture + Symbolic Parameterization

基于研究成果，推荐采用 **五层 Hook 叠加** 的混合捕获方案：

```
Layer 1: Module-level Hook（nn.Module forward hook）
         -> 捕获模型层次结构（TransformerLayer -> Attention -> MLP -> Linear）
         -> 记录每层输入输出 Tensor 的 shape、dtype、device

Layer 2: Communication Hook（Monkey-patch torch.distributed）
         -> 捕获 AllReduce / AllGather / ReduceScatter / P2P 通信事件
         -> 记录通信数据量、通信组大小、通信类型

Layer 3: Autograd Function Hook（torch.autograd.Function hook）
         -> 捕获自定义 autograd function（如 LinearWithGradAccumulationAndAsyncCommunication）
         -> 记录 forward / backward 的输入输出

Layer 4: Backward Hook（torch.autograd.graph.register_hooks）
         -> 捕获标准 PyTorch 算子的反向传播（matmul, add, etc.）
         -> 构建完整的反向计算图

Layer 5: PyTorch Profiler（torch.profiler with record_shapes=True）
         -> 采集运行时实际执行时间
         -> 验证 Hook 捕获的 shape 准确性
         -> 获取 PyTorch 内置 FLOPs 估算
```

### A.2.3 Layer 1: Module-level Hook（计算节点层次结构）

**目标**：捕获 TransformerLayer -> Attention -> MLP -> Linear 的层次结构，记录每层输入输出 shape。

**实现方式**：

```python
class GraphCaptureHook:
    def __init__(self, global_config):
        self.graph = ComputationalGraph()
        self.module_stack = []
        self.node_id = 0
        self.global_config = global_config

    def forward_pre_hook(self, module, input):
        path = self._get_module_path(module)
        input_shapes = [tuple(t.shape) if isinstance(t, torch.Tensor) else None
                        for t in input if isinstance(t, torch.Tensor)]
        self.current_node = self.graph.add_node(
            type="compute",
            op_type=module.__class__.__name__,
            module_path=path,
            input_shapes=input_shapes,
            params={name: tuple(p.shape) for name, p in module.named_parameters()}
        )

    def forward_hook(self, module, input, output):
        output_shapes = []
        if isinstance(output, torch.Tensor):
            output_shapes = [tuple(output.shape)]
        elif isinstance(output, (tuple, list)):
            output_shapes = [tuple(t.shape) if isinstance(t, torch.Tensor) else None
                             for t in output]
        self.current_node.output_shapes = output_shapes
        # 符号化 shape 推断
        self.current_node.symbolic_shapes = infer_symbolic_shapes(
            output_shapes, module.__class__.__name__,
            self.current_node.params, self.global_config
        )

    def _get_module_path(self, module):
        return module._get_name()
```

**注册 Hook**：

```python
def register_graph_capture_hooks(model, config):
    capturer = GraphCaptureHook(config)
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子模块
            module.register_forward_pre_hook(capturer.forward_pre_hook)
            module.register_forward_hook(capturer.forward_hook)
    return capturer
```

**关键实现细节**：
- 对于 `ColumnParallelLinear` / `RowParallelLinear` 等 Megatron 特定模块，需要识别其 TP 切分维度
- 输入 shape `[S, B, H]`（seq_first 格式）需要正确解析为 `(seq_len, micro_batch_size, hidden_size)`
- 参数 shape 需要识别哪些维度被 TP / EP 切分（如 `weight.shape = [H/tp, H]`）

### A.2.4 Layer 2: Communication Hook（通信算子捕获）

**目标**：在真实训练过程中精确捕获所有 NCCL 通信操作的位置、类型和数据量。

**实现方式**：Monkey-patch `torch.distributed` 中的通信原语。

```python
import torch.distributed as dist
import inspect

_orig_all_reduce = dist.all_reduce
_orig_all_gather = dist.all_gather_into_tensor
_orig_reduce_scatter = dist.reduce_scatter_tensor
_orig_batch_isend_irecv = dist.batch_isend_irecv

class CommunicationCapture:
    def __init__(self, graph):
        self.graph = graph
        self.comm_counter = 0

    def _get_call_context(self):
        stack = inspect.stack()
        for frame in stack:
            if "forward" in frame.function or "backward" in frame.function:
                return frame.function, frame.filename, frame.lineno
        return "unknown", "", 0

    def _add_comm_node(self, op_type, tensor_or_size, group=None, async_op=False):
        context = self._get_call_context()
        if isinstance(tensor_or_size, torch.Tensor):
            shape = tuple(tensor_or_size.shape)
            numel = tensor_or_size.numel()
            dtype_size = tensor_or_size.element_size()
        else:
            shape = None; numel = tensor_or_size; dtype_size = 2
        comm_size = dist.get_world_size(group) if group else dist.get_world_size()
        group_name = str(group) if group else "world"
        node = self.graph.add_node(
            type="comm", op_type=op_type, comm_size=comm_size,
            comm_group=group_name, tensor_shape=shape, tensor_numel=numel,
            tensor_dtype_size=dtype_size, comm_bytes=numel * dtype_size,
            symbolic_comm_bytes=f"{numel} * dtype_size",
            call_context=context, async_op=async_op)
        return node

    def patch_all_reduce(self, tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        self._add_comm_node("all_reduce", tensor, group, async_op)
        return _orig_all_reduce(tensor, op, group, async_op)

    def patch_all_gather(self, output_tensor, input_tensor, group=None, async_op=False):
        self._add_comm_node("all_gather", input_tensor, group, async_op)
        return _orig_all_gather(output_tensor, input_tensor, group, async_op)

    def patch_reduce_scatter(self, output_tensor, input_tensor, group=None, async_op=False):
        self._add_comm_node("reduce_scatter", input_tensor, group, async_op)
        return _orig_reduce_scatter(output_tensor, input_tensor, group, async_op)

    def patch_batch_isend_irecv(self, p2p_ops):
        for op in p2p_ops:
            op_type = "p2p_send" if op.op == dist.isend else "p2p_recv"
            self._add_comm_node(op_type, op.tensor, group=None, async_op=True)
        return _orig_batch_isend_irecv(p2p_ops)

    def install(self):
        dist.all_reduce = self.patch_all_reduce
        dist.all_gather_into_tensor = self.patch_all_gather
        dist.reduce_scatter_tensor = self.patch_reduce_scatter
        dist.batch_isend_irecv = self.patch_batch_isend_irecv
```

**需要 Patch 的通信原语清单**（基于 Megatron-LM 代码分析）：

| 通信原语 | Megatron 使用场景 | 所在文件 | 触发时机 |
|---------|----------------|---------|---------|
| `torch.distributed.all_reduce` | TP: ColumnParallelLinear backward dgrad; DP: 梯度同步 | `tensor_parallel/layers.py:344`, `distributed/param_and_grad_buffer.py` | Attention/MLP backward 后; Optimizer step 前 |
| `torch.distributed.all_gather_into_tensor` | TP: RowParallelLinear forward all-gather input | `tensor_parallel/layers.py:472` | RowParallelLinear forward 中 |
| `torch.distributed.reduce_scatter_tensor` | SP: Sequence Parallel grad reduce-scatter | `tensor_parallel/mappings.py` | Sequence Parallel backward 中 |
| `torch.distributed.P2POp` + `batch_isend_irecv` | PP: 阶段间激活值传递 | `pipeline_parallel/p2p_communication.py:72-100` | Pipeline 每 micro-batch forward/backward 阶段间 |
| `torch.distributed.all_to_all` | EP: MoE 专家分发 | `transformer/moe/token_dispatcher.py` | MoE layer forward/backward |
| `torch.distributed.ring_exchange` | PP: Ring-exchange P2P | `pipeline_parallel/p2p_communication.py:62` | 可选的 P2P 优化 |

### A.2.5 Layer 3: Autograd Function Hook（自定义反向传播捕获）

**目标**：捕获 Megatron 中自定义 `torch.autograd.Function` 的 forward/backward。

**关键 Autograd Function**（在 `tensor_parallel/layers.py` 中定义）：

- `LinearWithGradAccumulationAndAsyncCommunication`（line 432）: forward 中有 AllGather，backward 中有 AllReduce
- `LinearWithFrozenWeight`（line 314）: backward 中有 AllReduce

**实现方式**：

```python
def patch_autograd_function(func_cls, graph):
    orig_forward = func_cls.forward
    orig_backward = func_cls.backward

    @staticmethod
    def new_forward(ctx, *args, **kwargs):
        node = graph.add_node(
            type="autograd_fwd", op_type=func_cls.__name__,
            input_shapes=[tuple(a.shape) if isinstance(a, torch.Tensor) else None
                          for a in args])
        result = orig_forward(ctx, *args, **kwargs)
        if isinstance(result, torch.Tensor):
            node.output_shapes = [tuple(result.shape)]
        elif isinstance(result, (tuple, list)):
            node.output_shapes = [tuple(r.shape) if isinstance(r, torch.Tensor) else None
                                  for r in result]
        return result

    @staticmethod
    def new_backward(ctx, *grad_outputs):
        node = graph.add_node(
            type="autograd_bwd", op_type=func_cls.__name__,
            input_shapes=[tuple(g.shape) if isinstance(g, torch.Tensor) else None
                          for g in grad_outputs])
        result = orig_backward(ctx, *grad_outputs)
        return result

    func_cls.forward = new_forward
    func_cls.backward = new_backward
```

### A.2.6 Layer 4: Backward Hook（标准算子反向传播捕获）

**目标**：为每个 module 的输出 tensor 注册 backward hook，记录梯度 shape 和反向路径。

```python
def register_backward_hooks(model, graph):
    for name, module in model.named_modules():
        def make_hook(module_name):
            def hook(module, grad_input, grad_output):
                node = graph.add_node(
                    type="backward",
                    op_type=f"bwd_{module.__class__.__name__}",
                    module_path=module_name,
                    grad_input_shapes=[tuple(g.shape) if isinstance(g, torch.Tensor) else None
                                       for g in grad_input if g is not None],
                    grad_output_shapes=[tuple(g.shape) if isinstance(g, torch.Tensor) else None
                                        for g in grad_output if g is not None])
                return None
            return hook
        module.register_full_backward_hook(make_hook(name))
```

### A.2.7 Layer 5: PyTorch Profiler（运行时校准）

**目标**：利用 Megatron-LM 已有的 profiler 集成，在单次迭代中采集每个算子的实际执行时间。

Megatron-LM `training.py` 已支持 `torch.profiler`：

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        # 注：本项目基于 CPU 环境运行，不使用 CUDA
        # torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=0, warmup=0, active=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiling"),
    record_shapes=True,      # 关键：记录 tensor shape
    with_stack=True,
    with_flops=True,
) as prof:
    loss = model(input_ids)
    loss.backward()
    optimizer.step()
    prof.step()

prof.export_chrome_trace("trace.json")
```

**关键收益**：
- `record_shapes=True` 可以验证 Hook 捕获的 shape 是否准确
- `with_flops=True` 可以获取 PyTorch 内置的 FLOPs 估算（作为对比基准）
- Chrome trace 可以可视化完整的执行时间线（包括通信与计算的重叠）

### A.2.8 参数化：从具体数值到符号公式

**核心问题**：捕获到的图中所有 shape 都是具体数值（如 `torch.Size([2048, 1, 4096])`），但我们需要将其转换为符号公式（如 `(seq_len, micro_batch_size, hidden_size)`）。

**解决方案——启发式 Shape 符号化推断**：

```python
from sympy import Symbol

class SymbolicShapeInferer:
    def __init__(self, config):
        self.config = config
        self.symbols = {
            "B": config.micro_batch_size,
            "S": config.seq_length,
            "H": config.hidden_size,
            "TP": config.tensor_parallel_size,
            "PP": config.pipeline_model_parallel_size,
            "DP": config.data_parallel_size,
            "V": config.vocab_size,
            "F": config.ffn_hidden_size,
            "NH": config.num_attention_heads,
        }
        self.symbol_objs = {k: Symbol(k) for k in self.symbols}

    def infer_shape(self, concrete_shape, module_type, param_shapes):
        symbolic = []
        for dim_val in concrete_shape:
            sym = self._match_dimension(dim_val, module_type, param_shapes)
            symbolic.append(sym)
        return tuple(symbolic)

    def _match_dimension(self, val, module_type, param_shapes):
        if val == self.config.micro_batch_size:
            return self.symbol_objs["B"]
        elif val == self.config.seq_length:
            return self.symbol_objs["S"]
        elif val == self.config.hidden_size:
            return self.symbol_objs["H"]
        elif val == self.config.vocab_size:
            return self.symbol_objs["V"]
        elif val == self.config.ffn_hidden_size:
            return self.symbol_objs["F"]
        elif val == self.config.hidden_size // self.config.tensor_parallel_size:
            return self.symbol_objs["H"] / self.symbol_objs["TP"]
        elif val == self.config.ffn_hidden_size // self.config.tensor_parallel_size:
            return self.symbol_objs["F"] / self.symbol_objs["TP"]
        elif val == self.config.vocab_size // self.config.tensor_parallel_size:
            return self.symbol_objs["V"] / self.symbol_objs["TP"]
        elif val == self.config.num_attention_heads // self.config.tensor_parallel_size:
            return self.symbol_objs["NH"] / self.symbol_objs["TP"]
        return val
```

**Shape 符号化规则库**（基于 Megatron-LM 代码分析）：

| 模块类型 | 输入 Shape | 输出 Shape | 参数 Shape |
|---------|-----------|-----------|-----------|
| `ColumnParallelLinear` | `[S, B, H]` | `[S, B, F/TP]` | `[F/TP, H]` |
| `RowParallelLinear` | `[S, B, F/TP]` | `[S, B, H]` | `[H, F/TP]` |
| `ParallelAttention` | `[S, B, H]` | `[S, B, H]` | QKV: `[3*H/TP, H]`, Proj: `[H, H/TP]` |
| `ParallelMLP` | `[S, B, H]` | `[S, B, H]` | Gate/Up: `[2*F/TP, H]`, Down: `[H, F/TP]` |
| `VocabParallelEmbedding` | `[S, B]` | `[S, B, H]` | `[V/TP, H]` |
| `LayerNorm` | `[S, B, H]` | `[S, B, H]` | `[H]`, `[H]` |

### A.2.9 通信量与计算量的符号公式

**通信量公式**（自动从图中推导）：

| 通信类型 | 触发条件 | 数据量公式 | 说明 |
|---------|---------|-----------|------|
| TP AllReduce | Attention/MLP output 聚合 | `2 * B * S * H * dtype_size` | 每个 TP 组内聚合 |
| TP AllGather | RowParallelLinear input gather | `B * S * H * dtype_size` | 收集被 TP 切分的输入 |
| TP ReduceScatter | Sequence Parallel grad | `B * S * H * dtype_size` | 分散被 SP 切分的梯度 |
| DP AllReduce/ZeRO | 梯度同步 | `num_params * dtype_size` | 跨所有 DP  rank |
| PP P2P Send | 阶段间激活值传递 | `B * S * H * dtype_size` | forward: stage N -> N+1 |
| PP P2P Recv | 阶段间梯度传递 | `B * S * H * dtype_size` | backward: stage N -> N-1 |
| EP All2All | MoE 专家分发 | `B * S * H * topk * dtype_size` | token 路由到专家 rank |

**计算量公式**：

| 算子 | FLOPs 公式 | 说明 |
|------|-----------|------|
| QKV Projection | `6 * B * S * H * (H/TP)` | 3 个线性投影 |
| Attention Score | `2 * B * S^2 * H` | Q*K^T |
| Attention Output | `2 * B * S^2 * H` | Softmax(QK^T)*V |
| Output Projection | `2 * B * S * H * (H/TP)` | Attention 后线性层 |
| MLP fc1 | `2 * B * S * H * (F/TP) * gate_factor` | gate_factor=2 表示 gated activation |
| MLP fc2 | `2 * B * S * (F/TP) * H` | 投影回 hidden_size |
| LayerNorm | `5 * B * S * H` | 内存带宽受限 |
| Loss/CE | `2 * B * S * H * V` | Vocab projection + cross entropy |

---

## A.3 功能2：配置修改后的图重评估

### A.3.1 核心机制

参数化图中所有数值都是 sympy 符号表达式，修改配置后重新代入求值：

```python
def reevaluate_graph(graph, new_config):
    subs = {
        Symbol("B"): new_config.micro_batch_size,
        Symbol("S"): new_config.seq_length,
        Symbol("H"): new_config.hidden_size,
        Symbol("TP"): new_config.tensor_parallel_size,
        Symbol("PP"): new_config.pipeline_model_parallel_size,
        Symbol("DP"): new_config.data_parallel_size,
        Symbol("V"): new_config.vocab_size,
        Symbol("F"): new_config.ffn_hidden_size,
    }
    
    for node in graph.nodes:
        if hasattr(node, "symbolic_shapes"):
            node.evaluated_shapes = tuple(
                s.subs(subs) if hasattr(s, "subs") else s 
                for s in node.symbolic_shapes
            )
        if hasattr(node, "symbolic_comm_bytes"):
            node.evaluated_comm_bytes = node.symbolic_comm_bytes.subs(subs)
        if hasattr(node, "symbolic_flops"):
            node.evaluated_flops = node.symbolic_flops.subs(subs)
    
    # 更新 OOM 判断
    for node in graph.nodes:
        if node.type == "memory_peak":
            peak_mem = node.symbolic_peak_mem.subs(subs)
            node.will_oom = (peak_mem > new_config.gpu_memory_gb * 1e9)
```

### A.3.2 支持修改的配置维度

| 配置项 | 影响范围 | 重评估方式 |
|--------|---------|-----------|
| `tensor_parallel_size` | 所有 Linear 参数 shape、TP 通信量、Attention FLOPs | 直接代入新 TP 值，重新计算所有依赖表达式 |
| `pipeline_parallel_size` | 层分配、Pipeline Bubble、P2P 通信次数 | 重新计算每 stage 层数、Bubble 时间 |
| `data_parallel_size` | 梯度同步通信量、每 rank batch size | 调整 DP AllReduce 数据量、微批次数量 |
| `micro_batch_size` | 所有激活值 shape、所有 FLOPs、所有通信量 | 全局变量 B，影响所有节点 |
| `seq_length` | Attention FLOPs（S^2 项）、所有激活值 | 全局变量 S，影响所有节点 |
| `hidden_size` | 所有 Linear 维度、LayerNorm 维度 | 全局变量 H，影响所有节点 |
| `num_layers` | 总层数、总 FLOPs、Pipeline stage 层数 | 需要重新构建层结构（图结构变化） |
| `recompute_granularity` | 激活值缓存峰值 | 调整 Memory Model 中的缓存策略 |

### A.3.3 示例：TP 从 2 改到 4 后的自动更新

原始配置：TP=2, H=4096 -> ColumnParallelLinear 权重 shape = `[F/2, H]` = `[8192, 4096]`

修改后：TP=4, H=4096 -> 自动更新为 `[F/4, H]` = `[4096, 4096]`

同时更新：
- TP AllReduce 通信量不变（与 TP 无关，只与激活值 shape 有关）
- 但 AllReduce 的通信组大小从 2 变为 4
- 每 rank 的计算量减少一半（因为权重切分更细）
- 通信开销相对增加（更多 rank 参与 AllReduce）

---

## A.4 功能3：虚拟执行与性能统计

### A.4.1 虚拟执行引擎架构

```
参数化计算图（符号表达式）
    |
    v
[配置代入] -> 得到具体数值的图
    |
    v
[计算节点估算] -> Roofline Model 估算执行时间
    |
    v
[通信节点估算] -> 带宽/延迟模型 估算通信时间
    |
    v
[调度模拟] -> 事件驱动调度器模拟执行时序
    |
    v
[内存跟踪] -> 模拟显存分配/释放，记录峰值
    |
    v
[性能报告] -> 迭代时间 / MFU / 显存 / 瓶颈
```

### A.4.2 计算节点性能估算（Roofline Model）

```python
def estimate_compute_time(flops, mem_bytes, system_config, op_type):
    compute_time = flops / (system_config.peak_tflops * 1e12 
                            * system_config.compute_efficiency)
    mem_time = mem_bytes / (system_config.memory_bw_gbps * 1e9 
                            * system_config.mem_efficiency)
    
    # Roofline: 取计算时间和内存时间的较大值
    total_time = max(compute_time, mem_time)
    
    # 记录瓶颈类型（用于分析报告）
    bottleneck = "compute" if compute_time > mem_time else "memory"
    return total_time, bottleneck
```

**关键算子的内存访问量估算**：

| 算子 | 内存访问量 (bytes) | 说明 |
|------|-------------------|------|
| GEMM (B,S,H)x(H,F/TP) | `2 * B * S * H + 2 * H * F/TP + 2 * B * S * F/TP` | 读输入、读权重、写输出 |
| Attention Score | `4 * B * S^2 * H` | Q,K,V 读 + Attention 矩阵读写 |
| FlashAttention | `~2.5 * B * S^2 * H` | 融合后内存访问减少 |
| LayerNorm | `4 * B * S * H` | 读输入、读 gamma/beta、写输出 |

### A.4.3 通信节点性能估算

```python
def estimate_comm_time(comm_bytes, comm_type, comm_size, system_config):
    # Ring AllReduce 有效数据量: 2*(N-1)/N * data_size
    if comm_type == "all_reduce":
        effective_data = 2 * (comm_size - 1) / comm_size * comm_bytes
    elif comm_type in ["all_gather", "reduce_scatter"]:
        effective_data = (comm_size - 1) / comm_size * comm_bytes
    elif comm_type == "p2p":
        effective_data = comm_bytes
    elif comm_type == "all_to_all":
        effective_data = (comm_size - 1) / comm_size * comm_bytes
    
    # 区分节点内(NVLink)和节点间(IB)
    if is_intra_node(comm_size, system_config.gpus_per_node):
        bandwidth = system_config.nvlink_bw_gbps
    else:
        bandwidth = system_config.ib_bw_gbps
    
    return effective_data / (bandwidth * 1e9 * system_config.comm_efficiency) \
           + system_config.latency_us / 1e6
```

### A.4.4 内存跟踪机制

模拟显存分配/释放过程，精确跟踪峰值：

```python
class MemoryTracker:
    def __init__(self):
        self.allocations = {}  # tensor_id -> (size, allocated_at, freed_at)
        self.current_mem = 0
        self.peak_mem = 0
        self.timeline = []     # (timestep, delta_mem, current_mem)
    
    def allocate(self, tensor_id, size, timestep):
        self.allocations[tensor_id] = (size, timestep, None)
        self.current_mem += size
        self.peak_mem = max(self.peak_mem, self.current_mem)
        self.timeline.append((timestep, size, self.current_mem))
    
    def free(self, tensor_id, timestep):
        if tensor_id in self.allocations:
            size, _, _ = self.allocations[tensor_id]
            self.allocations[tensor_id] = (size, None, timestep)
            self.current_mem -= size
            self.timeline.append((timestep, -size, self.current_mem))
```

**显存组成估算**：

```
峰值显存 = 模型参数 + 梯度 + 优化器状态 + 激活值峰值 + 临时缓冲区

模型参数: num_params * dtype_size (BF16=2 bytes)
梯度: num_params * dtype_size
优化器状态 (Adam): num_params * (4+4+4) = num_params * 12 bytes (fp32)
ZeRO-1 优化器状态: num_params * 12 / DP bytes
激活值峰值 (无 Recompute): B * S * H * num_layers * activation_factor
激活值峰值 (Full Recompute): 仅保存每层输入，反向时重算
激活值峰值 (PP 1F1B): 第一阶段缓存 PP_size 个 micro-batch
```

### A.4.5 Pipeline 调度模拟

**1F1B 调度虚拟执行**：

```python
def simulate_1f1b(graph, pp_size, num_micro_batches, fwd_time, bwd_time, pp_comm_time):
    """
    1F1B 调度时间线：
    - Warmup: (pp_size - 1) 个 micro-batch 只做 forward
    - Steady: 交替 forward/backward
    - Cooldown: (pp_size - 1) 个 micro-batch 只做 backward
    """
    chunk_time = fwd_time + bwd_time + pp_comm_time * 2
    
    # 含 Bubble 的总时间
    total_time = (num_micro_batches + pp_size - 1) * chunk_time
    bubble_time = (pp_size - 1) * chunk_time
    
    return {
        "total_time": total_time,
        "bubble_time": bubble_time,
        "useful_time": num_micro_batches * chunk_time,
        "bubble_ratio": bubble_time / total_time
    }
```

### A.4.6 输出性能报告

虚拟执行完成后，生成结构化性能报告：

```python
class PerformanceReport:
    def __init__(self):
        self.iteration_time_ms = 0
        self.peak_memory_gb = 0
        self.mfu = 0.0          # Model FLOPs Utilization
        self.tgs = 0.0          # tokens/sec/device
        self.breakdown = {}     # 时间分解
        self.bottleneck = ""    # 主要瓶颈
    
    def generate(self, graph, config, system):
        # 1. 统计总迭代时间
        self.iteration_time_ms = sum(node.evaluated_time for node in graph.nodes)
        
        # 2. 统计显存峰值
        self.peak_memory_gb = max(node.peak_mem for node in graph.nodes) / 1e9
        
        # 3. 计算 MFU
        total_flops = sum(node.evaluated_flops for node in graph.nodes if node.type == "compute")
        peak_flops = system.peak_tflops * 1e12 * config.world_size
        self.mfu = total_flops / (self.iteration_time_ms / 1000 * peak_flops)
        
        # 4. 计算吞吐量
        self.tgs = (config.global_batch_size * config.seq_length) \
                   / (self.iteration_time_ms / 1000 * config.world_size)
        
        # 5. 时间分解
        self.breakdown = {
            "fwd_compute": sum(n.evaluated_time for n in graph.nodes if n.op_type == "fwd_compute"),
            "bwd_compute": sum(n.evaluated_time for n in graph.nodes if n.op_type == "bwd_compute"),
            "tp_comm": sum(n.evaluated_time for n in graph.nodes if n.comm_group == "tp"),
            "dp_comm": sum(n.evaluated_time for n in graph.nodes if n.comm_group == "dp"),
            "pp_comm": sum(n.evaluated_time for n in graph.nodes if n.comm_group == "pp"),
            "bubble": graph.pipeline_bubble_time,
            "optimizer": sum(n.evaluated_time for n in graph.nodes if n.op_type == "optimizer"),
        }
        
        # 6. 瓶颈识别
        self.bottleneck = max(self.breakdown, key=self.breakdown.get)
```

---

## A.5 与 Phase 1 静态仿真的关系

### A.5.1 能力对比

| 维度 | Phase 1 静态分析 | 参数化计算图方案 |
|------|----------------|----------------|
| 图结构来源 | 手工建模（参考 Megatron 论文公式） | 从真实代码自动捕获 |
| 通信算子位置 | 手工推导（基于并行策略理论） | 通过 Monkey-patch 精确捕获 |
| 反向传播结构 | 假设 forward=1x, backward=2x FLOPs | 真实捕获 backward 图结构 |
| 新模型适配 | 需要为新模型手工编写仿真逻辑 | 运行一次训练自动捕获 |
| 融合算子处理 | 使用理论 FLOPs 估算 | 基于真实 kernel 时间（经 Trace 校准） |
| 适用场景 | 快速原型验证、大规模策略搜索 | 精确分析、生产环境调优 |
| 运行依赖 | 纯 CPU，无需 GPU | 需要一次真实 CPU 训练用于捕获 |
| 开发复杂度 | 中等（需要理解并手工建模） | 较高（需要实现多层 Hook 和符号化） |

### A.5.2 互补关系

两个方案并非替代关系，而是 **互补协作**：

```
                    用户请求性能分析
                           |
           +---------------+---------------+
           |                               |
           v                               v
    [Phase 1 静态仿真]              [参数化计算图]
    - 纯 CPU 运行                    - 需要一次 CPU 捕获
    - 秒级响应                       - 分钟级响应
    - 误差 10-50%                    - 误差 5-10%（经校准后）
           |                               |
           v                               v
    快速筛选候选配置                 精确评估 Top-K 配置
    （如：1000 种策略）              （如：10 种最优策略）
           |                               |
           +---------------+---------------+
                           |
                           v
                    [最终推荐配置]
```

**协作流程**：
1. **粗筛**：使用 Phase 1 静态仿真在 CPU 上快速评估大量候选配置（秒级）
2. **精选**：对 Top-10 候选配置，使用参数化计算图进行精确评估（分钟级）
3. **验证**：在 CPU 环境下运行 Top-3 配置，对比实际训练性能与仿真结果，验证仿真精度

### A.5.3 建议实施路径

| 阶段 | 目标 | 时间 | 产出 |
|------|------|------|------|
| Step 1 | 实现 Phase 1 静态仿真核心 | 2-3 周 | 可运行的 CPU 性能估算器 |
| Step 2 | 在静态仿真验证通过后，叠加 Layer 1+2 Hook | 1-2 周 | 可捕获计算图 + 通信图 |
| Step 3 | 叠加 Layer 3+4+5，完成全量捕获 | 1-2 周 | 含反向传播 + Trace 校准的完整图 |
| Step 4 | 实现符号化和虚拟执行引擎 | 2-3 周 | What-if 分析能力 |
| Step 5 | 与静态仿真对比验证 | 1 周 | 精度对比报告 |

---

## A.6 下一步行动建议

### A.6.1 短期目标（1-2 周）

1. **实现 Layer 1 + Layer 2 Hook**
   - 在现有 Qwen3 0.6B CPU 训练环境上验证能否捕获完整图结构
   - 验证 Module Hook 能否正确识别 `ColumnParallelLinear`、`RowParallelLinear` 等 Megatron 特定模块
   - 验证 Communication Hook 能否捕获 `all_reduce`、`all_gather` 等通信原语

2. **输出验证**
   - 导出捕获的图结构（JSON 格式）
   - 人工检查图的完整性（节点数量、边关系、shape 是否正确）

### A.6.2 中期目标（2-4 周）

1. **叠加 Layer 3/4/5**
   - 捕获自定义 Autograd Function 的 forward/backward
   - 注册 Backward Hook 捕获标准算子反向传播
   - 使用 PyTorch Profiler 采集单次迭代的运行时数据

2. **符号化推断验证**
   - 验证 Shape 符号化推断的准确性（与真实 shape 对比）
   - 验证通信量公式与真实通信量的一致性

3. **虚拟执行引擎初版**
   - 实现 Roofline Model 计算时间估算
   - 实现带宽模型通信时间估算
   - 输出迭代时间、显存峰值的估算值

### A.6.3 长期目标（1-2 个月）

1. **What-if 分析**
   - 支持修改 TP/PP/DP/BS 后自动重评估
   - 支持 OOM 预判
   - 支持策略搜索（Grid Search）

2. **与 Phase 1 静态仿真集成**
   - 静态仿真用于粗筛，参数化图用于精选
   - 统一输出格式和报告生成

3. **多模型支持**
   - 利用 Pai-Patch 的 40+ 模型生态
   - 为每个模型运行一次捕获，建立模型库

### A.6.4 关键风险与应对

| 风险 | 影响 | 应对措施 |
|------|------|---------|
| CPU 环境使用 gloo 后端通信 | Communication Hook 捕获不到 NCCL 调用（CPU 无 NCCL） | 使用 gloo 后端模拟分布式通信行为；通过 Megatron 代码分析推导 NCCL 通信模式，在仿真层统一转换为通信事件 |
| 融合算子（FlashAttention）shape 不可见 | 注意力模块内部结构缺失 | 使用 PyTorch Profiler 的 `record_shapes=True` 捕获子 kernel 的 shape；或参考 Megatron 代码手工补充 |
| Pipeline Parallel 调度复杂 | 1F1B / Interleaved 调度逻辑分散 | 优先支持非 Pipeline 场景（TP+DP），逐步叠加 PP 支持 |
| 符号化推断错误 | What-if 分析结果不准确 | 建立符号化规则单元测试；使用已知配置验证推断结果 |

---

*本附录基于对 SimAI AICB、Lumos、STAGE、PyTorch FX、torch.export、Compiled Autograd 的深入研究，以及对 Megatron-LM-20250707 和 Pai-Megatron-Patch-12.0 代码的详细分析整理生成。*
