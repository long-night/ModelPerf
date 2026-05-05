# DistSim: A performance model of large-scale hybrid distributed DNN training - 技术分析文档

**原始论文**: Guandong Lu, et al. *DistSim: A performance model of large-scale hybrid distributed DNN training*. 20th ACM International Conference on Computing Frontiers (CF '23), May 9-11, 2023, Bologna, Italy. https://doi.org/10.1145/3587135.3592200

**机构**: 上海交通大学、上海期智研究院、华为技术有限公司

---

## 1. 概述 (Overview)

### 1.1 背景与动机

随着深度学习模型规模的指数级增长（从 BERT-Large 的 3.4 亿参数到 GPT-3 的 1750 亿参数），单机训练已无法满足需求，**分布式训练**成为必然选择。当前主流方案是结合数据并行（Data Parallelism）、模型并行（Model Parallelism）和流水线并行（Pipeline Parallelism）的**混合并行策略（Hybrid Parallelism）**。

然而，评估混合并行策略面临两大核心挑战：
- ** profiling 成本高昂**：在 2048 GPU 的大规模集群上直接 profile，每小时成本高达 7,168 美元（AWS 3.5 USD/GPU/小时）
- **依赖关系复杂**：不同并行策略之间存在层次化的数据依赖，现有工具（如 Daydream、dPRO）仅支持数据并行，无法处理混合并行

### 1.2 DistSim 核心思想

DistSim 通过两个关键洞察解决上述问题：

| 洞察 | 说明 | 收益 |
|------|------|------|
| **Profiling 冗余性** | 数据并行中各副本执行相同计算；流水线并行中不同 micro-batch 的计算相同；相同通信可被复用 | 将 profiling 规模从大规模集群缩减至 **2 个节点** |
| **层次化依赖** | 数据并行关注权重同步；流水线并行关注层间激活传递；模型并行关注层内张量分布 | 各策略独立建模，再逐层组合 |

---

## 2. 系统架构 (System Architecture)

### 2.1 整体工作流程

```
输入: 模型 + 并行策略配置 (MP/PP/DP)
  ↓
[事件生成器] 提取子模型中的计算/通信算子 → 生成 Event Set
  ↓
[事件 Profiler] 在 2 节点上 profile 所有事件耗时
  ↓
[层次化建模引擎]
  ├─ 模型并行建模 → 层到事件的映射
  ├─ 流水线并行建模 → 构建 micro-batch 调度时序
  └─ 数据并行建模 → 复制事件列表 + All-Reduce 通信
  ↓
输出: 每设备的详细执行时间线 (Timeline)
```

### 2.2 核心组件详解

#### 2.2.1 事件生成器 (Event Generator)

- 复用 PyTorch Distributed 的模型切分能力，获取每设备的 sub-model
- 解析所有计算算子（如 GEMM、LayerNorm）和通信算子（All-Reduce、P2P）
- **去重策略**：通过算子名称、参数、输入形状唯一标识事件
- 通信事件额外标记 **intra-node / inter-node** 属性（通过 rank 是否同节点判断）

**事件分类**：
| 类型 | 子类型 | 典型场景 |
|------|--------|----------|
| 计算事件 | - | Forward / Backward 计算 |
| 通信事件 | Point-to-Point | 流水线 stage 间激活传递 |
| 通信事件 | All-Reduce | 数据并行梯度同步、模型并行输出聚合 |

#### 2.2.2 事件 Profiler

**计算事件**：
- 使用 CUPTI 等工具直接 profile 单设备执行时间

**P2P 通信事件**：
- 关键挑战：存在**队列等待时间**（Queuing Time），发送方和接收方必须同时就绪才能开始传输
- 解决方案：同时 profile SEND 和 RECV，取两者耗时的 **最小值** 作为实际传输时间

**All-Reduce 通信事件**：
- 当设备数 ≤ 8 时：直接执行并 profile
- 当设备数 > 8 时：基于 Ring-AllReduce 理论公式推导
  - 传输量公式：`2(N-1) × P/N` （P 为张量大小，N 为设备数）
  - 实测对迭代时间预测的影响 < 2%

#### 2.2.3 层次化建模 (Hierarchical Modeling)

建模顺序：**模型并行 → 流水线并行 → 数据并行**，逐层扩展设备维度。

**1) 模型并行建模**
- 输入：事件集合 + MP size
- 输出：层到事件的映射 `MAP(L → E)`
- MP=1：单层映射到单一计算事件
- MP>1：层映射到复合事件（每设备包含计算事件 + All-Reduce 通信事件）

**2) 流水线并行建模**

核心算法（Pipeline Training Modeling Algorithm）：

```python
def pipeline_modeling(schedule S, layer_map MAP, mp_size MP):
    F = initialize_event_lists(D × MP)
    while S not empty:
        # 找到 schedule 中第一个可用 stage
        stage st(d, l, m), timestamp t = first_available(S)
        e = MAP[l]                              # 获取对应事件
        e_comm = get_comm_event(st)              # stage 间 P2P 通信事件
        for mp in range(1, MP+1):
            F[d × MP + mp].add(e, t, m)        # 添加计算事件
            F[d × MP + mp].add(e_comm, t + e.elapsed, m)  # 添加通信事件
        S.remove(st)
    return F
```

- 支持 GPipe 和 Dapple 两种同步流水线调度算法
- 输出：MP × PP 个设备的事件列表

**3) 数据并行建模**
- 将事件列表从 `MP × PP` 设备复制扩展到 `MP × PP × DP` 设备
- 在每个事件列表末尾根据梯度大小添加 **All-Reduce 通信事件**

---

## 3. 核心创新点 (Key Contributions)

### 3.1 冗余消除机制

DistSim 识别并消除四类 profiling 冗余：

| 冗余类型 | 描述 | 示例 |
|---------|------|------|
| Intra-model (①) | 同一设备在流水线中重复计算不同 micro-batch 的同一层 | Micro-batch 1/2/3 都计算 Layer1 |
| Inter-model (②) | 数据并行中不同 rank 执行完全相同的计算 | DP rank 0 和 rank 1 的 forward 相同 |
| 通信形状冗余 (③) | 相同形状的激活在 stage 间传递 | 所有 micro-batch 的激活张量形状相同 |
| 通信权重冗余 (④) | 相同大小的权重梯度需要同步 | 各层 weight All-Reduce 大小固定 |

### 3.2 层次化依赖处理

不同并行策略的依赖域相互独立：
- **模型并行**：处理层内（intra-layer）依赖 → 张量切分与聚合
- **流水线并行**：处理层间（inter-layer）依赖 → 激活传递与 micro-batch 调度
- **数据并行**：处理模型间（inter-model）依赖 → 权重同步

关键洞察：**修改模型并行的切分大小不会影响流水线并行的依赖关系**，因为流水线并行不关心层内的计算细节。

---

## 4. 实验评估 (Evaluation)

### 4.1 实验环境

| 项目 | 配置 |
|------|------|
| 集群 | 最多 16 × NVIDIA A40 GPU，分布于 4 台服务器 |
| 框架 | PyTorch Distributed + CUDA 11.6 + NCCL |
| 模型 | BERT-Large, GPT-2-345M, T5 |
| 策略标记 | `xMxPxD` = MP size × PP size × DP size |

### 4.2 整体精度评估 (Batch-Time / Iteration Time)

DistSim 在各种混合并行策略下的预测误差：

| 模型 | GPU 数 | 策略数 | 最大误差 | 与启发式方法对比 |
|------|--------|--------|----------|-----------------|
| BERT-Large | 4/8/16 | 10+ | **3.51%** | 启发式方法最高 40.4% |
| GPT-2-345M | 4/8/16 | 10+ | < 4% | - |
| T5 | 4/8/16 | 10+ | < 4% | - |

关键发现：误差与 GPU 数量无强相关性，4 GPU 配置误差（2.45%）可能高于 16 GPU（1.63%），主要来源于 100 次迭代的随机波动。

### 4.3 单 GPU 活动精度评估

- **最大误差**：4.19%
- 部分 GPU（如 GPT-2 2M2P4D 中的 GPU 9-12）误差高于平均，原因：
  1. Profile 波动在单个设备上累积并传播
  2. 使用 rank 0 时钟作为全局标准引入时间对齐问题
  3. 数据并行 rank 间在 All-Reduce 前无同步，存在时序差异
- **规律**：流水线并行规模越大，误差有增大趋势（早期 stage 误差向后传播）

### 4.4 逐 Stage 精度评估

在 `2M4P1D` 配置、micro-batch=4 下：
- 共 32 个 forward/backward stage，每 GPU 4 个
- **最大中位误差**：1.71%
- 同一模型并行组的 GPU（如 0&1, 2&3）误差分布基本一致，验证模型并行各部分计算可视为相同

### 4.5 大规模泛化评估

| 项目 | 配置 |
|------|------|
| 模型 | 145B 参数 GPT |
| 集群规模 | 128 GPU |
| 并行策略 | 8M16P1D (Megatron-LM 配置) |
| 对比基准 | Megatron-LM 报告数据 |
| 结论 | DistSim 与 Megatron-LM 的吞吐量增长趋势高度一致，验证大规模建模能力 |

---

## 5. 应用案例：自动并行策略搜索 (Use Case)

### 5.1 场景设定

- **目标模型**：BERT-exLarge（48 层 Transformer，未见过的模型）
- **硬件**：4 节点 × 16 × A10 GPU
- **约束**：全局 batch size = 16
- **搜索空间**：MP ∈ {1,2,4,8,16}，PP ∈ {1,2,4,8,16}，DP = 16/(MP×PP)，共 15 种有效配置

### 5.2 搜索结果

通过 DistSim 网格搜索：

| 指标 | DistSim 预测 | 实际测量 |
|------|-------------|----------|
| 最优策略 | 2M1P8D → 2.94 iter/s | 2M1P8D → 2.97 iter/s |
| 次优策略 | 2.92 iter/s | 2.90 iter/s |
| 最差策略 | 0.398 iter/s | 0.396 iter/s |
| 最优 vs 最差加速比 | **7.379×** | **7.488×** |

### 5.3 时间成本对比

| 环节 | DistSim | 直接运行 |
|------|---------|----------|
| 模拟耗时 | 0.14 s | - |
| GPU Profile 耗时 (gpu×s) | 49.18 | 380.35 |
| 相对比例 | **0.1296× (12.96%)** | 1× |

- 模拟时间占总时间 < 1%，主要开销在 profiling
- 随着规模增大，冗余消除收益更高（All-Reduce 的冗余消除比例随规模增加）

---

## 6. 与其他方法的对比 (Comparison)

| 方法 | 代表工作 | 数据并行 | 模型/流水线并行 | 低开销 | 精度 |
|------|---------|---------|----------------|--------|------|
| 直接 Profile | Megatron-LM | ✅ | ✅ | ❌ | ✅ |
| 解析模型 | DistIR, AccPar | ✅ | ✅ | ✅ | ❌ (平均 26.1%-40.4% 误差) |
| 仿真器 | Daydream, dPRO | ✅ | ❌ | ✅ | ✅ (仅 DP) |
| **DistSim (本文)** | - | ✅ | ✅ | ✅ | ✅ (<4% 误差) |

---

## 7. 可扩展性讨论 (Scalability)

| 新场景 | 支持方式 |
|--------|---------|
| **新策略** (ZeRO-DP, 3D-Parallelism) | 识别其依赖模式（层内 + 模型间），按层次分解后生成事件 |
| **新算法** (异步流水线如 PipeDream) | 移除全局同步事件，调整 pipeline schedule 即可建模 |
| **新算子/模型** | 视为新事件类型，正常 profile 和建模 |
| **无 profiling 设备** | 可接入 MGPUSim 或 Habitat 等预测器替代实际 profile（牺牲部分精度） |

---

## 8. 技术启示与总结

### 8.1 关键设计原则

1. **抽象分层**：通过 Event → Composed-Event → Timeline 的层次化抽象，将复杂分布式训练建模问题解耦
2. **冗余识别**：发现并利用"相同计算/通信在不同设备/micro-batch 间重复"这一特性，实现 O(1) 规模的 profile
3. **依赖隔离**：不同并行策略的依赖域正交，可独立建模后组合

### 8.2 适用场景

- **大规模集群策略预研**：在小型集群上评估数百种策略，再在最优策略上投入大规模资源
- **自动并行搜索**：作为成本模型驱动策略搜索（如 Piper、Alpa 的补充）
- **设备利用率分析**：精确定位流水线 bubble 和空闲时间，辅助故障恢复和计算填充

### 8.3 局限性

1. 当前仅支持**同步流水线并行**，异步流水线（如 PipeDream）需额外处理权重版本问题
2. 假设**同构设备**和**无网络层次结构**（如胖树拓扑），异构场景需扩展
3. 使用 rank 0 时钟作为全局标准存在时间对齐误差

---

## 9. 核心公式汇总

| 公式 | 用途 |
|------|------|
| All-Reduce 传输量：`2(N-1) × P/N` | 大设备数 All-Reduce 时间推导 |
| P2P 通信时间：`min(T_send, T_recv)` | 消除队列等待影响 |
| 设备总数：`D_total = MP × PP × DP` | 策略配置关系 |
| 最优策略搜索加速比：`throughput_best / throughput_worst` | 策略优化收益评估 |

---

*本文档基于 CF '23 会议论文《DistSim: A performance model of large-scale hybrid distributed DNN training》分析整理。*
