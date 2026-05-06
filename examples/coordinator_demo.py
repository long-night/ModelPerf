import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modelperf.capture.coordinator import CaptureCoordinator
from modelperf.capture.graph import OpType
from modelperf.simulation.virtual_executor import VirtualExecutor
from modelperf.simulation.execution_result import ExecutionConfig
from modelperf.analysis.what_if import WhatIfAnalyzer


class DemoModel(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden * 2)
        self.fc2 = nn.Linear(hidden * 2, hidden)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


def run_demo():
    print("=" * 60)
    print("ModelPerf CaptureCoordinator Demo")
    print("=" * 60)

    model = DemoModel(hidden=256)
    x = torch.randn(4, 128, 256, requires_grad=True)

    coordinator = CaptureCoordinator()
    coordinator.attach(model)

    print("\n[1] Forward + Backward capture with 5-layer hooks")
    with coordinator:
        y = model(x)
        loss = y.sum()
        loss.backward()

    graph = coordinator.get_graph()
    print(f"    Captured: {len(graph.nodes)} total nodes")
    print(f"    Forward nodes: {len(graph.forward_nodes)}")
    print(f"    Backward nodes: {len(graph.backward_nodes)}")
    print(f"    Edges: {len(graph.edges)}")

    compute_nodes = [n for n in graph.nodes.values() if n.node_type == OpType.AUTOGRAD_FWD]
    print(f"    Compute nodes: {len(compute_nodes)}")

    if graph.edges:
        print(f"    Sample edge: {graph.edges[0]}")

    print("\n[2] Performance simulation")
    graph.model_config = {"hidden_size": 256, "num_layers": 2}
    graph.strategy_config = {"micro_batch_size": 4, "seq_length": 128, "tp_size": 1}

    config = ExecutionConfig(
        peak_compute_tflops=312.0,
        peak_bandwidth_gbs=2039.0,
        network_bandwidth_gbs=600.0,
        memory_per_device_gb=80.0,
        enable_overlap=True,
    )
    executor = VirtualExecutor(config=config)
    result = executor.execute(graph)

    print(f"    Iteration time: {result.iteration_time_ms:.2f} ms")
    print(f"    Throughput: {result.throughput_tokens_per_sec:.2f} tokens/sec")
    print(f"    Peak memory: {result.peak_memory_mb:.2f} MB")
    print(f"    Bottleneck: {result.bottleneck}")

    print("\n[3] What-if analysis: increase batch size")
    analyzer = WhatIfAnalyzer(graph)
    new_config = analyzer.modify_config({"micro_batch_size": 8})
    new_result = analyzer.re_evaluate(new_config)

    print(f"    New iteration time: {new_result.iteration_time_ms:.2f} ms")
    print(f"    New throughput: {new_result.throughput_tokens_per_sec:.2f} tokens/sec")
    print(f"    New peak memory: {new_result.peak_memory_mb:.2f} MB")

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
