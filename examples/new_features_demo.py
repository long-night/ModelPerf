import tempfile
import torch
import torch.nn as nn

from modelperf.capture.graph import ComputationalGraph, GraphNode, OpType
from modelperf.capture.hierarchy import (
    infer_module_category,
    build_hierarchy_paths,
    get_hierarchy_summary,
    get_nodes_under_hierarchy,
)
from modelperf.capture.coordinator import CaptureCoordinator
from modelperf.testing.op_benchmark import OpBenchmark, OpAdapter
from modelperf.testing.op_test_generator import OpTestGenerator


class DemoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(576, 576),
            nn.Softmax(dim=-1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(576, 1536),
            nn.GELU(),
            nn.Linear(1536, 576),
        )
        self.norm = nn.LayerNorm(576)

    def forward(self, x):
        x = self.norm(x)
        x = self.attention(x) + x
        x = self.mlp(x) + x
        return x


def demo_csv_export():
    print("=" * 60)
    print("Demo 1: 计算图 CSV 导出")
    print("=" * 60)

    graph = ComputationalGraph()
    graph.add_node(
        node_id="emb", op_type="Embedding", node_type=OpType.COMPUTE,
        input_shapes=[(2, 128)], output_shapes=[(2, 128, 576)],
        params={"weight": (151936, 576)}, module_category="embedding",
    )
    graph.add_node(
        node_id="ln1", op_type="LayerNorm", node_type=OpType.COMPUTE,
        input_shapes=[(2, 128, 576)], output_shapes=[(2, 128, 576)],
        module_category="norm",
    )
    graph.add_node(
        node_id="qkv", op_type="Linear", node_type=OpType.COMPUTE,
        input_shapes=[(2, 128, 576)], output_shapes=[(2, 128, 1728)],
        params={"weight": (576, 1728)}, module_category="parallel_linear",
    )
    graph.add_node(
        node_id="comm1", op_type="all_reduce", node_type=OpType.COMMUNICATION,
        comm_size=8, comm_bytes=4194304, module_category="communication",
    )
    graph.add_edge("emb", "ln1")
    graph.add_edge("ln1", "qkv")
    graph.add_edge("qkv", "comm1")

    with tempfile.TemporaryDirectory() as tmpdir:
        graph.export_csv(tmpdir)
        import os
        print(f"CSV files exported to: {tmpdir}")
        for fname in os.listdir(tmpdir):
            fpath = os.path.join(tmpdir, fname)
            with open(fpath, "r") as f:
                lines = f.readlines()
                print(f"  {fname}: {len(lines)} lines")
    print()


def demo_hierarchy():
    print("=" * 60)
    print("Demo 2: 算子层级嵌套关系")
    print("=" * 60)

    graph = ComputationalGraph()
    nodes = [
        ("layer_0", "TransformerLayer", "model.layers.0"),
        ("ln_0", "LayerNorm", "model.layers.0.norm1"),
        ("qkv_0", "Linear", "model.layers.0.attention.q_proj"),
        ("attn_0", "SelfAttention", "model.layers.0.attention"),
        ("mlp_0", "MLP", "model.layers.0.mlp"),
        ("fc1_0", "Linear", "model.layers.0.mlp.fc1"),
        ("layer_1", "TransformerLayer", "model.layers.1"),
        ("ln_1", "LayerNorm", "model.layers.1.norm1"),
    ]
    for nid, op, path in nodes:
        graph.add_node(
            node_id=nid, op_type=op, node_type=OpType.COMPUTE,
            module_path=path, input_shapes=[(2, 128, 576)],
            output_shapes=[(2, 128, 576)],
        )

    build_hierarchy_paths(graph.nodes)

    print("层级树结构:")
    tree = graph.get_hierarchy_tree()
    def print_tree(d, indent=0):
        for k, v in d.items():
            if k.startswith("_"):
                continue
            print("  " * indent + f"- {k}")
            if isinstance(v, dict) and "_children" in v:
                print_tree(v["_children"], indent + 1)
                if v.get("_nodes"):
                    for nid in v["_nodes"]:
                        node = graph.nodes[nid]
                        print("  " * (indent + 1) + f"* [{nid}] {node.op_type} ({node.module_category})")
    print_tree(tree["root"])

    print("\n按类别统计:")
    summary = get_hierarchy_summary(graph.nodes)
    for cat, info in summary.items():
        print(f"  {cat}: {info['count']} nodes")

    print("\n查询 model.layers.0 下的所有节点:")
    sub_nodes = get_nodes_under_hierarchy(graph.nodes, "model.layers.0")
    for n in sub_nodes:
        print(f"  - {n.node_id}: {n.op_type} (level={n.module_level})")
    print()


def demo_capture_with_hierarchy():
    print("=" * 60)
    print("Demo 3: 真实模型捕获 + 层级关系")
    print("=" * 60)

    model = DemoModel()
    coord = CaptureCoordinator()
    coord.attach(model)
    coord.start()

    x = torch.randn(2, 128, 576)
    y = model(x)

    coord.stop()
    graph = coord.get_graph()
    build_hierarchy_paths(graph.nodes)

    print(f"Captured {len(graph.nodes)} nodes")
    for nid, node in graph.nodes.items():
        print(f"  [{nid}] {node.op_type:20s} path={node.module_path or 'N/A':30s} "
              f"cat={node.module_category:15s} parent={node.parent_module_id or 'N/A'}")
    print()


def demo_test_generation():
    print("=" * 60)
    print("Demo 4: 自动生成算子单元测试")
    print("=" * 60)

    graph = ComputationalGraph()
    graph.add_node(
        node_id="linear_1", op_type="Linear", node_type=OpType.COMPUTE,
        input_shapes=[(2, 128, 576)], output_shapes=[(2, 128, 1024)],
        params={"weight": (576, 1024)}, module_category="parallel_linear",
    )
    graph.add_node(
        node_id="layernorm_1", op_type="LayerNorm", node_type=OpType.COMPUTE,
        input_shapes=[(2, 128, 1024)], output_shapes=[(2, 128, 1024)],
        module_category="norm",
    )
    graph.add_node(
        node_id="comm_ar", op_type="all_reduce", node_type=OpType.COMMUNICATION,
        comm_size=8, comm_bytes=16777216, module_category="communication",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        gen = OpTestGenerator(output_dir=tmpdir)
        gen.generate_all(graph)

        import os
        print(f"Generated test files in: {tmpdir}")
        for fname in sorted(os.listdir(tmpdir)):
            fpath = os.path.join(tmpdir, fname)
            with open(fpath, "r") as f:
                content = f.read()
            print(f"\n--- {fname} ---")
            for line in content.split("\n")[:25]:
                print(line)
            if len(content.split("\n")) > 25:
                print("...")
    print()


def demo_benchmark():
    print("=" * 60)
    print("Demo 5: 算子性能测试 (CPU 模式)")
    print("=" * 60)

    node = GraphNode(
        node_id="test_linear", op_type="Linear", node_type=OpType.COMPUTE,
        input_shapes=[(4, 256, 576)], output_shapes=[(4, 256, 1024)],
        params={"weight": (576, 1024)},
    )

    benchmark = OpBenchmark(device="cpu", warmup=3, repeat=10)
    result = benchmark.benchmark_node(node)

    print(f"Op: {result.op_type}")
    print(f"  Correctness: {'PASS' if result.correctness_passed else 'FAIL'} "
          f"({result.correctness_error or 'N/A'})")
    print(f"  Avg Time: {result.avg_time_ms:.3f} ms")
    print(f"  FLOPs: {result.flops:.2e}")
    print(f"  FLOP/s: {result.flops_utilization:.3f} TFLOP/s")
    print(f"  Memory BW: {result.memory_bandwidth_gbs:.2f} GB/s")
    print(f"  Peak Memory: {result.peak_memory_mb:.2f} MB")
    print()


if __name__ == "__main__":
    demo_csv_export()
    demo_hierarchy()
    demo_capture_with_hierarchy()
    demo_test_generation()
    demo_benchmark()
    print("=" * 60)
    print("All demos completed!")
    print("=" * 60)
