

import json
import os
import sys


def load_captured_data():
    modelperf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_dir = os.path.join(modelperf_path, 'examples', 'output', 'captured_configs')
    graph_dir = os.path.join(modelperf_path, 'examples', 'output', 'captured_graph')

    data = {}

    for fname, key in [
        ('model_config.json', 'model_config'),
        ('strategy_config.json', 'strategy_config'),
        ('system_config.json', 'system_config'),
    ]:
        fpath = os.path.join(config_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                data[key] = json.load(f)
        else:
            print(f"[Warning] {fname} not found, using defaults")
            data[key] = {}

    graph_path = os.path.join(graph_dir, 'computational_graph.json')
    if os.path.exists(graph_path):
        with open(graph_path) as f:
            data['graph'] = json.load(f)
    else:
        data['graph'] = None

    return data


def build_graph_from_config(model_config, strategy_config):
    from modelperf.capture.graph import ComputationalGraph, GraphNode, OpType
    from modelperf.symbolic.shape_inferer import SymbolicShapeInferer

    graph = ComputationalGraph()
    graph.model_config = model_config
    graph.strategy_config = strategy_config

    inferer = SymbolicShapeInferer()
    B = inferer.get_symbol('B')
    S = inferer.get_symbol('S')
    H = inferer.get_symbol('H')

    batch = strategy_config.get('micro_batch_size', 1)
    seq = strategy_config.get('seq_length', 128)
    hidden = model_config.get('hidden_size', 1024)
    layers = model_config.get('num_layers', 4)
    ffn = model_config.get('ffn_hidden_size', 3072)
    vocab = model_config.get('vocab_size', 151936)
    if vocab <= 0:
        vocab = model_config.get('padded_vocab_size', 0)
    if vocab <= 0:
        vocab = 151936

    node_id = 0

    emb = graph.add_node(
        node_id=f"node_{node_id}", op_type="Embedding", node_type=OpType.COMPUTE,
        output_shapes=[(seq, batch, hidden)], symbolic_output_shapes=[(S, B, H)],
    )
    graph.forward_nodes.append(emb.node_id)
    node_id += 1

    for _ in range(layers):
        ops = [
            ("LayerNorm", (seq, batch, hidden), (S, B, H), {}, (seq, batch, hidden), (S, B, H)),
            ("Linear_qkv", (seq, batch, hidden * 3), (S, B, H * 3), {"weight": (hidden, hidden * 3)}, (seq, batch, hidden), (S, B, H)),
            ("SelfAttention", (seq, batch, hidden), (S, B, H), {}, (seq, batch, hidden * 3), (S, B, H * 3)),
            ("Linear_proj", (seq, batch, hidden), (S, B, H), {"weight": (hidden, hidden)}, (seq, batch, hidden), (S, B, H)),
            ("Linear_fc1", (seq, batch, ffn), (S, B, ffn), {"weight": (hidden, ffn)}, (seq, batch, hidden), (S, B, H)),
            ("Linear_fc2", (seq, batch, hidden), (S, B, H), {"weight": (ffn, hidden)}, (seq, batch, ffn), (S, B, ffn)),
        ]
        for op_type, out_shape, sym_shape, params, in_shape, in_sym in ops:
            node = graph.add_node(
                node_id=f"node_{node_id}", op_type=op_type, node_type=OpType.COMPUTE,
                input_shapes=[in_shape],
                output_shapes=[out_shape],
                symbolic_input_shapes=[in_sym],
                symbolic_output_shapes=[sym_shape],
                params=params,
            )
            graph.forward_nodes.append(node.node_id)
            node_id += 1

    lm_head = graph.add_node(
        node_id=f"node_{node_id}", op_type="Linear_lm_head", node_type=OpType.COMPUTE,
        input_shapes=[(seq, batch, hidden)],
        output_shapes=[(seq, batch, vocab)],
        symbolic_input_shapes=[(S, B, H)],
        symbolic_output_shapes=[(S, B, vocab)],
        params={"weight": (hidden, vocab)},
    )
    graph.forward_nodes.append(lm_head.node_id)

    for node_id_str in reversed(graph.forward_nodes):
        if node_id_str != emb.node_id:
            bwd = graph.add_node(
                node_id=f"{node_id_str}_bwd", op_type=f"{graph.nodes[node_id_str].op_type}_bwd",
                node_type=OpType.BACKWARD, output_shapes=graph.nodes[node_id_str].input_shapes,
            )
            graph.backward_nodes.append(bwd.node_id)

    return graph


def run_pipeline():
    print("=" * 60)
    print("ModelPerf End-to-End Pipeline (v2)")
    print("=" * 60)

    data = load_captured_data()
    model_config = data['model_config']
    strategy_config = data['strategy_config']
    system_config = data['system_config']
    graph_data = data['graph']

    print(f"\n[Step 1] Loaded captured configs")
    print(f"  Model config keys: {list(model_config.keys())[:5]}...")
    print(f"  Strategy config keys: {list(strategy_config.keys())[:5]}...")

    if graph_data:
        print(f"  Computational graph: {graph_data['summary']['total_nodes']} nodes, "
              f"{graph_data['summary']['comm_nodes']} comm, "
              f"{graph_data['summary']['compute_nodes']} compute")
    else:
        print("  No computational graph found, will build from config")

    print("\n[Step 2] Building simulation graph...")

    from modelperf.simulation.virtual_executor import VirtualExecutor
    from modelperf.simulation.execution_result import ExecutionConfig

    graph = build_graph_from_config(model_config, strategy_config)

    config = ExecutionConfig(
        gpu_type="a100",
        peak_compute_tflops=312.0,
        peak_bandwidth_gbs=2039.0,
        network_bandwidth_gbs=600.0,
        network_latency_us=2.0,
        memory_per_device_gb=80.0,
        tp_size=strategy_config.get('tensor_parallel_size', 1),
        pp_size=strategy_config.get('pipeline_parallel_size', 1),
        dp_size=strategy_config.get('data_parallel_size', 1),
        enable_overlap=True,
        pipeline_schedule="1f1b",
    )

    executor = VirtualExecutor(config=config)
    result = executor.execute(graph)

    print(f"\n[Step 3] Simulation results")
    print(f"  Iteration time: {result.iteration_time_ms:.2f} ms")
    print(f"  Forward time: {result.forward_time_ms:.2f} ms")
    print(f"  Backward time: {result.backward_time_ms:.2f} ms")
    print(f"  Compute time: {result.compute_time_ms:.2f} ms")
    print(f"  Comm time: {result.comm_time_ms:.2f} ms")
    print(f"  Peak memory: {result.peak_memory_mb:.2f} MB")
    print(f"  Bottleneck: {result.bottleneck}")
    print(f"  Throughput: {result.throughput_tokens_per_sec:.2f} tokens/sec")
    print(f"  Memory efficiency: {result.memory_efficiency * 100:.1f}%")

    print("\n[Step 4] What-if analysis: changing pipeline schedule to GPipe...")

    config_gpipe = ExecutionConfig(
        gpu_type="a100",
        peak_compute_tflops=312.0,
        peak_bandwidth_gbs=2039.0,
        network_bandwidth_gbs=600.0,
        network_latency_us=2.0,
        memory_per_device_gb=80.0,
        tp_size=strategy_config.get('tensor_parallel_size', 1),
        pp_size=strategy_config.get('pipeline_parallel_size', 1),
        dp_size=strategy_config.get('data_parallel_size', 1),
        enable_overlap=True,
        pipeline_schedule="gpipe",
    )
    executor_gpipe = VirtualExecutor(config=config_gpipe)
    result_gpipe = executor_gpipe.execute(graph)

    print(f"  GPipe iteration time: {result_gpipe.iteration_time_ms:.2f} ms")
    speedup = result_gpipe.iteration_time_ms / result.iteration_time_ms
    print(f"  1F1B vs GPipe ratio: {speedup:.2f}x")

    modelperf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    report_dir = os.path.join(modelperf_path, 'examples', 'output', 'reports')
    os.makedirs(report_dir, exist_ok=True)

    print("\n[Step 5] Generating report...")

    from modelperf.visualization.report_generator import generate_performance_report
    viz_report_path = generate_performance_report(result, report_dir)
    print(f"  Visualization report: {viz_report_path}")

    report = {
        "model_config": model_config,
        "strategy_config": strategy_config,
        "simulation": result.summary(),
        "compute_breakdown": result.compute_breakdown,
        "comm_breakdown": result.comm_breakdown,
        "memory_breakdown": result.memory_breakdown,
        "what_if": {
            "gpipe_iteration_time_ms": result_gpipe.iteration_time_ms,
            "1f1b_vs_gpipe_ratio": speedup,
        },
    }

    report_path = os.path.join(report_dir, 'end_to_end_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"  JSON report exported to {report_path}")

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
