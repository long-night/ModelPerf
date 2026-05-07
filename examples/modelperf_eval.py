#!/usr/bin/env python3
"""ModelPerf evaluation script for captured training configs.

Usage:
    python modelperf_eval.py --config-dir /path/to/captured_configs --graph-dir /path/to/captured_graph --output-dir /path/to/output
"""

import argparse
import json
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
modelperf_path = os.path.dirname(script_dir)
if modelperf_path not in sys.path:
    sys.path.insert(0, modelperf_path)


def load_captured_data(config_dir, graph_dir):
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
            print(f"[Warning] {fname} not found in {config_dir}, using defaults")
            data[key] = {}

    merged_graph = _load_and_merge_graphs(graph_dir)
    data['graph'] = merged_graph

    return data


def _load_and_merge_graphs(graph_dir):
    import glob
    from collections import defaultdict

    new_files = sorted(glob.glob(os.path.join(graph_dir, 'computational_graph_pp*_tp*_dp*.json')))
    legacy_rank_files = sorted(glob.glob(os.path.join(graph_dir, 'computational_graph_rank*.json')))
    legacy_file = os.path.join(graph_dir, 'computational_graph.json')

    graphs = []

    if new_files:
        for fpath in new_files:
            try:
                with open(fpath) as f:
                    graphs.append(json.load(f))
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[Warning] Failed to parse {os.path.basename(fpath)}: {e}")
    elif legacy_rank_files:
        for fpath in legacy_rank_files:
            try:
                with open(fpath) as f:
                    graphs.append(json.load(f))
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[Warning] Failed to parse {os.path.basename(fpath)}: {e}")
    elif os.path.exists(legacy_file):
        try:
            with open(legacy_file) as f:
                graphs.append(json.load(f))
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[Warning] Failed to parse computational_graph.json: {e}")

    if not graphs:
        print(f"[Warning] No computational graph files found in {graph_dir}")
        return None

    if len(graphs) == 1:
        return graphs[0]

    return _merge_hybrid_parallel_graphs(graphs)


def _merge_hybrid_parallel_graphs(graphs):
    from collections import defaultdict

    def get_parallel_id(g):
        pid = g.get("parallel_identity", {})
        return (
            pid.get("pp_rank", 0),
            pid.get("tp_rank", 0),
            pid.get("dp_rank", 0),
        )

    def merge_graphs(graph_list):
        merged = {
            "nodes": {},
            "edges": [],
            "model_config": graph_list[0].get("model_config", {}),
            "strategy_config": graph_list[0].get("strategy_config", {}),
            "system_config": graph_list[0].get("system_config", {}),
            "summary": {
                "total_nodes": 0,
                "compute_nodes": 0,
                "comm_nodes": 0,
                "forward_nodes": 0,
                "backward_nodes": 0,
                "edges": 0,
            },
        }
        seen_nodes = set()
        seen_edges = set()
        for g in graph_list:
            for node_id, node_data in g.get("nodes", {}).items():
                if node_id not in seen_nodes:
                    seen_nodes.add(node_id)
                    merged["nodes"][node_id] = node_data
            for edge in g.get("edges", []):
                edge_tuple = tuple(edge) if isinstance(edge, list) else edge
                if edge_tuple not in seen_edges:
                    seen_edges.add(edge_tuple)
                    merged["edges"].append(list(edge_tuple))
            gs = g.get("summary", {})
            s = merged["summary"]
            s["total_nodes"] += gs.get("total_nodes", 0)
            s["compute_nodes"] += gs.get("compute_nodes", 0)
            s["comm_nodes"] += gs.get("comm_nodes", 0)
            s["forward_nodes"] += gs.get("forward_nodes", 0)
            s["backward_nodes"] += gs.get("backward_nodes", 0)
            s["edges"] += gs.get("edges", 0)
        return merged

    print(f"[Step 1/3] Deduplicating DP replicas ({len(graphs)} total graphs)...")
    pp_tp_groups = defaultdict(list)
    for g in graphs:
        pp_rank, tp_rank, dp_rank = get_parallel_id(g)
        pp_tp_groups[(pp_rank, tp_rank)].append(g)

    deduped_graphs = []
    for key, group in pp_tp_groups.items():
        deduped = merge_graphs(group)
        deduped["parallel_identity"] = {
            "pp_rank": key[0],
            "tp_rank": key[1],
            "dp_rank": 0,
            "dp_size": 1,
        }
        deduped_graphs.append(deduped)
    print(f"  -> {len(deduped_graphs)} unique (PP, TP) combinations after DP dedup")

    print("[Step 2/3] Merging TP ranks within each PP stage...")
    pp_groups = defaultdict(list)
    for g in deduped_graphs:
        pp_rank = g.get("parallel_identity", {}).get("pp_rank", 0)
        pp_groups[pp_rank].append(g)

    tp_merged_graphs = []
    for pp_rank in sorted(pp_groups.keys()):
        group = pp_groups[pp_rank]
        merged = merge_graphs(group)
        merged["parallel_identity"] = {
            "pp_rank": pp_rank,
            "tp_rank": 0,
            "tp_size": 1,
            "dp_rank": 0,
            "dp_size": 1,
        }
        tp_merged_graphs.append(merged)
    print(f"  -> {len(tp_merged_graphs)} PP stages after TP merge")

    print("[Step 3/3] Stitching PP stages into global logical graph...")
    if len(tp_merged_graphs) == 1:
        final = tp_merged_graphs[0]
    else:
        final = merge_graphs(tp_merged_graphs)
        final["parallel_identity"] = {"stitched_pp": True}
    print(f"  -> Final graph: {len(final['nodes'])} nodes, {len(final['edges'])} edges")

    return final


def build_graph_from_config(model_config, strategy_config):
    from modelperf.capture.graph import ComputationalGraph, OpType
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


def run_simulation(config_dir, graph_dir, output_dir, label=""):
    print("=" * 60)
    print(f"ModelPerf Evaluation: {label}")
    print("=" * 60)

    data = load_captured_data(config_dir, graph_dir)
    model_config = data['model_config']
    strategy_config = data['strategy_config']
    system_config = data['system_config']
    graph_data = data['graph']

    print(f"\n[Step 1] Loaded captured configs from {config_dir}")
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
        pp_size=strategy_config.get('pipeline_model_parallel_size', 1),
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
        pp_size=strategy_config.get('pipeline_model_parallel_size', 1),
        dp_size=strategy_config.get('data_parallel_size', 1),
        enable_overlap=True,
        pipeline_schedule="gpipe",
    )
    executor_gpipe = VirtualExecutor(config=config_gpipe)
    result_gpipe = executor_gpipe.execute(graph)

    print(f"  GPipe iteration time: {result_gpipe.iteration_time_ms:.2f} ms")
    speedup = result_gpipe.iteration_time_ms / result.iteration_time_ms
    print(f"  1F1B vs GPipe ratio: {speedup:.2f}x")

    os.makedirs(output_dir, exist_ok=True)

    print("\n[Step 5] Generating report...")

    from modelperf.visualization.report_generator import generate_performance_report
    viz_report_path = generate_performance_report(result, output_dir)
    print(f"  Visualization report: {viz_report_path}")

    report = {
        "label": label,
        "model_config": model_config,
        "strategy_config": strategy_config,
        "system_config": system_config,
        "simulation": result.summary(),
        "compute_breakdown": result.compute_breakdown,
        "comm_breakdown": result.comm_breakdown,
        "memory_breakdown": result.memory_breakdown,
        "what_if": {
            "gpipe_iteration_time_ms": result_gpipe.iteration_time_ms,
            "1f1b_vs_gpipe_ratio": speedup,
        },
    }

    report_path = os.path.join(output_dir, f'report_{label.replace(" ", "_")}.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"  JSON report exported to {report_path}")

    print("\n" + "=" * 60)
    print("Evaluation completed successfully!")
    print("=" * 60)

    return result


def main():
    parser = argparse.ArgumentParser(description="ModelPerf evaluation for captured training data")
    parser.add_argument("--config-dir", required=True, help="Directory containing captured configs")
    parser.add_argument("--graph-dir", required=True, help="Directory containing captured graph")
    parser.add_argument("--output-dir", required=True, help="Directory for output reports")
    parser.add_argument("--label", default="", help="Label for this evaluation run")
    args = parser.parse_args()

    run_simulation(args.config_dir, args.graph_dir, args.output_dir, args.label)


if __name__ == "__main__":
    main()
