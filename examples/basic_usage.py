"""End-to-end example of ModelPerf: from synthetic graph to validation."""

import math
from typing import Dict, Any

from modelperf.capture.graph import (
    ComputationalGraph,
    GraphNode,
    OpType,
    CommType,
)
from modelperf.symbolic.shape_inferer import SymbolicShapeInferer
from modelperf.simulation.virtual_executor import VirtualExecutor
from modelperf.simulation.execution_result import ExecutionConfig
from modelperf.analysis.what_if import WhatIfAnalyzer, WhatIfConfig
from modelperf.utils.validator import Validator, ValidationReport


def create_synthetic_graph_qwen3_0_6b() -> ComputationalGraph:
    """Create a synthetic graph representing Qwen3 0.6B model."""
    graph = ComputationalGraph()
    
    # Qwen3 0.6B config
    model_config = {
        'hidden_size': 576,
        'num_layers': 28,
        'num_attention_heads': 16,
        'ffn_hidden_size': 1536,
        'seq_length': 4096,
        'vocab_size': 151936,
        'batch_size': 1,
    }
    
    strategy_config = {
        'tensor_parallel_size': 1,
        'pipeline_parallel_size': 1,
        'data_parallel_size': 1,
        'context_parallel_size': 1,
    }
    
    graph.model_config = model_config
    graph.strategy_config = strategy_config
    
    inferer = SymbolicShapeInferer()
    B = inferer.get_symbol('B')
    S = inferer.get_symbol('S')
    H = inferer.get_symbol('H')
    
    batch = model_config['batch_size']
    seq = model_config['seq_length']
    hidden = model_config['hidden_size']
    layers = model_config['num_layers']
    
    # Create forward nodes for one layer (repeated for num_layers)
    node_id = 0
    
    # Embedding
    emb_node = graph.add_node(
        node_id=f"node_{node_id}",
        op_type="Embedding",
        node_type=OpType.COMPUTE,
        output_shapes=[(seq, batch, hidden)],
        symbolic_output_shapes=[(S, B, H)],
    )
    graph.forward_nodes.append(emb_node.node_id)
    node_id += 1
    
    # For each layer
    for layer_idx in range(layers):
        # Input layernorm
        norm1 = graph.add_node(
            node_id=f"node_{node_id}",
            op_type="LayerNorm",
            node_type=OpType.COMPUTE,
            input_shapes=[(seq, batch, hidden)],
            output_shapes=[(seq, batch, hidden)],
            symbolic_input_shapes=[(S, B, H)],
            symbolic_output_shapes=[(S, B, H)],
        )
        graph.forward_nodes.append(norm1.node_id)
        node_id += 1
        
        # QKV projection
        qkv = graph.add_node(
            node_id=f"node_{node_id}",
            op_type="Linear_qkv",
            node_type=OpType.COMPUTE,
            input_shapes=[(seq, batch, hidden)],
            output_shapes=[(seq, batch, hidden * 3)],
            symbolic_input_shapes=[(S, B, H)],
            symbolic_output_shapes=[(S, B, H * 3)],
            params={"weight": (hidden, hidden * 3)},
        )
        graph.forward_nodes.append(qkv.node_id)
        node_id += 1
        
        # Self attention
        attn = graph.add_node(
            node_id=f"node_{node_id}",
            op_type="SelfAttention",
            node_type=OpType.COMPUTE,
            input_shapes=[(seq, batch, hidden * 3)],
            output_shapes=[(seq, batch, hidden)],
            symbolic_input_shapes=[(S, B, H * 3)],
            symbolic_output_shapes=[(S, B, H)],
        )
        graph.forward_nodes.append(attn.node_id)
        node_id += 1
        
        # Attention projection
        proj = graph.add_node(
            node_id=f"node_{node_id}",
            op_type="Linear_proj",
            node_type=OpType.COMPUTE,
            input_shapes=[(seq, batch, hidden)],
            output_shapes=[(seq, batch, hidden)],
            symbolic_input_shapes=[(S, B, H)],
            symbolic_output_shapes=[(S, B, H)],
            params={"weight": (hidden, hidden)},
        )
        graph.forward_nodes.append(proj.node_id)
        node_id += 1
        
        # MLP
        mlp_fc1 = graph.add_node(
            node_id=f"node_{node_id}",
            op_type="Linear_fc1",
            node_type=OpType.COMPUTE,
            input_shapes=[(seq, batch, hidden)],
            output_shapes=[(seq, batch, 1536)],
            symbolic_input_shapes=[(S, B, H)],
            symbolic_output_shapes=[(S, B, 1536)],
            params={"weight": (hidden, 1536)},
        )
        graph.forward_nodes.append(mlp_fc1.node_id)
        node_id += 1
        
        mlp_fc2 = graph.add_node(
            node_id=f"node_{node_id}",
            op_type="Linear_fc2",
            node_type=OpType.COMPUTE,
            input_shapes=[(seq, batch, 1536)],
            output_shapes=[(seq, batch, hidden)],
            symbolic_input_shapes=[(S, B, 1536)],
            symbolic_output_shapes=[(S, B, H)],
            params={"weight": (1536, hidden)},
        )
        graph.forward_nodes.append(mlp_fc2.node_id)
        node_id += 1
    
    # LM Head
    lm_head = graph.add_node(
        node_id=f"node_{node_id}",
        op_type="Linear_lm_head",
        node_type=OpType.COMPUTE,
        input_shapes=[(seq, batch, hidden)],
        output_shapes=[(seq, batch, 151936)],
        symbolic_input_shapes=[(S, B, H)],
        symbolic_output_shapes=[(S, B, 151936)],
        params={"weight": (hidden, 151936)},
    )
    graph.forward_nodes.append(lm_head.node_id)
    
    # Create backward nodes (reverse order, excluding embedding)
    for node_id_str in reversed(graph.forward_nodes):
        if node_id_str not in [emb_node.node_id]:
            graph.backward_nodes.append(f"{node_id_str}_bwd")
    
    return graph


def run_simulation_example():
    """Run the complete simulation and validation example."""
    print("=" * 60)
    print("ModelPerf Example: Qwen3 0.6B Training Simulation")
    print("=" * 60)
    print()
    
    # Step 1: Create synthetic graph
    print("Step 1: Creating synthetic computational graph...")
    graph = create_synthetic_graph_qwen3_0_6b()
    print(f"  Created graph with {len(graph.nodes)} nodes")
    print(f"  Forward nodes: {len(graph.forward_nodes)}")
    print(f"  Backward nodes: {len(graph.backward_nodes)}")
    print()
    
    # Step 2: Run virtual execution
    print("Step 2: Running virtual execution...")
    config = ExecutionConfig(
        gpu_type="a100",
        network_type="nvlink",
        peak_compute_tflops=312.0,
        peak_bandwidth_gbs=2039.0,
        network_bandwidth_gbs=600.0,
        network_latency_us=2.0,
        memory_per_device_gb=80.0,
        tp_size=1,
        pp_size=1,
        dp_size=1,
    )
    
    executor = VirtualExecutor(config=config)
    sim_result = executor.execute(graph)
    
    print(f"  Iteration time: {sim_result.iteration_time_ms:.2f} ms")
    print(f"  Compute time: {sim_result.compute_time_ms:.2f} ms")
    print(f"  Comm time: {sim_result.comm_time_ms:.2f} ms")
    print(f"  Peak memory: {sim_result.peak_memory_mb:.2f} MB")
    print(f"  Bottleneck: {sim_result.bottleneck}")
    print()
    
    # Step 3: What-if analysis
    print("Step 3: Running what-if analysis...")
    analyzer = WhatIfAnalyzer(graph)
    
    # Test different tensor parallel sizes
    test_configs = [
        {'tensor_parallel_size': 1},
        {'tensor_parallel_size': 2},
        {'tensor_parallel_size': 4},
    ]
    
    what_if_results = []
    for changes in test_configs:
        new_config = analyzer.modify_config(changes)
        result = analyzer.re_evaluate(new_config)
        what_if_results.append((changes, result))
        print(f"  TP={changes['tensor_parallel_size']}: {result.iteration_time_ms:.2f} ms")
    print()
    
    # Step 4: Validation against "real" training metrics
    print("Step 4: Running validation against real metrics...")
    
    # In a real scenario, these would come from actual training runs
    # Here we simulate "real" metrics with slight variations from simulation
    real_metrics = {
        'iteration_time_ms': sim_result.iteration_time_ms * 1.08,  # 8% higher in reality
        'peak_memory_mb': sim_result.peak_memory_mb * 0.95,  # 5% lower in reality
        'compute_time_ms': sim_result.compute_time_ms * 1.05,
        'comm_time_ms': sim_result.comm_time_ms * 1.15,
    }
    
    validator = Validator(tolerance_percent=15.0)
    
    validation_report = validator.validate_full(
        sim_result=sim_result,
        real_metrics_dict=real_metrics,
        simulation_config=config.to_dict(),
        real_config={'source': 'synthetic_training_run'},
    )
    
    print(f"  Overall score: {validation_report.overall_score:.2f}%")
    print(f"  Status: {validation_report.pass_fail}")
    print()
    
    for result in validation_report.results:
        status = "✓" if result.is_within_tolerance else "✗"
        print(f"  {status} {result.metric_name}: sim={result.simulated_value:.2f}, "
              f"real={result.real_value:.2f}, mape={result.mape:.2f}%")
    print()
    
    # Step 5: Generate markdown report
    print("Step 5: Generating validation report...")
    report_md = validator.generate_report(validation_report, format="markdown")
    print("  Report generated (length: {} chars)".format(len(report_md)))
    print()
    
    print("=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    
    return {
        'graph': graph,
        'sim_result': sim_result,
        'what_if_results': what_if_results,
        'validation_report': validation_report,
        'report_md': report_md,
    }


if __name__ == "__main__":
    results = run_simulation_example()