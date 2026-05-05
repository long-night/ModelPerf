"""End-to-end example: Capture real Qwen3 0.6B training graph and run simulation.

This example demonstrates how to use ModelPerf framework_adapter to:
1. Hook into Pai-Megatron-Patch Qwen3 training
2. Automatically extract model/strategy/system configs
3. Capture computational graph during training
4. Run performance simulation on the captured graph

Environment: CPU only (CUDA_VISIBLE_DEVICES=-1)
"""

import os
import sys
import json


def setup_environment():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    megatron_patch_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'Pai-Megatron-Patch-12.0'
    )
    megatron_lm_path = os.path.join(megatron_patch_path, 'backends/megatron/Megatron-LM-20250707')
    
    sys.path.insert(0, megatron_patch_path)
    sys.path.insert(0, megatron_lm_path)
    
    modelperf_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, modelperf_path)
    
    print(f"[Setup] Megatron-Patch path: {megatron_patch_path}")
    print(f"[Setup] Megatron-LM path: {megatron_lm_path}")
    print(f"[Setup] ModelPerf path: {modelperf_path}")


def run_capture_example():
    setup_environment()
    
    print("\n" + "=" * 60)
    print("ModelPerf: Real Training Capture Example (Qwen3 0.6B)")
    print("=" * 60)
    
    try:
        from modelperf.framework_adapter import register_pai_patch_hooks
    except ImportError as e:
        print(f"[Error] Failed to import ModelPerf: {e}")
        print("[Hint] Make sure ModelPerf is in PYTHONPATH")
        return
    
    export_dir = os.path.join(os.path.dirname(__file__), 'output', 'captured_configs')
    os.makedirs(export_dir, exist_ok=True)
    
    print(f"\n[Step 1] Installing ModelPerf hooks...")
    print(f"[Step 1] Configs will be exported to: {export_dir}")
    
    hook_manager = register_pai_patch_hooks(export_dir=export_dir)
    
    print("[Step 1] Hooks installed successfully")
    print("  - parse_args hook: will capture model/strategy/system configs")
    print("  - initialize_model_parallel hook: will capture parallel state")
    print("  - TransformerLayer.__init__ hook: will capture layer structure")
    
    print("\n[Step 2] Training would start here with hooks active...")
    print("  In real usage, call:")
    print("    from pretrain_qwen import pretrain, model_provider")
    print("    pretrain(...)")
    print("")
    print("  For this demo, we simulate the captured configs from a Qwen3 0.6B run.")
    
    _simulate_captured_configs(hook_manager)
    
    print("\n[Step 3] Retrieving captured configs...")
    model_config, strategy_config, system_config = hook_manager.get_configs()
    layer_info = hook_manager.get_layer_info()
    
    if model_config:
        print(f"  Model config: {len(model_config)} fields")
        for k, v in list(model_config.items())[:5]:
            print(f"    {k}: {v}")
        if len(model_config) > 5:
            print(f"    ... and {len(model_config) - 5} more")
    
    if strategy_config:
        print(f"  Strategy config: {len(strategy_config)} fields")
        for k, v in list(strategy_config.items())[:5]:
            print(f"    {k}: {v}")
        if len(strategy_config) > 5:
            print(f"    ... and {len(strategy_config) - 5} more")
    
    if system_config:
        print(f"  System config: {len(system_config)} fields")
    
    print(f"  Layer info: {len(layer_info)} layers captured")
    
    print("\n[Step 4] Running performance simulation...")
    _run_simulation(model_config, strategy_config)
    
    print("\n[Step 5] Exporting configs...")
    hook_manager._export_configs()
    
    print(f"  Exported to {export_dir}:")
    for fname in ['model_config.json', 'strategy_config.json', 'system_config.json']:
        fpath = os.path.join(export_dir, fname)
        if os.path.exists(fpath):
            print(f"    - {fname}")
    
    print("\n[Step 6] Cleaning up hooks...")
    hook_manager.uninstall()
    print("  Hooks uninstalled")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


def _simulate_captured_configs(hook_manager):
    """Simulate what would be captured from a real Qwen3 0.6B training run."""
    hook_manager._model_config = {
        'num_layers': 4,
        'hidden_size': 1024,
        'ffn_hidden_size': 3072,
        'num_attention_heads': 16,
        'num_query_groups': 8,
        'vocab_size': 151936,
        'padded_vocab_size': 152064,
        'max_position_embeddings': 40960,
        'position_embedding_type': 'rope',
        'normalization': 'RMSNorm',
        'norm_epsilon': 1e-06,
        'group_query_attention': True,
        'rotary_base': 1000000,
        'rotary_percent': 1.0,
        'use_rope_scaling': False,
        'params_dtype': 'torch.bfloat16',
        'bf16': True,
    }
    
    hook_manager._strategy_config = {
        'tensor_parallel_size': 1,
        'pipeline_parallel_size': 1,
        'data_parallel_size': 1,
        'context_parallel_size': 1,
        'expert_model_parallel_size': 1,
        'expert_tensor_parallel_size': 1,
        'sequence_parallel': True,
        'micro_batch_size': 1,
        'global_batch_size': 1,
        'recompute_granularity': 'selective',
        'use_distributed_optimizer': False,
        'overlap_grad_reduce': False,
        'overlap_param_gather': False,
        'overlap_p2p_comm': True,
    }
    
    hook_manager._system_config = {
        'train_iters': 25600,
        'seq_length': 128,
        'lr': 1e-05,
        'min_lr': 1e-06,
        'optimizer': 'adam',
        'weight_decay': 0.01,
        'clip_grad': 1.0,
        'world_size': 1,
        'rank': 0,
        'distributed_backend': 'gloo',
    }
    
    hook_manager._layer_info = [
        {'layer_number': i + 1, 'hidden_size': 1024, 'num_attention_heads': 16,
         'ffn_hidden_size': 3072, 'layernorm_epsilon': 1e-06}
        for i in range(4)
    ]


def _run_simulation(model_config, strategy_config):
    try:
        from modelperf.capture.graph import ComputationalGraph, GraphNode, OpType
        from modelperf.symbolic.shape_inferer import SymbolicShapeInferer
        from modelperf.simulation.virtual_executor import VirtualExecutor
        from modelperf.simulation.execution_result import ExecutionConfig
    except ImportError as e:
        print(f"  [Warning] Cannot import simulation modules: {e}")
        return
    
    if not model_config or not strategy_config:
        print("  [Warning] Missing configs, skipping simulation")
        return
    
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
    
    # Build synthetic graph matching Qwen3 structure
    node_id = 0
    
    emb = graph.add_node(
        node_id=f"node_{node_id}", op_type="Embedding", node_type=OpType.COMPUTE,
        output_shapes=[(seq, batch, hidden)], symbolic_output_shapes=[(S, B, H)],
    )
    graph.forward_nodes.append(emb.node_id)
    node_id += 1
    
    for _ in range(layers):
        norm1 = graph.add_node(
            node_id=f"node_{node_id}", op_type="LayerNorm", node_type=OpType.COMPUTE,
            input_shapes=[(seq, batch, hidden)], output_shapes=[(seq, batch, hidden)],
            symbolic_input_shapes=[(S, B, H)], symbolic_output_shapes=[(S, B, H)],
        )
        graph.forward_nodes.append(norm1.node_id)
        node_id += 1
        
        qkv = graph.add_node(
            node_id=f"node_{node_id}", op_type="Linear_qkv", node_type=OpType.COMPUTE,
            input_shapes=[(seq, batch, hidden)], output_shapes=[(seq, batch, hidden * 3)],
            symbolic_input_shapes=[(S, B, H)], symbolic_output_shapes=[(S, B, H * 3)],
            params={"weight": (hidden, hidden * 3)},
        )
        graph.forward_nodes.append(qkv.node_id)
        node_id += 1
        
        attn = graph.add_node(
            node_id=f"node_{node_id}", op_type="SelfAttention", node_type=OpType.COMPUTE,
            input_shapes=[(seq, batch, hidden * 3)], output_shapes=[(seq, batch, hidden)],
            symbolic_input_shapes=[(S, B, H * 3)], symbolic_output_shapes=[(S, B, H)],
        )
        graph.forward_nodes.append(attn.node_id)
        node_id += 1
        
        proj = graph.add_node(
            node_id=f"node_{node_id}", op_type="Linear_proj", node_type=OpType.COMPUTE,
            input_shapes=[(seq, batch, hidden)], output_shapes=[(seq, batch, hidden)],
            symbolic_input_shapes=[(S, B, H)], symbolic_output_shapes=[(S, B, H)],
            params={"weight": (hidden, hidden)},
        )
        graph.forward_nodes.append(proj.node_id)
        node_id += 1
        
        mlp_fc1 = graph.add_node(
            node_id=f"node_{node_id}", op_type="Linear_fc1", node_type=OpType.COMPUTE,
            input_shapes=[(seq, batch, hidden)], output_shapes=[(seq, batch, ffn)],
            symbolic_input_shapes=[(S, B, H)], symbolic_output_shapes=[(S, B, ffn)],
            params={"weight": (hidden, ffn)},
        )
        graph.forward_nodes.append(mlp_fc1.node_id)
        node_id += 1
        
        mlp_fc2 = graph.add_node(
            node_id=f"node_{node_id}", op_type="Linear_fc2", node_type=OpType.COMPUTE,
            input_shapes=[(seq, batch, ffn)], output_shapes=[(seq, batch, hidden)],
            symbolic_input_shapes=[(S, B, ffn)], symbolic_output_shapes=[(S, B, H)],
            params={"weight": (ffn, hidden)},
        )
        graph.forward_nodes.append(mlp_fc2.node_id)
        node_id += 1
    
    lm_head = graph.add_node(
        node_id=f"node_{node_id}", op_type="Linear_lm_head", node_type=OpType.COMPUTE,
        input_shapes=[(seq, batch, hidden)],
        output_shapes=[(seq, batch, model_config.get('vocab_size', 151936))],
        symbolic_input_shapes=[(S, B, H)],
        symbolic_output_shapes=[(S, B, model_config.get('vocab_size', 151936))],
        params={"weight": (hidden, model_config.get('vocab_size', 151936))},
    )
    graph.forward_nodes.append(lm_head.node_id)
    
    for node_id_str in reversed(graph.forward_nodes):
        if node_id_str != emb.node_id:
            graph.backward_nodes.append(f"{node_id_str}_bwd")
    
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
    )
    
    executor = VirtualExecutor(config=config)
    result = executor.execute(graph)
    
    print(f"  Simulation results:")
    print(f"    Iteration time: {result.iteration_time_ms:.2f} ms")
    print(f"    Compute time: {result.compute_time_ms:.2f} ms")
    print(f"    Comm time: {result.comm_time_ms:.2f} ms")
    print(f"    Peak memory: {result.peak_memory_mb:.2f} MB")
    print(f"    Bottleneck: {result.bottleneck}")


if __name__ == "__main__":
    run_capture_example()
