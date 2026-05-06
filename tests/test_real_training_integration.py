import os
import sys
import torch

sys.path.insert(0, '/mnt/d/ubuntu/opencode/training_framework/Pai-Megatron-Patch-12.0')
sys.path.insert(0, '/mnt/d/ubuntu/opencode/training_framework/Pai-Megatron-Patch-12.0/backends/megatron/Megatron-LM-20250707')
sys.path.insert(0, '/mnt/d/ubuntu/opencode/training_framework/ModelPerf')

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def test_model_creation_and_capture():
    print("=" * 60)
    print("ModelPerf Real Training Validation (Lightweight)")
    print("=" * 60)

    from megatron.training.arguments import parse_args
    from megatron_patch.arguments import get_patch_args

    sys.argv = [
        'pretrain_qwen.py',
        '--tensor-model-parallel-size', '1',
        '--pipeline-model-parallel-size', '1',
        '--num-layers', '2',
        '--hidden-size', '256',
        '--ffn-hidden-size', '768',
        '--num-attention-heads', '8',
        '--num-query-groups', '4',
        '--seq-length', '64',
        '--micro-batch-size', '1',
        '--global-batch-size', '1',
        '--train-iters', '1',
        '--lr', '1e-5',
        '--min-lr', '1e-6',
        '--lr-decay-iters', '1',
        '--lr-warmup-fraction', '0.01',
        '--init-method-std', '0.02',
        '--clip-grad', '1.0',
        '--weight-decay', '0.01',
        '--adam-beta1', '0.9',
        '--adam-beta2', '0.95',
        '--adam-eps', '1e-8',
        '--fp16',
        '--position-embedding-type', 'rope',
        '--tokenizer-type', 'HuggingFaceTokenizer',
        '--vocab-file', '/mnt/d/ubuntu/models/Qwen3-0.6B/vocab.json',
        '--merge-file', '/mnt/d/ubuntu/models/Qwen3-0.6B/merges.txt',
        '--data-path', '/mnt/d/ubuntu/datasets/pretrain/qwen3-datasets/mmap_qwen3_datasets_text_document',
        '--log-interval', '1',
        '--save-interval', '10000',
        '--eval-interval', '1000',
        '--eval-iters', '0',
        '--distributed-backend', 'gloo',
        '--transformer-impl', 'local',
        '--no-masked-softmax-fusion',
        '--no-bias-gelu-fusion',
        '--no-bias-dropout-fusion',
        '--no-gradient-accumulation-fusion',
        '--use-flash-attn', 'false',
        '--normalization', 'RMSNorm',
        '--norm-epsilon', '1e-06',
        '--rotary-percent', '1.0',
        '--rotary-base', '1000000',
        '--use-rotary-position-embeddings',
        '--swiglu',
        '--padded-vocab-size', '152064',
    ]

    args = parse_args(extra_args_provider=get_patch_args)
    print(f"\n[OK] Args parsed: layers={args.num_layers}, hidden={args.hidden_size}")

    from modelperf.framework_adapter.config_extractor import ConfigExtractor
    extractor = ConfigExtractor()
    extractor.extract_from_args(args)
    print(f"[OK] Configs extracted: model={len(extractor.model_config)}, strategy={len(extractor.strategy_config)}")

    from megatron_patch.tokenizer import build_tokenizer
    build_tokenizer(args)

    from megatron.training.arguments import core_transformer_config_from_args
    config = core_transformer_config_from_args(args)

    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    transformer_layer_spec = get_gpt_layer_local_spec(
        args.num_experts, args.moe_grouped_gemm,
        args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm,
        normalization=args.normalization,
    )

    from megatron.core.models.gpt import GPTModel
    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=True,
        post_process=True,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
    )
    print(f"[OK] GPTModel created: {sum(p.numel() for p in model.parameters())} params")

    from modelperf.capture.coordinator import CaptureCoordinator
    coordinator = CaptureCoordinator()
    coordinator.attach(model)
    print(f"[OK] CaptureCoordinator attached")

    with coordinator:
        x = torch.randint(0, args.padded_vocab_size, (1, 64))
        y = model(x)
        loss = y.sum()
        loss.backward()

    graph = coordinator.get_graph()
    print(f"\n[OK] Capture completed!")
    print(f"    Total nodes: {len(graph.nodes)}")
    print(f"    Forward nodes: {len(graph.forward_nodes)}")
    print(f"    Backward nodes: {len(graph.backward_nodes)}")
    print(f"    Edges: {len(graph.edges)}")
    print(f"    Compute nodes: {len(graph.get_compute_nodes())}")
    print(f"    Comm nodes: {len(graph.get_comm_nodes())}")

    import json
    export_dir = '/mnt/d/ubuntu/opencode/training_framework/ModelPerf/examples/output/captured_graph'
    os.makedirs(export_dir, exist_ok=True)
    with open(os.path.join(export_dir, 'computational_graph.json'), 'w') as f:
        json.dump(graph.to_dict(), f, indent=2)
    print(f"\n[OK] Graph exported to {export_dir}/computational_graph.json")

    from modelperf.simulation.virtual_executor import VirtualExecutor
    from modelperf.simulation.execution_result import ExecutionConfig
    from modelperf.analysis.what_if import WhatIfAnalyzer

    graph.model_config = extractor.model_config
    graph.strategy_config = extractor.strategy_config

    exec_config = ExecutionConfig(
        peak_compute_tflops=312.0,
        peak_bandwidth_gbs=2039.0,
        network_bandwidth_gbs=600.0,
        memory_per_device_gb=80.0,
        enable_overlap=True,
    )
    executor = VirtualExecutor(config=exec_config)
    result = executor.execute(graph)

    print(f"\n[OK] Simulation completed!")
    print(f"    Iteration time: {result.iteration_time_ms:.2f} ms")
    print(f"    Forward: {result.forward_time_ms:.2f} ms")
    print(f"    Backward: {result.backward_time_ms:.2f} ms")
    print(f"    Peak memory: {result.peak_memory_mb:.2f} MB")
    print(f"    Throughput: {result.throughput_tokens_per_sec:.2f} tokens/sec")
    print(f"    Bottleneck: {result.bottleneck}")

    analyzer = WhatIfAnalyzer(graph)
    new_config = analyzer.modify_config({'micro_batch_size': 2})
    new_result = analyzer.re_evaluate(new_config)
    print(f"\n[OK] What-if (BS=2): {new_result.iteration_time_ms:.2f} ms, {new_result.peak_memory_mb:.2f} MB")

    print("\n" + "=" * 60)
    print("Validation PASSED!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    try:
        success = test_model_creation_and_capture()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[FAILED] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
