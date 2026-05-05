"""Megatron-LM Monkey-patch hooks for automatic config extraction.

Hook points:
1. megatron.training.arguments.parse_args() -> capture all training args
2. megatron.core.parallel_state.initialize_model_parallel() -> capture parallel strategy
3. megatron.core.transformer.transformer_layer.TransformerLayer.__init__() -> capture layer structure
"""

import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
from functools import wraps

_captured_model_config: Optional[Dict] = None
_captured_strategy_config: Optional[Dict] = None
_captured_system_config: Optional[Dict] = None
_captured_layer_info: List[Dict] = []


def get_captured_configs() -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
    return _captured_model_config, _captured_strategy_config, _captured_system_config


def _extract_model_config_from_args(args: Any) -> Dict:
    config = {}
    
    config['num_layers'] = getattr(args, 'num_layers', None)
    config['hidden_size'] = getattr(args, 'hidden_size', None)
    config['ffn_hidden_size'] = getattr(args, 'ffn_hidden_size', None)
    config['num_attention_heads'] = getattr(args, 'num_attention_heads', None)
    config['num_query_groups'] = getattr(args, 'num_query_groups', 1)
    config['kv_channels'] = getattr(args, 'kv_channels', None)
    config['vocab_size'] = getattr(args, 'vocab_size', None)
    config['padded_vocab_size'] = getattr(args, 'padded_vocab_size', None)
    config['max_position_embeddings'] = getattr(args, 'max_position_embeddings', None)
    config['position_embedding_type'] = getattr(args, 'position_embedding_type', 'learned_absolute')
    
    config['normalization'] = getattr(args, 'normalization', 'LayerNorm')
    config['norm_epsilon'] = getattr(args, 'norm_epsilon', 1e-5)
    config['activation_func'] = getattr(args, 'activation_func', 'gelu')
    config['add_bias_linear'] = getattr(args, 'add_bias_linear', True)
    config['add_qkv_bias'] = getattr(args, 'add_qkv_bias', False)
    
    config['attention_backend'] = getattr(args, 'attention_backend', 'auto')
    config['group_query_attention'] = getattr(args, 'group_query_attention', False)
    config['rotary_base'] = getattr(args, 'rotary_base', 10000)
    config['rotary_percent'] = getattr(args, 'rotary_percent', 1.0)
    config['use_rope_scaling'] = getattr(args, 'use_rope_scaling', False)
    
    config['num_experts'] = getattr(args, 'num_experts', None)
    config['moe_router_topk'] = getattr(args, 'moe_router_topk', 1)
    config['moe_grouped_gemm'] = getattr(args, 'moe_grouped_gemm', False)
    
    config['fp16'] = getattr(args, 'fp16', False)
    config['bf16'] = getattr(args, 'bf16', False)
    config['params_dtype'] = str(getattr(args, 'params_dtype', 'float32'))
    
    return {k: v for k, v in config.items() if v is not None}


def _extract_strategy_config_from_args(args: Any) -> Dict:
    config = {}
    
    config['tensor_parallel_size'] = getattr(args, 'tensor_model_parallel_size', 1)
    config['pipeline_parallel_size'] = getattr(args, 'pipeline_model_parallel_size', 1)
    config['data_parallel_size'] = getattr(args, 'data_parallel_size', 1)
    config['context_parallel_size'] = getattr(args, 'context_parallel_size', 1)
    config['expert_model_parallel_size'] = getattr(args, 'expert_model_parallel_size', 1)
    config['expert_tensor_parallel_size'] = getattr(args, 'expert_tensor_parallel_size', 1)
    config['virtual_pipeline_model_parallel_size'] = getattr(args, 'virtual_pipeline_model_parallel_size', None)
    
    config['sequence_parallel'] = getattr(args, 'sequence_parallel', False)
    
    config['micro_batch_size'] = getattr(args, 'micro_batch_size', 1)
    config['global_batch_size'] = getattr(args, 'global_batch_size', 1)
    
    config['recompute_granularity'] = getattr(args, 'recompute_granularity', None)
    config['recompute_method'] = getattr(args, 'recompute_method', None)
    config['recompute_num_layers'] = getattr(args, 'recompute_num_layers', None)
    config['distribute_saved_activations'] = getattr(args, 'distribute_saved_activations', False)
    
    config['use_distributed_optimizer'] = getattr(args, 'use_distributed_optimizer', False)
    
    config['gradient_accumulation_fusion'] = getattr(args, 'gradient_accumulation_fusion', False)
    config['overlap_grad_reduce'] = getattr(args, 'overlap_grad_reduce', False)
    config['overlap_param_gather'] = getattr(args, 'overlap_param_gather', False)
    
    config['overlap_p2p_comm'] = getattr(args, 'overlap_p2p_comm', True)
    
    return config


def _extract_system_config_from_args(args: Any) -> Dict:
    config = {}
    
    config['train_iters'] = getattr(args, 'train_iters', None)
    config['max_position_embeddings'] = getattr(args, 'max_position_embeddings', None)
    config['seq_length'] = getattr(args, 'seq_length', None)
    
    config['lr'] = getattr(args, 'lr', None)
    config['min_lr'] = getattr(args, 'min_lr', None)
    config['lr_decay_iters'] = getattr(args, 'lr_decay_iters', None)
    config['lr_warmup_fraction'] = getattr(args, 'lr_warmup_fraction', None)
    config['optimizer'] = getattr(args, 'optimizer', 'adam')
    config['adam_beta1'] = getattr(args, 'adam_beta1', 0.9)
    config['adam_beta2'] = getattr(args, 'adam_beta2', 0.999)
    config['adam_eps'] = getattr(args, 'adam_eps', 1e-8)
    config['weight_decay'] = getattr(args, 'weight_decay', 0.01)
    config['clip_grad'] = getattr(args, 'clip_grad', 1.0)
    
    config['world_size'] = getattr(args, 'world_size', 1)
    config['rank'] = getattr(args, 'rank', 0)
    config['distributed_backend'] = getattr(args, 'distributed_backend', 'gloo')
    
    return config


class MegatronHookManager:
    def __init__(self, export_dir: Optional[str] = None):
        self.export_dir = export_dir
        self._original_parse_args: Optional[Callable] = None
        self._original_init_model_parallel: Optional[Callable] = None
        self._original_transformer_layer_init: Optional[Callable] = None
        self._model_config: Optional[Dict] = None
        self._strategy_config: Optional[Dict] = None
        self._system_config: Optional[Dict] = None
        self._layer_info: List[Dict] = []
        self._installed = False
    
    def install(self) -> None:
        if self._installed:
            return
        
        self._install_parse_args_hook()
        self._install_parallel_state_hook()
        self._install_transformer_layer_hook()
        self._installed = True
    
    def uninstall(self) -> None:
        if not self._installed:
            return
        
        if self._original_parse_args is not None:
            import megatron.training.arguments as args_module
            args_module.parse_args = self._original_parse_args
        
        if self._original_init_model_parallel is not None:
            import megatron.core.parallel_state as ps_module
            ps_module.initialize_model_parallel = self._original_init_model_parallel
        
        if self._original_transformer_layer_init is not None:
            from megatron.core.transformer.transformer_layer import TransformerLayer
            TransformerLayer.__init__ = self._original_transformer_layer_init
        
        self._installed = False
    
    def _install_parse_args_hook(self) -> None:
        try:
            import megatron.training.arguments as args_module
        except ImportError:
            print("[ModelPerf] Warning: megatron.training.arguments not found, skipping parse_args hook")
            return
        
        self._original_parse_args = args_module.parse_args
        original_func = self._original_parse_args
        
        @wraps(original_func)
        def hooked_parse_args(extra_args_provider=None, ignore_unknown_args=False):
            args = original_func(extra_args_provider, ignore_unknown_args)
            
            self._model_config = _extract_model_config_from_args(args)
            self._strategy_config = _extract_strategy_config_from_args(args)
            self._system_config = _extract_system_config_from_args(args)
            
            global _captured_model_config, _captured_strategy_config, _captured_system_config
            _captured_model_config = self._model_config
            _captured_strategy_config = self._strategy_config
            _captured_system_config = self._system_config
            
            print(f"[ModelPerf] Captured Megatron configs: "
                  f"model={len(self._model_config)} fields, "
                  f"strategy={len(self._strategy_config)} fields, "
                  f"system={len(self._system_config)} fields")
            
            if self.export_dir is not None:
                self._export_configs()
            
            return args
        
        args_module.parse_args = hooked_parse_args
    
    def _install_parallel_state_hook(self) -> None:
        try:
            import megatron.core.parallel_state as ps_module
        except ImportError:
            print("[ModelPerf] Warning: megatron.core.parallel_state not found, skipping parallel_state hook")
            return
        
        self._original_init_model_parallel = ps_module.initialize_model_parallel
        original_func = self._original_init_model_parallel
        
        @wraps(original_func)
        def hooked_initialize_model_parallel(
            tensor_model_parallel_size: int = 1,
            pipeline_model_parallel_size: int = 1,
            virtual_pipeline_model_parallel_size: Optional[int] = None,
            pipeline_model_parallel_split_rank: Optional[int] = None,
            pipeline_model_parallel_comm_backend: Optional[str] = None,
            use_sharp: bool = False,
            context_parallel_size: int = 1,
            hierarchical_context_parallel_sizes: Optional[List[int]] = None,
            expert_model_parallel_size: int = 1,
            num_distributed_optimizer_instances: int = 1,
            expert_tensor_parallel_size: Optional[int] = None,
            nccl_communicator_config_path: Optional[str] = None,
            distributed_timeout_minutes: int = 30,
            order: str = "tp-cp-ep-dp-pp",
            encoder_tensor_model_parallel_size: int = 0,
            encoder_pipeline_model_parallel_size: Optional[int] = 0,
            get_embedding_ranks=None,
            get_position_embedding_ranks=None,
            create_gloo_process_groups: bool = True,
            high_priority_stream_groups: Optional[List[str]] = None,
        ) -> None:
            result = original_func(
                tensor_model_parallel_size, pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size, pipeline_model_parallel_split_rank,
                pipeline_model_parallel_comm_backend, use_sharp, context_parallel_size,
                hierarchical_context_parallel_sizes, expert_model_parallel_size,
                num_distributed_optimizer_instances, expert_tensor_parallel_size,
                nccl_communicator_config_path, distributed_timeout_minutes, order,
                encoder_tensor_model_parallel_size, encoder_pipeline_model_parallel_size,
                get_embedding_ranks, get_position_embedding_ranks,
                create_gloo_process_groups, high_priority_stream_groups,
            )
            
            parallel_info = {
                'tensor_model_parallel_size': tensor_model_parallel_size,
                'pipeline_model_parallel_size': pipeline_model_parallel_size,
                'virtual_pipeline_model_parallel_size': virtual_pipeline_model_parallel_size,
                'context_parallel_size': context_parallel_size,
                'expert_model_parallel_size': expert_model_parallel_size,
                'expert_tensor_parallel_size': expert_tensor_parallel_size,
                'order': order,
                'create_gloo_process_groups': create_gloo_process_groups,
            }
            
            try:
                parallel_info['world_size'] = ps_module.get_data_parallel_world_size() * \
                    tensor_model_parallel_size * pipeline_model_parallel_size
            except Exception:
                pass
            
            if self._strategy_config is not None:
                self._strategy_config.update(parallel_info)
            
            global _captured_strategy_config
            if _captured_strategy_config is not None:
                _captured_strategy_config.update(parallel_info)
            
            print(f"[ModelPerf] Captured parallel state: TP={tensor_model_parallel_size}, "
                  f"PP={pipeline_model_parallel_size}, CP={context_parallel_size}, "
                  f"EP={expert_model_parallel_size}")
            
            return result
        
        ps_module.initialize_model_parallel = hooked_initialize_model_parallel
    
    def _install_transformer_layer_hook(self) -> None:
        try:
            from megatron.core.transformer.transformer_layer import TransformerLayer
        except ImportError:
            print("[ModelPerf] Warning: TransformerLayer not found, skipping layer hook")
            return
        
        self._original_transformer_layer_init = TransformerLayer.__init__
        original_init = self._original_transformer_layer_init
        
        @wraps(original_init)
        def hooked_init(self_layer, config: Any, submodules: Any, layer_number: int = 1,
                        hidden_dropout: Optional[float] = None,
                        model_comm_pgs: Optional[Any] = None,
                        vp_stage: Optional[int] = None):
            original_init(self_layer, config, submodules, layer_number, hidden_dropout,
                          model_comm_pgs, vp_stage)
            
            layer_info = {
                'layer_number': layer_number,
                'hidden_size': getattr(config, 'hidden_size', None),
                'num_attention_heads': getattr(config, 'num_attention_heads', None),
                'ffn_hidden_size': getattr(config, 'ffn_hidden_size', None),
                'layernorm_epsilon': getattr(config, 'layernorm_epsilon', None),
                'hidden_dropout': hidden_dropout if hidden_dropout is not None else getattr(config, 'hidden_dropout', None),
            }
            
            self._layer_info.append(layer_info)
            
            global _captured_layer_info
            _captured_layer_info.append(layer_info)
        
        TransformerLayer.__init__ = hooked_init
    
    def get_configs(self) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
        return self._model_config, self._strategy_config, self._system_config
    
    def get_layer_info(self) -> List[Dict]:
        return self._layer_info
    
    def _export_configs(self) -> None:
        if self.export_dir is None:
            return
        
        os.makedirs(self.export_dir, exist_ok=True)
        
        if self._model_config:
            with open(os.path.join(self.export_dir, 'model_config.json'), 'w') as f:
                json.dump(self._model_config, f, indent=2)
        
        if self._strategy_config:
            with open(os.path.join(self.export_dir, 'strategy_config.json'), 'w') as f:
                json.dump(self._strategy_config, f, indent=2)
        
        if self._system_config:
            with open(os.path.join(self.export_dir, 'system_config.json'), 'w') as f:
                json.dump(self._system_config, f, indent=2)
        
        print(f"[ModelPerf] Exported configs to {self.export_dir}")


def register_megatron_hooks(export_dir: Optional[str] = None) -> MegatronHookManager:
    manager = MegatronHookManager(export_dir=export_dir)
    manager.install()
    return manager


def unregister_megatron_hooks(manager: MegatronHookManager) -> None:
    manager.uninstall()
