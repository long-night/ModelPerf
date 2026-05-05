"""Config extraction utilities for Megatron-LM training arguments."""

import json
import os
from typing import Any, Dict, Optional


class ConfigExtractor:
    def __init__(self):
        self.model_config: Dict = {}
        self.strategy_config: Dict = {}
        self.system_config: Dict = {}
    
    def extract_from_args(self, args: Any) -> None:
        self.model_config = self._extract_model_config(args)
        self.strategy_config = self._extract_strategy_config(args)
        self.system_config = self._extract_system_config(args)
    
    def _extract_model_config(self, args: Any) -> Dict:
        config = {}
        
        fields = [
            'num_layers', 'hidden_size', 'ffn_hidden_size',
            'num_attention_heads', 'num_query_groups', 'kv_channels',
            'vocab_size', 'padded_vocab_size', 'max_position_embeddings',
            'position_embedding_type', 'normalization', 'norm_epsilon',
            'group_query_attention', 'rotary_base',
            'rotary_percent', 'use_rope_scaling', 'num_experts',
            'moe_router_topk', 'moe_grouped_gemm', 'fp16', 'bf16',
        ]
        
        for field in fields:
            val = getattr(args, field, None)
            if val is not None:
                config[field] = val
        
        # Handle non-serializable types
        attn_backend = getattr(args, 'attention_backend', None)
        if attn_backend is not None:
            config['attention_backend'] = str(attn_backend)
        
        config['params_dtype'] = str(getattr(args, 'params_dtype', 'float32'))
        
        return config
    
    def _extract_strategy_config(self, args: Any) -> Dict:
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
        
        config['use_distributed_optimizer'] = getattr(args, 'use_distributed_optimizer', False)
        config['overlap_grad_reduce'] = getattr(args, 'overlap_grad_reduce', False)
        config['overlap_param_gather'] = getattr(args, 'overlap_param_gather', False)
        config['overlap_p2p_comm'] = getattr(args, 'overlap_p2p_comm', True)
        
        return config
    
    def _extract_system_config(self, args: Any) -> Dict:
        config = {}
        
        fields = [
            'train_iters', 'max_position_embeddings', 'seq_length',
            'lr', 'min_lr', 'lr_decay_iters', 'lr_warmup_fraction',
            'optimizer', 'adam_beta1', 'adam_beta2', 'adam_eps',
            'weight_decay', 'clip_grad', 'world_size', 'rank',
        ]
        
        for field in fields:
            val = getattr(args, field, None)
            if val is not None:
                config[field] = val
        
        config['distributed_backend'] = getattr(args, 'distributed_backend', 'gloo')
        
        return config
    
    def to_dict(self) -> Dict[str, Dict]:
        return {
            'model_config': self.model_config,
            'strategy_config': self.strategy_config,
            'system_config': self.system_config,
        }
    
    def export_json(self, export_dir: str) -> None:
        os.makedirs(export_dir, exist_ok=True)
        
        with open(os.path.join(export_dir, 'model_config.json'), 'w') as f:
            json.dump(self.model_config, f, indent=2)
        
        with open(os.path.join(export_dir, 'strategy_config.json'), 'w') as f:
            json.dump(self.strategy_config, f, indent=2)
        
        with open(os.path.join(export_dir, 'system_config.json'), 'w') as f:
            json.dump(self.system_config, f, indent=2)


def extract_configs_from_args(args: Any) -> Dict[str, Dict]:
    extractor = ConfigExtractor()
    extractor.extract_from_args(args)
    return extractor.to_dict()
