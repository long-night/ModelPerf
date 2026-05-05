"""Configuration dataclasses for ModelPerf simulation."""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import json


@dataclass
class ModelConfig:
    num_layers: int = 12
    hidden_size: int = 768
    ffn_hidden_size: Optional[int] = None
    num_attention_heads: int = 12
    num_query_groups: int = 1
    kv_channels: Optional[int] = None
    vocab_size: int = 50257
    padded_vocab_size: Optional[int] = None
    max_position_embeddings: int = 1024
    position_embedding_type: str = 'learned_absolute'
    normalization: str = 'LayerNorm'
    norm_epsilon: float = 1e-5
    attention_backend: str = 'auto'
    group_query_attention: bool = False
    rotary_base: int = 10000
    rotary_percent: float = 1.0
    use_rope_scaling: bool = False
    num_experts: Optional[int] = None
    moe_router_topk: int = 1
    moe_grouped_gemm: bool = False
    params_dtype: str = 'float32'
    fp16: bool = False
    bf16: bool = False
    
    def __post_init__(self):
        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = self.hidden_size * 4
        if self.padded_vocab_size is None:
            self.padded_vocab_size = self.vocab_size
        if self.kv_channels is None:
            if self.num_attention_heads > 0:
                self.kv_channels = self.hidden_size // self.num_attention_heads
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ModelConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_json(cls, path: str) -> 'ModelConfig':
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))
    
    @classmethod
    def qwen3_0_6b(cls) -> 'ModelConfig':
        return cls(
            num_layers=4,
            hidden_size=1024,
            ffn_hidden_size=3072,
            num_attention_heads=16,
            num_query_groups=8,
            vocab_size=151936,
            padded_vocab_size=152064,
            max_position_embeddings=40960,
            position_embedding_type='rope',
            normalization='RMSNorm',
            norm_epsilon=1e-6,
            group_query_attention=True,
            rotary_base=1000000,
            params_dtype='bfloat16',
            bf16=True,
        )
    
    @classmethod
    def llama3_8b(cls) -> 'ModelConfig':
        return cls(
            num_layers=32,
            hidden_size=4096,
            ffn_hidden_size=14336,
            num_attention_heads=32,
            num_query_groups=8,
            vocab_size=128256,
            max_position_embeddings=8192,
            position_embedding_type='rope',
            normalization='RMSNorm',
            group_query_attention=True,
            rotary_base=500000,
            params_dtype='bfloat16',
            bf16=True,
        )


@dataclass
class StrategyConfig:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: Optional[int] = None
    
    sequence_parallel: bool = False
    micro_batch_size: int = 1
    global_batch_size: int = 1
    
    recompute_granularity: Optional[str] = None
    recompute_method: Optional[str] = None
    recompute_num_layers: Optional[int] = None
    distribute_saved_activations: bool = False
    
    use_distributed_optimizer: bool = False
    gradient_accumulation_fusion: bool = False
    overlap_grad_reduce: bool = False
    overlap_param_gather: bool = False
    overlap_p2p_comm: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'StrategyConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_json(cls, path: str) -> 'StrategyConfig':
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


@dataclass
class SystemConfig:
    train_iters: Optional[int] = None
    seq_length: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    
    lr: Optional[float] = None
    min_lr: Optional[float] = None
    lr_decay_iters: Optional[int] = None
    lr_warmup_fraction: Optional[float] = None
    optimizer: str = 'adam'
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 0.01
    clip_grad: float = 1.0
    
    world_size: int = 1
    rank: int = 0
    distributed_backend: str = 'gloo'
    
    # Hardware specs (theoretical values for simulation, not requiring real GPU)
    gpu_type: str = 'a100'
    peak_compute_tflops: float = 312.0
    peak_bandwidth_gbs: float = 2039.0
    network_bandwidth_gbs: float = 600.0
    network_latency_us: float = 2.0
    memory_per_device_gb: float = 80.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SystemConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_json(cls, path: str) -> 'SystemConfig':
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))
    
    @classmethod
    def a100(cls) -> 'SystemConfig':
        return cls(
            gpu_type='a100',
            peak_compute_tflops=312.0,
            peak_bandwidth_gbs=2039.0,
            network_bandwidth_gbs=600.0,
            network_latency_us=2.0,
            memory_per_device_gb=80.0,
        )
    
    @classmethod
    def h100(cls) -> 'SystemConfig':
        return cls(
            gpu_type='h100',
            peak_compute_tflops=989.0,
            peak_bandwidth_gbs=3350.0,
            network_bandwidth_gbs=900.0,
            network_latency_us=1.5,
            memory_per_device_gb=80.0,
        )
