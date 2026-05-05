"""Execution result dataclass for ModelPerf simulation."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class ExecutionResult:
    """Result of virtual execution simulation."""
    
    iteration_time_ms: float
    compute_time_ms: float
    comm_time_ms: float
    peak_memory_mb: float
    bottleneck: str
    
    forward_time_ms: float = 0.0
    backward_time_ms: float = 0.0
    optimizer_time_ms: float = 0.0
    
    compute_breakdown: Dict[str, float] = field(default_factory=dict)
    comm_breakdown: Dict[str, float] = field(default_factory=dict)
    memory_breakdown: Dict[str, float] = field(default_factory=dict)
    
    bubble_time_ms: float = 0.0
    num_microbatches: int = 1
    pipeline_stages: int = 1
    
    per_device_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary dict of results."""
        return {
            "iteration_time_ms": self.iteration_time_ms,
            "compute_time_ms": self.compute_time_ms,
            "comm_time_ms": self.comm_time_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "bottleneck": self.bottleneck,
            "forward_time_ms": self.forward_time_ms,
            "backward_time_ms": self.backward_time_ms,
            "compute_breakdown": self.compute_breakdown,
            "comm_breakdown": self.comm_breakdown,
            "memory_breakdown": self.memory_breakdown,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return {
            "iteration_time_ms": self.iteration_time_ms,
            "compute_time_ms": self.compute_time_ms,
            "comm_time_ms": self.comm_time_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "bottleneck": self.bottleneck,
            "forward_time_ms": self.forward_time_ms,
            "backward_time_ms": self.backward_time_ms,
            "optimizer_time_ms": self.optimizer_time_ms,
            "compute_breakdown": self.compute_breakdown,
            "comm_breakdown": self.comm_breakdown,
            "memory_breakdown": self.memory_breakdown,
            "bubble_time_ms": self.bubble_time_ms,
            "num_microbatches": self.num_microbatches,
            "pipeline_stages": self.pipeline_stages,
        }


@dataclass
class ExecutionConfig:
    """Configuration for virtual execution."""
    
    gpu_type: str = "a100"
    network_type: str = "nvlink"
    
    peak_compute_tflops: float = 312.0
    peak_bandwidth_gbs: float = 2039.0
    network_bandwidth_gbs: float = 600.0
    network_latency_us: float = 2.0
    
    memory_per_device_gb: float = 80.0
    
    tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1
    cp_size: int = 1
    
    gradient_accumulation_steps: int = 1
    num_microbatches: int = 1
    
    enable_activation_checkpointing: bool = False
    checkpointed_layers: Optional[set] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gpu_type": self.gpu_type,
            "network_type": self.network_type,
            "peak_compute_tflops": self.peak_compute_tflops,
            "peak_bandwidth_gbs": self.peak_bandwidth_gbs,
            "network_bandwidth_gbs": self.network_bandwidth_gbs,
            "network_latency_us": self.network_latency_us,
            "memory_per_device_gb": self.memory_per_device_gb,
            "tp_size": self.tp_size,
            "pp_size": self.pp_size,
            "dp_size": self.dp_size,
            "cp_size": self.cp_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_microbatches": self.num_microbatches,
            "enable_activation_checkpointing": self.enable_activation_checkpointing,
        }


__all__ = [
    "ExecutionResult",
    "ExecutionConfig",
]
