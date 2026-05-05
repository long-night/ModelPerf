"""Memory Tracker for monitoring memory usage during virtual execution."""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from ..capture.graph import OpType


class MemoryType(Enum):
    ACTIVATION = "activation"
    PARAMETER = "parameter"
    GRADIENT = "gradient"
    BUFFER = "buffer"
    TEMP = "temp"


@dataclass
class Allocation:
    size_bytes: int
    mem_type: MemoryType
    name: str = ""
    layer_id: Optional[int] = None


class DeviceMemory:
    """Memory tracker for a single device."""
    
    def __init__(self, device_id: int, total_memory_bytes: int = 0):
        self.device_id = device_id
        self.total_memory_bytes = total_memory_bytes
        self.current_usage = 0
        self.peak_usage = 0
        self.allocations: Dict[str, Allocation] = {}
        self.activation_checkpoints: Set[int] = set()
        
    def allocate(self, name: str, size_bytes: int, mem_type: MemoryType, layer_id: Optional[int] = None) -> bool:
        if size_bytes < 0:
            raise ValueError(f"Cannot allocate negative bytes: {size_bytes}")
        if size_bytes == 0:
            return True
            
        if name in self.allocations:
            existing = self.allocations[name]
            if existing.size_bytes != size_bytes:
                self.current_usage -= existing.size_bytes
                self.current_usage += size_bytes
                existing.size_bytes = size_bytes
        else:
            self.allocations[name] = Allocation(size_bytes, mem_type, name, layer_id)
            self.current_usage += size_bytes
            
        self.peak_usage = max(self.peak_usage, self.current_usage)
        
        if self.total_memory_bytes > 0 and self.current_usage > self.total_memory_bytes:
            return False
        return True
        
    def free(self, name: str) -> int:
        if name not in self.allocations:
            return 0
        size = self.allocations[name].size_bytes
        self.current_usage -= size
        del self.allocations[name]
        return size
        
    def get_usage_by_type(self, mem_type: MemoryType) -> int:
        return sum(a.size_bytes for a in self.allocations.values() if a.mem_type == mem_type)
        
    def get_peak(self) -> int:
        return self.peak_usage
        
    def clear(self):
        self.allocations.clear()
        self.current_usage = 0
        self.peak_usage = 0
        self.activation_checkpoints.clear()


class MemoryTracker:
    """Tracks memory usage across multiple devices during virtual execution.
    
    Supports:
    - Per-device memory tracking
    - Activation checkpointing
    - Communication buffer tracking
    - Peak memory estimation
    """
    
    def __init__(self, num_devices: int = 1, memory_per_device_gb: float = 80.0):
        self.num_devices = num_devices
        self.memory_per_device_bytes = int(memory_per_device_gb * 1e9)
        self.devices: Dict[int, DeviceMemory] = {}
        self.activation_checkpointing_enabled = False
        self.checkpointed_layers: Set[int] = set()
        
        for i in range(num_devices):
            self.devices[i] = DeviceMemory(i, self.memory_per_device_bytes)
            
    def allocate(
        self,
        device_id: int,
        name: str,
        size_bytes: int,
        mem_type: MemoryType = MemoryType.TEMP,
        layer_id: Optional[int] = None,
    ) -> bool:
        if device_id not in self.devices:
            raise ValueError(f"Invalid device_id: {device_id}")
        return self.devices[device_id].allocate(name, size_bytes, mem_type, layer_id)
        
    def free(self, device_id: int, name: str) -> int:
        if device_id not in self.devices:
            return 0
        return self.devices[device_id].free(name)
        
    def get_peak_memory(self, device_id: Optional[int] = None) -> int:
        if device_id is not None:
            return self.devices[device_id].get_peak()
        return sum(d.get_peak() for d in self.devices.values())
        
    def get_current_memory(self, device_id: Optional[int] = None) -> int:
        if device_id is not None:
            return self.devices[device_id].current_usage
        return sum(d.current_usage for d in self.devices.values())
        
    def get_memory_by_type(self, mem_type: MemoryType, device_id: Optional[int] = None) -> int:
        if device_id is not None:
            return self.devices[device_id].get_usage_by_type(mem_type)
        return sum(d.get_usage_by_type(mem_type) for d in self.devices.values())
        
    def enable_activation_checkpointing(self, checkpointed_layers: Optional[Set[int]] = None):
        self.activation_checkpointing_enabled = True
        if checkpointed_layers:
            self.checkpointed_layers = checkpointed_layers.copy()
            
    def is_layer_checkpointed(self, layer_id: int) -> bool:
        if not self.activation_checkpointing_enabled:
            return False
        return layer_id in self.checkpointed_layers
        
    def clear(self, device_id: Optional[int] = None):
        if device_id is not None:
            self.devices[device_id].clear()
        else:
            for device in self.devices.values():
                device.clear()
                
    def estimate_peak_memory(self, graph, total_memory_bytes: int = 0) -> int:
        peak = 0
        current = 0
        
        for node in graph.nodes.values():
            for shape in node.output_shapes:
                if shape:
                    numel = 1
                    for dim in shape:
                        numel *= dim
                    current += numel * 2
                    peak = max(peak, current)

            if node.node_type in (OpType.BACKWARD, OpType.OPTIMIZER):
                for shape in node.input_shapes:
                    if shape:
                        numel = 1
                        for dim in shape:
                            numel *= dim
                        current -= numel * 2
                        
        return peak

    def get_memory_summary(self) -> dict:
        summary = {
            "num_devices": self.num_devices,
            "memory_per_device_gb": self.memory_per_device_bytes / 1e9,
            "total_peak_gb": self.get_peak_memory() / 1e9,
            "total_current_gb": self.get_current_memory() / 1e9,
            "devices": {},
        }
        
        for device_id, device in self.devices.items():
            summary["devices"][device_id] = {
                "peak_gb": device.get_peak() / 1e9,
                "current_gb": device.current_usage / 1e9,
                "activations_gb": device.get_usage_by_type(MemoryType.ACTIVATION) / 1e9,
                "parameters_gb": device.get_usage_by_type(MemoryType.PARAMETER) / 1e9,
                "gradients_gb": device.get_usage_by_type(MemoryType.GRADIENT) / 1e9,
                "buffers_gb": device.get_usage_by_type(MemoryType.BUFFER) / 1e9,
            }
            
        return summary
