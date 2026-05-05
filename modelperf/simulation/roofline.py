"""Roofline Model for compute time estimation."""

from typing import Optional


class RooflineModel:
    """Roofline performance model for estimating compute kernel execution time.
    
    The Roofline model estimates execution time based on the bottleneck between
    compute throughput and memory bandwidth.
    """
    
    # GPU defaults (A100)
    DEFAULT_PEAK_COMPUTE_TFLOPS = 312.0  # FP16
    DEFAULT_PEAK_BANDWIDTH_GBS = 2039.0  # HBM bandwidth
    
    def __init__(
        self,
        peak_compute_tflops: float = DEFAULT_PEAK_COMPUTE_TFLOPS,
        peak_bandwidth_gbs: float = DEFAULT_PEAK_BANDWIDTH_GBS,
    ):
        """Initialize Roofline model.
        
        Args:
            peak_compute_tflops: Peak compute throughput in TFLOPS
            peak_bandwidth_gbs: Peak memory bandwidth in GB/s
        """
        self.peak_compute_tflops = peak_compute_tflops
        self.peak_bandwidth_gbs = peak_bandwidth_gbs
    
    def estimate_time(
        self,
        flops: float,
        arithmetic_intensity: float,
        peak_compute: Optional[float] = None,
        peak_bandwidth: Optional[float] = None,
    ) -> float:
        """Estimate compute kernel execution time using Roofline model.
        
        Time = max(FLOPs / peak_compute, bytes / peak_bandwidth)
        where bytes = FLOPs / arithmetic_intensity
        
        Args:
            flops: Number of floating-point operations
            arithmetic_intensity: FLOPs per byte (compute intensity)
            peak_compute: Override peak compute (TFLOPS), or None to use default
            peak_bandwidth: Override peak bandwidth (GB/s), or None to use default
            
        Returns:
            Estimated execution time in seconds
            
        Raises:
            ValueError: If arithmetic_intensity is zero or negative
        """
        if arithmetic_intensity <= 0:
            raise ValueError(f"arithmetic_intensity must be positive, got {arithmetic_intensity}")
        
        # Handle zero FLOPs edge case
        if flops <= 0:
            return 0.0
        
        # Use provided overrides or defaults
        pc = peak_compute if peak_compute is not None else self.peak_compute_tflops
        pb = peak_bandwidth if peak_bandwidth is not None else self.peak_bandwidth_gbs
        
        # Convert units for calculation
        # pc is in TFLOPS = 1e12 FLOPS
        # pb is in GB/s = 1e9 bytes/s
        
        # Compute time: FLOPs / (TFLOPS * 1e12) = seconds
        compute_time = flops / (pc * 1e12)
        
        # Memory bytes accessed
        bytes_accessed = flops / arithmetic_intensity
        
        # Memory time: bytes / (GB/s * 1e9) = seconds
        memory_time = bytes_accessed / (pb * 1e9)
        
        # Roofline: max of compute and memory bound
        return max(compute_time, memory_time)
    
    def estimate_time_simple(
        self,
        flops: float,
        bytes_accessed: float,
        peak_compute: Optional[float] = None,
        peak_bandwidth: Optional[float] = None,
    ) -> float:
        """Estimate time using explicit FLOPs and bytes (alternative interface).
        
        Args:
            flops: Number of floating-point operations
            bytes_accessed: Total bytes moved from/to memory
            peak_compute: Override peak compute (TFLOPS)
            peak_bandwidth: Override peak bandwidth (GB/s)
            
        Returns:
            Estimated execution time in seconds
        """
        if flops <= 0 or bytes_accessed <= 0:
            return 0.0
        
        pc = peak_compute if peak_compute is not None else self.peak_compute_tflops
        pb = peak_bandwidth if peak_bandwidth is not None else self.peak_bandwidth_gbs
        
        compute_time = flops / (pc * 1e12)
        memory_time = bytes_accessed / (pb * 1e9)
        
        return max(compute_time, memory_time)


# Convenience functions for common GPU types

def create_a100_model():
    """Create RooflineModel for NVIDIA A100."""
    return RooflineModel(
        peak_compute_tflops=312.0,  # FP16
        peak_bandwidth_gbs=2039.0,
    )


def create_h100_model():
    """Create RooflineModel for NVIDIA H100."""
    return RooflineModel(
        peak_compute_tflops=989.0,  # FP16
        peak_bandwidth_gbs=3350.0,
    )


def create_v100_model():
    """Create RooflineModel for NVIDIA V100."""
    return RooflineModel(
        peak_compute_tflops=125.0,  # FP16
        peak_bandwidth_gbs=900.0,
    )
