"""Bandwidth Model for communication time estimation."""

from enum import Enum
from typing import Optional


class CollectiveOp(Enum):
    """Collective communication operations."""
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_TO_ALL = "all_to_all"
    BROADCAST = "broadcast"
    REDUCE = "reduce"


class Algorithm(Enum):
    """Communication algorithms."""
    RING = "ring"
    TREE = "tree"
    DIRECT = "direct"
    CHAINED = "chained"


class BandwidthModel:
    """Bandwidth model for estimating collective communication time.
    
    Models the time for collective operations based on algorithm, bandwidth,
    latency, and number of participating ranks.
    """
    
    # Network defaults (NVLink A100)
    DEFAULT_BANDWIDTH_GBS = 600.0  # NVLink 3.0 per link
    DEFAULT_LATENCY_US = 2.0  # Software overhead
    
    def __init__(
        self,
        bandwidth_gbs: float = DEFAULT_BANDWIDTH_GBS,
        latency_us: float = DEFAULT_LATENCY_US,
    ):
        """Initialize bandwidth model.
        
        Args:
            bandwidth_gbs: Per-link bandwidth in GB/s
            latency_us: Base latency in microseconds
        """
        self.bandwidth_gbs = bandwidth_gbs
        self.latency_us = latency_us
    
    def estimate_comm_time(
        self,
        comm_bytes: float,
        comm_size: int,
        bandwidth: Optional[float] = None,
        latency: Optional[float] = None,
        algorithm: str = "ring",
        collective_op: str = "all_reduce",
    ) -> float:
        """Estimate communication time for a collective operation.
        
        Args:
            comm_bytes: Total bytes per rank to communicate
            comm_size: Number of ranks in communication group
            bandwidth: Override bandwidth (GB/s), or None for default
            latency: Override latency (us), or None for default
            algorithm: Communication algorithm ("ring", "tree", "direct")
            collective_op: Collective operation type ("all_reduce", "all_gather", etc.)
            
        Returns:
            Estimated communication time in seconds
            
        Raises:
            ValueError: If comm_size < 1 or comm_bytes < 0
        """
        if comm_size < 1:
            raise ValueError(f"comm_size must be >= 1, got {comm_size}")
        if comm_bytes < 0:
            raise ValueError(f"comm_bytes must be >= 0, got {comm_bytes}")
        
        if comm_size == 1:
            return 0.0
        
        bw = bandwidth if bandwidth is not None else self.bandwidth_gbs
        lat = latency if latency is not None else self.latency_us
        lat_sec = lat * 1e-6
        
        if comm_bytes == 0:
            return lat_sec
        
        # Convert bytes to gigabytes for bandwidth calculation
        bytes_gb = comm_bytes / 1e9
        
        # Route to specific collective implementation
        op = collective_op.lower()
        algo = algorithm.lower()
        
        if op == "all_reduce":
            return self._all_reduce_time(bytes_gb, comm_size, bw, lat_sec, algo)
        elif op == "all_gather":
            return self._all_gather_time(bytes_gb, comm_size, bw, lat_sec, algo)
        elif op == "reduce_scatter":
            return self._reduce_scatter_time(bytes_gb, comm_size, bw, lat_sec, algo)
        elif op == "all_to_all":
            return self._all_to_all_time(bytes_gb, comm_size, bw, lat_sec, algo)
        elif op == "broadcast":
            return self._broadcast_time(bytes_gb, comm_size, bw, lat_sec, algo)
        elif op == "reduce":
            return self._reduce_time(bytes_gb, comm_size, bw, lat_sec, algo)
        else:
            raise ValueError(f"Unknown collective operation: {collective_op}")
    
    def _all_reduce_time(
        self, bytes_gb: float, n: int, bw: float, lat: float, algo: str
    ) -> float:
        """Estimate All-Reduce time.
        
        Ring: time = 2*(n-1)/n * bytes / bw + latency
        Tree: time = 2 * bytes / bw + latency
        """
        if algo == "ring":
            factor = 2.0 * (n - 1) / n
        elif algo in ("tree", "direct"):
            factor = 2.0
        else:
            factor = 2.0 * (n - 1) / n  # Default to ring
        
        return factor * bytes_gb / bw + lat
    
    def _all_gather_time(
        self, bytes_gb: float, n: int, bw: float, lat: float, algo: str
    ) -> float:
        """Estimate All-Gather time.
        
        Ring: time = (n-1)/n * bytes / bw + latency
        """
        factor = (n - 1) / n
        return factor * bytes_gb / bw + lat
    
    def _reduce_scatter_time(
        self, bytes_gb: float, n: int, bw: float, lat: float, algo: str
    ) -> float:
        """Estimate Reduce-Scatter time.
        
        Ring: time = (n-1)/n * bytes / bw + latency
        """
        factor = (n - 1) / n
        return factor * bytes_gb / bw + lat
    
    def _all_to_all_time(
        self, bytes_gb: float, n: int, bw: float, lat: float, algo: str
    ) -> float:
        """Estimate All-to-All time.
        
        time = (n-1)/n * bytes / bw + latency
        """
        factor = (n - 1) / n
        return factor * bytes_gb / bw + lat
    
    def _broadcast_time(
        self, bytes_gb: float, n: int, bw: float, lat: float, algo: str
    ) -> float:
        """Estimate Broadcast time.
        
        Tree: time = log2(n) * bytes / bw + latency
        Chain: time = (n-1) * bytes / bw + latency
        """
        if algo in ("tree", "direct"):
            import math
            factor = math.log2(n) if n > 1 else 0
        else:
            factor = n - 1
        return factor * bytes_gb / bw + lat
    
    def _reduce_time(
        self, bytes_gb: float, n: int, bw: float, lat: float, algo: str
    ) -> float:
        """Estimate Reduce time.
        
        Tree: time = log2(n) * bytes / bw + latency
        """
        if algo in ("tree", "direct"):
            import math
            factor = math.log2(n) if n > 1 else 0
        else:
            factor = n - 1
        return factor * bytes_gb / bw + lat


# Factory functions for common network configurations

def create_nvlink_model(nvlink_version: int = 3):
    """Create BandwidthModel for NVLink.
    
    Args:
        nvlink_version: NVLink version (2 for V100, 3 for A100, 4 for H100)
    """
    configs = {
        2: {"bandwidth_gbs": 300.0, "latency_us": 2.0},
        3: {"bandwidth_gbs": 600.0, "latency_us": 2.0},
        4: {"bandwidth_gbs": 900.0, "latency_us": 1.5},
    }
    cfg = configs.get(nvlink_version, configs[3])
    return BandwidthModel(**cfg)


def create_infiniband_model(ib_speed: str = "hdr"):
    """Create BandwidthModel for InfiniBand.
    
    Args:
        ib_speed: IB speed ('sdr', 'ddr', 'qdr', 'fdr', 'edr', 'hdr', 'ndr')
    """
    speeds = {
        "sdr": 2.5,
        "ddr": 5.0,
        "qdr": 10.0,
        "fdr": 14.0,
        "edr": 25.0,
        "hdr": 50.0,
        "ndr": 100.0,
    }
    bw = speeds.get(ib_speed, 50.0)
    return BandwidthModel(bandwidth_gbs=bw, latency_us=5.0)


def create_ethernet_model(speed_gbps: float = 100.0):
    """Create BandwidthModel for Ethernet.
    
    Args:
        speed_gbps: Ethernet speed in Gbps (25, 100, etc.)
    """
    # Convert Gbps to GB/s (divide by 8)
    bw = speed_gbps / 8.0
    return BandwidthModel(bandwidth_gbs=bw, latency_us=10.0)
