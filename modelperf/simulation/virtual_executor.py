"""Virtual Execution Engine for ModelPerf simulation."""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from modelperf.capture.graph import ComputationalGraph, GraphNode, OpType, CommType
from modelperf.symbolic.shape_inferer import SymbolicShapeInferer
from modelperf.simulation.roofline import RooflineModel
from modelperf.simulation.bandwidth import BandwidthModel
from modelperf.simulation.memory_tracker import MemoryTracker, MemoryType
from modelperf.simulation.execution_result import ExecutionResult, ExecutionConfig


@dataclass
class NodeMetrics:
    compute_time_ms: float = 0.0
    comm_time_ms: float = 0.0
    flops: int = 0
    comm_bytes: int = 0
    memory_bytes: int = 0


class VirtualExecutor:
    """Virtual execution engine for estimating training performance."""
    
    def __init__(
        self,
        config: Optional[ExecutionConfig] = None,
        roofline_model: Optional[RooflineModel] = None,
        bandwidth_model: Optional[BandwidthModel] = None,
        shape_inferer: Optional[SymbolicShapeInferer] = None,
    ):
        self.config = config or ExecutionConfig()
        self.roofline = roofline_model or self._create_default_roofline()
        self.bandwidth = bandwidth_model or self._create_default_bandwidth()
        self.inferer = shape_inferer or SymbolicShapeInferer()
        
        self.memory_tracker: Optional[MemoryTracker] = None
        self.node_metrics: Dict[str, NodeMetrics] = {}
        
    def _create_default_roofline(self) -> RooflineModel:
        return RooflineModel(
            peak_compute_tflops=self.config.peak_compute_tflops,
            peak_bandwidth_gbs=self.config.peak_bandwidth_gbs,
        )
    
    def _create_default_bandwidth(self) -> BandwidthModel:
        return BandwidthModel(
            bandwidth_gbs=self.config.network_bandwidth_gbs,
            latency_us=self.config.network_latency_us,
        )
    
    def execute(
        self,
        graph: ComputationalGraph,
        model_config: Optional[Dict] = None,
        strategy_config: Optional[Dict] = None,
        system_config: Optional[Dict] = None,
    ) -> ExecutionResult:
        """Execute virtual simulation on the computational graph."""
        mc = model_config or graph.model_config or {}
        sc = strategy_config or graph.strategy_config or {}
        sysc = system_config or graph.system_config or {}
        
        self._setup_memory_tracker(sc)
        self.node_metrics = {}
        
        forward_time = self._simulate_pass(
            graph.forward_nodes, graph, mc, sc, is_backward=False
        )
        backward_time = self._simulate_pass(
            graph.backward_nodes, graph, mc, sc, is_backward=True
        )
        optimizer_time = self._simulate_optimizer(graph, mc, sc)
        
        compute_time = self._aggregate_compute_time()
        comm_time = self._aggregate_comm_time()
        
        pp_size = sc.get('pp_size', self.config.pp_size)
        num_microbatches = self.config.num_microbatches
        
        bubble_time = self._calculate_bubble_time(
            forward_time, backward_time, pp_size, num_microbatches
        )
        
        iteration_time = self._calculate_iteration_time(
            forward_time, backward_time, optimizer_time, bubble_time,
            pp_size, num_microbatches
        )
        
        peak_memory = self._get_peak_memory()
        bottleneck = self._identify_bottleneck(
            compute_time, comm_time, bubble_time, peak_memory
        )
        
        return ExecutionResult(
            iteration_time_ms=iteration_time * 1000,
            compute_time_ms=compute_time * 1000,
            comm_time_ms=comm_time * 1000,
            peak_memory_mb=peak_memory,
            bottleneck=bottleneck,
            forward_time_ms=forward_time * 1000,
            backward_time_ms=backward_time * 1000,
            optimizer_time_ms=optimizer_time * 1000,
            compute_breakdown=self._get_compute_breakdown(),
            comm_breakdown=self._get_comm_breakdown(),
            memory_breakdown=self._get_memory_breakdown(),
            bubble_time_ms=bubble_time * 1000,
            num_microbatches=num_microbatches,
            pipeline_stages=pp_size,
        )
    
    def _setup_memory_tracker(self, strategy_config: Dict):
        num_devices = (
            strategy_config.get('dp_size', 1) *
            strategy_config.get('pp_size', 1) *
            strategy_config.get('tp_size', 1)
        )
        self.memory_tracker = MemoryTracker(
            num_devices=num_devices,
            memory_per_device_gb=self.config.memory_per_device_gb,
        )
        if self.config.enable_activation_checkpointing:
            self.memory_tracker.enable_activation_checkpointing(
                self.config.checkpointed_layers
            )
    
    def _simulate_pass(
        self,
        node_ids: List[str],
        graph: ComputationalGraph,
        model_config: Dict,
        strategy_config: Dict,
        is_backward: bool,
    ) -> float:
        total_time = 0.0
        tp_size = strategy_config.get('tp_size', 1)
        
        for node_id in node_ids:
            if node_id not in graph.nodes:
                continue
            node = graph.nodes[node_id]
            metrics = NodeMetrics()
            
            if node.node_type == OpType.COMPUTE:
                flops = self._estimate_flops(node, model_config, tp_size, is_backward)
                time_sec = self.roofline.estimate_time_simple(
                    flops=flops,
                    bytes_accessed=flops / 100,
                )
                metrics.compute_time_ms = time_sec * 1000
                metrics.flops = int(flops)
                
            elif node.node_type == OpType.COMMUNICATION:
                comm_bytes = self._get_comm_bytes(node, model_config)
                time_sec = self.bandwidth.estimate_comm_time(
                    comm_bytes=comm_bytes,
                    comm_size=node.comm_size,
                    collective_op=node.comm_type.value if node.comm_type else "all_reduce",
                )
                metrics.comm_time_ms = time_sec * 1000
                metrics.comm_bytes = int(comm_bytes)
            
            self._track_memory(node, model_config, is_backward)
            
            self.node_metrics[node_id] = metrics
            total_time += metrics.compute_time_ms / 1000 + metrics.comm_time_ms / 1000
        
        return total_time
    
    def _estimate_flops(
        self,
        node: GraphNode,
        model_config: Dict,
        tp_size: int,
        is_backward: bool,
    ) -> float:
        op_type = node.op_type.lower()
        batch = model_config.get('batch_size', 1)
        seq = model_config.get('sequence_length', 2048)
        hidden = model_config.get('hidden_size', 4096)
        heads = model_config.get('num_attention_heads', 32)
        kv_heads = model_config.get('num_query_groups', heads)
        ffn_hidden = model_config.get('ffn_hidden_size', hidden * 4)
        
        head_dim = hidden // heads
        
        flops = 0.0
        
        if 'linear' in op_type or 'fc' in op_type or 'dense' in op_type:
            in_features = hidden
            out_features = ffn_hidden
            if 'qkv' in op_type or 'attention' in op_type:
                out_features = 3 * hidden // tp_size
            flops = 2 * batch * seq * in_features * out_features
            
        elif 'attention' in op_type and 'score' in op_type or 'matmul' in op_type:
            flops = 2 * batch * heads * seq * seq * head_dim
            
        elif 'attention' in op_type and 'output' in op_type:
            flops = 2 * batch * seq * seq * heads * head_dim
            
        elif 'mlp' in op_type or 'ffn' in op_type:
            flops = 2 * batch * seq * hidden * (ffn_hidden // tp_size) * 2
            
        elif 'norm' in op_type or 'layernorm' in op_type:
            flops = 5 * batch * seq * hidden
        
        if is_backward:
            flops *= 2
            
        return flops
    
    def _get_comm_bytes(self, node: GraphNode, model_config: Dict) -> float:
        if node.comm_bytes > 0:
            return float(node.comm_bytes)
        
        tensor_shape = node.tensor_shape
        if tensor_shape:
            numel = 1
            for dim in tensor_shape:
                numel *= dim if isinstance(dim, (int, float)) else 1
            dtype_size = node.tensor_dtype_size
            return float(numel * dtype_size)
        
        return 0.0
    
    def _track_memory(self, node: GraphNode, model_config: Dict, is_backward: bool):
        if self.memory_tracker is None:
            return
        
        for device_id in range(self.memory_tracker.num_devices):
            if node.output_shapes:
                for shape in node.output_shapes:
                    if shape:
                        numel = 1
                        for dim in shape:
                            numel *= dim if isinstance(dim, (int, float)) else 1
                        size_bytes = int(numel * 2)
                        mem_type = MemoryType.ACTIVATION
                        if is_backward:
                            mem_type = MemoryType.GRADIENT
                        self.memory_tracker.allocate(
                            device_id=device_id,
                            name=f"{node.node_id}_out",
                            size_bytes=size_bytes,
                            mem_type=mem_type,
                        )
    
    def _simulate_optimizer(self, graph: ComputationalGraph, model_config: Dict, strategy_config: Dict) -> float:
        total_params = 0
        for node in graph.nodes.values():
            if node.params:
                for param_shape in node.params.values():
                    numel = 1
                    for dim in param_shape:
                        numel *= dim if isinstance(dim, (int, float)) else 1
                    total_params += numel
        
        optimizer_flops = total_params * 2
        time_sec = self.roofline.estimate_time_simple(
            flops=optimizer_flops,
            bytes_accessed=optimizer_flops / 100,
        )
        return time_sec
    
    def _aggregate_compute_time(self) -> float:
        return sum(
            m.compute_time_ms for m in self.node_metrics.values()
        ) / 1000
    
    def _aggregate_comm_time(self) -> float:
        return sum(
            m.comm_time_ms for m in self.node_metrics.values()
        ) / 1000
    
    def _calculate_bubble_time(
        self,
        forward_time: float,
        backward_time: float,
        pp_size: int,
        num_microbatches: int,
    ) -> float:
        if pp_size <= 1:
            return 0.0
        if num_microbatches < 1:
            num_microbatches = 1
        
        stage_time = forward_time + backward_time
        bubble = (pp_size - 1) / pp_size * stage_time / num_microbatches
        return bubble
    
    def _calculate_iteration_time(
        self,
        forward_time: float,
        backward_time: float,
        optimizer_time: float,
        bubble_time: float,
        pp_size: int,
        num_microbatches: int,
    ) -> float:
        if pp_size > 1:
            base_time = (forward_time + backward_time + optimizer_time) / num_microbatches
            return base_time + bubble_time
        else:
            return forward_time + backward_time + optimizer_time
    
    def _get_peak_memory(self) -> float:
        if self.memory_tracker is None:
            return 0.0
        peak_bytes = self.memory_tracker.get_peak_memory()
        return peak_bytes / (1024 * 1024)
    
    def _identify_bottleneck(
        self,
        compute_time: float,
        comm_time: float,
        bubble_time: float,
        peak_memory_mb: float,
    ) -> str:
        max_time = max(compute_time, comm_time, bubble_time)
        
        if max_time == compute_time:
            return "compute"
        elif max_time == comm_time:
            return "communication"
        elif max_time == bubble_time:
            return "pipeline_bubble"
        
        memory_limit_mb = self.config.memory_per_device_gb * 1024
        if peak_memory_mb > memory_limit_mb * 0.9:
            return "memory"
        
        return "balanced"
    
    def _get_compute_breakdown(self) -> Dict[str, float]:
        breakdown = {}
        for node_id, metrics in self.node_metrics.items():
            if metrics.compute_time_ms > 0:
                breakdown[node_id] = metrics.compute_time_ms
        return breakdown
    
    def _get_comm_breakdown(self) -> Dict[str, float]:
        breakdown = {}
        for node_id, metrics in self.node_metrics.items():
            if metrics.comm_time_ms > 0:
                breakdown[node_id] = metrics.comm_time_ms
        return breakdown
    
    def _get_memory_breakdown(self) -> Dict[str, float]:
        if self.memory_tracker is None:
            return {}
        return {
            "activation": self.memory_tracker.get_memory_by_type(MemoryType.ACTIVATION) / (1024 * 1024),
            "parameter": self.memory_tracker.get_memory_by_type(MemoryType.PARAMETER) / (1024 * 1024),
            "gradient": self.memory_tracker.get_memory_by_type(MemoryType.GRADIENT) / (1024 * 1024),
            "buffer": self.memory_tracker.get_memory_by_type(MemoryType.BUFFER) / (1024 * 1024),
            "temp": self.memory_tracker.get_memory_by_type(MemoryType.TEMP) / (1024 * 1024),
        }
