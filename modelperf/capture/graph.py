from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


class OpType(Enum):
    COMPUTE = "compute"
    COMMUNICATION = "communication"
    MEMORY = "memory"
    AUTOGRAD_FWD = "autograd_fwd"
    AUTOGRAD_BWD = "autograd_bwd"
    BACKWARD = "backward"
    OPTIMIZER = "optimizer"


class CommType(Enum):
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_TO_ALL = "all_to_all"
    P2P_SEND = "p2p_send"
    P2P_RECV = "p2p_recv"
    BROADCAST = "broadcast"


@dataclass
class GraphNode:
    node_id: str
    op_type: str
    node_type: OpType
    module_path: Optional[str] = None
    input_shapes: List[Optional[Tuple]] = field(default_factory=list)
    output_shapes: List[Optional[Tuple]] = field(default_factory=list)
    symbolic_input_shapes: List[Any] = field(default_factory=list)
    symbolic_output_shapes: List[Any] = field(default_factory=list)
    params: Dict[str, Tuple] = field(default_factory=dict)
    comm_type: Optional[CommType] = None
    comm_size: int = 1
    comm_group: str = ""
    tensor_shape: Optional[Tuple] = None
    tensor_numel: int = 0
    tensor_dtype_size: int = 2
    comm_bytes: int = 0
    symbolic_comm_bytes: Optional[Any] = None
    async_op: bool = False
    call_context: Optional[Tuple] = None
    evaluated_time_ms: float = 0.0
    evaluated_flops: int = 0
    evaluated_mem_bytes: int = 0
    bottleneck: str = ""
    grad_input_shapes: List[Optional[Tuple]] = field(default_factory=list)
    grad_output_shapes: List[Optional[Tuple]] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "op_type": self.op_type,
            "node_type": self.node_type.value,
            "module_path": self.module_path,
            "input_shapes": [list(s) if s else None for s in self.input_shapes],
            "output_shapes": [list(s) if s else None for s in self.output_shapes],
            "symbolic_input_shapes": [str(s) for s in self.symbolic_input_shapes],
            "symbolic_output_shapes": [str(s) for s in self.symbolic_output_shapes],
            "params": {k: list(v) for k, v in self.params.items()},
            "comm_type": self.comm_type.value if self.comm_type else None,
            "comm_size": self.comm_size,
            "comm_group": self.comm_group,
            "tensor_shape": list(self.tensor_shape) if self.tensor_shape else None,
            "tensor_numel": self.tensor_numel,
            "comm_bytes": self.comm_bytes,
            "symbolic_comm_bytes": str(self.symbolic_comm_bytes) if self.symbolic_comm_bytes else None,
            "async_op": self.async_op,
            "evaluated_time_ms": self.evaluated_time_ms,
            "evaluated_flops": self.evaluated_flops,
            "inputs": self.inputs,
            "metadata": self.metadata,
        }


@dataclass
class ComputationalGraph:
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: List[Tuple[str, str]] = field(default_factory=list)
    model_config: Dict = field(default_factory=dict)
    strategy_config: Dict = field(default_factory=dict)
    system_config: Dict = field(default_factory=dict)
    capture_timestamp: Optional[str] = None
    framework_version: str = ""
    forward_nodes: List[str] = field(default_factory=list)
    backward_nodes: List[str] = field(default_factory=list)

    def add_node(self, **kwargs) -> GraphNode:
        node_id = kwargs.pop("node_id", f"node_{len(self.nodes)}")
        node = GraphNode(node_id=node_id, **kwargs)
        self.nodes[node_id] = node
        return node

    def add_edge(self, from_id: str, to_id: str):
        self.edges.append((from_id, to_id))
        if from_id in self.nodes:
            self.nodes[from_id].outputs.append(to_id)
        if to_id in self.nodes:
            self.nodes[to_id].inputs.append(from_id)

    def get_nodes_by_type(self, node_type: OpType) -> List[GraphNode]:
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def get_comm_nodes(self) -> List[GraphNode]:
        return self.get_nodes_by_type(OpType.COMMUNICATION)

    def get_compute_nodes(self) -> List[GraphNode]:
        return self.get_nodes_by_type(OpType.COMPUTE)

    def summary(self) -> Dict:
        return {
            "total_nodes": len(self.nodes),
            "compute_nodes": len(self.get_compute_nodes()),
            "comm_nodes": len(self.get_comm_nodes()),
            "forward_nodes": len(self.forward_nodes),
            "backward_nodes": len(self.backward_nodes),
            "edges": len(self.edges),
        }

    def to_dict(self) -> Dict:
        return {
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": self.edges,
            "model_config": self.model_config,
            "strategy_config": self.strategy_config,
            "system_config": self.system_config,
            "summary": self.summary(),
        }
