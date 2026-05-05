from typing import Dict, List, Optional, Tuple, Any, Callable
import torch
import torch.distributed as dist
from functools import wraps
from .graph import ComputationalGraph, GraphNode, OpType, CommType


def _get_comm_group_name(group: Any) -> str:
    if group is None:
        return "default"
    
    try:
        from megatron.core import parallel_state as mpu
        
        if group == mpu.get_tensor_model_parallel_group():
            return "tensor_parallel"
        elif group == mpu.get_data_parallel_group():
            return "data_parallel"
        elif group == mpu.get_pipeline_model_parallel_group():
            return "pipeline_parallel"
        elif hasattr(mpu, 'get_context_parallel_group'):
            if group == mpu.get_context_parallel_group():
                return "context_parallel"
        elif hasattr(mpu, 'get_expert_model_parallel_group'):
            if group == mpu.get_expert_model_parallel_group():
                return "expert_parallel"
        else:
            return "custom"
    except ImportError:
        return "unknown"
    except Exception:
        return "unknown"


def _get_tensor_info(tensor: Any) -> Tuple[Optional[Tuple], int, int, int]:
    if not isinstance(tensor, torch.Tensor):
        return None, 0, 2, 0
    
    shape = tuple(tensor.shape)
    numel = tensor.numel()
    dtype_size = 4 if tensor.dtype == torch.float32 else 2
    comm_bytes = numel * dtype_size
    
    return shape, numel, dtype_size, comm_bytes


def _get_dtype_size(tensor: Any) -> int:
    if isinstance(tensor, torch.Tensor):
        if tensor.dtype == torch.float32:
            return 4
        elif tensor.dtype == torch.float64:
            return 8
        elif tensor.dtype == torch.float16 or str(tensor.dtype).startswith("torch.bfloat16"):
            return 2
    return 2


class CommunicationCapture:
    _original_funcs: Dict[str, Callable] = {}
    _node_counter: int = 0
    
    def __init__(self, graph: Optional[ComputationalGraph] = None):
        self.graph = graph if graph else ComputationalGraph()
        self._is_active = False
        self._hooks_installed = False
        
    def _get_node_id(self) -> str:
        node_id = f"comm_{CommunicationCapture._node_counter}"
        CommunicationCapture._node_counter += 1
        return node_id
    
    def _create_all_reduce_wrapper(self, orig_func: Callable) -> Callable:
        @wraps(orig_func)
        def wrapper(tensor: torch.Tensor, op: Any = dist.ReduceOp.SUM, group: Any = None, async_op: bool = False):
            if self._is_active:
                node_id = self._get_node_id()
                shape, numel, dtype_size, comm_bytes = _get_tensor_info(tensor)
                comm_size = dist.get_world_size(group) if group else dist.get_world_size()
                group_name = _get_comm_group_name(group)
                
                node = GraphNode(
                    node_id=node_id,
                    op_type="all_reduce",
                    node_type=OpType.COMMUNICATION,
                    comm_type=CommType.ALL_REDUCE,
                    comm_size=comm_size,
                    comm_group=group_name,
                    tensor_shape=shape,
                    tensor_numel=numel,
                    tensor_dtype_size=dtype_size,
                    comm_bytes=comm_bytes,
                    async_op=async_op,
                )
                
                self.graph.nodes[node_id] = node
                self.graph.forward_nodes.append(node_id)
            
            return orig_func(tensor, op, group, async_op)
        
        return wrapper
    
    def _create_all_gather_wrapper(self, orig_func: Callable) -> Callable:
        @wraps(orig_func)
        def wrapper(output_tensor: torch.Tensor, input_tensor: torch.Tensor, group: Any = None, async_op: bool = False):
            if self._is_active:
                node_id = self._get_node_id()
                shape, numel, dtype_size, comm_bytes = _get_tensor_info(input_tensor)
                comm_size = dist.get_world_size(group) if group else dist.get_world_size()
                group_name = _get_comm_group_name(group)
                
                node = GraphNode(
                    node_id=node_id,
                    op_type="all_gather_into_tensor",
                    node_type=OpType.COMMUNICATION,
                    comm_type=CommType.ALL_GATHER,
                    comm_size=comm_size,
                    comm_group=group_name,
                    tensor_shape=shape,
                    tensor_numel=numel,
                    tensor_dtype_size=dtype_size,
                    comm_bytes=comm_bytes,
                    async_op=async_op,
                )
                
                self.graph.nodes[node_id] = node
                self.graph.forward_nodes.append(node_id)
            
            return orig_func(output_tensor, input_tensor, group, async_op)
        
        return wrapper
    
    def _create_reduce_scatter_wrapper(self, orig_func: Callable) -> Callable:
        @wraps(orig_func)
        def wrapper(output_tensor: torch.Tensor, input_tensor: torch.Tensor, op: Any = dist.ReduceOp.SUM, group: Any = None, async_op: bool = False):
            if self._is_active:
                node_id = self._get_node_id()
                shape, numel, dtype_size, comm_bytes = _get_tensor_info(input_tensor)
                comm_size = dist.get_world_size(group) if group else dist.get_world_size()
                group_name = _get_comm_group_name(group)
                
                node = GraphNode(
                    node_id=node_id,
                    op_type="reduce_scatter_tensor",
                    node_type=OpType.COMMUNICATION,
                    comm_type=CommType.REDUCE_SCATTER,
                    comm_size=comm_size,
                    comm_group=group_name,
                    tensor_shape=shape,
                    tensor_numel=numel,
                    tensor_dtype_size=dtype_size,
                    comm_bytes=comm_bytes,
                    async_op=async_op,
                )
                
                self.graph.nodes[node_id] = node
                self.graph.forward_nodes.append(node_id)
            
            return orig_func(output_tensor, input_tensor, op, group, async_op)
        
        return wrapper
    
    def _create_all_to_all_wrapper(self, orig_func: Callable) -> Callable:
        @wraps(orig_func)
        def wrapper(output_tensor: torch.Tensor, input_tensor: torch.Tensor, group: Any = None, async_op: bool = False):
            if self._is_active:
                node_id = self._get_node_id()
                shape, numel, dtype_size, comm_bytes = _get_tensor_info(input_tensor)
                comm_size = dist.get_world_size(group) if group else dist.get_world_size()
                group_name = _get_comm_group_name(group)
                
                node = GraphNode(
                    node_id=node_id,
                    op_type="all_to_all_single",
                    node_type=OpType.COMMUNICATION,
                    comm_type=CommType.ALL_TO_ALL,
                    comm_size=comm_size,
                    comm_group=group_name,
                    tensor_shape=shape,
                    tensor_numel=numel,
                    tensor_dtype_size=dtype_size,
                    comm_bytes=comm_bytes,
                    async_op=async_op,
                )
                
                self.graph.nodes[node_id] = node
                self.graph.forward_nodes.append(node_id)
            
            return orig_func(output_tensor, input_tensor, group, async_op)
        
        return wrapper
    
    def _create_broadcast_wrapper(self, orig_func: Callable) -> Callable:
        @wraps(orig_func)
        def wrapper(tensor: torch.Tensor, src: int = 0, group: Any = None, async_op: bool = False):
            if self._is_active:
                node_id = self._get_node_id()
                shape, numel, dtype_size, comm_bytes = _get_tensor_info(tensor)
                comm_size = dist.get_world_size(group) if group else dist.get_world_size()
                group_name = _get_comm_group_name(group)
                
                node = GraphNode(
                    node_id=node_id,
                    op_type="broadcast",
                    node_type=OpType.COMMUNICATION,
                    comm_type=CommType.BROADCAST,
                    comm_size=comm_size,
                    comm_group=group_name,
                    tensor_shape=shape,
                    tensor_numel=numel,
                    tensor_dtype_size=dtype_size,
                    comm_bytes=comm_bytes,
                    async_op=async_op,
                )
                
                self.graph.nodes[node_id] = node
                self.graph.forward_nodes.append(node_id)
            
            return orig_func(tensor, src, group, async_op)
        
        return wrapper
    
    def install_hooks(self):
        if self._hooks_installed:
            return
        
        CommunicationCapture._original_funcs['all_reduce'] = dist.all_reduce
        CommunicationCapture._original_funcs['all_gather_into_tensor'] = dist.all_gather_into_tensor
        CommunicationCapture._original_funcs['reduce_scatter_tensor'] = dist.reduce_scatter_tensor
        CommunicationCapture._original_funcs['all_to_all_single'] = dist.all_to_all_single
        CommunicationCapture._original_funcs['broadcast'] = dist.broadcast
        
        dist.all_reduce = self._create_all_reduce_wrapper(dist.all_reduce)
        dist.all_gather_into_tensor = self._create_all_gather_wrapper(dist.all_gather_into_tensor)
        dist.reduce_scatter_tensor = self._create_reduce_scatter_wrapper(dist.reduce_scatter_tensor)
        dist.all_to_all_single = self._create_all_to_all_wrapper(dist.all_to_all_single)
        dist.broadcast = self._create_broadcast_wrapper(dist.broadcast)
        
        if hasattr(dist, 'batch_isend_irecv'):
            CommunicationCapture._original_funcs['batch_isend_irecv'] = dist.batch_isend_irecv
            
        self._hooks_installed = True
    
    def uninstall_hooks(self):
        if not self._hooks_installed:
            return
        
        if 'all_reduce' in CommunicationCapture._original_funcs:
            dist.all_reduce = CommunicationCapture._original_funcs['all_reduce']
        if 'all_gather_into_tensor' in CommunicationCapture._original_funcs:
            dist.all_gather_into_tensor = CommunicationCapture._original_funcs['all_gather_into_tensor']
        if 'reduce_scatter_tensor' in CommunicationCapture._original_funcs:
            dist.reduce_scatter_tensor = CommunicationCapture._original_funcs['reduce_scatter_tensor']
        if 'all_to_all_single' in CommunicationCapture._original_funcs:
            dist.all_to_all_single = CommunicationCapture._original_funcs['all_to_all_single']
        if 'broadcast' in CommunicationCapture._original_funcs:
            dist.broadcast = CommunicationCapture._original_funcs['broadcast']
        
        self._hooks_installed = False
    
    def start(self):
        self._is_active = True
        self.install_hooks()
    
    def stop(self):
        self._is_active = False
    
    def reset(self):
        self.graph = ComputationalGraph()
        CommunicationCapture._node_counter = 0
    
    def get_graph(self) -> ComputationalGraph:
        return self.graph
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


def install_communication_hooks(graph: Optional[ComputationalGraph] = None) -> CommunicationCapture:
    capture = CommunicationCapture(graph)
    capture.start()
    return capture
