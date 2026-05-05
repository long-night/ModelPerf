from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
from .graph import ComputationalGraph, GraphNode, OpType, CommType


def _get_tensor_shape(t: Any) -> Optional[Tuple]:
    if isinstance(t, torch.Tensor):
        return tuple(t.shape)
    return None


def _get_tensor_dtype_size(t: Any) -> int:
    if isinstance(t, torch.Tensor) and t.dtype.is_floating_point:
        return 4 if t.dtype == torch.float32 else 2
    return 2


def _extract_shapes(args: Any) -> List[Optional[Tuple]]:
    shapes = []
    if isinstance(args, (tuple, list)):
        for arg in args:
            shapes.append(_get_tensor_shape(arg))
    elif isinstance(args, dict):
        for k in sorted(args.keys()):
            shapes.append(_get_tensor_shape(args[k]))
    else:
        shapes.append(_get_tensor_shape(args))
    return shapes


class ModuleCapture:
    def __init__(self, graph: Optional[ComputationalGraph] = None):
        self.graph = graph if graph else ComputationalGraph()
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._hook_handles: Dict[str, Any] = {}
        self._forward_stack: List[str] = []
        self._node_counter = 0
        self._is_active = False
        self._in_backward = False

    def _get_node_id(self) -> str:
        node_id = f"mod_{self._node_counter}"
        self._node_counter += 1
        return node_id

    def _create_forward_hook(self, module_path: str, module_class: str):
        def hook(module: nn.Module, input_args, output):
            if not self._is_active:
                return output
            
            self._forward_stack.append(module_path)
            
            node_id = self._get_node_id()
            input_shapes = _extract_shapes(input_args)
            output_shapes = _extract_shapes(output)
            
            params: Dict[str, Tuple] = {}
            for name, param in module.named_parameters(recurse=False):
                if param is not None:
                    params[name] = tuple(param.shape)
            
            phase = OpType.AUTOGRAD_BWD if self._in_backward else OpType.AUTOGRAD_FWD
            
            node = GraphNode(
                node_id=node_id,
                op_type=module_class,
                node_type=phase,
                module_path=module_path,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                params=params,
            )
            
            self.graph.nodes[node_id] = node
            self.graph.forward_nodes.append(node_id)
            
            return output
        
        return hook

    def _create_backward_hook(self, module_path: str):
        def hook(module: nn.Module, grad_input, grad_output):
            if not self._is_active:
                return
            
            if module_path in self._forward_stack:
                self._forward_stack.remove(module_path)
        
        return hook

    def register_module(self, module: nn.Module, path: str = ""):
        for name, child in module.named_children():
            child_path = f"{path}.{name}" if path else name
            child_class = child.__class__.__name__
            
            forward_hook = self._create_forward_hook(child_path, child_class)
            backward_hook = self._create_backward_hook(child_path)
            
            handle_fwd = child.register_forward_hook(forward_hook)
            handle_bwd = child.register_full_backward_hook(backward_hook)
            
            self._hooks.extend([handle_fwd, handle_bwd])
            self._hook_handles[child_path] = (handle_fwd, handle_bwd)
            
            self.register_module(child, child_path)

    def capture(self, model: nn.Module, sample_input: Any = None):
        self.register_module(model)
        return self.graph

    def start(self):
        self._is_active = True
        self._forward_stack = []

    def stop(self):
        self._is_active = False

    def reset(self):
        self.graph = ComputationalGraph()
        self._node_counter = 0
        self._forward_stack = []
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
        self._hook_handles = {}

    def get_graph(self) -> ComputationalGraph:
        return self.graph

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
