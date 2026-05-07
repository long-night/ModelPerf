from typing import Dict, List, Optional, Tuple, Any, Callable
import torch
import torch.nn as nn
from .graph import ComputationalGraph, GraphNode, OpType, CommType
from .hierarchy import infer_module_category


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
    def __init__(
        self,
        graph: Optional[ComputationalGraph] = None,
        backward_capture: Optional[Any] = None,
    ):
        self.graph = graph if graph else ComputationalGraph()
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._hook_handles: Dict[str, Any] = {}
        self._forward_stack: List[str] = []
        self._module_stack: List[Tuple[str, str]] = []
        self._node_counter = 0
        self._is_active = False
        self._in_backward = False
        self.backward_capture = backward_capture
        self._node_output_map: Dict[str, Any] = {}
        self._current_node_id: Optional[str] = None
        self._on_node_created_callbacks: List[Callable[[str], None]] = []
        self._module_node_map: Dict[str, str] = {}

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

            parent_module_id = None
            if self._module_stack:
                parent_module_id = self._module_stack[-1][1]

            module_level = len(module_path.split(".")) if module_path else 0
            category = infer_module_category(module_class, module_path)

            node = GraphNode(
                node_id=node_id,
                op_type=module_class,
                node_type=phase,
                module_path=module_path,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                params=params,
                hierarchy_path=module_path,
                parent_module_id=parent_module_id,
                module_level=module_level,
                module_category=category.value,
            )

            self.graph.nodes[node_id] = node
            self.graph.forward_nodes.append(node_id)
            self._current_node_id = node_id
            self._node_output_map[node_id] = output
            self._module_stack.append((module_path, node_id))
            self._module_node_map[module_path] = node_id

            for cb in self._on_node_created_callbacks:
                cb(node_id)

            if self.backward_capture is not None and self.backward_capture.enabled:
                self._register_output_backward_hooks(output, node_id)

            if len(self._forward_stack) > 1:
                prev_path = self._forward_stack[-2]
                prev_node_id = self._find_node_by_module_path(prev_path)
                if prev_node_id is not None:
                    self.graph.add_edge(prev_node_id, node_id)

            return output

        return hook

    def _create_forward_post_hook(self, module_path: str):
        def hook(module: nn.Module, input_args, output):
            if self._is_active:
                if self._module_stack and self._module_stack[-1][0] == module_path:
                    self._module_stack.pop()
            return output
        return hook

    def _find_node_by_module_path(self, module_path: str) -> Optional[str]:
        candidates = [
            nid for nid, node in self.graph.nodes.items()
            if node.module_path == module_path and nid in self.graph.forward_nodes
        ]
        return candidates[-1] if candidates else None

    def _register_output_backward_hooks(self, output: Any, node_id: str):
        if isinstance(output, torch.Tensor):
            if output.requires_grad:
                self.backward_capture.register_tensor_hook(output, node_id)
        elif isinstance(output, (tuple, list)):
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor) and out.requires_grad:
                    self.backward_capture.register_tensor_hook(out, f"{node_id}_out{i}")

    def register_module(self, module: nn.Module, path: str = ""):
        for name, child in module.named_children():
            child_path = f"{path}.{name}" if path else name
            child_class = child.__class__.__name__

            forward_hook = self._create_forward_hook(child_path, child_class)
            post_hook = self._create_forward_post_hook(child_path)

            handle_fwd = child.register_forward_hook(forward_hook)
            handle_post = child.register_forward_hook(post_hook)

            self._hooks.append(handle_fwd)
            self._hooks.append(handle_post)
            self._hook_handles[child_path] = handle_fwd

            self.register_module(child, child_path)

    def register_on_node_created(self, callback: Callable[[str], None]):
        self._on_node_created_callbacks.append(callback)

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
        self._module_stack.clear()
        self._module_node_map.clear()
        self._node_output_map.clear()
        self._current_node_id = None
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
