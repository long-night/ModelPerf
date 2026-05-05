from typing import Dict, List, Optional, Tuple, Any
import torch
from .graph import GraphNode, OpType, ComputationalGraph


class BackwardCapture:
    def __init__(self, graph: ComputationalGraph):
        self.graph = graph
        self.enabled = False
        self.tensor_hooks: Dict[str, Any] = {}
        self.node_map: Dict[str, str] = {}
        self.backward_nodes: List[str] = []
        self.tensor_to_node: Dict[int, str] = {}

    def register_tensor_hook(self, tensor: torch.Tensor, node_id: str):
        if not self.enabled or not tensor.requires_grad:
            return

        self.tensor_to_node[id(tensor)] = node_id

        def hook(grad: torch.Tensor) -> torch.Tensor:
            if not self.enabled:
                return grad

            bwd_node_id = f"{node_id}_bwd_{len(self.backward_nodes)}"
            grad_shape = tuple(grad.shape) if grad is not None else None

            bwd_node = GraphNode(
                node_id=bwd_node_id,
                op_type="backward_hook",
                node_type=OpType.BACKWARD,
                output_shapes=[grad_shape] if grad_shape else [],
                grad_output_shapes=[grad_shape] if grad_shape else [],
            )

            self.graph.nodes[bwd_node_id] = bwd_node
            self.graph.add_edge(node_id, bwd_node_id)
            self.backward_nodes.append(bwd_node_id)

            return grad

        handle = tensor.register_hook(hook)
        self.tensor_hooks[node_id] = handle

    def capture_module_backward(self, module: torch.nn.Module, forward_node_id: str):
        if not self.enabled:
            return

        for name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                self.register_tensor_hook(param, f"{forward_node_id}_param_{name}")

        for name, buffer in module.named_buffers(recurse=False):
            if buffer.requires_grad:
                self.register_tensor_hook(buffer, f"{forward_node_id}_buffer_{name}")

    def register_output_hooks(self, outputs: torch.Tensor, node_id: str):
        if not self.enabled:
            return

        if isinstance(outputs, torch.Tensor):
            self.register_tensor_hook(outputs, node_id)
        elif isinstance(outputs, (tuple, list)):
            for i, out in enumerate(outputs):
                if isinstance(out, torch.Tensor):
                    self.register_tensor_hook(out, f"{node_id}_output_{i}")

    def get_backward_nodes(self) -> List[str]:
        return self.backward_nodes.copy()

    def clear_hooks(self):
        for handle in self.tensor_hooks.values():
            try:
                handle.remove()
            except (AttributeError, RuntimeError):
                pass
        self.tensor_hooks.clear()
        self.tensor_to_node.clear()

    def start(self):
        self.enabled = True
        self.backward_nodes.clear()

    def stop(self):
        self.enabled = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.clear_hooks()
        return False
