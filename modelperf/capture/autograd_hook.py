from typing import Dict, List, Optional, Tuple, Any, Callable
import torch
from .graph import GraphNode, OpType, ComputationalGraph


class AutogradCapture:
    def __init__(self, graph: ComputationalGraph):
        self.graph = graph
        self.patched_functions: Dict[type, Tuple[Callable, Callable]] = {}
        self.node_map: Dict[str, str] = {}
        self.enabled = False

    def patch_function(self, func_class: type, node_prefix: str = "autograd"):
        if not hasattr(func_class, 'forward') or not hasattr(func_class, 'backward'):
            return

        original_forward = func_class.forward
        original_backward = func_class.backward
        capture = self

        @staticmethod
        def wrapped_forward(ctx, *args, **kwargs):
            if not capture.enabled:
                return original_forward(ctx, *args, **kwargs)

            node_id = f"{node_prefix}_{func_class.__name__}_{len(capture.graph.nodes)}"
            ctx._capture_node_id = node_id

            input_shapes = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    input_shapes.append(tuple(arg.shape))
                else:
                    input_shapes.append(None)

            outputs = original_forward(ctx, *args, **kwargs)

            output_shapes = []
            if isinstance(outputs, torch.Tensor):
                output_shapes.append(tuple(outputs.shape))
            elif isinstance(outputs, (tuple, list)):
                for out in outputs:
                    if isinstance(out, torch.Tensor):
                        output_shapes.append(tuple(out.shape))
                    else:
                        output_shapes.append(None)

            node = GraphNode(
                node_id=node_id,
                op_type=f"{func_class.__name__}.forward",
                node_type=OpType.AUTOGRAD_FWD,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
            )
            capture.graph.nodes[node_id] = node
            capture.node_map[node_id] = node_id

            return outputs

        @staticmethod
        def wrapped_backward(ctx, *grad_outputs):
            if not capture.enabled:
                return original_backward(ctx, *grad_outputs)

            node_id = getattr(ctx, '_capture_node_id', None)
            if node_id:
                bwd_node_id = f"{node_id}_backward"

                grad_input_shapes = []
                for grad in grad_outputs:
                    if isinstance(grad, torch.Tensor):
                        grad_input_shapes.append(tuple(grad.shape))
                    else:
                        grad_input_shapes.append(None)

                grad_inputs = original_backward(ctx, *grad_outputs)

                grad_output_shapes = []
                if isinstance(grad_inputs, torch.Tensor):
                    grad_output_shapes.append(tuple(grad_inputs.shape))
                elif isinstance(grad_inputs, (tuple, list)):
                    for grad in grad_inputs:
                        if isinstance(grad, torch.Tensor):
                            grad_output_shapes.append(tuple(grad.shape))
                        else:
                            grad_output_shapes.append(None)

                bwd_node = GraphNode(
                    node_id=bwd_node_id,
                    op_type=f"{func_class.__name__}.backward",
                    node_type=OpType.AUTOGRAD_BWD,
                    input_shapes=grad_input_shapes,
                    output_shapes=grad_output_shapes,
                    grad_input_shapes=grad_input_shapes,
                    grad_output_shapes=grad_output_shapes,
                )
                capture.graph.nodes[bwd_node_id] = bwd_node
                capture.graph.add_edge(node_id, bwd_node_id)
            else:
                grad_inputs = original_backward(ctx, *grad_outputs)

            return grad_inputs

        func_class.forward = wrapped_forward
        func_class.backward = wrapped_backward

        self.patched_functions[func_class] = (original_forward, original_backward)

    def unpatch_all(self):
        for func_class, (orig_fwd, orig_bwd) in self.patched_functions.items():
            func_class.forward = orig_fwd
            func_class.backward = orig_bwd
        self.patched_functions.clear()

    def start(self):
        self.enabled = True

    def stop(self):
        self.enabled = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
