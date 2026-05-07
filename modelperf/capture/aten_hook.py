"""Aten-level native operator capture via monkey-patching torch._ops.OpOverload.

Captures every ATen operator call (e.g. aten::mm, aten::addmm, aten::layer_norm)
as a graph node with full tensor metadata (dtype, stride, device) and op kwargs.
"""
from typing import Dict, List, Optional, Tuple, Any
import threading
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from .graph import ComputationalGraph, GraphNode, OpType

_ATEN_META_OPS = {
    "aten::size", "aten::stride", "aten::dim", "aten::is_leaf",
    "aten::storage_offset", "aten::is_contiguous", "aten::device",
    "aten::dtype", "aten::layout", "aten::detach", "aten::clone",
    "aten::_local_scalar_dense", "aten::item", "aten::numel",
    "aten::tolist", "aten::Int", "aten::Float", "aten::Bool",
    "aten::scalar_tensor", "aten::_tensor_str", "aten::empty",
    "aten::empty_strided", "aten::new_empty", "aten::new_zeros",
    "aten::fill_", "aten::zero_", "aten::copy_", "aten::set_",
    "aten::t", "aten::permute", "aten::transpose", "aten::view",
    "aten::reshape", "aten::unsqueeze", "aten::squeeze", "aten::expand",
    "aten::slice", "aten::select", "aten::index", "aten::cat",
    "aten::split", "aten::chunk", "aten::stack", "aten::unbind",
    "aten::as_strided", "aten::_reshape_alias", "aten::_unsafe_view",
    "aten::_to_copy", "aten::to", "aten::type_as", "aten::cast",
    "aten::alias", "aten::lift_fresh", "aten::lift_fresh_copy",
    "aten::_assert_async", "aten::record_stream", "aten::_backward",
    "aten::save_for_backward", "aten::set_data",
}


def _should_skip_op(op_name: str) -> bool:
    if any(op_name.endswith(suffix) for suffix in _ATEN_META_OPS):
        return True
    if "prim::" in op_name or "profiler" in op_name:
        return True
    return False


def _extract_tensor_meta(t: torch.Tensor) -> Dict[str, Any]:
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "stride": list(t.stride()),
        "numel": t.numel(),
        "device": str(t.device),
    }


def _extract_io_info(args, kwargs) -> Tuple[List[Dict], Dict[str, Any]]:
    input_infos: List[Dict] = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            input_infos.append(_extract_tensor_meta(arg))
        elif isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, torch.Tensor):
                    input_infos.append(_extract_tensor_meta(item))
    kw_info: Dict[str, Any] = {}
    if kwargs:
        for k, v in kwargs.items():
            if isinstance(v, (int, float, bool, str)):
                kw_info[k] = v
            elif isinstance(v, torch.Tensor):
                kw_info[k] = _extract_tensor_meta(v)
    return input_infos, kw_info


def _extract_output_info(outputs) -> List[Dict]:
    output_infos: List[Dict] = []
    if isinstance(outputs, torch.Tensor):
        output_infos.append(_extract_tensor_meta(outputs))
    elif isinstance(outputs, (list, tuple)):
        for out in outputs:
            if isinstance(out, torch.Tensor):
                output_infos.append(_extract_tensor_meta(out))
    return output_infos


class AtenCapture(TorchDispatchMode):
    """Capture every ATen op into the computational graph.

    Usage:
        capture = AtenCapture(graph)
        capture.start()
        # ... run model ...
        capture.stop()
    """

    def __init__(self, graph: Optional[ComputationalGraph] = None):
        super().__init__()
        self.graph = graph if graph else ComputationalGraph()
        self._enabled = False
        self._node_counter = 0
        self._in_optimizer = False
        self._local = threading.local()

    @staticmethod
    def _is_in_backward() -> bool:
        try:
            return torch._C._current_graph_task_id() >= 0
        except Exception:
            return False

    def _get_node_id(self, op_name: str) -> str:
        clean_name = op_name.replace("::", "_").replace(".", "_")
        node_id = f"aten_{self._node_counter}_{clean_name}"
        self._node_counter += 1
        return node_id

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        op_name = str(func)

        if not self._enabled or _should_skip_op(op_name):
            with torch._C._DisableTorchDispatch():
                return func(*args, **kwargs)

        if getattr(self._local, "in_dispatch", False):
            with torch._C._DisableTorchDispatch():
                return func(*args, **kwargs)

        self._local.in_dispatch = True
        try:
            input_infos, kw_info = _extract_io_info(args, kwargs)

            with torch._C._DisableTorchDispatch():
                outputs = func(*args, **kwargs)

            output_infos = _extract_output_info(outputs)

            node_id = self._get_node_id(op_name)

            if self._in_optimizer:
                node_type = OpType.OPTIMIZER
                phase = "optimizer"
            elif self._is_in_backward():
                node_type = OpType.BACKWARD
                phase = "backward"
            else:
                node_type = OpType.COMPUTE
                phase = "forward"

            node = GraphNode(
                node_id=node_id,
                op_type=op_name,
                node_type=node_type,
                input_shapes=[tuple(i["shape"]) for i in input_infos],
                output_shapes=[tuple(o["shape"]) for o in output_infos],
                metadata={
                    "input_tensors": input_infos,
                    "output_tensors": output_infos,
                    "aten_kwargs": kw_info,
                    "phase": phase,
                    "is_optimizer": self._in_optimizer,
                    "is_backward": phase == "backward",
                },
            )

            self.graph.nodes[node_id] = node
            if phase == "forward":
                self.graph.forward_nodes.append(node_id)
            elif phase == "backward":
                self.graph.backward_nodes.append(node_id)

            return outputs
        finally:
            self._local.in_dispatch = False

    def start(self):
        self._enabled = True
        super().__enter__()

    def stop(self):
        self._enabled = False
        super().__exit__(None, None, None)

    def reset(self):
        self.stop()
        self.graph = ComputationalGraph()
        self._node_counter = 0
        self._in_optimizer = False

    def set_in_optimizer(self, val: bool):
        self._in_optimizer = val

    def get_graph(self) -> ComputationalGraph:
        return self.graph

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
