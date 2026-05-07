from typing import Dict, List, Optional, Any, Callable
import torch
import torch.nn as nn

from .graph import ComputationalGraph
from .module_hook import ModuleCapture
from .comm_hook import CommunicationCapture
from .backward_hook import BackwardCapture
from .autograd_hook import AutogradCapture
from .profiler_hook import ProfilerCapture
from .aten_hook import AtenCapture
from .optimizer_hook import OptimizerCapture


class CaptureCoordinator:
    def __init__(
        self,
        graph: Optional[ComputationalGraph] = None,
        use_aten_mode: bool = False,
    ):
        self.graph = graph if graph else ComputationalGraph()
        self._use_aten_mode = use_aten_mode
        self.backward_capture = BackwardCapture(self.graph)
        self.module_capture = ModuleCapture(self.graph, backward_capture=self.backward_capture)
        self.comm_capture = CommunicationCapture(self.graph)
        self.autograd_capture = AutogradCapture(self.graph)
        self.profiler_capture = ProfilerCapture(self.graph)
        self.aten_capture: Optional[AtenCapture] = None
        self.optimizer_capture: Optional[OptimizerCapture] = None

        if use_aten_mode:
            self.aten_capture = AtenCapture(self.graph)
            self.optimizer_capture = OptimizerCapture(aten_capture=self.aten_capture)

        self._model: Optional[nn.Module] = None
        self._attached = False

        self._module_callbacks: List[Callable[[str], None]] = []

    def attach(self, model: nn.Module) -> "CaptureCoordinator":
        self._model = model
        if self._use_aten_mode:
            self.comm_capture.install_hooks()
        else:
            self.module_capture.register_module(model)
            self.comm_capture.install_hooks()
            self._register_autograd_functions()
            self._setup_module_to_comm_link()
        self._attached = True
        return self

    def _setup_module_to_comm_link(self):
        def on_node_created(node_id: str):
            self.comm_capture.set_current_compute_node(node_id)

        self.module_capture.register_on_node_created(on_node_created)
        self._module_callbacks.append(on_node_created)

    def _register_autograd_functions(self):
        try:
            from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
            self.autograd_capture.patch_function(bias_gelu_impl, "bias_gelu")
        except Exception:
            pass

        try:
            from megatron.core.tensor_parallel.layers import linear_with_grad_accumulation_and_async_allreduce
            self.autograd_capture.patch_function(
                linear_with_grad_accumulation_and_async_allreduce, "linear_async"
            )
        except Exception:
            pass

    def start(self):
        if self._use_aten_mode:
            if self.aten_capture is not None:
                self.aten_capture.start()
            if self.optimizer_capture is not None:
                self.optimizer_capture.install_hooks()
        self.module_capture.start()
        self.comm_capture.start()
        self.backward_capture.start()
        self.autograd_capture.start()

    def stop(self):
        if self._use_aten_mode:
            if self.aten_capture is not None:
                self.aten_capture.stop()
            if self.optimizer_capture is not None:
                self.optimizer_capture.uninstall_hooks()
        self.module_capture.stop()
        self.comm_capture.stop()
        self.backward_capture.stop()
        self.autograd_capture.stop()

    def reset(self):
        self.module_capture.reset()
        self.comm_capture.reset()
        self.backward_capture.clear_hooks()
        self.autograd_capture.unpatch_all()
        self.profiler_capture = ProfilerCapture(self.graph)
        if self.aten_capture is not None:
            self.aten_capture.reset()
        if self.optimizer_capture is not None:
            self.optimizer_capture.reset()
        self.graph = ComputationalGraph()
        self._model = None
        self._attached = False
        self._module_callbacks.clear()

    def get_graph(self) -> ComputationalGraph:
        return self.graph

    def start_profiler(self, **profiler_kwargs):
        self.profiler_capture.setup(**profiler_kwargs)
        self.profiler_capture.start()

    def stop_profiler(self) -> Dict[str, Any]:
        self.profiler_capture.stop()
        self.profiler_capture.process_events()
        return self.profiler_capture.get_calibration_data()

    def export_chrome_trace(self, path: str):
        self.profiler_capture.export_chrome_trace(path)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
