from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.profiler
from .graph import GraphNode, OpType, ComputationalGraph


class ProfilerCapture:
    def __init__(self, graph: ComputationalGraph):
        self.graph = graph
        self.enabled = False
        self.profiler: Optional[torch.profiler.profile] = None
        self.event_mapping: Dict[str, str] = {}
        self.profiler_available = True
        self.known_shapes: Dict[str, Tuple] = {}
        self.kernel_times: Dict[str, List[float]] = {}

    def _check_profiler_available(self) -> bool:
        try:
            import torch.profiler
            return hasattr(torch.profiler, 'profile')
        except ImportError:
            return False

    def setup(self, activities=None, record_shapes=True, with_stack=True, with_flops=True):
        if not self._check_profiler_available():
            self.profiler_available = False
            return

        if activities is None:
            activities = [
                torch.profiler.ProfilerActivity.CPU,
            ]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)

        self.profiler = torch.profiler.profile(
            activities=activities,
            record_shapes=record_shapes,
            with_stack=with_stack,
            with_flops=with_flops,
        )

    def start(self):
        self.enabled = True
        if self.profiler is not None:
            self.profiler.start()

    def stop(self):
        self.enabled = False
        if self.profiler is not None:
            self.profiler.stop()

    def process_events(self):
        if self.profiler is None or not hasattr(self.profiler, 'events'):
            return

        events = self.profiler.events()
        if events is None:
            return

        for event in events:
            self._process_event(event)

    def _process_event(self, event):
        event_name = getattr(event, 'name', None)
        if not event_name:
            return

        node_id = self._find_matching_node(event_name)
        if node_id and node_id in self.graph.nodes:
            node = self.graph.nodes[node_id]

            cuda_time = getattr(event, 'cuda_time_total', 0.0)
            cpu_time = getattr(event, 'cuda_time_total', 0.0)
            flops = getattr(event, 'flops', 0)

            if cuda_time > 0:
                if node_id not in self.kernel_times:
                    self.kernel_times[node_id] = []
                self.kernel_times[node_id].append(cuda_time / 1000.0)

            if flops > 0 and node.evaluated_flops == 0:
                node.evaluated_flops = flops

            input_shapes = getattr(event, 'input_shapes', None)
            if input_shapes and not node.input_shapes:
                shapes = []
                for shape in input_shapes:
                    if shape:
                        shapes.append(tuple(shape))
                    else:
                        shapes.append(None)
                node.input_shapes = shapes

    def _find_matching_node(self, event_name: str) -> Optional[str]:
        for node_id in self.graph.nodes:
            if event_name in node_id or node_id in event_name:
                return node_id

        event_parts = event_name.split('_')
        for node_id in self.graph.nodes:
            node_parts = node_id.split('_')
            common = set(event_parts) & set(node_parts)
            if len(common) >= min(len(event_parts), len(node_parts)) // 2:
                return node_id

        return None

    def get_calibration_data(self) -> Dict[str, Any]:
        calibration = {
            'kernel_times': {},
            'flops': {},
            'shapes': {},
        }

        for node_id, times in self.kernel_times.items():
            if times:
                calibration['kernel_times'][node_id] = {
                    'mean': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'count': len(times),
                }

        for node_id, node in self.graph.nodes.items():
            if node.evaluated_flops > 0:
                calibration['flops'][node_id] = node.evaluated_flops
            if node.input_shapes:
                calibration['shapes'][node_id] = node.input_shapes

        return calibration

    def export_chrome_trace(self, path: str):
        if self.profiler is not None and hasattr(self.profiler, 'export_chrome_trace'):
            self.profiler.export_chrome_trace(path)

    def __enter__(self):
        self.setup()
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.process_events()
        return False
