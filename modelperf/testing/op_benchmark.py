import time
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    op_type: str
    node_id: str
    correctness_passed: bool
    correctness_error: Optional[str]
    avg_time_ms: float
    flops: float
    flops_utilization: float
    memory_bandwidth_gbs: float
    comm_bandwidth_gbs: Optional[float]
    peak_memory_mb: float
    device: str


class OpAdapter:
    def __init__(self):
        self._registry: Dict[str, Callable[[Any], Tuple[Callable, List[torch.Tensor]]]] = {}
        self._register_defaults()

    def _register_defaults(self):
        self._registry["Linear"] = self._build_linear
        self._registry["linear"] = self._build_linear
        self._registry["aten_mm"] = self._build_matmul
        self._registry["aten_addmm"] = self._build_addmm
        self._registry["LayerNorm"] = self._build_layernorm
        self._registry["layer_norm"] = self._build_layernorm
        self._registry["SelfAttention"] = self._build_self_attention_stub
        self._registry["attention"] = self._build_self_attention_stub
        self._registry["ReLU"] = self._build_relu
        self._registry["relu"] = self._build_relu
        self._registry["GELU"] = self._build_gelu
        self._registry["gelu"] = self._build_gelu
        self._registry["Softmax"] = self._build_softmax
        self._registry["softmax"] = self._build_softmax
        self._registry["Dropout"] = self._build_dropout
        self._registry["dropout"] = self._build_dropout
        self._registry["Embedding"] = self._build_embedding
        self._registry["embedding"] = self._build_embedding

    def register(self, op_type: str, builder: Callable[[Any], Tuple[Callable, List[torch.Tensor]]]):
        self._registry[op_type] = builder

    def build(self, node) -> Tuple[Callable, List[torch.Tensor]]:
        op = node.op_type
        if op in self._registry:
            return self._registry[op](node)

        for key in self._registry:
            if key.lower() in op.lower() or op.lower() in key.lower():
                return self._registry[key](node)

        return self._build_fallback(node)

    @staticmethod
    def _make_tensors(shapes: List[Optional[Tuple]], device: str = "cpu", dtype=torch.float32):
        tensors = []
        for shape in shapes:
            if shape is None:
                continue
            try:
                t = torch.randn(shape, dtype=dtype, device=device)
                tensors.append(t)
            except Exception:
                pass
        return tensors

    def _build_linear(self, node):
        input_shapes = node.input_shapes
        params = node.params
        if not input_shapes or input_shapes[0] is None:
            input_shapes = [(2, 128, 576)]
        in_features = input_shapes[0][-1]
        out_features = None
        if "weight" in params:
            w = params["weight"]
            if len(w) >= 2:
                candidate_in = w[0]
                candidate_out = w[1]
                if candidate_in == in_features:
                    out_features = candidate_out
                elif candidate_out == in_features:
                    out_features = candidate_in
                else:
                    out_features = candidate_out
        if out_features is None and node.output_shapes and node.output_shapes[0] is not None:
            out_features = node.output_shapes[0][-1]
        if out_features is None:
            out_features = in_features
        module = nn.Linear(in_features, out_features, bias="bias" in params)
        inputs = self._make_tensors(input_shapes)
        return module, inputs

    def _build_matmul(self, node):
        shapes = node.input_shapes
        if len(shapes) < 2 or shapes[0] is None or shapes[1] is None:
            shapes = [(2, 128, 576), (2, 576, 576)]
        inputs = self._make_tensors(shapes)
        return lambda a, b: torch.matmul(a, b), inputs

    def _build_addmm(self, node):
        shapes = node.input_shapes
        if len(shapes) < 3 or any(s is None for s in shapes):
            shapes = [(128, 576), (128, 576), (576, 576)]
        inputs = self._make_tensors(shapes)
        return lambda bias, a, b: torch.addmm(bias, a, b), inputs

    def _build_layernorm(self, node):
        normalized_shape = None
        if "weight" in node.params:
            normalized_shape = node.params["weight"]
        elif node.input_shapes and node.input_shapes[0] is not None:
            normalized_shape = (node.input_shapes[0][-1],)
        else:
            normalized_shape = (576,)
        module = nn.LayerNorm(normalized_shape)
        inputs = self._make_tensors(node.input_shapes)
        return module, inputs

    def _build_self_attention_stub(self, node):
        inputs = self._make_tensors(node.input_shapes)
        return lambda x: x, inputs

    def _build_relu(self, node):
        module = nn.ReLU()
        inputs = self._make_tensors(node.input_shapes)
        return module, inputs

    def _build_gelu(self, node):
        module = nn.GELU()
        inputs = self._make_tensors(node.input_shapes)
        return module, inputs

    def _build_softmax(self, node):
        module = nn.Softmax(dim=-1)
        inputs = self._make_tensors(node.input_shapes)
        return module, inputs

    def _build_dropout(self, node):
        module = nn.Dropout(p=0.1)
        inputs = self._make_tensors(node.input_shapes)
        return module, inputs

    def _build_embedding(self, node):
        num_embeddings = 151936
        embedding_dim = 576
        if "weight" in node.params:
            w = node.params["weight"]
            if len(w) >= 2:
                num_embeddings, embedding_dim = w
        module = nn.Embedding(num_embeddings, embedding_dim)
        shape = node.input_shapes[0] if node.input_shapes else (2, 128)
        idx = torch.randint(0, num_embeddings, shape)
        return module, [idx]

    def _build_fallback(self, node):
        inputs = self._make_tensors(node.input_shapes)
        return lambda *args: args[0] if args else None, inputs


class OpBenchmark:
    def __init__(self, device: str = "cuda", warmup: int = 10, repeat: int = 100):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.warmup = warmup
        self.repeat = repeat
        self.adapter = OpAdapter()

    def benchmark_node(self, node) -> BenchmarkResult:
        op_func, inputs_cpu = self.adapter.build(node)
        inputs_gpu = [t.to(self.device) for t in inputs_cpu]
        if hasattr(op_func, "to"):
            op_func = op_func.to(self.device)

        correctness_passed, correctness_error = self._check_correctness(
            op_func, inputs_cpu, inputs_gpu
        )

        avg_time_ms = self._measure_time(op_func, inputs_gpu)
        flops = self._estimate_flops(node)
        flops_utilization = (flops / (avg_time_ms / 1000.0)) / 1e12 if avg_time_ms > 0 else 0.0
        mem_bw = self._estimate_memory_bandwidth(node, avg_time_ms)
        comm_bw = self._estimate_comm_bandwidth(node, avg_time_ms)
        peak_mem_mb = self._measure_peak_memory(op_func, inputs_gpu)

        return BenchmarkResult(
            op_type=node.op_type,
            node_id=node.node_id,
            correctness_passed=correctness_passed,
            correctness_error=correctness_error,
            avg_time_ms=avg_time_ms,
            flops=flops,
            flops_utilization=flops_utilization,
            memory_bandwidth_gbs=mem_bw,
            comm_bandwidth_gbs=comm_bw,
            peak_memory_mb=peak_mem_mb,
            device=self.device,
        )

    def _check_correctness(self, op_func, inputs_cpu, inputs_gpu, rtol=1e-4, atol=1e-5):
        try:
            with torch.no_grad():
                out_cpu = op_func(*inputs_cpu) if callable(op_func) else inputs_cpu[0]
                out_gpu = op_func(*inputs_gpu) if callable(op_func) else inputs_gpu[0]
                if isinstance(out_cpu, torch.Tensor) and isinstance(out_gpu, torch.Tensor):
                    out_gpu_cpu = out_gpu.cpu()
                    if out_cpu.dtype != out_gpu_cpu.dtype:
                        out_cpu = out_cpu.to(out_gpu_cpu.dtype)
                    if not torch.allclose(out_cpu, out_gpu_cpu, rtol=rtol, atol=atol):
                        diff = (out_cpu - out_gpu_cpu).abs().max().item()
                        return False, f"max_diff={diff}"
                elif isinstance(out_cpu, (tuple, list)):
                    for a, b in zip(out_cpu, out_gpu):
                        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                            if not torch.allclose(a, b.cpu(), rtol=rtol, atol=atol):
                                return False, "tuple mismatch"
            return True, None
        except Exception as e:
            return False, str(e)

    def _measure_time(self, op_func, inputs):
        for _ in range(self.warmup):
            with torch.no_grad():
                _ = op_func(*inputs)
        if self.device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        if self.device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(self.repeat):
                with torch.no_grad():
                    _ = op_func(*inputs)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event) / self.repeat
        else:
            for _ in range(self.repeat):
                with torch.no_grad():
                    _ = op_func(*inputs)
            elapsed_ms = (time.perf_counter() - start) * 1000.0 / self.repeat
        return elapsed_ms

    def _estimate_flops(self, node):
        op = node.op_type.lower()
        shapes = node.input_shapes
        params = node.params

        if "linear" in op or "mm" in op or "matmul" in op:
            if shapes and shapes[0] is not None:
                m = 1
                for d in shapes[0][:-1]:
                    m *= d
                k = shapes[0][-1]
                n = shapes[1][-1] if len(shapes) > 1 and shapes[1] is not None else k
                if "weight" in params and len(params["weight"]) >= 2:
                    n = params["weight"][0]
                return 2 * m * k * n

        if "layernorm" in op or "rmsnorm" in op or "norm" in op:
            if shapes and shapes[0] is not None:
                numel = 1
                for d in shapes[0]:
                    numel *= d
                return numel * 5

        if "softmax" in op:
            if shapes and shapes[0] is not None:
                numel = 1
                for d in shapes[0]:
                    numel *= d
                return numel * 5

        if "attention" in op or "selfattention" in op:
            if shapes and shapes[0] is not None:
                b, s, h = shapes[0][0], shapes[0][1], shapes[0][2]
                return 4 * b * s * h * h

        if "embedding" in op:
            if shapes and shapes[0] is not None and "weight" in params:
                numel = 1
                for d in shapes[0]:
                    numel *= d
                embed_dim = params["weight"][-1]
                return numel * embed_dim * 2

        return 0

    def _estimate_memory_bandwidth(self, node, time_ms):
        if time_ms <= 0:
            return 0.0
        total_bytes = 0
        for shape in node.input_shapes:
            if shape is not None:
                numel = 1
                for d in shape:
                    numel *= d
                total_bytes += numel * node.tensor_dtype_size
        for shape in node.output_shapes:
            if shape is not None:
                numel = 1
                for d in shape:
                    numel *= d
                total_bytes += numel * node.tensor_dtype_size
        for pshape in node.params.values():
            numel = 1
            for d in pshape:
                numel *= d
            total_bytes += numel * node.tensor_dtype_size
        return (total_bytes / (time_ms / 1000.0)) / 1e9

    def _estimate_comm_bandwidth(self, node, time_ms):
        if node.node_type.value != "communication" or time_ms <= 0:
            return None
        bytes_total = node.comm_bytes
        if bytes_total <= 0 and node.tensor_numel > 0:
            bytes_total = node.tensor_numel * node.tensor_dtype_size
        if bytes_total <= 0:
            return None
        return (bytes_total / (time_ms / 1000.0)) / 1e9

    def _measure_peak_memory(self, op_func, inputs):
        if self.device != "cuda":
            return 0.0
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = op_func(*inputs)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        return peak / (1024.0 * 1024.0)
