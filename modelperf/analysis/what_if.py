from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

import sympy
from sympy import Symbol, Expr, simplify

from ..capture.graph import ComputationalGraph, GraphNode, OpType
from ..symbolic.shape_inferer import SymbolicShapeInferer
from ..simulation.virtual_executor import VirtualExecutor
from ..simulation.execution_result import ExecutionResult
from ..simulation.roofline import RooflineModel
from ..simulation.bandwidth import BandwidthModel
from ..simulation.memory_tracker import MemoryTracker
from .config_variants import ConfigVariant, ConfigChangeType


@dataclass
class WhatIfConfig:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_parallel_size: int = 1
    hidden_size: int = 4096
    num_layers: int = 32
    num_attention_heads: int = 32
    ffn_hidden_size: int = 11008
    seq_length: int = 2048
    vocab_size: int = 32000
    micro_batch_size: int = 1
    global_batch_size: int = 8
    num_microbatches: int = 8
    activation_checkpointing: bool = False
    gpu_type: str = "A100"
    network_type: str = "IB"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'tensor_parallel_size': self.tensor_parallel_size,
            'pipeline_parallel_size': self.pipeline_parallel_size,
            'data_parallel_size': self.data_parallel_size,
            'context_parallel_size': self.context_parallel_size,
            'expert_parallel_size': self.expert_parallel_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_attention_heads': self.num_attention_heads,
            'ffn_hidden_size': self.ffn_hidden_size,
            'seq_length': self.seq_length,
            'vocab_size': self.vocab_size,
            'micro_batch_size': self.micro_batch_size,
            'global_batch_size': self.global_batch_size,
            'num_microbatches': self.num_microbatches,
            'activation_checkpointing': self.activation_checkpointing,
            'gpu_type': self.gpu_type,
            'network_type': self.network_type,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'WhatIfConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_graph(cls, graph: ComputationalGraph) -> 'WhatIfConfig':
        model_config = graph.model_config
        strategy_config = graph.strategy_config
        system_config = graph.system_config

        return cls(
            tensor_parallel_size=strategy_config.get('tensor_parallel_size', 1),
            pipeline_parallel_size=strategy_config.get('pipeline_parallel_size', 1),
            data_parallel_size=strategy_config.get('data_parallel_size', 1),
            context_parallel_size=strategy_config.get('context_parallel_size', 1),
            expert_parallel_size=strategy_config.get('expert_parallel_size', 1),
            hidden_size=model_config.get('hidden_size', 4096),
            num_layers=model_config.get('num_layers', 32),
            num_attention_heads=model_config.get('num_attention_heads', 32),
            ffn_hidden_size=model_config.get('ffn_hidden_size', 11008),
            seq_length=model_config.get('seq_length', 2048),
            vocab_size=model_config.get('vocab_size', 32000),
            micro_batch_size=strategy_config.get('micro_batch_size', 1),
            global_batch_size=strategy_config.get('global_batch_size', 8),
            num_microbatches=strategy_config.get('num_microbatches', 8),
            activation_checkpointing=system_config.get('activation_checkpointing', False),
            gpu_type=system_config.get('gpu_type', 'A100'),
            network_type=system_config.get('network_type', 'IB'),
        )


class WhatIfAnalyzer:
    def __init__(
        self,
        base_graph: ComputationalGraph,
        roofline_model: Optional[RooflineModel] = None,
        bandwidth_model: Optional[BandwidthModel] = None,
        memory_tracker: Optional[MemoryTracker] = None,
        max_workers: int = 4
    ):
        self.base_graph = base_graph
        self.inferer = SymbolicShapeInferer()
        self.roofline_model = roofline_model or RooflineModel()
        self.bandwidth_model = bandwidth_model or BandwidthModel()
        self.memory_tracker = memory_tracker or MemoryTracker()
        self.max_workers = max_workers

        self.base_config = WhatIfConfig.from_graph(base_graph)
        self._symbol_cache: Dict[str, Symbol] = {}

    def _get_symbol_values(self, config: WhatIfConfig) -> Dict[str, Union[int, float]]:
        return {
            'B': config.micro_batch_size,
            'S': config.seq_length,
            'H': config.hidden_size,
            'V': config.vocab_size,
            'F': config.ffn_hidden_size,
            'NH': config.num_attention_heads,
            'NG': config.num_attention_heads,
            'TP': config.tensor_parallel_size,
            'PP': config.pipeline_parallel_size,
            'DP': config.data_parallel_size,
            'CP': config.context_parallel_size,
            'EP': config.expert_parallel_size,
        }

    def modify_config(
        self,
        changes: Dict[str, Any],
        base_config: Optional[WhatIfConfig] = None
    ) -> WhatIfConfig:
        config = base_config or self.base_config
        new_config = WhatIfConfig.from_dict(config.to_dict())

        for key, value in changes.items():
            if hasattr(new_config, key):
                setattr(new_config, key, value)
            else:
                raise ValueError(f"Unknown config key: {key}")

        return new_config

    def _create_modified_graph(
        self,
        config: WhatIfConfig
    ) -> ComputationalGraph:
        graph = copy.deepcopy(self.base_graph)

        graph.model_config.update({
            'hidden_size': config.hidden_size,
            'num_layers': config.num_layers,
            'num_attention_heads': config.num_attention_heads,
            'ffn_hidden_size': config.ffn_hidden_size,
            'seq_length': config.seq_length,
            'vocab_size': config.vocab_size,
        })

        graph.strategy_config.update({
            'tensor_parallel_size': config.tensor_parallel_size,
            'pipeline_parallel_size': config.pipeline_parallel_size,
            'data_parallel_size': config.data_parallel_size,
            'context_parallel_size': config.context_parallel_size,
            'expert_parallel_size': config.expert_parallel_size,
            'micro_batch_size': config.micro_batch_size,
            'global_batch_size': config.global_batch_size,
            'num_microbatches': config.num_microbatches,
        })

        graph.system_config.update({
            'activation_checkpointing': config.activation_checkpointing,
            'gpu_type': config.gpu_type,
            'network_type': config.network_type,
        })

        symbol_values = self._get_symbol_values(config)

        for node in graph.nodes.values():
            self._update_node_symbols(node, symbol_values)

        return graph

    def _update_node_symbols(
        self,
        node: GraphNode,
        symbol_values: Dict[str, Union[int, float]]
    ):
        if node.symbolic_input_shapes:
            node.input_shapes = self._evaluate_shapes(
                node.symbolic_input_shapes,
                symbol_values
            )

        if node.symbolic_output_shapes:
            node.output_shapes = self._evaluate_shapes(
                node.symbolic_output_shapes,
                symbol_values
            )

        if node.symbolic_comm_bytes is not None:
            node.comm_bytes = int(self.inferer.evaluate(
                node.symbolic_comm_bytes,
                symbol_values
            ))

    def _evaluate_shapes(
        self,
        symbolic_shapes: List[Any],
        symbol_values: Dict[str, Union[int, float]]
    ) -> List[Optional[Tuple]]:
        result = []
        for shape in symbolic_shapes:
            if shape is None:
                result.append(None)
            else:
                evaluated = self.inferer.evaluate(shape, symbol_values)
                if isinstance(evaluated, (list, tuple)):
                    converted = []
                    for x in evaluated:
                        if hasattr(x, 'is_number') and x.is_number:
                            converted.append(int(x) if float(x) == int(float(x)) else float(x))
                        else:
                            converted.append(x)
                    result.append(tuple(converted))
                else:
                    result.append((evaluated,))
        return result

    def re_evaluate(
        self,
        new_config: WhatIfConfig,
        use_cached_executor: bool = False
    ) -> ExecutionResult:
        graph = self._create_modified_graph(new_config)

        executor = VirtualExecutor(
            roofline_model=self.roofline_model,
            bandwidth_model=self.bandwidth_model
        )

        result = executor.execute(graph)

        result.config_used = new_config.to_dict()

        return result

    def grid_search(
        self,
        param_grid: Dict[str, List[Any]],
        base_config: Optional[WhatIfConfig] = None,
        sort_by: str = "iteration_time_ms",
        ascending: bool = True,
        n_workers: Optional[int] = None
    ) -> List[Tuple[Dict[str, Any], ExecutionResult]]:
        from itertools import product

        config = base_config or self.base_config
        keys = list(param_grid.keys())
        values_lists = [param_grid[k] for k in keys]

        all_combinations = list(product(*values_lists))

        results = []
        workers = n_workers or self.max_workers

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {}
                for combo in all_combinations:
                    changes = dict(zip(keys, combo))
                    future = executor.submit(self._evaluate_single_config, config, changes)
                    futures[future] = changes

                for future in as_completed(futures):
                    changes = futures[future]
                    try:
                        result = future.result()
                        results.append((changes, result))
                    except Exception as e:
                        results.append((changes, None))
        else:
            for combo in all_combinations:
                changes = dict(zip(keys, combo))
                try:
                    result = self._evaluate_single_config(config, changes)
                    results.append((changes, result))
                except Exception as e:
                    results.append((changes, None))

        valid_results = [(c, r) for c, r in results if r is not None]

        if valid_results and hasattr(valid_results[0][1], sort_by):
            valid_results.sort(key=lambda x: getattr(x[1], sort_by), reverse=not ascending)

        return valid_results

    def _evaluate_single_config(
        self,
        base_config: WhatIfConfig,
        changes: Dict[str, Any]
    ) -> ExecutionResult:
        new_config = self.modify_config(changes, base_config)
        return self.re_evaluate(new_config)

    def predict_oom(
        self,
        config: WhatIfConfig,
        gpu_memory_gb: float = 80.0
    ) -> Tuple[bool, float, float]:
        graph = self._create_modified_graph(config)

        peak_memory = self.memory_tracker.estimate_peak_memory(
            graph,
            gpu_memory_gb * 1024**3
        )

        memory_gb = peak_memory / (1024**3)
        will_oom = memory_gb > gpu_memory_gb
        utilization = memory_gb / gpu_memory_gb

        return will_oom, memory_gb, utilization

    def batch_predict_oom(
        self,
        configs: List[WhatIfConfig],
        gpu_memory_gb: float = 80.0,
        n_workers: Optional[int] = None
    ) -> List[Tuple[bool, float, float]]:
        workers = n_workers or self.max_workers

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(self.predict_oom, config, gpu_memory_gb)
                    for config in configs
                ]
                return [f.result() for f in futures]
        else:
            return [self.predict_oom(config, gpu_memory_gb) for config in configs]

    def find_optimal_config(
        self,
        search_space: Dict[str, List[Any]],
        objective: str = "throughput",
        constraints: Optional[List[Callable[[WhatIfConfig], bool]]] = None,
        max_memory_gb: float = 80.0
    ) -> Tuple[WhatIfConfig, ExecutionResult]:
        from .config_variants import generate_grid_search

        grid_results = self.grid_search(
            search_space,
            sort_by="iteration_time_ms",
            ascending=True
        )

        valid_results = []
        for changes, result in grid_results:
            config = self.modify_config(changes)

            will_oom, _, _ = self.predict_oom(config, max_memory_gb)
            if will_oom:
                continue

            if constraints:
                all_pass = all(constraint(config) for constraint in constraints)
                if not all_pass:
                    continue

            valid_results.append((config, result))

        if not valid_results:
            raise ValueError("No valid configuration found within constraints")

        if objective == "throughput":
            valid_results.sort(key=lambda x: x[1].throughput_tokens_per_sec, reverse=True)
        elif objective == "latency":
            valid_results.sort(key=lambda x: x[1].iteration_time_ms)
        elif objective == "memory_efficiency":
            valid_results.sort(key=lambda x: x[1].memory_efficiency, reverse=True)

        return valid_results[0]


__all__ = [
    'WhatIfConfig',
    'WhatIfAnalyzer',
]
