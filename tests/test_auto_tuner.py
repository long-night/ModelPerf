import unittest
from typing import Any, Dict, List, Tuple

from modelperf.capture.graph import ComputationalGraph, GraphNode, OpType, CommType
from modelperf.simulation.execution_result import ExecutionResult
from modelperf.tuning.auto_tuner import (
    AutoTuner,
    ParallelStrategyPreset,
    TuningReport,
    recommend,
)


def _make_minimal_graph(
    hidden_size: int = 256,
    num_layers: int = 4,
    num_attention_heads: int = 8,
    world_size: int = 4,
    tp: int = 1,
    pp: int = 1,
    dp: int = 4,
    micro_batch_size: int = 1,
) -> ComputationalGraph:
    graph = ComputationalGraph()
    graph.model_config = {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "ffn_hidden_size": hidden_size * 4,
        "vocab_size": 1000,
        "seq_length": 64,
    }
    graph.strategy_config = {
        "tensor_parallel_size": tp,
        "pipeline_parallel_size": pp,
        "data_parallel_size": dp,
        "context_parallel_size": 1,
        "micro_batch_size": micro_batch_size,
        "global_batch_size": micro_batch_size * dp,
        "num_microbatches": micro_batch_size * dp,
    }
    graph.system_config = {
        "world_size": world_size,
        "gpu_type": "a100",
        "activation_checkpointing": False,
    }

    node = graph.add_node(
        node_id="linear_0",
        op_type="linear",
        node_type=OpType.COMPUTE,
        output_shapes=[(micro_batch_size, 64, hidden_size)],
        params={"weight": (hidden_size, hidden_size)},
    )
    graph.forward_nodes.append(node.node_id)
    graph.backward_nodes.append(node.node_id)

    return graph


class TestAutoTunerSearchSpace(unittest.TestCase):
    def test_generate_search_space_basic(self):
        graph = _make_minimal_graph(hidden_size=256, num_layers=4, num_attention_heads=8, world_size=4)
        tuner = AutoTuner(graph)
        space = tuner.generate_search_space(world_size=4)

        self.assertIn("tensor_parallel_size", space)
        self.assertIn("pipeline_parallel_size", space)
        self.assertIn("data_parallel_size", space)
        self.assertIn("micro_batch_size", space)

        self.assertTrue(all(isinstance(v, list) for v in space.values()))

    def test_tp_candidates_filter_hidden_size(self):
        graph = _make_minimal_graph(hidden_size=256, num_attention_heads=8, world_size=8)
        tuner = AutoTuner(graph)
        space = tuner.generate_search_space(world_size=8)

        for tp in space["tensor_parallel_size"]:
            self.assertEqual(256 % tp, 0)
            self.assertEqual(8 % tp, 0)

    def test_pp_candidates_filter_num_layers(self):
        graph = _make_minimal_graph(num_layers=4, world_size=4)
        tuner = AutoTuner(graph)
        space = tuner.generate_search_space(world_size=4)

        for pp in space["pipeline_parallel_size"]:
            self.assertEqual(4 % pp, 0)

    def test_search_space_respects_world_size(self):
        graph = _make_minimal_graph(world_size=8)
        tuner = AutoTuner(graph)
        space = tuner.generate_search_space(world_size=8)

        valid = set()
        for tp in space["tensor_parallel_size"]:
            for pp in space["pipeline_parallel_size"]:
                for dp in space["data_parallel_size"]:
                    if tp * pp * dp == 8:
                        valid.add((tp, pp, dp))

        self.assertGreater(len(valid), 0)

    def test_custom_tp_pp_bs_values(self):
        graph = _make_minimal_graph(world_size=4)
        tuner = AutoTuner(graph)
        space = tuner.generate_search_space(
            world_size=4,
            tp_values=[1, 2],
            pp_values=[1, 2],
            bs_values=[1, 4],
        )

        self.assertEqual(space["tensor_parallel_size"], [1, 2])
        self.assertEqual(space["pipeline_parallel_size"], [1, 2])
        self.assertEqual(space["micro_batch_size"], [1, 4])


class TestAutoTunerSearch(unittest.TestCase):
    def test_search_returns_report(self):
        graph = _make_minimal_graph(world_size=1)
        tuner = AutoTuner(graph)
        report = tuner.search(world_size=1, param_grid={
            "tensor_parallel_size": [1],
            "pipeline_parallel_size": [1],
            "data_parallel_size": [1],
            "micro_batch_size": [1, 2],
        })

        self.assertIsInstance(report, TuningReport)
        self.assertIn("tensor_parallel_size", report.recommended_config)
        self.assertGreater(len(report.all_results), 0)

    def test_search_balanced_preset(self):
        graph = _make_minimal_graph(world_size=1)
        tuner = AutoTuner(graph)
        report = tuner.search(
            preset=ParallelStrategyPreset.BALANCED,
            world_size=1,
            param_grid={
                "tensor_parallel_size": [1],
                "pipeline_parallel_size": [1],
                "data_parallel_size": [1],
                "micro_batch_size": [1, 2],
            },
        )

        self.assertGreater(report.expected_iteration_time_ms, 0)
        self.assertGreater(report.expected_peak_memory_mb, 0)

    def test_search_throughput_preset(self):
        graph = _make_minimal_graph(world_size=1)
        tuner = AutoTuner(graph)
        report = tuner.search(
            preset=ParallelStrategyPreset.THROUGHPUT,
            world_size=1,
            param_grid={
                "tensor_parallel_size": [1],
                "pipeline_parallel_size": [1],
                "data_parallel_size": [1],
                "micro_batch_size": [1, 2],
            },
        )

        if len(report.all_results) >= 2:
            best_throughput = report.all_results[0][1].throughput_tokens_per_sec
            for _, result in report.all_results[1:]:
                self.assertLessEqual(
                    result.throughput_tokens_per_sec, best_throughput
                )

    def test_search_memory_efficient_preset(self):
        graph = _make_minimal_graph(world_size=1)
        tuner = AutoTuner(graph)
        report = tuner.search(
            preset=ParallelStrategyPreset.MEMORY_EFFICIENT,
            world_size=1,
            param_grid={
                "tensor_parallel_size": [1],
                "pipeline_parallel_size": [1],
                "data_parallel_size": [1],
                "micro_batch_size": [1, 2],
            },
        )

        if len(report.all_results) >= 2:
            best_memory = report.all_results[0][1].peak_memory_mb
            for _, result in report.all_results[1:]:
                self.assertGreaterEqual(result.peak_memory_mb, best_memory)

    def test_filter_by_max_iteration_time(self):
        graph = _make_minimal_graph(world_size=1)
        tuner = AutoTuner(graph)

        report_unlimited = tuner.search(
            world_size=1,
            param_grid={
                "tensor_parallel_size": [1],
                "pipeline_parallel_size": [1],
                "data_parallel_size": [1],
                "micro_batch_size": [1, 2, 4],
            },
        )

        report_limited = tuner.search(
            world_size=1,
            max_iteration_time_ms=report_unlimited.expected_iteration_time_ms,
            param_grid={
                "tensor_parallel_size": [1],
                "pipeline_parallel_size": [1],
                "data_parallel_size": [1],
                "micro_batch_size": [1, 2, 4],
            },
        )

        self.assertLessEqual(
            len(report_limited.all_results), len(report_unlimited.all_results)
        )
        for _, result in report_limited.all_results:
            self.assertLessEqual(
                result.iteration_time_ms,
                report_unlimited.expected_iteration_time_ms,
            )

    def test_filter_by_memory(self):
        graph = _make_minimal_graph(world_size=1)
        tuner = AutoTuner(graph)

        report = tuner.search(
            gpu_memory_gb=0.00001,
            world_size=1,
            param_grid={
                "tensor_parallel_size": [1],
                "pipeline_parallel_size": [1],
                "data_parallel_size": [1],
                "micro_batch_size": [1, 2],
            },
        )

        self.assertEqual(len(report.all_results), 0)
        self.assertEqual(report.expected_iteration_time_ms, 0.0)

    def test_filter_by_min_throughput(self):
        graph = _make_minimal_graph(world_size=1)
        tuner = AutoTuner(graph)

        report = tuner.search(
            min_throughput=1e12,
            world_size=1,
            param_grid={
                "tensor_parallel_size": [1],
                "pipeline_parallel_size": [1],
                "data_parallel_size": [1],
                "micro_batch_size": [1, 2],
            },
        )

        self.assertEqual(len(report.all_results), 0)

    def test_pareto_frontier_non_empty(self):
        graph = _make_minimal_graph(world_size=1)
        tuner = AutoTuner(graph)
        report = tuner.search(
            world_size=1,
            param_grid={
                "tensor_parallel_size": [1],
                "pipeline_parallel_size": [1],
                "data_parallel_size": [1],
                "micro_batch_size": [1, 2, 4],
            },
        )

        self.assertGreaterEqual(len(report.pareto_frontier), 1)
        if len(report.pareto_frontier) > 1:
            for i in range(len(report.pareto_frontier) - 1):
                curr = report.pareto_frontier[i][1]
                nxt = report.pareto_frontier[i + 1][1]
                self.assertGreaterEqual(
                    curr.throughput_tokens_per_sec,
                    nxt.throughput_tokens_per_sec,
                )

    def test_pareto_frontier_no_dominated_points(self):
        graph = _make_minimal_graph(world_size=1)
        tuner = AutoTuner(graph)
        report = tuner.search(
            world_size=1,
            param_grid={
                "tensor_parallel_size": [1],
                "pipeline_parallel_size": [1],
                "data_parallel_size": [1],
                "micro_batch_size": [1, 2],
            },
        )

        pareto = report.pareto_frontier
        all_results = report.all_results

        for p_changes, p_result in pareto:
            for a_changes, a_result in all_results:
                if a_result is p_result:
                    continue
                dominates = (
                    a_result.throughput_tokens_per_sec
                    >= p_result.throughput_tokens_per_sec
                    and a_result.peak_memory_mb <= p_result.peak_memory_mb
                    and (
                        a_result.throughput_tokens_per_sec
                        > p_result.throughput_tokens_per_sec
                        or a_result.peak_memory_mb < p_result.peak_memory_mb
                    )
                )
                self.assertFalse(
                    dominates,
                    msg=f"Pareto point {p_changes} is dominated by {a_changes}",
                )

    def test_world_size_inference_from_graph(self):
        graph = _make_minimal_graph(world_size=4, tp=1, pp=1, dp=4)
        tuner = AutoTuner(graph)
        inferred = tuner._infer_world_size()
        self.assertEqual(inferred, 4)

    def test_valid_config_filter(self):
        graph = _make_minimal_graph(world_size=4)
        tuner = AutoTuner(graph)

        self.assertTrue(tuner._is_valid_config(
            {"tensor_parallel_size": 2, "pipeline_parallel_size": 1, "data_parallel_size": 2}, 4
        ))
        self.assertFalse(tuner._is_valid_config(
            {"tensor_parallel_size": 2, "pipeline_parallel_size": 2, "data_parallel_size": 2}, 4
        ))


class TestRecommendConvenience(unittest.TestCase):
    def test_recommend_returns_tuning_report(self):
        graph = _make_minimal_graph(world_size=1)
        report = recommend(
            graph,
            gpu_memory_gb=80.0,
            preset=ParallelStrategyPreset.BALANCED,
        )

        self.assertIsInstance(report, TuningReport)
        self.assertIn("tensor_parallel_size", report.recommended_config)

    def test_recommend_with_constraints(self):
        graph = _make_minimal_graph(world_size=1)
        report = recommend(
            graph,
            gpu_memory_gb=80.0,
            max_iteration_time_ms=1e6,
            min_throughput=0.0,
            preset=ParallelStrategyPreset.THROUGHPUT,
        )

        self.assertIsInstance(report, TuningReport)


class TestEdgeCases(unittest.TestCase):
    def test_empty_search_space_returns_empty_report(self):
        graph = _make_minimal_graph(world_size=1)
        tuner = AutoTuner(graph)
        report = tuner.search(
            world_size=1,
            param_grid={
                "tensor_parallel_size": [1],
                "pipeline_parallel_size": [1],
                "data_parallel_size": [1],
                "micro_batch_size": [],
            },
        )

        self.assertEqual(len(report.all_results), 0)
        self.assertEqual(report.expected_iteration_time_ms, 0.0)

    def test_single_result_pareto(self):
        graph = _make_minimal_graph(world_size=1)
        tuner = AutoTuner(graph)
        report = tuner.search(
            world_size=1,
            param_grid={
                "tensor_parallel_size": [1],
                "pipeline_parallel_size": [1],
                "data_parallel_size": [1],
                "micro_batch_size": [1],
            },
        )

        self.assertEqual(len(report.all_results), 1)
        self.assertEqual(len(report.pareto_frontier), 1)
        self.assertEqual(
            report.recommended_config,
            report.pareto_frontier[0][0],
        )


if __name__ == "__main__":
    unittest.main()
