import unittest

from modelperf.capture.graph import ComputationalGraph, GraphNode, OpType, CommType
from modelperf.symbolic.shape_inferer import SymbolicShapeInferer
from modelperf.simulation.virtual_executor import VirtualExecutor
from modelperf.simulation.execution_result import ExecutionConfig


class TestSymbolicBridge(unittest.TestCase):
    def test_symbolic_shape_evaluation(self):
        inferer = SymbolicShapeInferer()
        B = inferer.get_symbol('B')
        S = inferer.get_symbol('S')
        H = inferer.get_symbol('H')

        graph = ComputationalGraph()
        graph.model_config = {'hidden_size': 1024}
        graph.strategy_config = {'micro_batch_size': 2, 'seq_length': 128, 'tp_size': 1}

        node = graph.add_node(
            node_id="sym_node",
            op_type="Linear",
            node_type=OpType.COMPUTE,
            output_shapes=[(B * S, H)],
            symbolic_output_shapes=[(B * S, H)],
            params={"weight": (H, H)},
        )
        graph.forward_nodes.append(node.node_id)

        executor = VirtualExecutor()
        result = executor.execute(graph)

        self.assertGreater(result.iteration_time_ms, 0)
        self.assertGreater(result.peak_memory_mb, 0)

    def test_symbolic_comm_bytes_evaluation(self):
        inferer = SymbolicShapeInferer()
        B = inferer.get_symbol('B')
        S = inferer.get_symbol('S')
        H = inferer.get_symbol('H')

        graph = ComputationalGraph()
        graph.model_config = {'hidden_size': 1024}
        graph.strategy_config = {'micro_batch_size': 4, 'seq_length': 64, 'tp_size': 2}

        comm_node = graph.add_node(
            node_id="comm_node",
            op_type="all_reduce",
            node_type=OpType.COMMUNICATION,
            comm_type=CommType.ALL_REDUCE,
            comm_size=2,
            tensor_shape=(B, S, H),
            symbolic_comm_bytes=B * S * H * 2,
            tensor_dtype_size=2,
        )
        graph.forward_nodes.append(comm_node.node_id)

        executor = VirtualExecutor()
        result = executor.execute(graph)

        expected_bytes = 4 * 64 * 1024 * 2
        self.assertGreater(result.comm_time_ms, 0)

    def test_throughput_and_memory_efficiency(self):
        graph = ComputationalGraph()
        graph.model_config = {'hidden_size': 512}
        graph.strategy_config = {
            'micro_batch_size': 2,
            'global_batch_size': 8,
            'seq_length': 128,
            'tp_size': 1,
        }

        node = graph.add_node(
            node_id="compute",
            op_type="Linear",
            node_type=OpType.COMPUTE,
            output_shapes=[(128, 2, 512)],
            params={"weight": (512, 512)},
        )
        graph.forward_nodes.append(node.node_id)

        config = ExecutionConfig(memory_per_device_gb=80.0)
        executor = VirtualExecutor(config=config)
        result = executor.execute(graph)

        expected_tokens = 8 * 128
        expected_throughput = expected_tokens / (result.iteration_time_ms / 1000.0)
        self.assertAlmostEqual(result.throughput_tokens_per_sec, expected_throughput, places=2)

        self.assertGreaterEqual(result.memory_efficiency, 0.0)
        self.assertLessEqual(result.memory_efficiency, 1.0)


if __name__ == "__main__":
    unittest.main()
