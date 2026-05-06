import unittest

from modelperf.capture.graph import ComputationalGraph, GraphNode, OpType, CommType
from modelperf.simulation.virtual_executor import VirtualExecutor
from modelperf.simulation.execution_result import ExecutionConfig


class TestOverlapAndPipeline(unittest.TestCase):
    def test_overlap_reduces_time(self):
        graph = ComputationalGraph()

        comp_node = graph.add_node(
            node_id="compute",
            op_type="Linear",
            node_type=OpType.COMPUTE,
            output_shapes=[(128, 2, 512)],
        )
        comm_node = graph.add_node(
            node_id="comm",
            op_type="all_reduce",
            node_type=OpType.COMMUNICATION,
            comm_type=CommType.ALL_REDUCE,
            comm_size=2,
            comm_bytes=1024 * 1024,
        )
        graph.forward_nodes = [comp_node.node_id, comm_node.node_id]

        config_overlap = ExecutionConfig(enable_overlap=True)
        executor_overlap = VirtualExecutor(config=config_overlap)
        result_overlap = executor_overlap.execute(graph)

        config_no_overlap = ExecutionConfig(enable_overlap=False)
        executor_no_overlap = VirtualExecutor(config=config_no_overlap)
        result_no_overlap = executor_no_overlap.execute(graph)

        self.assertLessEqual(
            result_overlap.iteration_time_ms,
            result_no_overlap.iteration_time_ms,
            "Overlap should reduce or equal total time"
        )

    def test_pipeline_schedule_1f1b_vs_gpipe(self):
        graph = ComputationalGraph()
        graph.strategy_config = {'pp_size': 4, 'tp_size': 1}

        for i in range(3):
            fwd = graph.add_node(
                node_id=f"node_{i}",
                op_type="Linear",
                node_type=OpType.COMPUTE,
                output_shapes=[(128, 2, 512)],
            )
            bwd = graph.add_node(
                node_id=f"node_{i}_bwd",
                op_type="Linear_backward",
                node_type=OpType.BACKWARD,
                output_shapes=[(128, 2, 512)],
            )
            graph.forward_nodes.append(fwd.node_id)
            graph.backward_nodes.append(bwd.node_id)

        config_1f1b = ExecutionConfig(
            pp_size=4, num_microbatches=8, pipeline_schedule="1f1b"
        )
        executor_1f1b = VirtualExecutor(config=config_1f1b)
        result_1f1b = executor_1f1b.execute(graph)

        config_gpipe = ExecutionConfig(
            pp_size=4, num_microbatches=8, pipeline_schedule="gpipe"
        )
        executor_gpipe = VirtualExecutor(config=config_gpipe)
        result_gpipe = executor_gpipe.execute(graph)

        self.assertLess(
            result_1f1b.iteration_time_ms,
            result_gpipe.iteration_time_ms,
            "1F1B should be faster than GPipe for same config"
        )

    def test_no_pp_no_bubble(self):
        graph = ComputationalGraph()
        graph.strategy_config = {'pp_size': 1, 'tp_size': 1}

        node = graph.add_node(
            node_id="node_0",
            op_type="Linear",
            node_type=OpType.COMPUTE,
            output_shapes=[(128, 2, 512)],
        )
        graph.forward_nodes.append(node.node_id)

        config = ExecutionConfig(pp_size=1, num_microbatches=4)
        executor = VirtualExecutor(config=config)
        result = executor.execute(graph)

        self.assertEqual(result.bubble_time_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
