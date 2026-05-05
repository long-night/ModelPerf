import unittest
from modelperf.capture.graph import GraphNode, ComputationalGraph, OpType, CommType


class TestGraphNode(unittest.TestCase):
    def test_basic_creation(self):
        node = GraphNode(
            node_id="test_node",
            op_type="linear",
            node_type=OpType.COMPUTE
        )
        self.assertEqual(node.node_id, "test_node")
        self.assertEqual(node.op_type, "linear")
        self.assertEqual(node.node_type, OpType.COMPUTE)
        
    def test_node_with_shapes(self):
        node = GraphNode(
            node_id="compute_node",
            op_type="matmul",
            node_type=OpType.COMPUTE,
            input_shapes=[(8, 128, 4096), (4096, 1024)],
            output_shapes=[(8, 128, 1024)]
        )
        self.assertEqual(len(node.input_shapes), 2)
        self.assertEqual(node.output_shapes[0], (8, 128, 1024))
        
    def test_communication_node(self):
        node = GraphNode(
            node_id="allreduce_node",
            op_type="all_reduce",
            node_type=OpType.COMMUNICATION,
            comm_type=CommType.ALL_REDUCE,
            comm_size=8,
            comm_bytes=16777216
        )
        self.assertEqual(node.comm_type, CommType.ALL_REDUCE)
        self.assertEqual(node.comm_size, 8)
        self.assertEqual(node.comm_bytes, 16777216)
        
    def test_node_to_dict(self):
        node = GraphNode(
            node_id="dict_test",
            op_type="test_op",
            node_type=OpType.AUTOGRAD_FWD
        )
        d = node.to_dict()
        self.assertEqual(d["node_id"], "dict_test")
        self.assertEqual(d["op_type"], "test_op")
        self.assertEqual(d["node_type"], "autograd_fwd")
        
    def test_node_with_params(self):
        node = GraphNode(
            node_id="param_node",
            op_type="linear",
            node_type=OpType.COMPUTE,
            params={"weight": (4096, 1024), "bias": (1024,)}
        )
        self.assertEqual(node.params["weight"], (4096, 1024))
        self.assertEqual(node.params["bias"], (1024,))


class TestComputationalGraph(unittest.TestCase):
    def test_empty_graph(self):
        graph = ComputationalGraph()
        self.assertEqual(len(graph.nodes), 0)
        self.assertEqual(len(graph.edges), 0)
        
    def test_add_node(self):
        graph = ComputationalGraph()
        node = graph.add_node(
            node_id="node1",
            op_type="linear",
            node_type=OpType.COMPUTE
        )
        self.assertEqual(len(graph.nodes), 1)
        self.assertEqual(node.node_id, "node1")
        
    def test_add_node_auto_id(self):
        graph = ComputationalGraph()
        node1 = graph.add_node(op_type="linear", node_type=OpType.COMPUTE)
        node2 = graph.add_node(op_type="relu", node_type=OpType.COMPUTE)
        self.assertIn("node_0", graph.nodes)
        self.assertIn("node_1", graph.nodes)
        
    def test_add_edge(self):
        graph = ComputationalGraph()
        graph.add_node(node_id="n1", op_type="a", node_type=OpType.COMPUTE)
        graph.add_node(node_id="n2", op_type="b", node_type=OpType.COMPUTE)
        graph.add_edge("n1", "n2")
        
        self.assertEqual(len(graph.edges), 1)
        self.assertIn(("n1", "n2"), graph.edges)
        self.assertIn("n2", graph.nodes["n1"].outputs)
        self.assertIn("n1", graph.nodes["n2"].inputs)
        
    def test_get_nodes_by_type(self):
        graph = ComputationalGraph()
        graph.add_node(node_id="c1", op_type="compute", node_type=OpType.COMPUTE)
        graph.add_node(node_id="c2", op_type="comm", node_type=OpType.COMMUNICATION)
        graph.add_node(node_id="c3", op_type="compute", node_type=OpType.COMPUTE)
        
        compute_nodes = graph.get_nodes_by_type(OpType.COMPUTE)
        comm_nodes = graph.get_nodes_by_type(OpType.COMMUNICATION)
        
        self.assertEqual(len(compute_nodes), 2)
        self.assertEqual(len(comm_nodes), 1)
        
    def test_get_comm_nodes(self):
        graph = ComputationalGraph()
        graph.add_node(node_id="comm1", op_type="all_reduce", node_type=OpType.COMMUNICATION)
        graph.add_node(node_id="comp1", op_type="linear", node_type=OpType.COMPUTE)
        
        comm_nodes = graph.get_comm_nodes()
        self.assertEqual(len(comm_nodes), 1)
        
    def test_summary(self):
        graph = ComputationalGraph()
        graph.add_node(node_id="n1", op_type="a", node_type=OpType.COMPUTE)
        graph.add_node(node_id="n2", op_type="b", node_type=OpType.COMMUNICATION)
        graph.add_edge("n1", "n2")
        graph.forward_nodes = ["n1", "n2"]
        
        summary = graph.summary()
        self.assertEqual(summary["total_nodes"], 2)
        self.assertEqual(summary["compute_nodes"], 1)
        self.assertEqual(summary["comm_nodes"], 1)
        self.assertEqual(summary["edges"], 1)
        
    def test_to_dict(self):
        graph = ComputationalGraph()
        graph.add_node(node_id="n1", op_type="test", node_type=OpType.COMPUTE)
        graph.model_config = {"hidden_size": 4096}
        
        d = graph.to_dict()
        self.assertIn("nodes", d)
        self.assertIn("edges", d)
        self.assertIn("model_config", d)
        self.assertEqual(d["model_config"]["hidden_size"], 4096)
        
    def test_edge_updates_io(self):
        graph = ComputationalGraph()
        graph.add_node(node_id="a", op_type="op_a", node_type=OpType.COMPUTE)
        graph.add_node(node_id="b", op_type="op_b", node_type=OpType.COMPUTE)
        graph.add_node(node_id="c", op_type="op_c", node_type=OpType.COMPUTE)
        
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        
        self.assertEqual(graph.nodes["a"].outputs, ["b"])
        self.assertEqual(graph.nodes["b"].inputs, ["a"])
        self.assertEqual(graph.nodes["b"].outputs, ["c"])
        self.assertEqual(graph.nodes["c"].inputs, ["b"])


class TestEdgeCases(unittest.TestCase):
    def test_empty_graph_summary(self):
        graph = ComputationalGraph()
        summary = graph.summary()
        self.assertEqual(summary["total_nodes"], 0)
        
    def test_node_without_op_type(self):
        node = GraphNode(
            node_id="minimal",
            op_type="",
            node_type=OpType.MEMORY
        )
        self.assertEqual(node.op_type, "")
        
    def test_edge_between_nonexistent_nodes(self):
        graph = ComputationalGraph()
        graph.add_edge("nonexistent1", "nonexistent2")
        self.assertEqual(len(graph.edges), 1)


if __name__ == "__main__":
    unittest.main()
