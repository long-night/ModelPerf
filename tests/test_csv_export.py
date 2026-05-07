import os
import tempfile
import unittest
import csv
from modelperf.capture.graph import ComputationalGraph, GraphNode, OpType, CommType


class TestExportCSV(unittest.TestCase):
    def test_export_nodes_and_edges(self):
        graph = ComputationalGraph()
        graph.add_node(
            node_id="n1", op_type="Linear", node_type=OpType.COMPUTE,
            input_shapes=[(2, 128, 576)], output_shapes=[(2, 128, 1024)],
            params={"weight": (576, 1024)},
        )
        graph.add_node(
            node_id="n2", op_type="all_reduce", node_type=OpType.COMMUNICATION,
            comm_type=CommType.ALL_REDUCE, comm_size=8, comm_bytes=16777216,
        )
        graph.add_edge("n1", "n2")

        with tempfile.TemporaryDirectory() as tmpdir:
            graph.export_csv(tmpdir)

            node_path = os.path.join(tmpdir, "nodes.csv")
            self.assertTrue(os.path.exists(node_path))
            with open(node_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0]["node_id"], "n1")
                self.assertEqual(rows[1]["node_id"], "n2")

            edge_path = os.path.join(tmpdir, "edges.csv")
            self.assertTrue(os.path.exists(edge_path))
            with open(edge_path, "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
                self.assertEqual(header, ["from_id", "to_id"])
                rows = list(reader)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0], ["n1", "n2"])

    def test_export_empty_graph(self):
        graph = ComputationalGraph()
        with tempfile.TemporaryDirectory() as tmpdir:
            graph.export_csv(tmpdir)
            node_path = os.path.join(tmpdir, "nodes.csv")
            self.assertTrue(os.path.exists(node_path))

    def test_hierarchy_tree(self):
        graph = ComputationalGraph()
        graph.add_node(
            node_id="root", op_type="TransformerLayer", node_type=OpType.COMPUTE,
            hierarchy_path="model.layers.0"
        )
        graph.add_node(
            node_id="child", op_type="Linear", node_type=OpType.COMPUTE,
            hierarchy_path="model.layers.0.attention.q_proj"
        )

        tree = graph.get_hierarchy_tree()
        self.assertIn("root", tree)
        self.assertIn("node_map", tree)
        self.assertEqual(tree["node_map"]["child"], "model.layers.0.attention.q_proj")


if __name__ == "__main__":
    unittest.main()
