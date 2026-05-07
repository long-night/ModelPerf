import os
import tempfile
import unittest
from modelperf.capture.graph import GraphNode, OpType
from modelperf.testing.op_test_generator import OpTestGenerator, _sanitize_id


class TestSanitizeId(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(_sanitize_id("mod.0"), "mod_0")
        self.assertEqual(_sanitize_id("aten::mm"), "aten__mm")


class TestOpTestGenerator(unittest.TestCase):
    def test_generate_for_node(self):
        node = GraphNode(
            node_id="mod_0",
            op_type="Linear",
            node_type=OpType.COMPUTE,
            input_shapes=[(2, 128, 576)],
            output_shapes=[(2, 128, 1024)],
            params={"weight": (576, 1024)},
            module_category="parallel_linear",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_linear.py")
            gen = OpTestGenerator(output_dir=tmpdir)
            gen.generate_for_node(node, output_path)

            self.assertTrue(os.path.exists(output_path))
            with open(output_path, "r") as f:
                content = f.read()
                self.assertIn("def test_mod_0_correctness", content)
                self.assertIn("def test_mod_0_performance", content)
                self.assertIn("OpBenchmark", content)
                self.assertIn("OpAdapter", content)

    def test_generate_all(self):
        from modelperf.capture.graph import ComputationalGraph

        graph = ComputationalGraph()
        graph.add_node(
            node_id="n1", op_type="Linear", node_type=OpType.COMPUTE,
            input_shapes=[(2, 128, 576)], output_shapes=[(2, 128, 1024)],
            params={"weight": (576, 1024)}, module_category="parallel_linear",
        )
        graph.add_node(
            node_id="n2", op_type="LayerNorm", node_type=OpType.COMPUTE,
            input_shapes=[(2, 128, 1024)], output_shapes=[(2, 128, 1024)],
            module_category="norm",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            gen = OpTestGenerator(output_dir=tmpdir)
            gen.generate_all(graph)

            self.assertTrue(os.path.exists(os.path.join(tmpdir, "test_parallel_linear.py")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "test_norm.py")))


if __name__ == "__main__":
    unittest.main()
