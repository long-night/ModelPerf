import unittest
from modelperf.capture.graph import GraphNode, ComputationalGraph, OpType, ModuleCategory
from modelperf.capture.hierarchy import (
    infer_module_category,
    build_hierarchy_paths,
    get_hierarchy_summary,
    get_nodes_under_hierarchy,
)


class TestInferModuleCategory(unittest.TestCase):
    def test_parallel_linear(self):
        self.assertEqual(
            infer_module_category("RowParallelLinear"),
            ModuleCategory.PARALLEL_LINEAR
        )
        self.assertEqual(
            infer_module_category("Linear_qkv", "layers.0.attention.q_proj"),
            ModuleCategory.PARALLEL_LINEAR
        )

    def test_attention(self):
        self.assertEqual(
            infer_module_category("SelfAttention"),
            ModuleCategory.ATTENTION
        )
        self.assertEqual(
            infer_module_category("flash_attn_func"),
            ModuleCategory.ATTENTION
        )

    def test_transformer_layer(self):
        self.assertEqual(
            infer_module_category("TransformerLayer"),
            ModuleCategory.TRANSFORMER_LAYER
        )

    def test_optimizer(self):
        self.assertEqual(
            infer_module_category("Adam", "optimizer"),
            ModuleCategory.OPTIMIZER
        )

    def test_communication(self):
        self.assertEqual(
            infer_module_category("all_reduce"),
            ModuleCategory.COMMUNICATION
        )

    def test_other_fallback(self):
        self.assertEqual(
            infer_module_category("UnknownOp"),
            ModuleCategory.OTHER
        )


class TestBuildHierarchyPaths(unittest.TestCase):
    def test_basic_paths(self):
        graph = ComputationalGraph()
        n1 = graph.add_node(
            node_id="mod_0", op_type="TransformerLayer",
            node_type=OpType.AUTOGRAD_FWD, module_path="model.layers.0"
        )
        n2 = graph.add_node(
            node_id="mod_1", op_type="RowParallelLinear",
            node_type=OpType.AUTOGRAD_FWD, module_path="model.layers.0.attention.q_proj"
        )
        n3 = graph.add_node(
            node_id="mod_2", op_type="LayerNorm",
            node_type=OpType.AUTOGRAD_FWD, module_path="model.layers.0.norm1"
        )

        build_hierarchy_paths(graph.nodes)

        self.assertEqual(n1.module_level, 3)
        self.assertEqual(n1.hierarchy_path, "model.layers.0")
        self.assertEqual(n1.module_category, "transformer_layer")

        self.assertEqual(n2.module_level, 5)
        self.assertEqual(n2.hierarchy_path, "model.layers.0.attention.q_proj")
        self.assertEqual(n2.parent_module_id, "mod_0")
        self.assertEqual(n2.module_category, "parallel_linear")

        self.assertEqual(n3.module_level, 4)
        self.assertEqual(n3.parent_module_id, "mod_0")
        self.assertEqual(n3.module_category, "norm")

    def test_orphan_node(self):
        graph = ComputationalGraph()
        n = graph.add_node(
            node_id="mod_0", op_type="Linear",
            node_type=OpType.AUTOGRAD_FWD
        )
        build_hierarchy_paths(graph.nodes)
        self.assertEqual(n.module_level, 0)
        self.assertIsNone(n.hierarchy_path)
        self.assertEqual(n.module_category, "parallel_linear")


class TestGetHierarchySummary(unittest.TestCase):
    def test_summary(self):
        graph = ComputationalGraph()
        graph.add_node(
            node_id="a", op_type="Linear",
            node_type=OpType.COMPUTE, module_category="parallel_linear"
        )
        graph.add_node(
            node_id="b", op_type="LayerNorm",
            node_type=OpType.COMPUTE, module_category="norm"
        )
        graph.add_node(
            node_id="c", op_type="Linear",
            node_type=OpType.COMPUTE, module_category="parallel_linear"
        )

        summary = get_hierarchy_summary(graph.nodes)
        self.assertEqual(summary["parallel_linear"]["count"], 2)
        self.assertEqual(summary["norm"]["count"], 1)


class TestGetNodesUnderHierarchy(unittest.TestCase):
    def test_prefix_match(self):
        graph = ComputationalGraph()
        n1 = graph.add_node(
            node_id="a", op_type="Linear",
            node_type=OpType.COMPUTE, hierarchy_path="model.layers.0.attention.q_proj"
        )
        n2 = graph.add_node(
            node_id="b", op_type="LayerNorm",
            node_type=OpType.COMPUTE, hierarchy_path="model.layers.0.norm1"
        )
        n3 = graph.add_node(
            node_id="c", op_type="Linear",
            node_type=OpType.COMPUTE, hierarchy_path="model.layers.1.attention.q_proj"
        )

        results = get_nodes_under_hierarchy(graph.nodes, "model.layers.0")
        self.assertEqual(len(results), 2)
        self.assertIn(n1, results)
        self.assertIn(n2, results)
        self.assertNotIn(n3, results)


if __name__ == "__main__":
    unittest.main()
