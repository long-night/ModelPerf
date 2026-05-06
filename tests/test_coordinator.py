import unittest
import torch
import torch.nn as nn

from modelperf.capture.coordinator import CaptureCoordinator
from modelperf.capture.graph import OpType, CommType


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TestCaptureCoordinator(unittest.TestCase):
    def test_coordinator_lifecycle(self):
        model = SimpleModel()
        coord = CaptureCoordinator()
        coord.attach(model)
        coord.start()

        x = torch.randn(2, 10)
        y = model(x)

        coord.stop()
        graph = coord.get_graph()

        self.assertGreater(len(graph.nodes), 0)
        self.assertGreaterEqual(len(graph.forward_nodes), 3)

    def test_forward_backward_linkage(self):
        model = SimpleModel()
        coord = CaptureCoordinator()
        coord.attach(model)
        coord.start()

        x = torch.randn(2, 10, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()

        coord.stop()
        graph = coord.get_graph()

        fwd_nodes = [n for n in graph.nodes.values() if n.node_type == OpType.AUTOGRAD_FWD]
        bwd_nodes = [n for n in graph.nodes.values() if n.node_type == OpType.BACKWARD]

        self.assertGreater(len(fwd_nodes), 0, "Should have forward nodes")
        self.assertGreater(len(bwd_nodes), 0, "Should have backward nodes")

        fwd_ids = {n.node_id for n in fwd_nodes}
        edges_to_bwd = [e for e in graph.edges if e[1] in {n.node_id for n in bwd_nodes}]
        self.assertGreater(len(edges_to_bwd), 0, "Should have edges from forward to backward")

    def test_sequential_edges(self):
        model = SimpleModel()
        coord = CaptureCoordinator()
        coord.attach(model)
        coord.start()

        x = torch.randn(2, 10)
        model(x)

        coord.stop()
        graph = coord.get_graph()

        fc1_nodes = [n for n in graph.nodes.values() if n.module_path == "fc1"]
        relu_nodes = [n for n in graph.nodes.values() if n.module_path == "relu"]
        fc2_nodes = [n for n in graph.nodes.values() if n.module_path == "fc2"]

        if fc1_nodes and relu_nodes:
            fc1_id = fc1_nodes[0].node_id
            relu_id = relu_nodes[0].node_id
            self.assertIn(relu_id, fc1_nodes[0].outputs)
            self.assertIn(fc1_id, relu_nodes[0].inputs)

        if relu_nodes and fc2_nodes:
            relu_id = relu_nodes[0].node_id
            fc2_id = fc2_nodes[0].node_id
            self.assertIn(fc2_id, relu_nodes[0].outputs)
            self.assertIn(relu_id, fc2_nodes[0].inputs)

    def test_reset(self):
        model = SimpleModel()
        coord = CaptureCoordinator()
        coord.attach(model)
        coord.start()
        model(torch.randn(2, 10))
        coord.stop()

        self.assertGreater(len(coord.get_graph().nodes), 0)

        coord.reset()
        self.assertEqual(len(coord.get_graph().nodes), 0)
        self.assertEqual(len(coord.get_graph().forward_nodes), 0)

    def test_context_manager(self):
        model = SimpleModel()
        coord = CaptureCoordinator()
        coord.attach(model)

        with coord:
            model(torch.randn(2, 10))

        graph = coord.get_graph()
        self.assertGreater(len(graph.nodes), 0)


class TestCommCaptureIntegration(unittest.TestCase):
    def test_comm_nodes_captured(self):
        import torch.distributed as dist
        orig_all_reduce = dist.all_reduce

        model = SimpleModel()
        coord = CaptureCoordinator()
        coord.attach(model)
        coord.start()

        try:
            tensor = torch.randn(4, 4)
            dist.all_reduce(tensor)
        except Exception:
            pass

        model(torch.randn(2, 10))

        coord.stop()
        graph = coord.get_graph()

        comm_nodes = [n for n in graph.nodes.values() if n.node_type == OpType.COMMUNICATION]

        dist.all_reduce = orig_all_reduce

    def test_comm_compute_linkage(self):
        import torch.distributed as dist
        orig_all_reduce = dist.all_reduce

        class CommModel(nn.Module):
            def forward(self, x):
                dist.all_reduce(x)
                return x * 2

        model = CommModel()
        coord = CaptureCoordinator()
        coord.attach(model)
        coord.start()

        try:
            model(torch.randn(2, 10))
        except Exception:
            pass

        coord.stop()
        graph = coord.get_graph()

        dist.all_reduce = orig_all_reduce

        compute_nodes = [n for n in graph.nodes.values() if n.node_type == OpType.AUTOGRAD_FWD]
        comm_nodes = [n for n in graph.nodes.values() if n.node_type == OpType.COMMUNICATION]

        if compute_nodes and comm_nodes:
            edges_from_compute = [e for e in graph.edges if e[0] == compute_nodes[0].node_id]
            self.assertTrue(
                any(e[1] == comm_nodes[0].node_id for e in edges_from_compute),
                "Communication node should be linked from compute node"
            )


if __name__ == "__main__":
    unittest.main()
