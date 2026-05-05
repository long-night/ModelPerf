import unittest
import math
from modelperf.simulation.bandwidth import BandwidthModel, CollectiveOp, Algorithm


class TestBandwidthModel(unittest.TestCase):
    def test_default_initialization(self):
        model = BandwidthModel()
        self.assertEqual(model.bandwidth_gbs, 600.0)
        self.assertEqual(model.latency_us, 2.0)
        
    def test_custom_initialization(self):
        model = BandwidthModel(
            bandwidth_gbs=1000.0,
            latency_us=5.0
        )
        self.assertEqual(model.bandwidth_gbs, 1000.0)
        self.assertEqual(model.latency_us, 5.0)
        
    def test_all_reduce_ring_formula(self):
        model = BandwidthModel(bandwidth_gbs=100.0, latency_us=0.0)
        comm_bytes = 1e9
        comm_size = 8
        time = model.estimate_comm_time(
            comm_bytes=comm_bytes,
            comm_size=comm_size,
            algorithm="ring",
            collective_op="all_reduce"
        )
        bytes_gb = comm_bytes / 1e9
        factor = 2.0 * (comm_size - 1) / comm_size
        expected = factor * bytes_gb / 100.0
        self.assertAlmostEqual(time, expected, places=10)
        
    def test_all_reduce_tree_formula(self):
        model = BandwidthModel(bandwidth_gbs=100.0, latency_us=0.0)
        comm_bytes = 1e9
        comm_size = 8
        time = model.estimate_comm_time(
            comm_bytes=comm_bytes,
            comm_size=comm_size,
            algorithm="tree",
            collective_op="all_reduce"
        )
        bytes_gb = comm_bytes / 1e9
        factor = 2.0
        expected = factor * bytes_gb / 100.0
        self.assertAlmostEqual(time, expected, places=10)
        
    def test_all_gather_formula(self):
        model = BandwidthModel(bandwidth_gbs=100.0, latency_us=0.0)
        comm_bytes = 1e9
        comm_size = 4
        time = model.estimate_comm_time(
            comm_bytes=comm_bytes,
            comm_size=comm_size,
            algorithm="ring",
            collective_op="all_gather"
        )
        bytes_gb = comm_bytes / 1e9
        factor = (comm_size - 1) / comm_size
        expected = factor * bytes_gb / 100.0
        self.assertAlmostEqual(time, expected, places=10)
        
    def test_reduce_scatter_formula(self):
        model = BandwidthModel(bandwidth_gbs=100.0, latency_us=0.0)
        comm_bytes = 1e9
        comm_size = 4
        time = model.estimate_comm_time(
            comm_bytes=comm_bytes,
            comm_size=comm_size,
            algorithm="ring",
            collective_op="reduce_scatter"
        )
        bytes_gb = comm_bytes / 1e9
        factor = (comm_size - 1) / comm_size
        expected = factor * bytes_gb / 100.0
        self.assertAlmostEqual(time, expected, places=10)
        
    def test_single_rank(self):
        model = BandwidthModel(bandwidth_gbs=100.0, latency_us=5.0)
        time = model.estimate_comm_time(
            comm_bytes=1e9,
            comm_size=1,
            algorithm="ring",
            collective_op="all_reduce"
        )
        self.assertEqual(time, 0.0)
        
    def test_zero_bytes(self):
        model = BandwidthModel(bandwidth_gbs=100.0, latency_us=5.0)
        time = model.estimate_comm_time(
            comm_bytes=0,
            comm_size=8,
            algorithm="ring",
            collective_op="all_reduce"
        )
        self.assertEqual(time, 5.0 * 1e-6)
        
    def test_invalid_comm_size(self):
        model = BandwidthModel()
        with self.assertRaises(ValueError):
            model.estimate_comm_time(
                comm_bytes=1e9,
                comm_size=0,
                algorithm="ring",
                collective_op="all_reduce"
            )
        with self.assertRaises(ValueError):
            model.estimate_comm_time(
                comm_bytes=1e9,
                comm_size=-1,
                algorithm="ring",
                collective_op="all_reduce"
            )
            
    def test_invalid_comm_bytes(self):
        model = BandwidthModel()
        with self.assertRaises(ValueError):
            model.estimate_comm_time(
                comm_bytes=-1e9,
                comm_size=8,
                algorithm="ring",
                collective_op="all_reduce"
            )
            
    def test_latency_included(self):
        model = BandwidthModel(bandwidth_gbs=100.0, latency_us=10.0)
        time = model.estimate_comm_time(
            comm_bytes=0,
            comm_size=2,
            algorithm="ring",
            collective_op="all_reduce"
        )
        expected_latency = 10.0 * 1e-6
        self.assertAlmostEqual(time, expected_latency, places=10)
        
    def test_bandwidth_override(self):
        model = BandwidthModel(bandwidth_gbs=100.0, latency_us=0.0)
        time = model.estimate_comm_time(
            comm_bytes=1e9,
            comm_size=2,
            algorithm="ring",
            collective_op="all_reduce",
            bandwidth=200.0
        )
        bytes_gb = 1e9 / 1e9
        factor = 2.0 * (2 - 1) / 2
        expected = factor * bytes_gb / 200.0
        self.assertAlmostEqual(time, expected, places=10)
        
    def test_latency_override(self):
        model = BandwidthModel(bandwidth_gbs=100.0, latency_us=10.0)
        time = model.estimate_comm_time(
            comm_bytes=0,
            comm_size=2,
            algorithm="ring",
            collective_op="all_reduce",
            latency=20.0
        )
        expected = 20.0 * 1e-6
        self.assertAlmostEqual(time, expected, places=10)
        
    def test_broadcast_tree(self):
        model = BandwidthModel(bandwidth_gbs=100.0, latency_us=0.0)
        comm_bytes = 1e9
        comm_size = 8
        time = model.estimate_comm_time(
            comm_bytes=comm_bytes,
            comm_size=comm_size,
            algorithm="tree",
            collective_op="broadcast"
        )
        bytes_gb = comm_bytes / 1e9
        factor = math.log2(comm_size)
        expected = factor * bytes_gb / 100.0
        self.assertAlmostEqual(time, expected, places=10)
        
    def test_broadcast_chain(self):
        model = BandwidthModel(bandwidth_gbs=100.0, latency_us=0.0)
        comm_bytes = 1e9
        comm_size = 8
        time = model.estimate_comm_time(
            comm_bytes=comm_bytes,
            comm_size=comm_size,
            algorithm="ring",
            collective_op="broadcast"
        )
        bytes_gb = comm_bytes / 1e9
        factor = comm_size - 1
        expected = factor * bytes_gb / 100.0
        self.assertAlmostEqual(time, expected, places=10)


class TestCollectiveOp(unittest.TestCase):
    def test_enum_values(self):
        self.assertEqual(CollectiveOp.ALL_REDUCE.value, "all_reduce")
        self.assertEqual(CollectiveOp.ALL_GATHER.value, "all_gather")
        self.assertEqual(CollectiveOp.REDUCE_SCATTER.value, "reduce_scatter")
        self.assertEqual(CollectiveOp.ALL_TO_ALL.value, "all_to_all")
        self.assertEqual(CollectiveOp.BROADCAST.value, "broadcast")
        self.assertEqual(CollectiveOp.REDUCE.value, "reduce")


class TestAlgorithm(unittest.TestCase):
    def test_enum_values(self):
        self.assertEqual(Algorithm.RING.value, "ring")
        self.assertEqual(Algorithm.TREE.value, "tree")
        self.assertEqual(Algorithm.DIRECT.value, "direct")
        self.assertEqual(Algorithm.CHAINED.value, "chained")


if __name__ == "__main__":
    unittest.main()
