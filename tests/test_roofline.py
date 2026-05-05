import unittest
from modelperf.simulation.roofline import RooflineModel, create_a100_model, create_h100_model, create_v100_model


class TestRooflineModel(unittest.TestCase):
    def test_default_initialization(self):
        model = RooflineModel()
        self.assertEqual(model.peak_compute_tflops, 312.0)
        self.assertEqual(model.peak_bandwidth_gbs, 2039.0)
        
    def test_custom_initialization(self):
        model = RooflineModel(
            peak_compute_tflops=500.0,
            peak_bandwidth_gbs=3000.0
        )
        self.assertEqual(model.peak_compute_tflops, 500.0)
        self.assertEqual(model.peak_bandwidth_gbs, 3000.0)
        
    def test_estimate_time_compute_bound(self):
        model = RooflineModel(
            peak_compute_tflops=100.0,
            peak_bandwidth_gbs=1000.0
        )
        flops = 1e12
        ai = 1000
        time = model.estimate_time(flops, ai)
        expected_compute = flops / (100.0 * 1e12)
        self.assertAlmostEqual(time, expected_compute, places=10)
        
    def test_estimate_time_memory_bound(self):
        model = RooflineModel(
            peak_compute_tflops=1000.0,
            peak_bandwidth_gbs=10.0
        )
        flops = 1e12
        ai = 1
        time = model.estimate_time(flops, ai)
        bytes_accessed = flops / ai
        expected_memory = bytes_accessed / (10.0 * 1e9)
        self.assertAlmostEqual(time, expected_memory, places=10)
        
    def test_estimate_time_formula(self):
        model = RooflineModel(
            peak_compute_tflops=100.0,
            peak_bandwidth_gbs=100.0
        )
        flops = 1e12
        ai = 10
        time = model.estimate_time(flops, ai)
        compute_time = flops / (100.0 * 1e12)
        bytes_accessed = flops / ai
        memory_time = bytes_accessed / (100.0 * 1e9)
        expected = max(compute_time, memory_time)
        self.assertAlmostEqual(time, expected, places=10)
        
    def test_estimate_time_zero_flops(self):
        model = RooflineModel()
        time = model.estimate_time(0, 100)
        self.assertEqual(time, 0.0)
        
    def test_estimate_time_negative_flops(self):
        model = RooflineModel()
        time = model.estimate_time(-1e12, 100)
        self.assertEqual(time, 0.0)
        
    def test_estimate_time_invalid_arithmetic_intensity(self):
        model = RooflineModel()
        with self.assertRaises(ValueError):
            model.estimate_time(1e12, 0)
        with self.assertRaises(ValueError):
            model.estimate_time(1e12, -1)
            
    def test_estimate_time_with_overrides(self):
        model = RooflineModel(
            peak_compute_tflops=100.0,
            peak_bandwidth_gbs=100.0
        )
        flops = 1e12
        ai = 10
        time = model.estimate_time(
            flops, ai,
            peak_compute=200.0,
            peak_bandwidth=200.0
        )
        compute_time = flops / (200.0 * 1e12)
        bytes_accessed = flops / ai
        memory_time = bytes_accessed / (200.0 * 1e9)
        expected = max(compute_time, memory_time)
        self.assertAlmostEqual(time, expected, places=10)
        
    def test_estimate_time_simple(self):
        model = RooflineModel(
            peak_compute_tflops=100.0,
            peak_bandwidth_gbs=100.0
        )
        flops = 1e12
        bytes_accessed = 1e11
        time = model.estimate_time_simple(flops, bytes_accessed)
        compute_time = flops / (100.0 * 1e12)
        memory_time = bytes_accessed / (100.0 * 1e9)
        expected = max(compute_time, memory_time)
        self.assertAlmostEqual(time, expected, places=10)
        
    def test_estimate_time_simple_zero_flops(self):
        model = RooflineModel()
        time = model.estimate_time_simple(0, 1e11)
        self.assertEqual(time, 0.0)
        
    def test_estimate_time_simple_zero_bytes(self):
        model = RooflineModel()
        time = model.estimate_time_simple(1e12, 0)
        self.assertEqual(time, 0.0)


class TestGPUFactoryFunctions(unittest.TestCase):
    def test_create_a100_model(self):
        model = create_a100_model()
        self.assertEqual(model.peak_compute_tflops, 312.0)
        self.assertEqual(model.peak_bandwidth_gbs, 2039.0)
        
    def test_create_h100_model(self):
        model = create_h100_model()
        self.assertEqual(model.peak_compute_tflops, 989.0)
        self.assertEqual(model.peak_bandwidth_gbs, 3350.0)
        
    def test_create_v100_model(self):
        model = create_v100_model()
        self.assertEqual(model.peak_compute_tflops, 125.0)
        self.assertEqual(model.peak_bandwidth_gbs, 900.0)


class TestEdgeCases(unittest.TestCase):
    def test_very_small_flops(self):
        model = RooflineModel()
        time = model.estimate_time(1e-6, 100)
        self.assertGreaterEqual(time, 0.0)
        
    def test_very_large_flops(self):
        model = RooflineModel()
        time = model.estimate_time(1e18, 1000)
        self.assertGreater(time, 0.0)
        
    def test_arithmetic_intensity_boundary(self):
        model = RooflineModel(
            peak_compute_tflops=100.0,
            peak_bandwidth_gbs=100.0
        )
        ai_boundary = 100.0 * 1e12 / (100.0 * 1e9)
        flops = 1e12
        time = model.estimate_time(flops, ai_boundary)
        compute_time = flops / (100.0 * 1e12)
        self.assertAlmostEqual(time, compute_time, places=10)


if __name__ == "__main__":
    unittest.main()
