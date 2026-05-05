import unittest
from modelperf.simulation.memory_tracker import MemoryTracker, MemoryType, DeviceMemory, Allocation


class TestAllocation(unittest.TestCase):
    def test_basic_allocation(self):
        alloc = Allocation(size_bytes=1024, mem_type=MemoryType.ACTIVATION)
        self.assertEqual(alloc.size_bytes, 1024)
        self.assertEqual(alloc.mem_type, MemoryType.ACTIVATION)
        self.assertEqual(alloc.name, "")
        self.assertIsNone(alloc.layer_id)
        
    def test_allocation_with_name(self):
        alloc = Allocation(
            size_bytes=2048,
            mem_type=MemoryType.PARAMETER,
            name="layer1.weight",
            layer_id=0
        )
        self.assertEqual(alloc.name, "layer1.weight")
        self.assertEqual(alloc.layer_id, 0)


class TestDeviceMemory(unittest.TestCase):
    def test_basic_initialization(self):
        device = DeviceMemory(device_id=0)
        self.assertEqual(device.device_id, 0)
        self.assertEqual(device.current_usage, 0)
        self.assertEqual(device.peak_usage, 0)
        self.assertEqual(len(device.allocations), 0)
        
    def test_allocation_tracking(self):
        device = DeviceMemory(device_id=0)
        result = device.allocate("tensor1", 1024, MemoryType.ACTIVATION)
        self.assertTrue(result)
        self.assertEqual(device.current_usage, 1024)
        self.assertEqual(device.peak_usage, 1024)
        self.assertIn("tensor1", device.allocations)
        
    def test_multiple_allocations(self):
        device = DeviceMemory(device_id=0)
        device.allocate("t1", 1024, MemoryType.ACTIVATION)
        device.allocate("t2", 2048, MemoryType.PARAMETER)
        self.assertEqual(device.current_usage, 3072)
        self.assertEqual(device.peak_usage, 3072)
        
    def test_free_allocation(self):
        device = DeviceMemory(device_id=0)
        device.allocate("tensor1", 1024, MemoryType.ACTIVATION)
        freed = device.free("tensor1")
        self.assertEqual(freed, 1024)
        self.assertEqual(device.current_usage, 0)
        self.assertNotIn("tensor1", device.allocations)
        
    def test_free_nonexistent(self):
        device = DeviceMemory(device_id=0)
        freed = device.free("nonexistent")
        self.assertEqual(freed, 0)
        
    def test_reallocate_same_name(self):
        device = DeviceMemory(device_id=0)
        device.allocate("tensor", 1024, MemoryType.ACTIVATION)
        device.allocate("tensor", 2048, MemoryType.ACTIVATION)
        self.assertEqual(device.current_usage, 2048)
        self.assertEqual(device.allocations["tensor"].size_bytes, 2048)
        
    def test_get_usage_by_type(self):
        device = DeviceMemory(device_id=0)
        device.allocate("act1", 1024, MemoryType.ACTIVATION)
        device.allocate("act2", 2048, MemoryType.ACTIVATION)
        device.allocate("param", 4096, MemoryType.PARAMETER)
        
        act_usage = device.get_usage_by_type(MemoryType.ACTIVATION)
        param_usage = device.get_usage_by_type(MemoryType.PARAMETER)
        
        self.assertEqual(act_usage, 3072)
        self.assertEqual(param_usage, 4096)
        
    def test_get_peak(self):
        device = DeviceMemory(device_id=0)
        device.allocate("t1", 1024, MemoryType.ACTIVATION)
        self.assertEqual(device.get_peak(), 1024)
        device.free("t1")
        device.allocate("t2", 512, MemoryType.ACTIVATION)
        self.assertEqual(device.get_peak(), 1024)
        
    def test_clear(self):
        device = DeviceMemory(device_id=0)
        device.allocate("t1", 1024, MemoryType.ACTIVATION)
        device.allocate("t2", 2048, MemoryType.PARAMETER)
        device.clear()
        
        self.assertEqual(device.current_usage, 0)
        self.assertEqual(device.peak_usage, 0)
        self.assertEqual(len(device.allocations), 0)
        
    def test_memory_limit_exceeded(self):
        device = DeviceMemory(device_id=0, total_memory_bytes=2048)
        result1 = device.allocate("t1", 1024, MemoryType.ACTIVATION)
        self.assertTrue(result1)
        result2 = device.allocate("t2", 2048, MemoryType.ACTIVATION)
        self.assertFalse(result2)
        
    def test_allocate_zero_bytes(self):
        device = DeviceMemory(device_id=0)
        result = device.allocate("empty", 0, MemoryType.ACTIVATION)
        self.assertTrue(result)
        self.assertEqual(device.current_usage, 0)
        
    def test_allocate_negative_bytes(self):
        device = DeviceMemory(device_id=0)
        with self.assertRaises(ValueError):
            device.allocate("invalid", -1024, MemoryType.ACTIVATION)


class TestMemoryTracker(unittest.TestCase):
    def test_basic_initialization(self):
        tracker = MemoryTracker(num_devices=4, memory_per_device_gb=80.0)
        self.assertEqual(tracker.num_devices, 4)
        self.assertEqual(tracker.memory_per_device_bytes, int(80.0 * 1e9))
        self.assertEqual(len(tracker.devices), 4)
        
    def test_single_device(self):
        tracker = MemoryTracker(num_devices=1, memory_per_device_gb=40.0)
        self.assertEqual(tracker.num_devices, 1)
        self.assertIn(0, tracker.devices)
        
    def test_allocate(self):
        tracker = MemoryTracker(num_devices=2, memory_per_device_gb=80.0)
        result = tracker.allocate(
            device_id=0,
            name="tensor1",
            size_bytes=1024,
            mem_type=MemoryType.ACTIVATION
        )
        self.assertTrue(result)
        self.assertEqual(tracker.devices[0].current_usage, 1024)
        
    def test_allocate_invalid_device(self):
        tracker = MemoryTracker(num_devices=2, memory_per_device_gb=80.0)
        with self.assertRaises(ValueError):
            tracker.allocate(
                device_id=5,
                name="tensor1",
                size_bytes=1024,
                mem_type=MemoryType.ACTIVATION
            )
            
    def test_free(self):
        tracker = MemoryTracker(num_devices=2, memory_per_device_gb=80.0)
        tracker.allocate(0, "tensor1", 1024, MemoryType.ACTIVATION)
        freed = tracker.free(0, "tensor1")
        self.assertEqual(freed, 1024)
        self.assertEqual(tracker.devices[0].current_usage, 0)
        
    def test_free_invalid_device(self):
        tracker = MemoryTracker(num_devices=2, memory_per_device_gb=80.0)
        freed = tracker.free(5, "tensor1")
        self.assertEqual(freed, 0)
        
    def test_get_peak_memory(self):
        tracker = MemoryTracker(num_devices=2, memory_per_device_gb=80.0)
        tracker.allocate(0, "t1", 1024, MemoryType.ACTIVATION)
        tracker.allocate(1, "t2", 2048, MemoryType.ACTIVATION)
        
        peak_0 = tracker.get_peak_memory(device_id=0)
        peak_1 = tracker.get_peak_memory(device_id=1)
        peak_total = tracker.get_peak_memory()
        
        self.assertEqual(peak_0, 1024)
        self.assertEqual(peak_1, 2048)
        self.assertEqual(peak_total, 3072)
        
    def test_get_current_memory(self):
        tracker = MemoryTracker(num_devices=2, memory_per_device_gb=80.0)
        tracker.allocate(0, "t1", 1024, MemoryType.ACTIVATION)
        
        current_0 = tracker.get_current_memory(device_id=0)
        current_total = tracker.get_current_memory()
        
        self.assertEqual(current_0, 1024)
        self.assertEqual(current_total, 1024)
        
    def test_get_memory_by_type(self):
        tracker = MemoryTracker(num_devices=2, memory_per_device_gb=80.0)
        tracker.allocate(0, "act1", 1024, MemoryType.ACTIVATION)
        tracker.allocate(0, "act2", 2048, MemoryType.ACTIVATION)
        tracker.allocate(0, "param", 4096, MemoryType.PARAMETER)
        
        act_usage = tracker.get_memory_by_type(MemoryType.ACTIVATION, device_id=0)
        param_usage = tracker.get_memory_by_type(MemoryType.PARAMETER, device_id=0)
        total_act = tracker.get_memory_by_type(MemoryType.ACTIVATION)
        
        self.assertEqual(act_usage, 3072)
        self.assertEqual(param_usage, 4096)
        self.assertEqual(total_act, 3072)
        
    def test_enable_activation_checkpointing(self):
        tracker = MemoryTracker(num_devices=2, memory_per_device_gb=80.0)
        tracker.enable_activation_checkpointing(checkpointed_layers={0, 2, 4})
        
        self.assertTrue(tracker.activation_checkpointing_enabled)
        self.assertEqual(tracker.checkpointed_layers, {0, 2, 4})
        
    def test_is_layer_checkpointed(self):
        tracker = MemoryTracker(num_devices=2, memory_per_device_gb=80.0)
        self.assertFalse(tracker.is_layer_checkpointed(0))
        
        tracker.enable_activation_checkpointing(checkpointed_layers={0, 2})
        self.assertTrue(tracker.is_layer_checkpointed(0))
        self.assertFalse(tracker.is_layer_checkpointed(1))
        self.assertTrue(tracker.is_layer_checkpointed(2))
        
    def test_clear(self):
        tracker = MemoryTracker(num_devices=2, memory_per_device_gb=80.0)
        tracker.allocate(0, "t1", 1024, MemoryType.ACTIVATION)
        tracker.allocate(1, "t2", 2048, MemoryType.ACTIVATION)
        
        tracker.clear(device_id=0)
        self.assertEqual(tracker.devices[0].current_usage, 0)
        self.assertEqual(tracker.devices[1].current_usage, 2048)
        
        tracker.clear()
        self.assertEqual(tracker.devices[1].current_usage, 0)
        
    def test_get_memory_summary(self):
        tracker = MemoryTracker(num_devices=2, memory_per_device_gb=80.0)
        tracker.allocate(0, "t1", 1024 * 1024 * 1024, MemoryType.ACTIVATION)
        
        summary = tracker.get_memory_summary()
        self.assertEqual(summary["num_devices"], 2)
        self.assertEqual(summary["memory_per_device_gb"], 80.0)
        self.assertIn("devices", summary)
        self.assertIn(0, summary["devices"])


if __name__ == "__main__":
    unittest.main()
