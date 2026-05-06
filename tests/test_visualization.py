import os
import tempfile
import unittest

from modelperf.simulation.execution_result import ExecutionResult
from modelperf.visualization.report_generator import generate_performance_report


class TestReportGenerator(unittest.TestCase):
    def test_generate_report_creates_files(self):
        result = ExecutionResult(
            iteration_time_ms=100.0,
            compute_time_ms=60.0,
            comm_time_ms=20.0,
            peak_memory_mb=1024.0,
            bottleneck="compute",
            forward_time_ms=30.0,
            backward_time_ms=25.0,
            optimizer_time_ms=5.0,
            compute_breakdown={"matmul": 40.0, "attention": 20.0},
            comm_breakdown={"all_reduce": 15.0, "all_gather": 5.0},
            memory_breakdown={"activation": 512.0, "parameter": 256.0, "gradient": 256.0},
            bubble_time_ms=10.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = generate_performance_report(result, tmpdir)
            self.assertTrue(os.path.isfile(report_path))
            self.assertTrue(report_path.endswith("report.html"))

            files = os.listdir(tmpdir)
            self.assertIn("report.html", files)

    def test_generate_report_with_empty_breakdowns(self):
        result = ExecutionResult(
            iteration_time_ms=50.0,
            compute_time_ms=30.0,
            comm_time_ms=10.0,
            peak_memory_mb=512.0,
            bottleneck="comm",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = generate_performance_report(result, tmpdir)
            self.assertTrue(os.path.isfile(report_path))

    def test_generate_report_zero_values(self):
        result = ExecutionResult(
            iteration_time_ms=0.0,
            compute_time_ms=0.0,
            comm_time_ms=0.0,
            peak_memory_mb=0.0,
            bottleneck="none",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = generate_performance_report(result, tmpdir)
            self.assertTrue(os.path.isfile(report_path))

    def test_html_files_are_readable(self):
        result = ExecutionResult(
            iteration_time_ms=120.0,
            compute_time_ms=70.0,
            comm_time_ms=30.0,
            peak_memory_mb=2048.0,
            bottleneck="memory",
            forward_time_ms=35.0,
            backward_time_ms=30.0,
            optimizer_time_ms=5.0,
            memory_breakdown={"activation": 1024.0, "parameter": 512.0, "gradient": 512.0},
            bubble_time_ms=10.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_performance_report(result, tmpdir)
            report_path = os.path.join(tmpdir, "report.html")
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("ModelPerf Performance Report", content)
            self.assertIn("Summary Statistics", content)


if __name__ == "__main__":
    unittest.main()
