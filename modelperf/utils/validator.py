"""Validation framework for comparing simulation results with real training."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from modelperf.simulation.execution_result import ExecutionResult
from modelperf.utils.metrics import (
    calculate_mape,
    calculate_rmse,
    calculate_mae,
    calculate_r2,
)


@dataclass
class ValidationResult:
    """Result of comparing a single simulated metric against real measurement."""
    
    metric_name: str
    simulated_value: float
    real_value: float
    mape: float
    rmse: float
    mae: float
    error_margin: float
    is_within_tolerance: bool
    tolerance_percent: float = 10.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'metric_name': self.metric_name,
            'simulated_value': self.simulated_value,
            'real_value': self.real_value,
            'mape': self.mape,
            'rmse': self.rmse,
            'mae': self.mae,
            'error_margin': self.error_margin,
            'is_within_tolerance': self.is_within_tolerance,
            'tolerance_percent': self.tolerance_percent,
        }


@dataclass
class ValidationReport:
    """Complete validation report with all metrics and summary."""
    
    timestamp: str
    simulation_config: Dict[str, Any]
    real_config: Dict[str, Any]
    results: List[ValidationResult] = field(default_factory=list)
    overall_score: float = 0.0
    pass_fail: str = "UNKNOWN"
    
    def get_result_by_metric(self, metric_name: str) -> Optional[ValidationResult]:
        for result in self.results:
            if result.metric_name == metric_name:
                return result
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'simulation_config': self.simulation_config,
            'real_config': self.real_config,
            'results': [r.to_dict() for r in self.results],
            'overall_score': self.overall_score,
            'pass_fail': self.pass_fail,
        }


class Validator:
    """Validator for comparing simulation results against real training measurements."""
    
    def __init__(self, tolerance_percent: float = 10.0):
        self.tolerance_percent = tolerance_percent
    
    def _compare_single_metric(
        self,
        metric_name: str,
        simulated: float,
        real: float,
    ) -> ValidationResult:
        mape = calculate_mape(simulated, real)
        rmse = calculate_rmse(simulated, real)
        mae = calculate_mae(simulated, real)
        
        error_margin = abs(simulated - real)
        is_within_tolerance = mape <= self.tolerance_percent
        
        return ValidationResult(
            metric_name=metric_name,
            simulated_value=simulated,
            real_value=real,
            mape=mape,
            rmse=rmse,
            mae=mae,
            error_margin=error_margin,
            is_within_tolerance=is_within_tolerance,
            tolerance_percent=self.tolerance_percent,
        )
    
    def compare_iteration_time(
        self,
        sim_result: ExecutionResult,
        real_time_ms: float,
    ) -> ValidationResult:
        return self._compare_single_metric(
            'iteration_time_ms',
            sim_result.iteration_time_ms,
            real_time_ms,
        )
    
    def compare_memory_peak(
        self,
        sim_result: ExecutionResult,
        real_memory_mb: float,
    ) -> ValidationResult:
        return self._compare_single_metric(
            'peak_memory_mb',
            sim_result.peak_memory_mb,
            real_memory_mb,
        )
    
    def compare_communication_time(
        self,
        sim_result: ExecutionResult,
        real_comm_times_dict: Dict[str, float],
    ) -> Dict[str, ValidationResult]:
        results = {}
        for comm_type, real_time in real_comm_times_dict.items():
            sim_time = sim_result.comm_breakdown.get(comm_type, 0.0)
            results[comm_type] = self._compare_single_metric(
                f'comm_time_{comm_type}',
                sim_time,
                real_time,
            )
        return results
    
    def validate_full(
        self,
        sim_result: ExecutionResult,
        real_metrics_dict: Dict[str, float],
        simulation_config: Optional[Dict] = None,
        real_config: Optional[Dict] = None,
    ) -> ValidationReport:
        timestamp = datetime.now().isoformat()
        
        results = []
        
        if 'iteration_time_ms' in real_metrics_dict:
            results.append(self.compare_iteration_time(
                sim_result,
                real_metrics_dict['iteration_time_ms'],
            ))
        
        if 'peak_memory_mb' in real_metrics_dict:
            results.append(self.compare_memory_peak(
                sim_result,
                real_metrics_dict['peak_memory_mb'],
            ))
        
        if 'compute_time_ms' in real_metrics_dict:
            results.append(self._compare_single_metric(
                'compute_time_ms',
                sim_result.compute_time_ms,
                real_metrics_dict['compute_time_ms'],
            ))
        
        if 'comm_time_ms' in real_metrics_dict:
            results.append(self._compare_single_metric(
                'comm_time_ms',
                sim_result.comm_time_ms,
                real_metrics_dict['comm_time_ms'],
            ))
        
        for key in real_metrics_dict:
            if key.startswith('comm_breakdown_'):
                comm_type = key.replace('comm_breakdown_', '')
                sim_val = sim_result.comm_breakdown.get(comm_type, 0.0)
                results.append(self._compare_single_metric(
                    key,
                    sim_val,
                    real_metrics_dict[key],
                ))
        
        overall_score = 0.0
        if results:
            overall_score = sum(r.mape for r in results) / len(results)
        
        pass_fail = "PASS" if all(r.is_within_tolerance for r in results) else "FAIL"
        
        return ValidationReport(
            timestamp=timestamp,
            simulation_config=simulation_config or {},
            real_config=real_config or {},
            results=results,
            overall_score=overall_score,
            pass_fail=pass_fail,
        )
    
    def generate_report(
        self,
        validation_results: Union[ValidationReport, List[ValidationResult]],
        format: str = "markdown",
    ) -> str:
        if isinstance(validation_results, ValidationReport):
            results = validation_results.results
            report_meta = validation_results
        else:
            results = validation_results
            report_meta = None
        
        if format.lower() == "markdown":
            return self._generate_markdown_report(results, report_meta)
        elif format.lower() == "json":
            import json
            if report_meta:
                return json.dumps(report_meta.to_dict(), indent=2)
            else:
                return json.dumps([r.to_dict() for r in results], indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_markdown_report(
        self,
        results: List[ValidationResult],
        report_meta: Optional[ValidationReport] = None,
    ) -> str:
        lines = []
        
        lines.append("# ModelPerf Validation Report")
        lines.append("")
        
        if report_meta:
            lines.append(f"**Timestamp:** {report_meta.timestamp}")
            lines.append(f"**Overall Score:** {report_meta.overall_score:.2f}%")
            lines.append(f"**Status:** {report_meta.pass_fail}")
            lines.append("")
        
        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Simulated | Real | MAPE | RMSE | MAE | Status |")
        lines.append("|--------|-----------|------|------|------|-----|--------|")
        
        for r in results:
            status = "✅ PASS" if r.is_within_tolerance else "❌ FAIL"
            lines.append(
                f"| {r.metric_name} | {r.simulated_value:.2f} | "
                f"{r.real_value:.2f} | {r.mape:.2f}% | {r.rmse:.4f} | "
                f"{r.mae:.4f} | {status} |"
            )
        
        lines.append("")
        lines.append("## Detailed Analysis")
        lines.append("")
        
        for r in results:
            lines.append(f"### {r.metric_name}")
            lines.append("")
            lines.append(f"- **Simulated:** {r.simulated_value:.4f}")
            lines.append(f"- **Real:** {r.real_value:.4f}")
            lines.append(f"- **Error Margin:** {r.error_margin:.4f}")
            lines.append(f"- **MAPE:** {r.mape:.2f}%")
            lines.append(f"- **RMSE:** {r.rmse:.4f}")
            lines.append(f"- **MAE:** {r.mae:.4f}")
            lines.append(f"- **Tolerance:** {r.tolerance_percent:.1f}%")
            lines.append(f"- **Status:** {'PASS' if r.is_within_tolerance else 'FAIL'}")
            lines.append("")
        
        return "\n".join(lines)


__all__ = [
    'ValidationResult',
    'ValidationReport',
    'Validator',
]