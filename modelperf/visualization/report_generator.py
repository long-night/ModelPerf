"""Performance report generator for ModelPerf."""

import math
import os
from typing import Dict, List, Tuple

from modelperf.simulation.execution_result import ExecutionResult

_HAS_MPL = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    pass


def _ensure_dir(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)


def _default_iteration_breakdown(result: ExecutionResult) -> Dict[str, float]:
    breakdown = {
        "forward": result.forward_time_ms,
        "backward": result.backward_time_ms,
        "optimizer": result.optimizer_time_ms,
        "comm": result.comm_time_ms,
        "bubble": result.bubble_time_ms,
    }
    if all(v == 0.0 for v in breakdown.values()):
        breakdown["total"] = result.iteration_time_ms
    return breakdown


def _default_memory_breakdown(result: ExecutionResult) -> Dict[str, float]:
    if result.memory_breakdown:
        return dict(result.memory_breakdown)
    if result.peak_memory_mb > 0:
        return {"peak_memory": result.peak_memory_mb}
    return {"unknown": 0.0}


def _default_compute_comm_breakdown(
    result: ExecutionResult,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    compute = dict(result.compute_breakdown) if result.compute_breakdown else {}
    comm = dict(result.comm_breakdown) if result.comm_breakdown else {}
    if not compute:
        compute = {"total_compute": result.compute_time_ms}
    if not comm:
        comm = {"total_comm": result.comm_time_ms}
    return compute, comm


# --- Matplotlib renderers ---


def _mpl_iteration_time_breakdown(result: ExecutionResult, output_path: str) -> None:
    breakdown = _default_iteration_breakdown(result)
    labels = list(breakdown.keys())
    values = list(breakdown.values())
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336", "#607D8B"]
    fig, ax = plt.subplots(figsize=(8, 4))
    left = 0.0
    for i, (label, value) in enumerate(zip(labels, values)):
        ax.barh(["Iteration"], [value], left=[left], color=colors[i % len(colors)], label=label)
        left += value
    ax.set_xlabel("Time (ms)")
    ax.set_title("Iteration Time Breakdown")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _mpl_memory_breakdown(result: ExecutionResult, output_path: str) -> None:
    breakdown = _default_memory_breakdown(result)
    labels = list(breakdown.keys())
    values = list(breakdown.values())
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336", "#607D8B"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=[colors[i % len(colors)] for i in range(len(labels))],
    )
    ax.set_title("Memory Breakdown")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _mpl_compute_comm_breakdown(result: ExecutionResult, output_path: str) -> None:
    compute, comm = _default_compute_comm_breakdown(result)
    all_keys = sorted(set(compute.keys()) | set(comm.keys()))
    compute_vals = [compute.get(k, 0.0) for k in all_keys]
    comm_vals = [comm.get(k, 0.0) for k in all_keys]
    x = range(len(all_keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width / 2 for i in x], compute_vals, width, label="Compute", color="#2196F3")
    ax.bar([i + width / 2 for i in x], comm_vals, width, label="Comm", color="#F44336")
    ax.set_xticks(x)
    ax.set_xticklabels(all_keys, rotation=45, ha="right")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Compute vs Communication Breakdown")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# --- HTML fallback renderers ---


def _html_iteration_time_breakdown(result: ExecutionResult, output_path: str) -> None:
    breakdown = _default_iteration_breakdown(result)
    total = sum(breakdown.values()) or 1.0
    colors = {
        "forward": "#4CAF50",
        "backward": "#2196F3",
        "optimizer": "#FF9800",
        "comm": "#9C27B0",
        "bubble": "#F44336",
        "total": "#607D8B",
    }
    bars = ""
    legend = ""
    for label, value in breakdown.items():
        pct = (value / total) * 100
        color = colors.get(label, "#607D8B")
        bars += '<div style="width:%.2f%%;background:%s;height:100%%;"></div>' % (pct, color)
        legend += '<span style="margin-right:12px;font-size:12px;">'
        legend += '<span style="display:inline-block;width:10px;height:10px;background:%s;' % color
        legend += 'margin-right:4px;"></span>%s: %.2f ms</span>' % (label, value)
    html = (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
        + '<title>Iteration Time Breakdown</title>\n<style>\n'
        + 'body { font-family: Arial, sans-serif; margin: 20px; }\n'
        + '.container { max-width: 800px; margin: auto; }\n'
        + '.chart { display: flex; height: 40px; border: 1px solid #ccc; border-radius: 4px; overflow: hidden; }\n'
        + '.legend { margin-top: 12px; }\n'
        + '</style>\n</head>\n<body>\n<div class="container">\n'
        + '<h2>Iteration Time Breakdown</h2>\n'
        + '<div class="chart">' + bars + '</div>\n'
        + '<div class="legend">' + legend + '</div>\n'
        + '</div>\n</body>\n</html>'
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def _html_memory_breakdown(result: ExecutionResult, output_path: str) -> None:
    breakdown = _default_memory_breakdown(result)
    labels = list(breakdown.keys())
    values = list(breakdown.values())
    total = sum(values) or 1.0
    palette = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336", "#607D8B", "#795548"]
    cx, cy, r = 100, 100, 80
    start_angle = 0.0
    slices: List[str] = []
    legend_items: List[str] = []
    for i, (label, value) in enumerate(zip(labels, values)):
        angle = (value / total) * 360.0
        end_angle = start_angle + angle
        x1 = cx + r * math.cos(math.radians(start_angle))
        y1 = cy + r * math.sin(math.radians(start_angle))
        x2 = cx + r * math.cos(math.radians(end_angle))
        y2 = cy + r * math.sin(math.radians(end_angle))
        large_arc = 1 if angle > 180 else 0
        path = "M %d %d L %.2f %.2f A %d %d 0 %d 1 %.2f %.2f Z" % (cx, cy, x1, y1, r, r, large_arc, x2, y2)
        color = palette[i % len(palette)]
        slices.append('<path d="%s" fill="%s" stroke="#fff" stroke-width="1"/>' % (path, color))
        pct = (value / total) * 100
        legend_items.append(
            '<div style="margin:4px 0;font-size:12px;">'
            + '<span style="display:inline-block;width:10px;height:10px;background:%s;' % color
            + 'margin-right:6px;"></span>%s: %.2f MB (%.1f%%)</div>' % (label, value, pct)
        )
        start_angle = end_angle
    svg = '<svg width="220" height="220" viewBox="0 0 220 220">' + ''.join(slices) + '</svg>'
    html = (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
        + '<title>Memory Breakdown</title>\n<style>\n'
        + 'body { font-family: Arial, sans-serif; margin: 20px; }\n'
        + '.container { max-width: 600px; margin: auto; display: flex; align-items: center; gap: 20px; }\n'
        + '</style>\n</head>\n<body>\n<div class="container">\n'
        + '<div>' + svg + '</div>\n<div>\n<h2>Memory Breakdown</h2>\n'
        + ''.join(legend_items) + '\n</div>\n</div>\n</body>\n</html>'
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def _html_compute_comm_breakdown(result: ExecutionResult, output_path: str) -> None:
    compute, comm = _default_compute_comm_breakdown(result)
    all_keys = sorted(set(compute.keys()) | set(comm.keys()))
    compute_vals = [compute.get(k, 0.0) for k in all_keys]
    comm_vals = [comm.get(k, 0.0) for k in all_keys]
    max_val = max(max(compute_vals, default=0.0), max(comm_vals, default=0.0)) or 1.0
    rows = ""
    for key, c_val, m_val in zip(all_keys, compute_vals, comm_vals):
        c_pct = (c_val / max_val) * 100
        m_pct = (m_val / max_val) * 100
        rows += '<tr><td style="padding:4px 8px;font-size:12px;">' + key + '</td>'
        rows += '<td style="padding:4px 8px;"><div style="width:%.1f%%;background:#2196F3;height:16px;"></div></td>' % c_pct
        rows += '<td style="padding:4px 8px;font-size:11px;color:#555;">%.2f</td>' % c_val
        rows += '<td style="padding:4px 8px;"><div style="width:%.1f%%;background:#F44336;height:16px;"></div></td>' % m_pct
        rows += '<td style="padding:4px 8px;font-size:11px;color:#555;">%.2f</td></tr>' % m_val
    html = (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
        + '<title>Compute vs Communication Breakdown</title>\n<style>\n'
        + 'body { font-family: Arial, sans-serif; margin: 20px; }\n'
        + '.container { max-width: 900px; margin: auto; }\n'
        + 'table { border-collapse: collapse; width: 100%; }\n'
        + 'th { text-align: left; font-size: 12px; padding: 4px 8px; background: #f5f5f5; }\n'
        + '</style>\n</head>\n<body>\n<div class="container">\n'
        + '<h2>Compute vs Communication Breakdown</h2>\n<table>\n'
        + '<tr>\n<th>Node</th>\n<th colspan="2">Compute (ms)</th>\n<th colspan="2">Comm (ms)</th>\n</tr>\n'
        + rows
        + '\n</table>\n</div>\n</body>\n</html>'
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def _generate_report_html(
    result: ExecutionResult,
    output_dir: str,
    chart_files: List[str],
) -> str:
    summary_rows = ""
    summary = {
        "Iteration Time (ms)": "%0.2f" % result.iteration_time_ms,
        "Forward Time (ms)": "%0.2f" % result.forward_time_ms,
        "Backward Time (ms)": "%0.2f" % result.backward_time_ms,
        "Optimizer Time (ms)": "%0.2f" % result.optimizer_time_ms,
        "Compute Time (ms)": "%0.2f" % result.compute_time_ms,
        "Comm Time (ms)": "%0.2f" % result.comm_time_ms,
        "Bubble Time (ms)": "%0.2f" % result.bubble_time_ms,
        "Peak Memory (MB)": "%0.2f" % result.peak_memory_mb,
        "Bottleneck": result.bottleneck,
        "Throughput (tok/s)": "%0.2f" % result.throughput_tokens_per_sec,
        "Memory Efficiency (%)": "%0.1f" % (result.memory_efficiency * 100),
        "Pipeline Stages": str(result.pipeline_stages),
        "Num Microbatches": str(result.num_microbatches),
    }
    for key, value in summary.items():
        summary_rows += "<tr><td>%s</td><td>%s</td></tr>\n" % (key, value)

    chart_embeds = ""
    for cf in chart_files:
        basename = os.path.basename(cf)
        if basename.endswith(".html"):
            try:
                with open(cf, "r", encoding="utf-8") as f:
                    content = f.read()
                start = content.find("<body>")
                end = content.find("</body>")
                if start != -1 and end != -1:
                    body = content[start + 6 : end]
                else:
                    body = content
                chart_embeds += (
                    '<div style="margin: 24px 0; padding: 16px; '
                    'border: 1px solid #ddd; border-radius: 8px;">\n'
                    + body + "\n</div>\n"
                )
            except Exception:
                chart_embeds += "<p>Could not embed %s</p>\n" % basename
        else:
            chart_embeds += '<img src="%s" style="max-width:100%%;margin:16px 0;">\n' % basename

    html = (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
        + '<title>ModelPerf Performance Report</title>\n<style>\n'
        + 'body { font-family: "Segoe UI", Arial, sans-serif; margin: 0; padding: 0; background: #f8f9fa; color: #333; }\n'
        + '.container { max-width: 960px; margin: 32px auto; background: #fff; padding: 32px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }\n'
        + 'h1 { font-size: 24px; margin-bottom: 16px; }\n'
        + 'h2 { font-size: 18px; margin-top: 24px; margin-bottom: 12px; }\n'
        + 'table { width: 100%%; border-collapse: collapse; margin-top: 12px; }\n'
        + 'th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid #eee; font-size: 14px; }\n'
        + 'th { background: #fafafa; font-weight: 600; }\n'
        + '</style>\n</head>\n<body>\n<div class="container">\n'
        + '<h1>ModelPerf Performance Report</h1>\n'
        + '<h2>Summary Statistics</h2>\n<table>\n'
        + '<tr><th>Metric</th><th>Value</th></tr>\n'
        + summary_rows
        + '</table>\n'
        + '<h2>Charts</h2>\n'
        + chart_embeds
        + '</div>\n</body>\n</html>'
    )
    report_path = os.path.join(output_dir, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return report_path


def generate_performance_report(result: ExecutionResult, output_dir: str) -> str:
    """Generate performance comparison charts and an HTML summary report.

    Args:
        result: ExecutionResult from the virtual executor.
        output_dir: Directory where output files will be written.

    Returns:
        Absolute path to the generated report.html.
    """
    _ensure_dir(output_dir)

    if _HAS_MPL:
        it_path = os.path.join(output_dir, "iteration_time_breakdown.png")
        mem_path = os.path.join(output_dir, "memory_breakdown.png")
        cc_path = os.path.join(output_dir, "compute_comm_breakdown.png")
        _mpl_iteration_time_breakdown(result, it_path)
        _mpl_memory_breakdown(result, mem_path)
        _mpl_compute_comm_breakdown(result, cc_path)
    else:
        it_path = os.path.join(output_dir, "iteration_time_breakdown.html")
        mem_path = os.path.join(output_dir, "memory_breakdown.html")
        cc_path = os.path.join(output_dir, "compute_comm_breakdown.html")
        _html_iteration_time_breakdown(result, it_path)
        _html_memory_breakdown(result, mem_path)
        _html_compute_comm_breakdown(result, cc_path)

    chart_files = [it_path, mem_path, cc_path]
    report_path = _generate_report_html(result, output_dir, chart_files)
    return os.path.abspath(report_path)


__all__ = ["generate_performance_report"]



