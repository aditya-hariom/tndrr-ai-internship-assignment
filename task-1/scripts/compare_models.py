"""
scripts/compare_models.py

Aggregates baseline vs fine-tuned evaluation results:
1. Calculates absolute and relative improvements across all metrics.
2. Formats a comparison table.
3. Generates results/comparison.csv.
4. Generates an ASCII/SVG visualization artifact for reports.
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any

def generate_comparison():
    base_path = Path("results/baseline_metrics.json")
    ft_path = Path("results/finetuned_metrics.json")

    if not base_path.exists() or not ft_path.exists():
        print("[Error] Missing evaluation metric files in results/")
        return

    with open(base_path, "r", encoding="utf-8") as f:
        base = json.load(f)
    with open(ft_path, "r", encoding="utf-8") as f:
        ft = json.load(f)

    metrics_to_compare = [
        ("Raw JSON Validity Rate", "raw_json_validity_rate", True),
        ("Extracted JSON Validity Rate", "extracted_json_validity_rate", True),
        ("Intent Classification Accuracy", "intent_accuracy", True),
        ("Exact Match (EM) Accuracy", "exact_match_accuracy", True),
        ("Field-Level Micro Precision", "field_precision", True),
        ("Field-Level Micro Recall", "field_recall", True),
        ("Field-Level Micro F1 Score", "field_f1", True),
        ("Field-Level Micro Accuracy", "field_accuracy", True),
    ]

    rows = []
    print("\n" + "=" * 90)
    print(f"{'Metric':<35} | {'Base Model':<12} | {'Fine-Tuned':<12} | {'Abs Delta':<10} | {'Rel Gain':<10}")
    print("=" * 90)

    for label, key, is_pct in metrics_to_compare:
        b_val = base.get(key, 0.0)
        f_val = ft.get(key, 0.0)
        abs_delta = f_val - b_val
        rel_gain = ((f_val - b_val) / b_val * 100) if b_val > 0 else (100.0 if f_val > 0 else 0.0)

        if is_pct:
            b_str = f"{b_val * 100:.2f}%"
            f_str = f"{f_val * 100:.2f}%"
            d_str = f"{abs_delta * 100:+.2f}%"
            r_str = f"{rel_gain:+.1f}%" if b_val > 0 else "N/A (base=0)"
        else:
            b_str = f"{b_val:.4f}"
            f_str = f"{f_val:.4f}"
            d_str = f"{abs_delta:+.4f}"
            r_str = f"{rel_gain:+.1f}%"

        print(f"{label:<35} | {b_str:<12} | {f_str:<12} | {d_str:<10} | {r_str:<10}")

        rows.append({
            "metric": label,
            "baseline": b_val,
            "finetuned": f_val,
            "absolute_improvement": round(abs_delta, 4),
            "relative_improvement_pct": round(rel_gain, 2)
        })

    print("=" * 90)

    # Save to results/comparison.csv
    csv_path = Path("results/comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "baseline", "finetuned", "absolute_improvement", "relative_improvement_pct"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Comparison CSV saved to: {csv_path}")

    # Generate an SVG visualization chart for high-res rendering
    svg_path = Path("results/model_comparison_chart.svg")
    generate_svg_chart(rows, svg_path)
    print(f"Comparison chart saved to: {svg_path}")

def generate_svg_chart(data, svg_path: Path):
    """Generates a clean standalone SVG bar chart comparing Base vs Fine-Tuned."""
    width = 750
    height = 420
    margin_left = 220
    margin_right = 50
    margin_top = 50
    margin_bottom = 40
    chart_width = width - margin_left - margin_right
    bar_height = 18
    group_spacing = 42

    svg_elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="background:#0d1117; font-family:system-ui, -apple-system, sans-serif;">',
        f'<text x="{width/2}" y="28" text-anchor="middle" fill="#58a6ff" font-size="16" font-weight="600">Base Model vs LoRA Fine-Tuned (Held-Out Test Set, N=160)</text>',
        # Legend
        f'<rect x="{width - 240}" y="15" width="12" height="12" fill="#8b949e" rx="2" />',
        f'<text x="{width - 222}" y="26" fill="#c9d1d9" font-size="12">Base Model</text>',
        f'<rect x="{width - 130}" y="15" width="12" height="12" fill="#238636" rx="2" />',
        f'<text x="{width - 112}" y="26" fill="#c9d1d9" font-size="12">Fine-Tuned</text>',
    ]

    selected_metrics = [r for r in data if "Validity" not in r["metric"] or "Raw" in r["metric"]]

    for idx, r in enumerate(selected_metrics):
        y = margin_top + idx * group_spacing
        label = r["metric"].replace("Rate", "").replace("Accuracy", "Acc").replace("Score", "").strip()
        b_val = r["baseline"]
        f_val = r["finetuned"]

        b_len = chart_width * b_val
        f_len = chart_width * f_val

        # Label
        svg_elements.append(f'<text x="{margin_left - 12}" y="{y + 20}" text-anchor="end" fill="#c9d1d9" font-size="12">{label}</text>')
        # Base bar
        svg_elements.append(f'<rect x="{margin_left}" y="{y}" width="{b_len}" height="{bar_height}" fill="#8b949e" rx="3" />')
        svg_elements.append(f'<text x="{margin_left + b_len + 6}" y="{y + 14}" fill="#8b949e" font-size="11">{b_val*100:.1f}%</text>')
        # Fine-tuned bar
        svg_elements.append(f'<rect x="{margin_left}" y="{y + bar_height + 3}" width="{f_len}" height="{bar_height}" fill="#2ea043" rx="3" />')
        svg_elements.append(f'<text x="{margin_left + f_len + 6}" y="{y + bar_height + 17}" fill="#3fb950" font-size="11" font-weight="600">{f_val*100:.1f}%</text>')

    svg_elements.append('</svg>')
    svg_content = "\n".join(svg_elements)
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_content)

if __name__ == "__main__":
    generate_comparison()
