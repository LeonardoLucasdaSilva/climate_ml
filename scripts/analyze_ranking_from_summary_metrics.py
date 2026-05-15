#!/usr/bin/env python
"""
Standalone ranking analysis script that uses ONLY summary_metrics.json.

This script bypasses the problematic NPZ files and builds rankings for multiple
metrics found in summary_metrics.json across all runs.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.config.paths import PROJECT_ROOT
from src.utils.files import ensure_dir

# Configuration
METRICS = ["mae", "pearson_correlation", "nse", "kge", "dtw_distance"]
# Direction of optimality per metric
METRIC_DIRECTION = {
    "mae": "lower",
    "dtw_distance": "lower",
    "pearson_correlation": "higher",
    "nse": "higher",
    "kge": "higher",
}
INTERVALS = [
    (0.0, 10.0, "low"),
    (10.0, 30.0, "medium"),
    (30.0, None, "heavy"),
]


def process_metric(metric: str):
    """Analyze runs for a specific metric using ONLY summary_metrics.json."""
    if metric not in METRICS:
        print(f"Unknown metric '{metric}'. Skipping.")
        return

    better = METRIC_DIRECTION.get(metric, "lower")
    runs_root = PROJECT_ROOT / "runs"
    runs = sorted([p for p in runs_root.iterdir() if p.is_dir() and p.name != "summary"])

    if not runs:
        print("No runs found")
        return

    ranges = {f"{int(low)}_{int(high) if high else 'plus'}": (low, high)
              for low, high, _ in INTERVALS}

    # Track all runs' performance for this metric
    all_runs_performance = {}
    init_best = -np.inf if better == "higher" else np.inf
    cmp = (lambda x, y: x > y) if better == "higher" else (lambda x, y: x < y)
    global_best = {name: {"run_name": None, "metric_value": init_best} for name in ranges}

    print(f"\n=== Analyzing metric: {metric} ===")
    print(f"Found {len(runs)} runs. Reading summary_metrics.json...")

    for run_dir in runs:
        run_name = run_dir.name
        summary_metrics_path = run_dir / "summary_metrics.json"
        if not summary_metrics_path.exists():
            print(f"  SKIP {run_name}: No summary_metrics.json")
            continue

        try:
            with open(summary_metrics_path, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)

            # Collect metric values across stations
            values = []
            for _, station_metrics in metrics_data.items():
                if metric in station_metrics and station_metrics[metric] is not None:
                    values.append(station_metrics[metric])

            if not values:
                print(f"  SKIP {run_name}: No '{metric}' values in summary_metrics.json")
                continue

            avg_val = float(np.mean(values))
            direction_arrow = "↓" if better == "lower" else "↑"
            print(f"  OK   {run_name}: {metric} = {avg_val:.6f} ({len(values)} stations) {direction_arrow}")

            # Since we don't have range-specific values here, replicate overall across ranges
            run_range_summary = {}
            for range_name in ranges:
                run_range_summary[range_name] = {
                    "metric_mean": avg_val,
                    "n_stations": len(values),
                }

                # Update global best
                if cmp(avg_val, global_best[range_name]["metric_value"]):
                    global_best[range_name]["metric_value"] = avg_val
                    global_best[range_name]["run_name"] = run_name

            all_runs_performance[run_name] = run_range_summary

        except Exception as e:
            print(f"  ERROR {run_name}: {e}")
            continue

    if not all_runs_performance:
        print(f"No runs with '{metric}' in summary_metrics.json found")
        return

    print(f"Processed {len(all_runs_performance)} runs successfully for metric '{metric}'")

    # Create ranking data rows
    ranking_data = []
    for run_name, run_performance in all_runs_performance.items():
        row = {"run_name": run_name}
        for range_name in ranges:
            if range_name in run_performance:
                row[f"{range_name}_{metric}"] = run_performance[range_name]["metric_mean"]
                row[f"{range_name}_n_stations"] = run_performance[range_name]["n_stations"]
        ranking_data.append(row)

    # Create DataFrame and sort according to metric direction
    df_ranking = pd.DataFrame(ranking_data)
    metric_cols = [col for col in df_ranking.columns if f"_{metric}" in col and "n_stations" not in col]
    if metric_cols:
        overall_col = f"overall_{metric}"
        df_ranking[overall_col] = df_ranking[metric_cols].mean(axis=1)
        ascending = (better == "lower")
        df_ranking = df_ranking.sort_values(by=overall_col, ascending=ascending, na_position="last")

    # Save outputs
    summary_dir = runs_root / "summary"
    ensure_dir(summary_dir)

    ranking_csv = summary_dir / f"ranking_{metric}_from_summary_metrics.csv"
    df_ranking.to_csv(ranking_csv, index=False)
    print(f"✓ Saved ranking CSV to: {ranking_csv}")

    # Prepare JSON summary
    ranking_json = summary_dir / f"ranking_{metric}_from_summary_metrics.json"
    ranking_dict = {
        "metric": metric,
        "better": better,
        "data_source": "summary_metrics.json",
        "note": "Range-specific metrics use overall values (range-specific metrics are not available in summary_metrics.json).",
        "intervals": [{"low": low, "high": high, "label": label} for low, high, label in INTERVALS],
        "global_best": {
            name: {
                "best_run": info["run_name"],
                "best_metric": float(info["metric_value"]) if np.isfinite(info["metric_value"]) else None,
                "bounds": list(ranges[name]),
            }
            for name, info in global_best.items()
        },
        "all_runs": ranking_data,
    }
    with open(ranking_json, "w", encoding="utf-8") as f:
        json.dump(ranking_dict, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved ranking JSON to: {ranking_json}")

    print(f"\nTop 10 runs by {metric}:")
    overall_col = f"overall_{metric}"
    cols_to_show = ["run_name"] + ([overall_col] if overall_col in df_ranking.columns else [])
    print(df_ranking.head(10)[cols_to_show].to_string(index=False))


def analyze_runs_from_summary_metrics():
    """Analyze runs using ONLY summary_metrics.json for all configured metrics."""
    for metric in METRICS:
        process_metric(metric)
    print(f"\n{'='*70}")
    print("Analysis complete for all metrics!")
    print(f"{'='*70}")


if __name__ == "__main__":
    analyze_runs_from_summary_metrics()

