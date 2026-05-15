"""
Post-processing: filter best experiment results from GLOBAL_DEBUG_ERA5.

Reads test predictions saved as NPZ files, evaluates each configuration
against segment-wise metrics, and generates filtered timeseries plots
(showing only the samples that fall inside the metric's precipitation range)
in two versions — dashed prediction and solid prediction.

Usage
-----
    python scripts/filter_best_results.py
    python scripts/filter_best_results.py --debug_dir runs/GLOBAL_DEBUG_ERA5
"""

import argparse
import csv
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.config.paths import PROJECT_ROOT

# Matches two ISO dates joined by a hyphen, e.g. "2017-01-01-2020-12-31"
_DATE_RANGE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})-(\d{4}-\d{2}-\d{2})")


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------
METRICS = {
    "acc_0_2mm": {
        "description": "0–2 mm real: ≥ 80% das previsões também em 0–2 mm",
        "low": 0.0,
        "high": 2.0,
        "threshold": 0.80,
        "kind": "accuracy",
    },
    "mae_0_10mm": {
        "description": "0–10 mm real: MAE ≤ 2 mm",
        "low": 0.0,
        "high": 10.0,
        "threshold": 2.0,
        "kind": "mae",
    },
    "mae_10_30mm": {
        "description": "10–30 mm real: MAE ≤ 5 mm",
        "low": 10.0,
        "high": 30.0,
        "threshold": 5.0,
        "kind": "mae",
    },
    "mae_30plus_mm": {
        "description": "30+ mm real: MAE ≤ 40 mm",
        "low": 30.0,
        "high": None,
        "threshold": 25.0,
        "kind": "mae",
    },
    "kge_0_10mm": {
        "description": "0–10 mm real: KGE > 0.4",
        "low": 0.0,
        "high": 10.0,
        "threshold": 0.4,
        "kind": "kge",
    },
    "kge_10_30mm": {
        "description": "10–30 mm real: KGE > 0.4",
        "low": 10.0,
        "high": 30.0,
        "threshold": 0.4,
        "kind": "kge",
    },
    "kge_30plus_mm": {
        "description": "30+ mm real: KGE > 0.4",
        "low": 30.0,
        "high": None,
        "threshold": 0.4,
        "kind": "kge",
    },
}


# ---------------------------------------------------------------------------
# Helpers — metrics
# ---------------------------------------------------------------------------

def _segment_mask(y: np.ndarray, low: float, high: float | None) -> np.ndarray:
    if high is None:
        return y >= low
    return (y >= low) & (y <= high)


def _kge(yt: np.ndarray, yp: np.ndarray) -> float:
    """Kling-Gupta Efficiency. Returns -inf if inputs are degenerate."""
    if yt.std() == 0 or yt.mean() == 0 or len(yt) < 2:
        return float("-inf")
    r_num = np.sum((yt - yt.mean()) * (yp - yp.mean()))
    r_den = np.sqrt(np.sum((yt - yt.mean()) ** 2) * np.sum((yp - yp.mean()) ** 2))
    r = r_num / r_den if r_den != 0 else 0.0
    alpha = yp.std() / yt.std()
    beta = yp.mean() / yt.mean()
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    """Return metric value for each segment, or None if the segment is empty."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    results = {}
    for name, cfg in METRICS.items():
        mask = _segment_mask(y_true, cfg["low"], cfg["high"])
        if mask.sum() == 0:
            results[name] = None
            continue

        yt, yp = y_true[mask], y_pred[mask]

        if cfg["kind"] == "accuracy":
            results[name] = float(_segment_mask(yp, cfg["low"], cfg["high"]).mean())
        elif cfg["kind"] == "mae":
            results[name] = float(np.mean(np.abs(yt - yp)))
        elif cfg["kind"] == "kge":
            results[name] = _kge(yt, yp)

    return results


def passes(metric_name: str, value: float | None) -> bool:
    if value is None:
        return False
    cfg = METRICS[metric_name]
    if cfg["kind"] in ("accuracy", "kge"):
        return value >= cfg["threshold"]
    return value <= cfg["threshold"]


def _format_value(metric_name: str, value: float) -> str:
    kind = METRICS[metric_name]["kind"]
    if kind == "accuracy":
        return f"{value * 100:.1f}%"
    if kind == "kge":
        return f"KGE={value:.3f}"
    return f"MAE={value:.3f} mm"


# ---------------------------------------------------------------------------
# Helpers — plotting
# ---------------------------------------------------------------------------

def _extract_period(file_prefix: str) -> str:
    """Return 'YYYY-MM-DD_YYYY-MM-DD' extracted from the file prefix, or 'unknown'."""
    m = _DATE_RANGE_RE.search(file_prefix)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return "unknown_period"


def _range_label(cfg: dict) -> str:
    if cfg["high"] is None:
        return f"≥ {cfg['low']:.0f} mm"
    return f"{cfg['low']:.0f}–{cfg['high']:.0f} mm"


def _save_segment_plots(
    yt_seg: np.ndarray,
    yp_seg: np.ndarray,
    metric_name: str,
    file_prefix: str,
    metric_value: float,
    dest_dir: Path,
) -> None:
    """Generate dashed and solid timeseries plots for the filtered segment."""
    cfg = METRICS[metric_name]
    val_str = _format_value(metric_name, metric_value)
    title = (
        f"{file_prefix}\n"
        f"Segment: {_range_label(cfg)} | {cfg['description'].split(':')[0]} | {val_str}"
    )
    x = np.arange(len(yt_seg))

    for linestyle, suffix in [("--", "dashed"), ("-", "solid")]:
        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(x, yt_seg, label="Real", linewidth=2, color="steelblue")
        ax.plot(
            x, yp_seg,
            label="Predicted",
            linewidth=2,
            linestyle=linestyle,
            color="tomato",
        )

        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xlabel("Sample index (filtered segment)")
        ax.set_ylabel("Precipitation (mm)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_path = dest_dir / f"{file_prefix}_{metric_name}_{suffix}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Rankings
# ---------------------------------------------------------------------------

TOP_N = 25


def _save_rankings(
    all_results: dict[str, list[tuple[str, float]]],
    best_dir: Path,
) -> None:
    """Write one CSV per metric with the top-25 configurations.

    Columns: rank, period, config, value, passes_threshold
    Also writes best_results/ranking_summary.csv combining all metrics.
    """
    summary_rows: list[dict] = []

    for metric_name, entries in all_results.items():
        if not entries:
            continue

        cfg = METRICS[metric_name]
        higher_is_better = cfg["kind"] in ("accuracy", "kge")
        sorted_entries = sorted(
            entries,
            key=lambda x: -x[1] if higher_is_better else x[1],
        )
        top = sorted_entries[:TOP_N]

        csv_path = best_dir / f"ranking_{metric_name}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "rank",
                "period",
                "config",
                f"value ({cfg['kind']})",
                "passes_threshold",
            ])
            for rank, (prefix, val) in enumerate(top, start=1):
                writer.writerow([
                    rank,
                    _extract_period(prefix),
                    prefix,
                    f"{val:.6f}",
                    "yes" if passes(metric_name, val) else "no",
                ])
                summary_rows.append({
                    "metric": metric_name,
                    "rank": rank,
                    "period": _extract_period(prefix),
                    "config": prefix,
                    "value": f"{val:.6f}",
                    "passes_threshold": "yes" if passes(metric_name, val) else "no",
                })

    # Global summary CSV
    if summary_rows:
        summary_path = best_dir / "ranking_summary.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["metric", "rank", "period", "config", "value", "passes_threshold"],
            )
            writer.writeheader()
            writer.writerows(summary_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(global_debug_dir: Path) -> None:
    preds_dir = global_debug_dir / "predictions"
    best_dir = global_debug_dir / "best_results"

    if not preds_dir.exists():
        print(f"[ERROR] Predictions directory not found: {preds_dir}")
        print("        Run the sweep first so that NPZ files are saved there.")
        return

    best_dir.mkdir(parents=True, exist_ok=True)
    for metric_name in METRICS:
        (best_dir / metric_name).mkdir(exist_ok=True)

    npz_files = sorted(preds_dir.glob("*_test_predictions.npz"))
    if not npz_files:
        print(f"[WARN] No NPZ files found in {preds_dir}")
        return

    print(f"Evaluating {len(npz_files)} configurations...\n")

    passing: dict[str, list[tuple[str, float]]] = {m: [] for m in METRICS}
    all_results: dict[str, list[tuple[str, float]]] = {m: [] for m in METRICS}

    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        y_true = data["y_true"].flatten()
        y_pred = data["y_pred"].flatten()

        file_prefix = npz_path.stem.replace("_test_predictions", "")
        metrics = evaluate(y_true, y_pred)

        period = _extract_period(file_prefix)

        for metric_name, value in metrics.items():
            if value is not None:
                all_results[metric_name].append((file_prefix, value))

            if not passes(metric_name, value):
                continue

            passing[metric_name].append((file_prefix, value))

            # Destination: best_results/<metric>/<period>/
            dest_dir = best_dir / metric_name / period
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Filter arrays to the metric's precipitation segment
            cfg = METRICS[metric_name]
            mask = _segment_mask(y_true, cfg["low"], cfg["high"])
            yt_seg = y_true[mask]
            yp_seg = y_pred[mask]

            _save_segment_plots(
                yt_seg, yp_seg,
                metric_name, file_prefix, value,
                dest_dir,
            )

    # ------------------------------------------------------------------
    # Rankings (CSV files)
    # ------------------------------------------------------------------
    _save_rankings(all_results, best_dir)

    # ------------------------------------------------------------------
    # Terminal summary
    # ------------------------------------------------------------------
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    total_unique: set[str] = set()
    for metric_name, entries in passing.items():
        cfg = METRICS[metric_name]
        higher_is_better = cfg["kind"] in ("accuracy", "kge")
        sorted_entries = sorted(entries, key=lambda x: -x[1] if higher_is_better else x[1])

        print(f"\n[{metric_name}]  {cfg['description']}")
        print(f"  Threshold : {cfg['threshold']}")
        print(f"  Passing   : {len(entries)} / {len(all_results[metric_name])} evaluated")
        for rank, (prefix, val) in enumerate(sorted_entries[:TOP_N], start=1):
            print(f"    #{rank:02d}  {_format_value(metric_name, val)}  {prefix}")
        if len(entries) > TOP_N:
            print(f"    ... and {len(entries) - TOP_N} more")

        total_unique.update(p for p, _ in entries)

    print(f"\n{'='*70}")
    print(f"Unique configs passing at least one metric : {len(total_unique)}")
    print(f"Plots saved to                             : {best_dir}")
    print(f"Ranking CSVs saved to                      : {best_dir}/ranking_<metric>.csv")
    print(f"Global summary                             : {best_dir}/ranking_summary.csv")
    print("Each passing config generates two plots per metric: _dashed.png / _solid.png")


def main():
    parser = argparse.ArgumentParser(
        description="Filter best results and generate segment-filtered plots"
    )
    parser.add_argument(
        "--debug_dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "GLOBAL_DEBUG_ERA5",
        help="Path to the GLOBAL_DEBUG_* directory (default: runs/GLOBAL_DEBUG_ERA5)",
    )
    args = parser.parse_args()
    run(args.debug_dir)


if __name__ == "__main__":
    main()
