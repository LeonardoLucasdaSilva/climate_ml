import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config.paths import PROJECT_ROOT
from src.utils.files import ensure_dir
from src.evaluation.metrics import (
    mae,
    pearson_correlation,
    nse,
    kge,
    dtw_distance,
)

# ============================================================
# CONFIGURATION - Métricas e intervalos
# ============================================================

# Métricas a varrer (todas precisam existir em summary_metrics.json
# ou serão computadas dos NPZ conforme o recorte por faixa)
METRICS = ["mae", "pearson_correlation", "nse", "kge", "dtw_distance"]
# Direção ótima por métrica (para ordenação/ranking)
# "lower" => menor é melhor | "higher" => maior é melhor
METRIC_DIRECTION = {
    "mae": "lower",
    "dtw_distance": "lower",
    "pearson_correlation": "higher",
    "nse": "higher",
    "kge": "higher",
}

# Intervalos de precipitação: (low, high, label) (high=None => aberto)
INTERVALS = [
    (0.0, 10.0, "low"),       # 0-10mm
    (10.0, 30.0, "medium"),   # 10-30mm
    (30.0, None, "heavy"),    # 30mm+
]

# Shift alignment: Compare y_true[i] with y_pred[i+1] to account for 1-day shift
ALIGN_SHIFT = False  # Set to False to use y_true[i] vs y_pred[i] (original alignment)

# ============================================================


def align_predictions_for_shift(y_true, y_pred):
    """Align y_true and y_pred to account for 1-day prediction shift.

    Compares y_true[i] with y_pred[i+1], suppressing the extra prediction
    to maintain consistent array lengths.

    Parameters
    ----------
    y_true : array-like
        Ground truth values
    y_pred : array-like
        Predicted values (may be shifted by 1 day)

    Returns
    -------
    tuple of (y_true_aligned, y_pred_aligned)
        Both with same length, aligned for 1-day shift comparison
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if ALIGN_SHIFT:
        # Compare y_true[i] with y_pred[i+1]
        # Suppress the last element of y_true to match y_pred[1:]
        min_len = min(len(y_true) - 1, len(y_pred) - 1)
        if min_len <= 0:
            return y_true, y_pred
        return y_true[:min_len], y_pred[1:min_len + 1]
    else:
        # Original alignment: y_true[i] vs y_pred[i]
        min_len = min(len(y_true), len(y_pred))
        return y_true[:min_len], y_pred[:min_len]


def load_run_metrics(run_dir: Path):
    """Carrega métricas por estação do summary_metrics.json, se existir.

    Retorna dict estação -> dict de métricas.
    """
    summary_path = run_dir / "summary_metrics.json"
    if not summary_path.exists():
        return {}

    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_station_predictions(run_dir: Path, station: str):
    """Carrega predições de teste salvas por estação em um run.

    Espera arquivos salvos por save_station_artifacts em:
    <run_dir>/<station>/predictions/<station>_<run_name>_test_predictions.npz.
    """
    station_dir = run_dir / station
    preds_dir = station_dir / "predictions"
    if not preds_dir.exists():
        return None

    exact_path = preds_dir / f"{station}_{run_dir.name}_test_predictions.npz"
    if not exact_path.exists():
        return None

    data = np.load(exact_path)
    return {
        "y_true": data["y_true"],
        "y_pred": data["y_pred"],
        "source": str(data["source"]) if "source" in data.files else "era5",
    }


def load_run_level_predictions(run_dir: Path):
    """Carrega predições de teste no nível do run, ao lado do summary_metrics.json.

    Espera um arquivo `<run_name>_test_predictions.npz` onde `run_name`
    é o nome do diretório do run.
    """
    run_name = run_dir.name
    npz_path = run_dir / f"{run_name}_test_predictions.npz"

    if not npz_path.exists():
        return None

    data = np.load(npz_path)
    return {
        "y_true": data["y_true"],
        "y_pred": data["y_pred"],
        "source": str(data["source"]) if "source" in data.files else "era5",
    }


def compute_range_errors(y_true, y_pred, bounds, metric="mae"):
    """Computa uma métrica restrita às amostras dentro de uma faixa de precipitação.

    Parameters
    ----------
    y_true : array-like
        Valores verdadeiros
    y_pred : array-like
        Valores preditos
    bounds : tuple
        (low, high) onde high pode ser None
    metric : str
        Nome da métrica

    Returns
    -------
    float
        Valor da métrica ou NaN se não houver amostras na faixa
    """
    # Apply shift alignment
    y_true, y_pred = align_predictions_for_shift(y_true, y_pred)

    low, high = bounds
    if high is None:
        mask = y_true >= low
    else:
        mask = (y_true >= low) & (y_true < high)

    if not np.any(mask):
        return np.nan

    y_true_range = y_true[mask]
    y_pred_range = y_pred[mask]

    m = metric.lower()
    if m == "mae":
        return float(np.mean(np.abs(y_true_range - y_pred_range)))
    elif m == "rmse":
        return float(np.sqrt(np.mean((y_true_range - y_pred_range) ** 2)))
    elif m == "mape":
        nonzero_mask = y_true_range != 0
        if not np.any(nonzero_mask):
            return np.nan
        return float(
            np.mean(
                np.abs(
                    (y_true_range[nonzero_mask] - y_pred_range[nonzero_mask])
                    / y_true_range[nonzero_mask]
                )
            )
            * 100
        )
    elif m == "smape":
        denominator = (np.abs(y_true_range) + np.abs(y_pred_range)) / 2
        nonzero_denom = denominator != 0
        if not np.any(nonzero_denom):
            return np.nan
        return float(
            np.mean(
                np.abs(
                    y_true_range[nonzero_denom] - y_pred_range[nonzero_denom]
                )
                / denominator[nonzero_denom]
            )
            * 100
        )
    elif m == "pearson_correlation":
        return float(pearson_correlation(y_true_range, y_pred_range))
    elif m == "nse":
        return float(nse(y_true_range, y_pred_range))
    elif m == "kge":
        return float(kge(y_true_range, y_pred_range))
    elif m == "dtw_distance":
        return float(dtw_distance(y_true_range, y_pred_range))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def plot_range_scatter(y_true, y_pred, bounds, title_suffix=""):
    """Gráfico de dispersão simples por faixa, usado para visualização."""
    # Apply shift alignment
    y_true, y_pred = align_predictions_for_shift(y_true, y_pred)

    low, high = bounds
    if high is None:
        mask = y_true >= low
        range_label = f">={low}mm"
    else:
        mask = (y_true >= low) & (y_true < high)
        range_label = f"[{low}, {high})mm"

    if not np.any(mask):
        return None

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true[mask], y_pred[mask], alpha=0.4, s=8)

    min_val = min(y_true[mask].min(), y_pred[mask].min())
    max_val = max(y_true[mask].max(), y_pred[mask].max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black")

    ax.set_xlabel("y_true")
    ax.set_ylabel("y_pred")
    ax.set_title(f"Best config scatter {range_label}{title_suffix}")
    ax.grid(True)

    return fig


def process_metric(runs_root: Path, metric: str):
    """Analisa todos os runs para uma métrica específica usando NPZs e gera rankings."""
    if metric not in METRICS:
        print(f"[SKIP] Métrica desconhecida: {metric}")
        return

    if not runs_root.exists():
        print(f"Runs root not found: {runs_root}")
        return

    runs = sorted([p for p in runs_root.iterdir() if p.is_dir() and p.name != "summary"])
    if not runs:
        print(f"No runs found under {runs_root}")
        return

    better = METRIC_DIRECTION.get(metric, "lower")
    ascending = (better == "lower")

    # Dicionário de faixas a partir de INTERVALS
    ranges = {f"{int(low)}_{int(high) if high else 'plus'}": (low, high)
              for low, high, _ in INTERVALS}

    # Melhor global por faixa
    init_best = -np.inf if better == "higher" else np.inf
    cmp = (lambda x, y: x > y) if better == "higher" else (lambda x, y: x < y)
    global_best = {
        name: {"run_name": None, "metric_value": init_best}
        for name in ranges
    }

    # Desempenho por run para ranking
    all_runs_performance = {}

    print(f"\n=== Analyzing metric: {metric} ===")
    for run_dir in runs:
        run_name = run_dir.name
        print(f"Analyzing run: {run_name}")

        # Validador de arrays
        def _validate_arrays(y_true, y_pred):
            yt = np.asarray(y_true).flatten()
            yp = np.asarray(y_pred).flatten()
            ok = True
            reasons = []
            if yt.size == 0 or yp.size == 0:
                ok = False
                reasons.append("empty arrays")
            if yt.shape != yp.shape:
                ok = False
                reasons.append(f"shape mismatch {yt.shape} vs {yp.shape}")
            if not np.all(np.isfinite(yt)):
                ok = False
                reasons.append("y_true contains non-finite values")
            if not np.all(np.isfinite(yp)):
                ok = False
                reasons.append("y_pred contains non-finite values")
            return ok, reasons, yt, yp

        # 1) Tenta NPZ no nível do run
        aggregated_source = None
        y_true_all = None
        y_pred_all = None
        n_stations_used = 0

        run_level = load_run_level_predictions(run_dir)
        if run_level is not None:
            ok, reasons, yt, yp = _validate_arrays(run_level["y_true"], run_level["y_pred"])
            if not ok:
                print(f"  [WARN] Invalid run-level NPZ for {run_name}: {', '.join(reasons)}")
            else:
                y_true_all, y_pred_all = yt, yp
                aggregated_source = "run_level_npz"
                n_stations_used = 1

        # 2) Fallback: concatena NPZ por estação
        if y_true_all is None:
            station_dirs = [
                d for d in run_dir.iterdir() if d.is_dir() and (d / "predictions").exists()
            ]
            stations = [d.name for d in station_dirs]
            all_y_true = []
            all_y_pred = []
            for station in stations:
                preds = load_station_predictions(run_dir, station)
                if preds is None:
                    continue
                ok, reasons, yt, yp = _validate_arrays(preds["y_true"], preds["y_pred"])
                if not ok:
                    print(f"    [WARN] Invalid station NPZ '{station}': {', '.join(reasons)}")
                    continue
                all_y_true.append(yt)
                all_y_pred.append(yp)

            if all_y_true:
                y_true_all = np.concatenate(all_y_true)
                y_pred_all = np.concatenate(all_y_pred)
                aggregated_source = "per_station_npz"
                n_stations_used = len(all_y_true)

        if y_true_all is None or y_pred_all is None:
            print(f"  [SKIP] No valid NPZ predictions found in {run_name}")
            continue

        # Métrica por faixa
        per_run_range_metrics = {name: [] for name in ranges}
        for range_name, bounds in ranges.items():
            val = compute_range_errors(y_true_all, y_pred_all, bounds, metric=metric)
            per_run_range_metrics[range_name].append(val)
            if np.isnan(val):
                print(f"    Range {range_name}: no samples in range")
            else:
                arrow = "↓" if better == "lower" else "↑"
                print(f"    Range {range_name}: {metric.upper()}={val:.6f} {arrow}")

        # Métrica geral (log)
        overall_val = compute_range_errors(y_true_all, y_pred_all, (float("-inf"), None), metric=metric)
        if np.isnan(overall_val):
            print("    Overall metric: NaN (no valid samples)")
        else:
            print(f"    Overall {metric.upper()}: {overall_val:.6f}")

        # Média por faixa
        run_range_summary = {}
        for range_name in ranges:
            vals = [v for v in per_run_range_metrics[range_name] if not np.isnan(v)]
            metric_mean = float(np.mean(vals)) if vals else np.nan
            run_range_summary[range_name] = {
                "metric_mean": metric_mean,
                "n_stations": n_stations_used,
            }

            # Atualiza melhor global conforme direção
            if not np.isnan(metric_mean):
                current_best = global_best[range_name]["metric_value"]
                if cmp(metric_mean, current_best):
                    global_best[range_name]["metric_value"] = metric_mean
                    global_best[range_name]["run_name"] = run_name  # type: ignore[assignment]

        # Armazena desempenho do run
        all_runs_performance[run_name] = run_range_summary

        # Salva resumo por run (por métrica)
        best_mixed_dir = run_dir / "best_mixed_config"
        ensure_dir(best_mixed_dir)

        summary_ranges: dict[str, dict[str, object]] = {}
        for range_name in ranges:
            summary_ranges[range_name] = {  # type: ignore[assignment]
                "bounds": list(ranges[range_name]),
                "metric_mean": run_range_summary[range_name]["metric_mean"],
                "n_stations": run_range_summary[range_name]["n_stations"],
            }

        summary: dict[str, object] = {
            "run_name": run_name,
            "metric": metric,
            "data_source": aggregated_source or "npz",
            "note": "Metrics computed from test predictions (.npz)",
            "ranges": summary_ranges,
        }

        with open(best_mixed_dir / f"mixed_config_summary_{metric}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    # Resumo global por métrica
    global_summary = {}
    for range_name, info in global_best.items():
        best_val = info["metric_value"]
        global_summary[range_name] = {
            "best_run": info["run_name"],
            "best_metric": (float(best_val) if np.isfinite(best_val) else None),
            "bounds": list(ranges[range_name]),
        }

    global_out = runs_root / f"best_mixed_config_global_summary_{metric}.json"
    with open(global_out, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)

    print("Global best mixed-config summary written to", global_out)

    # ==========================================================
    # ARQUIVOS DE RANKING
    # ==========================================================
    summary_dir = runs_root / "summary"
    ensure_dir(summary_dir)

    ranking_data = []
    for run_name, run_performance in all_runs_performance.items():
        row = {"run_name": run_name}
        for range_name, bounds in ranges.items():
            if range_name in run_performance:
                row[f"{range_name}_{metric}"] = run_performance[range_name]["metric_mean"]
                row[f"{range_name}_n_stations"] = run_performance[range_name]["n_stations"]
        ranking_data.append(row)

    # DataFrame + ordenação conforme direção
    df_ranking = pd.DataFrame(ranking_data)
    metric_cols = [col for col in df_ranking.columns if f"_{metric}" in col and "n_stations" not in col]
    if metric_cols:
        overall_col = f"overall_{metric}"
        df_ranking[overall_col] = df_ranking[metric_cols].mean(axis=1)
        df_ranking = df_ranking.sort_values(by=overall_col, ascending=ascending, na_position="last")

    # CSV consolidado por métrica
    ranking_csv = summary_dir / f"ranking_{metric}.csv"
    df_ranking.to_csv(ranking_csv, index=False)
    print(f"Ranking summary written to {ranking_csv}")

    # JSON consolidado por métrica
    ranking_json = summary_dir / f"ranking_{metric}.json"
    ranking_dict = {
        "metric": metric,
        "better": better,
        "intervals": [{"low": low, "high": high, "label": label}
                      for low, high, label in INTERVALS],
        "global_best": global_summary,
        "all_runs": ranking_data,
    }
    with open(ranking_json, "w", encoding="utf-8") as f:
        json.dump(ranking_dict, f, indent=2, ensure_ascii=False)
    print(f"Ranking JSON written to {ranking_json}")

    # ----------------------------------------------------------
    # RANKINGS POR INTERVALO (ordenação depende da direção)
    # ----------------------------------------------------------
    per_interval_rows = []
    for range_name, bounds in ranges.items():
        col = f"{range_name}_{metric}"
        if col not in df_ranking.columns:
            continue
        df_int = (
            df_ranking[["run_name", col]]
            .dropna()
            .rename(columns={col: metric})
        )
        df_int = df_int.sort_values(by=metric, ascending=ascending, kind="mergesort").reset_index(drop=True)
        df_int.insert(0, "interval", range_name)
        df_int["rank"] = np.arange(1, len(df_int) + 1)
        interval_csv = summary_dir / f"ranking_{metric}_{range_name}.csv"
        df_int.to_csv(interval_csv, index=False)
        per_interval_rows.append(df_int)
        print(f"Per-interval ranking written to {interval_csv}")

    if per_interval_rows:
        df_per_interval = pd.concat(per_interval_rows, ignore_index=True)
        consolidated_csv = summary_dir / f"ranking_{metric}_per_interval.csv"
        df_per_interval.to_csv(consolidated_csv, index=False)
        print(f"Consolidated per-interval ranking written to {consolidated_csv}")

    # ==========================================================
    # CONFIGURAÇÃO MISTA VIRTUAL (por métrica)
    # ==========================================================
    virtual_config = {
        "description": (
            "Virtual mixed configuration selecting the best run per "
            f"precipitation range, based on test {metric.upper()}."
        ),
        "metric": metric,
        "ranges": {},
        "notes": [
            "For each precipitation range, use the listed run's model/checkpoints "
            "from its directory under 'runs/'.",
            "At inference time you can conceptually route samples into different "
            "models depending on the expected precipitation range.",
        ],
    }

    for range_name, info in global_best.items():
        best_run = info["run_name"]
        best_metric = info["metric_value"]
        bounds = list(ranges[range_name])

        if best_run is None or not np.isfinite(best_metric):
            entry = {
                "bounds": bounds,
                "best_run_name": None,
                f"best_{metric}": None,
                "run_dir": None,
                "per_run_summary_path": None,
            }
        else:
            entry = {
                "bounds": bounds,
                "best_run_name": best_run,
                f"best_{metric}": float(best_metric),
                "run_dir": f"runs/{best_run}",
                "per_run_summary_path": (
                    f"runs/{best_run}/best_mixed_config/mixed_config_summary_{metric}.json"
                ),
            }

        virtual_config["ranges"][range_name] = entry

    virtual_out = runs_root / f"best_mixed_config_virtual_{metric}.json"
    with open(virtual_out, "w", encoding="utf-8") as f:
        json.dump(virtual_config, f, indent=2, ensure_ascii=False)

    print("Virtual mixed configuration written to", virtual_out)


def visualize_global_mixed_config(runs_root: Path, metric: str):
    """Visualização simples do melhor por faixa (barras), por métrica."""
    summary_path = runs_root / f"best_mixed_config_global_summary_{metric}.json"
    if not summary_path.exists():
        print(f"Global summary not found: {summary_path}")
        return

    with open(summary_path, "r", encoding="utf-8") as f:
        global_summary = json.load(f)

    ranges = list(global_summary.keys())
    metrics_vals = [
        global_summary[r]["best_metric"] if global_summary[r]["best_metric"] is not None else np.nan
        for r in ranges
    ]
    labels = [
        global_summary[r]["best_run"] if global_summary[r]["best_run"] is not None else "(none)"
        for r in ranges
    ]

    x = np.arange(len(ranges))

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(x, metrics_vals, color="steelblue")

    ax.set_xticks(x)
    ax.set_xticklabels(ranges)
    ax.set_ylabel(f"Best {metric.upper()}")
    ax.set_title(f"Best mixed configuration per precipitation range ({metric.upper()})")

    # Anota cada barra com o nome do run
    for bar, label in zip(bars, labels):
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                label,
                ha="center",
                va="bottom",
                rotation=45,
                fontsize=8,
            )

    fig.tight_layout()

    out_path = runs_root / f"best_mixed_config_global_bar_{metric}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Mixed-setting visualization written to", out_path)


def build_mixed_timeseries_visualization(runs_root: Path, metric: str):
    """Visualização de série temporal da configuração mista, por métrica.

    Para cada faixa, usamos o run globalmente melhor nessa faixa (pela métrica).
    Concatena amostras de teste e constrói uma série "mista".
    """
    global_path = runs_root / f"best_mixed_config_global_summary_{metric}.json"
    if not global_path.exists():
        print(f"Global summary not found, skipping mixed timeseries: {global_path}")
        return

    with open(global_path, "r", encoding="utf-8") as f:
        global_summary = json.load(f)

    # Faixas
    ranges = {f"{int(low)}_{int(high) if high else 'plus'}": (low, high)
              for low, high, _ in INTERVALS}

    # Carrega predições dos melhores por faixa
    range_runs = {}
    for range_name, bounds in ranges.items():
        best_run = global_summary.get(range_name, {}).get("best_run")
        if best_run is None:
            continue

        run_dir = runs_root / best_run
        if not run_dir.exists():
            continue

        # Preferência: NPZ do run; senão, concatena por estação
        run_level = load_run_level_predictions(run_dir)
        if run_level is not None:
            range_runs[range_name] = {
                "run_name": best_run,
                "y_true": run_level["y_true"],
                "y_pred": run_level["y_pred"],
            }
            continue

        station_dirs = [
            d for d in run_dir.iterdir() if d.is_dir() and (d / "predictions").exists()
        ]
        stations = [d.name for d in station_dirs]
        if not stations:
            continue

        all_y_true = []
        all_y_pred = []
        for station in stations:
            preds = load_station_predictions(run_dir, station)
            if preds is None:
                continue
            all_y_true.append(preds["y_true"])
            all_y_pred.append(preds["y_pred"])

        if not all_y_true:
            continue

        y_true_concat = np.concatenate(all_y_true)
        y_pred_concat = np.concatenate(all_y_pred)
        range_runs[range_name] = {
            "run_name": best_run,
            "y_true": y_true_concat,
            "y_pred": y_pred_concat,
        }

    if not range_runs:
        print("No per-range best runs with predictions found; skipping mixed timeseries plot.")
        return

    base = next(iter(range_runs.values()))
    y_true_all = base["y_true"].astype(float).flatten()
    n = len(y_true_all)

    mixed_pred = np.zeros_like(y_true_all)

    # Máscaras por faixa
    masks = {}
    for range_name, bounds in ranges.items():
        low, high = bounds
        if high is None:
            mask = y_true_all >= low
        else:
            mask = (y_true_all >= low) & (y_true_all < high)
        masks[range_name] = mask

    # Atribui predições mistas por faixa
    for range_name, bounds in ranges.items():
        if range_name not in range_runs:
            continue
        run_preds = range_runs[range_name]["y_pred"].astype(float).flatten()
        m = min(len(run_preds), n)
        mask = masks[range_name][:m]
        mixed_pred[:m][mask] = run_preds[:m][mask]

    # Fallback: onde ainda zero e y_true != 0, usa predição base
    base_pred = base["y_pred"].astype(float).flatten()
    if len(base_pred) >= n:
        fallback_mask = (mixed_pred == 0) & (y_true_all != 0)
        mixed_pred[fallback_mask] = base_pred[:n][fallback_mask]

    # Função para computar valor geral da métrica na série mista
    def _overall_metric_value(y_true_v, y_pred_v, mname: str) -> float:
        if mname == "mae":
            return float(mae(y_true_v, y_pred_v))
        elif mname == "pearson_correlation":
            return float(pearson_correlation(y_true_v, y_pred_v))
        elif mname == "nse":
            return float(nse(y_true_v, y_pred_v))
        elif mname == "kge":
            return float(kge(y_true_v, y_pred_v))
        elif mname == "dtw_distance":
            return float(dtw_distance(y_true_v, y_pred_v))
        else:
            return float("nan")

    mixed_metric_val = _overall_metric_value(y_true_all, mixed_pred, metric)

    # Plot série temporal
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_true_all, label="y_true", linewidth=1.0)
    ax.plot(mixed_pred, label="y_pred_mixed", linewidth=1.0)

    ax.set_title(f"Mixed configuration timeseries (overall {metric.upper()}={mixed_metric_val:.3f})")
    ax.set_xlabel("Time index (test samples)")
    ax.set_ylabel("Precipitation (mm)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = runs_root / f"best_mixed_config_timeseries_{metric}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Mixed-setting timeseries visualization written to", out_path)


if __name__ == "__main__":
    runs_root = PROJECT_ROOT / "runs"
    # Varrer e analisar para cada métrica
    for _metric in METRICS:
        process_metric(runs_root, _metric)
        visualize_global_mixed_config(runs_root, _metric)
        build_mixed_timeseries_visualization(runs_root, _metric)
