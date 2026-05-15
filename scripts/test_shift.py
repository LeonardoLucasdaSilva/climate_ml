"""
STANDALONE DEBUG SCRIPT: Sequence -> Prediction Debugging
===========================================================

This script loads the test predictions from a saved NPZ file and prints:
- Input sequence (7 timesteps)
- Predicted precipitation
- Actual precipitation
- Error metrics

Configuration: lstm_daily_2017-01-01-2020-12-31_ts7_h1_lossquantile_0.9_scalerTrue

NO EXTERNAL DEPENDENCIES - all data is loaded from saved files.
"""

import sys
from pathlib import Path
import numpy as np
import yaml
import pandas as pd
import traceback

# ============================================================================
# CONSTANTS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
RUN_NAME = "lstm_daily_2017-01-01-2020-12-31_ts7_h1_lossquantile_0.9_scalerTrue"
RUN_DIR = PROJECT_ROOT / "runs" / RUN_NAME

# Paths to saved files
NPZ_FILE = RUN_DIR / f"{RUN_NAME}_test_predictions.npz"
CONFIG_FILE = RUN_DIR / "config.yaml"

# ============================================================================
# VERIFY FILES EXIST
# ============================================================================

print("=" * 90)
print("DEBUG SCRIPT: SEQUENCE -> PREDICTION ANALYSIS")
print("=" * 90)
print()

print(f"Configuration: {RUN_NAME}")
print()

if not NPZ_FILE.exists():
    print(f"ERROR: NPZ file not found at:")
    print(f"  {NPZ_FILE}")
    sys.exit(1)

if not CONFIG_FILE.exists():
    print(f"ERROR: Config file not found at:")
    print(f"  {CONFIG_FILE}")
    sys.exit(1)

print(f"[OK] NPZ file found: {NPZ_FILE.name}")
print(f"[OK] Config file found: {CONFIG_FILE.name}")
print()

# ============================================================================
# LOAD CONFIGURATION
# ============================================================================

with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)

print("Configuration Details:")
print(f"  Dataset: {config['data']['initial_date']} to {config['data']['end_date']}")
print(f"  Target: {config['data']['target']}")
print(f"  Timesteps: {config['data']['timesteps']}")
print(f"  Horizon: {config['data']['horizon']}")
print(f"  Loss function: {config['training']['loss']}")
print(f"  Use scaler: {config['preprocessing']['use_scaler']}")
print(f"  Use log transform: {config['preprocessing']['use_log']}")
print(f"  Station: {config['experiment']['single_station_name']}")
print()

# ============================================================================
# LOAD PREDICTIONS FROM NPZ
# ============================================================================

print("Loading predictions from NPZ file...")
data = np.load(NPZ_FILE)

print(f"Available arrays in NPZ file:")
for key in data.files:
    arr = data[key]
    print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
print()

# Extract predictions and true values
y_pred = data['y_pred'].flatten()
y_true = data['y_true'].flatten()

print(f"Loaded predictions:")
print(f"  y_pred length: {len(y_pred)}")
print(f"  y_true length: {len(y_true)}")
print()

# Load test sequences if available
try:
    X_test = data['X_test']
    print(f"  X_test shape: {X_test.shape}")
    has_sequences = True
except KeyError:
    print(f"  X_test NOT available in NPZ file")
    print(f"    (Attempting to load from data pipeline...)")

    # Try to load X_test from the data pipeline
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.data.pipeline import prepare_station_data

        station_name = config['experiment']['single_station_name']
        print(f"    Loading data for: {station_name}")

        splits, _, _ = prepare_station_data(station_name, config)
        X_test = splits["X_test"]

        print(f"  X_test loaded from pipeline: {X_test.shape}")
        has_sequences = True
    except Exception as e:
        print(f"    Could not load X_test from pipeline: {str(e)[:80]}")
        has_sequences = False
        X_test = None

print()

# ============================================================================
# DENORMALIZATION FUNCTION
# ============================================================================

def denormalize_sequence(scaled_seq, min_val=68, max_val=None):
    """
    Denormalize sequence from [0, 1] range back to original values.

    If max_val is None, assumes max_val can be inferred or we use a default range.
    For now, we infer max from the relationship: scaled 0 -> 68 (min), scaled 1 -> max

    Standard MinMaxScaler: X_scaled = (X - min) / (max - min)
    Inverse: X = X_scaled * (max - min) + min
    """
    if max_val is None:
        # If only min is given, assume a typical precipitation max (e.g., 200mm)
        # But we can also try to infer from data
        max_val = 200.0  # Default maximum precipitation

    denorm_seq = scaled_seq * (max_val - min_val) + min_val
    return denorm_seq

# ============================================================================
# DISPLAY SEQUENCE -> PREDICTION PAIRS
# ============================================================================

timesteps = config['data']['timesteps']
num_to_show = 400

print("=" * 90)
print(f"FIRST {num_to_show} TEST SAMPLES: SEQUENCE -> PREDICTION -> ACTUAL")
print("=" * 90)
print()

if has_sequences and X_test is not None:
    # We have input sequences
    for i in range(min(num_to_show, len(y_pred))):
        sequence = X_test[i]  # Shape: (timesteps, num_features)

        # Extract first variable (typically precipitation)
        if len(sequence.shape) > 1:
            seq_values = sequence[:, 0]  # First feature for all timesteps
        else:
            seq_values = sequence

        # Denormalize the sequence (min_val=68 based on your specification)
        seq_denorm = denormalize_sequence(seq_values, min_val=0, max_val=68)

        pred = y_pred[i]
        true = y_true[i]
        error = abs(pred - true)

        # Format sequences for display
        seq_str_scaled = " -> ".join([f"{v:.2f}" for v in seq_values])
        seq_str_denorm = " -> ".join([f"{v:.2f}" for v in seq_denorm])

        print(f"Sample {i+1:3d}:")
        print(f"  Scaled sequence (0-1):        {seq_str_scaled}")
        print(f"  Denormalized (0-68):        {seq_str_denorm}")
        print(f"  Predicted (next day):          {pred:.4f} mm")
        print(f"  Actual (next day):             {true:.4f} mm")
        print(f"  Absolute Error:                {error:.4f} mm")

        # Add indicator
        if error < 1.0:
            status = "[GOOD]"
        elif error < 5.0:
            status = "[OK]"
        else:
            status = "[HIGH_ERROR]"
        print(f"  Status:                        {status}")
        print()

# ============================================================================
# OVERALL STATISTICS
# ============================================================================

print("=" * 90)
print("OVERALL TEST SET STATISTICS")
print("=" * 90)
print()

print("Predictions:")
print(f"  Min:  {y_pred.min():.4f} mm")
print(f"  Max:  {y_pred.max():.4f} mm")
print(f"  Mean: {y_pred.mean():.4f} mm")
print(f"  Std:  {y_pred.std():.4f} mm")
print()

print("Actual values:")
print(f"  Min:  {y_true.min():.4f} mm")
print(f"  Max:  {y_true.max():.4f} mm")
print(f"  Mean: {y_true.mean():.4f} mm")
print(f"  Std:  {y_true.std():.4f} mm")
print()

# Calculate metrics
abs_errors = np.abs(y_pred - y_true)
mae = np.mean(abs_errors)
rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
correlation = np.corrcoef(y_pred, y_true)[0, 1]

print("Error Metrics:")
print(f"  MAE (Mean Absolute Error):     {mae:.4f} mm")
print(f"  RMSE (Root Mean Squared Error): {rmse:.4f} mm")
print(f"  Pearson Correlation:           {correlation:.4f}")
print()

# Distribution of errors
print("Error Distribution:")
print(f"  Errors < 1.0 mm:   {np.sum(abs_errors < 1.0):4d} ({100*np.sum(abs_errors < 1.0)/len(abs_errors):.1f}%)")
print(f"  Errors < 5.0 mm:   {np.sum(abs_errors < 5.0):4d} ({100*np.sum(abs_errors < 5.0)/len(abs_errors):.1f}%)")
print(f"  Errors < 10.0 mm:  {np.sum(abs_errors < 10.0):4d} ({100*np.sum(abs_errors < 10.0)/len(abs_errors):.1f}%)")
print(f"  Errors >= 10.0 mm: {np.sum(abs_errors >= 10.0):4d} ({100*np.sum(abs_errors >= 10.0)/len(abs_errors):.1f}%)")
print()

# Analyze by precipitation level
print("Performance by Actual Precipitation Level:")
no_rain = y_true == 0
light_rain = (y_true > 0) & (y_true <= 5)
moderate_rain = (y_true > 5) & (y_true <= 20)
heavy_rain = y_true > 20

for label, mask in [("No rain (0mm)", no_rain),
                     ("Light rain (0-5mm)", light_rain),
                     ("Moderate rain (5-20mm)", moderate_rain),
                     ("Heavy rain (>20mm)", heavy_rain)]:
    if np.sum(mask) > 0:
        mae_subset = np.mean(abs_errors[mask])
        count = np.sum(mask)
        print(f"  {label:20s}: MAE={mae_subset:.4f} mm (n={count})")

print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 90)
print("DEBUG ANALYSIS COMPLETE")
print("=" * 90)
print()
print("Key observations:")
print(f"  Total test samples: {len(y_pred)}")
print(f"  Mean prediction error: {mae:.4f} mm")
print(f"  Prediction-actual correlation: {correlation:.4f}")
print(f"  Good predictions (<1mm error): {100*np.sum(abs_errors < 1.0)/len(abs_errors):.1f}%")
print()
print("Suggestions for debugging:")
if correlation < 0.3:
    print(f"  [WARNING] Low correlation ({correlation:.4f}) - check data alignment")
if mae > 5.0:
    print(f"  [WARNING] High MAE ({mae:.4f}) - model may need tuning or more training")
if np.sum(abs_errors >= 10.0) > 0.3 * len(abs_errors):
    print(f"  [WARNING] High error rate - check for outliers in predictions")
print()

