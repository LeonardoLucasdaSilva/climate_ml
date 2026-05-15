"""
Simple Autocorrelation Analysis Module

This module provides simple functions to:
1. Calculate autocorrelation function (ACF) for time series
2. Plot ACF with confidence intervals
3. Load data from NetCDF files with variable filenames
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


def calculate_acf(data, max_lags=None):
    """
    Calculate autocorrelation function (ACF).

    Parameters
    ----------
    data : array-like
        Time series data
    max_lags : int, optional
        Maximum lags to compute. Default is 50% of data length

    Returns
    -------
    acf : ndarray
        Autocorrelation values from lag 0 to max_lags
    """
    # Remove NaN values
    data_clean = data[~np.isnan(data)]

    # Set default max_lags
    if max_lags is None:
        max_lags = len(data_clean) // 2

    # Normalize data (remove mean)
    data_norm = data_clean - np.mean(data_clean)

    # Calculate ACF using numpy correlate
    acf = np.correlate(data_norm, data_norm, mode='full')
    acf = acf[len(acf)//2:]
    acf = acf[:max_lags + 1] / acf[0]

    return acf


def load_netcdf(file_path, variable_name=None):
    """
    Load data from NetCDF file.

    Parameters
    ----------
    file_path : str
        Path to NetCDF file
    variable_name : str, optional
        Variable to extract. If None, uses first data variable

    Returns
    -------
    data : ndarray
        Flattened data array
    var_name : str
        Variable name used
    """
    ds = xr.open_dataset(file_path)

    # Find variable if not specified
    if variable_name is None:
        data_vars = [v for v in ds.data_vars if not v.startswith('_')]
        variable_name = data_vars[0]

    data = ds[variable_name].values.flatten()
    ds.close()

    return data, variable_name


def plot_acf(acf_values, max_lags=None, title="Autocorrelation Function",
             save_path=None, figsize=(12, 6)):
    """
    Plot autocorrelation function with confidence intervals.

    Parameters
    ----------
    acf_values : ndarray
        ACF values from calculate_acf()
    max_lags : int, optional
        Number of lags to display
    title : str
        Plot title
    save_path : str, optional
        Path to save figure (e.g., 'acf_plot.png')
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    fig : matplotlib figure
        Figure object
    """
    if max_lags is None:
        max_lags = len(acf_values) - 1

    acf_plot = acf_values[:max_lags + 1]
    lags = np.arange(len(acf_plot))

    # Calculate 95% confidence interval
    ci = 1.96 / np.sqrt(len(acf_plot))

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(lags, acf_plot, width=0.5, alpha=0.7, color='steelblue')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=ci, color='red', linestyle='--', linewidth=1.5, label='95% CI')
    ax.axhline(y=-ci, color='red', linestyle='--', linewidth=1.5)
    ax.fill_between(lags, -ci, ci, alpha=0.1, color='red')

    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('Autocorrelation', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
                   exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")

    return fig


# Example usage
if __name__ == "__main__":
    nc_file = r"C:\Users\usuario\PycharmProjects\climate_ml\data\interim\era5_precipitation_timeseries_porto alegre - jardim botanico_1D.nc"

    if os.path.exists(nc_file):
        # Load data
        data, var_name = load_netcdf(nc_file)
        print(f"Loaded variable: {var_name}")
        print(f"Data shape: {data.shape}")

        # Calculate ACF
        acf = calculate_acf(data, max_lags=180)

        # Plot
        fig = plot_acf(acf, max_lags=180,
                      title=f"ACF - {var_name}",
                      save_path=os.path.join(os.path.dirname(nc_file), "acf_plot.png"))

        plt.show()
        print("Done!")

