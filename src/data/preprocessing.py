import numpy as np

def create_sliding_windows(
    series,
    window_size,
    horizon=1,
    multi_step=False
):
    """
    Transform a time series into sliding windows for forecasting.
    """

    if multi_step:
        print("Multi-step not implemented yet")
    else:

        X,y = [],[]

        m = len(series) - (window_size + horizon)

        for i in range(m+1):
            X.append(series[i:i+window_size])
            y.append(series[i+window_size:i+window_size+horizon])

        return np.array(X), np.array(y)

