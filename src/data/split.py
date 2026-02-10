import numpy as np

def temporal_train_val_test_split(
    X,
    y,
    train_size=0.7,
    val_size=0.2
):
    """
    Splits time series data into train, validation and test sets
    preserving temporal order.

    Parameters
    ----------
    X : array-like
        Input sequences.
    y : array-like
        Target values.
    train_size : float
        Fraction of data used for training.
    val_size : float
        Fraction of data used for validation.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """

    if (train_size + val_size) >= 1:
        raise ValueError("Split proportions must sum to 1 or less.")

    n = len(X)

    train_end = int(n * train_size)
    val_end   = int(n * (train_size + val_size))

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]

    X_test = X[val_end:]
    y_test = y[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test

def random_train_val_test_split(
    X,
    y,
    train_size=0.7,
    val_size=0.2,
    test_size=None,
    random_state=None
):
    """
    Randomly split data into train / validation / test sets.

    Parameters
    ----------
    X : array-like
        Feature matrix (n_samples, ...).
    y : array-like
        Target array (n_samples, ...).
    train_size : float
        Proportion of samples for training.
    val_size : float
        Proportion of samples for validation.
    test_size : float or None
        Proportion of samples for testing. If None, uses remaining samples.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """

    if test_size is None:
        total = train_size + val_size
    else:
        total = train_size + val_size + test_size

    if total > 1.0:
        raise ValueError(
            f"Split proportions sum to {total:.2f}, must be ≤ 1.0."
        )

    n_samples = len(X)

    if random_state is not None:
        np.random.seed(random_state)

    indices = np.random.permutation(n_samples)

    train_end = int(train_size * n_samples)
    val_end   = train_end + int(val_size * n_samples)

    train_idx = indices[:train_end]
    val_idx   = indices[train_end:val_end]
    test_idx  = indices[val_end:]

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_val = X[val_idx]
    y_val = y[val_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]

    return X_train, X_val, X_test, y_train, y_val, y_test