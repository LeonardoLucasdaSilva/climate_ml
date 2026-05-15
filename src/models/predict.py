import torch
from torch.utils.data import DataLoader, TensorDataset


def predict_timeseries_model(
    model,
    X,
    batch_size=256,
    device=None,
):
    """Generates predictions for a PyTorch time-series model.

    Parameters
    ----------
    model : torch.nn.Module
    X : numpy array or torch tensor
    batch_size : int
    device : torch.device

    Returns
    -------
    numpy.ndarray
        Predictions with shape (n_samples, horizon)
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    # Convert to tensor if needed
    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float32)

    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predictions = []

    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(device)

            outputs = model(X_batch)

            # Backward-compatible: if model returns a tuple, take regression part
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            predictions.append(outputs.cpu())

    predictions = torch.cat(predictions, dim=0)

    return predictions.numpy()


def predict_timeseries_multitask(
    model,
    X,
    batch_size=256,
    device=None,
):
    """Predict using a multitask model, returning regression and classification.

    Returns
    -------
    dict with keys:
        - "regression": np.ndarray of shape (n_samples, horizon)
        - "logits": np.ndarray of shape (n_samples, horizon)
        - "probs": np.ndarray of shape (n_samples, horizon) with sigmoid applied
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float32)

    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    reg_preds = []
    cls_logits = []

    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(device)

            outputs = model(X_batch)

            if not isinstance(outputs, tuple) or len(outputs) != 2:
                raise ValueError(
                    "predict_timeseries_multitask expects model to return "
                    "(y_reg, y_cls_logits)."
                )

            y_reg, y_logits = outputs

            reg_preds.append(y_reg.cpu())
            cls_logits.append(y_logits.cpu())

    reg_preds = torch.cat(reg_preds, dim=0)
    cls_logits = torch.cat(cls_logits, dim=0)

    probs = torch.sigmoid(cls_logits)

    return {
        "regression": reg_preds.numpy(),
        "logits": cls_logits.numpy(),
        "probs": probs.numpy(),
    }
