import torch
from torch.utils.data import DataLoader, TensorDataset


def predict_timeseries_model(
    model,
    X,
    batch_size=256,
    device=None
):
    """
    Generates predictions for a PyTorch time-series model.

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

            predictions.append(outputs.cpu())

    predictions = torch.cat(predictions, dim=0)

    return predictions.numpy()