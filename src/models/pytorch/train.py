import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def train_regression_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=50,
    batch_size=64,
    patience=10,
    min_delta=0.0,
    lr=1e-3,
    device=None
):
    """
    Trains a PyTorch model for supervised regression with early stopping.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to train.
    X_train, y_train : array-like or torch.Tensor
        Training data.
    X_val, y_val : array-like or torch.Tensor
        Validation data.
    epochs : int, optional
        Maximum number of training epochs.
    batch_size : int, optional
        Batch size for training.
    patience : int, optional
        Number of epochs with no validation improvement before stopping.
    min_delta : float, optional
        Minimum change in validation loss to qualify as improvement.
    lr : float, optional
        Learning rate.
    device : torch.device, optional
        Device to run training on (CPU or CUDA).

    Returns
    -------
    history : dict
        Dictionary containing training and validation loss history.
    best_val_loss : float
        Best validation loss achieved during training.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # Convert numpy to torch tensors if needed
    if not torch.is_tensor(X_train):
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)

    # Fix shape: (N, 1, 1) -> (N, 1)
    if y_train.ndim == 3:
        y_train = y_train.squeeze(-1)

    if y_val.ndim == 3:
        y_val = y_val.squeeze(-1)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "val_loss": []
    }

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):

        # -------- TRAIN --------
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # -------- EARLY STOPPING --------
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()  # save best weights
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break

    model.load_state_dict(best_model_state)

    return history, best_val_loss