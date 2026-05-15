import copy
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.models.losses import MultitaskLoss


def _compute_loss_and_outputs(model, X_batch, y_batch, criterion):
    """Helper to handle both single-task and multitask outputs/loss.

    Returns
    -------
    loss : torch.Tensor
        Scalar total loss used for backprop.
    y_pred_reg : torch.Tensor
        Regression predictions (for metrics), shape (batch, horizon).
    """
    outputs = model(X_batch)

    if isinstance(criterion, MultitaskLoss):
        # Model is expected to return (y_reg, y_cls_logits)
        y_pred_reg, y_pred_cls = outputs
        loss, _, _ = criterion(y_pred_reg, y_pred_cls, y_batch)
        return loss, y_pred_reg

    # Standard regression case
    loss = criterion(outputs, y_batch)
    return loss, outputs


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
    device=None,
    criterion=nn.MSELoss(),
    debug=False,  # NEW: Add debug flag
):
    """Trains a PyTorch model for supervised regression with early stopping.

    This function is backward-compatible with single-task training but also
    supports multitask mode when ``criterion`` is an instance of
    :class:`MultitaskLoss` and the model returns ``(y_reg, y_cls_logits)``.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # DEBUG: Log initial shapes
    if debug:
        print(f"[TRAIN DEBUG] Initial X_train shape: {X_train.shape}")
        print(f"[TRAIN DEBUG] Initial y_train shape: {y_train.shape}")
        print(f"[TRAIN DEBUG] Initial X_val shape: {X_val.shape}")
        print(f"[TRAIN DEBUG] Initial y_val shape: {y_val.shape}")

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

    # DEBUG: Log after shape fixing
    if debug:
        print(f"[TRAIN DEBUG] After reshape: X_train {X_train.shape}, y_train {y_train.shape}")
        print(f"[TRAIN DEBUG] X_train[0, -1, :] (last day of first window): {X_train[0, -1, :].numpy() if hasattr(X_train[0, -1, :], 'numpy') else X_train[0, -1, :]}")
        print(f"[TRAIN DEBUG] y_train[0] (target for first window): {y_train[0].numpy() if hasattr(y_train[0], 'numpy') else y_train[0]}")
        print(f"[TRAIN DEBUG] y_train[1] (target for second window): {y_train[1].numpy() if hasattr(y_train[1], 'numpy') else y_train[1]}")

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    best_model_state = copy.deepcopy(model.state_dict())
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

            loss, _ = _compute_loss_and_outputs(model, X_batch, y_batch, criterion)

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

                loss, _ = _compute_loss_and_outputs(
                    model, X_batch, y_batch, criterion
                )

                val_loss += loss.item()

        val_loss /= len(val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # -------- EARLY STOPPING --------
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())  # save best weights
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break

    model.load_state_dict(best_model_state)

    return history, best_val_loss
