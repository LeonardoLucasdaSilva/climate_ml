import matplotlib.pyplot as plt

def plot_training_history(history):
    plt.figure(figsize=(8, 5))

    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_real_vs_predicted_scatter(y_true, y_pred):
    plt.figure(figsize=(6, 6))

    plt.scatter(y_true, y_pred, alpha=0.5)

    # Diagonal (perfect prediction)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.xlabel("Real values")
    plt.ylabel("Predicted values")
    plt.title("Real vs Predicted")
    plt.grid(True)
    plt.axis("equal")
    plt.show()

