from pathlib import Path

def save_figure(fig, path, dpi=300, bbox_inches="tight", close=True):
    """
    Save a matplotlib figure safely.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : str or Path
        Destination file path (including extension).
    dpi : int, default=300
        Resolution of saved image.
    bbox_inches : str, default="tight"
        Bounding box setting for matplotlib.
    close : bool, default=True
        Whether to close the figure after saving.

    Returns
    -------
    Path
        The resolved path where the figure was saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)

    if close:
        import matplotlib.pyplot as plt
        plt.close(fig)

    return path