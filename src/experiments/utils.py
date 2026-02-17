from src.config.paths import RUNS_DIR


def filter_stations(df, config):
    """
    Filter the stations DataFrame according to experiment configuration.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing station metadata with at least a "cidade" column.
    config : dict
        Experiment configuration dictionary. Expected structure:
        config["experiment"]["stations_mode"] can be:
            - "all" (default): keep all stations
            - "single": filter to one station specified by
              config["experiment"]["single_station_name"]

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame containing only the selected stations.

    Raises
    ------
    ValueError
        If stations_mode is "single" but the specified station name
        is not found in the DataFrame.
    """
    stations_mode = config.get("experiment", {}).get("stations_mode", "all")

    if stations_mode == "single":
        name = config["experiment"].get("single_station_name")
        df = df[df["cidade"] == name]

        if df.empty:
            raise ValueError(f"Station '{name}' not found.")

    return df


def resolve_output_directory(cidade, config, run_dir):
    """
    Resolve the base output directory for a given station according to
    experiment output configuration.

    Parameters
    ----------
    cidade : str
        Station name.
    config : dict
        Experiment configuration dictionary. Expected structure:
        config["experiment"]["output_mode"] can be:
            - "standard": per-station folder inside run_dir/locations/
            - "debug": shared debug folder inside run_dir/
            - "global_debug": shared global debug folder inside RUNS_DIR/
    run_dir : pathlib.Path
        Root directory of the current experiment run.

    Returns
    -------
    pathlib.Path
        Path object representing the base directory where outputs
        for the station should be saved.

    Raises
    ------
    ValueError
        If output_mode is not supported.
    """
    mode = config.get("experiment", {}).get("output_mode", "standard")

    if mode == "standard":
        return run_dir / "locations" / cidade

    if mode == "debug":
        return run_dir / "debug_outputs"

    if mode == "global_debug":
        root = RUNS_DIR / "_GLOBAL_DEBUG"
        root.mkdir(parents=True, exist_ok=True)
        return root

    raise ValueError("Unsupported output_mode")
