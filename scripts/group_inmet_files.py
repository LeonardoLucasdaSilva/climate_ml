import re
import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

RAW_DIR = INTERIM_DATA_DIR / "inmet"
PROCESSED_DIR = PROCESSED_DATA_DIR / "inmet"

file_regex = re.compile(
    r"INMET_.*?_([A-Z]{2})_([A-Z0-9]+)_.*?_(\d{2}-\d{2}-\d{4})_A_(\d{2}-\d{2}-\d{4})",
    re.IGNORECASE,
)

# --------------------------------------------
# TARGET SCHEMA (FINAL DATASET FORMAT)
# --------------------------------------------

TARGET_COLUMNS = [
    "PRECIPITACAO_TOTAL",
    "PRESSAO",
    "PRESSAO_MIN",
    "PRESSAO_MAX",
    "RADIACAO",
    "TEMPERATURA",
    "PONTO_ORVALHO",
    "TEMPERATURA_MAXIMA",
    "TEMPERATURA_MIN",
    "PONTO_ORVALHO_MAX",
    "PONTO_ORVALHO_MIN",
    "UMIDADE_MAX",
    "UMIDADE_MIN",
    "UMIDADE",
    "DIRECAO_VENTO",
    "RAJADA_VENTO",
    "VELOCIDADE_VENTO",
]

DAILY_TARGET_COLUMNS = [
    column for column in TARGET_COLUMNS if column != "DIRECAO_VENTO"
] + [
    "DIRECAO_VENTO_SIN",
    "DIRECAO_VENTO_COS",
]

# --------------------------------------------
# FILE PARSING
# --------------------------------------------

def parse_filename(path):

    m = file_regex.search(path.stem)
    if not m:
        return None

    state = m.group(1)
    station = m.group(2)

    start = pd.to_datetime(m.group(3), dayfirst=True)
    end = pd.to_datetime(m.group(4), dayfirst=True)

    return state, station, start, end


# --------------------------------------------
# COLLECT FILES
# --------------------------------------------

def collect_files(state_filter=None, station_filter=None):

    stations = {}

    source_files = set(RAW_DIR.rglob("*.csv")) | set(RAW_DIR.rglob("*.CSV"))

    for file in source_files:

        parsed = parse_filename(file)

        if not parsed:
            continue

        state, station, start, end = parsed

        if state_filter and state.upper() != state_filter.upper():
            continue

        if station_filter and station.upper() != station_filter.upper():
            continue

        stations.setdefault((state, station), []).append(
            {
                "path": file,
                "start": start,
                "end": end,
                "size": file.stat().st_size
            }
        )

    print("Stations detected:", len(stations))
    return stations


def infer_year_range(stations):

    years = [
        year
        for files in stations.values()
        for f in files
        for year in (f["start"].year, f["end"].year)
    ]

    if not years:
        raise RuntimeError("No INMET source files found to infer year range.")

    return min(years), max(years)


# --------------------------------------------
# HEADER DETECTION
# --------------------------------------------

def detect_header_and_metadata(file):

    metadata = {}

    with open(file, encoding="latin-1") as f:

        for i, line in enumerate(f):

            line = line.strip()

            upper_line = line.upper()

            if upper_line.startswith("DATA") and "HORA" in upper_line:
                return i, metadata

            if ":;" in line:
                k, v = line.split(":;", 1)
                metadata[k.strip()] = v.strip()

    return i, metadata


# --------------------------------------------
# COLUMN CLEANING
# --------------------------------------------

def normalize_columns(df):

    df = df.loc[:, ~df.columns.str.contains("^UNNAMED", case=False)]

    cols = (
        df.columns
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
        .str.strip()
        .str.upper()
        .str.replace(r"\(.*?\)", "", regex=True)
        .str.replace(",", "")
        .str.replace(" ", "_")
    )

    df.columns = cols

    rename_map = {
        "PRECIPITACAO_TOTAL": "PRECIPITACAO_TOTAL",
        "PRECIPITACAO_TOTAL_HORARIO_MM": "PRECIPITACAO_TOTAL",
        "PRECIPITACAO_TOTAL_HORARIO_": "PRECIPITACAO_TOTAL",
        "PRESSAO_ATMOSFERICA_AO_NIVEL_DA_ESTACAO_HORARIA_": "PRESSAO",
        "PRESSAO_ATMOSFERICA_MAX.NA_HORA_ANT.__": "PRESSAO_MAX",
        "PRESSAO_ATMOSFERICA_MIN._NA_HORA_ANT.__": "PRESSAO_MIN",
        "RADIACAO_GLOBAL_": "RADIACAO",
        "TEMPERATURA_DO_AR_-_BULBO_SECO_HORARIA_": "TEMPERATURA",
        "TEMPERATURA_DO_PONTO_DE_ORVALHO_": "PONTO_ORVALHO",
        "TEMPERATURA_MAXIMA_NA_HORA_ANT.__": "TEMPERATURA_MAXIMA",
        "TEMPERATURA_MINIMA_NA_HORA_ANT.__": "TEMPERATURA_MIN",
        "TEMPERATURA_ORVALHO_MAX._NA_HORA_ANT.__": "PONTO_ORVALHO_MAX",
        "TEMPERATURA_ORVALHO_MIN._NA_HORA_ANT.__": "PONTO_ORVALHO_MIN",
        "UMIDADE_REL._MAX._NA_HORA_ANT.__": "UMIDADE_MAX",
        "UMIDADE_REL._MIN._NA_HORA_ANT.__": "UMIDADE_MIN",
        "UMIDADE_RELATIVA_DO_AR_HORARIA_": "UMIDADE",
        "VENTO_DIRECAO": "DIRECAO_VENTO",
        "VENTO_DIRECAO_HORARIA__)": "DIRECAO_VENTO",
        "VENTO_RAJADA_MAXIMA": "RAJADA_VENTO",
        "VENTO_RAJADA_MAXIMA_": "RAJADA_VENTO",
        "VENTO_VELOCIDADE": "VELOCIDADE_VENTO",
        "VENTO_VELOCIDADE_HORARIA_": "VELOCIDADE_VENTO",
    }

    df = df.rename(columns=rename_map)

    return df


# --------------------------------------------
# ENFORCE DATASET SCHEMA
# --------------------------------------------

def enforce_schema(df, columns=TARGET_COLUMNS):

    for col in columns:
        if col not in df.columns:
            df[col] = -9999

    df = df[columns]

    return df


# --------------------------------------------
# CIRCULAR STATISTICS
# --------------------------------------------

def add_wind_direction_components(df):
    """Add sine/cosine components for valid wind directions in degrees."""

    directions = pd.to_numeric(df["DIRECAO_VENTO"], errors="coerce")
    directions = directions.where(directions.between(0, 360, inclusive="both"))
    directions = directions.replace(-9999, np.nan)
    radians = np.deg2rad(directions % 360)

    df = df.copy()
    df["DIRECAO_VENTO_SIN"] = np.sin(radians)
    df["DIRECAO_VENTO_COS"] = np.cos(radians)
    return df


# --------------------------------------------
# FIND DATE/HOUR COLUMNS
# --------------------------------------------

def find_column(cols, keyword):

    for c in cols:
        if keyword in c:
            return c

    return None


# --------------------------------------------
# DATA READER
# --------------------------------------------

def read_data(file):

    header_len, metadata = detect_header_and_metadata(file)

    df = pd.read_csv(
        file,
        sep=";",
        encoding="latin-1",
        skiprows=header_len,
        decimal=",",
        na_values=-9999,
        low_memory=False
    )

    df = normalize_columns(df)

    # print(len(df.columns))
    # for col in df.columns:
    #     print(repr(col))

    date_col = find_column(df.columns, "DATA")
    hour_col = find_column(df.columns, "HORA")

    if date_col is None or hour_col is None:
        raise Exception("Date/hour column not found")

    df[hour_col] = (
        df[hour_col]
        .astype(str)
        .str.replace(" UTC", "", regex=False)
        .str.replace(":", "", regex=False)
        .str.zfill(4)
        .str.replace(r"^(\d{2})(\d{2})$", r"\1:\2", regex=True)
    )

    df["datetime"] = pd.to_datetime(df[date_col] + " " + df[hour_col], errors="coerce")

    df = df.drop(columns=[date_col, hour_col])

    df = df.dropna(subset=["datetime"])

    df = df.set_index("datetime")

    df = enforce_schema(df)

    return df, metadata


# --------------------------------------------
# YEAR INDEX
# --------------------------------------------

def build_year_index(year):

    start = f"{year}-01-01 00:00"
    end = f"{year}-12-31 23:00"

    return pd.date_range(start, end, freq="h")


# --------------------------------------------
# REMOVE DUPLICATE FILES (KEEP BIGGEST)
# --------------------------------------------

def deduplicate_files(files):

    unique = {}

    for f in files:

        key = (f["start"], f["end"])

        if key not in unique:
            unique[key] = f
        else:
            if f["size"] > unique[key]["size"]:
                unique[key] = f

    return list(unique.values())


# --------------------------------------------
# PROCESS STATION
# --------------------------------------------

def process_station(args):

    state, station, files, start_year, end_year = args

    print("Processing", state, station)

    files = deduplicate_files(files)
    files = sorted(files, key=lambda x: (x["start"], x["end"], x["size"]))
    station_start = min(f["start"] for f in files)
    station_end = max(f["end"] for f in files)

    out_dir = PROCESSED_DIR / state / station
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = out_dir / f"{station}_{start_year}_{end_year}_daily.csv"

    first_write = True
    metadata_records = []

    # Aggregation rules
    AGG_RULES = {
        "PRECIPITACAO_TOTAL": "sum",
        "PRESSAO": "mean",
        "PRESSAO_MIN": "min",
        "PRESSAO_MAX": "max",
        "RADIACAO": "sum",
        "TEMPERATURA": "mean",
        "TEMPERATURA_MAXIMA": "max",
        "TEMPERATURA_MIN": "min",
        "PONTO_ORVALHO": "mean",
        "PONTO_ORVALHO_MAX": "max",
        "PONTO_ORVALHO_MIN": "min",
        "UMIDADE": "mean",
        "UMIDADE_MAX": "max",
        "UMIDADE_MIN": "min",
        "DIRECAO_VENTO_SIN": "mean",
        "DIRECAO_VENTO_COS": "mean",
        "RAJADA_VENTO": "max",
        "VELOCIDADE_VENTO": "mean",
    }

    for year in range(start_year, end_year + 1):

        dfs = []

        for f in files:

            if f["end"].year < year or f["start"].year > year:
                continue

            try:
                df, meta = read_data(f["path"])

                df = df[df.index.year == year]

                if not df.empty:
                    dfs.append(df)

                meta_record = dict(meta)
                meta_record["file_start"] = f["start"]
                meta_record["file_end"] = f["end"]
                meta_record["source_file"] = f["path"].name

                metadata_records.append(meta_record)

            except Exception as e:
                print("Error:", f["path"], e)

        if dfs:

            data = pd.concat(dfs)
            data = data.sort_index()
            data = data[~data.index.duplicated(keep="last")]
            data = add_wind_direction_components(data)

            data_daily = data.resample("D").agg(AGG_RULES)

        else:

            data_daily = pd.DataFrame(columns=DAILY_TARGET_COLUMNS)

        # Ensure a daily index only for the period covered by the source files.
        # This avoids padding partial current-year files, such as Jan-Jun 2026,
        # with artificial missing days through Dec 31.
        daily_start = max(pd.Timestamp(f"{year}-01-01"), station_start)
        daily_end = min(pd.Timestamp(f"{year}-12-31"), station_end)

        if daily_start > daily_end:
            continue

        daily_index = pd.date_range(
            daily_start,
            daily_end,
            freq="D"
        )

        data_daily = data_daily.reindex(daily_index)

        data_daily = enforce_schema(data_daily, DAILY_TARGET_COLUMNS)

        # Keep only DATE column
        data_daily.insert(0, "DATA", data_daily.index.strftime("%Y-%m-%d"))

        data_daily.reset_index(drop=True, inplace=True)

        data_daily.to_csv(
            dataset_path,
            sep=";",
            index=False,
            mode="w" if first_write else "a",
            header=first_write
        )

        first_write = False

    meta_df = pd.DataFrame(metadata_records)

    if not meta_df.empty:

        meta_df = meta_df.sort_values("file_start")
        meta_path = out_dir / "station_metadata_history.csv"
        meta_df.to_csv(meta_path, index=False)

    print("Saved", dataset_path)


# --------------------------------------------
# MAIN
# --------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description="Group standardized INMET hourly station CSV files into daily files."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of stations to process in parallel. Default is 1 to avoid "
            "large pandas memory spikes on the full INMET dataset."
        ),
    )
    parser.add_argument(
        "--state",
        help="Optional UF filter, for example RS or SP.",
    )
    parser.add_argument(
        "--station",
        help="Optional station code filter, for example A804.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report detected stations and year range; do not write outputs.",
    )
    return parser.parse_args()


def main():

    args = parse_args()

    stations = collect_files(args.state, args.station)

    start_year, end_year = infer_year_range(stations)
    print(f"Year range detected: {start_year}-{end_year}")
    print(f"Stations to process: {len(stations)}")

    tasks = [
        (state, station, files, start_year, end_year)
        for (state, station), files in stations.items()
    ]

    if args.dry_run:
        return

    max_workers = max(cpu_count() - 2, 1)
    workers = max(1, min(args.workers, max_workers, len(tasks)))
    print(f"Workers: {workers}")

    if workers == 1:
        for task in tasks:
            process_station(task)
        return

    with Pool(workers, maxtasksperchild=1) as pool:
        pool.map(process_station, tasks)


if __name__ == "__main__":
    main()
