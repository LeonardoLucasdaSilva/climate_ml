import re
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from src.config.paths import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

RAW_DIR = INTERIM_DATA_DIR / "inmet"
PROCESSED_DIR = PROCESSED_DATA_DIR / "inmet"

START_YEAR = 2000
END_YEAR = 2025

file_regex = re.compile(
    r"INMET_.*?_([A-Z]{2})_([A-Z0-9]+)_.*?_(\d{2}-\d{2}-\d{4})_A_(\d{2}-\d{2}-\d{4})",
    re.IGNORECASE,
)

# --------------------------------------------
# TARGET SCHEMA (FINAL DATASET FORMAT)
# --------------------------------------------

TARGET_COLUMNS = [
    "PRECIPITACAO_TOTAL_HORARIO",
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

def collect_files():

    stations = {}

    for file in RAW_DIR.rglob("*.csv"):

        parsed = parse_filename(file)

        if not parsed:
            continue

        state, station, start, end = parsed

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


# --------------------------------------------
# HEADER DETECTION
# --------------------------------------------

def detect_header_and_metadata(file):

    metadata = {}

    with open(file, encoding="latin-1") as f:

        for i, line in enumerate(f):

            line = line.strip()

            if line.startswith("DATA;"):
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
        "PRECIPITACAO_TOTAL": "PRECIPITACAO_TOTAL_HORARIO",
        "PRECIPITACAO_TOTAL_HORARIO_MM": "PRECIPITACAO_TOTAL_HORARIO",
        "VENTO_DIRECAO": "DIRECAO_VENTO",
        "VENTO_RAJADA_MAXIMA": "RAJADA_VENTO",
        "VENTO_VELOCIDADE": "VELOCIDADE_VENTO",
    }

    df = df.rename(columns=rename_map)

    return df


# --------------------------------------------
# ENFORCE DATASET SCHEMA
# --------------------------------------------

def enforce_schema(df):

    for col in TARGET_COLUMNS:
        if col not in df.columns:
            df[col] = -9999

    df = df[TARGET_COLUMNS]

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
        low_memory=True
    )

    df = normalize_columns(df)

    date_col = find_column(df.columns, "DATA")
    hour_col = find_column(df.columns, "HORA")

    if date_col is None or hour_col is None:
        raise Exception("Date/hour column not found")

    df[hour_col] = (
        df[hour_col]
        .astype(str)
        .str.replace(" UTC", "", regex=False)
        .str.zfill(5)
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

    state, station, files = args

    print("Processing", state, station)

    files = deduplicate_files(files)

    files = sorted(files, key=lambda x: x["start"])

    out_dir = PROCESSED_DIR / state / station
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = out_dir / f"{station}_2000_2025.csv"

    first_write = True

    metadata_records = []

    for year in range(START_YEAR, END_YEAR + 1):

        year_index = build_year_index(year)

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

        # ------------------------------------------------
        # FIX: ensure empty years still have correct schema
        # ------------------------------------------------

        if dfs:

            data = pd.concat(dfs)
            data = data[~data.index.duplicated(keep="first")]

        else:

            data = pd.DataFrame(columns=TARGET_COLUMNS)
            data.index = pd.DatetimeIndex([])

        data = enforce_schema(data)

        data = data.reindex(year_index)

        # Ensure correct column order and 2D structure
        data = data[TARGET_COLUMNS].copy()

        data.insert(0, "DATA", data.index.strftime("%Y-%m-%d"))
        data.insert(1, "HORA", data.index.strftime("%H:%M:%S"))

        data.reset_index(drop=True, inplace=True)

        data.to_csv(
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

def main():

    stations = collect_files()

    tasks = [(state, station, files) for (state, station), files in stations.items()]

    workers = max(cpu_count() - 2, 1)

    with Pool(workers) as pool:
        pool.map(process_station, tasks)


if __name__ == "__main__":
    main()
