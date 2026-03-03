
"""
check_inmet_completeness.py

For every processed INMET station CSV, find all contiguous date intervals
where the data is fully complete (no NaN and no -9999 sentinel values).

Output is printed to stdout and also saved as a CSV summary.
"""

import pandas as pd
from pathlib import Path

from src.config.paths import PROCESSED_DATA_DIR

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INMET_DIR = PROCESSED_DATA_DIR / "inmet"
OUTPUT_PATH = PROCESSED_DATA_DIR / "inmet_complete_intervals.csv"

NO_DATA_SENTINEL = -9999

# Columns to check for completeness (all climate variables, not the date col)

# ALL POSSIBLE INMET COLUMNS

# COLUMNS_TO_CHECK = [
#     "PRECIPITACAO_TOTAL",
#     "PRESSAO",
#     "PRESSAO_MIN",
#     "PRESSAO_MAX",
#     "RADIACAO",
#     "TEMPERATURA",
#     "PONTO_ORVALHO",
#     "TEMPERATURA_MAXIMA",
#     "TEMPERATURA_MIN",
#     "PONTO_ORVALHO_MAX",
#     "PONTO_ORVALHO_MIN",
#     "UMIDADE_MAX",
#     "UMIDADE_MIN",
#     "UMIDADE",
#     "DIRECAO_VENTO",
#     "RAJADA_VENTO",
#     "VELOCIDADE_VENTO",
# ]

COLUMNS_TO_CHECK = [
    "PRECIPITACAO_TOTAL"
]

# Control variables
STATE_TO_CHECK = "RS"  # Change to selected state or None for all states
START_DATE = "2020-01-01"  # Initial interval date (formato: AAAA-MM-DD)
END_DATE = "2024-06-30"  # End interval date (formato: AAAA-MM-DD)

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def find_complete_intervals(df: pd.DataFrame) -> list[dict]:
    """
    Dado um DataFrame diário com um DatetimeIndex, retorna uma lista de
    intervalos contíguos onde cada linha está completa (sem NaN ou -9999).

    Retorna
    -------
    lista de dicionários com chaves: start, end, n_days
    """

    cols = [c for c in COLUMNS_TO_CHECK if c in df.columns]
    df = df[cols].replace(NO_DATA_SENTINEL, float("nan"))
    complete_mask = df.notna().all(axis=1)

    intervals = []
    in_interval = False
    start = None

    for date, is_complete in complete_mask.items():
        if is_complete and not in_interval:
            start = date
            in_interval = True
        elif not is_complete and in_interval:
            end = date - pd.Timedelta(days=1)
            intervals.append({"start": start, "end": end, "n_days": (end - start).days + 1})
            in_interval = False

    if in_interval:
        end = complete_mask.index[-1]
        intervals.append({"start": start, "end": end, "n_days": (end - start).days + 1})

    return intervals


def load_station_csv(path: Path) -> pd.DataFrame:
    """Carrega um CSV diário processado do INMET com um DatetimeIndex adequado."""
    df = pd.read_csv(path, sep=";", low_memory=False)
    df["date"] = pd.to_datetime(df["DATA"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()

    # Filtra pelo intervalo de datas especificado
    if START_DATE or END_DATE:
        df = df.loc[START_DATE:END_DATE]

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    all_records = []
    fully_covered_stations = []

    station_files = sorted(INMET_DIR.rglob("*_2000_2025_daily.csv"))

    if not station_files:
        print(f"Nenhum arquivo de estação encontrado em {INMET_DIR}")
        return

    print(f"Encontrado(s) {len(station_files)} arquivo(s) de estação.\n")

    for csv_path in station_files:

        # Deriva o estado e a estação a partir da estrutura do caminho
        station = csv_path.parent.name
        state = csv_path.parent.parent.name

        # Filtra pelo estado especificado
        if STATE_TO_CHECK and state != STATE_TO_CHECK:
            continue

        try:
            df = load_station_csv(csv_path)
        except Exception as e:
            print(f"[ERRO] {state}/{station}: {e}")
            continue

        intervals = find_complete_intervals(df)

        if not intervals:
            print(f"  [{state}/{station}] — Nenhum intervalo completo encontrado.")
        else:
            print(f"  [{state}/{station}] — {len(intervals)} intervalo(s) completo(s):")
            for iv in intervals:
                print(f"      {iv['start'].date()} → {iv['end'].date()}  ({iv['n_days']} dias)")

        for iv in intervals:
            all_records.append({
                "state": state,
                "station": station,
                "start": iv["start"].date(),
                "end": iv["end"].date(),
                "n_days": iv["n_days"],
            })

            # Verifica se o intervalo cobre completamente o intervalo especificado
            if iv["start"].date() == pd.to_datetime(START_DATE).date() and iv["end"].date() == pd.to_datetime(END_DATE).date():
                fully_covered_stations.append(f"{state}/{station}")

    print()

    if all_records:
        summary = pd.DataFrame(all_records)
        summary.to_csv(OUTPUT_PATH, index=False)
        print(f"Resumo salvo em: {OUTPUT_PATH}")
    else:
        print("Nenhum intervalo completo encontrado em nenhuma estação.")

    # Imprime as estações que cobrem completamente o intervalo
    if fully_covered_stations:
        print("\nEstações que cobrem completamente o intervalo especificado:")
        for station in fully_covered_stations:
            print(f"  {station}")
    else:
        print("\nNenhuma estação cobre completamente o intervalo especificado.")


if __name__ == "__main__":
    main()

