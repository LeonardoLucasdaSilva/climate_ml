import zipfile
import shutil
from pathlib import Path

from src.config.paths import RAW_DATA_DIR, INTERIM_DATA_DIR
from src.utils.files import ensure_dir


RAW_INMET_DIR = RAW_DATA_DIR / "inmet"
INTERIM_INMET_DIR = INTERIM_DATA_DIR / "inmet"

TMP_DIR = INTERIM_DATA_DIR / "_tmp_inmet"
ensure_dir(TMP_DIR)


state_map = {
    "AC": "acre",
    "AL": "alagoas",
    "AP": "amapa",
    "AM": "amazonas",
    "BA": "bahia",
    "CE": "ceara",
    "DF": "distrito_federal",
    "ES": "espirito_santo",
    "GO": "goias",
    "MA": "maranhao",
    "MT": "mato_grosso",
    "MS": "mato_grosso_do_sul",
    "MG": "minas_gerais",
    "PA": "para",
    "PB": "paraiba",
    "PR": "parana",
    "PE": "pernambuco",
    "PI": "piaui",
    "RJ": "rio_de_janeiro",
    "RN": "rio_grande_do_norte",
    "RS": "rio_grande_do_sul",
    "RO": "rondonia",
    "RR": "roraima",
    "SC": "santa_catarina",
    "SP": "sao_paulo",
    "SE": "sergipe",
    "TO": "tocantins",
}


def parse_filename(name: str):
    """
    INMET_REGION_STATE_STATION_...
    returns (state, station)
    """
    parts = name.replace(".CSV", "").split("_")

    if len(parts) < 4:
        return None, None

    region = parts[1]
    uf = parts[2]
    station = parts[3]

    return uf, station


for zip_path in RAW_INMET_DIR.glob("*.zip"):

    print(f"\nProcessing {zip_path.name}")

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(TMP_DIR)

    for csv_file in TMP_DIR.rglob("*.CSV"):

        uf, station = parse_filename(csv_file.name)

        if uf is None:
            print(
                f"[PARSE ERROR] Could not parse structure → {csv_file.name}"
            )
            continue

        state = state_map.get(uf)

        if state is None:
            print(
                f"[STATE ERROR] UF '{uf}' not recognized → {csv_file.name}"
            )
            continue

        state_dir = INTERIM_INMET_DIR / state
        station_dir = state_dir / station

        try:
            ensure_dir(station_dir)

            target_path = station_dir / csv_file.name

            shutil.move(csv_file, target_path)

        except Exception as e:
            print(
                f"[MOVE ERROR]\n"
                f"file: {csv_file.name}\n"
                f"state: {state}\n"
                f"station: {station}\n"
                f"error: {e}"
            )

    shutil.rmtree(TMP_DIR)
    ensure_dir(TMP_DIR)


print("Finished organizing INMET data.")