import re
import difflib
from pathlib import Path

from src.config.paths import INTERIM_DATA_DIR


INMET_DIR = INTERIM_DATA_DIR / "inmet"

date_pattern = re.compile(r"\d{4}[-/]\d{2}[-/]\d{2};")

IGNORE_FIELDS = ("LATITUDE", "LONGITUDE", "ALTITUDE")

def read_header(file_path: Path):
    header_lines = []

    with open(file_path, "r", encoding="latin-1") as f:
        for line in f:
            if date_pattern.match(line):
                break
            header_lines.append(line.rstrip())

    return header_lines


def filter_ignored(lines):
    """
    Remove lines that start with ignored metadata fields.
    """
    filtered = []
    for line in lines:
        if line.startswith(IGNORE_FIELDS):
            continue
        filtered.append(line)
    return filtered


problems = 0


for state_dir in INMET_DIR.iterdir():

    if not state_dir.is_dir():
        continue

    for station_dir in state_dir.iterdir():

        if not station_dir.is_dir():
            continue

        files = list(station_dir.glob("*.CSV"))

        if len(files) <= 1:
            continue

        reference_file = files[0]
        reference_header = read_header(reference_file)
        reference_filtered = filter_ignored(reference_header)

        for file in files[1:]:

            header = read_header(file)
            header_filtered = filter_ignored(header)

            if header_filtered != reference_filtered:

                problems += 1

                print("\n==============================")
                print(f"STATE   : {state_dir.name}")
                print(f"STATION : {station_dir.name}")
                print(f"FILE    : {file.name}")
                print(f"REF     : {reference_file.name}")
                print("---------- HEADER DIFF ----------")

                diff = difflib.unified_diff(
                    reference_filtered,
                    header_filtered,
                    fromfile=reference_file.name,
                    tofile=file.name,
                    lineterm=""
                )

                for line in diff:
                    print(line)


if problems == 0:
    print("All station folders have identical headers (ignoring lat/long/alt).")
else:
    print(f"\nFound {problems} files with header differences.")