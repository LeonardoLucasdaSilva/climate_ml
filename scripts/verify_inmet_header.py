import re
import difflib
from pathlib import Path
import math

from src.config.paths import INTERIM_DATA_DIR


INMET_DIR = INTERIM_DATA_DIR / "inmet"
LOG_DIR = INTERIM_DATA_DIR / "inmet_logs"

LOG_DIR.mkdir(exist_ok=True)

date_pattern = re.compile(r"\d{4}[-/]\d{2}[-/]\d{2};")
year_pattern = re.compile(r"\d{4}")

IGNORE_FIELDS = ("LATITUDE", "LONGITUDE", "ALTITUDE")

# km threshold for suspicious station movement
DIST_THRESHOLD_KM = 10


def read_header(file_path: Path):
    header_lines = []

    with open(file_path, "r", encoding="latin-1") as f:
        for line in f:
            if date_pattern.match(line):
                break
            header_lines.append(line.rstrip())

    return header_lines


def filter_ignored(lines):

    filtered = []

    for line in lines:
        if line.startswith(IGNORE_FIELDS):
            continue
        filtered.append(line)

    return filtered


def parse_number(x):

    if x is None:
        return None

    x = x.replace(",", ".").strip()

    try:
        return float(x)
    except:
        return None


def extract_geo(header_lines):

    lat = None
    lon = None
    alt = None

    for line in header_lines:

        if line.startswith("LATITUDE"):
            lat = parse_number(line.split(";")[1])

        elif line.startswith("LONGITUDE"):
            lon = parse_number(line.split(";")[1])

        elif line.startswith("ALTITUDE"):
            alt = parse_number(line.split(";")[1])

    return lat, lon, alt


def extract_year(path: Path):

    m = year_pattern.search(path.name)

    if m:
        return int(m.group())

    return None


def haversine(lat1, lon1, lat2, lon2):

    R = 6371

    lat1, lon1, lat2, lon2 = map(
        math.radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


for state_dir in INMET_DIR.iterdir():

    if not state_dir.is_dir():
        continue

    DIFF_LOG = LOG_DIR / f"{state_dir.name}_header_differences.txt"
    COORD_LOG = LOG_DIR / f"{state_dir.name}_station_coordinates_intervals.txt"

    problems = 0

    big_changes = []

    station_intervals = []

    with open(DIFF_LOG, "w", encoding="utf-8") as diff_log:

        for station_dir in state_dir.iterdir():

            if not station_dir.is_dir():
                continue

            files = list(station_dir.glob("*.CSV"))

            if len(files) == 0:
                continue

            files.sort(key=lambda f: extract_year(f) or 0)

            reference_file = files[0]
            reference_header = read_header(reference_file)
            reference_filtered = filter_ignored(reference_header)

            coord_periods = {}

            for file in files:

                header = read_header(file)

                lat, lon, alt = extract_geo(header)

                year = extract_year(file)

                key = (lat, lon, alt)

                if key not in coord_periods:
                    coord_periods[key] = [year, year]
                else:
                    coord_periods[key][1] = year

            # ---- detect large movements WITH YEAR ----

            ordered_coords = sorted(
                coord_periods.items(),
                key=lambda x: x[1][0] if x[1][0] is not None else -1
            )

            for i in range(len(ordered_coords) - 1):

                (lat1, lon1, alt1), (start1, end1) = ordered_coords[i]
                (lat2, lon2, alt2), (start2, end2) = ordered_coords[i + 1]

                if None in (lat1, lon1, lat2, lon2):
                    continue

                dist = haversine(lat1, lon1, lat2, lon2)

                if dist > DIST_THRESHOLD_KM:

                    big_changes.append(
                        (
                            state_dir.name,
                            station_dir.name,
                            dist,
                            end1,
                            start2,
                            (lat1, lon1, alt1),
                            (lat2, lon2, alt2),
                        )
                    )

            text_block = (
                f"\n====================================\n"
                f"STATE   : {state_dir.name}\n"
                f"STATION : {station_dir.name}\n"
            )

            for (lat, lon, alt), (start, end) in coord_periods.items():

                text_block += (
                    f"{start} → {end} | "
                    f"LAT={lat}  LON={lon}  ALT={alt}\n"
                )

            text_block += "------------------------------------\n"

            station_intervals.append(text_block)

            # ---- header comparison ----

            for file in files[1:]:

                header = read_header(file)
                header_filtered = filter_ignored(header)

                if header_filtered != reference_filtered:

                    problems += 1

                    header_text = (
                        "\n==============================\n"
                        f"STATE   : {state_dir.name}\n"
                        f"STATION : {station_dir.name}\n"
                        f"FILE    : {file.name}\n"
                        f"REF     : {reference_file.name}\n"
                        "---------- HEADER DIFF ----------\n"
                    )

                    print(header_text)
                    diff_log.write(header_text)

                    diff = difflib.unified_diff(
                        reference_filtered,
                        header_filtered,
                        fromfile=reference_file.name,
                        tofile=file.name,
                        lineterm=""
                    )

                    for line in diff:
                        print(line)
                        diff_log.write(line + "\n")

        if problems == 0:
            msg = "All station folders have identical headers (ignoring coordinates)."
        else:
            msg = f"Found {problems} files with header differences."

        print(f"{state_dir.name}: {msg}")
        diff_log.write(msg + "\n")

    # ---- write coordinate file ----

    with open(COORD_LOG, "w", encoding="utf-8") as coord_log:

        coord_log.write(
            "#############################################\n"
            "# LARGE GEOLOCATION CHANGES\n"
            "#############################################\n\n"
        )

        for state, station, dist, end1, start2, c1, c2 in sorted(
            big_changes, key=lambda x: -x[2]
        ):

            coord_log.write(
                f"{state} | {station}\n"
                f"CHANGE: {end1} → {start2}\n"
                f"DISTANCE: {dist:.2f} km\n"
                f"{c1} -> {c2}\n\n"
            )

        coord_log.write(
            "\n\n#############################################\n"
            "# FULL STATION INTERVALS\n"
            "#############################################\n"
        )

        for block in station_intervals:
            coord_log.write(block)

    print(f"\nSaved logs for {state_dir.name} in {LOG_DIR}")
