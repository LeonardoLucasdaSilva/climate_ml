import re
from pathlib import Path

from src.config.paths import INTERIM_DATA_DIR

INMET_DIR = INTERIM_DATA_DIR / "inmet"

# Header replacements
replacements = {
    "ESTAÇÃO": "ESTACAO",
    "ESTAÇ?O": "ESTACAO",
    "ESTAC?O": "ESTACAO",
    "REGIÃO": "REGIAO",
    "REGI?O": "REGIAO",
    "DATA DE FUNDAÇÃO (YYYY-MM-DD)": "DATA DE FUNDACAO",
    "DATA DE FUNDAC?O": "DATA DE FUNDACAO",
    "DATA (YYYY-MM-DD)": "DATA",
    "Data": "DATA",
    "HORA (UTC)": "HORA",
    "Hora UTC": "HORA",
    "PRECIPITAÇÃO TOTAL": "PRECIPITACAO_TOTAL",
    "HORÁRIO (mm)": "HORARIO",
    "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)": "PRESSAO",
    "PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)": "PRESSAO_MIN",
    "PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)": "PRESSAO_MAX",
    "RADIACAO GLOBAL (KJ/m²)": "RADIACAO",
    "RADIACAO GLOBAL (Kj/m²)": "RADIACAO",
    "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)": "TEMPERATURA",
    "TEMPERATURA DO PONTO DE ORVALHO (°C)": "PONTO_ORVALHO",
    "TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)": "TEMPERATURA_MAXIMA",
    "TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)": "TEMPERATURA_MIN",
    "TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)": "PONTO_ORVALHO_MAX",
    "TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)": "PONTO_ORVALHO_MIN",
    "UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)": "UMIDADE_MAX",
    "UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)": "UMIDADE_MIN",
    "UMIDADE RELATIVA DO AR, HORARIA (%)": "UMIDADE",
    "VENTO, DIREÇÃO HORARIA (gr) (° (gr))": "DIRECAO_VENTO",
    "VENTO, RAJADA MAXIMA (m/s)": "RAJADA_VENTO",
    "VENTO, VELOCIDADE HORARIA (m/s)": "VELOCIDADE_VENTO",
    "BEBDOURO": "BEBEDOURO",
    "SAO LUIS DO PARAITINGA": "SAO LUIZ DO PARAITINGA",
    "FORTE DE COPACABANA": "RIO DE JANEIRO - FORTE DE COPACABANA",
    "VILA MILITAR": "RIO DE JANEIRO - VILA MILITAR",
    "PARATI": "PARATY",
    "MARAMBAIA": "RIO DE JANEIRO-MARAMBAIA",
    "ECOLOGIA AGRICOLA": "SEROPEDICA-ECOLOGIA AGRICOLA",
    "JANAUBA": "NOVA PORTEIRINHA (JANAUBA)",
    "PAMPULHA": "BELO HORIZONTE (PAMPULHA)",
    "BRASNORTE (MUNDO NOVO)": "BRASNORTE (NOVO MUNDO)",
    "PRES. KENNEDY": "PRESIDENTE KENNEDY",
    "FORMOSO DO RIO PRETO": "FORMOSA DO RIO PRETO",
    "LUIZ EDUARDO MAGALHAES": "LUIS EDUARDO MAGALHAES",
    "PORTO ALEGRE": "PORTO ALEGRE - JARDIM BOTANICO",
    "SAQUAREMA": "SAQUAREMA - SAMPAIO CORREA",
    "SAQUAREMA - SAMPAIO CORREIA": "SAQUAREMA - SAMPAIO CORREA",
    "NOVA FRIBURGO": "NOVA FRIBURGO - SALINAS",
    "SAO TOME": "CAMPOS DOS GOYTACAZES - SAO TOME",
    "BELO HORIZONTE (PAMPULHA)": "BELO HORIZONTE - PAMPULHA",
    "XEREM": "DUQUE DE CAXIAS - XEREM",
    "CAMPOS": "CAMPOS DOS GOYTACAZES",
    "TERESOPOLIS": "TERESOPOLIS-PARQUE NACIONAL",

}

# Date patterns
date_yyyy_mm_dd_slash = re.compile(r"\b(\d{4})/(\d{2})/(\d{2})\b")
date_dd_mm_yyyy = re.compile(r"\b(\d{2})/(\d{2})/(\d{4})\b")
date_dd_mm_yy = re.compile(r"\b(\d{2})/(\d{2})/(\d{2})\b")

# Detect first data line
data_line_pattern = re.compile(r"\s*\d{4}[-/]\d{2}[-/]\d{2};")


def normalize_header_names(line: str):

    line = line.rstrip("\n")

    # Metadata lines like KEY:;VALUE
    if ":;" in line:
        key, value = line.split(":;", 1)

        key = replacements.get(key.strip(), key.strip())
        value = replacements.get(value.strip(), value.strip())

        return f"{key}:;{value}\n"

    # Column header line
    for old, new in replacements.items():
        line = line.replace(old, new)

    return line + "\n"


def convert_dates(text: str):
    # YYYY/MM/DD -> YYYY-MM-DD
    text = date_yyyy_mm_dd_slash.sub(r"\1-\2-\3", text)

    # DD/MM/YYYY -> YYYY-MM-DD
    text = date_dd_mm_yyyy.sub(lambda m: f"{m.group(3)}-{m.group(2)}-{m.group(1)}", text)

    # DD/MM/YY -> YYYY-MM-DD
    def repl_short(m):
        d, mth, y = m.groups()
        y = int(y)
        y = 2000 + y if y < 50 else 1900 + y
        return f"{y:04d}-{mth}-{d}"

    text = date_dd_mm_yy.sub(repl_short, text)

    return text


def normalize_line(line: str, header=False):

    if header:
        line = normalize_header_names(line)

    line = convert_dates(line)

    return line


for file in INMET_DIR.rglob("*.CSV"):

    changed = False
    new_lines = []

    with open(file, "r", encoding="latin-1") as f:

        header = True

        for line in f:

            if header and data_line_pattern.match(line):
                header = False

            new_line = normalize_line(line, header)

            if new_line != line:
                changed = True

            new_lines.append(new_line)

    if changed:
        print(f"Updated → {file}")

        with open(file, "w", encoding="latin-1") as f:
            f.writelines(new_lines)

print("Normalization finished.")