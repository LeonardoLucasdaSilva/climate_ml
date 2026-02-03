from src.data.load import load_raw,load_processed,load_interim

try:
    path = load_raw("fake_data.nc")
except FileNotFoundError as e:
    print("OK",e)

try:
    path = load_interim("fake_data.nc")
except FileNotFoundError as e:
    print("OK",e)

try:
    path = load_processed("fake_data.nc")
except FileNotFoundError as e:
    print("OK",e)

