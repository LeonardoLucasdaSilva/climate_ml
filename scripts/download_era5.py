from src.data import download_era5

# This array contains the requests to get the datasets we want from the ERA5 API

all_years_2 = [
                "1979", "1980", "1981",
                "1982", "1983", "1984",
                "1985", "1986", "1987",
                "1988", "1989", "1990",
                "1991", "1992", "1993",
                "1994", "1995", "1996",
                "1997", "1998", "1999",
                "2000", "2001", "2002",
                "2003", "2004", "2005",
                "2006", "2007", "2008",
                "2009", "2010", "2011",
                "2012", "2013", "2014",
                "2015", "2016", "2017",
                "2018", "2019", "2020",
                "2021", "2022", "2023",
                "2024", "2025", "2026"
            ]

all_years = [
                "1988", "1989", "1990",
                "1991", "1992", "1993",
                "1994", "1995", "1996",
                "1997", "1998", "1999",
                "2000", "2001", "2002",
                "2003", "2004", "2005",
                "2006", "2007", "2008",
                "2009", "2010", "2011",
                "2012", "2013", "2014",
                "2015", "2016", "2017",
                "2018", "2019", "2020",
                "2021", "2022", "2023",
                "2024", "2025", "2026"
            ]

DATASETS = []


for year in all_years:
    requisition = {
        "name": "downward_uv_radiation_at_the_surface_"+year,
        "dataset": "reanalysis-era5-single-levels",
        "filename": "downward_uv_radiation_at_the_surface_"+year+".grib",
        "request": {
            "product_type": ["reanalysis"],
            "variable": ["downward_uv_radiation_at_the_surface"],
            "year": year,
            "month": [
                "01", "02", "03",
                "04", "05", "06",
                "07", "08", "09",
                "10", "11", "12"
            ],
            "day": [
                "01", "02", "03",
                "04", "05", "06",
                "07", "08", "09",
                "10", "11", "12",
                "13", "14", "15",
                "16", "17", "18",
                "19", "20", "21",
                "22", "23", "24",
                "25", "26", "27",
                "28", "29", "30",
                "31"
            ],
            "time": [
                "00:00", "01:00", "02:00",
                "03:00", "04:00", "05:00",
                "06:00", "07:00", "08:00",
                "09:00", "10:00", "11:00",
                "12:00", "13:00", "14:00",
                "15:00", "16:00", "17:00",
                "18:00", "19:00", "20:00",
                "21:00", "22:00", "23:00"
            ],
            "data_format": "grib",
            "grid": '0.25/0.25',
            "download_format": "unarchived",
            "area": [-25, -60, -35, -45]
        }
    }

    DATASETS.append(requisition)


def main():
    for spec in DATASETS:
        print(f"Downloading {spec['name']}")

        download_era5(
            dataset=spec["dataset"],
            request=spec["request"],
            filename=spec["filename"],
        )

if __name__ == "__main__":
    main()



