from datetime import datetime, timedelta


def days_before(date_str: str, days: int, fmt: str = "%Y-%m-%d") -> str:
    """
    Return the date that is a given number of days before the input date.

    Parameters
    ----------
    date_str : str
        Input date as a string (e.g., "2020-01-01").
    days : int
        Number of days to subtract.
    fmt : str, optional
        Date format string (default is "%Y-%m-%d").

    Returns
    -------
    str
        The resulting date as a string in the same format.
    """
    date_obj = datetime.strptime(date_str, fmt)
    new_date = date_obj - timedelta(days=days)
    return new_date.strftime(fmt)