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

def days_after(date_str: str, days: int, fmt: str = "%Y-%m-%d") -> str:
    """
    Return the date that is a given number of days after the input date.

    Parameters
    ----------
    date_str : str
        Input date as a string (e.g., "2020-01-01").
    days : int
        Number of days to add.
    fmt : str, optional
        Date format string (default is "%Y-%m-%d").

    Returns
    -------
    str
        The resulting date as a string in the same format.
    """
    date_obj = datetime.strptime(date_str, fmt)
    new_date = date_obj + timedelta(days=days)
    return new_date.strftime(fmt)

def days_between(
    start_date: str,
    end_date: str,
    fmt: str = "%Y-%m-%d"
) -> int:
    """
    Return the number of days between two dates.

    Parameters
    ----------
    start_date : str
        Start date as a string (e.g., "2020-01-01").
    end_date : str
        End date as a string (e.g., "2020-01-10").
    fmt : str, optional
        Date format string (default is "%Y-%m-%d").

    Returns
    -------
    int
        Number of days between the two dates.
        Positive if end_date is after start_date,
        negative if end_date is before start_date.
    """
    start_obj = datetime.strptime(start_date, fmt)
    end_obj = datetime.strptime(end_date, fmt)

    delta = end_obj - start_obj
    return delta.days