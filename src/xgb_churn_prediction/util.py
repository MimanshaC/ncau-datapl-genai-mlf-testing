import datetime
from pathlib import Path
from typing import Tuple


def get_resource_folder() -> Path:
    """Function to get the resource folder

    Args:
        None

    Returns:
        Path: Path to resource folder
    """
    return Path(__file__).parent.joinpath("resources")


def read_sql_file(path: Path) -> str:
    """Function to load and check sql code from .sql file

    Args:
        path (Path): path to file

    Returns:
        str: Loaded query
    """
    with open(path, mode="r", encoding="utf-8-sig") as file:
        query = file.read()

    # TODO: check query for correctness
    return query


def calulate_dates(days: int) -> Tuple[str, str]:
    """Function to calculate dates needed for training.
    Calcualtes today - one day and today - one day + however many days stated as parameter

    Args:
        days (int): how many days to subtract from today

    Returns:
        Tuple[str, str]: calculated dates as strings in the specified format
    """
    yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
    date_x_days_ago = yesterday - datetime.timedelta(days=days)
    yesterday_str = yesterday.strftime("%Y-%m-%d")
    date_x_days_ago_str = date_x_days_ago.strftime("%Y-%m-%d")

    return date_x_days_ago_str, yesterday_str
