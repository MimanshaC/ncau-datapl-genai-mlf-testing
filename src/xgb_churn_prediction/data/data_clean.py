# script for cleaning data after it is ingested
import pandas as pd


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Function to clean data
    Args:
        data (pd.DataFrame): dataset to clean

    Returns:
        pd.DataFrame: cleaned dataset
    """

    # TODO run your data cleaning functions here
    data = fill_missing_data(data)

    return data


def fill_missing_data(data: pd.DataFrame) -> pd.DataFrame:
    """Function to fill missing datapoints in dataset with default values

    Args:
        data (pd.DataFrame): dataset to be filled with missing data points

    Returns:
        pd.DataFrame: dataset with filled missing datapoints
    """

    # add logic to fill missing data points in the dataset

    return data


# TODO add any other functions required for data cleansing
