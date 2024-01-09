# script for splitting data into train/test datasets
from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class SplitDataResult:
    """A data class representing the result of splitting a dataset into training and testing data.

    Attributes:
        train_data (pd.DataFrame): The training data subset.
        test_data (pd.DataFrame): The testing data subset.
    """

    train_data: pd.DataFrame
    test_data: pd.DataFrame


def split_data(training_data: pd.DataFrame) -> SplitDataResult:
    """Function to split data into train and test

    Args:
        training_data (pd.DataFrame): dataset to split

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: _description_
    """
    # TODO define custom splitting logic
    train_data, test_data = train_test_split(training_data, test_size=0.2, random_state=42)

    return SplitDataResult(train_data=train_data, test_data=test_data)


def split_X_y(data: pd.DataFrame, label: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Function to split dataset into X and y

    Args:
        data (pd.DataFrame): dataset to split
        label (str): key for label

    Returns:
        Tuple[pd.DataFrame, pd.Series]: X, y values
    """
    X = data.drop(label, axis=1)
    y = data[label]
    return X, y
