import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class Featurizer(BaseEstimator, TransformerMixin):
    """Class for all feature engineering functions"""

    def __init__(self) -> None:
        """Initializes a new instance of Featurizer with the specified arguments.
        If arguments are specfified they need to be the same across model lifecycle.
        """

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "Featurizer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Function to transform dataest and create features

        Args:
            X (pd.DataFrame): dataset to transform

        Returns:
            pd.DataFrame: transformed dataframe
        """
        # run transformations on dataset

        # Sort the feature column order to ensure the input order at inference time
        # does not matter
        X.sort_index(axis=1, inplace=True)
        features_abc = self.generate_features_abc(X)

        transformed_data = pd.concat([features_abc, X], axis=1)

        # Sort the transformed column order to ensure the input order at inference time
        # does not matter
        transformed_data.sort_index(axis=1, inplace=True)

        return transformed_data

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def generate_features_abc(self, data: pd.DataFrame) -> pd.DataFrame:
        """Custom function to generate features

        Args:
            data (pd.DataFrame): dataset to create features for

        Returns:
            pd.DataFrame: generated features X
        """

        # TODO define specific feature generation functions here to isolate for better testing
        return pd.DataFrame()
