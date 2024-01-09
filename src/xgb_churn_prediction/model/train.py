import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from .features import Featurizer

# Define hyperparameters
HYPERPARAMETERS = {"test": "x"}


def train_model(train_data_x: pd.DataFrame, train_data_y: pd.Series) -> Pipeline:
    """Function to train a model

    Args:
        train_data_x (pd.DataFrame): Training data features
        train_data_y (pd.DataFrame): Training data targets

    Returns:
        Pipeline: Scikit learn pipeline incl preprocessing and model to use for predictions
    """

    # define type of model to train
    model = RandomForestClassifier(**HYPERPARAMETERS)

    # Define the preprocessing pipeline
    preprocessing_pipeline = Pipeline(
        [
            ("generate_features", Featurizer()),
            # Add additional preprocessing steps here
        ]
    )

    full_pipeline = Pipeline([("preprocessing", preprocessing_pipeline), ("model", model)])

    # Fit the full pipeline on the training data
    full_pipeline.fit(train_data_x, train_data_y)

    return full_pipeline
