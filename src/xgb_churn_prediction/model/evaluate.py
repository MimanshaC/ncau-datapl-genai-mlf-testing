# script for model evaluation
from typing import Dict

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from ..data.data_split import split_X_y


def evaluate_model(
    test_dataset: pd.DataFrame, trained_model: Pipeline, key: str
) -> Dict[str, float]:
    """Evaluate the predictions of the model against the true values of a test dataset.

    Args:
        test_dataset (pd.DataFrame): test dataset to run evaluation on
        trained_model_path (str): trained model to run evaluation on

    Returns:
        Dict[str, float]: metrics dict with results
    """

    # Split into X and y
    test_X, test_y = split_X_y(test_dataset, key)

    # Generate predictions
    y_hat = trained_model.predict(test_X)

    # Calculate desired metrics
    clfc_report = classification_report(test_y, y_hat, output_dict=True)
    accuracy = clfc_report["accuracy"]
    precision = clfc_report["weighted avg"]["precision"]
    recall = clfc_report["weighted avg"]["recall"]
    f1score = clfc_report["weighted avg"]["f1-score"]

    # store metrics in dict
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1score": float(f1score),
    }

    return metrics


def champion_challenger(
    metrics_champion: Dict, metrics_challenger: Dict, metrics_dict: Dict
) -> bool:
    """Function to compare to metrics of challenger and champion model.

    Args:
        metrics_champion (Dict): metrics as dict of champion model
        metrics_challenger (Dict): metrics as dict of challenger model
        metrics_dict (Dict): dict of metrics and their operator to evaluate challenger vs. champion
            e.g. metrics_dict = {"meanAbsoluteError": operator.le, "accuracy": operator.ge}

    Returns:
        bool: indication if model has improved or not
    """

    for metric_name, operator_func in metrics_dict.items():
        if operator_func(metrics_challenger[metric_name], metrics_champion[metric_name]):
            return True

    return False
