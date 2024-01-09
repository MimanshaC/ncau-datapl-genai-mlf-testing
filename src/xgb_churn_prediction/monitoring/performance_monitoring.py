from typing import Dict
from typing import Union

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import ClassificationPreset
from evidently.report import Report

from xgb_churn_prediction.monitoring.alerting_threshold import (
    PERFORMANCE_MONITORING_THRESHOLDS,
)


def generate_evidently_report(
    processed_data: pd.DataFrame, target_column: str, series_id_column: str
) -> Report:
    """Compute performance metrics using evidently to be later visualised as an
    evidently Report. The `column_mapping.task` and `evidently.metric_preset` need
    to be updated based on the model type (e.g. classification would use the
    ClassificationPreset).

    Args:
        processed_data (pd.DataFrame): data containing ground truth and predicted values

    Returns:
        Report: Evidently report
    """
    # format report structure
    column_mapping = ColumnMapping()
    column_mapping.task = "classification"
    column_mapping.target = target_column
    column_mapping.prediction = "prediction"
    column_mapping.id = series_id_column
    column_mapping.datetime = "timestamp"

    # create evidently report and generate model evaluation metrics
    report = Report(
        metrics=[
            ClassificationPreset(),
        ],
    )
    report.run(reference_data=None, current_data=processed_data, column_mapping=column_mapping)
    return report


def extract_metrics(report: Report) -> Dict[str, Union[float, bool]]:
    """Extracts and formats the metrics that are in the evidently report. Generates
    the boolean flag that will have a policy defined on it. This column will be
    determined by stepping through the `ALERTING_THRESHOLDS` defined in
    `alerting_threshold.py`.

    Args:
        report (Report): evidently report using ClassificationPreset

    Returns:
        Dict[str, float]: dictionary of metrics
    """
    # format metrics dictionary from report
    metrics = report.as_dict()["metrics"][0]["result"]["current"]
    metrics.pop("underperformance", "no key error")
    metrics = {k: v for k, v in metrics.items() if v is not None}

    # Generate alert column comparing calculated metrics to thresholds
    metrics["performance_alert"] = False
    for threshold in PERFORMANCE_MONITORING_THRESHOLDS:
        if threshold.compare(metrics[threshold.metric_name]):
            metrics["performance_alert"] = True

    return metrics


def process_data(
    ground_truth_data: pd.DataFrame,
    predictions_data: pd.DataFrame,
    target_column: str,
    timestamp_column: str,
    series_id_column: str,
) -> pd.DataFrame:
    """
    Process and merge two dataframes containing ground truth and prediction data.

    Args:
        ground_truth_data (pd.DataFrame): DataFrame containing ground truth data.
        predictions_data (pd.DataFrame): DataFrame containing prediction data.
        target_column (str): Name of the target column in the dataframes.
        timestamp_column (str): Name of the timestamp column in the dataframes.
        series_id_column (str): Name of the series ID column in the dataframes.

    Returns:
        pd.DataFrame: Merged DataFrame containing filtered and merged data.
    """
    ground_truth_data[target_column] = ground_truth_data[target_column].astype(int)
    predictions_data.prediction = predictions_data.prediction.astype(int)
    predictions_data = predictions_data.rename(columns={"series_id": series_id_column})

    # merge the two filtered df together
    return pd.merge(ground_truth_data, predictions_data, how="inner", on=[series_id_column])
