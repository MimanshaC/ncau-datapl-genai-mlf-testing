import json
from typing import Dict
from typing import Tuple
from typing import Union

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import TargetDriftPreset
from evidently.report import Report

# TODO: link to in docs
# https://www.evidentlyai.com/blog/ml-monitoring-do-i-need-data-drift
# https://www.evidentlyai.com/blog/data-and-prediction-drift


def evaluate(
    latest_inference: pd.DataFrame,
    historical_inference: pd.DataFrame,
) -> Tuple[Report, Dict[str, Union[float, bool]]]:
    """
    Evaluate drift between the latest and historical inferences.
    Produces an Evidently report and some metrics for storage and alerting.

    Args:
        latest_inference (pd.DataFrame): DataFrame containing the latest inference data.
        historical_inference (pd.DataFrame): DataFrame containing historical inference data.

    Returns:
        Tuple[Report, Dict[str, Union[float, bool]]]: A tuple containing an Evidently report
        and drift metrics.
    """
    # Create a column mapping for Evidently
    column_mapping = ColumnMapping()
    column_mapping.id = "series_id"
    column_mapping.task = "classification"
    column_mapping.prediction = "prediction"
    column_mapping.datetime = "timestamp"

    # Create an Evidently report with TargetDriftPreset
    report = Report(metrics=[TargetDriftPreset()])

    # Run the report to evaluate drift
    report.run(
        reference_data=historical_inference,
        current_data=latest_inference,
        column_mapping=column_mapping,
    )

    # Convert the report to JSON and extract drift metrics
    report_json = report.json()
    drift_metric = next(
        metric
        for metric in json.loads(report_json)["metrics"]
        if metric["metric"] == "ColumnDriftMetric"
    )

    # Extract drift score and drift detected information
    metrics = {
        "drift_score": drift_metric["result"]["drift_score"],
        "drift_detected": drift_metric["result"]["drift_detected"],
    }
    return (report, metrics)
