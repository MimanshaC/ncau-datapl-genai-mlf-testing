import warnings
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import pandas as pd
from numba import NumbaDeprecationWarning

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
    from evidently import ColumnMapping
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

# TODO: link to in docs
# https://www.evidentlyai.com/blog/ml-monitoring-do-i-need-data-drift
# https://www.evidentlyai.com/blog/data-and-prediction-drift


def create_report(
    latest_df: pd.DataFrame, hist_df: pd.DataFrame, numerical_features: List[str]
) -> Tuple[Report, Dict[str, Union[float, bool]], Dict[str, Union[float, bool]]]:
    """Evaluation of drift between latest and historical features.
    Produces an Evidently report and some metrics for storage and alerting.

    Args:
        latest_df (pd.DataFrame): current inference data
        hist_df (pd.DataFrame): training data
        numerical_features (List[str]): features to generate drift report for

    Returns:
        Tuple[Report, Dict[str, Union[float, bool]], Dict[str, Union[float, bool]]]
          report, metrics and metrics to add to montoring table
    """
    column_mapping = ColumnMapping()
    column_mapping.numerical_features = numerical_features
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=hist_df,
        current_data=latest_df,
        column_mapping=column_mapping,
    )
    report_dict = report.as_dict()
    metrics = report_dict["metrics"]

    drift_metric = next(metric for metric in metrics if metric["metric"] == "DataDriftTable")
    cols = drift_metric["result"]["drift_by_columns"]
    metrics = {
        "drift_detected": drift_metric["result"]["dataset_drift"],
        "drift_score": drift_metric["result"]["share_of_drifted_columns"],
        "number_of_drifted_columns": drift_metric["result"]["number_of_drifted_columns"],
        "number_of_columns": drift_metric["result"]["number_of_columns"],
    }
    for col_name, col_info in cols.items():
        metrics[f"{col_name}__drift_score"] = col_info["drift_score"]
        metrics[f"{col_name}__drift_detected"] = col_info["drift_detected"]

    monitoring_metrics = {
        "drift_detected": drift_metric["result"]["dataset_drift"],
        "drift_score": drift_metric["result"]["share_of_drifted_columns"],
    }
    return report, metrics, monitoring_metrics
