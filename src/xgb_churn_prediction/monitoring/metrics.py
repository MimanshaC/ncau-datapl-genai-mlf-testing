import logging
from datetime import datetime
from typing import Dict
from typing import Optional
from typing import Union

import google.api_core.exceptions
import numpy as np
from google.cloud import bigquery
from google.cloud import monitoring_v3


def python_type_to_bq_type(input_type: type, default: Optional[str] = None) -> str:
    type_map = {int: "INT64", float: "FLOAT64", bool: "BOOL", str: "STRING", np.float64: "FLOAT64"}
    return_val = type_map.get(input_type)
    if return_val is not None:
        return return_val
    if default is None:
        raise ValueError(f"No matching BQ type for {input_type} and no default given")
    return default


def write_metrics_to_table(
    project: str,
    dataset: str,
    bq_location: str,
    table_name: str,
    model_name: str,
    model_version: str,
    timestamp: datetime,
    metrics: Dict[str, Union[float, bool]],
) -> None:
    """
    Write a dictionary of metrics to a table, creating or adding columns to the table if required.
    The type of a metric cannot change; this will result in an error (columns are not coerced or
    otherwise change type). model_name and timestamp are stored also.

    Args:
        project (str): The Google Cloud project ID.
        dataset (str): The name of the BigQuery dataset.
        bq_location (str): The location of the BigQuery dataset.
        table_name (str): The name of the BigQuery table.
        model_name (str): The name of the model associated with the metrics.
        molde_version (str): Model Version
        timestamp (datetime): The timestamp when the metrics were recorded.
        metrics (Dict[str, Union[float, bool]]): A dictionary containing the
        metrics to be written to the table.

    Returns:
        None

    Raises:
        Exception: If google.api_core.exceptions.GoogleAPICallError - there is a job error
        loading the BigQuery table
    """
    # Create a dictionary to represent the row of metrics to be inserted
    bq_saved_metric_row: Dict[str, Union[str, float, bool]] = {
        "model_name": model_name,
        "model_version": model_version,
        "timestamp": timestamp.isoformat(),
        **metrics,
    }

    bq_client = bigquery.Client(project=project)
    job_config = bigquery.LoadJobConfig()
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
    job_config.create_disposition = bigquery.CreateDisposition.CREATE_IF_NEEDED
    job_config.schema_update_options = [bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION]
    job_config.schema = [
        bigquery.SchemaField("model_name", "STRING"),
        bigquery.SchemaField("model_version", "STRING"),
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
    ] + [
        bigquery.SchemaField(name, python_type_to_bq_type(type(value)))
        for name, value in metrics.items()
    ]
    job = bq_client.load_table_from_json(
        [bq_saved_metric_row],
        f"{project}.{dataset}.{table_name}",
        location=bq_location,
        job_config=job_config,
    )
    try:
        # Wait for the job to complete
        job.result()
    except google.api_core.exceptions.GoogleAPICallError:
        logging.error(f"Job errors: {job.errors}")
        raise


def write_metrics_to_cloud_monitoring(
    project: str,
    location: str,
    model_name: str,
    pipeline_job_name: str,
    timestamp: datetime,
    metrics: Dict[str, Union[float, bool]],
    prefix: str,
) -> None:
    """
    Write metrics to Cloud Monitoring under the prefix
    custom.googleapis.com/machine_learning/monitoring/

    Pass pipeline_job_name via the use of kfp.v2.dsl.PIPELINE_JOB_NAME_PLACEHOLDER within a
    pipeline function.

    Args:
        project (str): The Google Cloud project ID.
        location (str): The location where the metrics are recorded.
        model_name (str): The name of the machine learning model.
        pipeline_job_name (str): The name of the pipeline job where the metrics are recorded.
        timestamp (datetime): The timestamp when the metrics were recorded.
        metrics (Dict[str, Union[float, bool]]): A dictionary containing the metrics to be recorded.
        prefix (str): The prefix for metric names in Cloud Monitoring.

    Returns:
        None
    """

    # Define the type prefix for metric names
    type_prefix = f"custom.googleapis.com/machine_learning/monitoring/{prefix}"

    # Initialize a Cloud Monitoring client
    monitor_client = monitoring_v3.MetricServiceClient()

    # Iterate through the metrics and create time series
    for name, value in metrics.items():
        # Convert the metric value to the appropriate TypedValue
        if isinstance(value, bool):
            value = monitoring_v3.TypedValue(bool_value=value)
        elif isinstance(value, float):
            value = monitoring_v3.TypedValue(double_value=value)
        elif isinstance(value, int):
            value = monitoring_v3.TypedValue(double_value=float(value))
        else:
            logging.warning(f"Unable to record metric {value} of type {type(value)}")
            continue

        # Create a TimeSeries for the metric
        series = monitoring_v3.TimeSeries()
        series.metric.type = f"{type_prefix}/{name}"
        series.resource.type = "generic_task"
        series.resource.labels["project_id"] = project
        series.resource.labels["namespace"] = model_name
        series.resource.labels["job"] = model_name
        series.resource.labels["task_id"] = pipeline_job_name
        series.resource.labels["location"] = location

        # Define the time interval and create a data point
        interval = monitoring_v3.TimeInterval(
            end_time={"seconds": int(timestamp.timestamp()), "nanos": timestamp.microsecond * 1000}
        )
        point = monitoring_v3.Point(interval=interval, value=value)
        series.points = [point]

        # Create the time series in Cloud Monitoring
        monitor_client.create_time_series(name=f"projects/{project}", time_series=[series])
