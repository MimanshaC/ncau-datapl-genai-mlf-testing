from kfp.v2.dsl import HTML
from kfp.v2.dsl import Metrics
from kfp.v2.dsl import Output
from kfp.v2.dsl import component

from config import BASE_IMAGE


@component(base_image=BASE_IMAGE)
def evaluate_prediction_drift(
    project_id: str,
    dataset_id: str,
    table_id: str,
    data_limit: int,
    inference_table: str,
    lookback_days: int,
    model_name: str,
    model_version: str,
    location: str,
    bq_location: str,
    series_id_expr: str,
    prediction_expr: str,
    timestamp_expr: str,
    pipeline_job_name: str,
    metrics: Output[Metrics],
    report: Output[HTML],
) -> None:
    """Evaluate prediction drift over previous inferences.

    Args:
        project_id (str): project ID
        dataset_id (str): dataset ID
        table_id (str): table ID of current inference run
        data_limit (int): data limit for quicker loading time
        inference_table (str): inference table ID
        lookback_days (int): days to look back on
        model_name (str): model name
        model_version (str): model version
        location (str): cloud monitoring location
        bq_location (str): BQ location
        series_id_column (str): column name of series ID
        prediction_expr (str): column name of prediction
        timestamp_column (str): column name of timestamp
        pipeline_job_name (str): pipeline job name
        metrics (Output[Metrics]): metrics as Output Metrics of component
        report (Output[HTML]): report as Output report of component
    """
    import logging
    from datetime import datetime
    from datetime import timezone

    from xgb_churn_prediction.data.data_ingestion import execute_bq_query
    from xgb_churn_prediction.monitoring import prediction_drift
    from xgb_churn_prediction.monitoring.inference_history_table import (
        fetch_historical_inference,
    )
    from xgb_churn_prediction.monitoring.metrics import (
        write_metrics_to_cloud_monitoring,
    )
    from xgb_churn_prediction.monitoring.metrics import write_metrics_to_table

    logging.info("Fetching inference history data from Big Query")
    inference_history = fetch_historical_inference(
        project=project_id,
        dataset=dataset_id,
        table=table_id,
        data_limit=data_limit,
        lookback_days=lookback_days,
        model_version=model_version,
    )

    if inference_history.empty:
        logging.info(
            f"No historical inference data for model version {model_version} available \
                - skipping prediction drift"
        )
        return

    logging.info("Fetching latest inference data from Big Query")
    latest_inference = execute_bq_query(
        project_id,
        f"""
        SELECT
            CAST({series_id_expr} AS STRING) AS series_id,
            CAST({timestamp_expr} AS timestamp) AS timestamp,
            {prediction_expr} AS prediction
        FROM `{project_id}.{dataset_id}.{inference_table}`
        LIMIT {data_limit}
        """,
    )

    logging.info("Generating evidently performance report and metrics")
    result_report, result_metrics = prediction_drift.evaluate(latest_inference, inference_history)
    result_report.save_html(report.path)
    for name, value in result_metrics.items():
        metrics.log_metric(name, float(value))

    now = datetime.now(tz=timezone.utc)

    logging.info("Writing metrics to monitoring table in Big Query")
    write_metrics_to_table(
        project=project_id,
        dataset=dataset_id,
        table_name="monitor_prediction_drift",
        model_name=model_name,
        model_version=model_version,
        bq_location=bq_location,
        timestamp=now,
        metrics=result_metrics,
    )

    logging.info("Writing metrics to cloud monitoring")
    write_metrics_to_cloud_monitoring(
        project=project_id,
        location=location,
        timestamp=now,
        model_name=model_name,
        pipeline_job_name=pipeline_job_name,
        metrics=result_metrics,
        prefix="prediction_drift",
    )
