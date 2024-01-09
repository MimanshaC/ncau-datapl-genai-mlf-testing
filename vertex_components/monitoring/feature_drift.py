from kfp.v2.dsl import HTML
from kfp.v2.dsl import Artifact
from kfp.v2.dsl import Input
from kfp.v2.dsl import Metrics
from kfp.v2.dsl import Output
from kfp.v2.dsl import component

from config import BASE_IMAGE


@component(base_image=BASE_IMAGE)
def evaluate_feature_drift(
    project_id: str,
    dataset_id: str,
    table_id: str,
    data_limit: int,
    inference_dataset: Input[Artifact],
    metrics: Output[Metrics],
    report: Output[HTML],
    model_name: str,
    model_version: str,
    location: str,
    bq_location: str,
    pipeline_job_name: str,
) -> None:
    """Evaluate prediction drift over previous inferences.

    Args:
        project_id (str): project ID
        dataset_id (str): dataset ID
        table_id (str): table ID of historic dataset
        data_limit (int): data limit to reduce load time
        inference_dataset (Input[Artifact]): inference dataset to use for feature drift analysis
        metrics (Output[Metrics]): output metrics of evidently report
        report (Output[HTML]): output evidently report
        model_name (str): model name to use in table
        model_version (str): model version to use in table
        location (str): cloud monitoring location
        bq_location (str): BQ location
        pipeline_job_name (str): pipeline job name for monitoring purposes
    """
    import logging
    from datetime import datetime
    from datetime import timezone

    from xgb_churn_prediction.data.data_ingestion import execute_bq_query
    from xgb_churn_prediction.monitoring.feature_drift import create_report
    from xgb_churn_prediction.monitoring.metrics import (
        write_metrics_to_cloud_monitoring,
    )
    from xgb_churn_prediction.monitoring.metrics import write_metrics_to_table

    # Read in training history data (random selection of 100,000)
    logging.info("Fetching training history data from Big Query")
    training_query = f"""
        SELECT *
        FROM `{project_id}.{dataset_id}.{table_id}`
        ORDER BY RAND()
        LIMIT {data_limit}
    """
    training_data = execute_bq_query(project_id, sql_query=training_query)

    # Read in inference data (random selection of 100,000)
    logging.info("Fetching current inference data from Big Query")
    inference_query = f"""
        SELECT *
        FROM `{inference_dataset.metadata["datasetId"]}.{inference_dataset.metadata["tableId"]}`
        ORDER BY RAND()
        LIMIT {data_limit}
    """
    inference_data = execute_bq_query(project_id, sql_query=inference_query)

    # TODO: define features to check on for data drift
    # NOTE: limiting range to speed up processing
    features = ["feature_1"]

    logging.info(f"Generating evidently data drift report and metrics for columns {features}")
    result_report, result_metrics, monitoring_metrics = create_report(
        inference_data[features].astype(float), training_data[features].astype(float), features
    )

    logging.info("Logging report and metrics artifact as part of pipeline run")
    for name, value in result_metrics.items():
        metrics.log_metric(name, float(value))

    result_report.save_html(report.path)

    now = datetime.now(tz=timezone.utc)

    logging.info("Writing metrics to monitoring table in Big Query")
    write_metrics_to_table(
        project=project_id,
        dataset=dataset_id,
        table_name="monitor_feature_drift",
        model_name=model_name,
        model_version=model_version,
        bq_location=bq_location,
        timestamp=now,
        metrics=monitoring_metrics,
    )

    logging.info("Writing metrics to cloud monitoring")
    write_metrics_to_cloud_monitoring(
        project=project_id,
        location=location,
        timestamp=now,
        model_name=model_name,
        pipeline_job_name=pipeline_job_name,
        metrics=monitoring_metrics,
        prefix="feature_drift",
    )
