from typing import Any

from google_cloud_pipeline_components.v1.bigquery import BigqueryQueryJobOp


def add_to_inference_history_table(
    project: str,
    bq_location: str,
    dataset: str,
    job_config: dict,
    inference_result_table_name: str,
    series_id_expr: str,
    prediction_expr: str,
    timestamp_expr: str,
) -> Any:
    """
    Append the contents of a new inference table to the inference history table in BigQuery

    Args:
        project (str): Google Cloud project ID.
        bq_location (str): Location (gcp region & zone) for the BigQuery dataset
        dataset (str): Bigquery dataset name
        job_config (dict): BigQuery job configuration details
        inference_result_table_name (str): BigQuery table name for inference results
        series_id_expr (str): Series Id Expression
        prediction_expr (str): The name or expression representing the predictions made by the model
        timestamp_expr (str): The name or expression representing the timestamp
        associated with each data point


    Returns:
        None
    """

    return BigqueryQueryJobOp(
        project=project,
        location=bq_location,
        query=f"""
        SELECT
            CURRENT_TIMESTAMP() AS inserted_at,
            CAST({series_id_expr} AS STRING) AS series_id,
            CAST({timestamp_expr} AS timestamp) AS timestamp,
            {prediction_expr} AS prediction,
            model_version
        FROM
            `{project}.{dataset}.{inference_result_table_name}`
        """,
        job_configuration_query=job_config,
    )
