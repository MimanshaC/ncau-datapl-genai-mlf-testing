from google_cloud_pipeline_components.types import artifact_types
from google_cloud_pipeline_components.v1.bigquery import BigqueryQueryJobOp
from kfp.v2 import dsl
from kfp.v2.components import importer_node

from config import DATA_LIMIT
from config import DATASET
from config import INFERENCE_HISTORY_TABLE
from config import INTERIM_TABLE_EXPIRE_DAYS
from config import LOCATION
from config import LOCATION_BQ
from config import MODEL_NAME_CUSTOM
from config import PIPELINE_ROOT
from config import PREDICTION_COLUMN
from config import PREDICTION_DRIFT_LOOKBACK_DAYS
from config import PROJECT
from config import SERIES_ID_COLUMN
from config import SERVICE_ENDPOINT
from config import TIMESTAMP_COLUMN
from config import TRAINING_HISTORY_TABLE
from vertex_components.data import util
from vertex_components.data.data import create_inference_table
from vertex_components.model.predict import batch_predictions
from vertex_components.monitoring.feature_drift import evaluate_feature_drift
from vertex_components.monitoring.inference_history_table import (
    add_to_inference_history_table,
)
from vertex_components.monitoring.prediction_drift import evaluate_prediction_drift


# TODO: Change pipeline root to desired gcs bucket
@dsl.pipeline(
    # Default pipeline root. You can override it when submitting the pipeline.
    pipeline_root=PIPELINE_ROOT,
    # A name for the pipeline. Use to determine the pipeline Context.
    name="inference-pipeline-xgb-churn-prediction-custom",
)
def inference_pipeline_custom() -> None:
    """Inference pipeline using custom components to
    - load model
    - create data query
    - run batch inference on data query for horizon
    """
    # Define pipeline
    # Load model based on artifact uri and resource name (default)
    artifact_uri = f"https://{SERVICE_ENDPOINT}/v1/projects/{PROJECT}/locations/{LOCATION}/models/{MODEL_NAME_CUSTOM}@default"  # noqa
    resource_name = f"projects/{PROJECT}/locations/{LOCATION}/models/{MODEL_NAME_CUSTOM}@default"
    importer_spec = importer_node.importer(
        artifact_uri=artifact_uri,
        artifact_class=artifact_types.VertexModel,
        metadata={"resourceName": resource_name},
    )

    inference_data = create_inference_table(project=PROJECT, dataset_id=DATASET)

    # Set inference source table expiration
    BigqueryQueryJobOp(
        project=PROJECT,
        location=LOCATION_BQ,
        query=f"""
      ALTER TABLE `{PROJECT}.{DATASET}.{inference_data.outputs["table_id"]}`
      SET OPTIONS (expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL {INTERIM_TABLE_EXPIRE_DAYS} DAY))
      """,  # noqa: E501
    ).set_display_name("Set inference source data table expiration")

    # Load data, run batch predictions and push output into BQ table
    batch_predict = batch_predictions(
        project=PROJECT,
        dataset=DATASET,
        inference_data=inference_data.outputs["inference_dataset"],
        model=importer_spec.outputs["artifact"],
        timestamp_expr=TIMESTAMP_COLUMN,
        prediction_expr=PREDICTION_COLUMN,
        series_id_expr=SERIES_ID_COLUMN,
    )

    # Set inference result table expiration
    BigqueryQueryJobOp(
        project=PROJECT,
        location=LOCATION_BQ,
        query=f"""
      ALTER TABLE `{PROJECT}.{DATASET}.{batch_predict.outputs["result_table_id"]}`
      SET OPTIONS (expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL {INTERIM_TABLE_EXPIRE_DAYS} DAY))
      """,  # noqa: E501
    ).set_display_name("Set inference result table expiration")

    # Check for prediction drift before adding new inferences to the history/reference table
    prediction_drift = evaluate_prediction_drift(
        project_id=PROJECT,
        dataset_id=DATASET,
        table_id=INFERENCE_HISTORY_TABLE,
        data_limit=DATA_LIMIT,
        inference_table=batch_predict.outputs["result_table_id"],
        lookback_days=PREDICTION_DRIFT_LOOKBACK_DAYS,
        model_name=MODEL_NAME_CUSTOM,
        model_version=batch_predict.outputs["model_version"],
        location=LOCATION,
        bq_location=LOCATION_BQ,
        series_id_expr=SERIES_ID_COLUMN,
        timestamp_expr=TIMESTAMP_COLUMN,
        prediction_expr=PREDICTION_COLUMN,
        pipeline_job_name=dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
    ).set_display_name(  # type: ignore
        "Evaluate prediction drift"
    )

    # Set training history data table name for model
    training_data = f"{TRAINING_HISTORY_TABLE}_{batch_predict.outputs['model_version']}"

    # Evaluate data drift is set to run after batch_predict
    # there isn't a current need to, but eventually it will
    evaluate_feature_drift(
        project_id=PROJECT,
        dataset_id=DATASET,
        table_id=training_data,
        data_limit=DATA_LIMIT,
        inference_dataset=inference_data.outputs["inference_dataset"],
        model_name=MODEL_NAME_CUSTOM,
        model_version=batch_predict.outputs["model_version"],
        location=LOCATION,
        bq_location=LOCATION_BQ,
        pipeline_job_name=dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
    ).after(batch_predict).set_display_name("Evaluate data drift")

    # Copy to history table
    job_config = util.generate_job_config(
        project=PROJECT, dataset=DATASET, table_id=INFERENCE_HISTORY_TABLE
    )

    add_to_inference_history_table(
        project=PROJECT,
        bq_location=LOCATION_BQ,
        dataset=DATASET,
        job_config=job_config.output,
        inference_result_table_name=batch_predict.outputs["result_table_id"],
        series_id_expr=SERIES_ID_COLUMN,
        timestamp_expr=TIMESTAMP_COLUMN,
        prediction_expr=PREDICTION_COLUMN,
    ).after(prediction_drift).set_display_name("Copy to history table")
