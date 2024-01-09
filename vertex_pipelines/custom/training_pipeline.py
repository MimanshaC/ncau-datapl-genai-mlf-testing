from google_cloud_pipeline_components.v1.bigquery import BigqueryQueryJobOp
from kfp.v2 import dsl

from config import DATASET
from config import LOCATION
from config import LOCATION_BQ
from config import MODEL_NAME_CUSTOM
from config import PIPELINE_ROOT
from config import PROJECT
from config import SERVICE_ENDPOINT
from config import SERVING_CONTAINER_IMAGE
from config import TARGET_COLUMN
from config import TRAINING_HISTORY_TABLE
from vertex_components import util
from vertex_components.data import data
from vertex_components.model import evaluate
from vertex_components.model import train
from vertex_components.model import upload_deploy


@dsl.pipeline(
    # Default pipeline root. You can override it when submitting the pipeline.
    pipeline_root=PIPELINE_ROOT,
    # A name for the pipeline. Use to determine the pipeline Context.
    name="training-pipeline-xgb-churn-prediction-custom",
)
def training_pipeline_custom() -> None:
    """Training pipeline using custom components to run
    - data ingestion, cleaning, splitting
    - training of model
    - evaluation of model
    - uploading of model to Vertex AI registry
    - exporting of evaluations to Vertex AI registry
    - importing evaluations of current model
    - champion challenger

    Returns:
        None
    """

    # Define pipeline
    # Create dataset if not exists
    _ = BigqueryQueryJobOp(
        project=PROJECT,
        location=LOCATION_BQ,
        query=f"CREATE SCHEMA IF NOT EXISTS {DATASET}",
    )

    # Create train / test table in BQ
    dataset = data.create_train_test_table(project=PROJECT, dataset_id=DATASET).after(_)

    # Train model with train dataset
    model = train.train(
        project=PROJECT, dataset=dataset.outputs["training_dataset"], target_column=TARGET_COLUMN
    )  # type: ignore

    # Evaluate model with test dataset
    eval = evaluate.evaluate(
        project=PROJECT,
        dataset=dataset.outputs["training_dataset"],
        target_column=TARGET_COLUMN,
        model=model.outputs["model"],
    )

    # Upload model as Vertex Model to registry
    vertex_model = upload_deploy.upload_model(
        model_name=MODEL_NAME_CUSTOM,
        serving_container_image=SERVING_CONTAINER_IMAGE,
        model=model.outputs["model"],
    )  # type: ignore

    # Attach and upload evaluations to newly generated Vertex Model
    upload_evals = evaluate.upload_evaluations_classification(
        metrics=eval.outputs["metrics"],
        vertex_model=vertex_model.outputs["vertex_model"],
        service_endpoint=SERVICE_ENDPOINT,
    )

    # Compare current champion model to newly trained model
    evaluate.champion_challenger_classification(
        project_id=PROJECT,
        location=LOCATION,
        model_name=MODEL_NAME_CUSTOM,
        pipeline_job_name=dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
        vertex_model=vertex_model.outputs["vertex_model"],
    ).after(  # type: ignore
        upload_evals
    )

    # Rename training table to add model version
    util.change_table_name(
        project=PROJECT,
        dataset=DATASET,
        old_table_id=dataset.outputs["table_id"],
        new_table_id=f"{TRAINING_HISTORY_TABLE}_{vertex_model.outputs['version']}",
    ).set_display_name("Rename training data table")
