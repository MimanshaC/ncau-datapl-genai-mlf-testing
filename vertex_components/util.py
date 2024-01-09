from typing import NamedTuple
from typing import Optional

from kfp.v2.dsl import Artifact
from kfp.v2.dsl import Input
from kfp.v2.dsl import Output
from kfp.v2.dsl import component

from config import BASE_IMAGE


@component(base_image=BASE_IMAGE)
def unique_table_name(
    prefix: str,
    project: Optional[str] = None,
    dataset: Optional[str] = None,
) -> NamedTuple("Output", [("table_name", str), ("job_configuration_query", dict)]):  # type: ignore
    """
    Generate a unique table name based on the current timestamp.
    Uses the YYYY_MM_DDThh_mm_ss_sssZ format.
    Make sure to apply set_caching_options(enable_caching=False) to this component.
    """
    from collections import namedtuple
    from datetime import datetime
    from datetime import timezone

    Output = namedtuple("Output", ["table_name", "job_configuration_query"])
    # %f is microsecond so we strip the last 3 digits to get a truncated ms
    table_name = prefix + datetime.now(tz=timezone.utc).strftime("%Y_%m_%dT%H_%M_%S_%f")[:-3] + "Z"
    # Construct a job_configuration_query
    # Can't do this in the pipeline yet; see https://github.com/kubeflow/pipelines/issues/4802
    if project and dataset:
        job_configuration_query = {
            "destinationTable": {
                "projectId": project,
                "datasetId": dataset,
                "tableId": table_name,
            }
        }
    else:
        job_configuration_query = None
    return Output(
        table_name=table_name,
        job_configuration_query=job_configuration_query,
    )


@component(base_image=BASE_IMAGE)
def get_artifact_metadata_value(artifact: Input[Artifact], key: str) -> str:
    """
    Fetch a single metadata value from an artifact.
    Useful if you want to use a metdata value in a pipeline.
    """
    return artifact.metadata[key]


@component(base_image=BASE_IMAGE)
def generate_job_config(project: str, dataset: str, table_id: str) -> dict:  # type: ignore
    """Component to generate json job config for BQ query execution,
        otherwise JSON serializable error form pipeline param

    Args:
        project (str): project ID
        dataset (str): dataset ID
        table_id (str): table name

    Returns:
        dict: job config as dict
    """
    job_config = {
        "destinationTable": {
            "projectId": project,
            "datasetId": dataset,
            "tableId": table_id,
        },
        # Create the history table if it doesn't exist
        "createDisposition": "CREATE_IF_NEEDED",
        # Append if it does exist
        "writeDisposition": "WRITE_APPEND",
    }
    return job_config


@component(base_image=BASE_IMAGE)
def change_table_name(
    project: str, dataset: str, old_table_id: str, new_table_id: str, table: Output[Artifact]
) -> None:
    """Component to change table name and output the new table as an artifact

    Args:
        project (str): project ID
        dataset (str): dataset ID
        old_table_id (str): old table name
        new_table_id (str): new table name
        table (Output[Artifact]): new table as output artifact to access via Vertex UI
    """

    from xgb_churn_prediction.data import data_ingestion

    query = f"""
        ALTER TABLE `{project}.{dataset}.{old_table_id}`
        RENAME TO `{new_table_id}`
    """
    data_ingestion.execute_bq_query(project=project, sql_query=query)

    table.metadata = {
        "projectId": project,
        "datasetId": dataset,
        "tableId": new_table_id,
    }

    table.uri = f"https://www.googleapis.com/bigquery/v2/projects/{project}/datasets/{dataset}/tables/{new_table_id}"  # noqa
