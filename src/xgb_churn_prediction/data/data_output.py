# script for outputting data to a BigQuery table
import logging
from typing import List
from typing import Optional
from typing import Tuple

import google.api_core.exceptions
import pandas as pd
from google.cloud import bigquery


def output_data(
    project: str,
    dataset: pd.DataFrame,
    table_id: str,
    data_type_mapping: Optional[List[Tuple[str, str]]] = None,
) -> None:
    """
    Writes the given output data to the corresponding table name in BigQuery.
    Args:
        project (str): GCP project
        dataset (pd.DataFrame): Dataframe containing output data
        table_id (str): id of output table in the form of dataset.tablename
        data_type_mapping Optional(List[Tuple[str, str]]): list of tuples containing column name
        and data type

    Returns:
        None

    Raises:
        Exception: google.api_core.exceptions.GoogleAPICallError when there is an API call error
    """
    # Initialize a BigQuery client using the specified GCP project.
    client = bigquery.Client(project=project)

    # Check if a data type mapping is provided, and create a schema if available.
    if data_type_mapping:
        schema = [
            bigquery.SchemaField(col_name, col_type) for col_name, col_type in data_type_mapping
        ]
        job_config = bigquery.LoadJobConfig(schema=schema)

        # Load the data from the DataFrame into the specified BigQuery table.
        job = client.load_table_from_dataframe(
            dataframe=dataset, destination=table_id, job_config=job_config, project=project
        )
    else:
        # Load the data from the DataFrame into the specified BigQuery table without a schema.
        job = client.load_table_from_dataframe(
            dataframe=dataset, destination=table_id, project=project
        )

    try:
        # Wait for the job to complete and handle any errors.
        job.result()
    except google.api_core.exceptions.GoogleAPICallError:
        logging.error(f"Job errors: {job.errors}")
        raise
