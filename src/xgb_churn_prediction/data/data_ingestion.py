# script for data ingestion

from typing import Dict
from typing import Optional

import pandas as pd
from google.cloud import bigquery

from ..util import get_resource_folder
from ..util import read_sql_file

# data parameters for modelling


def execute_bq_query(project: str, sql_query: str, dtypes: Optional[Dict] = None) -> pd.DataFrame:
    """Function to execute a SQL query on BigQuery and parse the result into a pandas dataframe

    Args:
        sql_query (str): SQL query to be executed
        project (str): environment where to execute the query
        dtypes Optional(Dict): data types specifications for columns

    Returns:
        pd.DataFrame: query result parsed into a pandas dataframe
    """
    client = bigquery.Client(project=project)
    query_job = client.query(sql_query)

    if dtypes:
        return query_job.result().to_dataframe(dtypes=dtypes)
    else:
        return query_job.result().to_dataframe()


def create_data_query(param_1: str, param_2: str) -> str:
    """Function to create data query based on parameters

    Args:
        param_1 (str): delivery method to ingest into the query
        param_2 (str): min date to ingest into the query

    Returns:
        str: Query with populated data
    """

    # TODO adapt query with sql logic needed

    query = f"""
        SELECT *
        FROM {param_1}
        WHERE {param_2} == 1
    """

    return query


def load_sql_query_and_execute(project: str, query_file_name: str) -> pd.DataFrame:
    """Function to load sql query from path and execute to load dataset from BQ

    Args:
        project (str): project ID
        query_file_name (str): name of query file to load

    Returns:
        pd.DataFrame: dataset as a dataframe
    """
    # get path to sql query in resource folder
    path = get_resource_folder().joinpath(query_file_name)

    # load sql query as string
    sql_query = read_sql_file(path)

    # set dtypes
    dtypes = {"column_1": str}

    # execute query and parse into dataframe
    df = execute_bq_query(project=project, sql_query=sql_query, dtypes=dtypes)

    return df


def create_inference_data_query(param_1: str) -> str:
    """Function to create data query based on parameters

    Args:
        param_1 (str): delivery method to ingest into the query

    Returns:
        str: Query with populated data
    """

    # TODO adapt query with sql logic needed

    query = f"""
        SELECT *
        FROM {param_1}
    """

    return query
