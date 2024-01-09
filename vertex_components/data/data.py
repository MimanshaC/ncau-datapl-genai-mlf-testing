from typing import NamedTuple

from kfp.v2.dsl import Artifact
from kfp.v2.dsl import Output
from kfp.v2.dsl import component

from config import BASE_IMAGE


@component(base_image=BASE_IMAGE)
def create_train_test_sql_query(
    param_1: str = "test", param_2: str = "test", param_3: str = "test"
) -> NamedTuple("outputs", [("train_data_query", str), ("test_data_query", str)]):  # type: ignore
    """Component to generate SQL queries for train and test dataset

    All relevant libraries need to be imported within the component function

    Args:
        param_1 (str): parameter needed to generate data query
        param_2 (str): parameter needed to generate data query
        param_3 (str): parameter needed to generate data query

    Returns:
        NamedTuple: Tuple containing the train and test data query
    """
    from collections import namedtuple

    from xgb_churn_prediction.data import data_ingestion

    train_data_query = data_ingestion.create_data_query(param_1, param_2)
    test_data_query = data_ingestion.create_data_query(param_2, param_3)

    # Remove characters causing issues with json.load for pre-built components
    table = str.maketrans({i: " " for i in range(31)})
    test_data_query = test_data_query.translate(table)
    train_data_query = train_data_query.translate(table)

    output = namedtuple("output", ["train_data_query", "test_data_query"])
    return output(train_data_query=train_data_query, test_data_query=test_data_query)


@component(base_image=BASE_IMAGE)
def create_train_test_table(
    project: str, dataset_id: str, training_dataset: Output[Artifact]
) -> NamedTuple("output", [("table_id", str), ("timestamp_str", str)]):  # type: ignore
    """Component to run load train/test data as part of Vertex AI pipeline
    All relevant libraries need to be imported within the component function;
    otherwise they won't be found!

    Args:
        project (str): Project ID
        dataset (str): Dataset id
        training_dataset (Output[Artifact]): training dataset as output artifact
    Returns:
        output (namedtuple): table_id and timestamp_str for the generaterd training data table
    """
    import logging
    from collections import namedtuple
    from datetime import datetime
    from datetime import timezone

    import pandas as pd

    from xgb_churn_prediction.data import data_clean
    from xgb_churn_prediction.data import data_ingestion
    from xgb_churn_prediction.data import data_output
    from xgb_churn_prediction.data import data_split

    timestamp = datetime.now(tz=timezone.utc)

    # TODO: Add use case specific Data Ingestion query. Use
    # data_ingestion.load_sql_query_and_execute to load sql from .sql file
    # with parameters
    data_query = f"""SELECT * FROM
        `{project}.sample_dataset.sample_table`
    """
    dtypes = {"param1": int, "param2": int}

    # load dataset from BigQuery
    logging.info("Loading data from Big Query")
    dataset = data_ingestion.execute_bq_query(project, data_query, dtypes)

    # clean dataset
    logging.info("Cleaning training dataset")
    dataset_cleaned = data_clean.clean_data(dataset)

    if len(dataset_cleaned) == 0:
        raise ValueError("No data found that matches requirements.")

    # split into train/test dataset
    logging.info("Splitting training dataset into train/test")
    data_split_result = data_split.split_data(dataset_cleaned)
    train_data = data_split_result.train_data
    test_data = data_split_result.test_data

    # assign splt column for each split dataframe
    train_data["split"] = "TRAIN"
    test_data["split"] = "TEST"
    concatenated_df = pd.concat([train_data, test_data])

    # generate output table name with timestamp
    timestamp_str = timestamp.strftime("%Y_%m_%dT%H_%M_%S_%f")[:-3] + "Z"
    table_id = f"training_data_{timestamp_str}"
    full_table_name = f"{dataset_id}.{table_id}"

    # save dataframes to bigquery table
    data_type_mapping = None

    # output data to BigQuery
    logging.info("Storing training dataset in Big Query")
    data_output.output_data(project, concatenated_df, full_table_name, data_type_mapping)

    training_dataset.metadata = {
        "projectId": project,
        "datasetId": dataset_id,
        "tableId": table_id,
    }

    training_dataset.uri = f"https://www.googleapis.com/bigquery/v2/projects/{project}/datasets/{dataset_id}/tables/{table_id}"  # noqa

    # return complete table id
    output = namedtuple("output", ["table_id", "timestamp_str"])
    return output(table_id=table_id, timestamp_str=timestamp_str)


@component(base_image=BASE_IMAGE)
def create_inference_sql_query(
    param_1: str,
) -> NamedTuple("outputs", [("data_query", str)]):  # type: ignore
    """Component to run load train/test data as part of Vertex AI pipeline
    All relevant libraries need to be imported within the component function;
    otherwise they won't be found!

    Args:
        param1 (str): param 1

    Returns:
        NamedTuple: A named tuple containing the prepared data query.

    """

    from collections import namedtuple

    from xgb_churn_prediction.data import data_ingestion

    # Create the inference data query using a function from data_ingestion
    data_query = data_ingestion.create_inference_data_query(param_1)

    # Remove characters causing issues with json.load for pre-built components
    table = str.maketrans({i: " " for i in range(31)})
    data_query = data_query.translate(table)

    # Create a named tuple to store the prepared data query
    output = namedtuple("output", ["data_query"])
    return output(data_query=data_query)


@component(base_image=BASE_IMAGE)
def create_inference_table(
    project: str,
    dataset_id: str,
    inference_dataset: Output[Artifact],
) -> NamedTuple("output", [("table_id", str)]):  # type: ignore
    """Component to create interim table with inference data as part of Vertex AI pipeline
    All relevant libraries need to be imported within the component function;
    otherwise they won't be found!

    Args:
        project (str): GCP project
        dataset_id (str): dataset id for creating the training data table
        timestamp_column (str): column name for timestamp
        prediction_date (Optional[str]): prediction date if given, otherwise set as next Monday
        inference_dataset (Output[Artifact]): inference dataset as output artifact
    """
    import logging
    from collections import namedtuple
    from datetime import datetime
    from datetime import timezone

    from xgb_churn_prediction.data import data_ingestion
    from xgb_churn_prediction.data import data_output

    # TODO: Add use case specific Data Ingestion query. Use
    # data_ingestion.load_sql_query_and_execute to load sql from
    # .sql file with parameters
    query = f"""
        SELECT *
        FROM `{project}.sample_dataset.sample_inference_table`
    """
    dtypes = {"param1": int, "param2": int}

    # load dataset from BigQuery
    logging.info("Loading data from Big Query")
    dataset = data_ingestion.execute_bq_query(project, query, dtypes)

    # generate output table name with timestamp
    timestamp = datetime.now(tz=timezone.utc)
    timestamp_str = timestamp.strftime("%Y_%m_%dT%H_%M_%S_%f")[:-3] + "Z"
    table_id = f"inference_data_{timestamp_str}"
    full_table_name = f"{dataset_id}.{table_id}"

    # save dataframes to bigquery table
    data_type_mapping = None

    # output data to BigQuery
    logging.info("Storing inference dataset in Big Query")
    data_output.output_data(project, dataset, full_table_name, data_type_mapping)

    inference_dataset.metadata = {
        "projectId": project,
        "datasetId": dataset_id,
        "tableId": table_id,
    }

    inference_dataset.uri = f"https://www.googleapis.com/bigquery/v2/projects/{project}/datasets/{dataset_id}/tables/{table_id}"  # noqa

    # return complete table id
    output = namedtuple("output", ["table_id"])
    return output(table_id=table_id)
