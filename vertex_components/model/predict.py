from typing import NamedTuple

from kfp.v2.dsl import Artifact
from kfp.v2.dsl import Input
from kfp.v2.dsl import Model
from kfp.v2.dsl import component

from config import BASE_IMAGE


@component(base_image=BASE_IMAGE)
def batch_predictions(
    project: str,
    dataset: str,
    inference_data: Input[Artifact],
    model: Input[Model],
    timestamp_expr: str,
    prediction_expr: str,
    series_id_expr: str,
) -> NamedTuple("output", [("result_table_id", str), ["model_version", str]]):  # type: ignore
    """Component to run batch predictions as part of Vertex AI pipeline
    All relevant libraries need to be imported within the component function;
    otherwise they won't be found!

    Args:
        project (str): project ID
        dataset (str): dataset ID
        inference_data (Input[Artifact]): inference dataset as input artifact
        model (Input[Model]): Vertex model as Model artifact to evaluate
        timestamp_expr (str): column name for timestamp
        prediction_expr (str): column name for predictions
        series_id_expr (str): column name for series id

    Returns:
        NamedTuple: table id, modelversion
    """
    import logging
    from collections import namedtuple
    from datetime import datetime
    from datetime import timezone

    from xgb_churn_prediction.data import data_ingestion
    from xgb_churn_prediction.data import data_output
    from xgb_churn_prediction.model import predict
    from xgb_churn_prediction.model import save_load_model

    # Read in inference data
    logging.info("Fetching current inference data from Big Query")
    sql_query = f"""
        SELECT *
        FROM `{inference_data.metadata["datasetId"]}.{inference_data.metadata["tableId"]}`
    """
    data = data_ingestion.execute_bq_query(project, sql_query=sql_query)

    # load model from Google Cloud Storage
    logging.info("Loading model from Model Registry / GCS")
    model_resource_name = model.metadata["resourceName"]
    trained_model, model_version = save_load_model.load_model_from_gcs(model_resource_name)

    # make predictions on dataset within horizon
    logging.info("Running predicitions on inference data")
    predictions = predict.make_predictions(
        trained_model, data, prediction_expr, timestamp_expr, series_id_expr
    )
    predictions["model_version"] = int(model_version)

    # generate output table name with timestamp
    timestamp = datetime.now(tz=timezone.utc)
    timestamp_str = timestamp.strftime("%Y_%m_%dT%H_%M_%S_%f")[:-3] + "Z"
    table_id = f"predictions_{timestamp_str}"
    full_table_name = f"{project}.{dataset}.{table_id}"

    # output predicitons to output inference table
    logging.info("Storing predicitions in Big Query")
    data_output.output_data(project=project, dataset=predictions, table_id=full_table_name)

    # return generated table id
    output = namedtuple("output", ["result_table_id", "model_version"])
    return output(result_table_id=table_id, model_version=model_version)
