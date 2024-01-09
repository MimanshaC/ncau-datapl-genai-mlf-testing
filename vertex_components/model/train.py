from kfp.v2.dsl import Artifact
from kfp.v2.dsl import Input
from kfp.v2.dsl import Output
from kfp.v2.dsl import component

from config import BASE_IMAGE


@component(base_image=BASE_IMAGE)
def train(
    project: str, dataset: Input[Artifact], target_column: str, model: Output[Artifact]
) -> None:
    """Component to run training as part of Vertex AI pipeline
    All relevant libraries need to be imported within the component function;
    otherwise they won't be found!

    Args:
        project (str): project ID
        dataset (Input[Artifact]): training dataset to train model with as Input Dataset
        target_column (str): Target Column for the model training
        model (Output[Artifact]): model as Output Artifact of component
    """
    import logging

    from xgb_churn_prediction.data import data_ingestion
    from xgb_churn_prediction.data import data_split
    from xgb_churn_prediction.model import save_load_model
    from xgb_churn_prediction.model import train

    # Set model path
    model_path = str(model.path)

    # Read in training data
    logging.info("Fetching training data from Big Query")
    sql_query = f"""
        SELECT * EXCEPT(split)
        FROM `{dataset.metadata["datasetId"]}.{dataset.metadata["tableId"]}`
        WHERE split = 'TRAIN'
    """
    train_data_df = data_ingestion.execute_bq_query(project, sql_query=sql_query)

    # Split Features / Target
    train_X, train_y = data_split.split_X_y(train_data_df, target_column)

    # Train model
    logging.info("Start model training")
    trained_model = train.train_model(train_X, train_y)

    # Save model
    logging.info("Model training successful - storing model in GCS")
    save_load_model.save_model(trained_model, model_path)
