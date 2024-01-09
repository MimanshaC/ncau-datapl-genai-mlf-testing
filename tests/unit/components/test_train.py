import pickle

from vertex_components.model.train import train


def test_train_component(mocker, dataset_train_test_table, test_dataset, model_artifact):
    """Test model component train"""
    # mock training result
    mocker.patch(
        "xgb_churn_prediction.data.data_split.split_X_y",
        return_value=("X", "y"),
    )
    mocker.patch(
        "xgb_churn_prediction.data.data_ingestion.execute_bq_query",
        return_value=test_dataset,
    )

    mocker.patch("xgb_churn_prediction.model.train.train_model", return_value="MOCK")
    train.python_func("project", dataset_train_test_table, "label", model_artifact)

    assert pickle.load(open(model_artifact.uri + ".pkl", "rb")) == "MOCK"
