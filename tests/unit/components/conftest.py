import json
import pickle

import pandas as pd
from kfp.v2.dsl import Artifact
from kfp.v2.dsl import Dataset
from kfp.v2.dsl import Model
from pytest import fixture

from tests.unit.util import get_test_resource_folder
from tests.unit.util import make_test_artifact


@fixture()
def dataset_train_test_artifact(tmp_path):
    """Create Dataset artifacts in temporary folder"""
    path_train = tmp_path / "train.csv"
    path_test = tmp_path / "test.csv"
    path_val = tmp_path / "val.csv"
    pd.DataFrame([]).to_csv(path_train)
    pd.DataFrame([]).to_csv(path_test)
    pd.DataFrame([]).to_csv(path_val)
    ds_train = make_test_artifact(Dataset)(uri=str(path_train).replace(".csv", ""))
    ds_test = make_test_artifact(Dataset)(uri=str(path_test).replace(".csv", ""))
    ds_val = make_test_artifact(Dataset)(uri=str(path_val).replace(".csv", ""))

    yield ds_train, ds_val, ds_test


@fixture()
def dataset_train_test_table(tmp_path):
    """Create Dataset artifacts in temporary folder"""
    path = tmp_path / "bqml"
    bq_table = make_test_artifact(Artifact)(uri=str(path))
    bq_table.log_metric("datasetId", "test")
    bq_table.log_metric("tableId", "test")

    yield bq_table


@fixture()
def model_artifact(tmp_path):
    """Create model artifact in temporary folder"""
    path_model = tmp_path / "model.pkl"
    path_model.touch()
    model = "MOCK"
    with open(path_model, "wb") as file:
        pickle.dump(model, file)
    model = make_test_artifact(Model)(uri=str(path_model).replace(".pkl", ""))

    yield model


@fixture()
def metrics_artifacts(tmp_path):
    """Create metric artifacts in temporary folder"""
    path_metric = tmp_path / "metrics"
    path_cfmetric = tmp_path / "classification_metric"
    with open(path_metric, "w") as f:
        json.dump({"test": 1}, f)
    with open(path_cfmetric, "w") as f:
        json.dump({"test": 1}, f)
    metric = make_test_artifact(Model)(uri=str(path_metric))
    cf_metric = make_test_artifact(Model)(uri=str(path_cfmetric))
    yield metric, cf_metric


@fixture()
def test_dataset():
    test_dataset_path = get_test_resource_folder().joinpath("test-resource.csv")
    data = pd.read_csv(test_dataset_path)

    yield data
