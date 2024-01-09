import pickle

import pandas as pd
from pytest import fixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


@fixture()
def dummy_model(tmp_path):
    """Create dummy model file"""

    model_path = tmp_path / "model.pkl"
    model_path.touch()
    model = Pipeline([("RFC", RandomForestClassifier())])
    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    yield str(model_path).replace(".pkl", ""), model


@fixture()
def mock_model(tmp_path):
    """Create mock model file"""

    model_path = tmp_path / "model_mock.pkl"
    model_path.touch()
    model = "MOCK"
    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    yield str(model_path).replace(".pkl", ""), model


@fixture()
def dataset(tmp_path):
    """Create dummy dataset file"""

    dataset_path = tmp_path / "dataset.csv"
    df = pd.DataFrame([[0, 0, 1], [1, 1, 0]])
    df.to_csv(dataset_path, index=False)

    yield str(dataset_path).replace(".csv", ""), df


@fixture()
def test_X_y_dataset():
    test_data = pd.DataFrame(
        [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 0],
            [1, 1, 0],
        ],
        columns=["1", "2", "label"],
    )

    X, y = test_data.drop(["label"], axis=1), test_data["label"]

    expected_feat = X

    yield (test_data, X, y, expected_feat)
