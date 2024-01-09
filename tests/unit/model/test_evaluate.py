import operator

import pandas as pd

from xgb_churn_prediction.model.evaluate import champion_challenger
from xgb_churn_prediction.model.evaluate import evaluate_model


def test_evaluate_model(mocker, dummy_model, test_X_y_dataset):
    """Test evaluate_model function"""

    data = test_X_y_dataset[0]

    model = dummy_model[1]
    expected_pred = pd.DataFrame(
        [
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
        ],
        columns=["pred"],
    )

    expected_evaluation = {
        "accuracy": 0,
        "precision": 0,
        "recall": 0,
        "f1score": 0,
    }

    mocker.patch("sklearn.pipeline.Pipeline.predict", return_value=expected_pred)

    evaluation = evaluate_model(data, model, "label")

    assert evaluation.keys() == expected_evaluation.keys()


def test_champion_challenger():
    """Test model function evaluate"""
    metrics_challenger = {"accuracy": 0.8}
    metrics_champion = {"accuracy": 0.7}
    metrics_dict = {"accuracy": operator.ge}

    improved = champion_challenger(metrics_champion, metrics_challenger, metrics_dict)

    assert improved is True
