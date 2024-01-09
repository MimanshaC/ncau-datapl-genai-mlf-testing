from xgb_churn_prediction.model.save_load_model import load_model
from xgb_churn_prediction.model.save_load_model import save_model


def test_load_model(mock_model) -> None:
    path = mock_model[0]
    model = load_model(path)

    assert model == "MOCK"


def test_save_model(mock_model) -> None:
    path = mock_model[0]
    model = mock_model[1]

    save_model(model, path)
