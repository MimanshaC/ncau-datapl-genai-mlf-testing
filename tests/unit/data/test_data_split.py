import pandas as pd

from xgb_churn_prediction.data.data_split import split_data


def test_split_data():
    """Test split_data function"""

    data_set = pd.DataFrame(
        [
            ["2022-01-18", "Browns Bay", 9233, 800.2],
            ["2022-01-19", "Browns Bay", 9233, 800.2],
            ["2022-01-20", "Browns Bay", 9233, 800.2],
            ["2022-01-19", "Browns Bay", 9233, 800.2],
            ["2022-01-20", "Browns Bay", 9233, 800.2],
        ],
        columns=["receipt_date", "store_name", "store_id", "total_sales_excluding_gst"],
    )

    data_split_result = split_data(data_set)
    train = data_split_result.train_data
    test = data_split_result.test_data

    assert len(train) == 4
    assert len(test) == 1
