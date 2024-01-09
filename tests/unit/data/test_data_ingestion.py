import pandas as pd

from xgb_churn_prediction.data.data_ingestion import create_data_query
from xgb_churn_prediction.data.data_ingestion import execute_bq_query


def test_create_data_query():
    """Test create_data_query function"""

    param_1 = "table123"
    param_2 = "count"
    expected_query = """
        SELECT *
        FROM table123
        WHERE count == 1
    """
    query = create_data_query(param_1, param_2)

    assert query == expected_query


def test_execute_bq_query(mocker):
    # Set test values
    project = "test"
    query = "SELECT * FROM table123"
    df_expected = pd.DataFrame([1, 1, 1])

    # Mock BQ client and return values
    mock_client = mocker.patch("google.cloud.bigquery.Client")
    mock_result = mocker.Mock()
    mock_result.to_dataframe.return_value = df_expected
    mock_client.return_value.query.return_value.result.return_value = mock_result

    df = execute_bq_query(project, query)

    # Check that the query was executed with the expected SQL query
    mock_client.return_value.query.assert_called_once_with(query)

    # Check that the query result was converted to a pandas dataframe
    assert isinstance(df, pd.DataFrame)

    # Check that the columns and values of the dataframe are correct
    assert df.equals(df_expected)
