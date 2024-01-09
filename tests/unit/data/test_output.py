import pandas as pd
from google.cloud import bigquery

from xgb_churn_prediction.data.data_output import output_data


def test_output_data(mocker):
    """Test output_data function"""
    project = "test_project"
    table_id = "test_dataset.test_table"
    data_set = pd.DataFrame({"col": [1, 2, 3, 4]})

    mock_job = mocker.Mock(spec=bigquery.LoadJob)
    mock_job.result.return_value = None
    mock_load_table_from_dataframe = mocker.Mock(return_value=mock_job)

    mock_client = mocker.patch("google.cloud.bigquery.Client")
    mock_client_instance = mock_client.return_value
    mock_client_instance.load_table_from_dataframe = mock_load_table_from_dataframe

    # Call the output_data function
    output_data(project, data_set, table_id)

    # Assert the expected behavior
    mock_load_table_from_dataframe.assert_called_once_with(
        dataframe=data_set, destination=table_id, project=project
    )

    # Assert that the job was waited for
    mock_job.result.assert_called_once_with()
