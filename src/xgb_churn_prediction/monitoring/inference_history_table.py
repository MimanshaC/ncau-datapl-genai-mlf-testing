import google.api_core.exceptions
import pandas as pd

from xgb_churn_prediction.data.data_ingestion import execute_bq_query


def fetch_historical_inference(
    project: str,
    dataset: str,
    table: str,
    data_limit: int,
    lookback_days: int,
    model_version: str,
) -> pd.DataFrame:
    """Fetch the most recent historical inference data for each series.

    NOTE: The query can be changed to however the historical data is stored.
    e.g. to load the whole dataframe besides the insert date:
            SELECT * except(inserted_at)
            FROM `{project}.{dataset}.inference_history_{model_type}`
            WHERE timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {lookback_days} DAY)
            AND inserted_at =
                (SELECT MAX(inserted_at) FROM `{project}.{dataset}.inference_history_{model_type}`)
            LIMIT 10000
    Args:
        project (str): project ID
        dataset (str): dataset ID
        table (str): table ID
        data_limit (int): data limit to reduce loading time
        lookback_days (int): amount of days to look back at
        model_version (str): model version to limit data to

    Returns:
        pd.DataFrame: resulting dataframe with columns series_id, prediction, timestamp

    Raises:
        Exception: If google.api_core.exceptions.NotFound if the
        resource cannot be found (e.g. resource does
        not exist, incorrect name, insufficient permissions)
    """

    try:
        return execute_bq_query(
            project,
            f"""
            SELECT
                series_id,
                ANY_VALUE(prediction HAVING MAX inserted_at) AS prediction,
                timestamp
            FROM `{project}.{dataset}.{table}`
            WHERE timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {lookback_days} DAY)
            AND model_version = {model_version}
            GROUP BY series_id, timestamp
            LIMIT {data_limit}
            """,  # noqa: E501
        )
    except google.api_core.exceptions.NotFound as exc:
        if "Table" in exc.errors[0]["message"]:
            return pd.DataFrame(
                {
                    "series_id": pd.Series(dtype="object"),
                    "prediction": pd.Series(dtype="float64"),
                    "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
                }
            )
        else:
            raise
