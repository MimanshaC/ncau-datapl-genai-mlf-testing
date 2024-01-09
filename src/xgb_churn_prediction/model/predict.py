from datetime import datetime
from datetime import timezone

import pandas as pd
from sklearn.pipeline import Pipeline


def make_predictions(
    model: Pipeline,
    data: pd.DataFrame,
    prediction_expr: str,
    timestamp_expr: str,
    series_id_expr: str,
) -> pd.DataFrame:
    """Function to generate predictions

    Args:
        model (Pipeline): model to use for predictions
        data (pd.DataFrame): dataframe to run predicitions on
        prediction_expr (str): column name for predictions
        timestamp_expr (str): column name for timestamp

    Returns:
        pd.DataFrame: predictions in a dataframe
    """

    # set timestamp
    time_stamp = datetime.now(tz=timezone.utc)
    date_str = time_stamp.strftime("%Y-%m-%d")

    # generate predictions, if there are additional steps required, insert here
    result = model.predict(data)

    # add predictions to dataframe
    data[prediction_expr] = result
    data[prediction_expr] = data[prediction_expr].astype(int)
    data[timestamp_expr] = date_str
    data[series_id_expr] = data[series_id_expr].astype(str)

    return data
