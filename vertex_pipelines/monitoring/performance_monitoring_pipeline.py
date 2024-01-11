from kfp.v2 import dsl

from config import DATA_LIMIT
from config import DATASET
from config import INFERENCE_HISTORY_TABLE
from config import LOCATION
from config import LOCATION_BQ
from config import MODEL_NAME_CUSTOM
from config import PERFORMANCE_MONITORING_LOOKBACK_DAYS
from config import PIPELINE_ROOT
from config import PROJECT
from config import SERIES_ID_COLUMN
from config import TARGET_COLUMN
from config import TIMESTAMP_COLUMN
from vertex_components.monitoring import performance_monitoring


# TODO: Change pipeline root to desired gcs bucket
@dsl.pipeline(
    pipeline_root=PIPELINE_ROOT, name="performance-monitoring-xgb-churn-prediction"
)
def performance_monitoring_pipeline(model_version: str = "") -> None:
    """
    Performance monitoring to compare predictions with ground truth values.
    Generates Kubeflow Pipeline Metrics and an HTML report using evidently.

    Args:
        model_version (str): version of model to use
            defaults to empty string, i.e. highest model version available will be used
    """

    _ = performance_monitoring.evaluate_performance_monitoring(  # type: ignore
        project_id=PROJECT,
        dataset_id=DATASET,
        table_id=INFERENCE_HISTORY_TABLE,
        data_limit=DATA_LIMIT,
        model_name=MODEL_NAME_CUSTOM,
        model_version=model_version,
        location=LOCATION,
        bq_location=LOCATION_BQ,
        performance_monitoring_lookback_days=PERFORMANCE_MONITORING_LOOKBACK_DAYS,
        pipeline_job_name=dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
        target_column=TARGET_COLUMN,
        series_id_column=SERIES_ID_COLUMN,
        timestamp_column=TIMESTAMP_COLUMN,
    ).set_display_name("Evaluate model performance across inferences.")
