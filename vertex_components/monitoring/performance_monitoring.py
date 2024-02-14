from kfp.v2.dsl import HTML
from kfp.v2.dsl import Metrics
from kfp.v2.dsl import Output
from kfp.v2.dsl import component

from config import BASE_IMAGE


@component(base_image=BASE_IMAGE)
def evaluate_performance_monitoring(
    project_id: str,
    dataset_id: str,
    table_id: str,
    data_limit: int,
    model_name: str,
    model_version: str,
    location: str,
    bq_location: str,
    performance_monitoring_lookback_days: int,
    pipeline_job_name: str,
    target_column: str,
    series_id_column: str,
    timestamp_column: str,
    metrics: Output[Metrics],
    report: Output[HTML],
) -> None:
    """Evaluate the performance of a model over previous inferences.

    Args:
        project_id (str): project ID
        dataset_id (str): dataset ID
        table_id (str): table ID of current inference run
        data_limit (int): data limit for quicker loading time
        model_name (str): model name
        model_version (str): model version
        location (str): cloud monitoring location
        bq_location (str): BQ location
        performance_monitoring_lookback_days (int): days to look back on
        pipeline_job_name (str): pipeline job name
        target_column (str): column name of target variable
        series_id_column (str): column name of series ID
        timestamp_column (str): column name of timestamp
        metrics (Output[Metrics]): metrics as Output Metrics of component
        report (Output[HTML]): report as Output report of component
    """
    import logging
    from datetime import datetime
    from datetime import timezone

    from xgb_churn_prediction.data import data_ingestion
    from xgb_churn_prediction.monitoring import performance_monitoring
    from xgb_churn_prediction.monitoring.inference_history_table import (
        fetch_historical_inference,
    )
    from xgb_churn_prediction.monitoring.metrics import (
        write_metrics_to_cloud_monitoring,
    )
    from xgb_churn_prediction.monitoring.metrics import write_metrics_to_table

    # set model version as max if not set via pipeline
    if not model_version:
        query = f"""
            (SELECT MAX(model_version) AS model_version
            FROM `{project_id}.{dataset_id}.{table_id}`)
        """
        result = data_ingestion.execute_bq_query(project_id, query).astype(str)
        model_version = result["model_version"].iloc[0]

    # create sql queries for ground truth data and inference data
    logging.info("Fetching ground truth data from Big Query")
    # TODO: Add use case specific Data Ingestion query. Use
    # data_ingestion.load_sql_query_and_execute to load sql from .sql
    # file with parameters
    ground_truth_data_query = f"""
        SELECT *
        FROM `{project_id}.mlf_testing.customer_churn_data`
    """
    dtypes = {
         'user_id': int,
         'month': int,
         'package_duration': int,
         'segment': int,
         'brand_regional': int,
         'gender_F': int,
         'brand_metro': int,
         'brand_australian': int,
         'current_offer_intro_Y': int,
         'digital_subscription': int,
         'articleviews_30': int,
         'bill_shock_days': int,
         'current_offer_adj_weekly_price': float,
         'current_offer_rp_term_price': float,
         'dormant_freq_last_12_months': int,
         'dormant_freq_last_6_months': int,
         'frequency30': int,
         'length_of_relationship': int,
         'my_account_days': int,
         'num_previous_subs': int,
         'other_sections_30': int,
         'pageviews_30': int,
         'pricing_days': float,
         'ratio_views_to_article_views': float,
         'recency': int,
         'section_breadth30': int,
         'section_breadth7': int,
         'subscription_tenure': int,
         'visits_30': int,
         'visits_7': int,
         'bill_shock_days_rate': float,
         'pricing_days_rate': float,
         'frequency30_change_LT_0': int,
         'frequency30_change_LT_Neg_20': int,
         'frequency30_change_LT_Neg_60': int,
         'price_up': float,
         'section_sops_30': int,
         'target_binary': int,
         'subscription_id': int,
         'pcsid': int,
         'tenure_grouping': int,
         'mosaicsegments': int,
         'articleviews30_change_LT_0': int,
         'articleviews30_change_LT_Neg_20': int,
         'articleviews30_change_LT_Neg_60': int}
    
    # load dataset from BigQuery into pandas df
    ground_truth_dataset = data_ingestion.execute_bq_query(
        project_id, ground_truth_data_query, dtypes=dtypes
    )
    logging.info("Fetching inference history data from Big Query")
    inference_dataset = fetch_historical_inference(
        project=project_id,
        dataset=dataset_id,
        table=table_id,
        data_limit=data_limit,
        lookback_days=performance_monitoring_lookback_days,
        model_version=model_version,
    )

    if inference_dataset.empty:
        logging.info("No historical inference data available - skipping performance monitoring")
        return

    # perform performance monitoring
    processed_data = performance_monitoring.process_data(
        ground_truth_data=ground_truth_dataset,
        predictions_data=inference_dataset,
        target_column=target_column,
        timestamp_column=timestamp_column,
        series_id_column=series_id_column,
    )

    if processed_data.empty:
        logging.info(
            "No ground truth data for current predictions - skipping performance monitoring"
        )
        return
    else:
        logging.info(f"Assessing performance based on {len(processed_data)} datapoints")

    logging.info("Generating evidently performance report and metrics")
    result_report = performance_monitoring.generate_evidently_report(
        processed_data=processed_data,
        target_column=target_column,
        series_id_column=series_id_column,
    )

    result_metrics = performance_monitoring.extract_metrics(result_report)

    # output performance metrics to the report
    result_report.save_html(report.path)
    for name, value in result_metrics.items():
        metrics.log_metric(name, float(value))

    now = datetime.now(tz=timezone.utc)

    # output performance metrics to bigquery and cloud monitoring
    logging.info("Writing metrics to monitoring table in Big Query")
    write_metrics_to_table(
        project=project_id,
        dataset=dataset_id,
        table_name="monitor_performance",
        model_name=model_name,
        model_version=model_version,
        bq_location=bq_location,
        timestamp=now,
        metrics=result_metrics,
    )

    logging.info("Writing metrics to cloud monitoring")
    write_metrics_to_cloud_monitoring(
        project=project_id,
        location=location,
        timestamp=now,
        model_name=model_name,
        pipeline_job_name=pipeline_job_name,
        metrics=result_metrics,
        prefix="performance",
    )
