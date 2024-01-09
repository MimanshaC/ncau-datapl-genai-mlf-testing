from vertex_components.model.evaluate import evaluate


def test_evaluate_component(
    mocker, dataset_train_test_table, test_dataset, model_artifact, metrics_artifacts
):
    """Test model component evaluate"""
    # mock creation of SQL quyer + BQ result
    return_value = {"f1score": 0.8, "recall": 0.7, "precision": 0.9}
    mocker.patch(
        "xgb_churn_prediction.data.data_ingestion.execute_bq_query",
        return_value=test_dataset,
    )
    mocker.patch(
        "xgb_churn_prediction.model.evaluate.evaluate_model", return_value=return_value
    )

    metrics_artifact = metrics_artifacts[0]

    _ = evaluate.python_func(
        "project",
        dataset_train_test_table,
        "target_column",
        model_artifact,
        metrics_artifact,
    )

    assert metrics_artifact.metadata == return_value
