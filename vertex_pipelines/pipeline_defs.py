import dataclasses
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

from google.cloud.run_v2 import ServicesClient

import config
from vertex_pipelines.custom.inference_pipeline import inference_pipeline_custom
from vertex_pipelines.custom.training_pipeline import training_pipeline_custom
from vertex_pipelines.monitoring.performance_monitoring_pipeline import (
    performance_monitoring_pipeline,
)

# TODO: Add new schedule based on the use case
TIMEZONE_CRON_PREFIX = "TZ=Australia/Sydney"
WEEKLY_SUNDAY_MIDNIGHT = "0 0 * * 0"
DAILY_NINE_AM = "0 9 * * *"


@dataclasses.dataclass
class PipelineDefinition:
    """
    Represents a definition for a machine learning pipeline to be deployed

    Attributes:
        name (str): The name of the pipeline.
        pipeline_function (Callable): The Python function implementing the pipeline.
        schedule (Optional[str]): The schedule for running the pipeline (e.g., a cron expression).
        is_training (bool): Flag indicating whether the pipeline is used for training.
        pipeline_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments
            for the pipeline function.

    Properties:
        json_file_name (str): Name of the JSON file associated with this pipeline.
        submissions_endpoint (str): Endpoint for submitting pipeline executions.
    """

    name: str
    pipeline_function: Callable
    schedule: Optional[str] = None
    is_training: bool = False
    pipeline_kwargs: Optional[Dict[str, Any]] = None

    @property
    def json_file_name(self) -> str:
        """
        Returns the name of the JSON file associated with the pipeline.

        Returns:
            str: The JSON file name based on the pipeline's 'name' attribute.
        """
        return f"{self.name}.json"

    @property
    def submissions_endpoint(self) -> str:
        """
        Returns the endpoint for submitting pipeline executions.

        Returns:
            str: The endpoint for submitting pipeline executions.
        """
        cloud_run_client = ServicesClient()
        _parent = cloud_run_client.common_location_path(
            project=config.PROJECT, location=config.LOCATION
        )
        service = cloud_run_client.get_service(
            name=f"{_parent}/services/vertex-pipeline-submissions"
        )
        return f"{service.uri}/submit/{config.MODEL_NAME_PREFIX}/{self.name}"


# define pipeline parameters for each pipeline to be deployed
PIPELINE_DEFS = [
    PipelineDefinition(
        name="performance_monitoring_pipeline",
        pipeline_function=performance_monitoring_pipeline,
    ),
    PipelineDefinition(
        name="custom_training_pipeline",
        pipeline_function=training_pipeline_custom,
        is_training=True,
    ),
    PipelineDefinition(
        name="custom_inference_pipeline",
        pipeline_function=inference_pipeline_custom,
        schedule=f"{TIMEZONE_CRON_PREFIX} {DAILY_NINE_AM}",  # Example schedule
    ),
]

# run time check there are no duplicates
distinct_names = set()
duplicated_names = set()
for pipeline in PIPELINE_DEFS:
    duped = pipeline.name in distinct_names
    distinct_names.add(pipeline.name)
    if duped:
        duplicated_names.add(pipeline.name)

if duplicated_names:
    human_readable = ", ".join(duplicated_names)
    raise ValueError(
        f"The following pipeline names are duplicated, these must be unique: {human_readable}"
    )
