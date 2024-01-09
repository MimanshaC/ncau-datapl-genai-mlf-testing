import argparse
import os
from pathlib import Path
from typing import List

from google.cloud import aiplatform
from kfp.v2 import compiler

import config
from vertex_pipelines.pipeline_defs import PIPELINE_DEFS
from vertex_pipelines.pipeline_defs import PipelineDefinition


def compile(pipeline: PipelineDefinition) -> Path:
    """Function that uses the KFP SDK compiler to compile the written pipeline function
    in python into a static configuration json. This is needed to submit a pipeline for
    execution on Vertex AI.

    To make this function work across different pipelines, the functions are stored in a
    function dict and accessed via the passed parameter. Type checking is enabled to
    to ensure type consistency between components that pass data between one another.

    Args:
        pipeline (str): name of the pipeline to be compiled

    Returns:
        pipeline_file_path (str): the file path to the compiled pipeline json file
    """
    folder = Path("output_pipelines/")
    folder.mkdir(exist_ok=True)
    pipeline_file_path = folder / pipeline.json_file_name
    compiler.Compiler().compile(
        pipeline_func=pipeline.pipeline_function,
        package_path=str(pipeline_file_path),
        type_check=True,
        pipeline_parameters=pipeline.pipeline_kwargs,
    )
    return pipeline_file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile all pipelines into json workflows")
    parser.add_argument(
        "--include-schedules",
        action="store_true",
        required=False,
        help="Upload the schedules to GCP - github actions only",
    )
    args = parser.parse_args()

    schedules: List[PipelineDefinition] = []
    for pipeline_def in PIPELINE_DEFS:
        pipeline_file_path = compile(pipeline_def)
        cron = pipeline_def.schedule
        display_name = f"{config.MODEL_NAME_PREFIX}__{pipeline_def.name}"
        if not args.include_schedules or cron is None:
            continue

        template_path = os.path.join(os.getcwd(), pipeline_file_path)

        # Check if a schedule with the same display name already exists
        existing_schedules = aiplatform.PipelineJobSchedule.list(
            filter=f'display_name="{display_name}"'
        )

        if existing_schedules:
            # If a schedule with the same display name exists, update it
            pipeline_job_schedule = existing_schedules[0]
            pipeline_job_schedule.update(display_name=display_name, cron=cron)

        else:
            # If no schedule with the same display name exists, create a new one
            pipeline_job = aiplatform.PipelineJob(
                template_path=template_path,
                display_name=pipeline_def.name,
                project=config.PROJECT,
                location=config.LOCATION,
                enable_caching=False,
            )

            pipeline_job_schedule = pipeline_job.create_schedule(
                display_name=display_name,
                cron=cron,
                service_account=config.SUBMISSIONS_SERVICE_ACCOUNT,
            )