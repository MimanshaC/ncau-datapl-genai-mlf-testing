from google.cloud import aiplatform

import config
from vertex_pipelines.pipeline_defs import PIPELINE_DEFS

if __name__ == "__main__":
    # Filter pipeline definitions to get only training
    training = [pipeline_def for pipeline_def in PIPELINE_DEFS if pipeline_def.is_training is True]

    # Iterate through the training pipelines.
    for training_pipeline in training:
        pipeline_file_path = (
            config.PIPELINE_BUCKET
            + "/"
            + config.MODEL_NAME_PREFIX
            + "/"
            + training_pipeline.json_file_name
        )

        # Submit the pipeline job
        pipeline_job = aiplatform.PipelineJob(
            template_path=pipeline_file_path,
            display_name=training_pipeline.name,
            project=config.PROJECT,
            location=config.LOCATION,
            enable_caching=False,
        )

        pipeline_job.run(service_account=config.SUBMISSIONS_SERVICE_ACCOUNT)
