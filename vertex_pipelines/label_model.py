import logging
import os

from dotenv import load_dotenv
from google.cloud import aiplatform

from config import LOCATION
from config import MODEL_NAME_PREFIX
from config import PROJECT

load_dotenv(".env")
MODEL_TYPE = os.environ.get("LABELING_MODEL_TYPE", "custom")
MODEL_NAME = f"{MODEL_NAME_PREFIX}_{MODEL_TYPE}"


def label_model() -> None:
    """
    Update the latest model as the default model to be used for inference.

    Returns:
        None

    Raises:
        Exception: If there are no models in the model registry to be labelled
        Exception: If the specified model and model version is not in model registry
    """
    aiplatform.init(project=PROJECT, location=LOCATION)
    models = aiplatform.Model.list(
        project=PROJECT, location=LOCATION, filter=(f"display_name={MODEL_NAME}")
    )

    if len(models) == 0:
        logging.error(f"There are no models in the {MODEL_NAME} model registry to be labelled.")
        raise Exception(f"Model registry is empty for {MODEL_NAME}")
    else:
        parent_model = models[0]

    # fetch all model versions for the given model
    model_registry = aiplatform.ModelRegistry(model=parent_model)
    model_versions = model_registry.list_versions()

    version = os.environ["LABELING_MODEL_VERSION"]
    logging.info(f"Version = {version}")

    if version == "":
        logging.info("No version is specified, using 'latest' as model version")
        version = "latest"

    if version != "latest":
        latest_version = version
        model_version_ids = [model.version_id for model in model_versions]
        logging.info(f"Available model versions: {model_version_ids}, version: {latest_version}")
        if latest_version not in model_version_ids:
            logging.error(
                f"""There is no version {latest_version} of the model {MODEL_NAME} in the model
                registry."""
            )
            raise Exception(f"No {MODEL_NAME} with version {latest_version} in the model registry")
    else:
        latest_version = model_versions[len(model_versions) - 1].version_id

    # this call will both update the given version & remove old version as default
    logging.info(f"Marking version {latest_version} as default")
    model_registry.add_version_aliases(version=latest_version, new_aliases=["default"])


if __name__ == "__main__":
    label_model()
