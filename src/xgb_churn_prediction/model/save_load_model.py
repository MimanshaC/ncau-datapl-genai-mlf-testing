import pickle
from typing import Any
from typing import Tuple

from google.cloud import aiplatform
from google.cloud import storage

# Define type of model file, e.g.
TYPE = "pkl"


def save_model(model: Any, path: str) -> None:
    """Function to save the model to a dedicated path

    Args:
        model (Any): Model to be saved
        path (str): Path where model should be saved to

    Returns:
        None

    """
    path = f"{path}.{TYPE}"

    with open(path, "wb") as file:
        pickle.dump(model, file)


def load_model(path: str) -> Any:
    """Function to load the model from a dedicated path

    Args:
        path (str): Path to load model from

    Returns:
        Any: Loaded model
    """
    path = f"{path}.{TYPE}"

    with open(path, "rb") as file:
        model = pickle.load(file)

    return model


def load_model_from_gcs(model_name: str) -> Tuple[Any, str]:
    """Function to load model file from gcs

    Args:
        model_name (str): model as resource name

    Returns:
        Tuple[Any, str]: tuple of model and its model version
    """
    # Get default model
    vertex_model = aiplatform.Model(model_name=model_name)
    # Create a client to interact with Google Cloud Storage
    client = storage.Client()
    # Get the bucket and blob names from the artifact URI
    bucket_name, blob_name = f"{vertex_model.uri}model.{TYPE}".replace("gs://", "").split("/", 1)
    # Get the blob object
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    # Download the blob to a local file
    local_path = f"model.{TYPE}"
    blob.download_to_filename(local_path)
    # Load the pickle file
    with open(local_path, "rb") as file:
        trained_model = pickle.load(file)

    return trained_model, vertex_model.version_id
