from typing import NamedTuple

from kfp.v2.dsl import Artifact
from kfp.v2.dsl import Input
from kfp.v2.dsl import Model
from kfp.v2.dsl import Output
from kfp.v2.dsl import component

from config import BASE_IMAGE


@component(base_image=BASE_IMAGE)
def upload_model(
    model_name: str,
    serving_container_image: str,
    model: Input[Model],
    vertex_model: Output[Model],
) -> NamedTuple("output", [("version", str), ("resource_name", str)]):  # type: ignore
    """Component to upload a model into the Vertex AI registry as part of a pipeline
    To be able to version a model, existance of the model in registry is first checked to either
    create a new parent model or work with existing one.

    All relevant libraries need to be imported within the component function;
    otherwise they won't be found!

    Args:
        model_name (str): Name of model
        serving_container_image (str): Container image used to serve model
        model (Input[Model]): model as Model artifact
        service_endpoint (str): service endpoint to generate Vertex AI artifact uri
        vertex_model (Output[Model]): model as Vertex Model artifact

    Returns:
        NamedTuple: artifact_uri and resoruce_name needed to import Vertex AI model from registry
    """
    # import all necessary libraries within component
    from collections import namedtuple

    from google.cloud import aiplatform

    # check if model already exists (parent model) and the version just needs to be updated
    models = aiplatform.Model.list(filter=("display_name={}").format(model_name))

    if len(models) == 0:
        parent_model = None
        model_id = model_name
    else:
        parent_model = models[0].resource_name
        model_id = None

    # upload model
    uploaded_model = aiplatform.Model.upload(
        model_id=model_id,
        display_name=model_name,
        parent_model=parent_model,
        is_default_version=False,
        # remove "model" (the file name without its suffix of the model artifact on gcs) from uri
        # to only point to the directory containing the Model artifact
        # NOTE: This is the file name no matter what the parameter/display name of the artifact is
        artifact_uri=model.uri.replace("model", ""),
        # add all details needed for the custom container image that is used for serving
        serving_container_image_uri=serving_container_image,
        # reset entrypoint for serving image to initiate the server for handling requests
        serving_container_command=["sh", "entrypoint.sh"],
        # routes & port have to be set to be able to use BatchPredictOp
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
    )

    # save data to output param
    vertex_model.uri = uploaded_model.versioned_resource_name  # type: ignore

    # generate variables for importer node
    output = namedtuple("output", ["version", "resource_name"])
    resource_name = vertex_model.uri
    version = uploaded_model.version_id
    return output(version=version, resource_name=resource_name)


@component(base_image=BASE_IMAGE)
def deploy_model(model: Input[Model], vertex_endpoint: Output[Artifact]) -> None:
    """Component to deploy a model sitting in the Vertex AI registry to an endpoint
    as part of a Vertex AI pipeline.

    Args:
        model (Input[Model]): model as Model artifact
        vertex_endpoint (Output[Artifact]): model endpoint as output artifact

    Returns:
        None
    """
    # import all necessary libraries within component
    from google.cloud import aiplatform

    uploaded_model = aiplatform.Model(model.uri)

    # deploy model
    # TODO specify/create endpoint before, otherwise created by deploy function if non existant
    deployed_model = uploaded_model.deploy(
        machine_type="n1-standard-4",
        traffic_split={"0": 100},
        deployed_model_display_name=uploaded_model.display_name,
    )

    # Save data to the output params
    vertex_endpoint.uri = deployed_model.resource_name
