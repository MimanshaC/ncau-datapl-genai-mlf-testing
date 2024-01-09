# custom HTTP server using Flask to serve predictions from a custom-trained model
# doco: https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements#image
import io
import os
import pickle
from typing import Any
from typing import Dict
from typing import List

import pandas as pd
from flask import Flask
from flask import jsonify
from flask import request
from google.cloud import storage
from werkzeug.exceptions import HTTPException

health_endpoint = os.environ["AIP_HEALTH_ROUTE"]
predict_endpoint = os.environ["AIP_PREDICT_ROUTE"]
model_gcs_uri = os.environ["AIP_STORAGE_URI"]


def download_model() -> Any:
    """Function to load model from gcs uri

    Returns:
        object: The deserialized Python object, which is of the same type as the
        original pickled object.
    """
    client = storage.Client()
    buffer = io.BytesIO()
    client.download_blob_to_file(f"{model_gcs_uri}/model.pkl", buffer, raw_download=True)
    buffer.seek(0)
    return pickle.load(buffer)


model = download_model()

app = Flask(__name__)


@app.route(health_endpoint)
def healthy() -> Dict:
    """Endpoint to perform health checks on HTTP server (required by Vertex AI)

    Returns:
        Dict: empty dict as confirmation of health
    """
    return {}


@app.route(predict_endpoint, methods=["POST"])
def predict() -> Dict:
    """Endpoint to handle prediction requests if model is deployed to an endpoint

    Add all logic that is required to pre- or post-process data in here
    Currently, data is loaded from json body to then generate predicitons with the loaded model

    Returns:
        Dict: prediction results
    """
    obj = request.get_json(force=True, cache=False)
    instances: List[dict] = obj["instances"]
    data_df = pd.DataFrame(instances)

    # TODO Add any logic to pre process infence input
    predictions = model.predict(data_df)
    labels = [{"label": pred} for pred in predictions]
    output = {"predictions": labels}
    return jsonify(output)


@app.errorhandler(Exception)
def handle_exception(e: Exception) -> Any:
    if isinstance(e, HTTPException):
        return e

    return {"exception": repr(e)}, 500
