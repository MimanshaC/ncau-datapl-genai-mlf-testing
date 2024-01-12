import os

from dotenv import load_dotenv

load_dotenv(".env")

# Load environment variables into constants - key error raised if not set
ENV = "nprod"
#ENV = os.environ["ENVIRONMENT"]
#MODEL_NAME_PREFIX = os.environ["MODEL_NAME_PREFIX"]
MODEL_NAME_PREFIX = "xgb_churn_prediction"
MODEL_NAME_CUSTOM = f"{MODEL_NAME_PREFIX}_custom"

PIPELINE_BUCKET = "gs://mlops-ncau-data-nprod-aitrain"
#PIPELINE_BUCKET = os.environ["PIPELINE_BUCKET"]
PIPELINE_ROOT = f"{PIPELINE_BUCKET}/{MODEL_NAME_PREFIX}"

# Variables to change per project
PROJ_PREFIX = "ncau-data"
IMAGE_REGISTRY = f"ml-vertex-pipelines-v2-{ENV}"  # Change based on the project

# Define additional constants across project
PROJECT = f"{PROJ_PREFIX}-{ENV}-aitrain"
LOCATION = "australia-southeast1"
BASE_IMAGE = f"{LOCATION}-docker.pkg.dev/{PROJECT}/{IMAGE_REGISTRY}/{MODEL_NAME_PREFIX}:latest"
SERVING_CONTAINER_IMAGE = BASE_IMAGE
LOCATION_BQ = "australia-southeast1"
SERVICE_ENDPOINT = "australia-southeast1-aiplatform.googleapis.com"

# TODO: define model specific data constants
DATASET = MODEL_NAME_PREFIX
SERIES_ID_COLUMN = "user_id"
TARGET_COLUMN = "target_binary"
TIMESTAMP_COLUMN = "prediction_timestamp"
PREDICTION_COLUMN = "prediction_value"
INFERENCE_HISTORY_TABLE = "inference_data_history"
TRAINING_HISTORY_TABLE = "training_data_history_model_version"

PERFORMANCE_MONITORING_LOOKBACK_DAYS = 1200
PREDICTION_DRIFT_LOOKBACK_DAYS = 730
INTERIM_TABLE_EXPIRE_DAYS = 30
DATA_LIMIT = 100000

# The # prefix is required
# Set to None to disable alerting
ALERT_NOTIFICATION_SLACK_CHANNEL = f"#ml-ops-alerting-{ENV}"
SUBMISSIONS_SERVICE_ACCOUNT = f"mlops-service-account@{PROJECT}.iam.gserviceaccount.com"
