import dataclasses
import json
import logging
from typing import List

from google.api import metric_pb2 as ga_metric
from google.cloud import monitoring_v3

from config import ALERT_NOTIFICATION_SLACK_CHANNEL
from config import MODEL_NAME_PREFIX
from config import PROJECT

policy_prefix = "custom.googleapis.com/machine_learning/monitoring"


@dataclasses.dataclass
class AlertPolicy:
    policy_name: str
    metric: str
    model_name: str


POLICIES = [
    AlertPolicy(
        policy_name="Prediction drift",
        metric=f"{policy_prefix}/prediction_drift/drift_detected",
        model_name=f"{MODEL_NAME_PREFIX}_custom",
    ),
    AlertPolicy(
        policy_name="Model performance",
        metric=f"{policy_prefix}/performance/performance_alert",
        model_name=f"{MODEL_NAME_PREFIX}_custom",
    ),
    AlertPolicy(
        policy_name="Champion challenger",
        metric=f"{policy_prefix}/model_evaluation/improved_against_champion",
        model_name=f"{MODEL_NAME_PREFIX}_custom",
    )
    # TODO add any other policy here!
]


def gen_policies() -> List[monitoring_v3.AlertPolicy]:
    f"""
    Generates desired policies.

    Define all policies that should trigger alterts in the list. For each policy state:
        - "for_": name of the alert
        - "metric": metric name (defined when metrics are written to cloud monitoring)
        - "channel_name": channel to send alerts to; currently only Slack is configured
        - "model_name": exact model name for which the alerts should be configured
            like so {MODEL_NAME_PREFIX}_<type_of_model>
            e.g. {MODEL_NAME_PREFIX}_custom

    Returns:
        List: list of policies to be created/synced
    """
    if not ALERT_NOTIFICATION_SLACK_CHANNEL:
        return []

    channel_name = get_channel_name()

    policies = []
    for policy in POLICIES:
        check_metric(policy)
        policies.append(
            gen_policy(
                for_=policy.policy_name,
                metric=policy.metric,
                channel_name=channel_name,
                model_name=policy.model_name,
            )
        )
    return policies


def get_channel_name() -> str:
    """
    Find the already-existing notification channel for ALERT_NOTIFICATION_SLACK_CHANNEL, or error.

    Returns:
        (str): name of Slack notification channcel

    Raises:
        Exception: If the Slack notification channel is missing
    """
    channel_client = monitoring_v3.NotificationChannelServiceClient()
    channels = list(
        channel_client.list_notification_channels(
            monitoring_v3.ListNotificationChannelsRequest(
                name=f"projects/{PROJECT}",
                filter=(
                    "type = 'slack' AND "
                    f"labels.channel_name = '{ALERT_NOTIFICATION_SLACK_CHANNEL}'"
                ),
            )
        )
    )
    if channels:
        return channels[0].name
    else:
        raise Exception(f"Missing notification channel for {ALERT_NOTIFICATION_SLACK_CHANNEL}")


def gen_policy(
    for_: str, metric: str, channel_name: str, model_name: str
) -> monitoring_v3.AlertPolicy:
    """
    Generate an alert policy that triggers when metric (a boolean metric) is True.

    Args:
        for_ (str): A description or purpose for the alert
        metric (str): The name of the boolean metric that triggers the alert.
        channel_name (str): The notification channel to send alerts to.
        model_name (str): The name of the model associated with the alert.

    Returns:
        monitoring_v3.AlertPolicy: An alert policy object representing the generated policy.

    """
    # You can get JSON for these policies via a download button on the GCP Console
    # (it's easier to manipulate JSON than manually build protobuf objects)
    policy_object = {
        "displayName": f"{model_name} {for_.lower()}",
        "documentation": {
            "content": f"{for_} alert for model ${{resource.label.job}} in project ${{resource.project}}. See the pipeline for more details - https://console.cloud.google.com/vertex-ai/locations/${{resource.label.location}}/pipelines/runs/${{resource.label.task_id}}?project=${{resource.project}}.",  # noqa: E501
            "mimeType": "text/markdown",
        },
        "userLabels": {
            "severity": "warning",
            "model_name": model_name,
            "intent": for_.lower().replace(" ", "_"),
        },
        "conditions": [
            {
                "displayName": metric,
                "conditionThreshold": {
                    "aggregations": [
                        {"alignmentPeriod": "60s", "perSeriesAligner": "ALIGN_COUNT_TRUE"}
                    ],
                    "comparison": "COMPARISON_GT",
                    "duration": "0s",
                    "filter": f'resource.type = "generic_task" AND resource.labels.job = "{model_name}" AND metric.type = "{metric}"',  # noqa: E501
                    "trigger": {"count": 1},
                },
            }
        ],
        "alertStrategy": {"autoClose": "604800s"},  # 7 days
        "combiner": "OR",
        "enabled": True,
        "notificationChannels": [channel_name],
    }
    policy_json = json.dumps(policy_object)
    return monitoring_v3.AlertPolicy.from_json(policy_json)


def sync_policies() -> None:
    """
    Add or update desired policies and remove any other policies.

    Returns:
        None
    """
    policy_client = monitoring_v3.AlertPolicyServiceClient()
    existing = list(
        policy_client.list_alert_policies(
            monitoring_v3.ListAlertPoliciesRequest(
                name=f"projects/{PROJECT}",
                filter=f"user_labels.model_name = '{MODEL_NAME_PREFIX}_custom'",
            )
        )
    )
    # We match policies using the intent label
    existing_map = {
        policy.user_labels["intent"]: policy
        for policy in existing
        if "intent" in policy.user_labels
    }
    desired_policies = gen_policies()

    # Add or update desired policies
    for desired in desired_policies:
        matching = existing_map.pop(desired.user_labels["intent"], None)
        if matching:
            logging.info(f"{desired.display_name}: updating matching policy {matching.name}")
            desired.name = matching.name
            policy_client.update_alert_policy(alert_policy=desired)
        else:
            logging.info(f"{desired.display_name}: creating new policy")
            policy_client.create_alert_policy(name=f"projects/{PROJECT}", alert_policy=desired)

    # Remove any other policies
    for undesired in existing_map.values():
        logging.info(f"Removing policy {undesired.display_name} ({undesired.name})")
        policy_client.delete_alert_policy(name=undesired.name)


def check_metric(alert_policy: AlertPolicy) -> None:
    """
    Creates a metric descriptor for a metric that doesn't exit yet.
    If the metric exists no further action is taken.
    This is required to make sure an alert policy can be created.

    Args:
        metric_name (str): name of metric to check
        policy (str): name of policy

    Returns:
        None

    Raises:
        Exception: If metric doesn't exist - then creates the metric descriptor for metric
    """

    # Create a client
    client = monitoring_v3.MetricServiceClient()

    # Try to get metric descriptor - only possible if metric exists
    # if it does, no further actions need to be taken
    try:
        client.get_metric_descriptor(
            name=f"projects/{PROJECT}/metricDescriptors/{alert_policy.metric}"
        )
    # If exception is being raised, create metric descriptor for metric
    except Exception as e:
        logging.info(f"Exception {e}; creating metric descriptor for metric {alert_policy.metric}")
        descriptor = ga_metric.MetricDescriptor()
        descriptor.type = alert_policy.metric
        descriptor.metric_kind = ga_metric.MetricDescriptor.MetricKind.GAUGE
        # value type is always set to boolean at this stage, because all policies only
        # listen to bool metrics - if this changes, add value into policy definition
        descriptor.value_type = ga_metric.MetricDescriptor.ValueType.BOOL
        descriptor.description = f"Custom metrics for {alert_policy.policy_name.lower()}."

        # Make the request
        descriptor = client.create_metric_descriptor(
            name=f"projects/{PROJECT}", metric_descriptor=descriptor
        )


if __name__ == "__main__":
    sync_policies()
