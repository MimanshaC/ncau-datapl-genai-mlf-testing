resource "google_monitoring_notification_channel" "slack" {
  display_name = "Slack"
  type         = "slack"
  labels = {
    "channel_name" = "#ml-ops-alerting-${var.env}"
  }
  sensitive_labels {
    auth_token = var.slack_bot_token
  }
}
