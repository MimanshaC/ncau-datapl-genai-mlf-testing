resource "google_logging_metric" "logging_metric" {
  name   = "${var.project}-vertexai-api-disabled"
  filter = "resource.type=audited_resource AND resource.labels.service=serviceusage.googleapis.com AND resource.labels.method=google.api.serviceusage.v1.ServiceUsage.DisableService AND protoPayload.resourceName=projects/${var.project}/services/aiplatform.googleapis.com AND protoPayload.authenticationInfo.principalEmail=${var.monitoring_project}@appspot.gserviceaccount.com AND resource.labels.project_id=${var.project}"
}

resource "google_project_iam_binding" "project" {
  project = var.project
  role    = "roles/editor"

  members = [
    "serviceAccount:${var.monitoring_project}@appspot.gserviceaccount.com",
  ]
}
