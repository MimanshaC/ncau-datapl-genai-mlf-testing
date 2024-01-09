resource "google_artifact_registry_repository" "docker" {
  location      = var.location
  repository_id = "mlops"
  description   = "Docker repo for ML Ops"
  format        = "DOCKER"
}
