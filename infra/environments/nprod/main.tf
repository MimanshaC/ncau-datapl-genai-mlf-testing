module "project-services" {
  source  = "terraform-google-modules/project-factory/google//modules/project_services"
  version = "~> 14.4"

  project_id                  = var.project
  enable_apis                 = var.enable_apis
  disable_services_on_destroy = false

  // Add new required API's in the array below
  activate_apis = var.apis
}

module "docker_repo" {
  source    = "../../modules/artifact_registry"
  location  = var.location
}

module "vertex_pipelines_storage" {
  source        = "../../modules/cloud_storage"
  project_name  = var.project
  location      = var.location
}

module "slack_notification_channel" {
  source          = "../../modules/cm_notification_channel"
  env             = var.env
  slack_bot_token = var.slack_bot_token
}
