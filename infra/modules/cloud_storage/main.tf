resource "google_storage_bucket" "source" {
  name                        = "mlops-${var.project_name}"
  project                     = var.project_name
  location                    = var.location
  force_destroy               = false
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  storage_class               = "STANDARD"
}
