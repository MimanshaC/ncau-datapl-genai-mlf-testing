// Add new roles to the service account in the array below
resource "google_project_iam_member" "member" {
  project = var.project_name
  for_each = toset(var.roles)
  role   = each.key
  member = "serviceAccount:${var.sa_email}"
}
