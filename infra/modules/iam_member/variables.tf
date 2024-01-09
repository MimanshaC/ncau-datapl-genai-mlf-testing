variable "project_name" {
  type    = string
  description = "GCP project name"
}

variable "sa_email" {
  type    = string
  description = "Service Account email to attach roles to"
}

variable "roles" {
  type    = list(string)
  description = "List of roles to attach to the Service Account"
}
