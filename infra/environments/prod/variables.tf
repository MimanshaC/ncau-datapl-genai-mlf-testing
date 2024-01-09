variable "project" {
  description = "GCP project name"
  type    = string
}

variable "location" {
  default = "australia-southeast1"
  type    = string
}

variable "enable_apis" {
  default = true
  type    = bool
}

variable "apis" {
  type    = list(string)
  description = "List of GCP API to enable"
}

variable "env" {
  type    = string
  description = "GCP environment (e.g. nprod, prod)"
}

variable "slack_bot_token" {
  type    = string
  description = "OAuth token to authenticate with slack bot app"
}
