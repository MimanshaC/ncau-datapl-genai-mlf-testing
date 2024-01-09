variable "env" {
  type    = string
  description = "GCP environment (e.g. nprod, prod)"
}

variable "slack_bot_token" {
  type    = string
  description = "OAuth token to authenticate with slack bot app"
}
