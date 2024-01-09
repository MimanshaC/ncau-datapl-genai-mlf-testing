// Cannot use variables in backend specification
terraform {
  backend "gcs" {
    bucket = "${var.project.name}-tfstate"
    prefix = "env/state"
  }
}
