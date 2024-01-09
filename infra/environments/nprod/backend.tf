// Cannot use variables in backend specification
terraform {
  backend "gcs" {
    bucket = "ncau-data-nprod-aitrain-tfstate"
    prefix = "env/state"
  }
}
