#!/bin/bash

# shell script to execute as an entrypoint when base image is used as model serving image
# this is only used if image is used as a serving image when uploading a model to the registry
# doco: https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements#server
exec gunicorn -c app.gunicorn.conf.py app:app
