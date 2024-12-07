#!/bin/bash

# Exit on error
set -e

# Check if .dvc directory exists; if not, initialize DVC without SCM
if [ ! -d ".dvc" ]; then
    
    dvc init --subdir

    # Add and configure the MinIO remote
    dvc remote add -d minio s3://datasets
    dvc remote modify minio endpointurl http://localhost:9000
    dvc remote modify minio access_key_id minio_user
    dvc remote modify minio secret_access_key minio_password
    
fi

# Pass control to the container's main process
exec "$@"
