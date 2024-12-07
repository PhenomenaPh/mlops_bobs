#!/bin/bash

# Exit on error
set -e

# Set an alias for the MinIO server located at http://minio:9000 using credentials
mc alias set myminio http://minio:9000 minio_user minio_password

# Create bucket
if ! mc ls myminio/datasets; then
  mc mb myminio/datasets
else
  echo "Bucket 'datasets' already exists"
fi
