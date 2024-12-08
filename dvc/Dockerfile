# Base image
FROM python:3.10-slim

# Set working directory to /dvc
WORKDIR /dvc

# Install DVC and any other dependencies
RUN pip install --no-cache-dir dvc

# Copy the contents of the /dvc directory (from the build context) into the container
COPY . /dvc
