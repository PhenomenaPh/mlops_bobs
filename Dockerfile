# Base Python image
FROM python:3.12-slim-bullseye

# Set working directory
WORKDIR /app

# Ensure Python can locate custom modules
ENV PYTHONPATH /app

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy poetry configuration files
COPY pyproject.toml poetry.lock ./

# Install dependencies with Poetry
RUN poetry install --no-root

# Install ClearML (если не добавлено в pyproject.toml)
RUN poetry add clearml || pip install clearml

# Copy required files in working directory
COPY mlops_bobs/ mlops_bobs/
COPY dashboard/ dashboard/
COPY scripts/ scripts/

# Expose the port of REST API service
EXPOSE 8000

# Add ClearML environment variables for debugging (если не передаются в docker-compose.yml)
ENV CLEARML_API_HOST=http://clearml-server:8008
ENV CLEARML_API_ACCESS_KEY=clearml_user
ENV CLEARML_API_SECRET_KEY=clearml_password

# Run REST API service
CMD ["poetry", "run", "python", "scripts/start_rest.py"]
