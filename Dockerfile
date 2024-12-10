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
RUN poetry install

# Copy required files in working directory
COPY mlops_bobs/ mlops_bobs/
COPY dashboard/ dashboard/
COPY scripts/ scripts/

# Expose the port of REST API service
EXPOSE 8000

# Run REST API service
CMD ["poetry", "run", "python", "scripts/start_rest.py"]
