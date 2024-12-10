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
COPY dashboard/ dashboard/

# Expose the port of Streamlit dashboard
EXPOSE 8501

# Run Streamlit dashboard
CMD ["poetry", "run", "streamlit", "run", "dashboard/streamlit_dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]
