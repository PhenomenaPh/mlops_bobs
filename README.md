# ML Model Service

A service for managing and using ML models with both REST and gRPC APIs. This service allows you to create, train, use, and manage multiple machine learning models through a simple API interface.

## Features

- REST and gRPC APIs
- Support for multiple ML model types
- Model lifecycle management (create, train, predict, delete)
- Hyperparameter customization
- Interactive dashboard
- Comprehensive logging
- Swagger documentation
- Authentication support

## Supported Models

Currently supported model types:
- Linear Regression
- Random Forest

Each model type supports its own set of hyperparameters that can be customized during creation.

## Project Structure

```
mlops_bobs/
├── src/
│   └── mlops_bobs/      
│       ├── api/         # API implementations
│       │   ├── rest/    # REST API endpoints
│       │   └── grpc/    # gRPC service
│       ├── core/        # Core business logic
│       ├── config/      # Configuration
│       └── utils/       # Utility functions
├── dashboard/           # Streamlit dashboard
├── tests/               # Test files
├── sample_data/         # Sample datasets
├── scripts/             # Utility scripts
├── poetry.lock         
└── pyproject.toml      
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ml-service
```

2. Make sure you have Poetry installed:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install dependencies:
```bash
poetry install
```

## Running the Service

1. Start the REST API server:
```bash
poetry run python scripts/start_rest.py
```

2. Access the Swagger documentation at:
```
http://localhost:8000/docs
```

## Running the Dashboard

1.  Start the Streamlit Dashboard:
```bash
streamlit run dashboard/streamlit_dashboard.py
```

2.  Log In the Dashboard
- username = `hse_mlops_2024`
- password = `strong_password`

## API Usage

### REST API Endpoints

1. Health Check:
```bash
GET /api/v1/health
```

2. Create Model:
```bash
POST /api/v1/models
{
    "model_type": "LinearRegression",
    "hyperparameters": {
        "fit_intercept": true
    }
}
```

3. Train Model:
```bash
POST /api/v1/models/{model_id}/train
{
    "features": [[1], [2], [3]],
    "targets": [2, 4, 6]
}
```

4. Make Predictions:
```bash
POST /api/v1/models/{model_id}/predict
{
    "features": [[4], [5], [6]]
}
```

5. List Models:
```bash
GET /api/v1/models
```

6. Delete Model:
```bash
DELETE /api/v1/models/{model_id}
```

### Example Using Python Client

```python
from ml_service.client import MLServiceClient

client = MLServiceClient()

# Create a model
model_info = client.create_model(
    model_type="LinearRegression",
    hyperparameters={"fit_intercept": True}
)

# Train the model
client.train_model(
    model_id=model_info["model_id"],
    features=[[1], [2], [3]],
    targets=[2, 4, 6]
)

# Make predictions
predictions = client.predict(
    model_id=model_info["model_id"],
    features=[[4], [5]]
)
```

## Testing

Run the test suite:
```bash
poetry run python scripts/test_api.py
```

## Development

1. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and commit:
```bash
git add .
git commit -m "Add your feature"
```

3. Push your changes and create a merge request:
```bash
git push origin feature/your-feature-name
```