"""REST API request/response schemas."""
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class BaseSchema(BaseModel):
    """Base schema with common configurations."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=()  # This removes the warnings
    )


class TrainingData(BaseSchema):
    """Schema for training data input."""
    features: list[list[float]] = Field(..., description='Training features (X)')
    targets: list[float] = Field(..., description='Training targets (y)')


class ModelCreate(BaseSchema):
    """Schema for model creation request."""
    model_name: str = Field(..., description='Human-readable name provided by the client')
    model_type: str = Field(..., description="Type of model to create (e.g., 'LinearRegression', 'RandomForest')")
    hyperparameters: Optional[dict[str, Any]] = Field(
        default=None,
        description='Model hyperparameters'
    )


class ModelResponse(BaseSchema):
    """Schema for model information response."""
    model_id: str = Field(..., description='Unique identifier for the model')
    model_name: str = Field(..., description='Human-readable name provided by the client')
    model_type: str = Field(..., description='Type of model')
    created_at: str = Field(..., description='Timestamp of model creation')
    hyperparameters: dict[str, Any] = Field(..., description='Model hyperparameters')


class PredictionRequest(BaseSchema):
    """Schema for prediction request."""
    features: list[list[float]] = Field(..., description='Features to make predictions for')


class PredictionResponse(BaseSchema):
    """Schema for prediction response."""
    predictions: list[float] = Field(..., description='Model predictions')


class HealthResponse(BaseSchema):
    """Schema for health check response."""
    status: str = Field(..., description='Service status')
    version: str = Field(..., description='Service version')
