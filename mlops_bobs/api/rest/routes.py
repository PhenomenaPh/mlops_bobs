"""REST API routes implementation."""
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, HTTPException, status
from loguru import logger

from ...core.model_implementations import LinearRegressionModel, RandomForestModel
from ...core.models import ModelRegistry
from . import schemas


router = APIRouter()
model_registry = ModelRegistry(storage_path=Path('./models'))

MODEL_TYPES = {
    'LinearRegression': LinearRegressionModel,
    'RandomForest': RandomForestModel
}


@router.get('/health', response_model=schemas.HealthResponse)
async def health_check() -> schemas.HealthResponse:
    """Check service health."""
    return schemas.HealthResponse(
        status='healthy',
        version='0.1.0'
    )


@router.post('/models', response_model=schemas.ModelResponse, status_code=status.HTTP_201_CREATED)
async def create_model(request: schemas.ModelCreate) -> schemas.ModelResponse:
    """Create a new model."""
    try:
        if request.model_type not in MODEL_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f'Unsupported model type. Available types: {list(MODEL_TYPES.keys())}'
            )

        model_id = str(uuid4())
        model_class = MODEL_TYPES[request.model_type]

        try:
            model, metadata = model_class.create(
                model_id=model_id,
                model_name=request.model_name,
                **(request.hyperparameters or {})
            )
        except Exception as e:
            logger.error(f'Failed to create model instance: {e!s}')
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f'Failed to create model instance: {e!s}'
            )
        
        model_registry.save_model(model_id, model, metadata)

        return schemas.ModelResponse(
            model_id=metadata.model_id,
            model_name=metadata.model_name,
            model_type=metadata.model_type,
            created_at=metadata.created_at,
            hyperparameters=metadata.hyperparameters
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Failed to create model: {e!s}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post('/models/{model_id}/train')
async def train_model(model_id: str, data: schemas.TrainingData) -> dict:
    """Train a specific model."""
    try:
        model = model_registry.get_model(model_id)
        if model is None:
            raise HTTPException(status_code=404, detail=f'Model {model_id} not found')

        X = np.array(data.features)
        y = np.array(data.targets)

        model.train(X, y)
        model_registry.save_model(model_id, model, model_registry._models[model_id])

        return {'message': f'Model {model_id} trained successfully'}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Failed to train model: {e!s}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post('/models/{model_id}/predict', response_model=schemas.PredictionResponse)
async def predict(model_id: str, request: schemas.PredictionRequest) -> schemas.PredictionResponse:
    """Make predictions using a specific model."""
    try:
        model = model_registry.get_model(model_id)
        if model is None:
            raise HTTPException(status_code=404, detail=f'Model {model_id} not found')

        X = np.array(request.features)
        predictions = model.predict(X)

        return schemas.PredictionResponse(predictions=predictions.tolist())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Failed to make predictions: {e!s}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get('/models', response_model=list[schemas.ModelResponse])
async def list_models() -> list[schemas.ModelResponse]:
    """List all available models."""
    try:
        models = model_registry.list_models()
        return [
            schemas.ModelResponse(
                model_id=model.model_id,
                model_name=model.model_name,
                model_type=model.model_type,
                created_at=model.created_at,
                hyperparameters=model.hyperparameters
            )
            for model in models
        ]
    except Exception as e:
        logger.error(f'Failed to list models: {e!s}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete('/models/{model_id}')
async def delete_model(model_id: str) -> dict:
    """Delete a specific model."""
    try:
        if model_registry.delete_model(model_id):
            return {'message': f'Model {model_id} deleted successfully'}
        raise HTTPException(status_code=404, detail=f'Model {model_id} not found')
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Failed to delete model: {e!s}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
