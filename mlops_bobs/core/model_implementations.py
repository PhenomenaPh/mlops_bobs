"""Implementation of specific ML models."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from .models import BaseMLModel, ModelMetadata  # Updated import


class SklearnModelMixin:
    """Mixin class for scikit-learn based models."""

    def _validate_input(self, X: Any, action: str) -> None:
        """Validate input data."""
        if not isinstance(X, (np.ndarray, list)):
            raise ValueError(f'Input data for {action} must be numpy array or list')

        if isinstance(X, list):
            X = np.array(X)

        if len(X.shape) != 2:
            raise ValueError(f'Input data must be 2-dimensional, got shape {X.shape}')


class LinearRegressionModel(SklearnModelMixin, BaseMLModel):
    """Linear Regression model implementation."""

    def __init__(self):
        self.model = LinearRegression()
        self._hyperparameters = {
            'fit_intercept': True,
            'copy_X': True,
            'n_jobs': None,
            'positive': False
        }

    def train(self, X: Any, y: Any, **kwargs) -> None:
        """Train the linear regression model."""
        self._validate_input(X, "training")

        try:
            logger.info("Training LinearRegression model...")
            self.model.set_params(**self._hyperparameters)
            self.model.fit(X, y)
            logger.info("Model training completed successfully")
        except Exception as e:
            logger.error(f"Failed to train model: {str(e)}")
            raise

    def predict(self, X: Any) -> np.ndarray:
        """Make predictions using the trained model."""
        self._validate_input(X, "prediction")

        try:
            logger.debug("Making predictions with LinearRegression model...")
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Failed to make predictions: {str(e)}")
            raise

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get current hyperparameters."""
        return self._hyperparameters.copy()

    def set_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Set model hyperparameters."""
        valid_params = {
            k: v for k, v in params.items()
            if k in self._hyperparameters
        }
        self._hyperparameters.update(valid_params)
        logger.info(f"Updated hyperparameters: {valid_params}")

    @classmethod
    def create(cls, model_id: str, **kwargs) -> tuple[LinearRegressionModel, ModelMetadata]:
        """Create a new model instance with metadata."""
        model = cls()
        model.set_hyperparameters(kwargs)

        metadata = ModelMetadata(
            model_id=model_id,
            model_type="LinearRegression",
            created_at=datetime.utcnow().isoformat(),
            hyperparameters=model._hyperparameters
        )

        return model, metadata


class RandomForestModel(SklearnModelMixin, BaseMLModel):
    """Random Forest model implementation."""

    def __init__(self):
        self.model = RandomForestRegressor()
        self._hyperparameters = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': None,
            'n_jobs': None
        }

    @classmethod
    def create(cls, model_id: str, **kwargs) -> tuple[RandomForestModel, ModelMetadata]:
        """Create a new model instance with metadata."""
        model = cls()
        model.set_hyperparameters(kwargs)

        metadata = ModelMetadata(
            model_id=model_id,
            model_type="RandomForest",
            created_at=datetime.utcnow().isoformat(),
            hyperparameters=model._hyperparameters
        )

        return model, metadata

    def train(self, X: Any, y: Any, **kwargs) -> None:
        """Train the random forest model."""
        self._validate_input(X, 'training')

        try:
            logger.info('Training RandomForest model...')
            self.model.set_params(**self._hyperparameters)
            self.model.fit(X, y)
            logger.info('Model training completed successfully')
        except Exception as e:
            logger.error(f'Failed to train model: {e!s}')
            raise

    def predict(self, X: Any) -> np.ndarray:
        """Make predictions using the trained model."""
        self._validate_input(X, 'prediction')

        try:
            logger.debug('Making predictions with RandomForest model...')
            return self.model.predict(X)
        except Exception as e:
            logger.error(f'Failed to make predictions: {e!s}')
            raise

    def get_hyperparameters(self) -> dict[str, Any]:
        """Get current hyperparameters."""
        return self._hyperparameters.copy()

    def set_hyperparameters(self, params: dict[str, Any]) -> None:
        """Set model hyperparameters."""
        valid_params = {
            k: v for k, v in params.items()
            if k in self._hyperparameters
        }
        self._hyperparameters.update(valid_params)
        logger.info(f'Updated hyperparameters: {valid_params}')
