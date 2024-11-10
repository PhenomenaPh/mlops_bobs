"""Script to test the REST API endpoints."""
from typing import Any

import requests
from loguru import logger


class MLServiceError(Exception):
    """Custom error for ML service operations."""


class MLServiceClient:
    """Client for interacting with the ML Service API."""

    def __init__(self, base_url: str = 'http://localhost:8000/api/v1'):
        self.base_url = base_url

    def _handle_response(self, response: requests.Response) -> Any:
        """Handle API response and potential errors."""
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            error_detail = response.json().get('detail', str(e))
            logger.error(f'HTTP Error: {error_detail}')
            raise MLServiceError(error_detail) from e
        except requests.exceptions.RequestException as e:
            logger.error(f'Request failed: {e!s}')
            raise MLServiceError(str(e)) from e

    def health_check(self) -> dict[str, str]:
        """Check service health."""
        response = requests.get(f'{self.base_url}/health')
        return self._handle_response(response)

    def create_model(
        self,
        model_type: str,
        hyperparameters: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Create a new model."""
        data = {
            'model_type': model_type,
            'hyperparameters': hyperparameters or {}
        }
        response = requests.post(f'{self.base_url}/models', json=data)
        return self._handle_response(response)

    def train_model(
        self,
        model_id: str,
        features: list[list[float]],
        targets: list[float]
    ) -> dict[str, str]:
        """Train a specific model."""
        data = {
            'features': features,
            'targets': targets
        }
        response = requests.post(
            f'{self.base_url}/models/{model_id}/train',
            json=data
        )
        return self._handle_response(response)

    def predict(
        self,
        model_id: str,
        features: list[list[float]]
    ) -> dict[str, list[float]]:
        """Make predictions using a specific model."""
        data = {
            'features': features
        }
        response = requests.post(
            f'{self.base_url}/models/{model_id}/predict',
            json=data
        )
        return self._handle_response(response)

    def list_models(self) -> list[dict[str, Any]]:
        """List all available models."""
        response = requests.get(f'{self.base_url}/models')
        return self._handle_response(response)

    def delete_model(self, model_id: str) -> dict[str, str]:
        """Delete a specific model."""
        response = requests.delete(f'{self.base_url}/models/{model_id}')
        return self._handle_response(response)


def test_linear_regression():
    """Test Linear Regression model workflow."""
    client = MLServiceClient()

    try:
        # 1. Check health
        logger.info('Checking service health...')
        health = client.health_check()
        logger.info(f'Health status: {health}')

        # 2. Create model
        logger.info('Creating Linear Regression model...')
        model_info = client.create_model(
            model_type='LinearRegression',
            hyperparameters={'fit_intercept': True}
        )
        model_id = model_info['model_id']
        logger.info(f'Created model: {model_info}')

        # 3. Train model
        logger.info(f'Training model {model_id}...')
        # Simple linear relationship: y = 2x
        training_result = client.train_model(
            model_id=model_id,
            features=[[1], [2], [3], [4], [5]],
            targets=[2, 4, 6, 8, 10]
        )
        logger.info(f'Training result: {training_result}')

        # 4. Make predictions
        logger.info('Making predictions...')
        # Should predict approximately y = 2x for new values
        predictions = client.predict(
            model_id=model_id,
            features=[[6], [7], [8]]  # Should predict close to [12, 14, 16]
        )
        logger.info(f'Predictions: {predictions}')

        # 5. List all models
        logger.info('Listing all models...')
        models = client.list_models()
        logger.info(f'Available models: {models}')

        # 6. Delete model
        logger.info(f'Deleting model {model_id}...')
        delete_result = client.delete_model(model_id)
        logger.info(f'Delete result: {delete_result}')

        logger.info('Linear Regression test completed successfully!')

    except MLServiceError as e:
        logger.error(f'Linear Regression test failed: {e!s}')
        raise


def test_random_forest():
    """Test Random Forest model workflow."""
    client = MLServiceClient()

    try:
        # 1. Create model with specific hyperparameters
        logger.info('Creating Random Forest model...')
        model_info = client.create_model(
            model_type='RandomForest',
            hyperparameters={
                'n_estimators': 100,
                'max_depth': 5,
                'random_state': 42
            }
        )
        model_id = model_info['model_id']
        logger.info(f'Created model: {model_info}')

        # 2. Train with more complex data
        logger.info(f'Training model {model_id}...')
        # Multiple features per sample: [x1, x2] => y = x1 + x2
        features = [
            [1, 2],  # sum = 3
            [2, 3],  # sum = 5
            [3, 4],  # sum = 7
            [4, 5],  # sum = 9
            [5, 6]   # sum = 11
        ]
        targets = [3, 5, 7, 9, 11]

        training_result = client.train_model(
            model_id=model_id,
            features=features,
            targets=targets
        )
        logger.info(f'Training result: {training_result}')

        # 3. Make predictions
        logger.info('Making predictions...')
        # Should predict approximately y = x1 + x2 for new values
        test_features = [
            [6, 7],  # should predict ≈ 13
            [7, 8],  # should predict ≈ 15
            [8, 9]   # should predict ≈ 17
        ]
        predictions = client.predict(
            model_id=model_id,
            features=test_features
        )
        logger.info(f'Predictions: {predictions}')

        # 4. Cleanup
        logger.info(f'Deleting model {model_id}...')
        delete_result = client.delete_model(model_id)
        logger.info(f'Delete result: {delete_result}')

        logger.info('Random Forest test completed successfully!')

    except MLServiceError as e:
        logger.error(f'Random Forest test failed: {e!s}')
        raise


def main():
    """Run all tests."""
    try:
        logger.info('Starting API tests...')

        logger.info('\n=== Testing Linear Regression ===')
        test_linear_regression()

        logger.info('\n=== Testing Random Forest ===')
        test_random_forest()

        logger.success('All tests completed successfully!')

    except Exception as e:
        logger.error(f'Tests failed: {e!s}')
        raise


if __name__ == '__main__':
    main()
