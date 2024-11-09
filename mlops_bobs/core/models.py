from loguru import logger
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class MetaData(BaseModel):
    """Ml Model Metadata"""

    model_id: str = Field(..., description='Unique identifier for the model')
    model_type: str = Field(..., description='Type/class of the model')
    created_at: str = Field(..., description='Timestamp when model was created')
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description='Model hyperparameters')


class BaseModel(ABC):
    """Base class for all ML models in the service."""

    @abstractmethod
    def train(self, X: Any, y: Any, **kwargs) -> None:
        """Train the model with given data."""

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Make predictions using the model."""

    @abstractmethod
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get model hyperparameters."""

    @abstractmethod
    def set_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Set model hyperparameters."""


class ModelRegistry:
    """Registry for managing ML models."""

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, ModelMetadata] = {}
        self._load_existing_models()

    def _load_existing_models(self) -> None:
        """Load metadata for existing models from storage."""
        for model_path in self.storage_path.glob('*.joblib'):
            try:
                metadata = joblib.load(model_path.with_suffix('.meta'))
                self._models[metadata.model_id] = metadata
            except Exception as e:
                logger.error(f'Failed to load model metadata {model_path}: {e!s}')

    def save_model(self, model_id: str, model: BaseModel, metadata: ModelMetadata) -> None:
        """Save model and its metadata to storage."""
        try:
            model_path = self.storage_path / f'{model_id}.joblib'
            metadata_path = self.storage_path / f'{model_id}.meta'

            joblib.dump(model, model_path)
            joblib.dump(metadata, metadata_path)

            self._models[model_id] = metadata
            logger.info(f'Successfully saved model {model_id}')
        except Exception as e:
            logger.error(f'Failed to save model {model_id}: {e!s}')
            raise

    def load_model(self, model_id: str) -> Optional[BaseModel]:
        """Load model from storage."""
        try:
            model_path = self.storage_path / f'{model_id}.joblib'
            if not model_path.exists():
                logger.warning(f'Model {model_id} not found')
                return None

            model = joblib.load(model_path)
            logger.info(f'Successfully loaded model {model_id}')
            return model
        except Exception as e:
            logger.error(f'Failed to load model {model_id}: {e!s}')
            raise

    def delete_model(self, model_id: str) -> bool:
        """Delete model and its metadata from storage."""
        try:
            model_path = self.storage_path / f'{model_id}.joblib'
            metadata_path = self.storage_path / f'{model_id}.meta'

            if not model_path.exists():
                logger.warning(f'Model {model_id} not found')
                return False

            model_path.unlink()
            metadata_path.unlink()
            self._models.pop(model_id, None)

            logger.info(f'Successfully deleted model {model_id}')
            return True
        except Exception as e:
            logger.error(f'Failed to delete model {model_id}: {e!s}')
            raise

    def list_models(self) -> List[ModelMetadata]:
        """List all available models."""
        return list(self._models.values())
