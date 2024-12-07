from pathlib import Path
from typing import Optional

from dvc.config import Config
from dvc.repo import Repo

from . import minio_utils


class DVCStorage:
    def __init__(self):
        self.repo = Repo.init(force=True)
        self._configure_remote()

    def _configure_remote(self):
        """Configure MinIO as a DVC remote"""
        # Configure MinIO remote
        self.repo.config.set(
            "remote.minio.url",
            f"s3://{minio_utils.MINIO_BUCKET}",
            level=Config.LEVEL_REPO,
        )

        # Set MinIO credentials and endpoint
        self.repo.config.set(
            "remote.minio.endpointurl",
            f"http://{minio_utils.MINIO_ENDPOINT}",
            level=Config.LEVEL_REPO,
        )
        self.repo.config.set(
            "remote.minio.access_key_id",
            minio_utils.MINIO_ACCESS_KEY,
            level=Config.LEVEL_REPO,
        )
        self.repo.config.set(
            "remote.minio.secret_access_key",
            minio_utils.MINIO_SECRET_KEY,
            level=Config.LEVEL_REPO,
        )

        # Set as default remote
        self.repo.config.set("core.remote", "minio", level=Config.LEVEL_REPO)

    def add_dataset(self, df, name: str) -> str:
        """
        Add a dataset to DVC tracking

        Args:
            df: pandas DataFrame to save
            name: name for the dataset

        Returns:
            str: Path to the tracked dataset
        """
        # Save dataset locally first
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        file_path = data_dir / f"{name}.csv"
        df.to_csv(file_path, index=False)

        # Add to DVC
        self.repo.add(str(file_path))

        # Push to remote (MinIO)
        self.repo.push()

        return str(file_path)

    def get_dataset(self, name: str) -> Optional[str]:
        """
        Get a dataset from DVC storage

        Args:
            name: Name of the dataset to retrieve

        Returns:
            Optional[str]: Path to the pulled dataset or None if not found
        """
        file_path = Path("data") / f"{name}.csv"

        try:
            # Pull from remote
            self.repo.pull([str(file_path)])
            return str(file_path)
        except Exception as e:
            print(f"Error pulling dataset: {e}")
            return None

    def list_datasets(self) -> list[dict]:
        """
        List all datasets tracked by DVC

        Returns:
            list[dict]: List of dataset information
        """
        datasets = []
        data_dir = Path("data")

        if data_dir.exists():
            for file in data_dir.glob("*.csv"):
                if (file.parent / (file.name + ".dvc")).exists():
                    # Get DVC file info
                    dvc_info = self.repo.status([str(file)])[str(file)]
                    datasets.append(
                        {
                            "name": file.stem,
                            "path": str(file),
                            "status": dvc_info["status"] if dvc_info else "unknown",
                        }
                    )

        return datasets
