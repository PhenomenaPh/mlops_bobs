import subprocess

from pathlib import Path
from typing import Optional

from dvc.repo import Repo



class DVCStorage:


    def __init__(self):
        self.repo = Repo("./dvc")

    def add_dataset(self, df, name: str) -> str:
        """
        Add a dataset to DVC tracking.

        Args:
            df: pandas DataFrame to save
            name: name for the dataset

        Returns:
            str: Path to the tracked dataset
        """
        
        # Save dataset locally first
        data_dir = Path("dvc/data")
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
        Get a dataset from DVC storage.

        Args:
            name: Name of the dataset to retrieve

        Returns:
            Optional[str]: Path to the pulled dataset or None if not found
        """
        
        file_path = Path("dvc/data") / f"{name}.csv"

        # Pull from remote
        self.repo.pull([str(file_path)])
        return str(file_path)


    def remove_dataset(self, name: str) -> str:
        """
        Remove a dataset and its metadata from DVC and MinIO storage.

        Args:
            name: Name of the dataset to remove

        Returns:
            str: Path of the removed dataset
        """
        
        # Configure paths
        csv_path = Path("data") / f"{name}.csv"
        dvc_path = Path("data") / f"{name}.csv.dvc"
        cwd = "dvc"

        # Remove .dvc file with dataset metadata from local machine
        subprocess.run(["dvc", "remove", dvc_path], cwd=cwd)
        # Delete dataset .csv file from local machine
        subprocess.run(["rm", "-rf", csv_path], cwd=cwd)
        # Force garbage collection (deletion of unused files) in MinIO remote storage (cloud)
        subprocess.run(["dvc", "gc", "-w", "-c", "--force"], cwd=cwd)
        
        return str(csv_path)


    def list_datasets(self) -> list[dict]:
        """
        List all datasets tracked by DVC.

        Returns:
            list[dict]: List of dataset information
        """

        datasets = []
        data_dir = Path("dvc/data")

        if data_dir.exists():
            for file in data_dir.glob("*.csv"):
                dvc_file = file.parent / (file.name + ".dvc")
                if dvc_file.exists():
                    
                    try:
                        dvc_info = self.repo.status([str(file)]).get(str(file), {})
                        datasets.append(
                            {
                                "name": file.stem,
                                "path": str(file),
                                "status": dvc_info.get("status", "unchanged"),
                            }
                        )

                    except KeyError:
                        datasets.append(
                            {
                                "name": file.stem,
                                "path": str(file),
                                "status": "not tracked",
                            }
                        )

            return datasets
