from pathlib import Path
from typing import Optional

from dvc.config import Config
from dvc.repo import Repo



class DVCStorage:


    def __init__(self):
        self.repo = Repo("./dvc")


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
        Get a dataset from DVC storage

        Args:
            name: Name of the dataset to retrieve

        Returns:
            Optional[str]: Path to the pulled dataset or None if not found
        """
        
        file_path = Path("dvc/data") / f"{name}.csv"

        try:
            # Pull from remote
            self.repo.pull([str(file_path)])
            return str(file_path)
        except Exception as e:
            print(f"Error pulling dataset: {e}")
            return None


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
                        dvc_info = self.repo.status([str(file)])#.get(str(file), {})
                        datasets.append(
                            {
                                "name": file.stem,
                                "path": str(file),
                                "status": dvc_info.get("status", "unknown"),
                            }
                        )
                        print("Success!")
                    except KeyError:
                        print(f"File not tracked by DVC: {file}")
                        datasets.append(
                            {
                                "name": file.stem,
                                "path": str(file),
                                "status": "not tracked",
                            }
                        )
                else:
                    print(f"No DVC file found for: {file}")

        return dvc_info #datasets