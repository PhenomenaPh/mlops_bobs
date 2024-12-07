import os

import pandas as pd

from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from . import dvc_utils



# MinIO client configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "datasets")



# Initialize DVC storage
dvc_storage = dvc_utils.DVCStorage()



def save_dataframe_to_minio(df: pd.DataFrame, filename: str) -> str:
    """
    Save a pandas DataFrame using DVC with MinIO as remote storage.

    Args:
        df: The pandas DataFrame to save
        filename: Optional custom filename, if not provided will generate one

    Returns:
        str: The name of the saved dataset
    """
    
    # Remove .csv extension if present
    filename = Path(filename).stem

    # Save using DVC
    file_path = dvc_storage.add_dataset(df, filename)
    
    return Path(file_path)


def get_dataframe_from_minio(dataset_name: str) -> pd.DataFrame:
    """
    Retrieve a dataset using DVC.

    Args:
        dataset_name: The name of the dataset to retrieve

    Returns:
        pd.DataFrame: The loaded DataFrame
    """
    # Get dataset using DVC
    file_path = dvc_storage.get_dataset(dataset_name)
    if file_path is None:
        raise Exception(f"Dataset {dataset_name} not found")

    return pd.read_csv(file_path)


def list_datasets() -> list[dict]:
    """
    List all datasets tracked by DVC.

    Returns:
        List[Dict]: A list of dictionaries containing dataset information
    """
    
    datasets = dvc_storage.list_datasets()

    # Format the response to match the expected schema
    return datasets
    # return [
    #     {
    #         "dataset_name": dataset["name"],
    #         "size": Path(dataset["path"]).stat().st_size,
    #         "last_modified": datetime.fromtimestamp(
    #             Path(dataset["path"]).stat().st_mtime
    #         ).isoformat(),
    #     }
    #     for dataset in datasets
    # ]


def format_dataset_for_training(
    df: pd.DataFrame,
) -> Tuple[List[List[float]], List[float]]:
    """
    Format a DataFrame into the structure required by the training API.
    Assumes the last column is the target variable.

    Args:
        df: The pandas DataFrame to format

    Returns:
        Tuple containing:
            - features: List of lists containing feature values
            - targets: List containing target values
    """
    # Convert DataFrame to float64 for API compatibility
    df = df.astype("float64")

    # Split features and target
    features_df = df.iloc[:, :-1]  # all columns except last
    targets_series = df.iloc[:, -1]  # last column

    # Convert to the format required by the API
    features = features_df.values.tolist()
    targets = targets_series.values.tolist()

    return features, targets
