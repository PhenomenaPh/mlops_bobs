import io
import os
from datetime import datetime
from typing import List, Tuple

import pandas as pd
from minio import Minio

# MinIO client configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "datasets")


def get_minio_client():
    """Create and return a MinIO client instance."""
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,  # Set to True if using HTTPS
    )


def ensure_bucket_exists(client):
    """Ensure the bucket exists, create if it doesn't."""
    if not client.bucket_exists(MINIO_BUCKET):
        client.make_bucket(MINIO_BUCKET)


def save_dataframe_to_minio(df: pd.DataFrame, filename: str | None = None) -> str:
    """
    Save a pandas DataFrame to MinIO as a CSV file.

    Args:
        df: The pandas DataFrame to save
        filename: Optional custom filename, if not provided will generate one

    Returns:
        str: The object name (path) in MinIO where the file was saved
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dataset_{timestamp}.csv"

    # Convert DataFrame to CSV bytes
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    csv_buffer = io.BytesIO(csv_bytes)

    client = get_minio_client()
    ensure_bucket_exists(client)

    # Upload the file to MinIO
    client.put_object(
        bucket_name=MINIO_BUCKET,
        object_name=filename,
        data=csv_buffer,
        length=len(csv_bytes),
        content_type="text/csv",
    )

    return filename


def get_dataframe_from_minio(filename: str) -> pd.DataFrame:
    """
    Retrieve a CSV file from MinIO and return it as a pandas DataFrame.

    Args:
        filename: The name of the file to retrieve

    Returns:
        pd.DataFrame: The loaded DataFrame
    """
    client = get_minio_client()

    try:
        # Get the object from MinIO
        data = client.get_object(MINIO_BUCKET, filename)
        # Read the CSV data into a DataFrame
        return pd.read_csv(io.BytesIO(data.read()))
    except Exception as e:
        raise Exception(f"Error retrieving file from MinIO: {e!s}")


def list_datasets() -> list[dict]:
    """
    List all datasets stored in MinIO.

    Returns:
        List[Dict]: A list of dictionaries containing dataset information
        Each dictionary contains:
            - dataset_name: name of the dataset file
            - size: size in bytes
            - last_modified: last modification timestamp
    """
    client = get_minio_client()
    ensure_bucket_exists(client)

    try:
        objects = client.list_objects(MINIO_BUCKET)
        datasets = []

        for obj in objects:
            datasets.append(
                {
                    "dataset_name": obj.object_name,
                    "size": obj.size,
                    "last_modified": obj.last_modified.isoformat(),
                }
            )

        return datasets
    except Exception as e:
        raise Exception(f"Error listing datasets from MinIO: {e!s}")


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
