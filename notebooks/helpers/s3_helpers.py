import os
import boto3
from datetime import datetime
from typing import Optional, Union, List

def get_s3_client() -> boto3.client:
    """Initialize S3 client using environment variables."""
    endpoint_url = os.getenv("AWS_ENDPOINT_URL_S3") or os.getenv("MLFLOW_S3_ENDPOINT_URL")
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        endpoint_url=endpoint_url,
    )


def get_latest_s3_folder(
    bucket_name: str,
    prefix: str,
    date_formats: Union[str, List[str]] = ("%y%m%d%H%M%S", "%y%d%m%H%M%S", "%y%m%d%H%M", "%y%d%m%H%M"),
    s3_client: Optional[boto3.client] = None,
) -> Optional[str]:
    """
    Finds and returns the full S3 path of the most recent timestamp folder under a given bucket and prefix.

    Args:
        bucket_name (str): Name of the S3 bucket (e.g., 'data').
        prefix (str): Prefix/folder path in the bucket (e.g., 'student_performance').
        date_formats (str or list of str): strptime format pattern(s) to match (e.g., '%y%m%d%H%M%S').
        s3_client (boto3.client, optional): Active boto3 S3 client instance. If None, creates one.

    Returns:
        Optional[str]: Full S3 URI of the latest folder (e.g., 's3://data/student_performance/240730211830/')
                       or None if no matching timestamp folders are found.
    """
    if s3_client is None:
        s3_client = get_s3_client()

    # Ensure prefix ends with '/' for directory listing
    prefix = prefix.strip("/") + "/" if prefix else ""

    # Fetch subfolders using Delimiter='/'
    response = s3_client.list_objects_v2(
        Bucket=bucket_name,
        Prefix=prefix,
        Delimiter="/"
    )

    if "CommonPrefixes" not in response:
        return None

    if isinstance(date_formats, str):
        date_formats = [date_formats]

    parsed_folders = []

    for obj in response["CommonPrefixes"]:
        # Extract folder name from prefix (e.g., "student_performance/240730211830/" -> "240730211830")
        folder_name = obj["Prefix"].rstrip("/").split("/")[-1]

        # Try parsing with supported date formats
        for fmt in date_formats:
            try:
                dt = datetime.strptime(folder_name, fmt)
                parsed_folders.append((dt, folder_name))
                break  # Matched format successfully
            except ValueError:
                continue

    if not parsed_folders:
        return None

    # Get the latest folder by parsed datetime
    _, latest_folder = max(parsed_folders, key=lambda item: item[0])
    
    return f"s3://{bucket_name}/{prefix}{latest_folder}/"
