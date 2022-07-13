import boto3
import os

from typing import NewType, List
from botocore.client import ClientError

from constants import DATASTORE_PATH
from helper import file_age_in_days

s3 = boto3.resource(
    's3',
)

s3_client = boto3.client(
    's3'
)


def _s3_get_all_objects(bucket: str):
    """
    Gets all the objects from given s3 bucket.

    :param bucket: name of the bucket (str)
    :return: list of s3 bucket objects
    """

    s3_bucket = s3.Bucket(bucket)
    try:
        return s3_bucket.objects.all()
    except ClientError as err:
        print(f"error - {err}")


def _s3_get_latest_files(bucket_objects, day_limit: int) -> List:
    """
    Retain objects when their last_modified happened less than day_limit days ago.
    """

    latest_object: List = []

    for bo in bucket_objects:
        if file_age_in_days(bo.last_modified) <= day_limit:
            latest_object.append(bo)

    return latest_object


def _s3_download_object_to_local(latest_bucket_objects, download_location: str) -> List[str]:
    file_locations: List[str] = []

    for bo in latest_bucket_objects:
        s3_file_path, s3_file_name = os.path.split(bo.key)

        if not s3_file_name: continue

        local_file_location = os.path.join(download_location, s3_file_name)

        try:
            print(f"downloading file {s3_file_name}")
            s3_client.download_file(bo.bucket_name, bo.key, local_file_location)

            print(f"saving file {s3_file_name}\n")
            file_locations.append(local_file_location)

        except ClientError as err:
            print(f"Error downloading - {bo.key}, err: {err}")

    return file_locations


def fetch_and_download(bucket: str) -> List[str]:
    latest_files = _s3_get_latest_files(_s3_get_all_objects(bucket), 3)
    return _s3_download_object_to_local(latest_files, DATASTORE_PATH)
