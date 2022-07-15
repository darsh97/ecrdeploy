import timeit

import boto3
import boto3
import os

from typing import List
from botocore.client import ClientError

from constants import DATASTORE_PATH, s3_SOURCE_BUCKET
from helper import file_age_in_days

s3 = boto3.resource('s3')

s3_client = boto3.client('s3')


def _s3_get_all_objects_pagination(bucket, prefix, requester_pays=False):
    """Get s3 objects from a bucket/prefix
    optionally use requester-pays header
    """

    batch: int = 0

    extra_kwargs = {}
    if requester_pays:
        extra_kwargs = {'RequestPayer': 'requester'}

    next_token = 'init'

    while next_token is not None:
        kwargs = extra_kwargs.copy()
        if next_token != 'init':
            kwargs.update({'ContinuationToken': next_token})

        resp = s3_client.list_objects_v2(
            Bucket=bucket, Prefix=prefix, **kwargs)

        print(f"Fetched batch: {batch}.")

        try:
            next_token = resp['NextContinuationToken']
        except KeyError:
            next_token = None

        for contents in resp['Contents']:
            yield contents

        batch += 1


def _s3_get_latest_files(bucket_objects, day_limit: int) -> List:
    """
    Retain objects when their last_modified happened less than day_limit days ago.
    """

    latest_object: List = []
    for bo in bucket_objects:

        if file_age_in_days(bo.get("LastModified")) <= day_limit:
            latest_object.append(bo)

    return latest_object


def _s3_download_object_to_local(latest_bucket_objects, download_location: str, requester_pay=False) -> List[str]:
    file_locations: List[str] = []

    for bo in latest_bucket_objects:
        bo_key = bo.get("Key")
        s3_file_path, s3_file_name = os.path.split(bo_key)
        print(f"Processing {bo_key}!")
        if not s3_file_name: continue

        local_file_location = os.path.join(download_location, s3_file_name)

        try:
            print(f"downloading file {s3_file_name}")
            response = s3_client.download_file(s3_SOURCE_BUCKET, bo_key, local_file_location,
                                    {"RequestPayer": "requester"} if requester_pay else {})

            print(response)

            print(f"saving file {s3_file_name} to {local_file_location}\n")
            file_locations.append(local_file_location)

        except ClientError as err:
            print(f"Error downloading - {bo_key}, err: {err}")

    return file_locations


def fetch_and_download(bucket: str, day_limit: int, requester_pay: bool = False) -> List[str]:
    all_files = _s3_get_all_objects_pagination(bucket=bucket,
                                               prefix='',
                                               requester_pays=requester_pay)

    latest_files = _s3_get_latest_files(all_files, day_limit=day_limit)
    return _s3_download_object_to_local(latest_files, DATASTORE_PATH, requester_pay=requester_pay)
