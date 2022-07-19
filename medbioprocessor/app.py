import pandas

from datapuller import fetch_and_download
from dataparser import parse
from constants import s3_SOURCE_BUCKET, s3_DESTINATION_BUCKET, DATASTORE_PATH
from typing import List, Dict
from helper import upload_file
from botocore.exceptions import ClientError

import os
import time
import pandas as pd


def main():
    data_files: List[str] = fetch_and_download(s3_SOURCE_BUCKET, day_limit=30, requester_pay=True)
    parsed_data_files: List[Dict] = [parse(f) for f in data_files if os.path.isfile(f)]

    print("Converted Parsed-files to dataframe!\n")
    data_df = pd.DataFrame(parsed_data_files)

    xlsx_file_name: str = f"biomedrxiv_{str(int(time.time()))}.xlsx"
    local_destination_path: str = os.path.join(DATASTORE_PATH, xlsx_file_name)

    print(f"Converting and saving xlsx to {local_destination_path}")
    data_df.to_excel(local_destination_path)

    try:
        print(f"Uploading {xlsx_file_name} to {s3_DESTINATION_BUCKET}!")
        if os.path.isfile(local_destination_path):
            upload_response = upload_file(local_destination_path, s3_DESTINATION_BUCKET, xlsx_file_name)
            print(f"Uploaded {xlsx_file_name} to {s3_DESTINATION_BUCKET}!")
        else:
            print(f"Invalid Path {local_destination_path}")
    except ClientError as e:
        print(f"Err {e}")


if __name__ == '__main__':
    main()
