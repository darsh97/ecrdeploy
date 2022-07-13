import pandas

from datapuller import fetch_and_download
from dataparser import parse
from constants import s3_SOURCE_BUCKET, s3_DESTINATION_BUCKET
from typing import List
from helper import upload_file

import os


def main():
    data_files: List[str] = fetch_and_download(s3_SOURCE_BUCKET)
    """
    Traverse the downloaded datafiles, parse them and push the xlsx output to destionation bucket.
    """
    for data_file in data_files:
        if not data_file.endswith(".xml"): continue
        print(f"parsing {data_file}")
        data_file_df: pandas.DataFrame = parse(data_file)
        print(f"parsed {data_file}!\n")

        data_file_dir, data_file_name = os.path.split(data_file)
        xlsx_file_name: str = data_file_name.split(".")[0] + '.xlsx'
        xlsx_file_path: str = os.path.join(data_file_dir, xlsx_file_name)

        print(f"converting to xlsx {data_file}")
        data_file_df.to_excel(xlsx_file_path)
        print(f"converted to xlsx {data_file}\n")

        print(f"uploading {xlsx_file_name}")
        upload_response = upload_file(xlsx_file_path, s3_DESTINATION_BUCKET, xlsx_file_name)
        print(f"Uploaded {xlsx_file_name}: {upload_response}\n")


if __name__ == '__main__':
    main()
