import gzip
import os
from typing import Set

import pandas as pd
import boto3

from bs4 import BeautifulSoup as bs

from data_puller import ftp_puller, is_file_age_lt_x_months
from data_parser import parse_article
from constants import storage_folder, s3_destination_bucket, processed_file_log

from helper import upload_file
from botocore.exceptions import ClientError

from random import shuffle

s3_client = boto3.client('s3')


def main():
    print("Starting to fetch data", flush=True)

    # fetch_from_ftp_to_local: bool = ftp_puller()
    total: int = 0
    processed_successfully: int = 0
    failed_files = []

    print("Download Complete\n", flush=True)
    print(os.listdir(storage_folder), flush=True)

    downloaded_files = os.listdir(storage_folder)
    shuffle(downloaded_files)

    # Get the list of processed_files from s3

    local_processed_file_log = os.path.join(storage_folder, processed_file_log)
    try:
        response = s3_client.download_file(s3_destination_bucket, processed_file_log, local_processed_file_log)
    except ClientError as e:
        print("Could not load processed_files_log", flush=True)
        raise e

    processed_files: Set[str] = set(line.strip() for line in open(local_processed_file_log))
    print(f"Already processed files {processed_files}", flush=True)

    files_to_process: Set[str] = set(downloaded_files) - processed_files

    for data_file in files_to_process:
        if data_file.endswith(".gz"):

            article_data = []

            data_file_local_path: str = os.path.join(storage_folder, data_file)
            print(f"Processing {data_file}", flush=True)

            try:
                print(f"Extracting {data_file}", flush=True)
                _zip = gzip.open(data_file_local_path, "r")
                print(f"Extracted {data_file}, converting to beautifulsoup", flush=True)

                _xml = bs(_zip.read(), "lxml")

                _zip.close()

                print(f"Parsing {data_file}", flush=True)

                for article in _xml.findAll("pubmedarticle"):
                    article_data.append(parse_article(article))

                print(f"Converting {data_file} to Dataframe!", flush=True)
                article_df = pd.DataFrame(article_data)

                xlsx_file_name: str = data_file.split(".")[0] + ".xlsx"
                pkl_file_name: str = data_file.split(".")[0] + ".pkl"

                xlsx_local_file_path: str = os.path.join(storage_folder, xlsx_file_name)
                pkl_local_file_path: str = os.path.join(storage_folder, pkl_file_name)

                print(f"Converting {data_file}_dataframe to Pickle: {pkl_local_file_path}", flush=True)
                article_df.to_pickle(pkl_local_file_path)

                print(f"Converting {data_file}_dataframe to Excel: {xlsx_local_file_path}", flush=True)
                article_df.to_excel(xlsx_local_file_path)

                uploaded_pkl: bool = False

                if os.path.isfile(pkl_local_file_path):
                    upload_response = upload_file(pkl_local_file_path, s3_destination_bucket, pkl_file_name)
                    print(f"Uploaded {pkl_file_name} to {s3_destination_bucket}", flush=True)
                    uploaded_pkl = True

                    with open(local_processed_file_log, 'w') as processed_log:
                        processed_log.write(f"{data_file}\n")

                        try:
                            upload_file(local_processed_file_log, s3_destination_bucket, processed_file_log)
                            print(f"Uploaded {processed_log} to {s3_destination_bucket}", flush=True)
                        except ClientError as e:
                            print(f"Failed to upload logfile: {e}", flush=True)

                    processed_successfully += 1

                if os.path.isfile(xlsx_local_file_path):
                    upload_response = upload_file(xlsx_local_file_path, s3_destination_bucket, xlsx_file_name)
                    print(f"Uploaded {xlsx_file_name} to {s3_destination_bucket}", flush=True)

                if uploaded_pkl:
                    print(f"PostProcessing: Clear {pkl_file_name}, {xlsx_file_name}, {data_file}", flush=True)
                    if os.path.isfile(pkl_local_file_path): os.remove(pkl_local_file_path)
                    if os.path.isfile(xlsx_local_file_path): os.remove(xlsx_local_file_path)
                    if os.path.isfile(data_file_local_path): os.remove(data_file_local_path)

            except Exception as e:
                failed_files.append(data_file)
                print(f"{e} - {data_file}", flush=True)

            total += 1

        print(f"Processing completed, Uploading log to {s3_destination_bucket}", flush=True)

        log_file_local_path = os.path.join(storage_folder, "failed_text_logs.txt")

        with open(log_file_local_path, 'w') as f:
            for file_name in failed_files:
                f.write(f"{file_name}\n")

        try:
            upload_file(log_file_local_path, s3_destination_bucket, "failed_text_logs.txt")
            print(f"Uploaded {log_file_local_path} to {s3_destination_bucket}", flush=True)
        except ClientError as e:
            print(f"Failed to upload logfile: {e}", flush=True)

        print(
            f"Completed Processing! total: {total}, processed_successfully: {processed_successfully}, failed: {total - processed_successfully}",
            flush=True)


if __name__ == '__main__':
    main()
