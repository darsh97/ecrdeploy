import gzip
import os
import pandas as pd

from bs4 import BeautifulSoup as bs

from data_puller import ftp_puller, is_file_age_lt_x_months
from data_parser import parse_article
from constants import storage_folder, s3_destination_bucket

from helper import upload_file
from botocore.exceptions import ClientError

from random import shuffle


def main():
    print("Starting to fetch data", flush=True)

    # fetch_from_ftp_to_local: bool = ftp_puller()
    total: int = 0
    processed: int = 0
    failed_files = []

    print("Download Complete\n", flush=True)
    print(os.listdir(storage_folder), flush=True)

    downloaded_files = os.listdir(storage_folder)

    shuffle(downloaded_files)

    for data_file in downloaded_files:

        if data_file.endswith(".gz"):

            article_data = []

            data_file_local_path: str = os.path.join(storage_folder, data_file)
            print(f"Processing {data_file}", flush=True)

            try:
                print(f"Extracting {data_file}", flush=True)
                _zip = gzip.open(data_file_local_path, "r")
                print(f"Extracted {data_file}, converting to beautifulsoup", flush=True)

                _xml = bs(_zip.read(), "lxml")

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
                    processed += 1

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

        print(f"Completed Processing! total: {total}, processed: {processed}, failed: {total - processed}", flush=True)


if __name__ == '__main__':
    main()
