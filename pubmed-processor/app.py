import gzip
import os
import pandas as pd

from bs4 import BeautifulSoup as bs

from data_puller import ftp_puller, is_file_age_lt_x_months
from data_parser import parse_article
from constants import storage_folder, s3_destination_bucket

from helper import upload_file


def main():
    fetch_from_ftp_to_local: bool = ftp_puller()
    

    for data_file in os.listdir(storage_folder):

        if data_file.endswith(".gz"):

            article_data = []

            data_file_local_path: str = os.path.join(storage_folder, data_file)
            print(f"Processing {data_file}")

            try:
                print(f"Extracting {data_file}")

                with gzip.open(data_file_local_path, "r") as _zip:
                    _xml = bs(_zip.read(), "lxml")

                print(f"Parsing {data_file}")

                for article in _xml.findAll("pubmedarticle"):
                    article_data.append(parse_article(article))

                print(f"Converting {data_file} to Dataframe!")
                article_df = pd.DataFrame(article_data)

                xlsx_file_name: str = data_file.split(".")[0] + ".xlsx"
                pkl_file_name: str = data_file.split(".")[0] + ".pkl"

                xlsx_local_file_path: str = os.path.join(storage_folder, xlsx_file_name)
                pkl_local_file_path: str = os.path.join(storage_folder, pkl_file_name)

                print(f"Converting {data_file}_dataframe to Pickle: {pkl_local_file_path}")
                article_df.to_pickle(pkl_local_file_path)

                print(f"Converting {data_file}_dataframe to Excel: {xlsx_local_file_path}")
                article_df.to_excel(xlsx_local_file_path)

                if os.path.isfile(pkl_local_file_path):
                    upload_response = upload_file(pkl_local_file_path, s3_destination_bucket, pkl_file_name)
                    print(f"Uploaded {pkl_file_name} to {s3_destination_bucket}")

                if os.path.isfile(xlsx_local_file_path):
                    upload_response = upload_file(xlsx_local_file_path, s3_destination_bucket, xlsx_file_name)
                    print(f"Uploaded {xlsx_file_name} to {s3_destination_bucket}")


            except Exception as err:
                print(f"{data_file}: {err}")

        return


if __name__ == '__main__':
    main()
