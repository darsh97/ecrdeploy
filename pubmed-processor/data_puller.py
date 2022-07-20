import os

from subprocess import check_call
from datetime import datetime
from dateutil.relativedelta import relativedelta

from constants import full_files_ftp_url, inc_files_ftp_url, file_pattern, storage_folder


def is_file_age_lt_x_months(file_path: str, num_months: int) -> bool:
    doc_date = datetime.fromtimestamp(os.stat(file_path).st_mtime)
    today_minus_x_months_dt = datetime.now() + relativedelta(months=-num_months)
    return doc_date > today_minus_x_months_dt


def ftp_puller(_type: str = "weekly"):
    ftp_url = inc_files_ftp_url if _type == "weekly" else full_files_ftp_url
    bash_command_wget = 'wget -r -np -nH --cut-dirs=50 -A "' + file_pattern + '" -P ' + storage_folder + ' ' + ftp_url
    try:
        check_call(bash_command_wget, shell=True)
        return True
    except Exception as e:
        print("Exception occured while downloading files from the server.")
        raise e
