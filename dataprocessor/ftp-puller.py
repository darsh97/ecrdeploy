from subprocess import check_call
import timeit

file_pattern = '*.xml.gz'
storage_folder = 'datastore/'

full_files_ftp_url = 'ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/*'
inc_files_ftp_url = 'ftp://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/*'


def ftp_puller():
    bash_command_wget = 'wget -r -np -nH --cut-dirs=50 -A "' + file_pattern + '" -P ' + storage_folder + ' ' + full_files_ftp_url
    try:
        check_call(bash_command_wget, shell=True)
    except Exception as e:
        print("Exception occured while downloading files from the server.")
        raise e


start = timeit.default_timer()
ftp_puller()
print(timeit.default_timer() - start)