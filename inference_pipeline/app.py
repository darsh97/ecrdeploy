import json, os, urllib, datetime, uuid, traceback, time, math
import pandas as pd
from impsa_ranking import RankingDLModel
import aws_utils
import boto3
from boto3.dynamodb.conditions import Key, Attr
import csv

s3_client = aws_utils.S3Client()
s3 = boto3.client('s3')

S3_BUCKETNAME = os.getenv("S3_BUCKETNAME", "")
PREDICT_DATA_S3_PATH = os.getenv("PREDICT_DATA_S3_PATH", "")
MODEL_KEY_PATH = os.getenv("MODEL_KEY_PATH", "")
from functools import wraps
import timeit


def timer(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()
        rv = f(*args, **kwargs)
        print(f"{f.__name__} took {timeit.default_timer() - start_time}")
        return rv

    return wrapper


@timer
def predict_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))

    data_list = None

    bucket, key = None, None

    if data_list is None:
        content = os.path.join("datastore",
                               "pubmed_rsv_20220713_20220719_1a7716c4d0bd4321948782349e421bfb_20220720085325_16f0de2a-4296-4d46-871b-00568d5703d5.json")
        input_data = json.load(open(content))
        data_list = input_data["data"]

    if "trained_model_path" in event.keys():
        biobert_models_path = event["trained_model_path"]
    else:
        biobert_models_path = "s3://codebucket234/"
    # model_name = event['model_name'] if "model_name" in event.keys() else "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    batch_size = 16

    predict_file = os.path.join("kfolder-bert-model-predict.csv")

    if data_list is None or len(data_list) == 0:
        # do nothing when input data is blank, but need to trigger next step to generate blank html for mail merge
        # my_df_tmp = pd.DataFrame()
        # my_df_tmp.to_csv(predict_file, index=False)
        with open(predict_file, "w", encoding="utf-8") as f:
            f.write("title,abstract,relevance_prediction")
    else:
        model = RankingDLModel(bert_model=model_name,
                               cache_dir=os.path.join("/datastore", os.path.basename(model_name.strip("/"))))
        predict_params_dict = {
            "data_list": data_list,
            # "model_name":model_name,
            "trained_model_path": biobert_models_path,
            "batch_size": batch_size,
            "use_gpu": model.use_gpu,
            "predict_file": predict_file
        }

        ret = model.predict(**predict_params_dict)

    if os.path.isfile(predict_file):
        # predict file created, print contents out
        print_csv_file(predict_file)


def print_csv_file(csv_file):
    filename = csv_file

    # initializing the titles and rows list
    fields = []
    rows = []

    # reading csv file
    with open(filename, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        # extracting field names through first row
        fields = next(csvreader)

        # extracting each data row one by one
        for i, row in enumerate(csvreader):
            if i > 10:
                break
            rows.append(row)

        # get total number of rows
        print("Total no. of rows: %d" % (csvreader.line_num), flush=True)

    # printing the field names
    print('Field names are:' + ', '.join(field for field in fields), flush=True)

    # printing first 5 rows
    print('\nFirst 5 rows are:\n', flush=True)
    for row in rows[:10]:
        # parsing each column of a row
        for col in row:
            print("%10s" % col, end=" ", flush=True),
        print('\n', flush=True)


predict_handler({}, {})
