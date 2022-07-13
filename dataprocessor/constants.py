import json

with open("config.json", "r") as config_file:
    config_data = json.load(config_file)

    s3_SOURCE_BUCKET = config_data["s3_source_bucket"]
    s3_DESTINATION_BUCKET = config_data["s3_destination_bucket"]

DATASTORE_PATH: str = "datastore"