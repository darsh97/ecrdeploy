import boto3 
from boto3.dynamodb.conditions import Key, Attr
import os, sys, datetime
import traceback
from typing import Union

STR_DELIMETER="|"

class AWSClient():
    def __init__(self, profile_name=None, aws_access_key_id=None, aws_secret_access_key=None, region_name="us-east-1"):
        self.session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
            profile_name=profile_name
        )

class DynamoDBClient(AWSClient):
    def __init__(self, profile_name=None, aws_access_key_id=None, aws_secret_access_key=None, region_name="us-east-1"):
        super(DynamoDBClient, self).__init__(
            profile_name=profile_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        self.client = self.session.resource("dynamodb")


    def __remove_duplicate_table(self, table_list, keep_first=True):
        table_name_list = [x.table_name for x in table_list]
        if keep_first:
            return [i for n, i in enumerate(table_list) if i.table_name not in table_name_list[:n]]
        else:
            return [table_list[i] for i, x in enumerate(table_list) if x.table_name not in table_name_list[i+1:]]

        

    def list_tables(self, filter_keywords:Union[str,list]=None, exact_match=True):
        '''
            filter_keywords: filter used to filter the table name
            exact_match: True/False
                True: exact match, use = to filter the table name
                False: fuzzy match, user in to filter the table name
        '''
        all_tables = list(self.client.tables.all())
        # all_table_names = [x.table_name for x in all_tables]
        if filter_keywords is None: return all_tables
        my_filters = list()
        if isinstance(filter_keywords, str): my_filters.append(filter_keywords)
        elif isinstance(filter_keywords, list) and len(filter_keywords)>0: my_filters = filter_keywords
        if len(my_filters) == 0: return list()
        my_ret = list()
        # print([x.name.lower() for x in all_tables])
        for fk in my_filters:
            match_tables=list()
            if exact_match: match_tables = [x for x in all_tables if fk.lower() == x.table_name.lower()]
            else: match_tables = [x for x in all_tables if fk.lower() in x.table_name.lower()]
            # print(match_tables)
            if len(match_tables)>0: my_ret.extend(match_tables)
        if len(my_ret)>0: my_ret = self.__remove_duplicate_table(table_list=my_ret)
        return my_ret

    def create_dynamodb_table(self, table_definition:dict):
        db_client = self.client
        if table_definition is None or not isinstance(table_definition, dict): return None
        my_table_name = table_definition.get("tablename")
        if my_table_name is None:
            print("table name is not provided!")
            return None
        # my_tables = db_client.tables.filter()
        # my_tables = [x for x in list(my_tables) if x.table_name.lower() == my_table_name.lower()]
        my_tables = self.list_tables(filter_keywords=[my_table_name], exact_match=True)
        if my_tables is not None and len(my_tables)>0 : return my_tables[0]

        if table_definition.get("keyschema") is None or table_definition.get("attribute") is None:
            print("table definition is invalid!")
            print(table_definition) 
            return None
        # create a new table
        table = db_client.create_table(
                    TableName=my_table_name.lower(),
                    KeySchema=table_definition.get("keyschema"),
                    AttributeDefinitions=table_definition.get("attribute"),
                    ProvisionedThroughput=table_definition.get("throughput")
                )
        # Wait until the table exists.
        table.meta.client.get_waiter('table_exists').wait(TableName=my_table_name.lower())
        return table

    def delete_table(self, table_name):
        try:
            my_table = self.client.Table(table_name)
            # check if table exists or not
            my_table_arn = None
            try:
                my_table_arn = my_table.table_arn
            except Exception as exx:
                traceback.print_exc()
            if my_table_arn is None:
                print("table {} not exist, do nothing!".format(table_name))
                return True
            my_table.delete()
            return True
        except Exception as ex:
            traceback.print_exc()
            raise ex

    def persist_dynamodb(self, table_name:str, db_item:dict):
        db_client = self.client
        if db_client is None: 
            print("db connection is not ok!")
            return
        if table_name is None or len(table_name.strip()) == 0:
            print("table name is not specified!")
            return
        if not isinstance(db_item, dict): 
            print("item is invalid to persist")
            return
        try:
            table = self.create_dynamodb_table(
                table_definition={"tablename": table_name.lower()}
            )
            if table is None:
                print("table[{}] not exist!".format(table_name.lower()))
                
            # db_item["start_ts"]= datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            table.put_item(Item=db_item)
        except Exception as ex:
            traceback.print_exc()
            raise ex

    def query_dynamodb(self, table_name:str, query_dict:dict, limit=100):
        my_table = self.client.Table(table_name.lower())
        if query_dict is not None and len(query_dict.keys())>0:
            my_query_condition = None
            for k,v in query_dict.items():
                my_query_condition = Attr(k).eq(v) if my_query_condition is None else  (my_query_condition & Attr(k).eq(v))
            resp = my_table.scan(
                FilterExpression=my_query_condition,
                Limit=limit
            )
        else:
            resp = my_table.scan()
        if resp is None or "Items" not in resp.keys(): return list()
        return resp.get("Items")
        
    def batch_write_table(self, table_name:str, item_list:list):
        try:
            my_table = self.client.Table(table_name.lower())
            with my_table.batch_writer(overwrite_by_pkeys=['partition_key', 'sort_key']) as batch:
                for item in item_list:
                    batch.put_item(
                        Item=item
                    )
        except Exception as ex:
            traceback.print_exc()
            raise ex

class S3Client(AWSClient):
    def __init__(self, profile_name=None, aws_access_key_id=None, aws_secret_access_key=None, region_name="us-east-1"):
        super(S3Client, self).__init__(
            profile_name=profile_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        self.client = self.session.client("s3")
        self.resource = self.session.resource("s3")

    def upload_file(self, filename, bucket, s3_key):
        try:
            res = self.client.upload_file(
                Filename = filename,
                Bucket = bucket,
                Key = s3_key
            )
            ret = "s3://{}/{}".format(bucket, s3_key)
            return ret
        except Exception as ex:
            traceback.print_exc()
            return None

    def download_file(self, s3_key, bucket, filename):
        print("bucket:{}, s3_key:{}".format(bucket, s3_key))
        try:
            self.client.download_file(
                Bucket = bucket,
                Key = s3_key,
                Filename = filename
            )
            return True
        except Exception as ex:
            traceback.print_exc()
            # return False
            raise ex

    def download_file_from_s3(self, s3_path, output_folder):
        s3_input_list = s3_path.replace("s3://", "").split("/")
        if len(s3_input_list)<2:
            print("the input s3 file {} is invalid!")
            return None
        s3_bucket = s3_input_list[0]
        s3_key = "/".join(s3_input_list[1:])
        os.makedirs(output_folder, exist_ok=True)
        local_file = os.path.join(output_folder, s3_input_list[-1])
        download_flag = self.download_file(
            bucket=s3_bucket,
            s3_key=s3_key,
            filename=local_file
        )
        return local_file if download_flag else None

    def delete_file(self, s3_uri):
        try:
            s3_input_list = s3_uri.replace("s3://", "").split("/")
            bucket = s3_input_list[0]
            s3_key = "/".join(s3_input_list[1:])
            self.client.delete_object(
                Bucket=bucket,
                Key=s3_key
            )
            return True
        except Exception as ex:
            traceback.print_exc()
            raise ex

    def delete_folder(self, s3_uri):
        try:
            s3_input_list = s3_uri.replace("s3://", "").split("/")
            bucket_name = s3_input_list[0]
            s3_key = "/".join(s3_input_list[1:])
            bucket = self.resource.Bucket(bucket_name)
            bucket.objects.filter(Prefix=s3_key).delete()
            return True
        except Exception as ex:
            traceback.print_exc()
            raise ex

    def upload_obj(self, bucket, key, obj):
        pass

    def download_obj(self, bucket, key, obj):
        pass

    def download_obj_from_s3(self, s3_uri):
        pass

    def list_objects(self, bucket, prefix:str="")->list:
        ret = list()
        mybucket = self.resource.Bucket(name=bucket)
        for page in mybucket.objects.filter(Prefix=prefix).pages():
            for obj in page:
                print(obj.key)
                ret.append(obj.key)
        return ret

if __name__=="__main__":
    s3_client = S3Client()
    my_obj_list = s3_client.list_objects(bucket="ivy-lambda-data", prefix="videos/en-US")
    print(my_obj_list)
    assert(len(my_obj_list)>0)