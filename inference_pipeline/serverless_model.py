from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch
import boto3
import os
import tarfile
import io
import base64
import json
import re

s3 = boto3.client('s3')


class ServerlessS3Model:
    def __init__(self, model_path, s3_bucket, file_prefix, output_hidden_states=True):
        self.device = self.get_env_devicetype()
        self.output_hidden_states = output_hidden_states
        self.download_config(model_path=model_path, s3_bucket=s3_bucket, file_prefix=file_prefix)
        self.model, self.tokenizer = self.from_pretrained(
            model_path, s3_bucket, file_prefix)

    def get_env_devicetype(self):
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
            self.use_gpu = True
        return device

    def from_pretrained(self, model_path: str, s3_bucket: str, file_prefix: str):
        file_ext = os.path.basename(file_prefix).split(".")[-1]
        file_ext = file_ext.lower()
        if file_ext in ["tar", "gz"]:
            model = self.load_model_from_s3_tar(model_path, s3_bucket, file_prefix)
        else:
            model = self.load_model_from_s3(model_path, s3_bucket, file_prefix)
        tokenizer = self.load_tokenizer(model_path)
        return model, tokenizer

    def load_model_from_s3_tar(self, model_path: str, s3_bucket: str, file_prefix: str):
        if model_path and s3_bucket and file_prefix:
            obj = s3.get_object(Bucket=s3_bucket, Key=file_prefix)
            bytestream = io.BytesIO(obj['Body'].read())
            tar = tarfile.open(fileobj=bytestream, mode="r:gz")
            config = AutoConfig.from_pretrained(f'{model_path}/config.json')
            config.output_hidden_states=self.output_hidden_states
            for member in tar.getmembers():
                if member.name.endswith(".bin"):
                    f = tar.extractfile(member)
                    state = torch.load(io.BytesIO(f.read()), map_location=self.device)
                    model = AutoModel.from_pretrained(
                        pretrained_model_name_or_path=None, state_dict=state, config=config)
            return model
        else:
            raise KeyError('No S3 Bucket and Key Prefix provided')

    def load_model_from_s3(self, model_path: str, s3_bucket: str, file_prefix: str):
        s3_keys = self.list_objects(s3_bucket=s3_bucket, file_prefix=file_prefix)
        bin_files = [x for x in s3_keys if x.endswith(".bin")]
        if model_path and s3_bucket and file_prefix and len(bin_files)>0:
            obj = s3.get_object(Bucket=s3_bucket, Key=bin_files[0])
            bytestream = io.BytesIO(obj['Body'].read())
            state = torch.load(bytestream, map_location=self.device)
            config = AutoConfig.from_pretrained(f'{model_path}/config.json')
            config.output_hidden_states=self.output_hidden_states
            model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=None, state_dict=state, config=config)
            return model
        else:
            raise KeyError('No S3 Bucket and Key Prefix provided')

    def load_tokenizer(self, model_path: str):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return tokenizer

    def list_objects(self, s3_bucket: str, file_prefix: str="")->list:
        ret = list()
        my_objs = s3.list_objects(Bucket=s3_bucket, Prefix=file_prefix)
        if my_objs is None or not isinstance(my_objs, dict) or "Contents" not in my_objs.keys(): return ret 
        ret = [x["Key"] for x in my_objs["Contents"]]
        return ret


    def download_config(self, model_path: str, s3_bucket: str, file_prefix: str):
        keys_to_download = list()

        if file_prefix is None or len(file_prefix.strip()) == 0:
            keys_to_download = list()
        elif not file_prefix.endswith("/"): 
            keys_to_download = [file_prefix]
        else:
            keys_to_download = self.list_objects(s3_bucket=s3_bucket, file_prefix=file_prefix)
            keys_to_download = [x for x in keys_to_download if x.endswith(".json") or x.endswith(".txt")]

        print("model config to download:{}".format(keys_to_download))
        if len(keys_to_download)>0 and not os.path.isdir(model_path): os.makedirs(model_path, exist_ok=True)
        for k in keys_to_download:
            obj = s3.get_object(Bucket=s3_bucket, Key=k)
            with open(os.path.join(model_path, os.path.basename(k)), "wb") as f:
                f.write(obj['Body'].read())

    # def encode(self, question, context):
    #     encoded = self.tokenizer.encode_plus(question, context)
    #     return encoded["input_ids"], encoded["attention_mask"]

    # def decode(self, token):
    #     answer_tokens = self.tokenizer.convert_ids_to_tokens(
    #         token, skip_special_tokens=True)
    #     return self.tokenizer.convert_tokens_to_string(answer_tokens)

    # def predict(self, input_):
    #     """
    #     This is used for summmarization of text.
    #     :params input_: (String) input text like a pubmed abstract
    #     """

    #     return model(input_)