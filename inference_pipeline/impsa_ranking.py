import abc
import tensorflow as tf
import torch
from tqdm import tqdm
import random
import pandas as pd
import time
import glob
import numpy as np
# import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import os

from data_loader import DataLoader as data_load
from smart_open import open as smart_open
import io

import aws_utils
import serverless_model
import sys

class IMPSAModel(object, metaclass=abc.ABCMeta):

    def __init__(self):
        self.use_gpu = False
        self.device = self.get_env_devicetype()

    def get_env_devicetype(self):
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
            self.use_gpu = True
        return device

    @abc.abstractmethod
    def train(self, **kwargs):
        raise NotImplementedError("user must define fly to use the base class")

    @abc.abstractmethod
    def predict(self, **kwargs):
        raise NotImplementedError("user must define fly to use the base class")


class RankingDLModel(IMPSAModel):
    def __init__(self, bert_model, cache_dir=None):
        super(RankingDLModel, self).__init__()
        self.data_root_folder = os.path.join("config")
        self.s3_bucket = os.getenv("S3_BUCKETNAME", "ias-ml-eng-impsa-dev")
        self.s3_model_prefix = os.getenv("MODEL_KEY_PATH", "ranking/models/rankingDL")
        if bert_model.lower().startswith("s3://"):
            # load from s3 to memory
            s3_bucketname = bert_model.replace("s3://", "").split("/")[0]
            s3_key_model = "/".join(bert_model.replace("s3://", "").split("/")[1:])
            S3MODEL = serverless_model.ServerlessS3Model(
                model_path=cache_dir,
                s3_bucket=s3_bucketname,
                file_prefix=s3_key_model
            )
            self.bert_classification_model = S3MODEL.model
            self.tokenizer = S3MODEL.tokenizer
        else:
            if cache_dir is None:
                self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
                self.bert_classification_model = BertForSequenceClassification.from_pretrained(
                    bert_model,
                    num_labels=2,
                    output_attentions=False,
                    output_hidden_states=False,
                )
            else:
                self.tokenizer = BertTokenizer.from_pretrained(bert_model, cache_dir=cache_dir, do_lower_case=True)
                self.bert_classification_model = BertForSequenceClassification.from_pretrained(
                    bert_model,
                    cache_dir=cache_dir,
                    num_labels=2,
                    output_attentions=False,
                    output_hidden_states=False,
                )

    def train(self, **kwargs):
        kwargs["use_gpu"] = self.use_gpu
        return self.cross_val_train(**kwargs)

    def predict(self, **kwargs):
        data_list = kwargs.get("data_list")
        dl = data_load(data_list=data_list)
        test_data = dl.load_full_train()
        kwargs["test_data"] = test_data
        kwargs.pop("data_list", None)

        predict_file = kwargs.pop("predict_file", None)
        # ---- Predict with K-Folds ----
        test_df = self.kfold_bert_predict(**kwargs)
        if predict_file is not None: test_df.to_csv(predict_file, index=False)
        return test_df.to_dict("records")

    def f1_score_func(self, labels, preds):
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return f1_score(labels_flat, preds_flat)

    def roc_auc_score_func(self, labels, preds):
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return roc_auc_score(labels_flat, preds_flat)

    def accuracy_score_calc(self, labels, preds):
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return accuracy_score(labels_flat, preds_flat)

    def get_saved_model_path(self, epoch, fold_no, biobert_models_path, model_alias):
        model_path = biobert_models_path + f"/Fold_{fold_no}_{model_alias}_epoch_{epoch}.pt"
        return model_path

    def evaluate_model(self, model, val_dataloader, use_gpu=True):
        """After the completion of each training epoch, measure the model's performance
        on our validation set.
        """
        # Put the model into the evaluation mode. The dropout layers are disabled during
        # the test time.
        model.eval()

        loss_val_total = 0
        predictions, true_vals = [], []

        # For each batch in our validation set...
        for batch in val_dataloader:
            # Load batch to GPU
            if use_gpu: batch = tuple(b.to("cuda") for b in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2],
            }

            # Compute logits
            with torch.no_grad():
                outputs = model(**inputs)

            loss = outputs[0]
            logits = outputs[1]
            loss_val_total += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = inputs["labels"].cpu().numpy()
            predictions.append(logits)
            true_vals.append(label_ids)

        loss_val_avg = loss_val_total / len(val_dataloader)

        predictions = np.concatenate(predictions, axis=0)
        true_vals = np.concatenate(true_vals, axis=0)
        val_acc = self.accuracy_score_calc(true_vals, predictions)
        val_f1 = self.f1_score_func(true_vals, predictions)

        ###
        pred_probs = []
        for i in range(len(predictions)):
            pred_probs.append(predictions[i][1])
        _, cora_val = self.cora(true_vals, pred_probs)
        ###
        return loss_val_avg, val_acc, val_f1, cora_val, predictions, true_vals

    def cora(self, actual, predicted):
        """
        Computes the Centroid of Relevant Abstracts 
    
        Parameters
        ----------
        actual : list
                A list predicted relevance (0/1) for each abstract 
        predicted : list
                A list of true relevance [0,..,1] for each abstract 
        Returns
        -------
        cora : double
            The Centroid of Relevant Abstracts 
        cost_red: double 
            The cost reduction %  
        """
        # check same len 
        assert len(actual) == len(predicted)
        # check actual is composed by 0 and 1 
        assert len([i for i in actual if i != 0 and i != 1]) == 0

        # compute the max cost reduction , cora_zero_cost_red, cora_max_cost_red
        rel_abs = len([i for i in actual if i == 1])
        if rel_abs == 0:
            raise Exception("N. relevant abstracts == 0!")
        tot_abs = len(actual)
        if tot_abs == 0:
            raise Exception("N. abstracts == 0!")
        max_cost_red = (tot_abs - rel_abs) / tot_abs
        cora_zero_cost_red = (len(actual) + 1) / 2
        cora_max_cost_red = (rel_abs + 1) / 2

        ## compute cora 
        arr = np.column_stack((predicted, actual))
        arr_sort = arr[np.argsort(arr[:, 0], )[::-1]]
        rel_list = arr_sort[:, 1].tolist()
        cora = 0
        for i, r in enumerate(rel_list):
            cora = cora + r * (i + 1)
        cora = cora / sum(rel_list)

        ## compute cost_red 
        cost_red = max_cost_red - (cora - cora_max_cost_red) * max_cost_red / (cora_zero_cost_red - cora_max_cost_red)

        ## 
        if cost_red < 0:
            cost_red = 0

        return cora, cost_red

    # Create a function to tokenize a set of texts
    def preprocessing_for_bert(self, data):
        """Perform required preprocessing steps for pretrained BERT.
        @param    data (np.array): Array of texts to be processed.
        @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
        @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                    tokens should be attended to by the model.
        """
        # Create empty lists to store outputs
        input_ids = []
        attention_masks = []

        # Load the BERT tokenizer
        # tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
        tokenizer = self.tokenizer

        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_data_train = tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=list(data),  # data.values,
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            return_attention_mask=True,  # Return attention mask
            padding="max_length",  # Pad sentence to max length
            max_length=256,  # Max length to truncate/pad
            truncation=True,
            return_tensors="pt",  # Return PyTorch tensor
        )

        input_ids = encoded_data_train["input_ids"]
        attention_masks = encoded_data_train["attention_mask"]

        return input_ids, attention_masks

    def train_data(self, biobert_models_path, model_alias,
                   train_dataloader, val_dataloader, epochs=4, evaluation=False,
                   batch_size=16, seed_val=1973, fold_no=0, use_gpu=True):

        """Train the BertClassifier model.
        """
        # model = BertForSequenceClassification.from_pretrained(
        #     model_name,
        #     num_labels=2,
        #     output_attentions=False,
        #     output_hidden_states=False,
        # )
        model = self.bert_classification_model

        if use_gpu: model.cuda()
        optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * epochs
        )

        random.seed(seed_val)
        np.random.seed(seed_val)
        if use_gpu: torch.cuda.manual_seed_all(seed_val)

        best_acc = 0.0
        best_epoch = 1
        best_epoch_file = ""

        # Start training loop
        print("Start training...\n")
        for epoch in tqdm(range(1, epochs + 1), desc=f"FOLD {fold_no}:"):
            # =======================================
            #               Training
            # =======================================
            # Print the header of the result table
            print(
                f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Val F1':^7}| {'Val CORA':^7} | {'Elapsed':^9}")
            print("-" * 80)

            # Measure the elapsed time of each epoch
            t0_epoch, t0_batch = time.time(), time.time()
            # Put the model into the training mode
            model.train()

            loss_train_total = 0

            progress_bar = tqdm(
                train_dataloader,
                desc="Epoch {:1d}".format(epoch),
                leave=False,
                disable=False,
            )

            # For each batch of training data...
            for batch in progress_bar:
                # Zero out any previously calculated gradients
                model.zero_grad()
                if use_gpu: device = torch.device("cuda")
                # Load batch to GPU
                batch = tuple(b.to(device) for b in batch)

                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[2],
                }

                outputs = model(**inputs)

                # Compute loss and accumulate the loss values
                loss = outputs[0]
                loss_train_total += loss.item()
                # Perform a backward pass to calculate gradients
                loss.backward()
                # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters and the learning rate
                optimizer.step()
                scheduler.step()

                progress_bar.set_postfix(
                    {"training_loss": "{:.3f}".format(loss.item() / len(batch))}
                )

                # Reset batch tracking variables
                t0_batch = time.time()

            # tqdm.write(f"\nEpoch {epoch}")

            # Calculate the average loss over the entire training data
            avg_train_loss = loss_train_total / len(train_dataloader)
            # tqdm.write(f"Training loss: {avg_train_loss}")

            # =======================================
            #               Evaluation
            # =======================================
            if evaluation == True:
                # After the completion of each training epoch, measure the model's performance
                # on our validation set.
                val_loss, val_accuracy, val_f1, cora_val, predictions, true_vals = self.evaluate_model(model,
                                                                                                       val_dataloader,
                                                                                                       use_gpu=use_gpu)

                # Print performance over the entire training data
                time_elapsed = time.time() - t0_epoch

                print(
                    f"{epoch:^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.4f} | {val_f1:^9.4f} |{cora_val:^9.4f} | {time_elapsed:^9.2f}")
                print("-" * 80)

            ##########
            best_acc = 0.0 if epoch == 1 else best_acc
            best_epoch = 1 if epoch == 1 else best_epoch
            best_epoch_file = "" if epoch == 1 else best_epoch_file
            ##########
            if val_accuracy > best_acc:
                prev_best_epoch_file = self.get_saved_model_path(
                    best_epoch, fold_no, biobert_models_path, model_alias
                )

                if os.path.exists(prev_best_epoch_file):
                    os.remove(prev_best_epoch_file)

                best_acc = val_accuracy
                best_epoch = epoch
                best_epoch_file = self.get_saved_model_path(
                    best_epoch, fold_no, biobert_models_path, model_alias
                )
                # After the completion of each training epoch, measure the model's performance
                # on our validation set.            
                _, _, _, _, best_predictions, true_vals = self.evaluate_model(model, val_dataloader, use_gpu=use_gpu)

                # best_predictions = np.argmax(best_predictions, axis=1)
                print(
                    f"\nEpoch: {best_epoch} - New best accuracy! Accuracy: {best_acc}\n\n\n"
                )

                # torch.save(model.state_dict(), biobert_models_path+f"/Fold_{fold_no}_{model_alias}_epoch_{epoch}.model")
                if not os.path.isdir(biobert_models_path): os.makedirs(biobert_models_path, exist_ok=True)
                torch.save(
                    model,
                    biobert_models_path + f"/Fold_{fold_no}_{model_alias}_epoch_{epoch}.pt",
                )
        # prediction_labels = np.argmax(best_predictions, axis=1)
        prediction_labels = np.argmax(best_predictions, axis=1).flatten()
        return model, best_acc, best_predictions, prediction_labels, true_vals

    def bert_predict(self, test_data, trained_model_path, batch_size=16, use_gpu=True):
        """Perform a forward pass on the trained BERT model to predict probabilities
        on the test set.
        """
        # ---------------------
        # Run `preprocessing_for_bert` on the test set
        print('Tokenizing data...')
        test_inputs, test_masks = self.preprocessing_for_bert(test_data["abstract"])

        # Create the DataLoader for our test set
        test_dataset = TensorDataset(test_inputs, test_masks)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

        # model_path = glob.glob(model_path).pop()
        print('Loading model...')
        model = torch.load(trained_model_path, map_location=self.device)
        # if use_gpu: model.cuda()
        # ---------------------
        # model_path = biobert_models_path + f"/Fold_{fold}_ms_pubmed_bert_epoch_*"
        # model_path = glob.glob(model_path).pop()
        # model = torch.load(model_path)
        # model.cuda()

        # Put the model into the evaluation mode. The dropout layers are disabled during
        # the test time.
        model.eval()

        all_logits = []
        # For each batch in our validation set...
        for batch in test_dataloader:
            # Load batch to GPU
            if use_gpu: batch = tuple(b.to("cuda") for b in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
            }

            # Compute logits
            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs[0]

            logits = logits.detach().cpu().numpy()
            all_logits.append(logits)

        predictions = np.concatenate(all_logits, axis=0)

        prediction_labels = np.argmax(predictions, axis=1).flatten()
        pred_probs = tf.nn.softmax(predictions)
        pred_probs = pred_probs.numpy()

        # label_0 = []
        label_1 = []

        for i in range(len(pred_probs)):
            # label_0.append(pred_probs[i][0])
            label_1.append(pred_probs[i][1])

        # test_data["pred_prob_0"] = label_0
        # test_data["pred_prob_1"] = label_1
        test_data_df = test_data.copy()
        test_data_df["relevance_prediction"] = label_1
        test_data_df["pred_label"] = prediction_labels

        return test_data_df

    def __load_model_from_s3(self, s3_uri):
        model = None
        with smart_open(s3_uri, 'rb') as f:
            buffer = io.BytesIO(f.read())
            model = torch.load(buffer, map_location=self.device)
        return model

    def kfold_bert_predict(self, test_data, trained_model_path, batch_size=16, use_gpu=True):
        """Perform a forward pass on the trained BERT model to predict probabilities
        on the test set.
        """
        test_df = test_data.copy()
        print("source data columns:{}".format(test_df.columns))
        print("source abstract:{}".format(list(test_df["abstract"])[0]))
        # ---------------------
        # Run `preprocessing_for_bert` on the test set
        print('Tokenizing data...')
        test_inputs, test_masks = self.preprocessing_for_bert(test_data["abstract"])

        # Create the DataLoader for our test set
        test_dataset = TensorDataset(test_inputs, test_masks)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

        # ---------------------
        # n_folds = 5
        # for fold in range(n_folds):
        fold = 0
        # test_df["pred_prob_0"] = 0.0
        # test_df["pred_prob_1"] = 0.0
        test_df["relevance_prediction"] = 0.0

        models_list = list()
        if not trained_model_path.lower().startswith("s3://"):
            models_list = glob.glob(os.path.join(trained_model_path, '*.pt'))
        else:
            if trained_model_path.lower().endswith(".pt"):
                models_list = [trained_model_path]
            else:
                s3_input_list = trained_model_path.replace("s3://", "").split("/")
                my_bucket = s3_input_list[0]
                my_prefix = "/".join(s3_input_list[1:])
                if not my_prefix.endswith("/"): my_prefix += "/"
                s3_client = aws_utils.S3Client()
                model_key_list = s3_client.list_objects(bucket=my_bucket, prefix=my_prefix)
                models_list = ["s3://{}/{}".format(my_bucket, x) for x in model_key_list]
                models_list = [x for x in models_list if x.lower().endswith(".pt")]

        for model_path in models_list:
            print('Loading model...:', model_path)
            if not model_path.startswith("s3://"):
                model = torch.load(model_path, map_location=self.device)
            else:
                model = self.__load_model_from_s3(s3_uri=model_path)
            # if use_gpu: model.cuda()
            # ---------------------
            # Put the model into the evaluation mode. The dropout layers are disabled during
            # the test time.
            model.eval()

            all_logits = []
            # For each batch in our validation set...
            for batch in test_dataloader:
                # Load batch to GPU
                if use_gpu: batch = tuple(b.to("cuda") for b in batch)

                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                }

                # Compute logits
                with torch.no_grad():
                    outputs = model(**inputs)

                logits = outputs[0]

                logits = logits.detach().cpu().numpy()
                all_logits.append(logits)

            predictions = np.concatenate(all_logits, axis=0)

            pred_probs = tf.nn.softmax(predictions)
            pred_probs = pred_probs.numpy()

            # label_0 = []
            label_1 = []

            for i in range(len(pred_probs)):
                # label_0.append(pred_probs[i][0])
                label_1.append(pred_probs[i][1])

            fold += 1
            # test_df["pred_prob_0"] = test_df["pred_prob_0"] + label_0
            # test_df["pred_prob_1"] = test_df["pred_prob_1"] + label_1
            test_df["relevance_prediction"] = test_df["relevance_prediction"] + label_1

        # test_df["pred_prob_0"] = test_df["pred_prob_0"]/len(models_list)
        # test_df["pred_prob_1"] = test_df["pred_prob_1"]/len(models_list)
        test_df["relevance_prediction"] = test_df["relevance_prediction"] / len(models_list)

        return test_df

    def cross_val_train(
            self,
            dataset_name,
            epochs,
            batch_size,
            seed_val,
            biobert_models_path,
            model_alias,
            n_folds,
            use_gpu=True
    ):
        ###################
        start_time = time.time()

        best_acc_list = []
        cora_score_list = []
        f1_score_list = []
        roc_auc_score_score_list = []
        output_df = pd.DataFrame()

        data_folder = os.path.join("config", dataset_name)
        data_file = None
        for f in os.listdir(data_folder):
            if f.lower().startswith("relevance"): data_file = os.path.join(data_folder, f)
        assert (data_file is not None)
        data_df = pd.read_csv(data_file, encoding="latin-1")
        data_list = data_df.to_dict("records")
        dl = data_load(data_list=data_list)
        df = dl.load_full_train()
        assert ("abstract" in df.columns)
        assert ("relevance_label" in df.columns)
        X = df["abstract"]
        y = df["relevance_label"]
        skf = model_selection.StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1973)
        fold_no = 0
        # Perform subsequent 80-20 train test split
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[test_idx]
            y_train, y_val = y[train_idx], y[test_idx]

            y_train = [int(i) for i in y_train]
            y_val = [int(i) for i in y_val]

            # ---------------------------------------------------
            # Run function `preprocessing_for_bert` on the train set and the validation set
            print('Tokenizing data...')
            train_inputs, train_masks = self.preprocessing_for_bert(X_train)
            val_inputs, val_masks = self.preprocessing_for_bert(X_val)
            # Convert other data types to torch.Tensor
            train_labels = torch.tensor(y_train)
            val_labels = torch.tensor(y_val)

            # Create the DataLoader for our training set
            train_data = TensorDataset(train_inputs, train_masks, train_labels)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

            # Create the DataLoader for our validation set
            val_data = TensorDataset(val_inputs, val_masks, val_labels)
            val_sampler = SequentialSampler(val_data)
            val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
            ##--------------------------------------------------
            model, best_acc, best_predictions, prediction_labels, true_vals = self.train_data(
                biobert_models_path, model_alias,
                train_dataloader, val_dataloader, epochs=epochs, evaluation=True,
                batch_size=batch_size, seed_val=seed_val, fold_no=fold_no, use_gpu=use_gpu)
            ##--------------------------------------------------
            pred_probs = tf.nn.softmax(best_predictions)
            pred_probs = pred_probs.numpy()

            fold_no = fold_no + 1

            pred_df = X_val.to_frame()
            pred_df.columns = ["abstract"]
            # label_0 = []
            label_1 = []

            for i in range(len(pred_probs)):
                # label_0.append(pred_probs[i][0])
                label_1.append(pred_probs[i][1])

            # results["pred_prob_0"] = label_0
            # results["pred_prob_1"] = label_1
            pred_df["relevance_prediction"] = label_1
            pred_df["actual_label"] = true_vals
            pred_df["pred_label"] = prediction_labels

            val_f1 = round(self.f1_score_func(true_vals, best_predictions), 4)
            val_roc_auc_score = round(self.roc_auc_score_func(true_vals, best_predictions), 4)
            _, cora_score = self.cora(
                actual=pred_df["actual_label"], predicted=pred_df["relevance_prediction"]
            )
            cora_score = round(cora_score, 4)
            output_df = pd.concat([output_df, pred_df])
            best_acc_list.append(best_acc)
            f1_score_list.append(val_f1)
            roc_auc_score_score_list.append(val_roc_auc_score)
            cora_score_list.append(cora_score)

        print(
            "******************************************K-FOLD CV Results******************************************"
        )
        print(">>> CORA          [Dev]:", cora_score_list)
        print(">>> Avg cora      [Dev]:", round(sum(cora_score_list) / len(cora_score_list), 4))
        print(">>> accuracy      [Dev]:", best_acc_list)
        print(">>> Avg accuracy  [Dev]:", round(sum(best_acc_list) / len(best_acc_list), 4))
        print(">>> AUC           [Dev]:", roc_auc_score_score_list)
        print(">>> Avg AUC       [Dev]:", round(sum(roc_auc_score_score_list) / len(roc_auc_score_score_list), 4))
        print(">>> F1_Score      [Dev]:", f1_score_list)
        print(">>> Avg F1        [Dev]:", round(sum(f1_score_list) / len(f1_score_list), 4))

        # output_df.to_csv(dataset_name + "_Predictions.csv")

        seconds = time.time() - start_time
        mins = seconds / 60
        hours = mins / 60
        days = hours / 24
        print("------>>>>>>> elapsed seconds: " + str(seconds))
        print("------>>>>>>> elapsed minutes: " + str(mins))
        print("------>>>>>>> elapsed hours: " + str(hours))
        print("------>>>>>>> elapsed days: " + str(days))

        return cora_score
