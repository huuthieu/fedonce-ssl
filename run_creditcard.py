import os.path
import wget
import bz2
import shutil
import numpy as np

import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, log_loss
from sklearn.model_selection import KFold

import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from utils.data_utils import load_data_cross_validation, load_movielens, load_creditcardfraud
from model.fl_model import VerticalFLModel
from model.single_party_model import SingleParty
from model.split_nn_model import SplitNNModel

import shap

os.environ["PROTOBUF_PYTHON_IMPLEMENTATION"] = "python"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"


def calculate_cls_weight(label):
    labels_tensor = torch.tensor(label, dtype=torch.float32)

    class_frequencies = torch.bincount(labels_tensor.long()) / len(labels_tensor)

    # Calculate class weights as the inverse of class frequencies
    class_weights = 1.0 / class_frequencies

    # Normalize the class weights to sum to 1
    class_weights /= class_weights.sum()

    # Convert the class weights to a PyTorch tensor with shape (2,)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    
    return class_weights_tensor


def calculate_cls_weight_bin(label):
    labels_tensor = torch.tensor(label, dtype=torch.float32)
    print(labels_tensor)
    num_positive_samples = (labels_tensor == 1).sum().item()
    num_negative_samples = (labels_tensor == 0).sum().item()
    pos_weight = torch.tensor(num_negative_samples / num_positive_samples, dtype=torch.float32)
    return pos_weight


def train_split_nn(num_parties, model_path):

    xs_train_val, y_train_val, xs_test, y_test = load_creditcardfraud("data/creditcard/creditcard.csv", use_cache = False,
                                                        test_rate = 0.1)

    
    acc_list = []
    f1_list = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)

    for i, (train_idx, val_idx) in enumerate(kfold):
        print("Cross Validation Fold {}".format(i))
        xs_train = [data[train_idx] for data in xs_train_val]
        y_train = y_train_val[train_idx]
        xs_val = [data[val_idx] for data in xs_train_val]
        y_val = y_train_val[val_idx]
        start = datetime.now()
        name = "splitnn_creditcard_party_{}_fold_{}".format(num_parties, i)
        writer = SummaryWriter("runs/{}".format(name))
        # ncf_counts = [counts[:2], counts[2:]]
        embed_dims = [[32, 32], [1, 4, 10, 4, 15, 5]]
        # class_weights = calculate_cls_weight_bin(y_train)
        class_weights = None
        aggregate_model = SplitNNModel(
            num_parties=num_parties,
            name=name,
            num_epochs=100,
            local_hidden_layers=[32, 16],
            local_output_size=3,
            lr=3e-5,
            agg_hidden_layers=[10],
            batch_size=128,
            weight_decay=1e-5,
            writer=writer,
            device='cuda:{}'.format("0"),
            update_target_freq=1,
            task='binary_classification',
            test_batch_size=1000,
            test_freq=1,
            cuda_parallel=False,
            model_type='fc',
            optimizer='adam',
            privacy=None,
            batches_per_lot=5,
            epsilon=1,
            delta=1.0 / xs_train[0].shape[0],
            num_workers=0,
            cls_weight=class_weights,
            model_path=model_path,    
        )
        val_acc, val_f1, rmse, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val, xs_test, y_test, use_cache=False)
        y_pred_test, y_score_test = aggregate_model.eval(xs_test)
        test_f1 = f1_score(y_test, y_pred_test)
        f1_list.append(test_f1)
        print(aggregate_model.params)
        print("-------------------------------------------------")
        time_min = (datetime.now() - start).seconds / 60
        print("Time(min) {}: ".format(time_min))
    print("Best f1={}".format(f1_list))

def train_fed_once(num_parties):
    score_summary = []
    xs_train_val, y_train_val, xs_test, y_test = load_creditcardfraud("data/creditcard/creditcard.csv", use_cache = False,
                                                        test_rate = 0.1)

    for party_id in range(num_parties):
        kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)
        f1_list = []
        for i, (train_idx, val_idx) in enumerate(kfold):
            print("Cross Validation Fold {}".format(i))
            xs_train = [data[train_idx] for data in xs_train_val]
            y_train = y_train_val[train_idx]
            xs_val = [data[val_idx] for data in xs_train_val]
            y_val = y_train_val[val_idx]
            print("Train shape: {}".format(xs_train[0].shape))
            print("Val shape: {}".format(xs_val[0].shape))
            start = datetime.now()
            name = "fedonce_creditcard_party_{}_active_{}_fold_{}".format(num_parties, party_id, i)
            writer = SummaryWriter("runs/{}".format(name))
            # class_weights = calculate_cls_weight_bin(y_train)
            class_weights = None
            aggregate_model = VerticalFLModel(
                num_parties=num_parties,
                active_party_id=party_id,
                name=name,
                full_name=name,
                num_epochs=40 if party_id == 0 else 40,
                num_local_rounds=30 if party_id == 0 else 30,
                local_lr=1e-4,
                local_hidden_layers=[128] if party_id == 0 else [128],
                local_batch_size=64,
                local_weight_decay=1e-5,
                local_output_size=14 if party_id == 0 else 14,
                num_agg_rounds=1,
                agg_lr=5e-4 if party_id == 0 else 5e-4,
                agg_hidden_layers=[32, 32] if party_id == 0 else [32, 32],
                agg_batch_size=64,
                agg_weight_decay=1e-5 if party_id == 0 else 1e-5,
                writer=writer,
                device='cuda:{}'.format("0"),
                update_target_freq=1,
                task='binary_classification',
                # n_classes = 2,
                test_batch_size=1000,
                test_freq=1,
                cuda_parallel=False,
                n_channels=1,
                model_type='fc',
                optimizer='adam',
                privacy=None,
                num_workers=0,
                cls_weight = class_weights)

            _, _, rmse, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val, xs_test, y_test, use_cache=False)
            y_test_score = aggregate_model.predict_agg(xs_test)
            y_test_pred = np.where(y_test_score > 0.5, 1, 0)
            print("Sum of pred: ")
            print(np.sum(y_test_pred))
            print("Sum of y_test: ")
            print(np.sum(y_test))

            # expand axis 1
            # y_test = np.expand_dims(y_test, axis=1)

            test_f1 = f1_score(y_test, y_test_pred)
            f1_list.append(test_f1)
            print(aggregate_model.params)
            time_min = (datetime.now() - start).seconds / 60
            print("Time(min) {}: ".format(time_min))
        score_summary.append(f1_list)
        print("F1 for active party {}".format(party_id) + str(f1_list))
        print("-------------------------------------------------")

    for i, result in enumerate(score_summary):
        print("Party {}: F1={}".format(i, result))

def train_combine():
    num_parties = 1
    xs_train_val, y_train_val, xs_test, y_test = load_creditcardfraud("data/creditcard/creditcard.csv",num_parties = num_parties, use_cache = False,
                                                            test_rate = 0.1)
    x_train_val = np.concatenate(xs_train_val, axis=1)
    x_test = np.concatenate(xs_test, axis=1)
    print(x_train_val.shape)
    print(x_test.shape)
    kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)
    f1_list = []
    for i, (train_idx, val_idx) in enumerate(kfold):
        print("Cross Validation Fold {}".format(i))
        x_train = x_train_val[train_idx]
        y_train = y_train_val[train_idx]
        x_val = x_train_val[val_idx]
        y_val = y_train_val[val_idx]
        name = "combine_movielens_fold_{}".format(i)
        writer = SummaryWriter("runs/{}".format(name))
        single_model = SingleParty(
            party_id=0,
            num_epochs=100,
            lr=1e-4,
            hidden_layers=[64, 32,8],
            batch_size=128,
            weight_decay=1e-4,
            writer=writer,
            device='cuda:0',
            task="binary_classification",
            test_batch_size=1000,
            test_freq=1,
            model_type='fc',
            optimizer='adam',
            cuda_parallel=False,
            n_channels=1,
            n_classes = 1
        )
        _, _, _, _ = single_model.train(x_train, y_train, x_val, y_val, x_test, y_test)
        y_pred_test, y_score_test = single_model.predict(x_test)
        test_f1 = f1_score(y_test, y_pred_test)
        f1_list.append(test_f1)
        print(single_model.params)
    explainer = shap.Explainer(single_model.predict_score, x_train)
    shap_values = explainer.shap_values(x_test)
    feature_importance = np.abs(shap_values).mean(axis=0)
    print("Feature importance: ", feature_importance)   
    print("-------------------------------------------------")
    print("Best accuracy={}".format(f1_list))

if __name__ == "__main__":
    train_fed_once(2)
#     train_split_nn(num_parties = 2, model_path = "saved_model")
#     train_combine()