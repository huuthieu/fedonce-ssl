from utils.data_utils import load_data_cross_validation, load_data_train_test, get_random_noisy_row, load_creditcardfraud, load_uci
from model.fl_model import VerticalFLModel
from model.fl_model_ssl import VerticalFLModel as VerticalFLModelSSL
from model.single_party_model import SingleParty
from model.split_nn_model import SplitNNModel
from model.models import FC

from torch.utils.tensorboard import SummaryWriter
from sklearn.inspection import permutation_importance
from joblib import Parallel, delayed
import torch

import os.path
import wget
import bz2
import shutil
import zipfile
import numpy as np

import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


## FedOnce
def train_fedonce(remove_ratio = 0.1):
    num_parties = 2
    xs_train_val, y_train_val, xs_test, y_test = load_uci(test_rate = 0.1)
    active_party = 0
    print("Active party {} starts training".format(active_party))
    score_list = []
    f1_summary = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)

    for i, (train_idx, val_idx) in enumerate(kfold):
        print("Cross Validation Fold {}".format(i))
        xs_train = [data[train_idx] for data in xs_train_val]
        y_train = y_train_val[train_idx]
        xs_val = [data[val_idx] for data in xs_train_val]
        y_val = y_train_val[val_idx]
        print("Train shape: {}".format(xs_train[0].shape))
        print("Val shape: {}".format(xs_val[0].shape))
        model_name = "vertical_fl_phishing_party_{}_fold_{}".format(num_parties, i)
        name = "{}_active_{}".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            num_epochs=100,
            num_local_rounds=100,
            local_lr=3e-4,
            local_hidden_layers=[50, 30],
            local_batch_size=100,
            local_weight_decay=1e-5,
            local_output_size=10,
            num_agg_rounds=1,
            agg_lr=1e-4,
            agg_hidden_layers=[10],
            agg_batch_size=100,
            agg_weight_decay=1e-4,
            writer=writer,
            device='cuda:0',
            update_target_freq=1,
            task='binary_classification',
            n_classes=10,
            test_batch_size=1000,
            test_freq=1,
            cuda_parallel=False,
            n_channels=1,
            model_type='fc',
            optimizer='adam',
            privacy=None,
            batches_per_lot=5,
            epsilon=1,
            delta=1.0/xs_train[0].shape[0]
        )
        acc, _, _, _  = aggregate_model.train(xs_train, y_train, xs_val, y_val, xs_test, y_test, use_cache=False)
        y_test_score = aggregate_model.predict_agg(xs_test)
        y_test_pred = np.where(y_test_score > 0.5, 1, 0)
        test_f1 = f1_score(y_test, y_test_pred)

        print("Active party {} finished training.".format(active_party))
        score_list.append(acc)
        f1_summary.append(test_f1)
        print(aggregate_model.params)
    print("Accuracy for active party {}".format(active_party) + str(score_list))
    print("F1 for active party {} with: {}".format(active_party, str(f1_summary)))
    mean = np.mean(score_list)
    std = np.std(score_list)
    out = "Party {}: Accuracy mean={}, std={}".format(active_party, mean, std)
    print(out)

    mean = np.mean(f1_summary)
    std = np.std(f1_summary)
    out = "Party {}: Accuracy mean={}, std={}".format(active_party, mean, std)
    print(out)
    return mean, std

def train_fedonce_ssl(remove_ratio = 0.1, ssl = True, unlign_ratio = 0.5):

    num_parties = 2
    xs_train_val, y_train_val, xs_test, y_test = load_uci(test_rate = 0.1)

    active_party = 0
    print("Active party {} starts training".format(active_party))
    score_list = []
    f1_summary = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)

    for i, (train_idx, val_idx) in enumerate(kfold):
        print("Cross Validation Fold {}".format(i))
        xs_train = [data[train_idx] for data in xs_train_val]
        y_train = y_train_val[train_idx]
        xs_val = [data[val_idx] for data in xs_train_val]
        y_val = y_train_val[val_idx]
        print("Train shape: {}".format(xs_train[0].shape))
        print("Val shape: {}".format(xs_val[0].shape))
        model_name = "vertical_fl_phishing_party_{}_fold_{}".format(num_parties, i)
        name = "{}_active_{}".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))

        if ssl:
            unalign_index = get_random_noisy_row(xs_train[0], unlign_ratio)       

        aggregate_model = VerticalFLModelSSL(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            num_epochs=100,
            num_local_rounds=100,
            local_lr=3e-4,
            local_hidden_layers=[50, 30],
            local_batch_size=100,
            local_weight_decay=1e-5,
            local_output_size=10,
            num_agg_rounds=1,
            agg_lr=1e-4,
            agg_hidden_layers=[10],
            agg_batch_size=100,
            agg_weight_decay=1e-4,
            writer=writer,
            device='cuda:0',
            update_target_freq=1,
            task='binary_classification',
            n_classes=10,
            test_batch_size=1000,
            test_freq=1,
            cuda_parallel=False,
            n_channels=1,
            model_type='fc',
            optimizer='adam',
            privacy=None,
            batches_per_lot=5,
            epsilon=1,
            delta=1.0/xs_train[0].shape[0]
        )
        acc, _, _, _  = aggregate_model.train(xs_train, y_train, xs_val, y_val, ssl = ssl, unalign_index = unalign_index)
        y_test_score = aggregate_model.predict_agg(xs_test)
        y_test_pred = np.where(y_test_score > 0.5, 1, 0)
        test_f1 = f1_score(y_test, y_test_pred)

        print("Active party {} finished training.".format(active_party))
        score_list.append(acc)
        f1_summary.append(test_f1)
        print(aggregate_model.params)
    print("Accuracy for active party {}".format(active_party) + str(score_list))
    print("F1 for active party {} with: {}".format(active_party, str(f1_summary)))
    mean = np.mean(score_list)
    std = np.std(score_list)
    out = "Party {}: Accuracy mean={}, std={}".format(active_party, mean, std)
    print(out)

    mean = np.mean(f1_summary)
    std = np.std(f1_summary)
    out = "Party {}: Accuracy mean={}, std={}".format(active_party, mean, std)
    print(out)
    return mean, std


# combine
def train_combine(remove_ratio = 0.1):
    num_parties = 2
    xs_train_val, y_train_val, xs_test, y_test = load_uci(test_rate = 0.1)

    active_party = 0
    print("Active party {} starts training".format(active_party))
    score_list = []
    f1_summary = []
    x_train_val = np.concatenate(xs_train_val, axis=1)
    print("x_train_val shape: {}".format(x_train_val.shape))
    print("ratio of positive samples: {}".format(np.sum(y_train_val) / len(y_train_val)))

    x_test = np.concatenate(xs_test, axis=1)
    print("x_test shape: {}".format(x_test.shape))
    print("ratio of positive samples: {}".format(np.sum(y_test) / len(y_test)))


    kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)

    f1_summary = []
    acc_summary = []
    for i, (train_idx, val_idx) in enumerate(kfold):
        print("Cross Validation Fold {}".format(i))
        name = "combine_phishing_fold_{}".format(i)
        x_train = x_train_val[train_idx]
        y_train = y_train_val[train_idx]
        x_val = x_train_val[val_idx]
        y_val = y_train_val[val_idx]
        writer = SummaryWriter("runs/{}".format(name))
        single_model = SingleParty(
            party_id=0,
            num_epochs=100,
            lr=1e-4,
            hidden_layers=[100, 50],
            batch_size=100,
            weight_decay=1e-4,
            writer=writer,
            device='cuda:0',
            task="binary_classification",
            n_classes=10,
            test_batch_size=1000,
            test_freq=1,
            n_channels=1,
            model_type='fc',
            optimizer='adam',
            cuda_parallel=False
        )
        acc, f1, _,_ = single_model.train(x_train, y_train, x_test, y_test)
        acc_summary.append(acc)
        f1_summary.append(f1)
        print(single_model.params)

    print("Accuracy summary: " + repr(acc_summary))
    print("F1 score summary: " + repr(f1_summary))
    for i, result in enumerate(acc_summary):
        mean = np.mean(result)
        std = np.std(result)
        print("Party {}: Accuracy mean={}, std={}".format(i, mean, std))
    for i, result in enumerate(f1_summary):
        mean = np.mean(result)
        std = np.std(result)
        print("Party {}: F1-score mean={}, std={}".format(i, mean, std))


if __name__ == '__main__':
    # train_combine(remove_ratio=0.6)
    train_fedonce_ssl(unlign_ratio = 0.9)