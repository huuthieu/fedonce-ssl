from utils.data_utils import load_data_cross_validation, load_data_train_test, get_random_noisy_row, load_creditcardfraud
# from model.fl_model import VerticalFLModel
from model.fl_model_ssl import VerticalFLModel
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



# combine
def train_combine(remove_ratio = 0.1):
    num_parties = 1
    xs_train_val, y_train_val, xs_test, y_test = load_creditcardfraud("data/creditcard/creditcard.csv",num_parties = num_parties, use_cache = False,
                                                                test_rate = 0.1,
                                                                remove_ratio = remove_ratio,
                                                                )
    x_train_val = np.concatenate(xs_train_val, axis=1)
    print("x_train_val shape: {}".format(x_train_val.shape))
    print("ratio of positive samples: {}".format(np.sum(y_train_val) / len(y_train_val)))

    x_test = np.concatenate(xs_test, axis=1)
    print("x_test shape: {}".format(x_test.shape))
    print("ratio of positive samples: {}".format(np.sum(y_test) / len(y_test)))


    kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)

    f1_summary = []
    acc_summary = []
    for party_id in range(num_parties):
        print("Party {} starts training".format(party_id))
        acc_list = []
        f1_list = []
        for i, (train_idx, val_idx) in enumerate(kfold):
            print("Cross Validation Fold {}".format(i))
            name = "combine_phishing_fold_{}".format(i)
            x_train = x_train_val[train_idx]
            y_train = y_train_val[train_idx]
            x_val = x_train_val[val_idx]
            y_val = y_train_val[val_idx]
            writer = SummaryWriter("runs/{}".format(name))
            single_model = SingleParty(
                party_id=party_id,
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
            acc_list.append(acc)
            f1_list.append(f1)
            print(single_model.params)

        f1_summary.append(f1_list)
        acc_summary.append(acc_list)
        print("Accuracy for party {}".format(party_id) + str(acc_list))
        print("F1 score for party {}".format(party_id, str(f1_list)))
        print("-------------------------------------------------")
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
    train_combine(remove_ratio=0.6)
    