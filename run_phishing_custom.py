from utils.data_utils import load_data_cross_validation
from model.fl_model import VerticalFLModel
from model.fl_model_dae import VerticalFLModel as VerticalFLModelDae

from model.simple_fl_model import PCAVerticalFLModel
from model.single_party_model import SingleParty
from model.split_nn_model import SplitNNModel
from utils.split_train_test import split_train_test

from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, precision_score, recall_score

import joblib
from joblib import Parallel, delayed

from torch.utils.tensorboard import SummaryWriter

import os.path
import wget
import bz2
import shutil
import numpy as np

import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', type=int, default=0, help='Index of GPU')
args = parser.parse_args()


if not os.path.isdir("data"):
    os.mkdir("data")
if not os.path.isfile("data/phishing"):
    print("Downloading phishing data")
    wget.download("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/phishing",
                  "data/phishing")

if not os.path.isfile("data/phishing.train") or not os.path.isfile("data/phishing.test"):
    split_train_test("phishing", file_type='libsvm', test_rate=0.1)


# FedOnce
def train_fedonce(active_party = 0,  k_percent = 100,  k1_percent = 100, 
                  random_state = 10, reps = 0.0):

    num_parties = 2
    x_scaler_wrapper = []
    y_scaler_wrapper = []
    x_normalizer_wrapper = []
    cross_valid_data = load_data_cross_validation("phishing.train", num_parties=num_parties,
                                                file_type='libsvm', n_fold=5, use_cache=False,
                                                x_scaler_wrapper=x_scaler_wrapper, y_scaler_wrapper=y_scaler_wrapper,
                                                x_normalizer_wrapper=x_normalizer_wrapper,
                                                random_state = random_state)
    xs_test, y_test = load_data_cross_validation("phishing.test", num_parties=num_parties,
                                                file_type='libsvm', n_fold=1, use_cache=False,
                                                x_scaler_wrapper=x_scaler_wrapper, y_scaler_wrapper=y_scaler_wrapper,
                                                x_normalizer_wrapper=x_normalizer_wrapper,
                                                random_state=random_state)[0]
    acc_summary = []
    f1_summary = []

    prec_list = []
    recall_list = []
    score_list = []
    
    acc_list = []
    f1_list = []
    for i, (xs_train, y_train, xs_val, y_val) in enumerate(cross_valid_data):
        start = datetime.now()
        print("Cross Validation Fold {}".format(i))
        print("Active Party is {}".format(active_party))
        model_name = "fedonce_phishing_party_{}_fold_{}".format(num_parties, i)
        name = "{}_active_{}_k1_{}_k_{}_random_state_{}".format(model_name, active_party, k1_percent, k_percent, random_state)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=name,
            full_name=name,
            num_epochs=300,
            num_local_rounds=100,
            local_lr=1e-4,
            local_hidden_layers=[30, 30],
            local_batch_size=100,
            local_weight_decay=1e-5,
            local_output_size=48,
            num_agg_rounds=1,
            agg_lr=1e-4,
            agg_hidden_layers=[30],
            agg_batch_size=100,
            agg_weight_decay=1e-4,
            writer=writer,
            device='cuda:{}'.format(args.gpu),
            update_target_freq=1,
            task='binary_classification',
            n_classes=10,
            test_batch_size=1000,
            test_freq=1,
            cuda_parallel=False,
            n_channels=1,
            model_type='fc',
            optimizer='adam',
            num_workers=0
        )
        selected_features = []
        if k1_percent < 100:
            acc, _, _, _, selected_features  = aggregate_model.train(xs_train, y_train, xs_val, y_val, k_percent = k1_percent)
        else:                                        
            acc, f1, _, _, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val)
        y_score_test = aggregate_model.predict_agg(xs_test, selection_features = selected_features)
        y_pred_test = np.where(y_score_test > 0.5, 1, 0)
        
        test_acc = accuracy_score(y_test, y_pred_test)
        test_f1 = f1_score(y_test, y_pred_test)

        test_prec = precision_score(y_test, y_pred_test)
        test_recall = recall_score(y_test, y_pred_test)

        print("Test Accuracy: ", test_acc)
        print("Test F1: ", test_f1)
        acc_list.append(test_acc)
        f1_list.append(test_f1)
        score_list.append(acc)
        prec_list.append(test_prec)
        recall_list.append(test_recall)

        print(aggregate_model.params)
        time_min = (datetime.now() - start).seconds / 60
        print("Time(min) {}: ".format(time_min))
    
    f1_summary.append(f1_list)
    acc_summary.append(acc_list)
    print("Active party {} finished training.".format(active_party))
    print("Accuracy for party {}".format(active_party) + str(acc_list))
    print("F1 score for party {}".format(active_party) + str(f1_list))
    print("-------------------------------------------------")
    
    mean_acc = np.mean(acc_list)
    mean_f1 = np.mean(f1_list)
    mean_prec = np.mean(prec_list)
    mean_recall = np.mean(recall_list)

    out = "Party {}, k_percent {:.1f}, k1_percent {:.1f}, random_state {:.1f}: F1 mean={}, prec mean={}, recall mean={}".format(active_party, k_percent, k1_percent, random_state, mean_f1, mean_prec,  mean_recall)
    print(out)
    return mean_acc, mean_f1, mean_prec, mean_recall

def train_fedonce_dae(active_party = 0,  k_percent = 100,  k1_percent = 100, 
                  random_state = 10, reps = 0.0):

    num_parties = 2
    x_scaler_wrapper = []
    y_scaler_wrapper = []
    x_normalizer_wrapper = []
    cross_valid_data = load_data_cross_validation("phishing.train", num_parties=num_parties,
                                                file_type='libsvm', n_fold=5, use_cache=False,
                                                x_scaler_wrapper=x_scaler_wrapper, y_scaler_wrapper=y_scaler_wrapper,
                                                x_normalizer_wrapper=x_normalizer_wrapper,
                                                random_state = random_state)    
    xs_test, y_test = load_data_cross_validation("phishing.test", num_parties=num_parties,
                                                file_type='libsvm', n_fold=1, use_cache=False,
                                                x_scaler_wrapper=x_scaler_wrapper, y_scaler_wrapper=y_scaler_wrapper,
                                                x_normalizer_wrapper=x_normalizer_wrapper,
                                                random_state= random_state)[0]
    acc_summary = []
    f1_summary = []
    
    acc_list = []
    f1_list = []

    prec_list = []
    recall_list = []
    score_list = []

    for i, (xs_train, y_train, xs_val, y_val) in enumerate(cross_valid_data):
        start = datetime.now()
        print("Cross Validation Fold {}".format(i))
        print("Active Party is {}".format(active_party))
        model_name = "fedonce_dae_phishing_party_{}_fold_{}".format(num_parties, i)
        name = "{}_active_{}_k1_{}_k_{}_random_state_{}".format(model_name, active_party, k1_percent, k_percent, random_state)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModelDae(
            num_parties=num_parties,
            active_party_id=active_party,
            name=name,
            full_name=name,
            num_epochs=300,
            num_local_rounds=100,
            local_lr=1e-4,
            local_hidden_layers=[30, 30],
            local_batch_size=100,
            local_weight_decay=1e-5,
            local_output_size=48,
            num_agg_rounds=1,
            agg_lr=1e-4,
            agg_hidden_layers=[30],
            agg_batch_size=100,
            agg_weight_decay=1e-4,
            writer=writer,
            device='cuda:{}'.format(args.gpu),
            update_target_freq=1,
            task='binary_classification',
            n_classes=10,
            test_batch_size=1000,
            test_freq=1,
            cuda_parallel=False,
            n_channels=1,
            model_type='fc',
            optimizer='adam',
            num_workers=0
        )
        selected_features = []
        if k1_percent < 100:
            acc, _, _, _, selected_features  = aggregate_model.train(xs_train, y_train, xs_val, y_val, k_percent = k1_percent)
        else:                                        
            acc, f1, _, _, _ = aggregate_model.train(xs_train, y_train, xs_val, y_val)
       
        y_score_test = aggregate_model.predict_agg(xs_test, selection_features = selected_features)
        y_pred_test = np.where(y_score_test > 0.5, 1, 0)
        
        test_acc = accuracy_score(y_test, y_pred_test)
        test_f1 = f1_score(y_test, y_pred_test)

        test_prec = precision_score(y_test, y_pred_test)
        test_recall = recall_score(y_test, y_pred_test)

        print("Test Accuracy: ", test_acc)
        print("Test F1: ", test_f1)
        acc_list.append(test_acc)
        f1_list.append(test_f1)
        score_list.append(acc)
        prec_list.append(test_prec)
        recall_list.append(test_recall)

        print(aggregate_model.params)
        time_min = (datetime.now() - start).seconds / 60
        print("Time(min) {}: ".format(time_min))

    f1_summary.append(f1_list)
    acc_summary.append(acc_list)
    print("Active party {} finished training.".format(active_party))
    print("Accuracy for party {}".format(active_party) + str(acc_list))
    print("F1 score for party {}".format(active_party) + str(f1_list))
    print("-------------------------------------------------")

    mean_acc = np.mean(acc_list)
    mean_f1 = np.mean(f1_list)
    mean_prec = np.mean(prec_list)
    mean_recall = np.mean(recall_list)

    out = "Party {}, k_percent {:.1f}, k1_percent {:.1f}, random_state {:.1f}: F1 mean={}, prec mean={}, recall mean={}".format(active_party, k_percent, k1_percent, random_state, mean_f1, mean_prec,  mean_recall)
    print(out)
    return mean_acc, mean_f1, mean_prec, mean_recall


def run_vertical_fl_ft_selection_all_ration_select_guest(active_party, reps = 0.0):
    ratios = np.arange(0.1, 1.0, 0.1)
    Parallel(n_jobs=-1)(delayed(train_fedonce)(active_party = active_party, k_percent = 100, k1_percent = ratio*100,
                                                reps = reps) for ratio in ratios)
    

if __name__ == "__main__":
    # train_fedonce()
    # train_fedonce_dae()
    run_vertical_fl_ft_selection_all_ration_select_guest(active_party=0)