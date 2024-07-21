from utils.data_utils import load_data_cross_validation, load_data_train_test, get_random_noisy_row, load_creditcardfraud, load_uci
from model.fl_model import VerticalFLModel
# from model.fl_model_ssl import VerticalFLModel as VerticalFLModelSSL
# from model.fl_model_sup import VerticalFLModel as VerticalFLModelSup
# from model.fl_model_scarf import VerticalFLModel as VerticalFLModelScarf
from model.fl_model_two_side import VerticalFLModel as VerticalFLModelTwoSide
from model.fl_model_split import VerticalFLModel as VerticalFLModelSplit
from model.fl_model_dae import VerticalFLModel as VerticalFLModelDae
from model.single_party_model import SingleParty
from model.split_nn_model import SplitNNModel
from model.split_nn_model_dae import SplitNNModel as SplitNNModelDae
from model.models import FC

from torch.utils.tensorboard import SummaryWriter
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectFromModel, SelectPercentile, f_classif

import joblib
from joblib import Parallel, delayed
import torch

import os.path
import wget
import bz2
import shutil
import zipfile
import numpy as np

import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold



## Feature importance sort

def get_feature_importance_uci():
    # load data
    xs_train_val, y_train_val, xs_test, y_test = load_uci(test_rate = 0.2)

    x_train_val = np.concatenate(xs_train_val, axis=1)

    x_test = np.concatenate(xs_test, axis=1)

    # calculate XGBoost feature importance
    print("Starts training XGBoost on uci")
    if os.path.exists("cache/feature_importance_uci.txt"):
        importance = np.loadtxt("cache/feature_importance_uci.txt")
    else:
        xg_cls = xgb.XGBClassifier(objective='binary:logistic',
                                    learning_rate=0.1,
                                    max_depth=6,
                                    n_estimators=100,
                                    reg_alpha=10,
                                    verbosity=2)

        xg_cls.fit(x_train_val, y_train_val, eval_set=[(x_train_val, y_train_val), (x_test, y_test)], eval_metric='auc')
        y_pred = xg_cls.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        importance = xg_cls.feature_importances_
        print("Finished training. Overall accuracy {}".format(acc))
        print("F1 score: ", f1)
        # save feature importance
        np.savetxt("cache/feature_importance_uci.txt", importance)

        # load importance from file
        importance = np.loadtxt("cache/feature_importance_uci.txt")
    
    return importance

def feature_selection(x_train_val, y_train_val, x_test, y_test, k_percent, name):
    
    sfm = SelectPercentile(f_classif, percentile=k_percent)
    sfm.fit(x_train_val, y_train_val)

    # Transform the data to include only selected features
    x_train_val_selected = sfm.transform(x_train_val)
    x_test_selected = sfm.transform(x_test)

    print("x_train_val_selected.shape: ", x_train_val_selected.shape)
    print("x_test_selected.shape: ", x_test_selected.shape)

    return x_train_val_selected, x_test_selected

## FedOnce
def train_fedonce(remove_ratio = 0, active_party = 0, beta = 0.5, noise_ratio = 0, k_percent = 100,  k1_percent = 100, select_host = True,
                  remain_selection = False, random_state = 10):
    num_parties = 2
    xs_train_val, y_train_val, xs_test, y_test = load_uci(test_rate = 0.2, remove_ratio=remove_ratio, feature_order=None,
                                                        feature_ratio_beta = beta, noise_ratio = noise_ratio,
                                                        random_state = random_state
                                                        )

    if k_percent < 100 and True:
        xs_train_val[active_party], xs_test[active_party] = feature_selection(xs_train_val[active_party], y_train_val, xs_test[active_party], y_test, k_percent, f"fedonce_active_{active_party}")

    # xs_train_val[active_party], xs_test[active_party] = feature_selection(xs_train_val[active_party], y_train_val, xs_test[active_party], y_test, 60, f"fedonce_active_{active_party}")

    print("Active party {} starts training".format(active_party))
    score_list = []
    prec_list = []
    recall_list = []
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
        name = "{}_active_{}_k1_{}_k_{}_random_state_{}".format(model_name, active_party, k1_percent, k_percent, random_state)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            full_name=name,
            num_epochs=100,
            num_local_rounds=100,
            local_lr=3e-4,
            local_hidden_layers=[50, 30],
            local_batch_size=100,
            local_weight_decay=1e-5,
            local_output_size=48,
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
        selected_features = []
        if False == False and k1_percent < 100:
            acc, _, _, _, selected_features  = aggregate_model.train(xs_train, y_train, xs_val, y_val, k_percent = k1_percent,
                                                                    remain_selection = remain_selection)
        else:
            acc, _, _, _ , _ = aggregate_model.train(xs_train, y_train, xs_val, y_val)
        
        y_test_score = aggregate_model.predict_agg(xs_test, selection_features = selected_features)
        y_test_pred = np.where(y_test_score > 0.5, 1, 0)
        test_f1 = f1_score(y_test, y_test_pred)
        test_prec = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)

        print("Active party {} finished training.".format(active_party))
        score_list.append(acc)
        f1_summary.append(test_f1)
        prec_list.append(test_prec)
        recall_list.append(test_recall)
        print(aggregate_model.params)
    
    print("Accuracy for active party {}".format(active_party) + str(score_list))
    print("F1 for active party {} with: {}".format(active_party, str(f1_summary)))
    
    mean_acc = np.mean(score_list)
    std = np.std(score_list)
    out = "Party {}, remove_ratio {:.1f}, beta {:.1f}, noise_ratio {:.1f}, k_percent {:.1f}, k1_percent {:.1f}, random_state {:.1f}: Accuracy mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, k_percent, k1_percent, random_state, mean_acc, std)
    print(out)

    mean_f1 = np.mean(f1_summary)
    std_f1 = np.std(f1_summary)

    mean_prec = np.mean(prec_list)
    std_prec = np.std(prec_list)

    mean_recall = np.mean(recall_list)
    std_recall = np.std(recall_list)

    # out = "Party {}, remove_ratio {:.1f}, beta {:.1f}: noise_ratio {:.1f}, k_percent {:.1f}, k1_percent {:.1f}: F1 mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, k_percent, k1_percent, mean, std)
    out = "Party {}, remove_ratio {:.1f}, beta {:.1f}: noise_ratio {:.1f}, k_percent {:.1f}, k1_percent {:.1f}, random_state {:.1f}: F1 mean={}, std={}, prec mean={}, std={}, recall mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, k_percent, k1_percent, random_state, mean_f1, std_f1, mean_prec, std_prec, mean_recall, std_recall)
    print(out)
    return mean_acc, mean_f1, mean_prec, mean_recall

## FedOnce L1
def train_fedonce_l1(remove_ratio = 0, active_party = 0, beta = 0.5, noise_ratio = 0, k_percent = 100,  k1_percent = 100, select_host = True,
                  remain_selection = False, random_state = 10):
    num_parties = 2
    xs_train_val, y_train_val, xs_test, y_test = load_uci(test_rate = 0.2, remove_ratio=remove_ratio, feature_order=None,
                                                        feature_ratio_beta = beta, noise_ratio = noise_ratio,
                                                        random_state = random_state
                                                        )

    if k_percent < 100 and True:
        xs_train_val[active_party], xs_test[active_party] = feature_selection(xs_train_val[active_party], y_train_val, xs_test[active_party], y_test, k_percent, f"fedonce_active_{active_party}")

    # xs_train_val[active_party], xs_test[active_party] = feature_selection(xs_train_val[active_party], y_train_val, xs_test[active_party], y_test, 60, f"fedonce_active_{active_party}")

    print("Active party {} starts training".format(active_party))
    score_list = []
    prec_list = []
    recall_list = []
    f1_summary = []
    eps = 2 
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
            local_output_size=48,
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
            privacy="MA",
            batches_per_lot=1,
            delta=1e-5,
            inter_party_comp_method="MA",
            grad_norm_C=1.5,
            epsilon=2048,
        )
        selected_features = []
        if False == False and k1_percent < 100:
            acc, _, _, _, selected_features  = aggregate_model.train(xs_train, y_train, xs_val, y_val, k_percent = k1_percent,
                                                                    remain_selection = remain_selection)
        else:
            acc, _, _, _ , _ = aggregate_model.train(xs_train, y_train, xs_val, y_val)
        
        y_test_score = aggregate_model.predict_agg(xs_test, selection_features = selected_features)
        y_test_pred = np.where(y_test_score > 0.5, 1, 0)
        test_f1 = f1_score(y_test, y_test_pred)
        test_prec = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)

        print("Active party {} finished training.".format(active_party))
        score_list.append(acc)
        f1_summary.append(test_f1)
        prec_list.append(test_prec)
        recall_list.append(test_recall)
        print(aggregate_model.params)
    
    print("Accuracy for active party {}".format(active_party) + str(score_list))
    print("F1 for active party {} with: {}".format(active_party, str(f1_summary)))
    
    mean_acc = np.mean(score_list)
    std = np.std(score_list)
    out = "Party {}, remove_ratio {:.1f}, beta {:.1f}, noise_ratio {:.1f}, k_percent {:.1f}, k1_percent {:.1f}: Accuracy mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, k_percent, k1_percent, mean_acc, std)
    print(out)

    mean_f1 = np.mean(f1_summary)
    std_f1 = np.std(f1_summary)

    mean_prec = np.mean(prec_list)
    std_prec = np.std(prec_list)

    mean_recall = np.mean(recall_list)
    std_recall = np.std(recall_list)

    # out = "Party {}, remove_ratio {:.1f}, beta {:.1f}: noise_ratio {:.1f}, k_percent {:.1f}, k1_percent {:.1f}: F1 mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, k_percent, k1_percent, mean, std)
    out = "Party {}, remove_ratio {:.1f}, beta {:.1f}: noise_ratio {:.1f}, k_percent {:.1f}, k1_percent {:.1f}: F1 mean={}, std={}, prec mean={}, std={}, recall mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, k_percent, k1_percent, mean_f1, std_f1, mean_prec, std_prec, mean_recall, std_recall)
    print(out)
    return mean_acc, mean_f1, mean_prec, mean_recall


## FedOnce L1
def train_fedonce_l1(remove_ratio = 0, active_party = 0, beta = 0.5, noise_ratio = 0, k_percent = 100,  k1_percent = 100, select_host = True,
                  remain_selection = False, random_state = 10):
    num_parties = 2
    xs_train_val, y_train_val, xs_test, y_test = load_uci(test_rate = 0.2, remove_ratio=remove_ratio, feature_order=None,
                                                        feature_ratio_beta = beta, noise_ratio = noise_ratio,
                                                        random_state = random_state
                                                        )

    if k_percent < 100 and True:
        xs_train_val[active_party], xs_test[active_party] = feature_selection(xs_train_val[active_party], y_train_val, xs_test[active_party], y_test, k_percent, f"fedonce_active_{active_party}")

    # xs_train_val[active_party], xs_test[active_party] = feature_selection(xs_train_val[active_party], y_train_val, xs_test[active_party], y_test, 60, f"fedonce_active_{active_party}")

    print("Active party {} starts training".format(active_party))
    score_list = []
    prec_list = []
    recall_list = []
    f1_summary = []
    eps = 2 
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
            local_output_size=48,
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
            privacy="MA",
            batches_per_lot=1,
            delta=1e-5,
            inter_party_comp_method="MA",
            grad_norm_C=1.5
        )
        selected_features = []
        if False == False and k1_percent < 100:
            acc, _, _, _, selected_features  = aggregate_model.train(xs_train, y_train, xs_val, y_val, k_percent = k1_percent,
                                                                    remain_selection = remain_selection)
        else:
            acc, _, _, _ , _ = aggregate_model.train(xs_train, y_train, xs_val, y_val)
        
        y_test_score = aggregate_model.predict_agg(xs_test, selection_features = selected_features)
        y_test_pred = np.where(y_test_score > 0.5, 1, 0)
        test_f1 = f1_score(y_test, y_test_pred)
        test_prec = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)

        print("Active party {} finished training.".format(active_party))
        score_list.append(acc)
        f1_summary.append(test_f1)
        prec_list.append(test_prec)
        recall_list.append(test_recall)
        print(aggregate_model.params)
    
    print("Accuracy for active party {}".format(active_party) + str(score_list))
    print("F1 for active party {} with: {}".format(active_party, str(f1_summary)))
    
    mean_acc = np.mean(score_list)
    std = np.std(score_list)
    out = "Party {}, remove_ratio {:.1f}, beta {:.1f}, noise_ratio {:.1f}, k_percent {:.1f}, k1_percent {:.1f}: Accuracy mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, k_percent, k1_percent, mean_acc, std)
    print(out)

    mean_f1 = np.mean(f1_summary)
    std_f1 = np.std(f1_summary)

    mean_prec = np.mean(prec_list)
    std_prec = np.std(prec_list)

    mean_recall = np.mean(recall_list)
    std_recall = np.std(recall_list)

    # out = "Party {}, remove_ratio {:.1f}, beta {:.1f}: noise_ratio {:.1f}, k_percent {:.1f}, k1_percent {:.1f}: F1 mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, k_percent, k1_percent, mean, std)
    out = "Party {}, remove_ratio {:.1f}, beta {:.1f}: noise_ratio {:.1f}, k_percent {:.1f}, k1_percent {:.1f}: F1 mean={}, std={}, prec mean={}, std={}, recall mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, k_percent, k1_percent, mean_f1, std_f1, mean_prec, std_prec, mean_recall, std_recall)
    print(out)
    return mean_acc, mean_f1, mean_prec, mean_recall


## FedOnce multi round
def train_fedonce_multi_round(remove_ratio = 0, active_party = 0, beta = 0.5, noise_ratio = 0, k_percent = 100,  k1_percent = 100, select_host = True,
                  remain_selection = False, random_state = 10):
    num_parties = 2
    xs_train_val, y_train_val, xs_test, y_test = load_uci(test_rate = 0.2, remove_ratio=remove_ratio, feature_order=None,
                                                        feature_ratio_beta = beta, noise_ratio = noise_ratio,
                                                        random_state = random_state
                                                        )

    if k_percent < 100 and True:
        xs_train_val[active_party], xs_test[active_party] = feature_selection(xs_train_val[active_party], y_train_val, xs_test[active_party], y_test, k_percent, f"fedonce_active_{active_party}")

    # xs_train_val[active_party], xs_test[active_party] = feature_selection(xs_train_val[active_party], y_train_val, xs_test[active_party], y_test, 60, f"fedonce_active_{active_party}")

    print("Active party {} starts training".format(active_party))
    score_list = []
    prec_list = []
    recall_list = []
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
        model_name = "vertical_fl_uci_party_{}_fold_{}".format(num_parties, i)
        name = "{}_active_{}".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            full_name=name,
            num_epochs=100,
            num_local_rounds=100,
            local_lr=3e-4,
            local_hidden_layers=[50, 30],
            local_batch_size=100,
            local_weight_decay=1e-5,
            local_output_size=48,
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
        selected_features = []
        if False == False and k1_percent < 100:
            acc, _, _, _, selected_features  = aggregate_model.train(xs_train, y_train, xs_val, y_val, k_percent = k1_percent,
                                                                    remain_selection = remain_selection)
        else:
            acc, _, _, _ , _ = aggregate_model.train(xs_train, y_train, xs_val, y_val)
        
        y_test_score = aggregate_model.predict_agg(xs_test, selection_features = selected_features)
        y_test_pred = np.where(y_test_score > 0.5, 1, 0)
        test_f1 = f1_score(y_test, y_test_pred)
        test_prec = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)

        print("Active party {} finished training.".format(active_party))
        score_list.append(acc)
        f1_summary.append(test_f1)
        prec_list.append(test_prec)
        recall_list.append(test_recall)
        print(aggregate_model.params)
    
    print("Accuracy for active party {}".format(active_party) + str(score_list))
    print("F1 for active party {} with: {}".format(active_party, str(f1_summary)))
    
    mean_acc = np.mean(score_list)
    std = np.std(score_list)
    out = "Party {}, remove_ratio {:.1f}, beta {:.1f}, noise_ratio {:.1f}, k_percent {:.1f}, k1_percent {:.1f}: Accuracy mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, k_percent, k1_percent, mean_acc, std)
    print(out)

    mean_f1 = np.mean(f1_summary)
    std_f1 = np.std(f1_summary)

    mean_prec = np.mean(prec_list)
    std_prec = np.std(prec_list)

    mean_recall = np.mean(recall_list)
    std_recall = np.std(recall_list)

    # out = "Party {}, remove_ratio {:.1f}, beta {:.1f}: noise_ratio {:.1f}, k_percent {:.1f}, k1_percent {:.1f}: F1 mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, k_percent, k1_percent, mean, std)
    out = "Party {}, remove_ratio {:.1f}, beta {:.1f}: noise_ratio {:.1f}, k_percent {:.1f}, k1_percent {:.1f}: F1 mean={}, std={}, prec mean={}, std={}, recall mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, k_percent, k1_percent, mean_f1, std_f1, mean_prec, std_prec, mean_recall, std_recall)
    print(out)

    # post fedOnce
    num_parties = 2
    xs_train_val, y_train_val, xs_test, y_test = load_uci(test_rate = 0.2, remove_ratio=remove_ratio, feature_order=None,
                                                        feature_ratio_beta = beta, noise_ratio = noise_ratio,
                                                        random_state = random_state
                                                        )
    
    ## only get random 10% xs_train_val and y_train_val
    random_indices = np.random.choice(xs_train_val[0].shape[0], int(xs_train_val[0].shape[0] * 0.2), replace=False)
    xs_train_val = [data[random_indices] for data in xs_train_val]
    y_train_val = y_train_val[random_indices]
    

    score_list = []
    prec_list = []
    recall_list = []
    f1_summary = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)

    print("Starts training")
    
    for i, (train_idx, val_idx) in enumerate(kfold):
        print("Cross Validation Fold {}".format(i))
        xs_train = [data[train_idx] for data in xs_train_val]
        y_train = y_train_val[train_idx]
        xs_val = [data[val_idx] for data in xs_train_val]
        y_val = y_train_val[val_idx]
        print("Train shape: {}".format(xs_train[0].shape))
        print("Val shape: {}".format(xs_val[0].shape))
        model_name = "post_vertical_fl_uci_party_{}_fold_{}".format(num_parties, i)
        name = "{}/".format(model_name)
        writer = SummaryWriter("runs/{}".format(name))
        
        cache_local_name = "vertical_fl_uci_party_{}_fold_{}".format(num_parties, i)
        cache_agg_name = "{}_active_{}".format(cache_local_name, active_party)
        

        aggregate_model = SplitNNModel(
            num_parties=num_parties,
            name=model_name,
            num_epochs=10,
            local_hidden_layers=[50, 30],
            local_output_size=48,
            lr=3e-4,
            agg_hidden_layers=[10],
            batch_size=100,
            weight_decay=1e-5,
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
            delta=1.0/xs_train[0].shape[0],
            num_workers=2,
            cache_local_name=cache_local_name,
            cache_agg_name=cache_agg_name,
            active_party=active_party
        )

        acc, _, _, _  = aggregate_model.train(xs_train, y_train, xs_val, y_val, use_cache=True)
        
        _, y_test_score = aggregate_model.predict(xs_test)
        y_test_pred = np.where(y_test_score > 0.5, 1, 0)
        test_f1 = f1_score(y_test, y_test_pred)
        test_prec = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)

        print("Active party {} finished training.".format(active_party))
        score_list.append(acc)
        f1_summary.append(test_f1)
        prec_list.append(test_prec)
        recall_list.append(test_recall)
        print(aggregate_model.params)
    
    print("Accuracy for active party {}".format(active_party) + str(score_list))
    print("F1 for active party {} with: {}".format(active_party, str(f1_summary)))
    
    mean_acc = np.mean(score_list)
    std = np.std(score_list)
    out = "Party {}, remove_ratio {:.1f}, beta {:.1f}, noise_ratio {:.1f}, k_percent {:.1f}, k1_percent {:.1f}: Accuracy mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, k_percent, k1_percent, mean_acc, std)
    print(out)

    mean_f1 = np.mean(f1_summary)
    std_f1 = np.std(f1_summary)

    mean_prec = np.mean(prec_list)
    std_prec = np.std(prec_list)

    mean_recall = np.mean(recall_list)
    std_recall = np.std(recall_list)

    # out = "Party {}, remove_ratio {:.1f}, beta {:.1f}: noise_ratio {:.1f}, k_percent {:.1f}, k1_percent {:.1f}: F1 mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, k_percent, k1_percent, mean, std)
    out = "Post Fed Party {}, remove_ratio {:.1f}, beta {:.1f}: noise_ratio {:.1f}, k_percent {:.1f}, k1_percent {:.1f}: F1 mean={}, std={}, prec mean={}, std={}, recall mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, k_percent, k1_percent, mean_f1, std_f1, mean_prec, std_prec, mean_recall, std_recall)
    print(out)

    return mean_acc, mean_f1, mean_prec, mean_recall


def train_fedonce_dae_multi_round(remove_ratio = 0, active_party = 0, beta = 0.5, noise_ratio = 0, k_percent = 100,  k1_percent = 100, select_host = True,
                                  remain_selection = False, random_state = 10):
    
    num_parties = 2
    xs_train_val, y_train_val, xs_test, y_test = load_uci(test_rate = 0.2, remove_ratio=remove_ratio, feature_order=None,
                                                        feature_ratio_beta = beta, noise_ratio = noise_ratio,
                                                        random_state = random_state
                                                        )

    if k_percent < 100 and True:
        xs_train_val[active_party], xs_test[active_party] = feature_selection(xs_train_val[active_party], y_train_val, xs_test[active_party], y_test, k_percent, f"fedonce_active_{active_party}")

    # xs_train_val[active_party], xs_test[active_party] = feature_selection(xs_train_val[active_party], y_train_val, xs_test[active_party], y_test, 60, f"fedonce_active_{active_party}")

    print("Active party {} starts training".format(active_party))
    score_list = []
    prec_list = []
    recall_list = []
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
        model_name = "vertical_fl_dae_uci_party_{}_fold_{}".format(num_parties, i)
        name = "{}_active_{}".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModelDae(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            full_name=name,
            num_epochs=100,
            num_local_rounds=100,
            local_lr=3e-4,
            local_hidden_layers=[50, 30],
            local_batch_size=100,
            local_weight_decay=1e-5,
            local_output_size=48,
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
        selected_features = []
        if False == False and k1_percent < 100:
            acc, _, _, _, selected_features  = aggregate_model.train(xs_train, y_train, xs_val, y_val, k_percent = k1_percent,
                                                                    remain_selection = remain_selection)
        else:
            acc, _, _, _ , _ = aggregate_model.train(xs_train, y_train, xs_val, y_val)
        
        y_test_score = aggregate_model.predict_agg(xs_test, selection_features = selected_features)
        y_test_pred = np.where(y_test_score > 0.5, 1, 0)
        test_f1 = f1_score(y_test, y_test_pred)
        test_prec = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)

        print("Active party {} finished training.".format(active_party))
        score_list.append(acc)
        f1_summary.append(test_f1)
        prec_list.append(test_prec)
        recall_list.append(test_recall)
        print(aggregate_model.params)
    
    print("Accuracy for active party {}".format(active_party) + str(score_list))
    print("F1 for active party {} with: {}".format(active_party, str(f1_summary)))
    
    mean_acc = np.mean(score_list)
    std = np.std(score_list)
    out = "Party {}, remove_ratio {:.1f}, beta {:.1f}, noise_ratio {:.1f}, k_percent {:.1f}, k1_percent {:.1f}: Accuracy mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, k_percent, k1_percent, mean_acc, std)
    print(out)

    mean_f1 = np.mean(f1_summary)
    std_f1 = np.std(f1_summary)

    mean_prec = np.mean(prec_list)
    std_prec = np.std(prec_list)

    mean_recall = np.mean(recall_list)
    std_recall = np.std(recall_list)

    # out = "Party {}, remove_ratio {:.1f}, beta {:.1f}: noise_ratio {:.1f}, k_percent {:.1f}, k1_percent {:.1f}: F1 mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, k_percent, k1_percent, mean, std)
    out = "Party {}, remove_ratio {:.1f}, beta {:.1f}: noise_ratio {:.1f}, k_percent {:.1f}, k1_percent {:.1f}: F1 mean={}, std={}, prec mean={}, std={}, recall mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, k_percent, k1_percent, mean_f1, std_f1, mean_prec, std_prec, mean_recall, std_recall)
    print(out)

    # post fedOnce
    num_parties = 2
    xs_train_val, y_train_val, xs_test, y_test = load_uci(test_rate = 0.2, remove_ratio=remove_ratio, feature_order=None,
                                                        feature_ratio_beta = beta, noise_ratio = noise_ratio,
                                                        random_state = random_state
                                                        )
    
    ## only get random 10% xs_train_val and y_train_val
    random_indices = np.random.choice(xs_train_val[0].shape[0], int(xs_train_val[0].shape[0] * 0.2), replace=False)
    xs_train_val = [data[random_indices] for data in xs_train_val]
    y_train_val = y_train_val[random_indices]
    

    score_list = []
    prec_list = []
    recall_list = []
    f1_summary = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)

    print("Starts training")
    
    for i, (train_idx, val_idx) in enumerate(kfold):
        print("Cross Validation Fold {}".format(i))
        xs_train = [data[train_idx] for data in xs_train_val]
        y_train = y_train_val[train_idx]
        xs_val = [data[val_idx] for data in xs_train_val]
        y_val = y_train_val[val_idx]
        print("Train shape: {}".format(xs_train[0].shape))
        print("Val shape: {}".format(xs_val[0].shape))
        model_name = "post_vertical_fl_dae_uci_party_{}_fold_{}".format(num_parties, i)
        name = "{}/".format(model_name)
        writer = SummaryWriter("runs/{}".format(name))
        
        cache_local_name = "vertical_fl_dae_uci_party_{}_fold_{}".format(num_parties, i)
        cache_agg_name = "{}_active_{}".format(cache_local_name, active_party)
        

        aggregate_model = SplitNNModelDae(
            num_parties=num_parties,
            name=model_name,
            num_epochs=10,
            local_hidden_layers=[50, 30],
            local_output_size=48,
            lr=3e-4,
            agg_hidden_layers=[10],
            batch_size=100,
            weight_decay=1e-5,
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
            delta=1.0/xs_train[0].shape[0],
            num_workers=2,
            cache_local_name=cache_local_name,
            cache_agg_name=cache_agg_name,
            active_party=active_party
        )

        acc, _, _, _  = aggregate_model.train(xs_train, y_train, xs_val, y_val, use_cache=True)
        
        _, y_test_score = aggregate_model.predict(xs_test)
        y_test_pred = np.where(y_test_score > 0.5, 1, 0)
        test_f1 = f1_score(y_test, y_test_pred)
        test_prec = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)

        print("Active party {} finished training.".format(active_party))
        score_list.append(acc)
        f1_summary.append(test_f1)
        prec_list.append(test_prec)
        recall_list.append(test_recall)
        print(aggregate_model.params)
    
    print("Accuracy for active party {}".format(active_party) + str(score_list))
    print("F1 for active party {} with: {}".format(active_party, str(f1_summary)))
    
    mean_acc = np.mean(score_list)
    std = np.std(score_list)
    out = "Post Fed Party {}, remove_ratio {:.1f}, beta {:.1f}, noise_ratio {:.1f}, k_percent {:.1f}, k1_percent {:.1f}: Accuracy mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, k_percent, k1_percent, mean_acc, std)
    print(out)

    mean_f1 = np.mean(f1_summary)
    std_f1 = np.std(f1_summary)

    mean_prec = np.mean(prec_list)
    std_prec = np.std(prec_list)

    mean_recall = np.mean(recall_list)
    std_recall = np.std(recall_list)

    # out = "Party {}, remove_ratio {:.1f}, beta {:.1f}: noise_ratio {:.1f}, k_percent {:.1f}, k1_percent {:.1f}: F1 mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, k_percent, k1_percent, mean, std)
    out = "Post Fed Party {}, remove_ratio {:.1f}, beta {:.1f}: noise_ratio {:.1f}, k_percent {:.1f}, k1_percent {:.1f}: F1 mean={}, std={}, prec mean={}, std={}, recall mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, k_percent, k1_percent, mean_f1, std_f1, mean_prec, std_prec, mean_recall, std_recall)
    print(out)

    return mean_acc, mean_f1, mean_prec, mean_recall


## FedOnce Split
def train_fedonce_split(remove_ratio = 0, active_party = 1, beta = 0.5, noise_ratio = 0):
    num_parties = 2
    xs_train_val, y_train_val, xs_test, y_test = load_uci(test_rate = 0.2)

    print("Active party {} starts training".format(active_party))
    score_list = []
    f1_summary = []
    prec_list = []
    recall_list = []

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
        aggregate_model = VerticalFLModelSplit(
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
        acc, _, _, _  = aggregate_model.train(xs_train, y_train, xs_val, y_val)
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
    out = "Party {}, remove_ratio {:.1f}, beta {:.1f}, noise_ratio {:.1f}: Accuracy mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, mean, std)
    print(out)

    mean = np.mean(f1_summary)
    std = np.std(f1_summary)
    out = "Party {}, remove_ratio {:.1f}, beta {:.1f}: noise_ratio {:.1f}: F1 mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, mean, std)
    print(out)
    return mean, std

## FedOnce Scarf
# def train_fedonce_scarf(remove_ratio = 0, active_party = 1, beta = 0.5, noise_ratio = 0, k_percent = 100, select_host = True):
    # num_parties = 2
    # xs_train_val, y_train_val, xs_test, y_test = load_uci(test_rate = 0.2, remove_ratio=remove_ratio, feature_order=None,
    #                                                     feature_ratio_beta = beta, noise_ratio = noise_ratio)

    # if k_percent < 100 and select_host:
    #     xs_train_val[active_party], xs_test[active_party] = feature_selection(xs_train_val[active_party], y_train_val, xs_test[active_party], y_test, k_percent, f"fedonce_active_{active_party}")


    # print("Active party {} starts training".format(active_party))
    # score_list = []
    # f1_summary = []
    # kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)

    # for i, (train_idx, val_idx) in enumerate(kfold):
    #     print("Cross Validation Fold {}".format(i))
    #     xs_train = [data[train_idx] for data in xs_train_val]
    #     y_train = y_train_val[train_idx]
    #     xs_val = [data[val_idx] for data in xs_train_val]
    #     y_val = y_train_val[val_idx]
    #     print("Train shape: {}".format(xs_train[0].shape))
    #     print("Val shape: {}".format(xs_val[0].shape))
    #     model_name = "vertical_fl_scarf_uci_party_{}_fold_{}".format(num_parties, i)
    #     name = "{}_active_{}".format(model_name, active_party)
    #     writer = SummaryWriter("runs/{}".format(name))
    #     aggregate_model = VerticalFLModelScarf(
    #         num_parties=num_parties,
    #         active_party_id=active_party,
    #         name=model_name,
    #         num_epochs=200,
    #         num_local_rounds=200,
    #         local_lr=3e-4,
    #         local_hidden_layers=[50, 30],
    #         local_batch_size=100,
    #         local_weight_decay=1e-5,
    #         local_output_size=10,
    #         num_agg_rounds=1,
    #         agg_lr=1e-4,
    #         agg_hidden_layers=[10],
    #         agg_batch_size=100,
    #         agg_weight_decay=1e-4,
    #         writer=writer,
    #         device='cuda:0',
    #         update_target_freq=1,
    #         task='binary_classification',
    #         n_classes=10,
    #         test_batch_size=1000,
    #         test_freq=1,
    #         cuda_parallel=False,
    #         n_channels=1,
    #         model_type='fc',
    #         optimizer='adam',
    #         privacy=None,
    #         batches_per_lot=5,
    #         epsilon=1,
    #         delta=1.0/xs_train[0].shape[0]
    #     )
    #     selected_features = []
    #     if select_host == False and k_percent < 100:
    #         acc, _, _, _, selected_features  = aggregate_model.train(xs_train, y_train, xs_val, y_val, k_percent = k_percent)
    #     else:
    #         acc, _, _, _, _  = aggregate_model.train(xs_train, y_train, xs_val, y_val)
    #     y_test_score = aggregate_model.predict_agg(xs_test, selection_features = selected_features)
    #     y_test_pred = np.where(y_test_score > 0.5, 1, 0)
    #     test_f1 = f1_score(y_test, y_test_pred)

    #     print("Active party {} finished training.".format(active_party))
    #     score_list.append(acc)
    #     f1_summary.append(test_f1)
    #     print(aggregate_model.params)
    
    # print("Accuracy for active party {}".format(active_party) + str(score_list))
    # print("F1 for active party {} with: {}".format(active_party, str(f1_summary)))
    
    # mean = np.mean(score_list)
    # std = np.std(score_list)
    # out = "Party {}, remove_ratio {:.1f}, beta {:.1f}, noise_ratio {:.1f}, k_percent {:.1f}: Accuracy mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, k_percent, mean, std)
    # print(out)

    # mean = np.mean(f1_summary)
    # std = np.std(f1_summary)
    # out = "Party {}, remove_ratio {:.1f}, beta {:.1f}: noise_ratio {:.1f}, k_percent {:.1f}: F1 mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, k_percent, mean, std)
    # print(out)
    # return mean, std

##
def train_fedonce_dae(remove_ratio = 0, active_party = 0, beta = 0.5, noise_ratio = 0, k_percent = 100, select_host = True, random_state = 10,
                      k1_percent = 100):
    num_parties = 2
    xs_train_val, y_train_val, xs_test, y_test = load_uci(test_rate = 0.2, remove_ratio=remove_ratio, feature_order=None,
                                                        feature_ratio_beta = beta, noise_ratio = noise_ratio, random_state = random_state)

    if k_percent < 100 and select_host:
        xs_train_val[active_party], xs_test[active_party] = feature_selection(xs_train_val[active_party], y_train_val, xs_test[active_party], y_test, k_percent, f"fedonce_active_{active_party}")


    print("Active party {} starts training".format(active_party))
    score_list = []
    f1_summary = []
    eval_f1_summary = []
    prec_list = []
    recall_list = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)

    for i, (train_idx, val_idx) in enumerate(kfold):
        print("Cross Validation Fold {}".format(i))
        xs_train = [data[train_idx] for data in xs_train_val]
        y_train = y_train_val[train_idx]
        xs_val = [data[val_idx] for data in xs_train_val]
        y_val = y_train_val[val_idx]
        print("Train shape: {}".format(xs_train[0].shape))
        print("Val shape: {}".format(xs_val[0].shape))
        model_name = "vertical_fl_dae_uci_party_{}_fold_{}".format(num_parties, i)
        name = "{}_active_{}_k1_{}_k_{}_random_state_{}".format(model_name, active_party, k1_percent, k_percent, random_state)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModelDae(
            num_parties=num_parties,
            active_party_id=active_party,
            name=name,
            full_name=name,
            num_epochs=100,
            num_local_rounds=100,
            local_lr=3e-4,
            local_hidden_layers=[50, 30],
            local_batch_size=100,
            local_weight_decay=1e-5,
            local_output_size=48,
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
        selected_features = []
        if False == False and k1_percent < 100:
            acc, eval_f1, _, _, selected_features  = aggregate_model.train(xs_train, y_train, xs_val, y_val, k_percent = k1_percent)
        else:
            acc, eval_f1, _, _, _  = aggregate_model.train(xs_train, y_train, xs_val, y_val)
        y_test_score = aggregate_model.predict_agg(xs_test, selection_features = selected_features)
        y_test_pred = np.where(y_test_score > 0.5, 1, 0)
        test_f1 = f1_score(y_test, y_test_pred)
        test_prec = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)

        print("Active party {} finished training.".format(active_party))
        score_list.append(acc)
        f1_summary.append(test_f1)
        eval_f1_summary.append(eval_f1)
        prec_list.append(test_prec)
        recall_list.append(test_recall)
        print(aggregate_model.params)
    
    print("Accuracy for active party {}".format(active_party) + str(score_list))
    print("F1 for active party {} with: {}".format(active_party, str(f1_summary)))
    
    mean_acc = np.mean(score_list)
    std = np.std(score_list)
    out = "Party {}, remove_ratio {:.1f}, beta {:.1f}, noise_ratio {:.1f}, k_percent {:.1f}: Accuracy mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, k_percent, mean_acc, std)
    print(out)

    mean_eval_f1 = np.mean(eval_f1_summary)

    mean_f1 = np.mean(f1_summary)
    std_f1 = np.std(f1_summary)

    mean_prec = np.mean(prec_list)
    std_prec = np.std(prec_list)

    mean_recall = np.mean(recall_list)
    std_recall = np.std(recall_list)

    # out = "Party {}, remove_ratio {:.1f}, beta {:.1f}: noise_ratio {:.1f}, k_percent {:.1f}: F1 mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, k_percent, mean, std)
    out = "Party {}, remove_ratio {:.1f}, beta {:.1f}: noise_ratio {:.1f}, k_percent {:.1f}, k1_percent {:.1f}, random_state {:.1f}: F1 mean={}, std={}, eval F1 mean={}, prec mean={}, std={}, recall mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, k_percent, k1_percent, random_state, mean_f1, std_f1, mean_eval_f1, mean_prec, std_prec, mean_recall, std_recall)

    print(out)
    return mean_acc, mean_f1, mean_prec, mean_recall, random_state, k_percent, k1_percent


def train_fedonce_dae_l1(remove_ratio = 0, active_party = 0, beta = 0.5, noise_ratio = 0, k_percent = 100, select_host = True, random_state = 10,
                      k1_percent = 100):
    num_parties = 2
    xs_train_val, y_train_val, xs_test, y_test = load_uci(test_rate = 0.2, remove_ratio=remove_ratio, feature_order=None,
                                                        feature_ratio_beta = beta, noise_ratio = noise_ratio, random_state = random_state)

    if k_percent < 100 and select_host:
        xs_train_val[active_party], xs_test[active_party] = feature_selection(xs_train_val[active_party], y_train_val, xs_test[active_party], y_test, k_percent, f"fedonce_active_{active_party}")


    print("Active party {} starts training".format(active_party))
    score_list = []
    f1_summary = []
    eval_f1_summary = []
    prec_list = []
    recall_list = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)

    for i, (train_idx, val_idx) in enumerate(kfold):
        print("Cross Validation Fold {}".format(i))
        xs_train = [data[train_idx] for data in xs_train_val]
        y_train = y_train_val[train_idx]
        xs_val = [data[val_idx] for data in xs_train_val]
        y_val = y_train_val[val_idx]
        print("Train shape: {}".format(xs_train[0].shape))
        print("Val shape: {}".format(xs_val[0].shape))
        model_name = "vertical_fl_dae_uci_party_{}_fold_{}".format(num_parties, i)
        name = "{}_active_{}_k1_{}_k_{}_random_state_{}".format(model_name, active_party, k1_percent, k_percent, random_state)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModelDae(
            num_parties=num_parties,
            active_party_id=active_party,
            name=name,
            num_epochs=100,
            num_local_rounds=100,
            local_lr=3e-4,
            local_hidden_layers=[50, 30],
            local_batch_size=100,
            local_weight_decay=1e-5,
            local_output_size=48,
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
            privacy="MA",
            batches_per_lot=1,
            epsilon=8,
            delta=1e-5,
            inter_party_comp_method="MA",
            grad_norm_C=1.5
        )
        selected_features = []
        if False == False and k1_percent < 100:
            acc, eval_f1, _, _, selected_features  = aggregate_model.train(xs_train, y_train, xs_val, y_val, k_percent = k1_percent)
        else:
            acc, eval_f1, _, _, _  = aggregate_model.train(xs_train, y_train, xs_val, y_val)
        y_test_score = aggregate_model.predict_agg(xs_test, selection_features = selected_features)
        y_test_pred = np.where(y_test_score > 0.5, 1, 0)
        test_f1 = f1_score(y_test, y_test_pred)
        test_prec = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)

        print("Active party {} finished training.".format(active_party))
        score_list.append(acc)
        f1_summary.append(test_f1)
        eval_f1_summary.append(eval_f1)
        prec_list.append(test_prec)
        recall_list.append(test_recall)
        print(aggregate_model.params)
    
    print("Accuracy for active party {}".format(active_party) + str(score_list))
    print("F1 for active party {} with: {}".format(active_party, str(f1_summary)))
    
    mean_acc = np.mean(score_list)
    std = np.std(score_list)
    out = "Party {}, remove_ratio {:.1f}, beta {:.1f}, noise_ratio {:.1f}, k_percent {:.1f}: Accuracy mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, k_percent, mean_acc, std)
    print(out)

    mean_eval_f1 = np.mean(eval_f1_summary)

    mean_f1 = np.mean(f1_summary)
    std_f1 = np.std(f1_summary)

    mean_prec = np.mean(prec_list)
    std_prec = np.std(prec_list)

    mean_recall = np.mean(recall_list)
    std_recall = np.std(recall_list)

    # out = "Party {}, remove_ratio {:.1f}, beta {:.1f}: noise_ratio {:.1f}, k_percent {:.1f}: F1 mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, k_percent, mean, std)
    out = "Party {}, remove_ratio {:.1f}, beta {:.1f}: noise_ratio {:.1f}, k_percent {:.1f}, k1_percent {:.1f}: F1 mean={}, std={}, eval F1 mean={}, prec mean={}, std={}, recall mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, k_percent, k1_percent, mean_f1, std_f1, mean_eval_f1, mean_prec, std_prec, mean_recall, std_recall)

    print(out)
    return mean_acc, mean_f1, mean_prec, mean_recall, random_state, k_percent, k1_percent

def train_fedonce_dae_l1(remove_ratio = 0, active_party = 0, beta = 0.5, noise_ratio = 0, k_percent = 100, select_host = True, random_state = 10,
                      k1_percent = 100):
    num_parties = 2
    xs_train_val, y_train_val, xs_test, y_test = load_uci(test_rate = 0.2, remove_ratio=remove_ratio, feature_order=None,
                                                        feature_ratio_beta = beta, noise_ratio = noise_ratio, random_state = random_state)

    if k_percent < 100 and select_host:
        xs_train_val[active_party], xs_test[active_party] = feature_selection(xs_train_val[active_party], y_train_val, xs_test[active_party], y_test, k_percent, f"fedonce_active_{active_party}")


    print("Active party {} starts training".format(active_party))
    score_list = []
    f1_summary = []
    eval_f1_summary = []
    prec_list = []
    recall_list = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)

    for i, (train_idx, val_idx) in enumerate(kfold):
        print("Cross Validation Fold {}".format(i))
        xs_train = [data[train_idx] for data in xs_train_val]
        y_train = y_train_val[train_idx]
        xs_val = [data[val_idx] for data in xs_train_val]
        y_val = y_train_val[val_idx]
        print("Train shape: {}".format(xs_train[0].shape))
        print("Val shape: {}".format(xs_val[0].shape))
        model_name = "vertical_fl_dae_uci_party_{}_fold_{}".format(num_parties, i)
        name = "{}_active_{}_k1_{}_k_{}_random_state_{}".format(model_name, active_party, k1_percent, k_percent, random_state)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModelDae(
            num_parties=num_parties,
            active_party_id=active_party,
            name=name,
            num_epochs=100,
            num_local_rounds=100,
            local_lr=3e-4,
            local_hidden_layers=[50, 30],
            local_batch_size=100,
            local_weight_decay=1e-5,
            local_output_size=48,
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
            privacy="MA",
            batches_per_lot=1,
            delta=1e-5,
            inter_party_comp_method="MA",
            grad_norm_C=1.5
        )
        selected_features = []
        if False == False and k1_percent < 100:
            acc, eval_f1, _, _, selected_features  = aggregate_model.train(xs_train, y_train, xs_val, y_val, k_percent = k1_percent)
        else:
            acc, eval_f1, _, _, _  = aggregate_model.train(xs_train, y_train, xs_val, y_val)
        y_test_score = aggregate_model.predict_agg(xs_test, selection_features = selected_features)
        y_test_pred = np.where(y_test_score > 0.5, 1, 0)
        test_f1 = f1_score(y_test, y_test_pred)
        test_prec = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)

        print("Active party {} finished training.".format(active_party))
        score_list.append(acc)
        f1_summary.append(test_f1)
        eval_f1_summary.append(eval_f1)
        prec_list.append(test_prec)
        recall_list.append(test_recall)
        print(aggregate_model.params)
    
    print("Accuracy for active party {}".format(active_party) + str(score_list))
    print("F1 for active party {} with: {}".format(active_party, str(f1_summary)))
    
    mean_acc = np.mean(score_list)
    std = np.std(score_list)
    out = "Party {}, remove_ratio {:.1f}, beta {:.1f}, noise_ratio {:.1f}, k_percent {:.1f}: Accuracy mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, k_percent, mean_acc, std)
    print(out)

    mean_eval_f1 = np.mean(eval_f1_summary)

    mean_f1 = np.mean(f1_summary)
    std_f1 = np.std(f1_summary)

    mean_prec = np.mean(prec_list)
    std_prec = np.std(prec_list)

    mean_recall = np.mean(recall_list)
    std_recall = np.std(recall_list)

    # out = "Party {}, remove_ratio {:.1f}, beta {:.1f}: noise_ratio {:.1f}, k_percent {:.1f}: F1 mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, k_percent, mean, std)
    out = "Party {}, remove_ratio {:.1f}, beta {:.1f}: noise_ratio {:.1f}, k_percent {:.1f}, k1_percent {:.1f}: F1 mean={}, std={}, eval F1 mean={}, prec mean={}, std={}, recall mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, k_percent, k1_percent, mean_f1, std_f1, mean_eval_f1, mean_prec, std_prec, mean_recall, std_recall)

    print(out)
    return mean_acc, mean_f1, mean_prec, mean_recall, random_state, k_percent, k1_percent


## FedOnce Two Side
def train_fedonce_two_side(remove_ratio = 0, active_party = 1, beta = 0.5, noise_ratio = 0, k_percent = 100, select_host = True):
    num_parties = 2
    xs_train_val, y_train_val, xs_test, y_test = load_uci(test_rate = 0.2, remove_ratio=remove_ratio, feature_order=None,
                                                        feature_ratio_beta = beta, noise_ratio = noise_ratio)

    if k_percent < 100 and select_host:
        xs_train_val[active_party], xs_test[active_party] = feature_selection(xs_train_val[active_party], y_train_val, xs_test[active_party], y_test, k_percent, f"fedonce_active_{active_party}")


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
        model_name = "vertical_fl_scarf_uci_party_{}_fold_{}".format(num_parties, i)
        name = "{}_active_{}".format(model_name, active_party)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModelTwoSide(
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
        selected_features = []
        if select_host == False and k_percent < 100:
            acc, _, _, _, selected_features  = aggregate_model.train(xs_train, y_train, xs_val, y_val, k_percent = k_percent)
        else:
            acc, _, _, _, _  = aggregate_model.train(xs_train, y_train, xs_val, y_val)
        y_test_score = aggregate_model.predict_agg(xs_test, selection_features = selected_features)
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
    out = "Party {}, remove_ratio {:.1f}, beta {:.1f}, noise_ratio {:.1f}, k_percent {:.1f}: Accuracy mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, k_percent, mean, std)
    print(out)

    mean = np.mean(f1_summary)
    std = np.std(f1_summary)
    out = "Party {}, remove_ratio {:.1f}, beta {:.1f}: noise_ratio {:.1f}, k_percent {:.1f}: F1 mean={}, std={}".format(active_party, remove_ratio, beta, noise_ratio, k_percent, mean, std)
    print(out)
    return mean, std

# combine
def train_combine(remove_ratio = 0, active_party = -1, k_percent = 100, random_state = 10):
    xs_train_val, y_train_val, xs_test, y_test = load_uci(test_rate = 0.2, remove_ratio=remove_ratio,
                                                          random_state = random_state)

    if active_party == -1:
    
        print("Active party {} starts training".format(active_party))
        x_train_val = np.concatenate(xs_train_val, axis=1)
        # x_train_val = xs_train_val[active_party]
        print("x_train_val shape: {}".format(x_train_val.shape))
        print("ratio of positive samples: {}".format(np.sum(y_train_val) / len(y_train_val)))

        x_test = np.concatenate(xs_test, axis=1)
        # x_test = xs_test[active_party]
        print("x_test shape: {}".format(x_test.shape))
        print("ratio of positive samples: {}".format(np.sum(y_test) / len(y_test)))
    else:
        print("Active party {} starts training".format(active_party))
        x_train_val = xs_train_val[active_party]
        print("x_train_val shape: {}".format(x_train_val.shape))
        print("ratio of positive samples: {}".format(np.sum(y_train_val) / len(y_train_val)))

        x_test = xs_test[active_party]
        print("x_test shape: {}".format(x_test.shape))
        print("ratio of positive samples: {}".format(np.sum(y_test) / len(y_test)))
    if k_percent < 100:
        x_train_val, x_test = feature_selection(x_train_val, y_train_val, x_test, y_test, k_percent, f"combine_active_{active_party}")

    kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)

    f1_summary = []
    acc_summary = []
    prec_list = []
    recall_list = []

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
        acc, _, _,_ = single_model.train(x_train, y_train, x_val, y_val)
        _, y_test_score = single_model.predict(x_test)
        y_test_pred = np.where(y_test_score > 0.5, 1, 0)
        test_f1 = f1_score(y_test, y_test_pred)
        acc_summary.append(acc)
        f1_summary.append(test_f1)
        prec = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)
        prec_list.append(prec)
        recall_list.append(recall)
        print(single_model.params)

    print("Accuracy summary: " + repr(acc_summary))
    print("F1 score summary: " + repr(f1_summary))
    for i, result in enumerate(acc_summary):
        mean = np.mean(result)
        std = np.std(result)
    print("remove_ratio {:.1f}, active_party {:.1f}, k_percent {:.1f}: accuracy mean={}, std={}".format(remove_ratio, active_party, k_percent, mean, std))
    for i, result in enumerate(f1_summary):
        mean = np.mean(result)
        std = np.std(result)
    print("remove_ratio {:.1f}, active_party {:.1f}, k_percent {:.1f}: F1 mean={}, std={}".format(remove_ratio, active_party, k_percent ,mean, std))
    
    mean_acc = np.mean(acc_summary)

    mean_f1 = np.mean(f1_summary)
    std_f1 = np.std(f1_summary)

    mean_prec = np.mean(prec_list)
    std_prec = np.std(prec_list)

    mean_recall = np.mean(recall_list)
    std_recall = np.std(recall_list)

    print("remove_ratio {:.1f}, active_party {:.1f}, k_percent {:.1f}: accuracy mean={}, std={}".format(remove_ratio, active_party, k_percent, mean_acc, std))
    print("remove_ratio {:.1f}, active_party {:.1f}, k_percent {:.1f}: F1 mean={}, std={}".format(remove_ratio, active_party, k_percent ,mean_f1, std_f1))
    print("remove_ratio {:.1f}, active_party {:.1f}, k_percent {:.1f}: precision mean={}, std={}".format(remove_ratio, active_party, k_percent ,mean_prec, std_prec))
    print("remove_ratio {:.1f}, active_party {:.1f}, k_percent {:.1f}: recall mean={}, std={}".format(remove_ratio, active_party, k_percent ,mean_recall, std_recall))

    return mean_acc, mean_f1, mean_prec, mean_recall
 

def run_vertical_fl_remove_all_ration():
    ratios = np.arange(0.2, 1.0, 0.1)
    Parallel(n_jobs=6)(delayed(train_fedonce)(remove_ratio = ratio) for ratio in ratios)

def run_vertical_fl_split_all_ration():
    betas = np.arange(0.2, 0.8, 0.1)
    Parallel(n_jobs=6)(delayed(train_fedonce)(beta = beta) for beta in betas)

def run_vertical_fl_multiple_seed():
    seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # seeds = [20]
    results = Parallel(n_jobs=6)(delayed(train_fedonce)(random_state = seed) for seed in seeds)
    acc_mean = np.mean([result[0] for result in results])
    f1_mean = np.mean([result[1] for result in results])
    prec_mean = np.mean([result[2] for result in results])
    recall_mean = np.mean([result[3] for result in results])
    print("Overall accuracy mean: ", acc_mean)
    print("Overall f1 mean: ", f1_mean)
    print("Overall precision mean: ", prec_mean)
    print("Overall recall mean: ", recall_mean)    

def run_vertical_fl_dae_multiple_seed():
    seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    results = Parallel(n_jobs=6)(delayed(train_fedonce_dae)(random_state = seed) for seed in seeds)
    acc_mean = np.mean([result[0] for result in results])
    f1_mean = np.mean([result[1] for result in results])
    prec_mean = np.mean([result[2] for result in results])
    recall_mean = np.mean([result[3] for result in results])
    print("Overall accuracy mean: ", acc_mean)
    print("Overall f1 mean: ", f1_mean)
    print("Overall precision mean: ", prec_mean)
    print("Overall recall mean: ", recall_mean)   

def run_combine_multiple_seed():
    seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    results = Parallel(n_jobs=6)(delayed(train_combine)(random_state = seed) for seed in seeds)
    acc_mean = np.mean([result[0] for result in results])
    f1_mean = np.mean([result[1] for result in results])
    prec_mean = np.mean([result[2] for result in results])
    recall_mean = np.mean([result[3] for result in results])
    print("Overall accuracy mean: ", acc_mean)
    print("Overall f1 mean: ", f1_mean)
    print("Overall precision mean: ", prec_mean)
    print("Overall recall mean: ", recall_mean) 

def run_vertical_fl_noise_all_ration():
    ratios = np.arange(0.1, 1.0, 0.1)
    Parallel(n_jobs=6)(delayed(train_fedonce)(noise_ratio = ratio) for ratio in ratios)

def run_combine_all_ration():
    ratios = np.arange(0.1, 1.0, 0.1)
    Parallel(n_jobs=6)(delayed(train_combine)(remove_ratio = ratio) for ratio in ratios)

def run_combine_ft_selection_all_ration(active_party):
    ratios = np.arange(0.1, 1.0, 0.1)
    Parallel(n_jobs=6)(delayed(train_combine)(active_party = active_party, k_percent = ratio*100) for ratio in ratios)

def run_vertical_fl_ft_selection_all_ration(active_party, select_host = True, remain = False, k1_percent = 100):
    ratios = np.arange(0.1, 1.0, 0.1)
    Parallel(n_jobs=6)(delayed(train_fedonce)(active_party = active_party, k_percent = ratio*100, k1_percent = k1_percent,
                                                  select_host= select_host, remain_selection = remain) for ratio in ratios)

def run_vertical_fl_dae_ft_selection_all_ration(active_party, select_host = True, k1_percent = 100):
    ratios = np.arange(0.1, 1.0, 0.1)
    Parallel(n_jobs=6)(delayed(train_fedonce_dae)(active_party = active_party, k_percent = ratio*100, k1_percent = k1_percent,
                                                  select_host= select_host) for ratio in ratios)

def run_vertical_fl_dae_select_all_ration_multi_seed():
    def select_host(k1_percent = 100):
        ratios = np.arange(0.1, 1.0, 0.1)
        seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        results = Parallel(n_jobs=6)(delayed(train_fedonce_dae)(k_percent = ratio*100, k1_percent = k1_percent,
                                                                random_state = seed) for ratio in ratios for seed in seeds)
        return results
    
    from collections import defaultdict
    final_res = defaultdict(list)
    for k in range(1, 10):
        print(">>>>>>>>>>>>>>>>")
        print("Value of k: ", k)
        results = select_host(k1_percent = k*10)

        # mean_acc, mean_f1, mean_prec, mean_recall, random_state, k_percent, k1_percent
        for res in results:
            k_percent = str(res[5])[:4]
            k1_percent = str(res[6])[:4]
            random_state = res[4]
            final_res[(random_state, k_percent, k1_percent)].append(res)
    print(final_res)


if __name__ == '__main__':
    # importance = get_feature_importance_uci()
    # print("Feature importance: ", importance)
    # train_combine()
    # run_vertical_fl_ssl_all_ration()
    # run_vertical_fl_all_ration()
    # run_vertical_fl_split_all_ration()
    # run_vertical_fl_noise_all_ration()
    # train_fedonce(remove_ratio = 0, active_party= 0, random_state= 20)
    # train_fedonce_scarf(remove_ratio = 0, active_party= 1, k_percent = 100, select_host = True)
    # train_fedonce_two_side(remove_ratio = 0, active_party= 1, k_percent = 100, select_host = True)
    # train_fedonce_split(remove_ratio = 0, active_party= 1)
    # train_fedonce_dae(remove_ratio = 0, active_party= 0, random_state= 50)
    # run_combine_all_ration()    
#     run_vertical_fl_sup_all_ration()
    # run_combine_ft_selection_all_ration(active_party = 0)
    # run_vertical_fl_ft_selection_all_ration(active_party = 1, select_host = True)
    # for k in range(1, 10):
        # print(">>>>>>>>>>>>>>>>")
        # print("Value of k: ", k)
        # run_vertical_fl_ft_selection_all_ration(active_party = 0, select_host = False, remain=False, k1_percent = k*10)
    
    # run_vertical_fl_multiple_seed()

    # run_vertical_fl_dae_multiple_seed()
    # train_fedonce_multi_round(remove_ratio=0, active_party=1)   
    # train_fedonce_dae_multi_round(remove_ratio=0, active_party=1)

    # run_vertical_fl_dae_select_all_ration_multi_seed()

    # train_fedonce_dae(k1_percent=90)
    # run_vertical_fl_dae_ft_selection_all_ration(active_party = 1, select_host = True)
    # for k in range(1, 10):
    #     print(">>>>>>>>>>>>>>>>")
    #     print("Value of k: ", k)
    #     run_vertical_fl_dae_ft_selection_all_ration(active_party = 1, select_host = False, k1_percent = k*10)
    
    # train_fedonce_l1(active_party=1)
    # train_fedonce_dae_l1(active_party=1)
