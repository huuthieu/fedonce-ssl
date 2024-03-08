from utils.data_utils import load_data_cross_validation, load_data_train_test, get_random_noisy_row, load_creditcardfraud, load_uci
from model.fl_model import VerticalFLModel
from model.fl_model_ssl import VerticalFLModel as VerticalFLModelSSL
from model.fl_model_sup import VerticalFLModel as VerticalFLModelSup
from model.single_party_model import SingleParty
from model.split_nn_model import SplitNNModel
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
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


## Feature importance sort

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
    importance = xg_cls.feature_importances_
    print("Finished training. Overall accuracy {}".format(acc))

    # save feature importance
    np.savetxt("cache/feature_importance_uci.txt", importance)

# load importance from file
    importance = np.loadtxt("cache/feature_importance_uci.txt")

def feature_selection(x_train_val, y_train_val, x_test, y_test, k_percent, name):
    print("Starts training XGBoost on uci")
    if os.path.exists(f"cache/uci_feature_{name}.joblib"):
        xg_cls = joblib.load(f"cache/uci_feature_{name}.joblib")
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
        importance = xg_cls.feature_importances_
        print("Finished training. Overall accuracy {}".format(acc))

        # save model
        joblib.dump(xg_cls, f"cache/uci_feature_{name}.joblib")

    # sfm = SelectFromModel(xg_cls, threshold=threshold)
    # sfm.fit(x_train_val, y_train_val)

    importance = xg_cls.feature_importances_

    # Select top k% features
    num_features_to_select = int(len(importance) * k_percent )
    sfm = SelectPercentile(f_classif, percentile=k_percent)
    sfm.fit(x_train_val, y_train_val)

    # Transform the data to include only selected features
    x_train_val_selected = sfm.transform(x_train_val)
    x_test_selected = sfm.transform(x_test)

    print("x_train_val_selected.shape: ", x_train_val_selected.shape)
    print("x_test_selected.shape: ", x_test_selected.shape)

    return x_train_val_selected, x_test_selected

## FedOnce
def train_fedonce(remove_ratio = 0, active_party = 1, beta = 0.5, noise_ratio = 0):
    num_parties = 2
    xs_train_val, y_train_val, xs_test, y_test = load_uci(test_rate = 0.2, remove_ratio=remove_ratio, feature_order=np.argsort(importance),
                                                          feature_ratio_beta = beta, noise_ratio = noise_ratio)
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

def train_fedonce_ssl(remove_ratio = 0.1, ssl = True, unlign_ratio = 0.5):

    num_parties = 2
    xs_train_val, y_train_val, xs_test, y_test = load_uci(test_rate = 0.2)

    active_party = 1
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
            unalign_index = get_random_noisy_row(xs_train[active_party], unlign_ratio)       

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
    out = "Party {}, unlign ratio {:.1f}: Accuracy mean={}, std={}".format(active_party, unlign_ratio, mean, std)
    print(out)

    mean = np.mean(f1_summary)
    std = np.std(f1_summary)
    out = "Party {}, unlign ratio {:.1f}: F1 mean={}, std={}".format(active_party, unlign_ratio, mean, std)
    print(out)
    return mean, std

def train_fedonce_sup(unlign_ratio = 0.1):
    num_parties = 2
    xs_train_val, y_train_val, xs_test, y_test = load_uci(test_rate = 0.2)
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

        unalign_index = get_random_noisy_row(xs_train[active_party], unlign_ratio)       

        aggregate_model = VerticalFLModelSup(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            num_epochs=100,
            num_local_rounds=50,
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
        acc, _, _, _  = aggregate_model.train(xs_train, y_train, xs_val, y_val, unalign_index = unalign_index)
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
    out = "Party {}, unlign_ratio {:.1f}: Accuracy mean={}, std={}".format(active_party, unlign_ratio, mean, std)
    print(out)

    mean = np.mean(f1_summary)
    std = np.std(f1_summary)
    out = "Party {}, unlign_ratio {:.1f}: F1 mean={}, std={}".format(active_party, unlign_ratio, mean, std)
    print(out)
    return mean, std

# combine
def train_combine(remove_ratio = 0, active_party = -1, k_percent = 1):
    xs_train_val, y_train_val, xs_test, y_test = load_uci(test_rate = 0.2, remove_ratio=remove_ratio)

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
    if k_percent < 1:
        x_train_val, x_test = feature_selection(x_train_val, y_train_val, x_test, y_test, k_percent, f"combine_active_{active_party}")

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
        acc, _, _,_ = single_model.train(x_train, y_train, x_val, y_val)
        _, y_test_score = single_model.predict(x_test)
        y_test_pred = np.where(y_test_score > 0.5, 1, 0)
        test_f1 = f1_score(y_test, y_test_pred)
        acc_summary.append(acc)
        f1_summary.append(test_f1)
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

def run_vertical_fl_remove_all_ration():
    ratios = np.arange(0.2, 1.0, 0.1)
    Parallel(n_jobs=6)(delayed(train_fedonce)(remove_ratio = ratio) for ratio in ratios)

def run_vertical_fl_split_all_ration():
    betas = np.arange(0.2, 0.8, 0.1)
    Parallel(n_jobs=6)(delayed(train_fedonce)(beta = beta) for beta in betas)

def run_vertical_fl_noise_all_ration():
    ratios = np.arange(0.1, 1.0, 0.1)
    Parallel(n_jobs=6)(delayed(train_fedonce)(noise_ratio = ratio) for ratio in ratios)

def run_vertical_fl_ssl_all_ration():
    ratios = np.arange(0.2, 1.0, 0.1)
    Parallel(n_jobs=6)(delayed(train_fedonce_ssl)(unlign_ratio = ratio) for ratio in ratios)

def run_vertical_fl_sup_all_ration():
    ratios = np.arange(0.2, 1.0, 0.1)
    Parallel(n_jobs=6)(delayed(train_fedonce_sup)(unlign_ratio = ratio) for ratio in ratios)

def run_combine_all_ration():
    ratios = np.arange(0.1, 1.0, 0.1)
    Parallel(n_jobs=6)(delayed(train_combine)(remove_ratio = ratio) for ratio in ratios)

def run_combine_ft_selection_all_ration(active_party):
    ratios = np.arange(0.1, 1.0, 0.1)
    Parallel(n_jobs=6)(delayed(train_combine)(active_party = active_party, k_percent = ratio) for ratio in ratios)

if __name__ == '__main__':
    # train_combine(remove_ratio=0, active_party = 0, k_percent = 0.1)
    # train_fedonce_ssl(unlign_ratio = 0.1)
    # run_vertical_fl_ssl_all_ration()
    # run_vertical_fl_all_ration()
    # run_vertical_fl_split_all_ration()
    # run_vertical_fl_noise_all_ration()
    # train_fedonce(remove_ratio = 0, active_party= 1)
    # run_combine_all_ration()    
    # train_fedonce_sup(unlign_ratio = 0.9)
#     run_vertical_fl_sup_all_ration()
    run_combine_ft_selection_all_ration(active_party = 0)