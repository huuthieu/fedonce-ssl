import os.path
import wget
import bz2
import shutil
import numpy as np

import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, log_loss, precision_score, recall_score
from sklearn.model_selection import KFold

import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from utils.data_utils import load_data_cross_validation, load_movielens, load_creditcardfraud
from model.fl_model import VerticalFLModel
from model.single_party_model import SingleParty
from model.split_nn_model import SplitNNModel
from model.fl_model_dae import VerticalFLModel as VerticalFLModelDae

import joblib
from joblib import Parallel, delayed

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

def train_fedonce(remove_ratio = 0, active_party = 0, beta = 0.5, noise_ratio = 0, k_percent = 100,  k1_percent = 100, select_host = True,
                  remain_selection = False, random_state = 50, reps = 0.5):
    num_parties = 2
    xs_train_val, y_train_val, xs_test, y_test = load_creditcardfraud("data/creditcard/creditcard.csv", use_cache = False,
                                                        random_state= random_state,
                                                        test_rate = 0.2)

    
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
        model_name = "vertical_fl_creditcard_party_{}_fold_{}".format(num_parties, i)
        name = "{}_active_{}_k1_{}_k_{}_random_state_{}".format(model_name, active_party, k1_percent, k_percent, random_state)
        writer = SummaryWriter("runs/{}".format(name))
        aggregate_model = VerticalFLModel(
            num_parties=num_parties,
            active_party_id=active_party,
            name=model_name,
            full_name=name,
            num_epochs=100,
            num_local_rounds=100,
            local_lr=1e-4,
            local_hidden_layers=[100, 100, 50],
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
            delta=1.0/xs_train[0].shape[0],
            repr_noise = reps
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

def train_fed_once(num_parties):
    score_summary = []
    xs_train_val, y_train_val, xs_test, y_test = load_creditcardfraud("data/creditcard/creditcard.csv", use_cache = False,
                                                        test_rate = 0.2)

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

def train_fedonce_dae(remove_ratio = 0, active_party = 0, beta = 0.5, noise_ratio = 0, k_percent = 100, select_host = True, random_state = 50,
                      k1_percent = 100, reps = 0.0):
    num_parties = 2
    xs_train_val, y_train_val, xs_test, y_test = load_creditcardfraud("data/creditcard/creditcard.csv", use_cache = False,
                                                        random_state= random_state,
                                                        test_rate = 0.2)

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
            delta=1.0/xs_train[0].shape[0],
            repr_noise = reps
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

def train_solo(random_state = 20):
    num_parties = 1
    xs_train_val, y_train_val, xs_test, y_test = load_creditcardfraud("data/creditcard/creditcard.csv", use_cache = False,
                                                        test_rate = 0.2, random_state = random_state)
    # x_train_val = np.concatenate(xs_train_val, axis=1)
    # x_test = np.concatenate(xs_test, axis=1)
    x_train_val = xs_train_val[0]
    x_test = xs_test[0]
    print(x_train_val.shape)
    print(x_test.shape)
    kfold = KFold(n_splits=5, shuffle=True, random_state=0).split(y_train_val)
    f1_list = []
    prec_list = []
    recall_list = []
    score_list = []
    for i, (train_idx, val_idx) in enumerate(kfold):
        print("Cross Validation Fold {}".format(i))
        x_train = x_train_val[train_idx]
        y_train = y_train_val[train_idx]
        x_val = x_train_val[val_idx]
        y_val = y_train_val[val_idx]
        name = "combine_movielens_fold_{}_random_state_{}".format(i, random_state)
        writer = SummaryWriter("runs/{}".format(name))
        single_model = SingleParty(
            party_id=0,
            num_epochs=100,
            lr=1e-4,
            name = name,
            hidden_layers=[50, 30, 10],
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
        acc, _, _, _ = single_model.train(x_train, y_train, x_val, y_val)
        _, y_score_test = single_model.predict(x_test)
        y_test_pred = np.where(y_score_test > 0.5, 1, 0)
        test_f1 = f1_score(y_test, y_test_pred)
        f1_list.append(test_f1)
        score_list.append(acc)

        test_prec = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)

        prec_list.append(test_prec)
        recall_list.append(test_recall)
        
        print(single_model.params)
        
    # explainer = shap.Explainer(single_model.predict_score, x_train)
    # shap_values = explainer.shap_values(x_test)
    # feature_importance = np.abs(shap_values).mean(axis=0)
    # print("Feature importance: ", feature_importance)   
    # print("-------------------------------------------------")
    print("Best accuracy={}".format(f1_list))

    mean_acc = np.mean(score_list)
    
    mean_f1 = np.mean(f1_list)
    std_f1 = np.std(f1_list)

    mean_prec = np.mean(prec_list)
    std_prec = np.std(prec_list)

    mean_recall = np.mean(recall_list)
    std_recall = np.std(recall_list)
    return mean_acc, mean_f1, mean_prec, mean_recall


def run_solo_multiple_seed():
    # seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    seeds = [10]
    results = Parallel(n_jobs=6)(delayed(train_solo)(random_state = seed) for seed in seeds)
    acc_mean = np.mean([result[0] for result in results])
    f1_mean = np.mean([result[1] for result in results])
    prec_mean = np.mean([result[2] for result in results])
    recall_mean = np.mean([result[3] for result in results])
    print("Overall accuracy mean: ", acc_mean)
    print("Overall f1 mean: ", f1_mean)
    print("Overall precision mean: ", prec_mean)
    print("Overall recall mean: ", recall_mean)  


def run_vertical_fl_multiple_seed():
    # seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    seeds = [10, 20]
    # seeds = [60, 70, 80, 90, 100]
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
    # seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # seeds = [10, 20, 30, 40, 50]
    seeds = [60, 70, 80, 90, 100]
    results = Parallel(n_jobs=6)(delayed(train_fedonce_dae)(random_state = seed) for seed in seeds)
    acc_mean = np.mean([result[0] for result in results])
    f1_mean = np.mean([result[1] for result in results])
    prec_mean = np.mean([result[2] for result in results])
    recall_mean = np.mean([result[3] for result in results])
    print("Overall accuracy mean: ", acc_mean)
    print("Overall f1 mean: ", f1_mean)
    print("Overall precision mean: ", prec_mean)
    print("Overall recall mean: ", recall_mean)   

if __name__ == "__main__":
#     # train_fed_once(2)
# #     train_split_nn(num_parties = 2, model_path = "saved_model")
# #     train_combine()
#     # train_fedonce(active_party=0)
    
    # run_vertical_fl_multiple_seed()
    train_fedonce_dae(active_party=0, random_state = 50)
    # run_vertical_fl_dae_multiple_seed()
    # run_solo_multiple_seed()

#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--number", type=int, default=0)
#     args = parser.parse_args()
#     number = args.number
#     if number == 0:
# #         run_vertical_fl_l1_multiple_seed()
# #         train_fedonce_l1(active_party=1, random_state = 50)
#         run_vertical_fl_multiple_seed()

#     elif number == 1:
# #         train_fedonce_dae_l1(active_party=1, random_state = 50)
# #         run_vertical_fl_dae_l1_multiple_seed()
#         run_vertical_fl_dae_multiple_seed()
