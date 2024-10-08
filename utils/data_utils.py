import warnings

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler

from scipy.sparse import csr_matrix
from imblearn.under_sampling import RandomUnderSampler

import numpy as np
import torch
from torch.utils.data import TensorDataset, Dataset
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, KMNIST
from torchvision.transforms import transforms

from typing import List
import os.path
import pickle
import patoolib
import wget
import zipfile

import pandas as pd

from utils.utils import get_closest_factor

from utils.exceptions import *

from ucimlrepo import fetch_ucirepo 


np.random.seed(10)


def vertical_split(x, num_parties):
    if isinstance(x, csr_matrix):
        x = x.todense()

    num_features = x.shape[1]
    xs = []
    for i in range(num_parties):
        if i == num_parties - 1:
            x_train_party_i = x[:, i * num_features // num_parties:]
        else:
            x_train_party_i = x[:, i * num_features // num_parties: (i + 1) * num_features // num_parties]
        xs.append(x_train_party_i)
    return xs


def bias_vertical_split_ratio(x_train, x_test, num_good_features, good_feature_ratio_alpha):
    # split data into biased two parts
    if isinstance(x_train, csr_matrix):
        x_train = x_train.todense()
    if isinstance(x_train, csr_matrix):
        x_test = x_test.todense()

    num_features = x_train.shape[1]
    assert num_good_features <= num_features // 2, "Too many good features"
    split_index = int(num_good_features * good_feature_ratio_alpha)
    good_feature_range = np.arange(num_features - num_good_features, num_features)
    np.random.shuffle(good_feature_range)
    a_good_feature_range = good_feature_range[:split_index]
    b_good_feature_range = good_feature_range[split_index:]
    other_feature_range = np.arange(0, num_features - num_good_features)
    np.random.shuffle(other_feature_range)
    a_other_feature_range = other_feature_range[:num_features // 2 - a_good_feature_range.size]
    b_other_feature_range = other_feature_range[num_features // 2 - a_good_feature_range.size:]
    a_feature_range = np.concatenate([a_good_feature_range, a_other_feature_range])
    b_feature_range = np.concatenate([b_good_feature_range, b_other_feature_range])
    assert np.abs(a_feature_range.size - b_feature_range.size) <= 1
    a_train = x_train[:, a_feature_range]
    b_train = x_train[:, b_feature_range]
    a_test = x_test[:, a_feature_range]
    b_test = x_test[:, b_feature_range]
    return [a_train, b_train], [a_test, b_test]

def bias_vertical_split(x, beta, noise_ratio = 0, active_party = 0):
    np.random.seed(0)
    # split data into biased two parts
    if isinstance(x, csr_matrix):
        x = x.todense()

    num_features = x.shape[1]
    split_index = int(num_features * beta)
    a = x[:, :split_index]
    b = x[:, split_index:]
    
    if noise_ratio > 0:
        if active_party == 0:
            b = add_noise(b, noise_ratio)
        else:
            a = add_noise(a, noise_ratio)
    
    return [a, b]

def bias_vertical_split_overlap(x, beta, overlap_ratio = 0.2):
    np.random.seed(0)
    # split data into biased two parts
    if isinstance(x, csr_matrix):
        x = x.todense()
        
    num_features = x.shape[1]
    num_overlap_features = int(num_features * overlap_ratio)
    overlap_index = np.random.choice(num_features, num_overlap_features, replace=False)
    overlap_part = x[:, overlap_index]
    
    x_remain = x = np.delete(x, overlap_index, axis=1)

    num_remain_cols = num_features - num_overlap_features
    cols_a = int(num_remain_cols * beta)
    
    indices_a = np.random.choice(num_remain_cols, cols_a, replace=False)
    indices_b = np.setdiff1d(np.arange(num_remain_cols), indices_a)

    a = np.concatenate([x_remain[:,indices_a], overlap_part], axis =1)
    b = np.concatenate([x_remain[:,indices_b], overlap_part], axis =1)
    
    return [a, b]

def add_noise(x, ratio = 0.3):
    """
    Thêm nhiễu vào làm biến đổi tính chất của một số hàng trong dữ liệu.

    Parameters:
    - x: numpy array, ma trận dữ liệu
    - ratio: float, tỷ lệ hàng có nhiễu (0 <= ratio <= 1)

    Returns:
    - numpy array, ma trận dữ liệu mới với nhiễu
    """
    np.random.seed(0)

    # Số hàng cần thêm nhiễu
    num_noisy_rows = int(ratio * x.shape[0])

    # Chọn ngẫu nhiên các hàng để thêm nhiễu
    noisy_rows_indices = np.random.choice(x.shape[0], num_noisy_rows, replace=False)
#     print(noisy_rows_indices)

    # Tạo ma trận nhiễu với giá trị ngẫu nhiên từ phân phối Gaussian
    noise = np.random.normal(0, 1, size=(num_noisy_rows, x.shape[1]))

    # Thêm nhiễu vào các hàng đã chọn
    x[noisy_rows_indices, :] += noise

    return x

def get_random_noisy_row(x, ratio):
    np.random.seed(0)
    num_noisy_rows = int(ratio * x.shape[0])

    # Chọn ngẫu nhiên các hàng để thêm nhiễu
    noisy_rows_indices = np.random.choice(x.shape[0], num_noisy_rows, replace=False)

    return noisy_rows_indices

def rm_noise(x, ratio = 0.95):
    noisy_rows_indices = get_random_noisy_row(x, ratio)

    x = np.delete(x, noisy_rows_indices, axis=0)

    return x
    
def rm_noise_list(list_x, ratio = 0.95):
    result = []
    for x in list_x:
        result.append(rm_noise(x, ratio))
    return result

def vertical_split_image(x, num_parties):
    if len(x.shape) != 4:
        print("Wrong format of image, got {}".format(str(x.shape)))
        raise UnsupportedFormatError

    m, n = x.shape[1], x.shape[2]
    a, b = get_closest_factor(num_parties)  # if num_parties is 9, then m=3, n=3
    if a != b:
        warnings.warn("num_parties is recommended to be perfect square numbers. a={}, b={}".format(a, b))
    if m % a != 0 or n % b != 0:
        warnings.warn("The image size for each party may be not equal. m={}, n={}, a={}, b={}".format(m,n,a,b))
    xs = []
    for i in range(a):
        for j in range(b):
            if i != m - 1 and j != n - 1:
                x_i_j = x[:, i * m // a: (i + 1) * m // a, j * n // b: (j + 1) * n // b, :]
            elif i == m - 1 and j != n - 1:
                x_i_j = x[:, i * m // a:, j * n // b: (j + 1) * n // b, :]
            elif i != m - 1 and j == n - 1:
                x_i_j = x[:, i * m // a: (i + 1) * m // a, j * n // b:, :]
            else:
                x_i_j = x[:, i * m // a:, j * n // b:, :]
            xs.append(x_i_j)
    return xs


def load_data_train_test(train_file_name, test_file_name=None, num_parties=1, test_size=0.2, root="data/",
                         file_type='libsvm', X=None, y=None):
    print("Data splitting")
    if file_type == 'libsvm':
        x_train, y_train = load_svmlight_file(root + train_file_name)
        x_train = x_train.todense()
        x_train = np.asarray(x_train)
        normalizer = Normalizer().fit(x_train)
        x_scaler = MinMaxScaler(feature_range=(0, 1))
        x_scaler.fit(x_train)
        x_train = normalizer.transform(x_scaler.transform(x_train))
    elif file_type == 'csv':
        dataset = np.loadtxt(root + train_file_name, delimiter=',', skiprows=1)
        x_train = dataset[:, :-1]
        y_train = dataset[:, -1].reshape(-1)
        normalizer = Normalizer().fit(x_train)
        x_scaler = MinMaxScaler(feature_range=(0, 1))
        x_scaler.fit(x_train)
        x_train = normalizer.transform(x_scaler.transform(x_train))
    elif file_type == 'torch':
        if train_file_name == 'cifar10':
            # trainset will be normalized later in vertical_fl.py after data argumentation
            dataset = CIFAR10(root=root, train=True, transform=None, download=True)
            x_train, y_train = dataset.data, np.array(dataset.targets)
        elif train_file_name == 'mnist':
            dataset = MNIST(root=root, train=True, transform=None, download=True)
            x_train, y_train = dataset.data.detach().numpy()[:, :, :, None], np.array(dataset.targets)
        elif train_file_name == 'fashion_mnist':
            dataset = FashionMNIST(root=root, train=True, transform=None, download=True)
            x_train, y_train = dataset.data.detach().numpy()[:, :, :, None], np.array(dataset.targets)
        elif train_file_name == 'kmnist':
            dataset = KMNIST(root=root, train=True, transform=None, download=True)
            x_train, y_train = dataset.data.detach().numpy()[:, :, :, None], np.array(dataset.targets)
        else:
            raise UnsupportedFormatError
    elif file_type == 'numpy':
        print("Loading existing numpy array")
        assert X is not None and y is not None and test_file_name is None
        x_train, y_train = X, y
        normalizer = Normalizer().fit(x_train)
        x_scaler = MinMaxScaler(feature_range=(0, 1))
        x_scaler.fit(x_train)
        x_train = normalizer.transform(x_scaler.transform(x_train))
    else:
        raise UnsupportedFormatError

    if test_file_name:
        if file_type == 'libsvm':
            x_test, y_test = load_svmlight_file(root + test_file_name)
            x_test = x_test.todense()
            x_test = np.asarray(x_test)
            x_test = normalizer.transform(x_scaler.transform(x_test))
        elif file_type == 'csv':
            dataset = np.loadtxt(root + test_file_name, delimiter=',', skiprows=1)
            x_test = dataset[:, :-1]
            y_test = dataset[:, -1].reshape(-1)
            x_test = normalizer.transform(x_scaler.transform(x_test))
        elif file_type == 'torch':
            if test_file_name == 'cifar10':
                dataset = CIFAR10(root=root, train=False, transform=None, download=True)
                x_test, y_test = dataset.data, np.array(dataset.targets)
            elif test_file_name == 'mnist':
                dataset = MNIST(root=root, train=False, transform=None, download=True)
                x_test, y_test = dataset.data.detach().numpy()[:, :, :, None], np.array(dataset.targets)
            elif test_file_name == 'fashion_mnist':
                dataset = FashionMNIST(root=root, train=False, transform=None, download=True)
                x_test, y_test = dataset.data.detach().numpy()[:, :, :, None], np.array(dataset.targets)
            elif test_file_name == 'kmnist':
                dataset = KMNIST(root=root, train=False, transform=None, download=True)
                x_test, y_test = dataset.data.detach().numpy()[:, :, :, None], np.array(dataset.targets)
            else:
                raise UnsupportedFormatError
        else:
            raise UnsupportedFormatError
    else:
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_size, random_state=0)

    # scale labels if regression or binary classification
    if file_type in ['libsvm', 'csv', 'numpy']:
        scalar = MinMaxScaler(feature_range=(0, 1))
        scalar.fit(y_train.reshape(-1, 1))
        y_train = scalar.transform(y_train.reshape(-1, 1)).reshape(-1)
        y_test = scalar.transform(y_test.reshape(-1, 1)).reshape(-1)
    elif file_type in ['torch']:
        pass
    else:
        raise UnsupportedFormatError

    # split data into parties
    if file_type in ['libsvm', 'csv', 'numpy']:
        xs_train = vertical_split(x_train, num_parties)
        xs_test = vertical_split(x_test, num_parties)
    elif file_type in ['torch']:
        xs_train = vertical_split_image(x_train, num_parties)
        xs_test = vertical_split_image(x_test, num_parties)
    else:
        raise UnsupportedFormatError
    print("Finished loading data")

    return xs_train, y_train, xs_test, y_test, x_train, x_test


def dedup_data(X, y):
    _, unique_indices = np.unique(X, axis=0, return_index=True)
    X = X[unique_indices]
    y = y[unique_indices]
    return X, y


def load_data_cross_validation(file_name, num_parties=1, root="data/", file_type='libsvm', n_fold=5,
                               use_cache=False, cache_path=None, csv_skiprows=1, feature_order=None,
                               shift_alpha=None, feature_ratio_beta=None,
                               num_good_features=None, good_feature_ratio_alpha=None,
                               x_scaler_wrapper=None, y_scaler_wrapper=None, x_normalizer_wrapper=None,
                               add_noise = False,
                              remove_noise = False,
                              overlap_ratio =  0,
                              random_state = 10):
    if use_cache and os.path.isfile(cache_path):
        print("Loading data from cache: " + cache_path)
        with open(cache_path, 'rb') as f:
            results = pickle.load(f)
            return results

    print("Loading data from file")
    if file_type == 'libsvm':
        x, y = load_svmlight_file(root + file_name)
        x = x.todense()
        x = np.asarray(x)
        x, y = dedup_data(x, y)

        print("Percentage of positive labels: " + str(np.sum(y) / len(y) * 100))
        
        if x_normalizer_wrapper is not None and len(x_normalizer_wrapper) == 1:
            normalizer = x_normalizer_wrapper[0]
        else:
            normalizer = Normalizer().fit(x)
            if x_normalizer_wrapper is not None:
                x_normalizer_wrapper.append(normalizer)
        if x_scaler_wrapper is not None and len(x_scaler_wrapper) == 1:
            x_scaler = x_scaler_wrapper[0]
        else:
            x_scaler = MinMaxScaler(feature_range=(0, 1))
            x_scaler.fit(x)
            if x_scaler_wrapper is not None:
                x_scaler_wrapper.append(x_scaler)
        x = normalizer.transform(x_scaler.transform(x))
    elif file_type == 'csv':
        try:
            dataset = np.loadtxt(root + file_name, delimiter=',', skiprows=csv_skiprows)
        except ValueError:
            dataset = np.genfromtxt(root + file_name, delimiter=',', skip_header=csv_skiprows, filling_values=0)
        x = dataset[:, :-1]
        y = dataset[:, -1].reshape(-1)
        if x_normalizer_wrapper is not None and len(x_normalizer_wrapper) == 1:
            normalizer = x_normalizer_wrapper[0]
        else:
            normalizer = Normalizer().fit(x)
            if x_normalizer_wrapper is not None:
                x_normalizer_wrapper.append(normalizer)
        if x_scaler_wrapper is not None and len(x_scaler_wrapper) == 1:
            x_scaler = x_scaler_wrapper[0]
        else:
            x_scaler = MinMaxScaler(feature_range=(0, 1))
            x_scaler.fit(x)
            if x_scaler_wrapper is not None:
                x_scaler_wrapper.append(x_scaler)
        x = normalizer.transform(x_scaler.transform(x))

    elif file_type == 'torch':
        if file_name == 'cifar10':
            # trainset will be normalized later in vertical_fl.py after data argumentation
            dataset = CIFAR10(root=root, train=True, transform=None, download=True)
            x, y = dataset.data, np.array(dataset.targets)
        elif file_name == 'mnist':
            dataset = MNIST(root=root, train=True, transform=None, download=True)
            x, y = dataset.data.detach().numpy()[:, :, :, None], np.array(dataset.targets)
        else:
            raise UnsupportedFormatError
    else:
        raise UnsupportedFormatError

    print("Scaling labels")
    # scale labels if regression or binary classification
    if file_type in ['libsvm', 'csv']:
        if y_scaler_wrapper is not None and len(y_scaler_wrapper) == 1:
            scalar = y_scaler_wrapper[0]
        else:
            scalar = MinMaxScaler(feature_range=(0, 1))
            scalar.fit(y.reshape(-1, 1))
            if y_scaler_wrapper is not None:
                y_scaler_wrapper.append(scalar)
        y = scalar.transform(y.reshape(-1, 1)).reshape(-1)
    elif file_type in ['torch']:
        pass
    else:
        raise UnsupportedFormatError

    if feature_order is not None:
        assert feature_order.size == x.shape[1], "Feature orders mismatch the number of features"
        x = x[:, feature_order]

    rng = np.random.RandomState(random_state)

    # Tạo một chỉ số hoán đổi
    permutation_indices = rng.permutation(x.shape[1])   
    x = x[:, permutation_indices] 

    if shift_alpha is not None:
        shift = -int(x.shape[1] * shift_alpha)
        x = np.roll(x, shift, axis=1)

    print("{} fold spliltting".format(n_fold))
    results = []
    if n_fold > 1:
        k_fold = KFold(n_splits=n_fold, shuffle=True, random_state=10)
        for i, (train_idx, test_idx) in enumerate(k_fold.split(x, y)):
            x_train = x[train_idx]
            y_train = y[train_idx]
            x_test = x[test_idx]
            y_test = y[test_idx]

            # split data into parties
            if file_type in ['libsvm', 'csv']:
                if feature_ratio_beta is not None:
                    assert num_parties == 2
                    if remove_noise:
                        xs_train = bias_vertical_split(x_train, feature_ratio_beta)
                        xs_train = rm_noise_list([*xs_train])
                        y_train = rm_noise(y_train)
                        xs_test = bias_vertical_split(x_test, feature_ratio_beta)
                    elif overlap_ratio > 0:
                        xs_train = bias_vertical_split_overlap(x_train, feature_ratio_beta, overlap_ratio)
                        xs_test = bias_vertical_split_overlap(x_test, feature_ratio_beta, overlap_ratio)
                    else:
                        xs_train = bias_vertical_split(x_train, feature_ratio_beta, add_noise)
                        xs_test = bias_vertical_split(x_test, feature_ratio_beta)
                elif good_feature_ratio_alpha is not None and num_good_features is not None:
                    assert num_parties == 2
                    xs_train, xs_test = bias_vertical_split_ratio(
                        x_train, x_test, num_good_features, good_feature_ratio_alpha)
                else:
                    if remove_noise:
                        xs_train = vertical_split(x_train, num_parties)
                        xs_test = vertical_split(x_test, num_parties)
                        xs_train = rm_noise_list([*xs_train])
                        y_train = rm_noise(y_train)
                    else:
                        xs_train = vertical_split(x_train, num_parties)
                        xs_test = vertical_split(x_test, num_parties)
            elif file_type in ['torch']:
                xs_train = vertical_split_image(x_train, num_parties)
                xs_test = vertical_split_image(x_test, num_parties)
            else:
                raise UnsupportedFormatError
            
            results.append([xs_train, y_train, xs_test, y_test])
            print("Fold {} finished".format(i))
    else:       # fold = 1
        # split data into parties
        if file_type in ['libsvm', 'csv']:
            if feature_ratio_beta is not None:
                assert num_parties == 2
                xs = bias_vertical_split(x, feature_ratio_beta)
            elif good_feature_ratio_alpha is not None and num_good_features is not None:
                assert False, "Unsupported"
                # assert num_parties == 2
                # xs_train, xs_test = bias_vertical_split_ratio(
                #     x, x, num_good_features, good_feature_ratio_alpha)
            else:
                xs = vertical_split(x, num_parties)
        elif file_type in ['torch']:
            xs = vertical_split_image(x, num_parties)
        else:
            raise UnsupportedFormatError
        results.append([xs, y])

    if use_cache and cache_path is not None:
        with open(cache_path, 'wb') as f:
            pickle.dump(results, f)
        print("Saved results to cache: " + str(cache_path))

    return results


def move_item_to_end_(arr, items):
    for item in items:
        arr.insert(len(arr), arr.pop(arr.index(item)))


def move_item_to_start_(arr, items):
    for item in items[::-1]:
        arr.insert(0, arr.pop(arr.index(item)))

class NYBikeTaxiLoader:
    def __init__(self, bike_path, taxi_path=None, link=False):
        print("Loading bike from {}".format(bike_path))
        self.bike_data = pd.read_pickle(bike_path)
        # self.bike_data = self.bike_data.head(10000)
        # print("Remove N/A from bike")
        # self.bike_data.dropna()
        print("Loaded.")
        if taxi_path is not None:
            print("Loading taxi from {}".format(taxi_path))
            self.taxi_data = pd.read_pickle(taxi_path)
            print("Loaded.")

        if link:
            self.labels = self.bike_data['tripduration'].to_numpy()
            self.bike_data.drop(columns=['tripduration'], inplace=True)

            # move lon and lat to end of airbnb
            bike_cols = list(self.bike_data)
            move_item_to_end_(bike_cols, ['start_lon', 'start_lat', 'end_lon', 'end_lat',
                                          'start_hour', 'end_hour'])
            self.bike_data = self.bike_data[bike_cols]
            self.bike_data.drop(columns=['start_hour', 'end_hour'], inplace=True)
            print("Current bike columns: " + str(list(self.bike_data)))
            self.bike_data = self.bike_data.to_numpy()

            # move lon and lat to the front of taxi
            tx_cols = list(self.taxi_data)
            move_item_to_start_(tx_cols, ['start_lon', 'start_lat', 'end_lon', 'end_lat',
                                          'start_hour', 'end_hour'])
            self.taxi_data = self.taxi_data[tx_cols]
            self.taxi_data.drop(columns=['start_hour', 'end_hour'], inplace=True)
            print("Current taxi columns: " + str(list(self.taxi_data)))
            self.taxi_data = self.taxi_data.to_numpy()
        else:
            print("Remove columns that are used for linkage")
            self.bike_data.drop(columns=['start_lon', 'start_lat', 'end_lon', 'end_lat',
                                         'start_hour', 'end_hour'], inplace=True)
            print('Extract labels')
            self.labels = self.bike_data['tripduration'].to_numpy()
            print("Extract data")
            self.bike_data = self.bike_data.drop(columns=['tripduration']).to_numpy()

    def load_single(self):
        return self.bike_data, self.labels

    def load_parties(self):
        return [self.bike_data, self.taxi_data], self.labels


def load_bike(train_file_name, num_parties=1, test_size=0.2, root="data/"):
    loader = NYBikeTaxiLoader(os.path.join(root, train_file_name))
    X, y = loader.load_single()
    print("Shape of X: {}".format(X.shape))
    return load_data_train_test(train_file_name, num_parties=num_parties, test_size=test_size, root=root,
                                file_type='numpy', X=X, y=y)


def load_nus_wide(path, download=True, label_type='airport', use_cache=True, balance=False, num_parties=2, raw_split=False):
    if use_cache and os.path.exists("cache/nus-wide.npy"):
        print("Loading nus-wide from cache")
        result = np.load("cache/nus-wide.npy", allow_pickle=True)
    else:
        if download:
            if not os.path.isdir(path):
                os.mkdir(path)
            if not (os.path.isdir(path + "/Low_Level_Features/") and
                    os.path.isdir(path + "/TrainTestLabels") and
                    os.path.isfile(path + "/Train_Tags1k.dat") and
                    os.path.isfile(path + "/Test_Tags1k.dat")):
                if not os.path.isfile(path + "/Low_Level_Features.rar"):
                    print("Start Downloading NUS-WIDE features")
                    wget.download(
                        "https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS_WID_Low_Level_Features.rar",
                        out=path + "/Low_Level_Features.rar")
                if not os.path.isfile(path + "/Groundtruth.zip"):
                    print("Start Downloading NUS-WIDE labels")
                    wget.download(
                        "https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/Groundtruth.zip",
                        out=path + "/Groundtruth.zip")
                if not (os.path.isfile(path + "/NUS_WID_Tags.zip") and os.path.isfile(path + "/Test_Tags1k.dat")):
                    print("Starting Downloading NUS-WIDE tags")
                    wget.download(
                        "https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS_WID_Tags.zip",
                        out=path + "/NUS_WID_Tags.zip")
                print("Extracting ...")
                patoolib.extract_archive(path + "/Low_Level_Features.rar", outdir=path)
                with zipfile.ZipFile(path + "/Groundtruth.zip", 'r') as zip_ref:
                    zip_ref.extractall(path)
                with zipfile.ZipFile(path + "/NUS_WID_Tags.zip", 'r') as zip_ref:
                    zip_ref.extractall(path)
            else:
                print("Found existed NUS-WIDE data")

        # read image features
        print("Read image features ...")
        img_file_ids = ['CH', 'CORR', 'EDH', 'WT', 'CM55']
        train_img_features = []
        test_img_features = []
        for id in img_file_ids:
            print("Reading " + id)
            train_data = np.loadtxt(path + "/Low_Level_Features/Train_Normalized_" + id + ".dat")
            test_data = np.loadtxt(path + "/Low_Level_Features/Test_Normalized_" + id + ".dat")
            train_img_features.append(train_data)
            test_img_features.append(test_data)
        train_img_data = np.concatenate(train_img_features, axis=1)
        test_img_data = np.concatenate(test_img_features, axis=1)

        # read labels
        print("Reading labels")
        train_label_path = path + "/TrainTestLabels/Labels_" + label_type + "_Train.txt"
        test_label_path = path + "/TrainTestLabels/Labels_" + label_type + "_Test.txt"
        train_labels = np.loadtxt(train_label_path)
        test_labels = np.loadtxt(test_label_path)

        if raw_split:
            if num_parties == 5:
                result = train_img_features, train_labels, test_img_features, test_labels
            elif num_parties == 1:
                result = train_img_data, train_labels, test_img_data, test_labels
            else:
                assert False
        else:
            # train_img_data = np.concatenate(train_img_features, axis=1)
            # test_img_data = np.concatenate(test_img_features, axis=1)

            # read text features
            print("Reading text features ...")
            train_text_data = np.loadtxt(path + "/Train_Tags1k.dat")
            test_text_data = np.loadtxt(path + "/Test_Tags1k.dat")

            if num_parties == 2:
                result = [train_img_data, train_text_data], train_labels, [test_img_data, test_text_data], test_labels
            else:
                train_text_data = vertical_split(train_text_data, num_parties - 1)
                test_text_data = vertical_split(test_text_data, num_parties - 1)
                result = [train_img_data] + train_text_data, train_labels, [test_img_data] + test_text_data, test_labels

        if use_cache:
            print("Saving to cache ...")
            np.save("cache/nus-wide.npy", result, allow_pickle=True)

    print("Finished loading NUS-WIDE data")

    if balance:
        assert num_parties == 2
        [train_img_data, train_text_data], train_labels, [test_img_data, test_text_data], test_labels = result

        # under sampling to balance the labels
        train_sampler = RandomUnderSampler(random_state=0)
        train_img_data_sample, train_labels_sample = train_sampler.fit_sample(train_img_data, train_labels)
        train_text_data_sample = train_text_data[train_sampler.sample_indices_, :]

        test_sampler = RandomUnderSampler(random_state=0)
        test_img_data_sample, test_labels_sample = test_sampler.fit_sample(test_img_data, test_labels)
        test_text_data_sample = test_text_data[test_sampler.sample_indices_, :]

        result = [train_img_data_sample, train_text_data_sample], train_labels_sample, \
                 [test_img_data_sample, test_text_data_sample], test_labels_sample

    return result


def load_uci(test_rate = 0.2, num_parties = 2, remove_ratio = 0, 
             feature_order = None, feature_ratio_beta = None, 
             noise_ratio = 0, active_party = 0, random_state = 10):
    df = pd.read_csv('data/uci/default_of_credit_card_clients.csv', index_col='ID')
    df.rename(columns={'default payment next month':'DEFAULT'}, inplace=True)
    df.rename(columns={'PAY_0': 'PAY_1'}, inplace=True)
    df.rename(columns=lambda x: x.upper(), inplace=True)
    # remove useless and incorrect information
    print(f"Dataset size before:\t{df.shape[0]}")
    df = df.drop(df[df['MARRIAGE']==0].index)
    df = df.drop(df[df['EDUCATION']==0].index)
    df = df.drop(df[df['EDUCATION']==5].index)
    df = df.drop(df[df['EDUCATION']==6].index)
    print(f"Dataset size after:\t{df.shape[0]}")

    df['GRAD_SCHOOL'] = (df['EDUCATION'] == 1).astype('float')
    df['UNIVERSITY'] = (df['EDUCATION'] == 2).astype('float')
    df['HIGH_SCHOOL'] = (df['EDUCATION'] == 3).astype('float')
    df.drop('EDUCATION', axis=1, inplace=True)

    df['MALE'] = (df['SEX'] == 1).astype('float')
    df.drop('SEX', axis=1, inplace=True)

    df['MARRIED'] = (df['MARRIAGE'] == 1).astype('float')
    df.drop('MARRIAGE', axis=1, inplace=True)

    y = df['DEFAULT']
    X = df.drop('DEFAULT', axis=1, inplace=False)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=test_rate, random_state=24, stratify=y)
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    X_train_std = X_train_raw.copy()
    X_test_std = X_test_raw.copy()

    X_train_std['LIMIT_BAL'] = scaler.fit_transform(X_train_raw['LIMIT_BAL'].values.reshape(-1, 1))
    X_test_std['LIMIT_BAL'] = scaler.transform(X_test_raw['LIMIT_BAL'].values.reshape(-1, 1))
    X_train_std['AGE'] = scaler.fit_transform(X_train_raw['AGE'].values.reshape(-1, 1))
    X_test_std['AGE'] = scaler.transform(X_test_raw['AGE'].values.reshape(-1, 1))
    for i in range(1,7):
        X_train_std['PAY_' + str(i)] = scaler.fit_transform(X_train_raw['PAY_' + str(i)].values.reshape(-1, 1))
        X_test_std['PAY_' + str(i)] = scaler.transform(X_test_raw['PAY_' + str(i)].values.reshape(-1, 1))
        X_train_std['BILL_AMT' + str(i)] = scaler.fit_transform(X_train_raw['BILL_AMT' + str(i)].values.reshape(-1, 1))
        X_test_std['BILL_AMT' + str(i)] = scaler.transform(X_test_raw['BILL_AMT' + str(i)].values.reshape(-1, 1))
        X_train_std['PAY_AMT' + str(i)] = scaler.fit_transform(X_train_raw['PAY_AMT' + str(i)].values.reshape(-1, 1))
        X_test_std['PAY_AMT' + str(i)] = scaler.transform(X_test_raw['PAY_AMT' + str(i)].values.reshape(-1, 1))
    

    X_train_std = X_train_std.to_numpy()
    X_test_std = X_test_std.to_numpy()

    rng = np.random.RandomState(random_state)

    # Tạo một chỉ số hoán đổi
    permutation_indices = rng.permutation(X_train_std.shape[1])

    # Shuffle các cột dựa trên chỉ số hoán đổi
    X_train_std = X_train_std[:, permutation_indices]
    X_test_std = X_test_std[:, permutation_indices]

    if feature_order is not None:
        assert feature_order.size == X_train_std.shape[1], "Feature orders mismatch the number of features"
        X_train_std = X_train_std[:, feature_order]
        X_test_std = X_test_std[:, feature_order]

    if feature_ratio_beta is not None:
        assert num_parties == 2
        x_train = bias_vertical_split(X_train_std, feature_ratio_beta, noise_ratio, active_party)
        x_test = bias_vertical_split(X_test_std, feature_ratio_beta)
    else:
        x_train = bias_vertical_split(X_train_std, 0.5, noise_ratio, active_party)
        x_test = bias_vertical_split(X_test_std, 0.5)
    
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    if remove_ratio > 0:
        x_train = rm_noise_list([*x_train], remove_ratio)
        y_train = rm_noise(y_train, remove_ratio)

    result = [x_train, y_train, x_test, y_test]
    # np.save("cache/creditcardfraud.npy", result, allow_pickle=True)
    
    return result
    

def load_creditcardfraud(path, use_cache = True, test_rate=0.2, num_parties=2,
                         remove_ratio = 0.2, random_state = 10):
    if use_cache:
        print("Loading creditcardfraud from cache")
        result = np.load("cache/creditcardfraud.npy", allow_pickle=True)
    else:
        print("Loading creditcardfraud from file")
        data = pd.read_csv(path)
#         data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

        ## sort by time
        data = data.sort_values('Time')

        ## Drop time column
        data = data.drop(['Time'], axis=1)

        ## Extract label
        label = data['Class']

        ## drop label column
        data = data.drop(['Class'], axis=1)

        for column in data.columns:
            # Create a StandardScaler for each column
            scaler = StandardScaler()
            # Fit and transform the data for that column
            data[column] = scaler.fit_transform(data[column].values.reshape(-1, 1))


        data = data.sample(frac=1, axis = 1, random_state=random_state)
        # data = data.reset_index(drop=True)
        ## add label column to the end
        data['Class'] = label

        # data = data.sample(frac=1) 
        
        data = data.values

        ## SPlit into train and test
        train_data = data[:int((1 - test_rate) * len(data)), :]
        test_data = data[int((1 - test_rate) * len(data)):, :]
        train_label = train_data[:, -1]
        test_label = test_data[:, -1]
        train_data = train_data[:, :-1]
        test_data = test_data[:, :-1]
        ## split into 2 parties with random number of features
#         train_data = vertical_split(train_data, num_parties)
#         test_data = vertical_split(test_data, num_parties)

        train_data = bias_vertical_split(train_data, 0.5)
        test_data = bias_vertical_split(test_data, 0.5)
        
        if remove_ratio > 0:
            train_data = rm_noise_list([*train_data], remove_ratio)
            train_label = rm_noise(train_label, remove_ratio)
        
        result = [train_data, train_label, test_data, test_label]
        # np.save("cache/creditcardfraud.npy", result, allow_pickle=True)
        
    return result

def load_movielens(path, download=True, use_cache=True, test_rate=0.2, num_parties=2):
    if use_cache and os.path.isfile("cache/movielens.npy"):
        print("Loading MovieLens from cache")
        result = np.load("cache/movielens.npy", allow_pickle=True)
    else:
        movie_path = path + "/movies.dat"
        rating_path = path + "/ratings.dat"
        user_path = path + "/users.dat"
        if download:
            if not os.path.isdir(path):
                os.mkdir(path)
            if not os.path.isfile(movie_path):
                print("Start downloading movies")
                wget.download("https://raw.githubusercontent.com/khanhnamle1994/movielens/master/dat/movies.dat",
                              out=movie_path)
            if not os.path.isfile(rating_path):
                print("Start downloading ratings")
                wget.download("https://raw.githubusercontent.com/khanhnamle1994/movielens/master/dat/ratings.dat",
                              out=rating_path)
            if not os.path.isfile(user_path):
                print("Start downloading users")
                wget.download("https://raw.githubusercontent.com/khanhnamle1994/movielens/master/dat/users.dat",
                              out=user_path)

        AGES = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"}
        OCCUPATIONS = {0: "other or not specified", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
                       4: "college/grad student", 5: "customer service", 6: "doctor/health care",
                       7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student", 11: "lawyer",
                       12: "programmer", 13: "retired", 14: "sales/marketing", 15: "scientist", 16: "self-employed",
                       17: "technician/engineer", 18: "tradesman/craftsman", 19: "unemployed", 20: "writer"}

        print("Loading ratings")
        ratings = pd.read_csv(rating_path,
                              sep='::',
                              engine='python',
                              encoding='latin-1',
                              names=['user_id', 'movie_id', 'rating', 'timestamp'])

        print("Loading users")
        users = pd.read_csv(user_path,
                            sep='::',
                            engine='python',
                            encoding='latin-1',
                            names=['user_id', 'gender', 'age', 'occupation', 'zipcode'],
                            usecols=['user_id', 'gender', 'age', 'occupation'])

        print("Loading movies")
        movies = pd.read_csv(movie_path,
                             sep='::',
                             engine='python',
                             encoding='latin-1',
                             names=['movie_id', 'title', 'genres'])

        df = ratings.merge(users, on='user_id').merge(movies, on='movie_id')

        df['user_id'] = df['user_id'] - 1
        df['movie_id'] = df['movie_id'] - 1

        nusers = df['user_id'].max() + 1  # 6040
        nmovies = df['movie_id'].max() + 1  # 3952

        # extract years
        df['date'] = pd.to_datetime(df['timestamp'], unit='s')
        df['year_rating'] = pd.DatetimeIndex(df['date']).year
        df['year_movie'] = df['title'].str.extract(r'\((\d+)\)').astype('int64')
        df['genre'] = df['genres'].transform(lambda s: s.split('|')[0])

        df = df.drop(columns=['timestamp', 'date', 'title', 'genres'])

        cols = ['gender', 'age', 'year_rating', 'year_movie', 'genre']
        cat2is = []  # category to int
        for col in cols:
            cats = sorted(df[col].unique().tolist())
            cat2i = {cat: i for i, cat in enumerate(cats)}
            cat2is.append(cat2i)
            df[col] = df[col].transform(lambda cat: cat2i[cat])

        df = df.sample(frac=1., random_state=0)
        nsamples = df.shape[0]
        split = int(nsamples * (1 - test_rate))
        df_train = df.iloc[:split, :]
        df_test = df.iloc[split:, :]

        train_rating = np.vstack([df_train.user_id.to_numpy(), df_train.movie_id.to_numpy()]).T
        train_aux = np.vstack([df_train.gender.to_numpy(), df_train.age.to_numpy(), df_train.occupation.to_numpy(),
                     df_train.year_rating.to_numpy(), df_train.year_movie.to_numpy(), df_train.genre.to_numpy()]).T
        test_rating = np.vstack([df_test.user_id.to_numpy(), df_test.movie_id.to_numpy()]).T
        test_aux = np.vstack([df_test.gender.to_numpy(), df_test.age.to_numpy(), df_test.occupation.to_numpy(),
                    df_test.year_rating.to_numpy(), df_test.year_movie.to_numpy(), df_test.genre.to_numpy()]).T
        train_label = df_train.rating.to_numpy()
        test_label = df_test.rating.to_numpy()
        counts = [nusers, nmovies] + [len(cat2i) for cat2i in cat2is[:2]] + \
                 [len(df['occupation'].unique())] + [len(cat2i) for cat2i in cat2is[2:]]

        if num_parties == 2:
            result = [[train_rating, train_aux], train_label,
                  [test_rating, test_aux], test_label, counts]
        else:
            train_text_data = vertical_split(train_aux, num_parties - 1)
            test_text_data = vertical_split(test_aux, num_parties - 1)
            result = [train_rating] + train_text_data, train_label, [test_rating] + test_text_data, test_label, counts

        if use_cache:
            print("Saving to cache ...")
            np.save("cache/movielens.npy", result, allow_pickle=True)

    return result


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        if self.transform != None:
            out1 = self.transform(inp)
            out2 = self.transform(inp)
        else:
            out1 = inp
            out2 = inp

        return out1, out2

class LocalDataset(Dataset):
    def __init__(self, data, target, cluster_label = None, transform=None):
        self.data = data
        self.target = target
        self.cluster_label = cluster_label
        self.transform = transform

    def __getitem__(self, index):
        if self.transform is not None:
            data_i = self.transform(self.data[index])
        else:
            data_i = self.data[index]
        if self.cluster_label is not None:
            return index, data_i, self.target[index], self.cluster_label[index]
        return index, data_i, self.target[index]

    def __len__(self):
        return self.target.shape[0]

    def update_targets(self, indices: List[int], new_targets):
        self.target[indices, :] = new_targets

class LocalDatasetSSLLabel(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform

    def __getitem__(self, index):
        if self.transform is not None:
            data_i = self.transform(self.data[index])
        else:
            data_i = self.data[index]
        return index, data_i, self.target[index]
    

    def __len__(self):
        return self.target.shape[0]

class LocalDatasetSSLUnLabel(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        # self.target = target
        self.target = np.array([-1 for i in range(len(target))])
        self.transform = TransformTwice(transform)

    def __getitem__(self, index):
        if self.transform is not None:
            data_i = self.transform(self.data[index])
        else:
            data_i = self.data[index]
        return index, data_i, self.target[index]

    def __len__(self):
        return self.target.shape[0]


class AggDataset(Dataset):
    def __init__(self, X, Z, y, transform=None):
        self.X = X      # X can be images
        self.Z = Z
        self.transform = transform
        self.y = y

    def __getitem__(self, index):
        if self.transform is not None:
            X_i = self.transform(self.X[index])
        else:
            X_i = self.X[index]
        return index, X_i, self.Z[index], self.y[index]

    def __len__(self):
        return self.y.shape[0]

    def update_Z(self, indices: List[int], new_Z):
        self.Z[indices] = new_Z


class ImageDataset(Dataset):
    def __init__(self, xs: list, y=None, transform=None):
        if y is not None:
            for x in xs:
                assert x.shape[0] == y.shape[0]
        self.xs = xs
        self.y = y
        self.transform = transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        result = []
        for x in self.xs:
            x_i = x[idx]
            if self.transform:
                x_i = self.transform(x_i)
            result.append(x_i)

        if self.y is not None:
            result.append(self.y[idx])

        return result

    def __len__(self):
        return self.xs[0].shape[0]


