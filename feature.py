#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Apr-18-20 19:49
# @Author  : Your Name (you@example.org)
# @Link    : https://github.com/KellyHwong/kaggle_digit_recognizer
# @Link    : https://www.kaggle.com/c/digit-recognizer

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from sklearn.utils import shuffle
from torchvision import datasets, transforms
from sklearn.decomposition import PCA

from model import LeNet5
from utils import makedir_exist_ok


def extract_feature():
    with open('./config.json', 'r') as f:
        CONFIG = json.load(f)
    DATA_ROOT = CONFIG["DATA_ROOT"]
    MODEL_DIR = CONFIG["MODEL_DIR"]
    FEATURE_DIR = CONFIG["FEATURE_DIR"]

    npzfile = np.load("train_val_idx.npz")
    train_idx = npzfile["train_idx"]
    val_idx = npzfile["val_idx"]

    os.chdir(DATA_ROOT)
    print(f"Change current work directory to: {DATA_ROOT}")

    print("Step 1: Loading data...")
    data_train = pd.read_csv("data/train.csv").values
    # 一般不应该对 data 进行 shuffle
    # data_train = shuffle(data_train)  # shuffle after loading train csv
    data_test = pd.read_csv("data/test.csv").values
    num_all = len(data_train)  # 42000
    num_test = len(data_test)  # 28000

    use_gpu = torch.cuda.is_available()

    model = LeNet5()
    model_path = os.path.join(
        MODEL_DIR, "model_epoch20_loss0.0023428495042026043.pth")
    model.load_state_dict(torch.load(model_path))

    print("Step 2: Converting data...")
    X = data_train[:, 1:].reshape(data_train.shape[0], 1, 28, 28)
    X = X.astype(float)
    X /= 255.0
    X = torch.from_numpy(X)
    y = data_train[:, 0]
    y = y.astype(int)
    y = torch.from_numpy(y)
    y = y.view(data_train.shape[0], -1)

    print("load train val split...")
    train_val_split = 0.8
    num_train = int(np.ceil(num_all*train_val_split))
    num_val = int(num_all - num_train)
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    # data_test = pd.read_csv("data/test.csv").values
    # X_test = data_test.reshape(data_test.shape[0], 1, 28, 28)
    # X_test = X_test.astype(float)
    # X_test /= 255.0
    # X_test = torch.from_numpy(X_test)
    # num_test = data_test.shape[0]
    # print(f"num_test:{num_test}")

    model.eval()
    feature = torch.Tensor()
    for i in range(num_val):
        # extract features for validation set
        X_batch = Variable(X_val[i:i+1].clone())
        X_batch = X_batch.type(torch.FloatTensor)
        if use_gpu:
            X_batch = X_batch.cuda()
        batch_out = model(X_batch)
        batch_feature = model.feature
        feature = torch.cat([feature, batch_feature])
        if (i+1) % 1000 == 0:
            print(f"Extracting: [{i+1}/{num_val}]")

    makedir_exist_ok(FEATURE_DIR)
    feature_path = os.path.join(FEATURE_DIR, "LeNet_epoch20_val.pth")
    torch.save(feature, feature_path)


def PCA_on_test():
    with open('./config.json', 'r') as f:
        CONFIG = json.load(f)
    DATA_ROOT = CONFIG["DATA_ROOT"]
    MODEL_DIR = CONFIG["MODEL_DIR"]
    FEATURE_DIR = CONFIG["FEATURE_DIR"]

    npzfile = np.load("train_val_idx.npz")
    train_idx = npzfile["train_idx"]
    val_idx = npzfile["val_idx"]

    os.chdir(DATA_ROOT)
    print(f"Change current work directory to: {DATA_ROOT}")

    print("Step 1: Loading data...")
    data_train = pd.read_csv("data/train.csv").values
    # 一般不应该对 data 进行 shuffle
    # data_train = shuffle(data_train)  # shuffle after loading train csv
    data_test = pd.read_csv("data/test.csv").values
    num_all = len(data_train)  # 42000
    num_test = len(data_test)  # 28000

    use_gpu = torch.cuda.is_available()

    model = LeNet5()
    model_path = os.path.join(
        MODEL_DIR, "model_epoch20_loss0.0023428495042026043.pth")
    model.load_state_dict(torch.load(model_path))

    print("Step 2: Converting data...")
    X = data_train[:, 1:].reshape(data_train.shape[0], 1, 28, 28)
    X = X.astype(float)
    X /= 255.0
    X = torch.from_numpy(X)
    y = data_train[:, 0]
    y = y.astype(int)
    y = torch.from_numpy(y)
    y = y.view(data_train.shape[0], -1)

    print("load train val split...")
    train_val_split = 0.8
    num_train = int(np.ceil(num_all*train_val_split))
    num_val = int(num_all - num_train)
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    feature_path = os.path.join(FEATURE_DIR, "LeNet_epoch20.pth")
    feature = torch.load(feature_path)

    pca = PCA(n_components=2)
    feature_pca = pca.fit_transform(feature)
    print(pca.explained_variance_ratio_)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = range(2)
    colors = ['r', 'g']
    ax.scatter(feature_pca[:, 0], feature_pca[:, 1])
    ax.legend(('bad', 'good'))
    ax.grid()
    plt.savefig('PCA.png')


def PCA_on_val():
    with open('./config.json', 'r') as f:
        CONFIG = json.load(f)
    DATA_ROOT = CONFIG["DATA_ROOT"]
    MODEL_DIR = CONFIG["MODEL_DIR"]
    FEATURE_DIR = CONFIG["FEATURE_DIR"]

    npzfile = np.load("train_val_idx.npz")
    train_idx = npzfile["train_idx"]
    val_idx = npzfile["val_idx"]

    os.chdir(DATA_ROOT)
    print(f"Change current work directory to: {DATA_ROOT}")

    print("Step 1: Loading data...")
    data_train = pd.read_csv("data/train.csv").values
    # 一般不应该对 data 进行 shuffle
    # data_train = shuffle(data_train)  # shuffle after loading train csv
    data_test = pd.read_csv("data/test.csv").values
    num_all = len(data_train)  # 42000
    num_test = len(data_test)  # 28000

    use_gpu = torch.cuda.is_available()

    model = LeNet5()
    model_path = os.path.join(
        MODEL_DIR, "model_epoch20_loss0.0023428495042026043.pth")
    model.load_state_dict(torch.load(model_path))

    print("Step 2: Converting data...")
    X = data_train[:, 1:].reshape(data_train.shape[0], 1, 28, 28)
    X = X.astype(float)
    X /= 255.0
    X = torch.from_numpy(X)
    y = data_train[:, 0]
    y = y.astype(int)
    y = torch.from_numpy(y)
    y = y.view(data_train.shape[0], -1)

    print("load train val split...")
    train_val_split = 0.8
    num_train = int(np.ceil(num_all*train_val_split))
    num_val = int(num_all - num_train)
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    feature_path = os.path.join(FEATURE_DIR, "LeNet_epoch20_val.pth")
    feature = torch.load(feature_path)

    pca = PCA(n_components=2)
    feature_pca = pca.fit_transform(feature)
    print("pca.explained_variance_ratio_: ", pca.explained_variance_ratio_)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)

    num_classes = 10
    targets = np.arange(num_classes)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r']
    color_dict = dict(zip(targets, colors))

    total_target = 0
    for target in targets:
        target_idx = [i for i, v in enumerate(y_val) if v == target]
        color = color_dict[target]
        print(f"target {target} components shape: ",
              feature_pca[target_idx].shape)
        ax.scatter(feature_pca[target_idx, 0],
                   feature_pca[target_idx, 1], c=color, s=50)
        total_target += len(target_idx)
    print("total_target: ", total_target)
    ax.legend([str(i) for i in range(num_classes)])
    ax.grid()
    plt.savefig('PCA_val.png')
    plt.show()


def main():
    # extract_feature()
    PCA_on_val()


if __name__ == "__main__":
    main()
