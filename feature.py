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
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

from model import LeNet5
from utils import makedir_exist_ok


def extract_feature():
    with open("./config.json", "r") as f:
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
    num_val = int(num_all-num_train)
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
    with open("./config.json", "r") as f:
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
    ax.set_xlabel("Principal Component 1", fontsize=15)
    ax.set_ylabel("Principal Component 2", fontsize=15)
    ax.set_title("2 component PCA", fontsize=20)
    targets = range(2)
    colors = ["r", "g"]
    ax.scatter(feature_pca[:, 0], feature_pca[:, 1])
    ax.legend(("bad", "good"))
    ax.grid()
    plt.savefig("PCA.png")


def PCA_on_val():
    with open("./config.json", "r") as f:
        CONFIG = json.load(f)
    DATA_ROOT = CONFIG["DATA_ROOT"]
    MODEL_DIR = CONFIG["MODEL_DIR"]
    FEATURE_DIR = CONFIG["FEATURE_DIR"]

    npzfile = np.load("train_val_idx.npz")
    train_idx = npzfile["train_idx"]
    val_idx = npzfile["val_idx"]

    print("Step 1: Loading data...")
    data_train = pd.read_csv(os.path.join(DATA_ROOT, "data/train.csv")).values
    # 一般不应该对 data 进行 shuffle
    # data_train = shuffle(data_train)  # shuffle after loading train csv
    data_test = pd.read_csv(os.path.join(DATA_ROOT, "data/test.csv")).values
    num_all = len(data_train)  # 42000
    num_test = len(data_test)  # 28000
    num_classes = 10

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
    ax.set_xlabel("Principal Component 1", fontsize=15)
    ax.set_ylabel("Principal Component 2", fontsize=15)
    ax.set_title("2 component PCA", fontsize=20)

    targets = np.arange(num_classes)

    total_target = 0
    for target in targets:
        target_idx = [i for i, v in enumerate(y_val) if v == target]
        color = color_dict[target]
        print(f"target {target} components shape: ",
              feature_pca[target_idx].shape)
        ax.scatter(feature_pca[target_idx, 0],
                   feature_pca[target_idx, 1], cmap="magma", s=10)  # c=color
        total_target += len(target_idx)
    print("total_target: ", total_target)
    ax.legend([str(i) for i in range(num_classes)])
    ax.grid()
    plt.savefig("PCA_val.png")
    plt.show()


def normalize(X, feature_range=(0, 1)):
    """MinMaxScaler normalize
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    normalized_data = scaler.fit_transform(X)
    return normalized_data


def isolation_forests():
    """Isolation_Forests
    Dataset data should be of dim 2
    """
    with open("./config.json", "r") as f:
        CONFIG = json.load(f)
    DATA_ROOT = CONFIG["DATA_ROOT"]
    MODEL_DIR = CONFIG["MODEL_DIR"]
    FEATURE_DIR = CONFIG["FEATURE_DIR"]

    npzfile = np.load("train_val_idx.npz")
    train_idx = npzfile["train_idx"]
    val_idx = npzfile["val_idx"]

    print("Step 1: Loading data...")
    data_train = pd.read_csv(os.path.join(DATA_ROOT, "data/train.csv")).values
    # 一般不应该对 data 进行 shuffle
    # data_train = shuffle(data_train)  # shuffle after loading train csv
    data_test = pd.read_csv(os.path.join(DATA_ROOT, "data/test.csv")).values
    num_all = len(data_train)  # 42000
    num_test = len(data_test)  # 28000

    # print("Step 2: Converting data...")
    # X = data_train[:, 1:].reshape(data_train.shape[0], 1, 28, 28)
    # X = X.astype(float)
    # X /= 255.0
    # X = torch.from_numpy(X)
    y = data_train[:, 0]
    y = y.astype(int)
    y = torch.from_numpy(y)
    y = y.view(data_train.shape[0], -1)

    print("load train val split...")
    train_val_split = 0.8
    num_train = int(np.ceil(num_all*train_val_split))
    num_val = int(num_all - num_train)
    y_val = y[val_idx]
    num_classes = 10

    feature_path = os.path.join(FEATURE_DIR, "LeNet_epoch20_val.pth")
    feature = torch.load(feature_path)
    label = y_val

    pca = PCA(n_components=2)
    feature_pca = pca.fit_transform(feature)
    print("pca.explained_variance_ratio_: ", pca.explained_variance_ratio_)

    targets = np.arange(num_classes)
    target_label = targets[0]
    target_idx = [i for i, v in enumerate(y_val) if v == target_label]

    feature_range = (-1, 1)  # set normalize limited

    random_state = np.random.RandomState(169)
    # n_targets = feature_pca.shape[0]
    n_targets = len(target_idx)
    print(f"n_targets: {n_targets}")

    feature_pca_norm = normalize(feature_pca, feature_range)
    x_data = feature_pca_norm[target_idx, 0]
    y_data = feature_pca_norm[target_idx, 1]
    ifm = IsolationForest(n_estimators=100, verbose=2, n_jobs=2,
                          max_samples=n_targets, random_state=random_state, max_features=2)

    Iso_train_dt = np.column_stack((x_data, y_data))
    ifm.fit(Iso_train_dt)
    scores_pred = ifm.decision_function(Iso_train_dt)

    outlier_fraction = 0.05  # 使用预测值取5%分位数来定义阈值（基于小概率事件5%）
    threshold = stats.scoreatpercentile(scores_pred, 100*outlier_fraction)

    # 根据训练样本中异常样本比例，得到阈值，用于绘图
    # matplotlib
    # plot the line, the samples, and the nearest vectors to the plane
    xx, yy = np.meshgrid(np.linspace(feature_range[0], feature_range[1], 50),
                         np.linspace(feature_range[0], feature_range[1], 50))  # 画格子
    Z = ifm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
    plt.title("Isolation Forest")
    outlier_proportion = int(outlier_fraction*n_targets)
    # 绘制异常点区域，从阈值附近到最外围
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold,
                                               outlier_proportion), cmap=plt.cm.hot)
    # 绘制异常点区域和正常点区域的边界
    ax_a = plt.contour(xx, yy, Z, levels=[
                       threshold], linewidths=2, colors="red")
    plt.contourf(xx, yy, Z, levels=[threshold,
                                    Z.max()], colors="palevioletred")
    # palevioletred 紫罗兰
    # 绘制正常点区域，值从阈值到最大的那部分

    anomal_idx, is_anomal = [], []
    for i in scores_pred:
        if i <= threshold:
            # print(i)
            is_anomal.append(1)
            anomal_idx.append(i)
        else:
            is_anomal.append(0)

    # scatter anomalies by class labels
    anomal_data = np.column_stack([x_data, y_data, scores_pred, is_anomal])
    anomal_df = pd.DataFrame(data=anomal_data, columns=[
                             "x_data", "y_data", "isolation_score", "is_anomal"])

    ax_b = plt.scatter(anomal_df["x_data"][anomal_df["is_anomal"] == 0], anomal_df["y_data"]
                       [anomal_df["is_anomal"] == 0], s=20, edgecolor="k", c="white")
    ax_c = plt.scatter(anomal_df["x_data"][anomal_df["is_anomal"] == 1], anomal_df["y_data"]
                       [anomal_df["is_anomal"] == 1], s=20, edgecolor="k", c="black")
    anomal_df.to_csv("anamal_isolation_forests.csv")

    plt.axis("tight")
    plt.xlim(feature_range)
    plt.ylim(feature_range)
    plt.legend([ax_a.collections[0], ax_b, ax_c],
               ["learned decision function", "true inliers", "true outliers"],
               loc="upper left")
    print(f"Isolation forests threshold: {threshold}")
    print(f"Total samples: {len(feature_pca)}")
    print(f"Num of anomal data detected: {len(anomal_idx)}")
    plt.show()


def main():
    # extract_feature()
    PCA_on_val()
    # isolation_forests()


if __name__ == "__main__":
    main()
