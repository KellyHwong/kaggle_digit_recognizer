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

    os.chdir(DATA_ROOT)
    print(f"Change current work directory to: {DATA_ROOT}")

    use_gpu = torch.cuda.is_available()

    data_test = pd.read_csv("data/test.csv").values

    model = LeNet5()
    model_path = os.path.join(
        MODEL_DIR, "model_epoch20_loss0.000484.pth")
    model.load_state_dict(torch.load(model_path))

    y_test = data_test.reshape(data_test.shape[0], 1, 28, 28)
    y_test = y_test.astype(float)
    y_test /= 255.0
    y_test = torch.from_numpy(y_test)
    num_test = data_test.shape[0]
    print(f"num_test:{num_test}")

    model.eval()
    feature = torch.Tensor()
    final_prediction = np.ndarray(shape=(num_test, 2), dtype=int)
    for i in range(num_test):
        X_batch = Variable(y_test[i:i+1].clone())
        X_batch = X_batch.type(torch.FloatTensor)
        if use_gpu:
            X_batch = X_batch.cuda()
        batch_out = model(X_batch)
        batch_feature = model.feature
        feature = torch.cat([feature, batch_feature])
        _, pred = torch.max(batch_out, 1)
        final_prediction[i][0] = 1 + i
        final_prediction[i][1] = pred.data[0]
        if (i + 1) % 2000 == 0:
            print(f"Testing: [{i+1}/{num_test}]")

    makedir_exist_ok(FEATURE_DIR)
    feature_path = os.path.join(FEATURE_DIR, "LeNet_epoch20.pth")
    torch.save(feature, feature_path)


def PCA_on_test():
    with open('./config.json', 'r') as f:
        CONFIG = json.load(f)
    DATA_ROOT = CONFIG["DATA_ROOT"]
    MODEL_DIR = CONFIG["MODEL_DIR"]
    FEATURE_DIR = CONFIG["FEATURE_DIR"]

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
    pass


def main():
    # extract_feature()
    PCA_on_test()


if __name__ == "__main__":
    main()
