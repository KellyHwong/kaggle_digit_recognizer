#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Apr-18-20 19:46
# @Author  : Kelly Hwong (you@example.org)
# @Link    : http://example.org

import os
import json
import pandas as pd
import numpy as np

import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from utils import makedir_exist_ok


def visualize_data():
    """ Load Config """
    with open('./config.json', 'r') as f:
        CONFIG = json.load(f)
    DATA_ROOT = CONFIG["DATA_ROOT"]
    DATA_ROOT_MNIST = CONFIG["DATA_ROOT_MNIST"]
    BATCH_SIZE = CONFIG["BATCH_SIZE"]
    MODEL_DIR = CONFIG["MODEL_DIR"]
    makedir_exist_ok(MODEL_DIR)

    os.chdir(DATA_ROOT)
    print(f"Change current work directory to: {DATA_ROOT}")

    print("Step 1: Loading data...")
    data_train = pd.read_csv("data/train.csv").values
    data_train = shuffle(data_train)  # shuffle after loading train csv
    data_test = pd.read_csv("data/test.csv").values
    num_train = data_train.shape[0]

    print("Step 2: Converting data...")

    X = data_train[:, 1:].reshape(data_train.shape[0], 1, 28, 28)
    X = X.astype(float)
    X /= 255.0
    X = torch.from_numpy(X)

    y = data_train[:, 0]
    y = y.astype(int)
    y = torch.from_numpy(y)
    y = y.view(data_train.shape[0], -1)

    for i in range(100):
        img = X[i]
        label = y[i]
        print(img)  # PIL.Image.Image
        # print(img.mode)  # mode L
        print(label)
        img = np.asarray(img)
        print(img.shape)
        img = img.reshape(28, 28)
        plt.imshow(img, cmap="gray")
        plt.show()


def train_val_mask(num_all=42000, train_val_split=0.8, seed=42):
    """
    Inputs:
        seed: default 42: Answer to the Ultimate Question of Life, the Universe, and Everything
    """
    np.random.seed(seed)
    num_train = int(np.ceil(num_all*train_val_split))
    num_val = int(num_all-num_train)
    print(
        f"split indices to num_train and num_val: [{num_train}/{num_all}] and [{num_val}/{num_all}]")
    all_idx = np.arange(num_all)
    # all_idx = np.random.permutation(all_idx)  # shuffle with permutation
    all_idx = np.random.shuffle(all_idx)  # shuffle with shuffle
    train_idx = all_idx[0:num_train]
    val_idx = all_idx[num_train:]

    return train_idx, val_idx


def main():
    train_idx, val_idx = train_val_mask()
    np.savez("train_val_idx.npz", train_idx=train_idx, val_idx=val_idx)


if __name__ == "__main__":
    main()
