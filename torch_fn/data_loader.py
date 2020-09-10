#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Apr-16-20 08:24
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


def load_data(data_dir):
    """load kaggle digit recognizer dataset
    Returns:
        train_images: 
        train_labels: 1 dimensional np.array, class index of each sample
        test_images: 
    """
    # load data
    data_train = pd.read_csv(os.path.join(data_dir, "train.csv")).values
    data_test = pd.read_csv(os.path.join(data_dir, "test.csv")).values
    num_train = data_train.shape[0]
    num_test = data_test.shape[0]

    # convert data
    train_images = data_train[:, 1:].reshape(
        num_train, 28, 28, 1)  # NHWC, channel last
    train_images = train_images.astype(float)
    train_images /= 255.0  # map to [0.0, 1.0]
    train_labels = data_train[:, 0].astype(int)
    num_classes = np.max(train_labels) + 1  # 10

    test_images = data_test.reshape(
        num_test, 28, 28, 1)  # NHWC, channel last
    test_images = test_images.astype(float)
    test_images /= 255.0  # map to [0.0, 1.0]

    return train_images, train_labels, test_images


def funcname(train_images, train_labels):
    num_train = train_images.shape[0]
    # train val split
    mask = np.arange(num_train)
    np.random.seed(42)
    np.random.shuffle(mask)

    val_split = int(0.2*num_train)

    X_train = train_images[mask[val_split:]]
    y_train = train_labels[mask[val_split:]]
    X_val = train_images[mask[:val_split]]
    y_val = train_labels[mask[:val_split]]


def main():
    # paths
    competition_name = "digit-recognizer"
    data_dir = os.path.expanduser(f"~/.kaggle/competitions/{competition_name}")
    train_images, train_labels, test_images = load_data(data_dir)
    print(test_images.dtype)


if __name__ == "__main__":
    main()
