#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Apr-17-20 02:55
# @Author  : Kelly Hwong (you@example.org)
# @Link    : https://github.com/KellyHwong/kaggle_digit_recognizer
# @Link    : https://www.kaggle.com/c/digit-recognizer

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
from sklearn.utils import shuffle

from model import LeNet5
from utils import makedir_exist_ok


def main():
    print("Load config...")
    with open('./config.json', 'r') as f:
        CONFIG = json.load(f)
    DATA_ROOT = CONFIG["DATA_ROOT"]
    BATCH_SIZE = CONFIG["BATCH_SIZE"]
    MODEL_DIR = CONFIG["MODEL_DIR"]
    makedir_exist_ok(MODEL_DIR)

    npzfile = np.load("train_val_idx.npz")
    train_idx = npzfile["train_idx"]
    val_idx = npzfile["val_idx"]

    os.chdir(DATA_ROOT)
    print(f"Change current work directory to: {DATA_ROOT}")

    print("Step 1: Loading data...")
    data_train = pd.read_csv("data/train.csv").values
    data_train = shuffle(data_train)  # shuffle after loading train csv
    data_test = pd.read_csv("data/test.csv").values
    num_all = len(data_train)  # 42000
    num_test = len(data_test)  # 28000

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
    y_val = X[val_idx]

    train_loader = torch.utils.data.DataLoader(dataset=data_train,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset=data_test,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=4)

    print("Step 3: Training phase...")
    num_epoch = 20  # 50000
    batch_size = BATCH_SIZE
    steps_per_epoch = int(np.ceil(num_train//batch_size))

    use_gpu = torch.cuda.is_available()

    model = LeNet5()
    model.train()

    criterion = nn.CrossEntropyLoss(size_average=False)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, betas=(0.9, 0.99))

    for epoch in range(num_epoch):
        _batch_size = batch_size
        for step in range(steps_per_epoch):
            if (step+1) * batch_size >= num_train:
                _batch_size = num_train - step*batch_size
            batch_idx = np.arange(step*batch_size, step*batch_size+_batch_size)
            X_batch = Variable(X_train[batch_idx].clone())
            y_batch = Variable(y_train[batch_idx].clone(), requires_grad=False)

            X_batch = X_batch.type(torch.FloatTensor)
            y_batch = y_batch.type(torch.LongTensor)
            y_batch = y_batch.view(batch_size)
            if use_gpu:
                X_batch, y_batch = X_batch.cuda(), y_batch.cuda()

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print(
                    f"Train epoch: {epoch+1}, [{step*BATCH_SIZE}/{num_train} ({step*BATCH_SIZE/num_train*100:.2f}%)].\tLoss: {loss:.6f}")

        if (epoch + 1) % 10 == 0:
            model_path = os.path.join(
                MODEL_DIR, f"model_epoch{epoch+1}_loss{loss:.6f}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model of epoch {epoch+1} saved to {model_path}.")

    print("Step 4: Testing phase...")
    X_test = data_test.reshape(data_test.shape[0], 1, 28, 28)
    X_test = X_test.astype(float)
    X_test /= 255.0
    X_test = torch.from_numpy(X_test)
    print(f"num_test:{num_test}")

    model.eval()
    final_prediction = np.ndarray(shape=(num_test, 2), dtype=int)
    for i in range(num_test):
        X_batch = Variable(X_test[i:i+1].clone())
        X_batch = X_batch.type(torch.FloatTensor)
        if use_gpu:
            X_batch = X_batch.cuda()
        batch_out = model(X_batch)
        batch_feature = model.feature
        _, pred = torch.max(batch_out, 1)
        final_prediction[i][0] = 1 + i
        final_prediction[i][1] = pred.data[0]
        if (i+1) % 2000 == 0:
            print(f"Testing: [{i+1}/{num_test}]")

    print("Step 5: Generating submission file...")
    submission = pd.DataFrame(final_prediction, dtype=int,
                              columns=['ImageId', 'Label'])
    submission.to_csv("pytorch_LeNet.csv", index=False, header=True)

    print("Finished!")


if __name__ == "__main__":
    main()
