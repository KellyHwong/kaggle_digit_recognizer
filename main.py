#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Apr-17-20 02:55
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

from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from model import LeNet5
from utils import makedir_exist_ok


def main():
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

    train_loader = torch.utils.data.DataLoader(dataset=data_train,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset=data_test,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=4)

    print("Step 3: Training phase...")
    num_epoch = 20  # 50000
    num_train = len(data_train)
    batch_size = BATCH_SIZE
    steps_per_epoch = int(np.ceil(len(data_train)//batch_size))

    use_gpu = torch.cuda.is_available()

    model = LeNet5()
    model.train()

    criterion = nn.CrossEntropyLoss(size_average=False)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, betas=(0.9, 0.99))

    for epoch in range(num_epoch):
        num_index = 0
        for step in range(steps_per_epoch):
            if num_index + batch_size >= num_train:
                batch_size = num_train - num_index
                X_batch = Variable(X[num_index:(num_index+batch_size)].clone())
                y_batch = Variable(
                    y[num_index:(num_index+batch_size)].clone(), requires_grad=False)
                num_index = 0
            else:
                X_batch = Variable(X[num_index:(num_index+batch_size)].clone())
                y_batch = Variable(
                    y[num_index:(num_index+batch_size)].clone(), requires_grad=False)
                num_index = num_index + batch_size

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
            if num_index == 0:
                break

        if (epoch + 1) % 10 == 0:
            model_path = os.path.join(
                MODEL_DIR, f"model_epoch{epoch+1}_loss{loss}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model of epoch {epoch+1} saved to {model_path}.")

    print("Step 4: Testing phase...")
    y_test = data_test.reshape(data_test.shape[0], 1, 28, 28)
    y_test = y_test.astype(float)
    y_test /= 255.0
    y_test = torch.from_numpy(y_test)
    num_test = data_test.shape[0]
    print(f"num_test:{num_test}")
    model.eval()

    final_prediction = np.ndarray(shape=(num_test, 2), dtype=int)
    for i in range(num_test):
        sample_data = Variable(y_test[i:i+1].clone())
        sample_data = sample_data.type(torch.FloatTensor)
        if use_gpu:
            sample_data = sample_data.cuda()
        sample_out = model(sample_data)
        _, pred = torch.max(sample_out, 1)
        final_prediction[i][0] = 1 + i
        final_prediction[i][1] = pred.data[0]
        if (i + 1) % 2000 == 0:
            print(f"Already tested: {i+1}")

    print("Step 5: Generating submission file...")
    submission = pd.DataFrame(final_prediction, dtype=int,
                              columns=['ImageId', 'Label'])
    submission.to_csv("pytorch_LeNet.csv", index=False, header=True)

    print("Finished!")


if __name__ == "__main__":
    main()
