#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : May-18-20
# @Update  : Sep-09-20
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)
# @RefLink : https://github.com/KellyHwong/kaggle_digit_recognizer
# @RefLink : https://www.kaggle.com/c/digit-recognizer

import os
import datetime
import pandas as pd
import numpy as np

import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

from torchvision import datasets, transforms
from sklearn.utils import shuffle

from torch_fn.model import LeNet5
from torch_fn.data_loader import load_data
from utils.dir_utils import makedir_exist_ok


def main():
    # paths
    competition_name = "digit-recognizer"
    data_dir = os.path.expanduser(f"~/.kaggle/competitions/{competition_name}")
    ckpt_dir = os.path.expanduser(
        f"~/Documents/DeepLearningData/{competition_name}/ckpts")
    log_dir = os.path.expanduser(
        f"~/Documents/DeepLearningData/{competition_name}/logs")
    makedir_exist_ok(ckpt_dir)
    makedir_exist_ok(log_dir)
    # experiment time
    date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # load data
    train_images, train_labels, test_images = load_data(
        data_dir)  # train_images: N, H, W
    num_train = train_images.shape[0]
    num_test = test_images.shape[0]

    # TODO
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.ToTensor()
        ]),
    }

    # training parameters
    IF_FAST_RUN = True
    start_epoch = 0
    # epochs = input("input training epochs: ")  # user epochs
    epochs = 1 if IF_FAST_RUN else epochs  # fast run epochs
    batch_size = 32
    use_gpu = torch.cuda.is_available()

    # prepare model, LeNet-5
    model_type = "LeNet-5"
    model = LeNet5()
    print(f"Using model: {model_type}.")

    model.train()  # training
    model_ckpt_dir = os.path.join(
        ckpt_dir, model_type, date_time)  # weights save path
    makedir_exist_ok(model_ckpt_dir)

    # TODO not implemented
    # train_loader = torch.utils.data.DataLoader(dataset=data_train,
    #                                            batch_size=batch_size,
    #                                            shuffle=True, num_workers=4)

    # model and model properties
    model = LeNet5()
    steps_per_epoch = int(np.ceil(num_train/batch_size))
    use_gpu = torch.cuda.is_available()
    model = model.cuda() if use_gpu else model  # to GPU
    model.train()

    criterion = nn.CrossEntropyLoss(size_average=False)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, betas=(0.9, 0.99))

    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            _batch_size = batch_size  # tmp batch_size
            if (step+1) * batch_size >= num_train:
                _batch_size = num_train - step*batch_size  # take left samples
            batch_idx = np.arange(step*batch_size, step*batch_size+_batch_size)
            X_batch = train_images[batch_idx]  # train_images is N, H, W, C
            y_batch = train_labels[batch_idx]
            # to Tensor
            X_batch, y_batch = torch.from_numpy(
                X_batch).float(), torch.from_numpy(y_batch).long()
            X_batch = X_batch.permute(0, 3, 1, 2)  # permute it to N, C, H, W
            if use_gpu:  # to GPU
                X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            # args.log_interval
            log_interval = 100
            if step % log_interval == 0:
                trained_samples = step*batch_size + _batch_size
                print(
                    f"Training Epoch: [{epoch+1}/{epochs}]. Step: [{step+1}/{steps_per_epoch}]. Samples: [{trained_samples}/{num_train} ({trained_samples/num_train*100: .2f} % )]. Loss: {loss: .6f}"
                )

    latest_weights_path = os.path.join(
        model_ckpt_dir, f"{model_type}-latest-weights.pth")
    torch.save(model.state_dict(), latest_weights_path)
    print(f"Model saved to {latest_weights_path}.")


if __name__ == "__main__":
    main()
