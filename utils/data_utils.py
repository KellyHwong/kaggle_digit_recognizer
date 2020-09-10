#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Apr-18-20 19:46
# @Update  : Sep-08-20 16:52
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)
# @Link    : http://example.org

import os
import json
import numpy as np
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from utils.dir_utils import makedir_exist_ok


def pad_images(images, padding=2):
    """padding images
    Inputs:
        images: numpy array
    Return:
        padded_images: images after padded
    """
    if len(images.shape) == 3:  # N, H, W, no channel, pad single channel
        padded_images = np.zeros(
            (len(images),
             images.shape[1]+2*padding,
             images.shape[2]+2*padding)
        )
        for i in tqdm(range(len(images))):
            img = images[i]

            h_ones = np.zeros((padding, img.shape[1]))
            img = np.vstack([h_ones, img, h_ones])

            v_ones = np.zeros((img.shape[0], padding))
            img = np.hstack([v_ones, img, v_ones])

            padded_images[i] = img

        return padded_images
    else:
        raise NotImplementedError


def pad_images_test():
    # test
    images = np.random.normal(size=(50000, 28, 28))
    padded_images = pad_images(images, padding=2)
    print(padded_images.shape)


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


def train_val_mask_test():
    train_idx, val_idx = train_val_mask()
    np.savez("train_val_idx.npz", train_idx=train_idx, val_idx=val_idx)


def main():
    train_val_mask_test()
    pad_images_test()


if __name__ == "__main__":
    main()
