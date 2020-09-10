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
