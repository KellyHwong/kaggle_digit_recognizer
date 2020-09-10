#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Apr-16-20 08:24
# @Update  : Sep-10-20 18:18
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)
# @RefLink : https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py#L15


import os
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union
import warnings
from PIL import Image

import matplotlib.pyplot as plt


class Kaggle_Digit(datasets.VisionDataset):
    """`Kaggle Digit Recognizer <https://www.kaggle.com/c/digit-recognizer>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``train.csv``
            and  ``test.csv`` exist. Example: root=os.path.expanduser(f"~/.kaggle/competitions/digit-recognizer").
        train (bool, optional): If True, creates dataset from ``train.csv``,
            otherwise from ``test.csv``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    train_file = "train.pt"
    test_file = "test.pt"
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            # download: bool = False, # not implemented
    ) -> None:
        super(Kaggle_Digit, self).__init__(root, transform=transform,
                                           target_transform=target_transform)
        self.train = train  # training set or test set

        self.process()

        if self.train:
            data_file = self.train_file
            self.data, self.targets = torch.load(
                os.path.join(self.processed_folder, data_file))
        else:
            data_file = self.test_file
            self.data, self.targets = torch.load(
                os.path.join(self.processed_folder, data_file)), None

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.data[index], int(self.targets[index])

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img.numpy(), mode='L')

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target
        else:
            img = self.data[index]  # only image

            if self.transform is not None:
                img = self.transform(img)

            return img

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        # return os.path.join(self.root, self.__class__.__name__, 'raw')
        return os.path.join(self.root)

    @property
    def processed_folder(self) -> str:
        # return os.path.join(self.root, self.__class__.__name__, 'processed')
        return os.path.join(self.root, 'processed')

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    # not implemented
    def _check_exists(self) -> bool:
        pass

    def process(self) -> None:
        """Process csv files to Tensor and save them.
        """

        os.makedirs(self.processed_folder, exist_ok=True)

        # process and save as torch files
        print('Processing...')

        images, labels = read_csv_file(
            os.path.join(self.root, "train.csv"), train=True)
        images_test = read_csv_file(os.path.join(
            self.root, "test.csv"), train=False)
        training_set = (images, labels)
        test_set = images_test

        with open(os.path.join(self.processed_folder, self.train_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')


def read_csv_file(path: str, train: bool = True) -> torch.Tensor:
    data = pd.read_csv(path, dtype=np.uint8).values
    if train:
        images = data[:, 1:].reshape(len(data), 28, 28)  # N, H, W
        labels = data[:, 0]
        return torch.from_numpy(images), torch.from_numpy(labels)
    else:
        images = data.reshape(len(data), 28, 28)  # N, H, W
        return torch.from_numpy(images)


def Kaggle_Digit_test():
    # paths
    competition_name = "digit-recognizer"
    data_dir = os.path.expanduser(f"~/.kaggle/competitions/{competition_name}")

    kaggle_digit = Kaggle_Digit(data_dir, train=True)


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


def load_data_test():
    # paths
    competition_name = "digit-recognizer"
    data_dir = os.path.expanduser(f"~/.kaggle/competitions/{competition_name}")

    train_images, train_labels, test_images = load_data(data_dir)
    print(test_images.dtype)


def train_val_split(train_images, train_labels):
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
    Kaggle_Digit_test()


if __name__ == "__main__":
    main()
