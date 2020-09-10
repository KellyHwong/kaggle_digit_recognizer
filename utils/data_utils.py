#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Sep-08-20 16:52
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)
# @Link    : http://example.org

import numpy as np
from tqdm import tqdm


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


def main():
    # test
    images = np.random.normal(size=(50000, 28, 28))
    padded_images = pad_images(images, padding=2)
    print(padded_images.shape)


if __name__ == "__main__":
    main()
