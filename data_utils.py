#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Apr-18-20 19:46
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os


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


if __name__ == "__main__":
    main()
