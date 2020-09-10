#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-03-20 23:44
# @Update  : Sep-08-20 17:36
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)
# @Link    : http://example.org

import os
import datetime
import json
import random
import pickle
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras  # keras-tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy, CategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

from keras_fn.resnet import model_depth, resnet_v2, lr_schedule
from utils.dir_utils import makedir_exist_ok
from utils.data_utils import pad_images

print(f"If in eager mode: {tf.executing_eagerly()}.")
print(f"Use tensorflow version {tf.__version__}.")
assert tf.__version__[0] == "2"


def main():
    # task mode
    mode = "train"  # default mode, "train" or "eval"
    if mode not in ["train", "eval"]:
        raise Exception("Mode error!")

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
    train_labels = keras.utils.to_categorical(train_labels)

    test_images = data_test.reshape(
        num_test, 28, 28, 1)  # NHWC, channel last
    test_images = test_images.astype(float)
    test_images /= 255.0  # map to [0.0, 1.0]

    # training parameters
    start_epoch = 0
    IF_FAST_RUN = False
    epochs = 200  # default epochs
    batch_size = 32

    # prepare model, ResNetv2
    n, version = 2, 2  # n, order of ResNetv2, 2 or 6
    depth = model_depth(n, version)
    model_type = 'ResNet%dv%d' % (depth, version)
    metrics = [
        BinaryAccuracy(),
        CategoricalAccuracy()
    ]

    # data feed pipeline
    padded = False  # default not to pad images
    print(f"Using model: {model_type}.")
    padded = True if model_type == "ResNet20v2" else padded

    if mode == "train":
        input_images = train_images
    elif mode == "eval":
        input_images = test_images

    if padded:
        input_images = input_images[:, :, :, 0]
        # pad (28, 28) to (32, 32)
        input_images = pad_images(input_images, padding=2)
        input_images = np.expand_dims(input_images, -1)

    input_shape = input_images.shape[1:]
    model = resnet_v2(input_shape=input_shape,
                      depth=depth, num_classes=num_classes)
    model.compile(loss='categorical_crossentropy',  # CCE loss
                  optimizer=Adam(learning_rate=lr_schedule(0)),
                  metrics=metrics)

    # define callbacks for model weights saving
    # ckpt_name = "%s-epoch-{epoch:%03d}-categorical_accuracy-{categorical_accuracy:.4f}.h5" % (
    #     model_type) # 不能用百分号！
    model_ckpt_dir = os.path.join(ckpt_dir, model_type, date_time)
    makedir_exist_ok(model_ckpt_dir)
    ckpt_name = "weights-epoch-{epoch:03d}-loss-{loss:.4f}-val_loss-{val_loss:.4f}.hdf5"
    ckpt_name = "weights-epoch-{epoch:03d}-binary_accuracy-{binary_accuracy:.4f}-categorical_accuracy-{categorical_accuracy:.4f}.hdf5"
    filepath = os.path.join(model_ckpt_dir, ckpt_name)
    checkpoint = ModelCheckpoint(
        filepath=filepath, monitor="categorical_accuracy", verbose=1)

    # define callbacks for logs and learning rate adjustment.
    model_log_dir = os.path.join(log_dir, model_type, date_time)
    makedir_exist_ok(model_log_dir)
    file_writer = tf.summary.create_file_writer(
        model_log_dir + "/metrics")  # custom scalars
    file_writer.set_as_default()

    csv_logger = CSVLogger(os.path.join(
        model_log_dir, "training.log.csv"), append=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        model_log_dir, histogram_freq=1)
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
    callbacks = [csv_logger, tensorboard_callback,
                 lr_scheduler, checkpoint]  # checkpoint

    latest_weights_path = os.path.join(
        model_ckpt_dir, f"./{model_type}-latest-weights.hdf5")

    if mode == "train":
        # train val split
        mask = np.arange(num_train)
        np.random.seed(42)
        np.random.shuffle(mask)

        val_split = int(0.2*num_train)

        X_train = input_images[mask[val_split:]]
        y_train = train_labels[mask[val_split:]]
        X_val = input_images[mask[:val_split]]
        y_val = train_labels[mask[:val_split]]

        # fit model
        print("Start training...")
        epochs = 3 if IF_FAST_RUN else epochs
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks
        )

        print("Save model's final weights...")
        model.save_weights(latest_weights_path)

        print("Save training history...")
        with open('./history.pickle', 'wb') as pickle_file:
            pickle.dump(history.history, pickle_file)

    elif mode == "eval":
        print("Start evaluating...")
        model.load_weights(latest_weights_path)
        predict = model.predict(
            input_images, batch_size=batch_size, workers=4, verbose=1)
        np.save(os.path.join(data_dir, f"./{model_type}-predict.npy"), predict)

        image_id = np.arange(num_test) + 1
        label = np.argmax(predict, axis=1)
        submission_df = pd.DataFrame({
            "ImageId": image_id,
            "Label": label
        })
        submission_df.to_csv(os.path.join(
            data_dir, f"{model_type}-submission.csv"), index=False)


if __name__ == "__main__":
    main()
