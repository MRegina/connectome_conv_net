"""
Created on Nov 07 2019
Code for connectome based regression with convolutional networks (CCNN)
@author: mregina
"""

import numpy as np
import tensorflow as tf
from datetime import datetime
import os

# input data paths
TRAIN_DATA_PATH = '/data/train_data.npz'
VALID_DATA_PATH = '/data/valid_data.npz'
TEST_DATA_PATH = '/data/test_data.npz'

# otput paths
LOGDIR = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(os.path.join(LOGDIR, "checkpoints"), exist_ok=False)
CHECKPOINT_PATH = os.path.join(LOGDIR, "checkpoints", "cp-{epoch:04d}.ckpt")

BATCH_SIZE = 32
INPUT_SHAPE = (100, 100, 1)
OUTPUT_SIZE = 100

# architecture parameters
filter_num1 = 256
filter_num2 = 256

# hyperparameters
LEARNING_RATE = 0.001
EPOCH = 100


def load_dataset(path):
    with np.load(path) as dataset:
        data = dataset['data']
        label = dataset['label']
    return data, label


# load data
train_data, train_label = load_dataset(TRAIN_DATA_PATH)
valid_data, valid_label = load_dataset(VALID_DATA_PATH)
test_data, test_label = load_dataset(TEST_DATA_PATH)



# create tensorflow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
valid_dataset = tf.data.Dataset.from_tensor_slices((valid_data, valid_label))
test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_label))

# batch (and shuffle) datasets
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)
valid_dataset = valid_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# create model
input = tf.keras.Input(shape=INPUT_SHAPE, name='input')
conv1 = tf.keras.layers.Conv2D(filters=filter_num1, kernel_size=(INPUT_SHAPE[0], 1), strides=(1, 1),
                               padding='valid',
                               kernel_initializer='he_uniform', activation='relu', name='conv1')(input)
conv2 = tf.keras.layers.Conv2D(filters=filter_num2, kernel_size=(1, INPUT_SHAPE[1]), strides=(1, 1),
                               padding='valid',
                               kernel_initializer='he_uniform', activation='relu', name='conv2')(conv1)
output = tf.keras.layers.Dense(units=OUTPUT_SIZE, name='output')(conv2)

model = tf.keras.Model(inputs=[input], outputs=[output])

# create callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOGDIR)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH, save_weights_only=False, period=5)

# compile model for training
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              # Loss function to minimize
              loss=tf.keras.losses.MeanSquaredError(),
              # List of metrics to monitor
              metrics=[tf.keras.losses.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])

# training loop
model.fit(train_dataset, epochs=EPOCH, validation_data=valid_dataset, callbacks=[tensorboard_callback, cp_callback])

# evaluate on the test set
test_loss = model.evaluate(test_dataset)
print(test_loss)
