import tensorflow as tf
import tensorflow.keras as keras

import numpy as np


def get_max_index(arr):
    index = 0
    for i in range(len(arr)):
        if arr[i] > arr[index]:
            index = i
    return index


# tf.device('/device:GPU:0')
config = tf.ConfigProto(log_device_placement=True, device_count={'GPU': 1})
sess = tf.Session(config=config)
keras.backend.set_session(sess)

train_binary_numbers = np.array([
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 0],
    [1, 0, 1, 0],
    [0, 1, 1, 0],
    [1, 1, 1, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 0, 1],
    [1, 1, 0, 1],
    [0, 0, 1, 1],
    [1, 0, 1, 1],
    [0, 1, 1, 1],
    [1, 1, 1, 1],
])
train_binary_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(4,)),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(16, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_binary_numbers, train_binary_labels, epochs=1)

predictions_test = np.array([
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 0],
    [1, 0, 1, 0],
    [0, 1, 1, 0],
    [1, 1, 1, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 0, 1],
    [1, 1, 0, 1],
    [0, 0, 1, 1],
    [1, 0, 1, 1],
    [0, 1, 1, 1],
    [1, 1, 1, 1],
])

predictions = model.predict(predictions_test)

for i in range(len(predictions_test)):
    print('{0} -> {1}'.format(predictions_test[i], get_max_index(predictions[i])))
