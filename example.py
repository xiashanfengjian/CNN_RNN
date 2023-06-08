import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

(train_im, train_lable),(test_im, test_lable) = tf.keras.datasets.mnist.load_data()
train_im = train_im/255
test_im = test_im/255

model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
                             tf.keras.layers.Dense(32, activation='softmax'),
                             tf.keras.layers.Dense(32, activation='softmax'),
                             tf.keras.layers.Dense(32, activation='softmax'),
                             tf.keras.layers.Dense(32, activation='softmax'),
                             tf.keras.layers.Dense(10,activation='relu')])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy',metrics=['acc'])
model.fit(train_im,train_lable,epochs=10)
hist = model.fit(train_im,train_lable,epochs=100)
plt.plot(hist.epoch,hist.history.get('loss'))
