#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

# import dataset
data = pd.read_csv("./data/train.csv")
data['Sex'] = data['Sex'].apply(lambda s: 1 if s == 'male' else 0)
data = data.fillna(0)
dataset_X = data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]
dataset_X = dataset_X.as_matrix()

# make softmax output
data['Deceased'] = data['Survived'].apply(lambda s: int(not s))
dataset_Y = data[['Deceased', 'Survived']]
dataset_Y = dataset_Y.as_matrix()

# split the dataset to train data and test data
X_train, X_val, y_train, y_val = train_test_split(dataset_X, dataset_Y, test_size=0.2, random_state=42)

# give the X, y, W and b
X = tf.placeholder(tf.float32, shape=[None, 6])
y = tf.placeholder(tf.float32, shape=[None, 2])

W = tf.Variable(tf.random_normal([6, 2]), name='weights')
bias = tf.Variable(tf.zeros([2]), name='bias')

# make the network
y_pred = tf.nn.softmax(tf.matmul(input, W) + bias)

# compute the cost
cross_entropy = - tf.reduce_sum(y * tf.log(y_pred + 1e-10), reduction_indices = 1)
cost = tf.reduce_mean(cross_entropy)

# add the optimization
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

with tf.Session() as sess:
    # initialize the variables
    tf.global_variables_initializer().run()

    # start thr train
    for epoch in range(10):
        total_loss = 0
        for i in range(len(X_train)):
            feed = {X: [X_train[i]], y: [y_train[i]]}
            _, loss = sess.run([train_op, cost], feed_dict=feed)
            total_loss += loss
        print('Epoch: %04d, total loss=%.9f' % (epoch + 1), total_loss)
    print('Training complete!')

    # # calculate the accuracy use tensorflow
    # accuracy = sess.run(acc_op, feed_dict={X: X_val, y: y_val})
    # print("Accuracy on validation set: %.9f" % accuracy)

    # Accuracy calculated by NumPy
    pred = sess.run(y_pred, feed_dict={X: X_val})
    correct = np.equal(np.argmax(pred, 1), np.argmax(y_val, 1))
    numpy_accuracy = np.mean(correct.astype(np.float32))
    print("Accuracy on validation set (numpy): %.9f" % numpy_accuracy)