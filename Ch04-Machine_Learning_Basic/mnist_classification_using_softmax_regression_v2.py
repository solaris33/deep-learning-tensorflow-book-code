# -*- coding: utf-8 -*-

import tensorflow as tf

# MNIST 데이터를 다운로드 합니다.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# 이미지들을 float32 데이터 타입으로 변경합니다.
x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
# 28*28 형태의 이미지를 784차원으로 flattening 합니다.
x_train, x_test = x_train.reshape([-1, 784]), x_test.reshape([-1, 784])
# [0, 255] 사이의 값을 [0, 1]사이의 값으로 Normalize합니다.
x_train, x_test = x_train / 255., x_test / 255.

# tf.data API를 이용해서 데이터를 섞고 batch 형태로 가져옵니다.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(60000).batch(100)
train_data_iter = iter(train_data)

# Softmax Regression 모델을 위한 tf.Variable들을 정의합니다.
W = tf.Variable(tf.zeros(shape=[784, 10]))
b = tf.Variable(tf.zeros(shape=[10]))

# Softmax Regression 모델을 정의합니다.
@tf.function
def create_softmax_regression(x):
  logits = tf.matmul(x, W) + b
  return tf.nn.softmax(logits)

# cross-entropy 손실 함수를 정의합니다.
@tf.function
def create_cross_entropy_loss(y_pred, y):
  return tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(y_pred), axis=[1]))
  #return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logtis, labels=y)) # tf.nn.softmax_cross_entropy_with_logits API를 이용한 구현

# 모델의 정확도를 출력하는 함수를 정의합니다.
@tf.function
def compute_accuracy(y_pred, y):
  correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  return accuracy

# 최적화를 위한 그라디언트 디센트 옵티마이저를 정의합니다.
optimizer = tf.optimizers.SGD(0.5)

# 최적화를 위한 function을 정의합니다.
@tf.function
def train_step(x, y):
  with tf.GradientTape() as tape:
    y_pred = create_softmax_regression(x)
    loss = create_cross_entropy_loss(y_pred, y)
  gradients = tape.gradient(loss, [W, b])
  optimizer.apply_gradients(zip(gradients, [W, b]))

# 1000번 반복을 수행하면서 최적화를 수행합니다.
for i in range(1000):
  batch_xs, batch_ys = next(train_data_iter)
  batch_ys = tf.one_hot(batch_ys, depth=10)
  train_step(batch_xs, batch_ys)

# 학습이 끝나면 학습된 모델의 정확도를 출력합니다.
print("정확도(Accuracy): %f" % compute_accuracy(create_softmax_regression(x_test), tf.one_hot(y_test, depth=10))) # 정확도 : 약 91%