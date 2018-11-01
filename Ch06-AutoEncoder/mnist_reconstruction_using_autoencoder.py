# -*- coding: utf-8 -*-
# AutoEncoder를 이용한 MNIST Reconstruction

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# MNIST 데이터를 다운로드 합니다.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 학습에 필요한 설정값들을 정의합니다.
learning_rate = 0.02
training_epochs = 50    # 반복횟수
batch_size = 256        # 배치개수
display_step = 1        # 손실함수 출력 주기
examples_to_show = 10   # 보여줄 MNIST Reconstruction 이미지 개수
input_size = 784        # 28*28
hidden1_size = 256 
hidden2_size = 128

# 입력을 받기 위한 플레이스홀더를 정의합니다.
# Autoencoder는 Unsupervised Learning이므로 타겟 레이블(label) y가 필요하지 않습니다.
x = tf.placeholder(tf.float32, shape=[None, input_size])

# Autoencoder 구조를 정의합니다.
def build_autoencoder(x):
  # 인코딩(Encoding) - 784 -> 256 -> 128
  W1 = tf.Variable(tf.random_normal(shape=[input_size, hidden1_size]))
  b1 = tf.Variable(tf.random_normal(shape=[hidden1_size]))
  H1_output = tf.nn.sigmoid(tf.matmul(x,W1) + b1)
  W2 = tf.Variable(tf.random_normal(shape=[hidden1_size, hidden2_size]))
  b2 = tf.Variable(tf.random_normal(shape=[hidden2_size]))
  H2_output = tf.nn.sigmoid(tf.matmul(H1_output,W2) + b2)
  # 디코딩(Decoding) 128 -> 256 -> 784
  W3 = tf.Variable(tf.random_normal(shape=[hidden2_size, hidden1_size]))
  b3 = tf.Variable(tf.random_normal(shape=[hidden1_size]))
  H3_output = tf.nn.sigmoid(tf.matmul(H2_output,W3) + b3)
  W4 = tf.Variable(tf.random_normal(shape=[hidden1_size, input_size]))
  b4 = tf.Variable(tf.random_normal(shape=[input_size]))
  reconstructed_x = tf.nn.sigmoid(tf.matmul(H3_output,W4) + b4)

  return reconstructed_x

# Autoencoder를 선언합니다.
y_pred = build_autoencoder(x)
# 타겟데이터는 인풋데이터와 같습니다.
y_true = x

# 손실함수와 옵티마이저를 정의합니다.
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))   # MSE(Mean of Squared Error) 손실함수
train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# 세션을 열고 그래프를 실행합니다.
with tf.Session() as sess:
  # 변수들의 초기값을 할당합니다.
  sess.run(tf.global_variables_initializer())

  # 지정된 횟수만큼 최적화를 수행합니다.
  for epoch in range(training_epochs):
    # 전체 배치를 불러옵니다.
    total_batch = int(mnist.train.num_examples/batch_size)
    # 모든 배치들에 대해서 최적화를 수행합니다.
    for i in range(total_batch):
      batch_xs, batch_ys = mnist.train.next_batch(batch_size)
      _, current_loss = sess.run([train_step, loss], feed_dict={x: batch_xs})
    # 지정된 epoch마다 학습결과를 출력합니다.
    if epoch % display_step == 0:
      print("반복(Epoch): %d, 손실 함수(Loss): %f" % ((epoch+1), current_loss))

  # 테스트 데이터로 Reconstruction을 수행합니다.
  reconstructed_result = sess.run(y_pred, feed_dict={x: mnist.test.images[:examples_to_show]})
  # 원본 MNIST 데이터와 Reconstruction 결과를 비교합니다.
  f, a = plt.subplots(2, 10, figsize=(10, 2))
  for i in range(examples_to_show):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    a[1][i].imshow(np.reshape(reconstructed_result[i], (28, 28)))
  f.savefig('reconstructed_mnist_image.png')  # reconstruction 결과를 png로 저장합니다.
  f.show()
  plt.draw()
  plt.waitforbuttonpress()