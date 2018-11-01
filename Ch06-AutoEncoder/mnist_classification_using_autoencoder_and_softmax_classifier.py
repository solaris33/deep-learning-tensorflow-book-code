# -*- coding: utf-8 -*-
# MNIST 숫자 분류를 위한 Autoencoder+Softmax 분류기 예제 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# MNIST 데이터를 다운로드 합니다.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 학습에 필요한 설정값들을 정의합니다.
learning_rate_RMSProp = 0.02
learning_rate_GradientDescent = 0.5
num_epochs = 100         # 반복횟수
batch_size = 256          
display_step = 1         # 몇 Step마다 log를 출력할지 결정합니다.
input_size = 784         # MNIST 데이터 input (이미지 크기: 28*28)
hidden1_size = 128       # 첫번째 히든레이어의 노드 개수 
hidden2_size = 64        # 두번째 히든레이어의 노드 개수 

# 입력을 받기 위한 플레이스홀더를 정의합니다.
x = tf.placeholder(tf.float32, shape=[None, input_size])   # 인풋을 위한 플레이스홀더를 정의합니다.
y = tf.placeholder(tf.float32, shape=[None, 10])           # True MNIST 숫자값

# Autoencoder 구조를 정의합니다.
def build_autoencoder(x):
  # 인코딩(Encoding) - 784 -> 128 -> 64
  Wh_1 = tf.Variable(tf.random_normal([input_size, hidden1_size]))   
  bh_1 = tf.Variable(tf.random_normal([hidden1_size]))
  H1_output = tf.nn.sigmoid(tf.matmul(x, Wh_1) +bh_1)
  Wh_2 = tf.Variable(tf.random_normal([hidden1_size, hidden2_size]))
  bh_2 = tf.Variable(tf.random_normal([hidden2_size]))
  H2_output = tf.nn.sigmoid(tf.matmul(H1_output, Wh_2) +bh_2)
  # 디코딩(Decoding) 64 -> 128 -> 784
  Wh_3 = tf.Variable(tf.random_normal([hidden2_size, hidden1_size]))
  bh_3 = tf.Variable(tf.random_normal([hidden1_size]))
  H3_output = tf.nn.sigmoid(tf.matmul(H2_output, Wh_3) +bh_3)
  Wo = tf.Variable(tf.random_normal([hidden1_size, input_size]))
  bo = tf.Variable(tf.random_normal([input_size]))
  X_reconstructed = tf.nn.sigmoid(tf.matmul(H3_output,Wo) + bo)
  
  return X_reconstructed, H2_output 

# Softmax 분류기를 정의합니다.
def build_softmax_classifier(x):
  W_softmax = tf.Variable(tf.zeros([hidden2_size, 10]))    # 원본 MNIST 이미지(784) 대신 오토인코더의 압축된 특징(64)을 입력값으로 받습니다.
  b_softmax = tf.Variable(tf.zeros([10]))
  y_pred = tf.nn.softmax(tf.matmul(x, W_softmax) + b_softmax)

  return y_pred

# Autoencoder를 선언합니다.
y_pred, extracted_features = build_autoencoder(x) # Autoencoder의 Reconstruction 결과(784), 압축된 Features(64)
# 타겟데이터는 인풋데이터와 같습니다.
y_true = x
# Softmax 분류기를 선언합니다. (입력으로 Autoencoder의 압축된 특징을 넣습니다.)
y_pred_softmax = build_softmax_classifier(extracted_features)

# 1. Pre-Training : MNIST 데이터 재구축을 목적으로하는 손실함수와 옵티마이저를 정의합니다.
pretraining_loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))     # MSE 손실 함수
pretraining_train_step = tf.train.RMSPropOptimizer(learning_rate_RMSProp).minimize(pretraining_loss)
# 2. Fine-Tuning :  MNIST 데이터 분류를 목적으로하는 손실함수와 옵티마이저를 정의합니다.
finetuning_loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred_softmax), reduction_indices=[1]))     # cross-entropy loss 함수
finetuning_train_step = tf.train.GradientDescentOptimizer(learning_rate_GradientDescent).minimize(finetuning_loss)

# 세션을 열고 그래프를 실행합니다.
with tf.Session() as sess:
  # 변수들의 초기값을 할당합니다.
  sess.run(tf.global_variables_initializer())

  # 전체 배치 개수를 불러옵니다.
  total_batch = int(mnist.train.num_examples/batch_size)

  # Step 1: MNIST 데이터 재구축을 위한 오토인코더 최적화(Pre-Training)를 수행합니다.
  for epoch in range(num_epochs):
    # 모든 배치들에 대해서 최적화를 수행합니다.
    for i in range(total_batch):
      batch_xs, batch_ys = mnist.train.next_batch(batch_size)
      _, pretraining_loss_print = sess.run([pretraining_train_step, pretraining_loss], feed_dict={x: batch_xs})
    # 지정된 epoch마다 학습결과를 출력합니다.
    if epoch % display_step == 0:
      print("반복(Epoch): %d, Pre-Training 손실 함수(pretraining_loss): %f" % ((epoch+1), pretraining_loss_print))
  print("Step 1 : MNIST 데이터 재구축을 위한 오토인코더 최적화 완료(Pre-Training)")

  # Step 2: MNIST 데이터 분류를 위한 오토인코더+Softmax 분류기 최적화(Fine-tuning)를 수행합니다.
  for epoch in range(num_epochs + 100):
    # 모든 배치들에 대해서 최적화를 수행합니다.
    for i in range(total_batch):
      batch_xs, batch_ys = mnist.train.next_batch(batch_size)
      _, finetuning_loss_print = sess.run([finetuning_train_step, finetuning_loss], feed_dict={x: batch_xs,  y: batch_ys})
    # 지정된 epoch마다 학습결과를 출력합니다.
    if epoch % display_step == 0:
      print("반복(Epoch): %d, Fine-tuning 손실 함수(finetuning_loss): %f" % ((epoch+1), finetuning_loss_print))
  print("Step 2 : MNIST 데이터 분류를 위한 오토인코더+Softmax 분류기 최적화 완료(Fine-Tuning)")

  # 오토인코더+Softmax 분류기 모델의 정확도를 출력합니다.
  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_pred_softmax,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print("정확도(오토인코더+Softmax 분류기): %f" % sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})) # 정확도 : 약 96%