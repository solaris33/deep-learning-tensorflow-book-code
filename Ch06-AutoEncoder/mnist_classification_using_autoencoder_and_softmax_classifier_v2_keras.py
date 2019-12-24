# -*- coding: utf-8 -*-
# MNIST 숫자 분류를 위한 Autoencoder+Softmax 분류기 예제 - Keras API를 이용한 구현

import tensorflow as tf

# MNIST 데이터를 다운로드 합니다.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# 이미지들을 float32 데이터 타입으로 변경합니다.
x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
# 28*28 형태의 이미지를 784차원으로 flattening 합니다.
x_train, x_test = x_train.reshape([-1, 784]), x_test.reshape([-1, 784])
# [0, 255] 사이의 값을 [0, 1]사이의 값으로 Normalize합니다.
x_train, x_test = x_train / 255., x_test / 255.

# 학습에 필요한 설정값들을 정의합니다.
learning_rate_RMSProp = 0.02
learning_rate_GradientDescent = 0.5
num_epochs = 100         # 반복횟수
batch_size = 256
display_step = 1         # 몇 Step마다 log를 출력할지 결정합니다.
input_size = 784         # MNIST 데이터 input (이미지 크기: 28*28)
hidden1_size = 128       # 첫번째 히든레이어의 노드 개수 
hidden2_size = 64        # 두번째 히든레이어의 노드 개수 

# tf.data API를 이용해서 데이터를 섞고 batch 형태로 가져옵니다.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.shuffle(60000).batch(batch_size)

def random_normal_intializer_with_stddev_1():
  return tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None)

# tf.keras.Model을 이용해서 Autoencoder 모델을 정의합니다.
class AutoEncoder(tf.keras.Model):
  def __init__(self):
    super(AutoEncoder, self).__init__()
    # 인코딩(Encoding) - 784 -> 128 -> 64
    self.hidden_layer_1 = tf.keras.layers.Dense(hidden1_size,
                                                activation='sigmoid',
                                                kernel_initializer=random_normal_intializer_with_stddev_1(),
                                                bias_initializer=random_normal_intializer_with_stddev_1())
    self.hidden_layer_2 = tf.keras.layers.Dense(hidden2_size,
                                                activation='sigmoid',
                                                kernel_initializer=random_normal_intializer_with_stddev_1(),
                                                bias_initializer=random_normal_intializer_with_stddev_1())
    # 디코딩(Decoding) 64 -> 128 -> 784
    self.hidden_layer_3 = tf.keras.layers.Dense(hidden1_size,
                                                activation='sigmoid',
                                                kernel_initializer=random_normal_intializer_with_stddev_1(),
                                                bias_initializer=random_normal_intializer_with_stddev_1())
    self.output_layer = tf.keras.layers.Dense(input_size,
                                                activation='sigmoid',
                                                kernel_initializer=random_normal_intializer_with_stddev_1(),
                                                bias_initializer=random_normal_intializer_with_stddev_1())

  def call(self, x):
    H1_output = self.hidden_layer_1(x)
    H2_output = self.hidden_layer_2(H1_output)
    H3_output = self.hidden_layer_3(H2_output)
    X_reconstructed = self.output_layer(H3_output)

    return X_reconstructed, H2_output

# tf.keras.Model을 이용해서 Softmax 분류기를 정의합니다.
class SoftmaxClassifier(tf.keras.Model):
  def __init__(self):
    super(SoftmaxClassifier, self).__init__()
    # 원본 MNIST 이미지(784) 대신 오토인코더의 압축된 특징(64)을 입력값으로 받습니다.
    self.softmax_layer = tf.keras.layers.Dense(10,
                                               activation='softmax',
                                               kernel_initializer='zeros',
                                               bias_initializer='zeros')

  def call(self, x):
    y_pred = self.softmax_layer(x)

    return y_pred

@tf.function
def pretraining_mse_loss(y_pred, y_true):
  return tf.reduce_mean(tf.pow(y_true - y_pred, 2)) # MSE(Mean of Squared Error) 손실함수

@tf.function
def finetuning_cross_entropy_loss(y_pred_softmax, y):
  return tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(y_pred_softmax), axis=[1]))     # cross-entropy loss 함수

# 1. Pre-Training : MNIST 데이터 재구축을 목적으로하는 옵티마이저와 최적화를 위한 function 정의합니다.
pretraining_optimizer = tf.optimizers.RMSprop(learning_rate_RMSProp, epsilon=1e-10)
@tf.function
def pretraining_train_step(autoencoder_model, x):
  # 타겟데이터는 인풋데이터와 같습니다.
  y_true = x
  with tf.GradientTape() as tape:
    y_pred, _ = autoencoder_model(x)
    pretraining_loss = pretraining_mse_loss(y_pred, y_true)
  gradients = tape.gradient(pretraining_loss, autoencoder_model.trainable_variables)
  pretraining_optimizer.apply_gradients(zip(gradients, autoencoder_model.trainable_variables))

# 2. Fine-Tuning :  MNIST 데이터 분류를 목적으로하는 옵티마이저와 최적화를 위한 function 정의합니다.
finetuning_optimizer = tf.optimizers.SGD(learning_rate_GradientDescent)
@tf.function
def finetuning_train_step(autoencoder_model, softmax_classifier_model, x, y):
  with tf.GradientTape() as tape:
    y_pred, extracted_features = autoencoder_model(x)
    y_pred_softmax = softmax_classifier_model(extracted_features)
    finetuning_loss = finetuning_cross_entropy_loss(y_pred_softmax, y)
  autoencoder_encoding_variables = autoencoder_model.hidden_layer_1.trainable_variables + autoencoder_model.hidden_layer_2.trainable_variables
  gradients = tape.gradient(finetuning_loss, autoencoder_encoding_variables + softmax_classifier_model.trainable_variables)
  finetuning_optimizer.apply_gradients(zip(gradients, autoencoder_encoding_variables + softmax_classifier_model.trainable_variables))

# 모델의 정확도를 출력하는 함수를 정의합니다.
@tf.function
def compute_accuracy(y_pred_softmax, y):
  correct_prediction = tf.equal(tf.argmax(y_pred_softmax,1), tf.argmax(y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  return accuracy

# Autoencoder 모델을 선언합니다.
AutoEncoder_model = AutoEncoder()
# Softmax 분류기 모델을 선언합니다. (입력으로 Autoencoder의 압축된 특징을 넣습니다.)
SoftmaxClassifier_model = SoftmaxClassifier()

# Step 1: MNIST 데이터 재구축을 위한 오토인코더 최적화(Pre-Training)를 수행합니다.
for epoch in range(num_epochs):
  # 모든 배치들에 대해서 최적화를 수행합니다.
  for batch_x, _ in train_data:
    _, pretraining_loss_print = pretraining_train_step(AutoEncoder_model, batch_x), pretraining_mse_loss(AutoEncoder_model(batch_x)[0], batch_x)
  # 지정된 epoch마다 학습결과를 출력합니다.
  if epoch % display_step == 0:
    print("반복(Epoch): %d, Pre-Training 손실 함수(pretraining_loss): %f" % ((epoch + 1), pretraining_loss_print))
print("Step 1 : MNIST 데이터 재구축을 위한 오토인코더 최적화 완료(Pre-Training)")

# Step 2: MNIST 데이터 분류를 위한 오토인코더+Softmax 분류기 최적화(Fine-tuning)를 수행합니다.
for epoch in range(num_epochs + 100):
  # 모든 배치들에 대해서 최적화를 수행합니다.
  for batch_x, batch_y in train_data:
    batch_y = tf.one_hot(batch_y, depth=10)
    _, finetuning_loss_print = finetuning_train_step(AutoEncoder_model, SoftmaxClassifier_model, batch_x, batch_y), finetuning_cross_entropy_loss(SoftmaxClassifier_model(AutoEncoder_model(batch_x)[1]), batch_y)
  # 지정된 epoch마다 학습결과를 출력합니다.
  if epoch % display_step == 0:
    print("반복(Epoch): %d, Fine-tuning 손실 함수(finetuning_loss): %f" % ((epoch + 1), finetuning_loss_print))
print("Step 2 : MNIST 데이터 분류를 위한 오토인코더+Softmax 분류기 최적화 완료(Fine-Tuning)")

# 오토인코더+Softmax 분류기 모델의 정확도를 출력합니다.
print("정확도(오토인코더+Softmax 분류기): %f" % compute_accuracy(SoftmaxClassifier_model(AutoEncoder_model(x_test)[1]), tf.one_hot(y_test, depth=10)))  # 정확도 : 약 96%