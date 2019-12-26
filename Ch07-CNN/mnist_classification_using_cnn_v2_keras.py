# -*- coding: utf-8 -*-
# Convolutional Neural Networks(CNN)을 이용한 MNIST 분류기(Classifier) - Keras API를 이용한 구현

import tensorflow as tf

# MNIST 데이터를 다운로드 합니다.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# 이미지들을 float32 데이터 타입으로 변경합니다.
x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
# 28*28 형태의 이미지를 784차원으로 flattening 합니다.
x_train, x_test = x_train.reshape([-1, 784]), x_test.reshape([-1, 784])
# [0, 255] 사이의 값을 [0, 1]사이의 값으로 Normalize합니다.
x_train, x_test = x_train / 255., x_test / 255.
# 레이블 데이터에 one-hot encoding을 적용합니다.
y_train, y_test = tf.one_hot(y_train, depth=10), tf.one_hot(y_test, depth=10)

# tf.data API를 이용해서 데이터를 섞고 batch 형태로 가져옵니다.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(60000).batch(50)
train_data_iter = iter(train_data)

# tf.keras.Model을 이용해서 CNN 모델을 정의합니다.
class CNN(tf.keras.Model):
  def __init__(self):
    super(CNN, self).__init__()
    # 첫번째 Convolution Layer
    # 5x5 Kernel Size를 가진 32개의 Filter를 적용합니다.
    self.conv_layer_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu')
    self.pool_layer_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)

    # 두번째 Convolutional Layer
    # 5x5 Kernel Size를 가진 64개의 Filter를 적용합니다.
    self.conv_layer_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')
    self.pool_layer_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)

    # Fully Connected Layer
    # 7x7 크기를 가진 64개의 activation map을 1024개의 특징들로 변환합니다.
    self.flatten_layer = tf.keras.layers.Flatten()
    self.fc_layer_1 = tf.keras.layers.Dense(1024, activation='relu')

    # Output Layer
    # 1024개의 특징들(feature)을 10개의 클래스-one-hot encoding으로 표현된 숫자 0~9-로 변환합니다.
    self.output_layer = tf.keras.layers.Dense(10, activation=None)

  def call(self, x):
    # MNIST 데이터를 3차원 형태로 reshape합니다. MNIST 데이터는 grayscale 이미지기 때문에 3번째차원(컬러채널)의 값은 1입니다.
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    # 28x28x1 -> 28x28x32
    h_conv1 = self.conv_layer_1(x_image)
    # 28x28x32 -> 14x14x32
    h_pool1 = self.pool_layer_1(h_conv1)
    # 14x14x32 -> 14x14x64
    h_conv2 = self.conv_layer_2(h_pool1)
    # 14x14x64 -> 7x7x64
    h_pool2 = self.pool_layer_2(h_conv2)
    # 7x7x64(3136) -> 1024
    h_pool2_flat = self.flatten_layer(h_pool2)
    h_fc1 = self.fc_layer_1(h_pool2_flat)
    # 1024 -> 10
    logits = self.output_layer(h_fc1)
    y_pred = tf.nn.softmax(logits)

    return y_pred, logits

# cross-entropy 손실 함수를 정의합니다.
@tf.function
def cross_entropy_loss(logits, y):
  return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

# 최적화를 위한 Adam 옵티마이저를 정의합니다.
optimizer = tf.optimizers.Adam(1e-4)

# 최적화를 위한 function을 정의합니다.
@tf.function
def train_step(model, x, y):
  with tf.GradientTape() as tape:
    y_pred, logits = model(x)
    loss = cross_entropy_loss(logits, y)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# 모델의 정확도를 출력하는 함수를 정의합니다.
@tf.function
def compute_accuracy(y_pred, y):
  correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  return accuracy

# Convolutional Neural Networks(CNN) 모델을 선언합니다.
CNN_model = CNN()

# 10000 Step만큼 최적화를 수행합니다.
for i in range(10000):
  # 50개씩 MNIST 데이터를 불러옵니다.
  batch_x, batch_y = next(train_data_iter)
  # 100 Step마다 training 데이터셋에 대한 정확도를 출력합니다.
  if i % 100 == 0:
    train_accuracy = compute_accuracy(CNN_model(batch_x)[0], batch_y)
    print("반복(Epoch): %d, 트레이닝 데이터 정확도: %f" % (i, train_accuracy))
  # 옵티마이저를 실행해 파라미터를 한스텝 업데이트합니다.
  train_step(CNN_model, batch_x, batch_y)

# 학습이 끝나면 학습된 모델의 정확도를 출력합니다.
print("정확도(Accuracy): %f" % compute_accuracy(CNN_model(x_test)[0], y_test))