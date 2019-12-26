# -*- coding: utf-8 -*-

"""
CIFAR-10 Convolutional Neural Networks(CNN) 예제 - Keras API를 이용한 구현
"""

import tensorflow as tf

# CIFAR-10 데이터를 다운로드하고 데이터를 불러옵니다.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# 이미지들을 float32 데이터 타입으로 변경합니다.
x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
# [0, 255] 사이의 값을 [0, 1]사이의 값으로 Normalize합니다.
x_train, x_test = x_train / 255., x_test / 255.
# scalar 형태의 레이블(0~9)을 One-hot Encoding 형태로 변환합니다.
y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10), axis=1)

# tf.data API를 이용해서 데이터를 섞고 batch 형태로 가져옵니다.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train_one_hot))
train_data = train_data.repeat().shuffle(50000).batch(128)
train_data_iter = iter(train_data)

test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test_one_hot))
test_data = test_data.batch(1000)
test_data_iter = iter(test_data)

# tf.keras.Model을 이용해서 CNN 모델을 정의합니다.
class CNN(tf.keras.Model):
  def __init__(self):
    super(CNN, self).__init__()
    # 첫번째 convolutional layer - 하나의 RGB 이미지를 64개의 특징들(feature)으로 맵핑(mapping)합니다.
    self.conv_layer_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')
    self.pool_layer_1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)
    # 두번째 convolutional layer - 64개의 특징들(feature)을 64개의 특징들(feature)로 맵핑(mapping)합니다.
    self.conv_layer_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')
    self.pool_layer_2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)
    # 세번째 convolutional layer
    self.conv_layer_3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')
    # 네번째 convolutional layer
    self.conv_layer_4 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')
    # 다섯번째 convolutional layer
    self.conv_layer_5 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')
    # Fully Connected Layer 1 - 2번의 downsampling 이후에, 우리의 32x32 이미지는 8x8x128 특징맵(feature map)이 됩니다.
    # 이를 384개의 특징들로 맵핑(mapping)합니다.
    self.flatten_layer = tf.keras.layers.Flatten()
    self.fc_layer_1 = tf.keras.layers.Dense(384, activation='relu')
    self.dropout = tf.keras.layers.Dropout(0.2)

    # Fully Connected Layer 2 - 384개의 특징들(feature)을 10개의 클래스-airplane, automobile, bird...-로 맵핑(mapsping)합니다.
    self.output_layer = tf.keras.layers.Dense(10, activation=None)

  def call(self, x, is_training):
    # 입력 이미지
    h_conv1 = self.conv_layer_1(x)
    h_pool1 = self.pool_layer_1(h_conv1)
    h_conv2 = self.conv_layer_2(h_pool1)
    h_pool2 = self.pool_layer_2(h_conv2)
    h_conv3 = self.conv_layer_3(h_pool2)
    h_conv4 = self.conv_layer_4(h_conv3)
    h_conv5 = self.conv_layer_5(h_conv4)
    h_conv5_flat = self.flatten_layer(h_conv5)
    h_fc1 = self.fc_layer_1(h_conv5_flat)
    # Dropout - 모델의 복잡도를 컨트롤합니다. 특징들의 co-adaptation을 방지합니다.
    h_fc1_drop = self.dropout(h_fc1, training=is_training)
    logits = self.output_layer(h_fc1_drop)
    y_pred = tf.nn.softmax(logits)

    return y_pred, logits

# cross-entropy 손실 함수를 정의합니다.
@tf.function
def cross_entropy_loss(logits, y):
  return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

# 최적화를 위한 RMSprop 옵티마이저를 정의합니다.
optimizer = tf.optimizers.RMSprop(1e-3)

# 최적화를 위한 function을 정의합니다.
@tf.function
def train_step(model, x, y, is_training):
  with tf.GradientTape() as tape:
    y_pred, logits = model(x, is_training)
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
  batch_x, batch_y = next(train_data_iter)

  # 100 Step마다 training 데이터셋에 대한 정확도와 loss를 출력합니다.
  if i % 100 == 0:
    train_accuracy = compute_accuracy(CNN_model(batch_x, False)[0], batch_y)
    loss_print = cross_entropy_loss(CNN_model(batch_x, False)[1], batch_y)

    print("반복(Epoch): %d, 트레이닝 데이터 정확도: %f, 손실 함수(loss): %f" % (i, train_accuracy, loss_print))
  # 20% 확률의 Dropout을 이용해서 학습을 진행합니다.
  train_step(CNN_model, batch_x, batch_y, True)

# 학습이 끝나면 테스트 데이터(10000개)에 대한 정확도를 출력합니다.
test_accuracy = 0.0
for i in range(10):
  test_batch_x, test_batch_y = next(test_data_iter)
  test_accuracy = test_accuracy + compute_accuracy(CNN_model(test_batch_x, False)[0], test_batch_y).numpy()
test_accuracy = test_accuracy / 10
print("테스트 데이터 정확도: %f" % test_accuracy)