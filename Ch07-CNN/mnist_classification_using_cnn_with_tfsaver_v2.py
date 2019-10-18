# -*- coding: utf-8 -*-
# CNN을 이용한 MNIST 분류기 - tf.train.CheckpointManager API 예제

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

# CNN 모델을 정의합니다.
class CNN(object):
  # CNN 모델을 모델을 위한 tf.Variable들을 정의합니다.
  def __init__(self):
    # 첫번째 Convolution Layer
    # 5x5 Kernel Size를 가진 32개의 Filter를 적용합니다.
    self.W_conv1 = tf.Variable(tf.random.truncated_normal(shape=[5, 5, 1, 32], stddev=5e-2))
    self.b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

    # 두번째 Convolutional Layer
    # 5x5 Kernel Size를 가진 64개의 Filter를 적용합니다.
    self.W_conv2 = tf.Variable(tf.random.truncated_normal(shape=[5, 5, 32, 64], stddev=5e-2))
    self.b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

    # Fully Connected Layer
    # 7x7 크기를 가진 64개의 activation map을 1024개의 특징들로 변환합니다.
    self.W_fc1 = tf.Variable(tf.random.truncated_normal(shape=[7 * 7 * 64, 1024], stddev=5e-2))
    self.b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

    # Output Layer
    # 1024개의 특징들(feature)을 10개의 클래스-one-hot encoding으로 표현된 숫자 0~9-로 변환합니다.
    self.W_output = tf.Variable(tf.random.truncated_normal(shape=[1024, 10], stddev=5e-2))
    self.b_output = tf.Variable(tf.constant(0.1, shape=[10]))

  def __call__(self, x):
    # MNIST 데이터를 3차원 형태로 reshape합니다. MNIST 데이터는 grayscale 이미지기 때문에 3번째차원(컬러채널)의 값은 1입니다.
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    # 28x28x1 -> 28x28x32
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, self.W_conv1, strides=[1, 1, 1, 1], padding='SAME') + self.b_conv1)
    # 28x28x32 -> 14x14x32
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 14x14x32 -> 14x14x64
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, self.W_conv2, strides=[1, 1, 1, 1], padding='SAME') + self.b_conv2)
    # 14x14x64 -> 7x7x64
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 7x7x64(3136) -> 1024
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1) + self.b_fc1)
    # 1024 -> 10
    logits = tf.matmul(h_fc1, self.W_output) + self.b_output
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
  gradients = tape.gradient(loss, vars(model).values())
  optimizer.apply_gradients(zip(gradients, vars(model).values()))

# 모델의 정확도를 출력하는 함수를 정의합니다.
@tf.function
def compute_accuracy(y_pred, y):
  correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  return accuracy

# Convolutional Neural Networks(CNN) 모델을 선언합니다.
CNN_model = CNN()

# tf.train.CheckpointManager를 이용해서 파라미터를 저장합니다.
SAVER_DIR = "./model"
ckpt = tf.train.Checkpoint(**vars(CNN_model))
ckpt_manager = tf.train.CheckpointManager(
      ckpt, directory=SAVER_DIR, max_to_keep=5)
latest_ckpt = tf.train.latest_checkpoint(SAVER_DIR)

# 만약 저장된 모델과 파라미터가 있으면 이를 불러오고 (Restore)
# Restored 모델을 이용해서 테스트 데이터에 대한 정확도를 출력하고 프로그램을 종료합니다.
if latest_ckpt:
  ckpt.restore(latest_ckpt)
  print("테스트 데이터 정확도 (Restored) : %f" % compute_accuracy(CNN_model(x_test)[0], y_test))
  exit()

# 10000 Step만큼 최적화를 수행합니다.
for step in range(10000):
  # 50개씩 MNIST 데이터를 불러옵니다.
  batch_x, batch_y = next(train_data_iter)
  # 100 Step마다 training 데이터셋에 대한 정확도를 출력하고 tf.train.CheckpointManager를 이용해서 파라미터를 저장합니다.
  if step % 100 == 0:
    ckpt_manager.save(checkpoint_number=step)
    train_accuracy = compute_accuracy(CNN_model(batch_x)[0], batch_y)
    print("반복(Epoch): %d, 트레이닝 데이터 정확도: %f" % (step, train_accuracy))
  # 옵티마이저를 실행해 파라미터를 한스텝 업데이트합니다.
  train_step(CNN_model, batch_x, batch_y)

# 학습이 끝나면 테스트 데이터에 대한 정확도를 출력합니다.
print("정확도(Accuracy): %f" % compute_accuracy(CNN_model(x_test)[0], y_test))