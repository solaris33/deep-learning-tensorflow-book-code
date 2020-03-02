# -*- coding: utf-8 -*-

import tensorflow as tf

# tf.keras.Model을 이용해서 선형회귀 모델(Wx + b)을 정의합니다.
class LinearRegression(tf.keras.Model):
  def __init__(self):
    super(LinearRegression, self).__init__()
    self.linear_layer = tf.keras.layers.Dense(1, activation=None)

  def call(self, x):
    y_pred = self.linear_layer(x)

    return y_pred

# 손실 함수를 정의합니다.
@tf.function
def mse_loss(y_pred, y):
  return tf.reduce_mean(tf.square(y_pred - y)) # MSE 손실함수 \mean{(y' - y)^2}

# 최적화를 위한 그라디언트 디센트 옵티마이저를 정의합니다.
optimizer = tf.optimizers.SGD(0.01)

# 텐서보드 summary 정보들을 저장할 폴더 경로를 설정하고 FileWriter를 선언합니다.
summary_writer = tf.summary.create_file_writer('./tensorboard_log')

# 최적화를 위한 function을 정의합니다.
@tf.function
def train_step(model, x, y):
  with tf.GradientTape() as tape:
    y_pred = model(x)
    loss = mse_loss(y_pred, y)
  # 매 step마다 tf.summary.scalar 텐서보드 로그를 기록합니다.
  with summary_writer.as_default():
    tf.summary.scalar('loss', loss, step=optimizer.iterations)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# 트레이닝을 위한 입력값과 출력값을 준비합니다. 
x_train = [1.0, 2.0, 3.0, 4.0]
y_train = [2.0, 4.0, 6.0, 8.0]

# tf.data API를 이용해서 데이터를 batch 형태로 가져옵니다.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().batch(1)
train_data_iter = iter(train_data)

# LinearRegression 모델을 선언합니다.
LinearRegression_model = LinearRegression()

# 경사하강법을 1000번 수행합니다.
for i in range(1000):
  batch_xs, batch_ys = next(train_data_iter)
  # tf.keras.layers.Dense API의 최소 input dimension인 2-dimension를 맞추기 위한 차원 확장
  batch_xs = tf.expand_dims(batch_xs, 0)
  train_step(LinearRegression_model, batch_xs, batch_ys)

# 테스트를 위한 입력값을 준비합니다.
x_test = [3.5, 5.0, 5.5, 6.0]
test_data = tf.data.Dataset.from_tensor_slices((x_test))
test_data = test_data.batch(1)

for batch_x_test in test_data:
  batch_x_test = tf.expand_dims(batch_x_test, 0)
  # 테스트 데이터를 이용해 학습된 선형회귀 모델이 데이터의 경향성(y=2x)을 잘 학습했는지 측정합니다.
  # 예상되는 참값 : [7, 10, 11, 12]
  print(tf.squeeze(LinearRegression_model(batch_x_test), 0).numpy())