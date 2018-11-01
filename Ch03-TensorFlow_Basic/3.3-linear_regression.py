# -*- coding: utf-8 -*-

import tensorflow as tf

# 선형회귀 모델(Wx + b)을 정의합니다.
W = tf.Variable(tf.random_normal(shape=[1]))   
b = tf.Variable(tf.random_normal(shape=[1]))
x = tf.placeholder(tf.float32)
linear_model = W*x + b

# True Value를 입력받기위한 플레이스홀더를 정의합니다.
y = tf.placeholder(tf.float32)

# 손실 함수를 정의합니다.
loss = tf.reduce_mean(tf.square(linear_model - y)) # MSE 손실함수 \mean{(y' - y)^2}

# 최적화를 위한 그라디언트 디센트 옵티마이저를 정의합니다.
optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(loss)

# 트레이닝을 위한 입력값과 출력값을 준비합니다. 
x_train = [1, 2, 3, 4]
y_train = [2, 4, 6, 8]

# 세션을 실행하고 파라미터(W,b)를 noraml distirubtion에서 추출한 임의의 값으로 초기화합니다.
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 경사하강법을 1000번 수행합니다.
for i in range(1000):
  sess.run(train_step, feed_dict={x: x_train, y: y_train})

# 테스트를 위한 입력값을 준비합니다.
x_test = [3.5, 5, 5.5, 6]
# 테스트 데이터를 이용해 학습된 선형회귀 모델이 데이터의 경향성(y=2x)을 잘 학습했는지 측정합니다.
# 예상되는 참값 : [7, 10, 11, 12]
print(sess.run(linear_model, feed_dict={x: x_test}))

sess.close()