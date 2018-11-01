# -*- coding: utf-8 -*-

import tensorflow as tf

# 선형회귀 모델(Wx + b)을 정의합니다.
W = tf.Variable(tf.random_normal(shape=[1]), name="W")   
b = tf.Variable(tf.random_normal(shape=[1]), name="b")
x = tf.placeholder(tf.float32, name="x")
linear_model = W*x + b

# True Value를 입력받기위한 플레이스홀더를 정의합니다.
y = tf.placeholder(tf.float32, name="y")

# 손실 함수를 정의합니다.
loss = tf.reduce_mean(tf.square(linear_model - y)) # MSE 손실함수 \mean{(y' - y)^2}
# 텐서보드를 위한 요약정보(scalar)를 정의합니다.
tf.summary.scalar('loss', loss)

# 최적화를 위한 옵티마이저를 정의합니다.
optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(loss)

# 트레이닝을 위한 입력값과 출력값을 준비합니다. 
x_train = [1, 2, 3, 4]
y_train = [2, 4, 6, 8]

# 세션을 실행하고 파라미터(W,b)를 noraml distirubtion에서 추출한 임의의 값으로 초기화합니다.
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 텐서보드 요약정보들을 하나로 합칩니다.
merged = tf.summary.merge_all()
# 텐서보드 summary 정보들을 저장할 폴더 경로를 설정합니다.
tensorboard_writer = tf.summary.FileWriter('./tensorboard_log', sess.graph)

# 경사하강법을 1000번 수행합니다.
for i in range(1000):
  sess.run(train_step, feed_dict={x: x_train, y: y_train})

  # 매스텝마다 텐서보드 요약정보값들을 계산해서 지정된 경로('./tensorboard_log')에 저장합니다.
  summary = sess.run(merged, feed_dict={x: x_train, y: y_train})
  tensorboard_writer.add_summary(summary, i)

# 테스트를 위한 입력값을 준비합니다.
x_test = [3.5, 5, 5.5, 6]
# 테스트 데이터를 이용해 학습된 선형회귀 모델이 데이터의 경향성(y=2x)을 잘 학습했는지 측정합니다.
# 예상되는 참값 : [7, 10, 11, 12]
print(sess.run(linear_model, feed_dict={x: x_test}))

sess.close()