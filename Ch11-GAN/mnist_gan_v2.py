# -*- coding: utf-8 -*-
# GAN(Generative Adversarial Networks)을 이용한 MNIST 데이터 생성
# Reference : https://github.com/TengdaHan/GAN-TensorFlow

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# MNIST 데이터를 다운로드 합니다.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# 이미지들을 float32 데이터 타입으로 변경합니다.
x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
# 28*28 형태의 이미지를 784차원으로 flattening 합니다.
x_train, x_test = x_train.reshape([-1, 784]), x_test.reshape([-1, 784])
# [0, 255] 사이의 값을 [0, 1]사이의 값으로 Normalize합니다.
x_train, x_test = x_train / 255., x_test / 255.

# 생성된 MNIST 이미지를 8x8 Grid로 보여주는 plot 함수를 정의합니다.
def plot(samples):
  fig = plt.figure(figsize=(8, 8))
  gs = gridspec.GridSpec(8, 8)
  gs.update(wspace=0.05, hspace=0.05)

  for i, sample in enumerate(samples):
    ax = plt.subplot(gs[i])
    plt.axis('off')
    plt.imshow(sample.reshape(28, 28))

  return fig

# 학습에 필요한 설정값들을 선언합니다.
num_epoch = 100000
batch_size = 64
num_input = 28 * 28
num_latent_variable = 100   # 잠재 변수 z의 차원
num_hidden = 128
learning_rate = 0.001

# tf.data API를 이용해서 데이터를 섞고 batch 형태로 가져옵니다.
train_data = tf.data.Dataset.from_tensor_slices(x_train)
train_data = train_data.repeat().shuffle(60000).batch(batch_size)
train_data_iter = iter(train_data)

# Generator 모델을 정의합니다.
# Inputs:
#   X : 인풋 Latent Variable
# Outputs:
#   generated_mnist_image : 생성된 MNIST 이미지
class Generator(object):
  # Generator 모델을 위한 tf.Variable들을 정의합니다.
  def __init__(self):
    # 100 -> 128 -> 784
    # 히든 레이어 파라미터
    self.G_W1 = tf.Variable(tf.random.normal(shape=[num_latent_variable, num_hidden], stddev=5e-2))
    self.G_b1 = tf.Variable(tf.constant(0.1, shape=[num_hidden]))
    # 아웃풋 레이어 파라미터
    self.G_W2 = tf.Variable(tf.random.normal(shape=[num_hidden, num_input], stddev=5e-2))
    self.G_b2 = tf.Variable(tf.constant(0.1, shape=[num_input]))

  def __call__(self, X):
    hidden_layer = tf.nn.relu((tf.matmul(X, self.G_W1) + self.G_b1))
    output_layer = tf.matmul(hidden_layer, self.G_W2) + self.G_b2
    generated_mnist_image = tf.nn.sigmoid(output_layer)

    return generated_mnist_image

# Discriminator 모델을 정의합니다.
# Inputs:
#   X : 인풋 이미지
# Outputs:
#   predicted_value : Discriminator가 판단한 True(1) or Fake(0)
#   logits : sigmoid를 씌우기전의 출력값
class Discriminator(object):
  # Discriminator 모델을 위한 tf.Variable들을 정의합니다.
  def __init__(self):
    # 784 -> 128 -> 1
    # 히든 레이어 파라미터
    self.D_W1 = tf.Variable(tf.random.normal(shape=[num_input, num_hidden], stddev=5e-2))
    self.D_b1 = tf.Variable(tf.constant(0.1, shape=[num_hidden]))
    # 아웃풋 레이어 파라미터
    self.D_W2 = tf.Variable(tf.random.normal(shape=[num_hidden, 1], stddev=5e-2))
    self.D_b2 = tf.Variable(tf.constant(0.1, shape=[1]))

  def __call__(self, X):
    hidden_layer = tf.nn.relu((tf.matmul(X, self.D_W1) + self.D_b1))
    logits = tf.matmul(hidden_layer, self.D_W2) + self.D_b2
    predicted_value = tf.nn.sigmoid(logits)

    return predicted_value, logits

# Generator의 손실 함수를 정의합니다.
@tf.function
def generator_loss(D_fake_logits):
  return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)))         # log(D(G(z))

# Discriminator의 손실 함수를 정의합니다.
@tf.function
def discriminator_loss(D_real_logits, D_fake_logits):
  d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real_logits)))  # log(D(x))
  d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake_logits)))  # log(1-D(G(z)))
  d_loss = d_loss_real + d_loss_fake  # log(D(x)) + log(1-D(G(z)))

  return d_loss

# 생성자(Generator) 모델을 선언합니다.
Generator_model = Generator()

# 구분자(Discriminator) 모델을 선언합니다.
Discriminator_model = Discriminator()

# Discriminator와 Generator를 위한 Optimizer들을 정의합니다.
discriminator_optimizer = tf.optimizers.Adam(learning_rate)
generator_optimizer = tf.optimizers.Adam(learning_rate)

# Discriminator 최적화를 위한 function을 정의합니다.
@tf.function
def d_train_step(discriminator_model, real_image, fake_image):
  with tf.GradientTape() as disc_tape:
    D_real, D_real_logits = discriminator_model(real_image)  # D(x)
    D_fake, D_fake_logits = discriminator_model(fake_image)  # D(G(z))
    loss = discriminator_loss(D_real_logits, D_fake_logits)
  gradients = disc_tape.gradient(loss, vars(discriminator_model).values())
  discriminator_optimizer.apply_gradients(zip(gradients, vars(discriminator_model).values()))

# Generator 최적화를 위한 function을 정의합니다.
@tf.function
def g_train_step(generator_model, discriminator_model, z):
  with tf.GradientTape() as gen_tape:
    G = generator_model(z)
    D_fake, D_fake_logits = discriminator_model(G)  # D(G(z))
    loss = generator_loss(D_fake_logits)
  gradients = gen_tape.gradient(loss, vars(generator_model).values())
  generator_optimizer.apply_gradients(zip(gradients, vars(generator_model).values()))

# 생성된 이미지들을 저장할 generated_output 폴더를 생성합니다.
num_img = 0
if not os.path.exists('generated_output/'):
  os.makedirs('generated_output/')

# num_epoch 횟수만큼 최적화를 수행합니다.
for i in range(num_epoch):
  # MNIST 이미지를 batch_size만큼 불러옵니다.
  batch_X = next(train_data_iter)

  # Latent Variable의 인풋으로 사용할 noise를 Uniform Distribution에서 batch_size 개수만큼 샘플링합니다.
  batch_noise = np.random.uniform(-1., 1., [batch_size, 100]).astype('float32')

  # 500번 반복할때마다 생성된 이미지를 저장합니다.
  if i % 500 == 0:
    samples = Generator_model(np.random.uniform(-1., 1., [64, 100]).astype('float32')).numpy()
    fig = plot(samples)
    plt.savefig('generated_output/%s.png' % str(num_img).zfill(3), bbox_inches='tight')
    num_img += 1
    plt.close(fig)

  # Discriminator 최적화를 수행하고 Discriminator의 손실함수를 return합니다.
  _, d_loss_print = d_train_step(Discriminator_model, batch_X, Generator_model(batch_noise)), discriminator_loss(Discriminator_model(batch_X)[1], Discriminator_model(Generator_model(batch_noise))[1])

  # Generator 최적화를 수행하고 Generator 손실함수를 return합니다.
  _, g_loss_print = g_train_step(Generator_model, Discriminator_model, batch_noise), generator_loss(Discriminator_model(Generator_model(batch_noise))[1])

  # 100번 반복할때마다 Discriminator의 손실함수와 Generator 손실함수를 출력합니다.
  if i % 100 == 0:
    print('반복(Epoch): %d, Generator 손실함수(g_loss): %f, Discriminator 손실함수(d_loss): %f' % (i, g_loss_print, d_loss_print))