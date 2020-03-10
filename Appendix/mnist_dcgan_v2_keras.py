# -*- coding: utf-8 -*-
# DCGAN(Deep Convolutional Generative Adversarial Network) 예제 - Keras API를 이용한 구현
# Reference : https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/dcgan.ipynb

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import glob
import imageio
import matplotlib.pyplot as plt
import os
import time

def generate_and_save_images(model, epoch, test_input):
  # batch normalization를 inference mode 모드로 실행하기 위해서 is_training을 False로 지정합니다.
  predictions = model(test_input, is_training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i+1)
    plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
    plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.close()

def make_and_save_gif_image(anim_file):
  with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)
    last = -1
    for i, filename in enumerate(filenames):
      frame = 2 * (i ** 0.5)
      if round(frame) > round(last):
        last = frame
      else:
        continue
      image = imageio.imread(filename)
      writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)

# MNIST 데이터를 다운로드 합니다.
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
# 이미지들을 float32 데이터 타입으로 변경하고 (28, 28, 1) 크기로 reshape합니다.
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
# 이미지를 [-1, 1] 사이 값으로 Normalize합니다.
train_images = (train_images - 127.5) / 127.5

# 학습을 위한 설정값들을 지정합니다.
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16
BUFFER_SIZE = 60000
BATCH_SIZE = 256

# tf.data API를 이용해서 데이터를 섞고 batch 형태로 가져옵니다.
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# tf.keras.Model을 이용해서 Generator 모델을 정의합니다.
class Generator(tf.keras.Model):
  def __init__(self):
    super(Generator, self).__init__()
    self.hidden_layer_1 = tf.keras.layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,))
    self.batch_norm_layer_1 = tf.keras.layers.BatchNormalization()
    self.hidden_layer_2 = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)
    self.batch_norm_layer_2 = tf.keras.layers.BatchNormalization()
    self.hidden_layer_3 = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)
    self.batch_norm_layer_3 = tf.keras.layers.BatchNormalization()
    self.output_layer = tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')

  def call(self, x, is_training):
    H1_output = self.hidden_layer_1(x)
    H1_output = self.batch_norm_layer_1(H1_output, training=is_training)
    H1_output = tf.keras.layers.LeakyReLU()(H1_output)
    H1_output = tf.keras.layers.Reshape((7, 7, 256))(H1_output)
    assert H1_output.shape == (x.shape[0], 7, 7, 256)

    H2_output = self.hidden_layer_2(H1_output)
    assert H2_output.shape == (x.shape[0], 7, 7, 128)
    H2_output = self.batch_norm_layer_2(H2_output, training=is_training)
    H2_output = tf.keras.layers.LeakyReLU()(H2_output)

    H3_output = self.hidden_layer_3(H2_output)
    assert H3_output.shape == (x.shape[0], 14, 14, 64)
    H3_output = self.batch_norm_layer_3(H3_output, training=is_training)
    H3_output = tf.keras.layers.LeakyReLU()(H3_output)

    generated_image = self.output_layer(H3_output)
    assert generated_image.shape == (x.shape[0], 28, 28, 1)

    return generated_image

# tf.keras.Model을 이용해서 Discriminator 모델을 정의합니다.
class Discriminator(tf.keras.Model):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.hidden_layer_1 = tf.keras.layers.Conv2D(64,
                                                 (5, 5),
                                                 strides=(2, 2),
                                                 padding='same',
                                                 input_shape=[28, 28, 1])
    self.dropout_layer_1 = tf.keras.layers.Dropout(0.3)
    self.hidden_layer_2 = tf.keras.layers.Conv2D(128,
                                                 (5, 5),
                                                 strides=(2, 2),
                                                 padding='same')
    self.dropout_layer_2 = tf.keras.layers.Dropout(0.3)
    self.flatten_layer = tf.keras.layers.Flatten()
    self.output_layer = tf.keras.layers.Dense(1)

  def call(self, x, is_training):
    H1_output = self.hidden_layer_1(x)
    H1_output = tf.keras.layers.LeakyReLU()(H1_output)
    H1_output = self.dropout_layer_1(H1_output, training=is_training)

    H2_output = self.hidden_layer_2(H1_output)
    H2_output = tf.keras.layers.LeakyReLU()(H2_output)
    H2_output = self.dropout_layer_2(H2_output, training=is_training)

    H2_output_flat = self.flatten_layer(H2_output)

    logits = self.output_layer(H2_output_flat)

    return logits

# 생성자(Generator) 모델을 선언합니다.
Generator_model = Generator()
# 구분자(Discriminator) 모델을 선언합니다.
Discriminator_model = Discriminator()

# cross entropy loss 계산을 위한 helper function을 정의합니다.
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Discriminator의 손실 함수를 정의합니다.
def discriminator_loss(real_output, fake_output):
  real_loss = cross_entropy(tf.ones_like(real_output), real_output)
  fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
  total_loss = real_loss + fake_loss

  return total_loss

# Generator의 손실 함수를 정의합니다.
def generator_loss(fake_output):
  return cross_entropy(tf.ones_like(fake_output), fake_output)

# Discriminator와 Generator를 위한 Optimizer들을 정의합니다.
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(generator_model, discriminator_model, images):
  noise = tf.random.normal([BATCH_SIZE, noise_dim])

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator_model(noise, is_training=True)

    real_output = discriminator_model(images, is_training=True)
    fake_output = discriminator_model(generated_images, is_training=True)

    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)

  gradients_of_generator = gen_tape.gradient(gen_loss, generator_model.trainable_variables)
  gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator_model.trainable_variables)

  generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_model.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator_model.trainable_variables))

# tf.train.Checkpoint API를 이용해서 파라미터를 저장합니다.
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=Generator_model,
                                 discriminator=Discriminator_model)

# epoch별 개선결과를 한눈에 확인하기 위해서 동일한 latent variable값을 seed로 사용합니다.
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# EPOCHS 횟수만큼 최적화를 수행합니다.
for epoch in range(EPOCHS):
  start = time.time()

  for image_batch in train_dataset:
    train_step(Generator_model, Discriminator_model, image_batch)

  generate_and_save_images(Generator_model,
                           epoch + 1,
                           seed)

  # 15회 반복마다 파라미터값을 저장합니다.
  if (epoch + 1) % 15 == 0:
    checkpoint.save(file_prefix = checkpoint_prefix)

  print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

# epoch별 Generator에 의해 생성된 이미지를 gif로 저장합니다.
make_and_save_gif_image('dcgan.gif')