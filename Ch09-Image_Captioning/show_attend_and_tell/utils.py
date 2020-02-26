# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# Inception v3의 input에 적합한 형태로 image_path 경로에서 이미지를 불러옵니다.
def load_image(image_path):
  img = tf.io.read_file(image_path)
  img = tf.image.decode_jpeg(img, channels=3)
  img = tf.image.resize(img, (299, 299))
  img = tf.keras.applications.inception_v3.preprocess_input(img)
  return img, image_path

# 전체 dataset에 존재하는 caption의 maximum length를 찾습니다.
def calc_max_length(tensor):
  return max(len(t) for t in tensor)

# attention 결과를 시각화합니다.
def plot_attention(image, result, attention_plot):
  temp_image = np.array(Image.open(image))

  fig = plt.figure(figsize=(10, 10))

  len_result = len(result)
  for l in range(len_result):
    temp_att = np.resize(attention_plot[l], (8, 8))
    ax = fig.add_subplot(len_result // 2, len_result // 2, l + 1)
    ax.set_title(result[l])
    img = ax.imshow(temp_image)
    ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

  plt.tight_layout()
  plt.savefig(image.split(os.path.sep)[-1].split('.')[-2] + ' attention' + '.png')
  plt.show()