# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import os
import json
from sklearn.utils import shuffle
from tqdm import tqdm
from utils import load_image

def cache_bottlenecks(img_name_vector, image_features_extract_model):
  # unique한 image name 집합을 만듭니다.
  encode_train = sorted(set(img_name_vector))

  # tf.data API를 이용해서 이미지를 batch 개수(=16)만큼 불러옵니다.
  image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
  image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

  # 동일 이미지에 대한 feature map 변환 연산을 반복수행하는 부분을 제거하기 위해서
  # 한번 feature map 형태로 변환한 값들을 disk에 저장해서 caching합니다.
  for img, path in tqdm(image_dataset):
    batch_features = image_features_extract_model(img)
    # 16x8x8x2048 이미지를 16x64x2048 형태로 reshape합니다.
    batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))

    for bf, p in zip(batch_features, path):
      path_of_feature = p.numpy().decode("utf-8")
      np.save(path_of_feature, bf.numpy())

def maybe_download_and_extract():
  # Caption annotation 압축파일을 다운받고, annotations 폴더에 압축을 풉니다.
  annotation_folder = '/annotations/'
  if not os.path.exists(os.path.abspath('.') + annotation_folder):
    annotation_zip = tf.keras.utils.get_file('captions.zip',
                                            cache_subdir=os.path.abspath('.'),
                                            origin = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                            extract = True)
    annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_train2014.json'
    os.remove(annotation_zip)
  else:
    annotation_file = os.path.abspath('.') + annotation_folder + 'captions_train2014.json'

  # 이미지 압축파일을 다운받고, train2014 폴더에 압축을 풉니다.
  image_folder = '/train2014/'
  if not os.path.exists(os.path.abspath('.') + image_folder):
    image_zip = tf.keras.utils.get_file('train2014.zip',
                                        cache_subdir=os.path.abspath('.'),
                                        origin = 'http://images.cocodataset.org/zips/train2014.zip',
                                        extract = True)
    PATH = os.path.dirname(image_zip) + image_folder
    os.remove(image_zip)
  else:
    PATH = os.path.abspath('.') + image_folder

  return annotation_file, PATH

def prepare_image_and_caption_data(num_examples=30000):
  annotation_file, PATH = maybe_download_and_extract()

  # annotation json 파일을 읽습니다.
  with open(annotation_file, 'r') as f:
    annotations = json.load(f)

  # caption과 image name을 vector로 저장합니다.
  all_captions = []
  all_img_name_vector = []

  for annot in annotations['annotations']:
    caption = '<start> ' + annot['caption'] + ' <end>'
    image_id = annot['image_id']
    full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)

    all_img_name_vector.append(full_coco_image_path)
    all_captions.append(caption)

  # caption과 image name들을 섞습니다.(shuffle)
  train_captions, img_name_vector = shuffle(all_captions,
                                            all_img_name_vector,
                                            random_state=1)

  # 빠른 학습을 위해서 shuffle된 set에서 처음부터 시작해서 num_examples 개수만큼만 선택합니다.
  train_captions = train_captions[:num_examples]
  img_name_vector = img_name_vector[:num_examples]

  print('selected samples :', len(train_captions))
  print('all samples :', len(all_captions))

  return train_captions, img_name_vector