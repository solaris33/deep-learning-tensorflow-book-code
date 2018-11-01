# -*- coding: utf-8 -*-

import numpy as np
import os
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob

import TensorflowUtils as utils

# MIT Scene Parsing 데이터를 다운로드 받을 경로
DATA_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'

# 다운받은 MIT Scene Parsing 데이터를 읽습니다.
def read_dataset(data_dir):
  pickle_filename = "MITSceneParsing.pickle"
  pickle_filepath = os.path.join(data_dir, pickle_filename)
  # MITSceneParsing.pickle 파일이 없으면 다운 받은 MITSceneParsing 데이터를 pickle 파일로 저장합니다.
  if not os.path.exists(pickle_filepath):
    utils.maybe_download_and_extract(data_dir, DATA_URL, is_zipfile=True)
    SceneParsing_folder = os.path.splitext(DATA_URL.split("/")[-1])[0]
    result = create_image_lists(os.path.join(data_dir, SceneParsing_folder))
    print ("Pickling ...")
    with open(pickle_filepath, 'wb') as f:
      pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
  else:
    print ("Found pickle file!")

  # 데이터가 저장된 pickle 파일을 읽고 데이터를 training 데이터와 validation 데이터로 분리합니다.
  with open(pickle_filepath, 'rb') as f:
    result = pickle.load(f)
    training_records = result['training']
    validation_records = result['validation']
    del result

  return training_records, validation_records

# training 폴더와 validation 폴더에서 
# raw 인풋이미지(.jpg)와 annotaion된 타겟이미지(.png)를 읽어서 리스트 형태로 만들어 리턴합니다.
def create_image_lists(image_dir):
  if not gfile.Exists(image_dir):
    print("Image directory '" + image_dir + "' not found.")
    return None
  directories = ['training', 'validation']
  image_list = {}

  for directory in directories:
    file_list = []
    image_list[directory] = []
    file_glob = os.path.join(image_dir, "images", directory, '*.' + 'jpg')
    file_list.extend(glob.glob(file_glob))

    if not file_list:
      print('No files found')
    else:
      for f in file_list:
        filename = os.path.splitext(f.split("/")[-1])[0]
        annotation_file = os.path.join(image_dir, "annotations", directory, filename + '.png')
        if os.path.exists(annotation_file):
          record = {'image': f, 'annotation': annotation_file, 'filename': filename}
          image_list[directory].append(record)
        else:
          print("Annotation file not found for %s - Skipping" % filename)

    random.shuffle(image_list[directory])
    no_of_images = len(image_list[directory])
    print ('No. of %s files: %d' % (directory, no_of_images))

  return image_list