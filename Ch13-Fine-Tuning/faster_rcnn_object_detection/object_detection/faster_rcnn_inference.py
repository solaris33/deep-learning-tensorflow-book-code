# -*- coding: utf-8 -*-

# 필요한 라이브러리들을 임포트합니다.
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# 현재 디렉토리를 모듈 경로로 추가합니다.
sys.path.append("..")

# 물체인식을 위한 유틸리티 함수들을 임포트합니다.
from utils import label_map_util
from utils import visualization_utils as vis_util

# 물체 인식을 위해 사용할 Pre-Trained Faster R-CNN 모델을 다운받기 위한 URL을 설정합니다. 다운 가능한 전체 Pre-Trained 모델은 아래 URL에서 확인하실 수 있습니다.
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
MODEL_NAME = 'faster_rcnn_resnet50_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# 다운받은 학습이 완료된(Frozen) Object Detection을 위한 그래프 pb 파일 경로를 지정합니다.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# 추론에 사용할 레이블들이 저장된 pbtxt 파일의 경로를 지정합니다.
# 기본값인 mscoco_label_map.pbtxt는 90개의 레이블로 구성되어 있습니다.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# 지정한 URL에서 Pre-Trained Faster R-CNN 모델을 다운받고 압축을 해제합니다.
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())

# 다운받은 그래프가 저장된 pb파일을 읽어서 그래프를 생성합니다.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# 레이블별로 id가 할당된 레이블맵을 불러옵니다.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# 이미지를 numpy 형태로 읽어오기 위한 유틸리티 함수를 정의합니다.
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
    (im_height, im_width, 3)).astype(np.uint8)


# test_images 폴더에 있는 기본 테스트용 이미지 2개(image1.jpg, imge2.jpg)를 불러옵니다.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# inch로 표현한 출력 이미지의 크기
IMAGE_SIZE = (12, 8)

# 불러온 테스트 이미지들을 하나씩 화면에 출력합니다.
# 다음으로 넘어가기 위해서는 아무 버튼을 한번 입력해야합니다.(plt.waitforbuttonpress())
for image_path in TEST_IMAGE_PATHS:
  image = Image.open(image_path)    
  plt.figure(figsize=IMAGE_SIZE)
  plt.imshow(image)
  plt.draw()
  plt.waitforbuttonpress()

# 불러온 이미지에 대해서 물체인식 추론(Inference)을 진행하고 추론 결과를 화면에 출력합니다.
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # 이미지를 입력받을 인풋 플레이스홀더를 지정합니다.
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # 물체 인식 결과인 Bounding Box들을 리턴받을 변수를 지정합니다.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # 추론 결과에 대한 확신의 정도를 리턴받을 scores텐서와 labels에 대한 추론 결과를 리턴받을 classes 변수를 지정합니다.
    # 출력결과 : 바운딩 박스 + 레이블 + 확신의 정도(score)
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    for image_path in TEST_IMAGE_PATHS:
      image = Image.open(image_path)
      # 입력받은 이미지를 numpy 형태의 배열로 변환합니다.
      image_np = load_image_into_numpy_array(image)
      # 모델의 인풋인 [1, None, None, 3] 형태로 차원을 확장(Expand)합니다. 
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # 인풋 이미지를 플레이스 홀더에 넣고 추론을 지행합니다.
      (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
      # 추론 결과를 화면에 출력합니다.
      # 출력결과 : 바운딩 박스 + 레이블 + 확신의 정도(score)
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      plt.figure(figsize=IMAGE_SIZE)
      plt.imshow(image_np)
      plt.draw()
      plt.waitforbuttonpress()

# 나만의 이미지로 추론을 진행해봅시다.
image = Image.open('test_chartreux.jpg')    
plt.figure(figsize=IMAGE_SIZE)
plt.imshow(image)
plt.draw()
plt.waitforbuttonpress()

# 불러온 이미지에 대해서 물체인식 추론(Inference)을 진행하고 추론 결과를 화면에 출력합니다.
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # 이미지를 입력받을 인풋 플레이스홀더를 지정합니다.
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # 물체 인식 결과인 Bounding Box들을 리턴받을 변수를 지정합니다.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # 추론 결과에 대한 확신의 정도를 리턴받을 scores텐서와 labels에 대한 추론 결과를 리턴받을 classes 변수를 지정합니다.
    # 출력결과 : 바운딩 박스 + 레이블 + 확신의 정도(score)
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # 입력받은 이미지를 numpy 형태의 배열로 변환합니다.
    image_np = load_image_into_numpy_array(image)
    # 모델의 인풋인 [1, None, None, 3] 형태로 차원을 확장(Expand)합니다. 
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # 인풋 이미지를 플레이스 홀더에 넣고 추론을 진행합니다.
    (boxes, scores, classes, num) = sess.run(
      [detection_boxes, detection_scores, detection_classes, num_detections],
      feed_dict={image_tensor: image_np_expanded})
    # 추론 결과를 화면에 출력합니다.
    # 출력결과 : 바운딩 박스 + 레이블 + 확신의 정도(score)
    vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      np.squeeze(boxes),
      np.squeeze(classes).astype(np.int32),
      np.squeeze(scores),
      category_index,
      use_normalized_coordinates=True,
      line_thickness=8)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)
    plt.draw()
    plt.waitforbuttonpress()
    # 추론 결과를 파일로 저장합니다.
    plt.savefig("test_chartreux_result.jpg")