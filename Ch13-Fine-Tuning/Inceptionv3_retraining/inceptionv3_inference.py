# -*- coding: utf-8 -*-

"""Inception v3 retrain을 통해 만든 나만의 분류기를 이용한 Inference"""

import numpy as np
import tensorflow as tf

image_path = '/tmp/test_chartreux.jpg'                                       # 추론을 진행할 이미지 파일경로
graph_pb_file_path = '/tmp/output_graph.pb'                                  # 읽어들일 graph 파일 경로
labels_txt_file_path = '/tmp/output_labels.txt'                              # 읽어들일 labels 파일 경로

# 저장된 output_graph.pb파일을 읽어서 그래프를 생성합니다.
def create_graph():
  with tf.gfile.FastGFile(graph_pb_file_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

# 이미지에 대한 추론(Inference)을 진행합니다.
def run_inference_on_image():
  answer = None

  # 만약 경로에 이미지 파일이 없을 경우 오류 로그를 출력합니다.
  if not tf.gfile.Exists(image_path):
    tf.logging.fatal('추론할 이미지 파일이 존재하지 않습니다. %s', image_path)
    return answer

  # 이미지 파일을 읽습니다.
  image_data = tf.gfile.FastGFile(image_path, 'rb').read()

  # 그래프를 생성합니다.
  create_graph()

  # 세션을 열고 그래프를 실행합니다.
  with tf.Session() as sess:
    # 최종 소프트 맥스 출력 레이어를 지정합니다.
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    # 추론할 이미지를 인풋으로 넣고 추론 결과인 소프트 맥스 행렬을 리턴 받습니다. 
    predictions = sess.run(softmax_tensor, feed_dict={'DecodeJpeg/contents:0': image_data})
    # 불필요한 차원을 제거합니다.
    predictions = np.squeeze(predictions)

    # 가장 높은 확률을 가진 5개(top 5)의 예측값(predictions)들의 인덱스를 가져옵니다.
    # e.g. [0 3 2 4 1]]
    top_k = predictions.argsort()[-5:][::-1]

    # output_labels.txt 파일로부터 정답 레이블들을 list 형태로 가져옵니다.
    f = open(labels_txt_file_path, 'rb')
    lines = f.readlines()
    labels = [str(w).replace("\n", "") for w in lines]

    # 가장 높은 확률을 가진 인덱스들부터 추론 결과(Top-5)를 출력합니다.
    print("Top-5 추론 결과:")
    for node_id in top_k:
      label_name = labels[node_id]
      probability = predictions[node_id]
      print('%s (확률 = %.5f)' % (label_name, probability))

    # 가장 높은 확류을 가진 Top-1 추론 결과를 출력합니다.
    print("\nTop-1 추론 결과:")
    answer = labels[top_k[0]]
    probability = predictions[top_k[0]]
    print('%s (확률 = %.5f)' % (answer, probability))

if __name__ == '__main__':
  run_inference_on_image()