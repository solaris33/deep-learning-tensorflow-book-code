# -*- coding: utf-8 -*-

import codecs
import os
import collections
from six.moves import cPickle
import numpy as np


class TextLoader():
  def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.encoding = encoding

    input_file = os.path.join(data_dir, "input.txt")
    vocab_file = os.path.join(data_dir, "vocab.pkl")
    tensor_file = os.path.join(data_dir, "data.npy")

    # 전처리된 파일들("vocab.pkl", "data.npy")이 이미 존재하면 이를 불러오고 없으면 데이터 전처리를 진행합니다.
    if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
      print("reading text file")
      self.preprocess(input_file, vocab_file, tensor_file)
    else:
      print("loading preprocessed files")
      self.load_preprocessed(vocab_file, tensor_file)
      # 배치를 생성하고 배치 포인터를 배치의 시작지점으로 리셋합니다.
      self.create_batches()
      self.reset_batch_pointer()

  # 데이터 전처리를 진행합니다.
  def preprocess(self, input_file, vocab_file, tensor_file):
    with codecs.open(input_file, "r", encoding=self.encoding) as f:
      data = f.read()
      # 데이터에서 문자(character)별 등장횟수를 셉니다.
      counter = collections.Counter(data)
      count_pairs = sorted(counter.items(), key=lambda x: -x[1])
      self.chars, _ = zip(*count_pairs) # 전체 문자들(Chracters)
      self.vocab_size = len(self.chars) # 전체 문자(단어) 개수
      self.vocab = dict(zip(self.chars, range(len(self.chars)))) # 단어들을 (charcter, id) 형태의 dictionary로 만듭니다.
      # vocab dictionary를 "vocab.pkl" 파일로 저장합니다.
      with open(vocab_file, 'wb') as f:
        cPickle.dump(self.chars, f)
      # 데이터의 각각의 character들을 id로 변경합니다.
      self.tensor = np.array(list(map(self.vocab.get, data)))
      # id로 변경한 데이터를 "data.npy" binary numpy 파일로 저장합니다.
      np.save(tensor_file, self.tensor)

  # 전처리한 데이터가 파일로 저장되어 있다면 파일로부터 전처리된 정보들을 읽어옵니다.
  def load_preprocessed(self, vocab_file, tensor_file):
    with open(vocab_file, 'rb') as f:
      self.chars = cPickle.load(f)
      self.vocab_size = len(self.chars)
      self.vocab = dict(zip(self.chars, range(len(self.chars))))
      self.tensor = np.load(tensor_file)
      self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length))

  # 전체 데이터를 배치 단위로 묶습니다.
  def create_batches(self):
    self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length))

    # 데이터 양이 너무 적어서 1개의 배치도 만들수없을 경우, 에러 메세지를 출력합니다.
    if self.num_batches == 0:
      assert False, "Not enough data. Make seq_length and batch_size small."

    # 배치에 필요한 정수만큼의 데이터만을 불러옵니다. e.g. 1115394 -> 1115000
    self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
    xdata = self.tensor
    ydata = np.copy(self.tensor)
    # 타겟 데이터는 인풋 데이터를 한칸 뒤로 민 형태로 구성합니다.
    ydata[:-1] = xdata[1:]
    ydata[-1] = xdata[0]
    # batch_size 크기의 배치를 num_batches 개수 만큼 생성합니다. 
    self.x_batches = np.split(xdata.reshape(self.batch_size, -1),
      self.num_batches, 1)
    self.y_batches = np.split(ydata.reshape(self.batch_size, -1),
      self.num_batches, 1)

  # 다음 배치롤 불러오고 배치 포인터를 1만큼 증가시킵니다.  
  def next_batch(self):
    x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
    self.pointer += 1
    return x, y

  # 배치의 시작점을 데이터의 시작지점으로 리셋합니다.
  def reset_batch_pointer(self):
    self.pointer = 0