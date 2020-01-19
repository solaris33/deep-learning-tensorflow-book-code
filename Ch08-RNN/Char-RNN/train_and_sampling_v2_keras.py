# -*- coding: utf-8 -*-
# Char-RNN 예제 - Keras API를 이용한 구현
# Reference : https://github.com/tensorflow/docs/blob/master/site/en/tutorials/text/text_generation.ipynb

from __future__ import absolute_import, division, print_function, unicode_literals

from absl import app
import tensorflow as tf

import numpy as np
import os
import time

# input 데이터와 input 데이터를 한글자씩 뒤로 민 target 데이터를 생성하는 utility 함수를 정의합니다.
def split_input_target(chunk):
  input_text = chunk[:-1]
  target_text = chunk[1:]

  return input_text, target_text

# 학습에 필요한 설정값들을 지정합니다.
data_dir = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')  # shakespeare
#data_dir = './data/linux/input.txt'  # linux
batch_size = 64      # Training : 64, Sampling : 1
seq_length = 100     # Training : 100, Sampling : 1
embedding_dim = 256  # Embedding 차원
hidden_size = 1024   # 히든 레이어의 노드 개수
num_epochs = 10

# 학습에 사용할 txt 파일을 읽습니다.
text = open(data_dir, 'rb').read().decode(encoding='utf-8')
# 학습데이터에 포함된 모든 character들을 나타내는 변수인 vocab과
# vocab에 id를 부여해 dict 형태로 만든 char2idx를 선언합니다.
vocab = sorted(set(text))  # 유니크한 character 개수
vocab_size = len(vocab)
print('{} unique characters'.format(vocab_size))
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# 학습 데이터를 character에서 integer로 변환합니다.
text_as_int = np.array([char2idx[c] for c in text])

# split_input_target 함수를 이용해서 input 데이터와 input 데이터를 한글자씩 뒤로 민 target 데이터를 생성합니다.
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
dataset = sequences.map(split_input_target)

# tf.data API를 이용해서 데이터를 섞고 batch 형태로 가져옵니다.
dataset = dataset.shuffle(10000).batch(batch_size, drop_remainder=True)

# tf.keras.Model을 이용해서 RNN 모델을 정의합니다.
class RNN(tf.keras.Model):
 def __init__(self, batch_size):
   super(RNN, self).__init__()
   self.embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                                    batch_input_shape=[batch_size, None])
   self.hidden_layer_1 = tf.keras.layers.LSTM(hidden_size,
                                             return_sequences=True,
                                             stateful=True,
                                             recurrent_initializer='glorot_uniform')
   self.output_layer = tf.keras.layers.Dense(vocab_size)

 def call(self, x):
   embedded_input = self.embedding_layer(x)
   features = self.hidden_layer_1(embedded_input)
   logits = self.output_layer(features)

   return logits

# sparse cross-entropy 손실 함수를 정의합니다.
def sparse_cross_entropy_loss(labels, logits):
  return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))

# 최적화를 위한 Adam 옵티마이저를 정의합니다.
optimizer = tf.keras.optimizers.Adam()

# 최적화를 위한 function을 정의합니다.
@tf.function
def train_step(model, input, target):
  with tf.GradientTape() as tape:
    logits = model(input)
    loss = sparse_cross_entropy_loss(target, logits)
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  return loss

def generate_text(model, start_string):
  num_sampling = 4000  # 생성할 글자(Character)의 개수를 지정합니다.

  # start_sting을 integer 형태로 변환합니다.
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # 샘플링 결과로 생성된 string을 저장할 배열을 초기화합니다.
  text_generated = []

  # 낮은 temperature 값은 더욱 정확한 텍스트를 생성합니다.
  # 높은 temperature 값은 더욱 다양한 텍스트를 생성합니다.
  temperature = 1.0

  # 여기서 batch size = 1 입니다.
  model.reset_states()
  for i in range(num_sampling):
    predictions = model(input_eval)
    # 불필요한 batch dimension을 삭제합니다.
    predictions = tf.squeeze(predictions, 0)

    # 모델의 예측결과에 기반해서 랜덤 샘플링을 하기위해 categorical distribution을 사용합니다.
    predictions = predictions / temperature
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

    # 예측된 character를 다음 input으로 사용합니다.
    input_eval = tf.expand_dims([predicted_id], 0)
    # 샘플링 결과를 text_generated 배열에 추가합니다.
    text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

def main(_):
  # Recurrent Neural Networks(RNN) 모델을 선언합니다.
  RNN_model = RNN(batch_size=batch_size)

  # 데이터 구조 파악을 위해서 예제로 임의의 하나의 배치 데이터 에측하고, 예측결과를 출력합니다.
  for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = RNN_model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

  # 모델 정보를 출력합니다.
  RNN_model.summary()

  # checkpoint 데이터를 저장할 경로를 지정합니다.
  checkpoint_dir = './training_checkpoints'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

  for epoch in range(num_epochs):
    start = time.time()

    # 매 반복마다 hidden state를 초기화합니다. (최초의 hidden 값은 None입니다.)
    hidden = RNN_model.reset_states()

    for (batch_n, (input, target)) in enumerate(dataset):
      loss = train_step(RNN_model, input, target)

      if batch_n % 100 == 0:
        template = 'Epoch {} Batch {} Loss {}'
        print(template.format(epoch+1, batch_n, loss))

    # 5회 반복마다 파라미터를 checkpoint로 저장합니다.
    if (epoch + 1) % 5 == 0:
      RNN_model.save_weights(checkpoint_prefix.format(epoch=epoch))

    print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

  RNN_model.save_weights(checkpoint_prefix.format(epoch=epoch))
  print("트레이닝이 끝났습니다!")

  sampling_RNN_model = RNN(batch_size=1)
  sampling_RNN_model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
  sampling_RNN_model.build(tf.TensorShape([1, None]))
  sampling_RNN_model.summary()

  # 샘플링을 시작합니다.
  print("샘플링을 시작합니다!")
  print(generate_text(sampling_RNN_model, start_string=u' '))

if __name__ == '__main__':
  # main 함수를 호출합니다.
  app.run(main)