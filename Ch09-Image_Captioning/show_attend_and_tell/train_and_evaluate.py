# -*- coding: utf-8 -*-
# Image Captioning 예제 - Keras API를 이용한 Show, attend and tell 모델 구현
# Reference : https://github.com/tensorflow/docs/blob/master/site/en/tutorials/text/image_captioning.ipynb

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import numpy as np
import time

from absl import app
from absl import flags

from model import BahdanauAttention, CNN_Encoder, RNN_Decoder
from utils import load_image, calc_max_length, plot_attention
from data_utils import cache_bottlenecks, maybe_download_and_extract, prepare_image_and_caption_data

flags.DEFINE_bool(
    'do_caching',
    default=True,
    help='Do bottleneck caching')

FLAGS = flags.FLAGS

# 학습을 위한 설정값들을 지정합니다.
num_examples = 30000
BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
top_k = 5000
vocab_size = top_k + 1
attention_features_shape = 64
EPOCHS = 20

# 빠른 학습을 위해서 shuffle된 set에서 처음 30000개만을 선택해서 데이터를 불러옵니다.
train_captions, img_name_vector = prepare_image_and_caption_data(num_examples)

# sparse cross-entropy 손실 함수를 정의합니다.
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

# 최적화를 위한 Adam 옵티마이저를 정의합니다.
optimizer = tf.keras.optimizers.Adam()

# 최적화를 위한 function을 정의합니다.
@tf.function
def train_step(img_tensor, target, tokenizer, encoder, decoder):
  loss = 0

  # 매 batch마다 hidden state를 0으로 초기화합니다.
  hidden = decoder.reset_state(batch_size=target.shape[0])

  # <start>로 decoding 문장을 시작합니다.
  dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

  with tf.GradientTape() as tape:
    features = encoder(img_tensor)

    for i in range(1, target.shape[1]):
      # feature를 decoder의 input으로 넣습니다.
      predictions, hidden, _ = decoder(dec_input, features, hidden)

      loss += loss_function(target[:, i], predictions)

      # teacher forcing 방식으로 학습을 진행합니다.
      dec_input = tf.expand_dims(target[:, i], 1)

  total_loss = (loss / int(target.shape[1]))

  trainable_variables = encoder.trainable_variables + decoder.trainable_variables
  gradients = tape.gradient(loss, trainable_variables)
  optimizer.apply_gradients(zip(gradients, trainable_variables))

  return loss, total_loss

# evaluation을 위한 function을 정의합니다.
def evaluate(image, max_length, attention_features_shape, encoder, decoder, image_features_extract_model, tokenizer):
  attention_plot = np.zeros((max_length, attention_features_shape))

  hidden = decoder.reset_state(batch_size=1)

  temp_input = tf.expand_dims(load_image(image)[0], 0)
  img_tensor_val = image_features_extract_model(temp_input)
  img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

  features = encoder(img_tensor_val)

  dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
  result = []

  for i in range(max_length):
    predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

    attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

    predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
    result.append(tokenizer.index_word[predicted_id])

    if tokenizer.index_word[predicted_id] == '<end>':
      return result, attention_plot

    dec_input = tf.expand_dims([predicted_id], 0)

  attention_plot = attention_plot[:len(result), :]

  return result, attention_plot

def main(_):
  # Imagenet 데이터셋에 대해 Pre-train된 Inception v3 모델의 Weight를 불러오고,
  # Softmax Layer 한칸 앞에서 8x8x2048 형태의 Feature map을 추출하는 hidden layer를 output으로 하는
  # image_features_extract_model을 tf.keras.Model을 이용해서 선언합니다.
  image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
  new_input = image_model.input
  hidden_layer = image_model.layers[-1].output

  image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

  # do_caching flag가 True일 경우 이미지들에 대한 bottleneck caching을 수행합니다.
  if FLAGS.do_caching == True:
    cache_bottlenecks(img_name_vector, image_features_extract_model)
  else:
    print('Already bottleneck cached !')

  # 가장 빈도수가 높은 5000개의 단어를 선택해서 Vocabulary set을 만들고,
  # Vocabulary set에 속하지 않은 단어들은 <unk>로 지정합니다.
  tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                    oov_token="<unk>",
                                                    filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
  tokenizer.fit_on_texts(train_captions)
  # 가장 긴 문장보다 작은 문장들은 나머지 부분은 <pad>로 padding합니다.
  tokenizer.word_index['<pad>'] = 0
  tokenizer.index_word[0] = '<pad>'

  # caption 문장을 띄어쓰기 단위로 split해서 tokenize 합니다.
  train_seqs = tokenizer.texts_to_sequences(train_captions)
  # 길이가 짧은 문장들에 대한 padding을 진행합니다.
  cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
  # attetion weights를 위해서 가장 긴 문장의 길이를 저장합니다.
  max_length = calc_max_length(train_seqs)

  # 데이터의 80%를 training 데이터로, 20%를 validation 데이터로 split합니다.
  img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                      cap_vector,
                                                                      test_size=0.2,
                                                                      random_state=0)

  print('train image size:', len(img_name_train), 'train caption size:', len(cap_train))
  print('validation image size:',len(img_name_val), 'validation caption size:', len(cap_val))

  num_steps = len(img_name_train) // BATCH_SIZE

  # disk에 caching 해놓은 numpy 파일들을 읽습니다.
  def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8') + '.npy')
    return img_tensor, cap

  dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
  # numpy 파일들을 병렬적(parallel)으로 불러옵니다.
  dataset = dataset.map(lambda item1, item2: tf.numpy_function(
            map_func, [item1, item2], [tf.float32, tf.int32]),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

  # tf.data API를 이용해서 데이터를 섞고(shuffle) batch 개수(=64)로 묶습니다.
  dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  # encoder와 decoder를 선언합니다.
  encoder = CNN_Encoder(embedding_dim)
  decoder = RNN_Decoder(embedding_dim, units, vocab_size)

  # checkpoint 데이터를 저장할 경로를 지정합니다.
  checkpoint_path = "./checkpoints/train"
  ckpt = tf.train.Checkpoint(encoder=encoder,
                             decoder=decoder,
                             optimizer = optimizer)
  ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

  start_epoch = 0
  if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    # checkpoint_path에서 가장 최근의 checkpoint를 restore합니다.
    ckpt.restore(ckpt_manager.latest_checkpoint)

  loss_plot = []

  # 지정된 epoch 횟수만큼 optimization을 진행합니다.
  for epoch in range(start_epoch+1, EPOCHS+1):
    start = time.time()
    total_loss = 0

    for (batch, (img_tensor, target)) in enumerate(dataset):
      batch_loss, t_loss = train_step(img_tensor, target, tokenizer, encoder, decoder)
      total_loss += t_loss

      if batch % 100 == 0:
        print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch, batch, batch_loss.numpy() / int(target.shape[1])))
    # 추후에 plot을 위해서 epoch별 loss값을 저장합니다.
    loss_plot.append(total_loss / num_steps)

    # 5회 반복마다 파라미터값을 저장합니다.
    if epoch % 5 == 0:
      ckpt_manager.save(checkpoint_number=epoch)

    print ('Epoch {} Loss {:.6f}'.format(epoch, total_loss/num_steps))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

  print('Training Finished !')
  plt.plot(loss_plot)
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Loss Plot')
  plt.savefig('Loss plot.png')
  plt.show()

  # validation set에서 random하게 1장의 이미지를 뽑아 해당 이미지에 대한 captioning을 진행합니다.
  rid = np.random.randint(0, len(img_name_val))
  image = img_name_val[rid]
  real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
  result, attention_plot = evaluate(image, max_length, attention_features_shape, encoder, decoder, image_features_extract_model, tokenizer)

  print ('Real Caption:', real_caption)
  print ('Prediction Caption:', ' '.join(result))
  plot_attention(image, result, attention_plot)

  # test를 위해서 surfer 이미지 한장을 다운받은뒤, 해당 이미지에 대한 captioning을 진행해봅니다.
  image_url = 'https://tensorflow.org/images/surf.jpg'
  image_extension = image_url[-4:]
  image_path = tf.keras.utils.get_file('image' + image_extension, origin=image_url)

  result, attention_plot = evaluate(image_path, max_length, attention_features_shape, encoder, decoder, image_features_extract_model, tokenizer)
  print('Prediction Caption:', ' '.join(result))
  plot_attention(image_path, result, attention_plot)

if __name__ == '__main__':
  # main 함수를 호출합니다.
  app.run(main)