# -*- coding: utf-8 -*-
# Embedding & tf.nn.embedding_lookup 예제

import tensorflow as tf
import numpy as np

vocab_size = 100			# one-hot encoding된 vocab 크기
embedding_size = 25		# embedding된 vocab 크기

# 인풋데이터를 받기 위한 플레이스홀더를 선언합니다.
inputs = tf.placeholder(tf.int32, shape=[None])
 
# 인풋데이터를 변환하기 위한 Embedding Matrix(100x25)를 선언합니다.  
embedding = tf.Variable(tf.random_normal([vocab_size, embedding_size]), dtype=tf.float32)
# tf.nn.embedding_lookup :
# int32나 int64 형태의 스칼라 형태의 인풋데이터를 vocab 사이즈만큼의 ebmedding된 vector로 변환합니다. 
embedded_inputs = tf.nn.embedding_lookup(embedding, inputs)

# 세션을 열고 모든 변수에 초기값을 할당합니다.
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# tf.nn.embedding_lookup 테스트 케이스 1
input_data = np.array([7])
print("Embedding 전 인풋데이터 : ")
# shape : [1, 100]
print(sess.run(tf.one_hot(input_data, vocab_size)))
print(tf.one_hot(input_data, vocab_size).shape)
print("Embedding 결과 : ")
# shape : [1, 25]
print(sess.run([embedded_inputs], feed_dict={inputs : input_data}))
print(sess.run([embedded_inputs], feed_dict={inputs : input_data})[0].shape)	# embedding된 차원을 출력합니다.


# tf.nn.embedding_lookup 테스트 케이스 2
input_data = np.array([7, 11, 67, 42, 21])
print("Embedding 전 인풋데이터 : ")
# shape : [5, 100]
print(sess.run(tf.one_hot(input_data, vocab_size)))
print(tf.one_hot(input_data, vocab_size).shape)
print("Embedding 결과 : ")
# shape : [5, 25]
print(sess.run([embedded_inputs], feed_dict={inputs : input_data}))
print(sess.run([embedded_inputs], feed_dict={inputs : input_data})[0].shape)	# embedding된 차원을 출력합니다.