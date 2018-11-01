# -*- coding: utf-8 -*-
# 텐서플로우 설치 체크 - 텐서플로우가 제대로 설치되었는지 체크해봅시다.

import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')

sess = tf.Session()
print(sess.run(hello)) 
#'Hello, TensorFlow!'

a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a + b))
#42

sess.close()