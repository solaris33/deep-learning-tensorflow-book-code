# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

# 2개의 값을 더하는 function을 정의합니다.
@tf.function
def add_two_values(x, y):
  return x + y

# 세션을 열고 그래프를 실행합니다.
# 출력값 :
# 7.5
# [ 3.  7.]
print(add_two_values(3, 4.5).numpy())
print(add_two_values(np.array([1, 3]), np.array([2, 4])).numpy())

# 노드를 추가해서 더 복잡한 그래프 형태를 만들어봅시다.
@tf.function
def add_two_values_and_multiply_three(x, y):
  return 3 * add_two_values(x, y)

# 출력값 : 22.5
print(add_two_values_and_multiply_three(3, 4.5).numpy())