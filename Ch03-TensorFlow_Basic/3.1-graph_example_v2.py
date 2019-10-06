# -*- coding: utf-8 -*-

import tensorflow as tf

# 그래프 노드를 정의하고 출력합니다.
# 출력값 : <tf.Tensor: id=0, shape=(), dtype=float32, numpy=3.0>, <tf.Tensor: id=1, shape=(), dtype=float32, numpy=4.0>
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # 암시적으로 tf.float32 타입으로 선언될 것입니다.
print(node1, node2)

# 그래프를 실행합니다.
# 출력값 : [3.0, 4.0]
print(node1.numpy(), node2.numpy())

# 두개의 노드의 값을 더하는 연산을 수행하는 node3을 정의합니다.
# 출력값:
# node3: <tf.Tensor: id=2, shape=(), dtype=float32, numpy=7.0>
# node3.numpy(): 7.0
node3 = tf.add(node1, node2)
print("node3:", node3)
print("node3.numpy():", node3.numpy())