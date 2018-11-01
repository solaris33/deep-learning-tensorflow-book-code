# -*- coding: utf-8 -*-

import tensorflow as tf

# 플레이스홀더 노드와 add 노드를 정의합니다.
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # 암시적으로 tf.add(a, b) 형태로 정의될 것입니다.

# 세션을 열고 그래프를 실행합니다.
# 출력값 :
# 7.5
# [ 3.  7.]
sess = tf.Session()
print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}))

# 노드를 추가해서 더 복잡한 그래프 형태를 만들어봅시다.
# 출력값 : 22.5
add_and_triple = adder_node * 3	# 암시적으로 tf.multiply(adder_node,3) 형태로 정의될 것입니다.
print(sess.run(add_and_triple, feed_dict={a: 3, b: 4.5}))

sess.close()