import tensorflow.contrib.slim as slim
weights = slim.variable('weights',
	shape=[10, 10, 3 , 3],
	initializer=tf.truncated_normal_initializer(stddev=0.1),
	regularizer=slim.l2_regularizer(0.05),
	device='/CPU:0')