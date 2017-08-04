#Scale equation: y = (x*(.9-.1)+.1*max(x)-.1*min(x))/(max(x)-min(x))
#Re-scale equation: yr=(y*(max(x)-min(x))-.1*max(x)+.9*min(x))/(.9-.1) 


import tensorflow as tf 
import numpy as np 

DATA_FILE = "encoded.csv"

neurons_layer1 = 100
neurons_layer2 = 100

x = tf.placeholder(tf.float32, [None, 7], name="input")
y = tf.placeholder(tf.int32, name="targets")

#Reusable method used to read data from .csv file (By Connor Smith)
def get_data(DATA_FILE):
	filename_queue = tf.train.string_input_producer([DATA_FILE])

	reader = tf.TextLineReader()
	key, value = reader.read(filename_queue)

	record_defaults = [[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.]]
	col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12 = tf.decode_csv(value, record_defaults=record_defaults)
	features = tf.stack([col1, col2, col3, col4, col5, col6, col8])
	targets = tf.stack([col7, col9, col10, col11, col12])

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		feature_list = list()
		label_list = list()
		test_feature_list = list()
		test_label_list = list()

		data_size = 5

		for i in range(data_size):
			example, label = sess.run([features, targets])
			print(example)
			print(label)
			if i <= 5:
				feature_list.append(example)
				label_list.append(label)
			else:
				test_feature_list.append(example)
				test_label_list.append(label)

	coord.request_stop()
	coord.join(threads)


	return feature_list, label_list, test_feature_list, test_label_list


