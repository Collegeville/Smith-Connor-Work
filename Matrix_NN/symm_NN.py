#In current form, this program is trained on matrices of size 5x5 
#to determine whether or not they are symmetric

import tensorflow as tf
import numpy as np

DATA_FILE = "matrices_test.csv"

neurons_layer1 = 100

x = tf.placeholder(tf.float32, name="input")
y = tf.placeholder(tf.int32, name="targets")

#Reusable method used to read data from .csv file (By Connor Smith)
def get_data(filename):
	filename_queue = tf.train.string_input_producer([filename])

	reader = tf.TextLineReader(skip_header_lines=1)
	key, value = reader.read(filename_queue)

	record_defaults = [[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.]]
	col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16, col17, col18, col19, col20, col21, col22, col23, col24, col25, col26 = tf.decode_csv(value, record_defaults=record_defaults)
	features = tf.stack([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16, col17, col18, col19, col20, col21, col22, col23, col24, col25])
	targets = tf.stack([col26])

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		feature_list = list()
		label_list = list()
		test_feature_list = list()
		test_label_list = list()

		data_size = 24

		for i in range(data_size):
			example, label = sess.run([features, targets])
			if i <= 17:
				feature_list.append(example)
				label_list.append(label)
			else:
				test_feature_list.append(example)
				test_label_list.append(label)

	coord.request_stop()
	coord.join(threads)

	return feature_list, label_list, test_feature_list, test_label_list


	#Design model architecture for best possible accuracy
def model(input_data):
	hidden1 = {'weights': tf.Variable(tf.random_normal([25, neurons_layer1])),
				'biases': tf.Variable(tf.zeros(neurons_layer1))}		
	output = {'weights': tf.Variable(tf.random_normal([neurons_layer1, 2])),
				'biases': tf.Variable(tf.zeros(2))}

	layer1 = tf.add(tf.matmul(input_data, hidden1['weights']), hidden1['biases'], name='layer1')
	layer1 = tf.nn.relu(layer1)

	output = tf.add(tf.matmul(layer1, output['weights']), output['biases'], name='output')

	return output


def train_model(x,y):
	batch_size = 5

	pred = model(x)

	cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred))

	optimizer = tf.train.AdamOptimizer(.001).minimize(cost)

	epochs = 1000

	saver = tf.train.Saver()

	#Run the processes built into the computation graph
	#Iterates through graph for number of specified epochs
	with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			train_x, train_y, test_x, test_y = get_data(DATA_FILE)

			for epoch in range(epochs):
				epoch_loss = 0

				i = 0

				while i < len(train_x) / batch_size:
					start = i
					end = i + batch_size 

					batch_x = np.array(train_x[start:end])
					batch_y = np.array(train_y[start:end])
					batch_y = np.reshape(batch_y, (batch_size))

					_, c = sess.run([optimizer, cost], feed_dict= {x: batch_x, y: batch_y})

					epoch_loss += c

					i += batch_size

				print("Epoch: ", epoch, " loss: ", epoch_loss)

			print(pred.eval({x:test_x}))


train_model(x,y)