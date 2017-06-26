import tensorflow as tf 
import numpy as np

DATA_FILE = 'wine.csv'

neurons_layer1 = 100
neurons_layer2 = 100
neurons_layer3 = 100

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.int32)

def get_data(filename):
	filename_queue = tf.train.string_input_producer([filename])

	reader = tf.TextLineReader()
	key, value = reader.read(filename_queue)

	record_defaults = [[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.],]
	col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14 = tf.decode_csv(value, record_defaults=record_defaults)
	features = tf.stack([col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14])
	targets = tf.stack([col1])

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		feature_list = list()
		label_list = list()
		test_feature_list = list()
		test_label_list = list()

		data_size = 179

		for i in range(data_size):
			example, label = sess.run([features, targets])
			if i <= 125:
				feature_list.append(example)
				label_list.append(label)
			else:
				test_feature_list.append(example)
				test_label_list.append(label)

	coord.request_stop()
	coord.join(threads)


	return feature_list, label_list, test_feature_list, test_label_list

def model(input_data):
	hidden1 = {'weights': tf.Variable(tf.random_normal([13, neurons_layer1])),
						'biases': tf.Variable(tf.zeros(neurons_layer1))}
	hidden2 = {'weights': tf.Variable(tf.random_normal([neurons_layer1, neurons_layer2])),
						'biases': tf.Variable(tf.zeros(neurons_layer2))}
	hidden3 = {'weights': tf.Variable(tf.random_normal([neurons_layer2, neurons_layer3])),
						'biases': tf.Variable(tf.zeros(neurons_layer3))}
	output = {'weights': tf.Variable(tf.random_normal([neurons_layer1, 4])),
						'biases': tf.Variable(tf.zeros(4))}

	layer1 = tf.add(tf.matmul(input_data, hidden1['weights']), hidden1['biases'])
	layer1 = tf.nn.relu(layer1)

	#layer2 = tf.add(tf.matmul(layer1, hidden2['weights']), hidden2['biases'])
	#layer2 = tf.nn.relu(layer2)

	#layer3 = tf.add(tf.matmul(layer2, hidden3['weights']), hidden3['biases'])
	#layer3 = tf.nn.relu(layer3)

	output = tf.add(tf.matmul(layer1, output['weights']), output['biases'])

	return output

def train_net(x,y):
	batch_size = 1

	pred = model(x)

	cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(y, [batch_size]), logits=pred))

	optimizer = tf.train.AdamOptimizer(.0001).minimize(cost)

	epochs = 1000

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		train_x, train_y, test_x, test_y = get_data(DATA_FILE)

		for epoch in range(epochs):
			epoch_loss = 0

			i = 0
			while i < (len(train_x)):
				start = i 
				end = i + batch_size

				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])

				_, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
				epoch_loss += c

				i += batch_size

			print('Epoch: ', epoch, ' loss: ', epoch_loss)

		correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		feed_dict = {x: test_x, y: test_y}
		print(accuracy.eval(feed_dict=feed_dict))

train_net(x,y)