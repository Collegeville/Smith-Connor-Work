'''
Best acc: ~.8
2 layers; 20 neurons each
5000 epochs
.00099 learning rate
Batch_size = 64
No dropout
'''
import tensorflow as tf 
import numpy as np 

DATA_FILE = "digits.csv"

neurons_layer1 = 15
neurons_layer2 = 20
neurons_layer3 = 10

x = tf.placeholder(tf.float32, [None, 16])
y = tf.placeholder(tf.int32)

#Reusable method used to read data from .csv file (By Connor Smith)
def get_data(filename):
	filename_queue = tf.train.string_input_producer([filename])

	reader = tf.TextLineReader()
	key, value = reader.read(filename_queue)

	record_defaults = [[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.]]
	col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16, col17 = tf.decode_csv(value, record_defaults=record_defaults)
	features = tf.stack([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16])
	targets = tf.stack([col17])

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		feature_list = list()
		label_list = list()
		test_feature_list = list()
		test_label_list = list()

		data_size = 10991

		for i in range(data_size):
			example, label = sess.run([features, targets])
			if i <= 7800:
				feature_list.append(example)
				label_list.append(label)
			else:
				test_feature_list.append(example)
				test_label_list.append(label)

	coord.request_stop()
	coord.join(threads)


	return feature_list, label_list, test_feature_list, test_label_list

def model(input_data):
	hidden1 = {'weights': tf.Variable(tf.random_normal([16, neurons_layer1])),
				'biases': tf.Variable(tf.zeros(neurons_layer1))}
	hidden2 = {'weights': tf.Variable(tf.random_normal([neurons_layer1, neurons_layer2])),
				'biases': tf.Variable(tf.zeros(neurons_layer2))}
	hidden3 = {'weights': tf.Variable(tf.random_normal([neurons_layer2, neurons_layer3])),
				'biases': tf.Variable(tf.zeros(neurons_layer3))}
	output = {'weights': tf.Variable(tf.random_normal([neurons_layer1, 10])),
				'biases': tf.Variable(tf.zeros(10))}

	layer1 = tf.add(tf.matmul(input_data, hidden1['weights']), hidden1['biases'])
	layer1 = tf.nn.relu(layer1)

	#layer2 = tf.add(tf.matmul(layer1, hidden2['weights']), hidden2['biases'])
	#layer2 = tf.nn.relu(layer2)

	#layer3 = tf.add(tf.matmul(layer2, hidden3['weights']), hidden3['biases'])
	#layer3 = tf.nn.relu(layer3)

	output = tf.transpose(tf.add(tf.matmul(layer1, output['weights']), output['biases']))

	return output

def train_model(x,y):
	batch_size = 1

	pred = model(x)

	cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred))
	#tf.reduce_mean(tf.square(pred - y))

	optimizer = tf.train.AdamOptimizer(.001).minimize(cost)

	epochs = 10

	with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			train_x, train_y, test_x, test_y = get_data(DATA_FILE)

			for epoch in range(epochs):
				epoch_loss = 0

				i = 0

				while i < len(train_x):
					start = i
					end = i + batch_size

					batch_x = np.zeros(())
					batch_y = np.array((batch_size))
					batch_x = np.append(batch_x, train_x[start])
					batch_y = np.append(batch_y, train_y[start])


					_, c = sess.run([optimizer, cost], feed_dict= {x: batch_x, y: batch_y})

					epoch_loss += c

					i += batch_size

				print("Epoch: ", epoch, " loss: ", epoch_loss)


				#if epoch % 5 == 0:

				#	correct_prediction = tf.equal(tf.round(pred), tf.round(y))

				#	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

				#	print("Accuracy: ", accuracy.eval(feed_dict={x: test_x, y: test_y}))

train_model(x,y)
