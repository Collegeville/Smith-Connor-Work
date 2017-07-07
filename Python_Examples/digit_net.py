'''
Best acc: ~.94
2 layers; 400 neurons each
7500 epochs
Default learning rate
Batch_size = 256
.5 dropout on first layer
'''
import tensorflow as tf 
import numpy as np 

DATA_FILE = "digits.csv"

neurons_layer1 = 700
neurons_layer2 = 700

x = tf.placeholder(tf.float32, [None, 16], name="input")
y = tf.placeholder(tf.int32, name="targets")

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
			if i <= 9000:
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
	hidden1 = {'weights': tf.Variable(tf.random_normal([16, neurons_layer1])),
				'biases': tf.Variable(tf.zeros(neurons_layer1))}
	hidden2 = {'weights': tf.Variable(tf.random_normal([neurons_layer1, neurons_layer2])),
				'biases': tf.Variable(tf.zeros(neurons_layer2))}
	output = {'weights': tf.Variable(tf.random_normal([neurons_layer2, 10])),
				'biases': tf.Variable(tf.zeros(10))}

	#First NN layer utilizes dropout to prevent overfitting of training data
	layer1 = tf.add(tf.matmul(input_data, hidden1['weights']), hidden1['biases'])
	layer1 = tf.nn.dropout(tf.nn.relu(layer1), .75)

	layer2 = tf.add(tf.matmul(layer1, hidden2['weights']), hidden2['biases'])
	layer2 = tf.nn.relu(layer2)

	output = tf.add(tf.matmul(layer2, output['weights']), output['biases'])

	return output

def train_model(x,y):
	batch_size = 256

	pred = model(x)

	#Sparse softmax cost function treats integer values as index with highest value within an array
	cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred))

	optimizer = tf.train.AdamOptimizer(.001).minimize(cost)

	epochs = 7500

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


			#Calculate accuracy on part of test set (whole test set inefficient with current method)
			#Calculate accuracy every 10 training iterations 
			t = 0
			equal = 0

			while t < 1000:

				start = t
				end = t + 1

				tbatch_x = np.array(test_x[start:end])
				tbatch_y = np.array(test_y[start:end])

				correct_prediction = tf.equal(tf.argmax(pred, 1), tf.cast(y, tf.int64))
				correct = correct_prediction.eval({x: tbatch_x, y: tbatch_y})

				if correct == True:
					equal += 1

				t += 1

			print("Accuracy: ", equal / 1000)
train_model(x,y)
