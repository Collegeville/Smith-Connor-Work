#Scale equation: y = (x*(.9-.1)+.1*max(x)-.1*min(x))/(max(x)-min(x))
#Re-scale equation: yr=(y*(max(x)-min(x))-.1*max(x)+.9*min(x))/(.9-.1) 


import tensorflow as tf 
import numpy as np 

DATA_FILE = "encoded.csv"

neurons_layer1 = 100
neurons_layer2 = 100

x = tf.placeholder(tf.float32, name="input")
y = tf.placeholder(tf.float32, name="targets")

#Reusable method used to read data from .csv file (By Connor Smith)
#Use in different file?
def get_data(filename):
	filename_queue = tf.train.string_input_producer([filename])

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

		data_size = 475

		for i in range(data_size):
			example, label = sess.run([features, targets])
			if i <= 390:
				feature_list.append(example)
				label_list.append(label)
			else:
				test_feature_list.append(example)
				test_label_list.append(label)

	coord.request_stop()
	coord.join(threads)

	features = list()
	labels = list()
	test_features = list()
	test_labels = list()

	features = [(r-min(feature_list))/(max(feature_list)-min(feature_list)) for r in feature_list]
	labels = [(s-min(label_list))/(max(label_list)-min(label_list)) for s in label_list]
	test_features = [(tr-min(test_feature_list))/(max(test_feature_list)-min(test_feature_list)) for tr in test_feature_list]
	test_labels = [(ts-min(test_label_list))/(max(test_label_list)-min(test_label_list)) for ts in test_label_list]

	#return features, labels, test_features, test_labels
	return features, labels, test_features, test_labels

#Design model architecture for best possible accuracy
def model(input_data):
	hidden1 = {'weights': tf.Variable(tf.random_normal([7, neurons_layer1])),
				'biases': tf.Variable(tf.zeros(neurons_layer1))}
	hidden2 = {'weights': tf.Variable(tf.random_normal([neurons_layer1, neurons_layer2])),
				'biases': tf.Variable(tf.zeros(neurons_layer2))}
	output = {'weights': tf.Variable(tf.random_normal([neurons_layer2, 5])),
				'biases': tf.Variable(tf.zeros(5))}

	#First NN layer utilizes dropout to prevent overfitting of training data
	layer1 = tf.add(tf.matmul(input_data, hidden1['weights']), hidden1['biases'])
	layer1 = tf.nn.dropout(tf.nn.relu(layer1), .75)

	layer2 = tf.add(tf.matmul(layer1, hidden2['weights']), hidden2['biases'])
	layer2 = tf.nn.relu(layer2)

	output = tf.add(tf.matmul(layer2, output['weights']), output['biases'])

	return output

def train_model(x,y):
	batch_size = 50

	pred = model(x)

	y_shape = tf.shape(y)
	#Sparse softmax cost function treats integer values as index with highest value within an array
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))

	optimizer = tf.train.AdamOptimizer(.001).minimize(cost)

	epochs = 1000

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

					_, c = sess.run([optimizer, cost], feed_dict= {x: batch_x, y: batch_y})

					epoch_loss += c

					i += batch_size

				print("Epoch: ", epoch, " loss: ", epoch_loss)


			correct_prediction = tf.equal(tf.round(pred), tf.round(y))

			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

			print("Accuracy: ", accuracy.eval(feed_dict={x: test_x, y: test_y}))

train_model(x,y)