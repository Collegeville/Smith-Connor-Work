'''
author: Connor Smith

Creates a feed-forward neural network that is designed to 
approximate sin function. Training and testing data read
from a .csv file.
'''


import tensorflow as tf
import numpy as np

DATA_FILE = 'sin_data.csv'


#Number of neurons in each layer
neurons_layer1 = 50
neurons_layer2 = 60
neurons_layer3 = 50

#batch_size =  (not large enough data set)

x = tf.placeholder('float')
y = tf.placeholder('float')

#Extracts training and testing data from single file
#Returns training and testing inputs and outputs (four values)
def get_data(filename):
	filename_queue = tf.train.string_input_producer([filename])

	reader = tf.TextLineReader()
	key, value = reader.read(filename_queue)

    # Default values, in case of empty columns. Also specifies the type of the
    # decoded result.
	record_defaults = [[1.], [1.]]
	col1, col2 = tf.decode_csv(value, record_defaults=record_defaults)
	features = tf.stack([col1])

	#Utilizes built-in TensorFlow functions to 
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		rad_list = list()
		sin_list = list()
		test_rad_list = list()
		test_val_list = list()

		data_size = 721

		for i in range(data_size):
			example, label = sess.run([features, col2])
			if i <= 500:
				rad_list.append(example)
				sin_list.append(label)
			else:
				test_rad_list.append(example)
				test_val_list.append(label)

		#Normalize data between 0 and 1(temporary fix for debugging)
		rad = [(r-min(rad_list))/(max(rad_list)-min(rad_list)) for r in rad_list]
		sin = [(s-min(sin_list))/(max(sin_list)-min(sin_list)) for s in sin_list]
		test_rad = [(tr-min(test_rad_list))/(max(test_rad_list)-min(test_rad_list)) for tr in test_rad_list]
		test_sin = [(ts-min(test_val_list))/(max(test_val_list)-min(test_val_list)) for ts in test_val_list]


	coord.request_stop()
	coord.join(threads)

	return rad, sin, test_rad, test_sin



#Creates neural network model with three hidden layers
def model(input_data):
	hidden1 = {'weights': tf.Variable(tf.random_normal([1, neurons_layer1])),
						'biases': tf.Variable(tf.zeros(neurons_layer1))}
	hidden2 = {'weights': tf.Variable(tf.random_normal([neurons_layer1, neurons_layer2])),
						'biases': tf.Variable(tf.zeros(neurons_layer2))}
	hidden3 = {'weights': tf.Variable(tf.random_normal([neurons_layer2, neurons_layer3])),
						'biases': tf.Variable(tf.zeros(neurons_layer3))}
	output = {'weights': tf.Variable(tf.random_normal([neurons_layer3, 1])),
						'biases': tf.Variable(tf.zeros(1))}


	layer1 = tf.add(tf.multiply(input_data, hidden1['weights']), hidden1['biases'])
	layer1 = tf.nn.relu(layer1)

	layer2 = tf.add(tf.matmul(layer1, hidden2['weights']), hidden2['biases'])
	layer2 = tf.nn.relu(layer2)

	layer3 = tf.add(tf.matmul(layer2, hidden3['weights']), hidden3['biases'])
	layer3 = tf.nn.relu(layer3)

	output = tf.add(tf.matmul(layer3, output['weights']), output['biases'])

	return output

#Train the neural network model
def train_net(x,y):
	pred = model(x)
	cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred,labels=y))

	#Optimize weights and biases in order to minimize cost function
	optimizer = tf.train.AdamOptimizer(learning_rate=.1).minimize(cost)

	epochs = 2

	#Run computation graph that feeds 
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		train_x, train_y, test_x, test_y = get_data(DATA_FILE)

		for epoch in range(epochs):
			epoch_loss = 0
			step = 0
			for _ in range((len(train_x))):
				_, c = sess.run([optimizer, cost], feed_dict = {x: train_x, y: train_y})
				epoch_loss += c

				step += 1

			print('Epoch: ', epoch, ' loss: ', epoch_loss)

		correct = tf.equal(tf.argmax(pred), tf.argmax(y))
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))

train_net(x,y)
				




