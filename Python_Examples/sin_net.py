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
neurons_layer2 = 10
neurons_layer3 = 10


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

	#Utilizes built-in TensorFlow functions to read data from .csv file
	#with particular number of feature columns (in this case 1)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		deg_list = list()
		sin_list = list()
		test_deg_list = list()
		test_val_list = list()

		data_size = 25000

		for i in range(data_size):
			example, label = sess.run([features, col2])
			if i <= 20000:
				deg_list.append(example)
				sin_list.append(label)
			else:
				test_deg_list.append(example)
				test_val_list.append(label)

		#Normalize data between 0 and 1(temporary fix for debugging)
		#deg = [(r-min(rad_list))/(max(rad_list)-min(rad_list)) for r in rad_list]
		#sin = [(s-min(sin_list))/(max(sin_list)-min(sin_list)) for s in sin_list]
		#test_deg = [(tr-min(test_rad_list))/(max(test_rad_list)-min(test_rad_list)) for tr in test_rad_list]
		#test_sin = [(ts-min(test_val_list))/(max(test_val_list)-min(test_val_list)) for ts in test_val_list]


	coord.request_stop()
	coord.join(threads)


	return deg_list, sin_list, test_deg_list, test_val_list



#Creates neural network model with three hidden layers
def model(input_data):
	hidden1 = {'weights': tf.Variable(tf.random_normal([1, neurons_layer1])),
						'biases': tf.Variable(tf.zeros(neurons_layer1))}
	hidden2 = {'weights': tf.Variable(tf.random_normal([neurons_layer1, neurons_layer2])),
						'biases': tf.Variable(tf.zeros(neurons_layer2))}
	hidden3 = {'weights': tf.Variable(tf.random_normal([neurons_layer2, neurons_layer3])),
						'biases': tf.Variable(tf.zeros(neurons_layer3))}
	output = {'weights': tf.Variable(tf.random_normal([neurons_layer1, 1])),
						'biases': tf.Variable(tf.zeros(1))}


	layer1 = tf.add(tf.multiply(input_data, hidden1['weights']), hidden1['biases'])
	layer1 = tf.tanh(layer1)

	#layer2 = tf.add(tf.matmul(layer1, hidden2['weights']), hidden2['biases'])
	#layer2 = tf.tanh(layer2)

	#layer3 = tf.add(tf.matmul(layer2, hidden3['weights']), hidden3['biases'])
	#layer3 = tf.tanh(layer3)

	output = tf.add(tf.matmul(layer1, output['weights']), output['biases'])

	return output

#Train the neural network model
def train_net(x,y):
	#Size of each data sample to feed through network
	batch_size = 100

	pred = model(x)
	cost = tf.reduce_mean(tf.square(pred - y))
	#tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred))

	#Optimize weights and biases in order to minimize cost function
	optimizer = tf.train.GradientDescentOptimizer(.001).minimize(cost)

	epochs = 20

	saver = tf.train.Saver()

	#Run computation graph that feeds pre-processed data through network
	#and the cost and optimizer functions
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

		#saver.save(sess,'66')

		#Calculate accuracy of model
		#correct = tf.equal(tf.round(tf.multiply(x,100)), tf.round(tf.multiply(y,100)))
		#accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		#print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))
		print(pred.eval({x: test_x}))

		#Allow up to 100 user inputs to make predictions based on model
		test_inputs = 100
		for _ in range(test_inputs):
			input_for_pred = input("Enter degree value to be approximated \n")
			input_for_pred = float(input_for_pred)
			feed_dict = {x: input_for_pred}
			approx = pred.eval(feed_dict)
			print(approx)

train_net(x,y)
				




