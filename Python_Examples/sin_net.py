'''
author: Connor Smith

Creates a feed-forward neural network that is designed to 
approximate sin function. Training and testing data read
from same .csv file.

Best accuracy: 89%
Learning rate should stay at .0001 in order to prevent vanishing gradient (because there is only one feature)
'''


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

DATA_FILE = 'sin_data_random.csv'


#Number of neurons in each layer
neurons_layer1 = 20
#neurons_layer2 = 15
#neurons_layer3 = 5


x = tf.placeholder('float', name='input')
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
	targets = tf.stack([col2])

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

		data_size = 23000

		for i in range(data_size):
			example, label = sess.run([features, targets])
			if i <= 15000:
				deg_list.append(example)
				sin_list.append(label)
			else:
				test_deg_list.append(example)
				test_val_list.append(label)

		#Normalize data between 0 and 1(temporary fix for debugging)
		deg = [(r-min(deg_list))/(max(deg_list)-min(deg_list)) for r in deg_list]
		sin = [(s-min(sin_list))/(max(sin_list)-min(sin_list)) for s in sin_list]
		test_deg = [(tr-min(test_deg_list))/(max(test_deg_list)-min(test_deg_list)) for tr in test_deg_list]
		test_sin = [(ts-min(test_val_list))/(max(test_val_list)-min(test_val_list)) for ts in test_val_list]


	coord.request_stop()
	coord.join(threads)


	return deg_list, sin_list, test_deg_list, test_val_list



#Creates neural network model with three hidden layers
def model(input_data):
	hidden1 = {'weights': tf.Variable(tf.random_normal([1, neurons_layer1]), name='w1'),
						'biases': tf.Variable(tf.zeros(neurons_layer1), name='b1')}
	#hidden2 = {'weights': tf.Variable(tf.random_normal([neurons_layer1, neurons_layer2])),
						#'biases': tf.Variable(tf.zeros(neurons_layer2))}
	#hidden3 = {'weights': tf.Variable(tf.random_normal([neurons_layer2, neurons_layer3])),
						#'biases': tf.Variable(tf.zeros(neurons_layer3))}
	output = {'weights': tf.Variable(tf.random_normal([neurons_layer1, 1]), name='wO'),
						'biases': tf.Variable(tf.zeros(1), name='bO')}


	layer1 = tf.add(tf.multiply(input_data, hidden1['weights']), hidden1['biases'], name='layer1')

	#Use hyperbolic tangent activation function because output is between -1 and 1,
	#resulting in fastest reduction of cost
	layer1 = tf.tanh(layer1)

	#Only one layer required for optimal output, as there is only one input feature
	#and one output, but infinite possible output values
	#layer2 = tf.add(tf.matmul(layer1, hidden2['weights']), hidden2['biases'])
	#layer2 = tf.tanh(layer2)
	#layer3 = tf.add(tf.matmul(layer2, hidden3['weights']), hidden3['biases'])
	#layer3 = tf.tanh(layer3)

	output = tf.add(tf.matmul(layer1, output['weights']), output['biases'], name='output')

	return output

#Train the neural network model
def train_net(x,y):
	#Size of each data sample to feed through network
	batch_size = 256

	pred = model(x)

	#Utilize mean-squared cost function
	cost = tf.reduce_mean(tf.square(pred - y))
	tf.summary.scalar("cost", cost)

	#Optimize weights and biases in order to minimize cost function
	###Changed LR from .0001 to .001
	optimizer = tf.train.AdamOptimizer(.0001).minimize(cost)

	epochs = 1000

	#Initialize Saver object to save state of weights and biases for later use
	saver = tf.train.Saver()

	#Run computation graph that feeds pre-processed data through network
	#and the cost and optimizer functions
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		#Get data from user-created function that extracts data from .csv file
		train_x, train_y, test_x, test_y = get_data(DATA_FILE)

		#For loop runs in range of epochs variable, each loop feeds data through
		#network using TensorFlow "magic"
		for epoch in range(epochs):
			epoch_loss = 0

			epoch_array = np.append(epoch_array, epoch)

			i = 0
			while i < len(train_x):
				start = i 
				end = i + batch_size

				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])

				_, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
				epoch_loss += c


				i += batch_size

			print('Epoch: ', epoch, ' loss: ', epoch_loss)

		saver.save(sess,'.\sin-model')

		correct_prediction = tf.equal(tf.round(tf.multiply(pred,100)), tf.round(tf.multiply(y,100)))

		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		print("Accuracy: ", accuracy.eval(feed_dict={x: test_x, y: test_y}))

		#Allow up to 100 user inputs to make predictions based on model
		#Deprecated because of second program to approximate sin function (sin_model.py)
		#test_inputs = 100
		#for _ in range(test_inputs):
		#	input_for_pred = input("Enter degree value to be approximated \n")
		#	input_for_pred = float(input_for_pred)
		#	feed_dict = {x: input_for_pred}
		#	approx = pred.eval(feed_dict)
		#	print(approx)

train_net(x,y)
				




