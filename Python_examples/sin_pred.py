'''
Uses previously saved weights and biases to predict 
sin model
'''
import tensorflow as tf  

def approx():

	x = tf.placeholder(tf.float32)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		loader = tf.train.import_meta_graph('sin-model.meta')
		loader.restore(sess, 'sin-model')
	
		graph = tf.get_default_graph()

		x = graph.get_tensor_by_name("input:0")

		input_val = input("Enter value to be approximated: \n")

		input_val = float(input_val)

		feed_dict = {x:input_val}

		output = graph.get_tensor_by_name("output:0")

		print(sess.run(output, feed_dict))