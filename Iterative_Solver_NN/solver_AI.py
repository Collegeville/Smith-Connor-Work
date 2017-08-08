
import tensorflow as tf  
import preprocess
import numpy as np

def predict():
	x = tf.placeholder(tf.float32)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		loader = tf.train.import_meta_graph('iterative-model.meta')
		loader.restore(sess, 'iterative-model')
	
		graph = tf.get_default_graph()

		input_array = np.zeros([1,7])

		x = graph.get_tensor_by_name("input:0")

		size = float(input("Enter size of square matrix (one side): \n"))
		input_array[0,0] = size

		nnz = float(input("Enter number of nonzeros: \n"))
		input_array[0,1] = nnz

		sym = float(input("Enter symmetry (1 = True, 0 = False): \n"))
		input_array[0,2] = sym

		discr = float(input("Enter 2D/3D discretization (1 = True, 0 = False): \n"))
		input_array[0,3] = discr

		posdef = float(input("Enter whether positive definite (1 = True, 0 = False): \n"))
		input_array[0,4] = posdef

		kind_string = input("Enter kind of matrix (area of research) : \n")
		kind = preprocess.encode(5," " + kind_string)
		input_array[0,5] = kind

		tol = float(input("Enter desired convergence tolerance: \n"))
		input_array[0,6] = tol

		print(input_array)

		feed_dict = {x:input_array}

		output = graph.get_tensor_by_name("output:0")

		nn_output = sess.run(output, feed_dict)

		print(nn_output)