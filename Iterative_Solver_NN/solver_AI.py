
import tensorflow as tf  
import preprocess
import numpy as np

def predict():
	x = tf.placeholder(tf.float32)

	with tf.Session() as sess:
		solver_loader = tf.train.import_meta_graph('Saved/solver_model.meta')
		solver_loader.restore(sess, 'Saved/solver_model')

		graph = tf.get_default_graph()

		x = graph.get_tensor_by_name("input:0")

		input_array = np.zeros([1,7])

		sess.run(tf.global_variables_initializer())
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

		intvals, kind_options = preprocess.to_int(5)
		options_list = ""
		for i in range(0,len(kind_options)):
			options_list = options_list + "\n" + kind_options[i]

		kind_string = input("Choose kind: " + options_list + "\n")
		kind = preprocess.encode(5," " + kind_string)
		input_array[0,5] = kind

		tol = float(input("Enter desired convergence tolerance: \n"))
		input_array[0,6] = tol

		feed_dict = {x:input_array}
	
		output = graph.get_tensor_by_name("output:0")

		print(input_array)

		solver_int = sess.run(output, feed_dict)
		print(solver_int)
		solver = preprocess.decode(6, np.argmax(solver_int))

		print(solver)

predict()



		#predictions = list()

		#for i in range(0,5):
		#	p = nn_output[0,i]
		#	if p < 1e-8:
		#		p = 0.0
		#	predictions.append(p)
		
		#solver_int = int(round(predictions[0]))
		#solver = preprocess.decode(6,solver_int)

		#precond_int = int(round(predictions[2]))
		#precond = preprocess.decode(9, precond_int)

		#maxit = round(predictions[1])

		#droptol = predictions[3]

		#diagcomp = predictions[4]

		#print("Recommended solver: " + solver + "\n")
		#print("Recommended maximum number of iterations: " + str(maxit) + "\n")
		#print("Recommended preconditioner: " + precond + "\n")
		#print("Recommended drop tolerance (if applicable): " + str(droptol) + "\n")
		#print("Recommended diagcomp (if applicable): " + str(diagcomp) + "\n")