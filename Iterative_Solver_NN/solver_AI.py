
import tensorflow as tf  
import preprocess
import numpy as np

def predict():
	x = tf.placeholder(tf.float32)

	input_array = np.zeros([1,7])

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

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		solver_loader = tf.train.import_meta_graph('Saved/solver/solver_model.meta')
		solver_loader.restore(sess, 'Saved/solver/solver_model')

		graph = tf.get_default_graph()

		x = graph.get_tensor_by_name("input:0")

		feed_dict = {x:input_array}
	
		output = graph.get_tensor_by_name("output:0")

		solver_int = sess.run(output, feed_dict)
		if np.around(solver_int) < 0:
			solver = preprocess.decode(6, 0)
		else:
			solver = preprocess.decode(6, 0)


		with tf.Session() as maxit_sess:
			maxit_sess.run(tf.global_variables_initializer())
			maxit_loader = tf.train.import_meta_graph('Saved/maxit/maxit_model.meta')
			maxit_loader.restore(maxit_sess, 'Saved/maxit/maxit_model')

			maxit_graph = tf.get_default_graph()

			x = maxit_graph.get_tensor_by_name("input:0")
	
			maxit_output = maxit_graph.get_tensor_by_name("output:0")

			maxit = maxit_sess.run(maxit_output, feed_dict)

		with tf.Session() as precond_sess:
			precond_sess.run(tf.global_variables_initializer())
			precond_loader = tf.train.import_meta_graph('Saved/precond/precond_model.meta')
			precond_loader.restore(precond_sess, 'Saved/precond/precond_model')

			precond_graph = tf.get_default_graph()

			x = precond_graph.get_tensor_by_name("input:0")
	
			precond_output = precond_graph.get_tensor_by_name("output:0")

			precond_int = precond_sess.run(precond_output, feed_dict)
			if np.around(precond_int) < 0:
				precond = preprocess.decode(6, 0)
			else:
				precond = preprocess.decode(9, int(np.around(precond_int)))


		with tf.Session() as droptol_sess:
			droptol_sess.run(tf.global_variables_initializer())
			droptol_loader = tf.train.import_meta_graph('Saved/droptol/droptol_model.meta')
			droptol_loader.restore(droptol_sess, 'Saved/droptol/droptol_model')

			droptol_graph = tf.get_default_graph()

			x = droptol_graph.get_tensor_by_name("input:0")
	
			droptol_output = droptol_graph.get_tensor_by_name("output:0")

			droptol = droptol_sess.run(droptol_output, feed_dict)

		x = tf.placeholder(tf.float32, [None,8])

		with tf.Session() as diagcomp_sess:
			diagcomp_sess.run(tf.global_variables_initializer())
			diagcomp_loader = tf.train.import_meta_graph('Saved/diagcomp/diagcomp_model.meta')
			diagcomp_loader.restore(diagcomp_sess, 'Saved/diagcomp/diagcomp_model')

			diagcomp_input = np.zeros([1,8])

			for i in range(len(input_array)):
				diagcomp_input[0,i] = input_array[0,i]

			diagcomp_input[0,7] = droptol


			diagcomp_graph = tf.get_default_graph()

			diagcomp_x = diagcomp_graph.get_tensor_by_name("diagcomp_input:0")

			feed_dict = {x: diagcomp_input}
	
			diagcomp_output = diagcomp_graph.get_tensor_by_name("diagcomp_output:0")

			diagcomp = diagcomp_sess.run(diagcomp_output, feed_dict)

			print(diagcomp)

			

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