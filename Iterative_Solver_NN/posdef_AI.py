#This program uses trained Neural Networks to predict parameters used
#in MatLab iterative solvers
import tensorflow as tf  
import preprocess
import numpy as np

def predict():
	x = tf.placeholder(tf.float32)

	##########################################################
	###Ask and retrieve user input for array specifications###
	##########################################################
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
	##########################################################


	####################
	##Solver parameter##
	####################
	solver_graph = tf.Graph()
	solver_sess = tf.Session(graph=solver_graph)

	with solver_graph.as_default():
		solver_loader = tf.train.import_meta_graph('Saved/PosDef/solver/solver_model.meta', clear_devices=True)
		solver_loader.restore(solver_sess, 'Saved/PosDef/solver/solver_model')

		x = solver_graph.get_tensor_by_name("input:0")

		solver_array = np.zeros([1,1])

		solver_array[0,0] = input_array[0,4]

		feed_dict = {x: solver_array}
	
		output = solver_graph.get_tensor_by_name("output:0")

		solver_int = solver_sess.run(output, feed_dict)

		#####################################################
		####Use solver value to determine what AI to use!####
		#####################################################

		#solver = preprocess.decode(6, np.argmax(solver_int))

		solver = "pcg"
	####################


	###################
	##Maxit parameter##
	###################
	maxit_graph = tf.Graph()
	maxit_sess = tf.Session(graph=maxit_graph)

	with maxit_graph.as_default():
		maxit_loader = tf.train.import_meta_graph('Saved/PosDef/maxit/maxit_model.meta', clear_devices=True)
		maxit_loader.restore(maxit_sess, 'Saved/PosDef/maxit/maxit_model')

		x = maxit_graph.get_tensor_by_name("input:0")

		maxit_input = np.zeros([1,5])

		maxit_input[0,0] = input_array[0,1]

		maxit_input[0,1] = input_array[0,2]

		maxit_input[0,2] = input_array[0,3]

		maxit_input[0,3] = input_array[0,4]

		maxit_input[0,4] = input_array[0,5]

		feed_dict = {x: maxit_input}
	
		output = maxit_graph.get_tensor_by_name("output:0")

		maxit = maxit_sess.run(output, feed_dict)
		maxit = abs(maxit[0][0])
	###################


	##############################
	##Recommended preconditioner##
	##############################
	precond_graph = tf.Graph()
	precond_sess = tf.Session(graph=precond_graph)

	with precond_graph.as_default():
		precond_loader = tf.train.import_meta_graph('Saved/PosDef/precond/precond_model.meta', clear_devices=True)
		precond_loader.restore(precond_sess, 'Saved/PosDef/precond/precond_model')

		x = precond_graph.get_tensor_by_name("input:0")

		feed_dict = {x: solver_array}
	
		output = precond_graph.get_tensor_by_name("output:0")

		precond_int = precond_sess.run(output, feed_dict)

		#precond = preprocess.decode(9, np.argmax(precond_int))

		precond = "ichol"
	##############################


	#####################
	##Droptol parameter##
	#####################
	droptol_graph = tf.Graph()
	droptol_sess = tf.Session(graph=droptol_graph)

	with droptol_graph.as_default():
		droptol_loader = tf.train.import_meta_graph('Saved/PosDef/droptol/droptol_model.meta', clear_devices=True)
		droptol_loader.restore(droptol_sess, 'Saved/PosDef/droptol/droptol_model')

		x = droptol_graph.get_tensor_by_name("input:0")

		feed_dict = {x: input_array}
	
		output = droptol_graph.get_tensor_by_name("output:0")

		droptol = droptol_sess.run(output, feed_dict)
		droptol = abs(droptol[0][0])
	#####################


	######################
	##Diagcomp parameter##
	######################
	diagcomp_graph = tf.Graph()
	diagcomp_sess = tf.Session(graph=diagcomp_graph)

	with diagcomp_graph.as_default():
		diagcomp_loader = tf.train.import_meta_graph('Saved/PosDef/diagcomp/diagcomp_model.meta', clear_devices=True)
		diagcomp_loader.restore(diagcomp_sess, 'Saved/PosDef/diagcomp/diagcomp_model')

		diagcomp_input = np.zeros([1,8])

		for i in range(len(input_array)):
			diagcomp_input[0,i] = input_array[0,i]

		diagcomp_input[0,7] = droptol

		x = diagcomp_graph.get_tensor_by_name("diagcomp_input:0")

		feed_dict = {x: diagcomp_input}
	
		output = diagcomp_graph.get_tensor_by_name("diagcomp_output:0")

		diagcomp = diagcomp_sess.run(output, feed_dict)
		diagcomp = abs(diagcomp[0][0])
	######################


	#####################################
	###Print parameter recommendations###
	#####################################
	print("\n\nRecommended solver: " + solver + "\n")
	print("Recommended maximum number of iterations: " + str(maxit) + "\n")
	print("Recommended preconditioner: " + precond + "\n")
	print("Recommended drop tolerance (if applicable): " + str(droptol) + "\n")
	print("Recommended diagcomp (if applicable): " + str(diagcomp) + "\n")	
	#####################################

predict()

