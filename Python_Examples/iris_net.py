'''
Created on Jun 19, 2017

@author: conds

Builds deep neural network with 3 hidden layers using high-level TensorFlow api.
'''
import tensorflow as tf
import numpy as np

TRAIN_DATA = "iris_training.csv"

TEST_DATA = "iris_test.csv"

def main():
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=TRAIN_DATA,target_dtype=np.int,features_dtype=np.float32)
    
    testing_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=TEST_DATA,target_dtype=np.int,features_dtype=np.float32)

    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
    
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[20, 20], n_classes=3)


    def get_train_inputs():
        x = tf.constant(training_set.data)
        y = tf.constant(training_set.target)

        return x,y


    classifier.fit(input_fn=get_train_inputs, steps=1000)

    def get_test_inputs():
        x = tf.constant(testing_set.data)
        y = tf.constant(testing_set.target)

        return x,y

    accuracy_score = classifier.evaluate(input_fn=get_test_inputs, steps=1)["accuracy"]

    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
                        
