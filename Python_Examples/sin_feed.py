'''
Created on Jun 19, 2017

@author: conds

Neural network that approximates sin curve using low-level TensorFlow api.
'''

import tensorflow as tf

TRAIN_FILE = 'sin_train.csv'
TEST_FILE = 'sin_test.csv'

FLAGS = None

def create_file_reader(filenames):
    reader = tf.TextLineReader(skip_header_lines=1)
    _, csv_row = reader.read(filename_queue)
    record_defaults = [[0.],[0.]]
    radians, sin_val = tf.decode_csv(csv_row, record_defaults=record_defaults)
    features = tf.pack([radians])
    return features, sin_val

def placeholder_inputs(batch_size):
    radians_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 1))
    sin_val_placeholder = tf.placeholder(tf.float32, shape=(batch_size))
    return radians_placeholder, labels_placeholder

def fill_feed_dict(data_set, radians_pl, sin_val_pl):
    
