'''
Created on Jun 19, 2017

@author: conds

Neural network that approximates sin curve using low-level TensorFlow api.
'''

import tensorflow as tf
import numpy as np
import math


FLAGS = None

#Create 2 hidden layers for neural network; activation: tanh
def inference(radians, hidden1_units, hidden2_units):
    #First hidden layer (only 1 input because sine function only takes 1)
    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal([1, hidden1_units], stddev=1), name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')        
        hidden1 = tf.tanh(tf.matmul(radians, weights) + biases)
    #Second hidden layer
    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units], stddev=1.0 / math.sqrt(float(hidden1_units)), name='weights'))
        biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
        hidden2 = tf.tanh(tf.matmul(radians, weights) + biases)

        logits = tf.matmul(hidden2, weights) + biases
    return logits

#Compute cost function and use reduce_mean to compute mean of elements
def loss(logits, labels):
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')

#Initialize optimizer and return one training step
def training(loss, learning_rate):
    tf.summary.scalar('Loss', loss)

    optimizer = tf.train.AdamOptimizer(learning_rate)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

#Compute sum of correct elements 
def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum((tf.cast(correct, tf.int32))
        
    
                              
                              

        
        
