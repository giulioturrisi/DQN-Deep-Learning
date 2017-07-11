#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from numpy import *
import os
from pylab import *
#from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import tensorflow as tf

#


class Q_Net():
  
  def __init__(self):      

        self.input = tf.placeholder(shape=[None, 84, 84, 3], dtype=tf.float32, name="X")
        self.target = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        self.actions = tf.placeholder(shape=[None,4], dtype=tf.float32, name="actions")


        self.W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 3, 32],stddev=0.01))
        self.b_conv1 = tf.Variable(tf.zeros(32))

        self.W_conv2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64],stddev=0.01))
        self.b_conv2 = tf.Variable(tf.zeros(64))

        self.W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64],stddev=0.01))
        self.b_conv3 = tf.Variable(tf.zeros(64))
    
        
      
        self.h_conv1 = tf.nn.relu((tf.nn.conv2d(self.input, self.W_conv1, strides = [1, 4, 4, 1], padding = "VALID")) + self.b_conv1)


        self.h_conv2 = tf.nn.relu(tf.nn.conv2d(self.h_conv1, self.W_conv2, strides = [1, 2, 2, 1], padding = "VALID") + self.b_conv2)


        self.h_conv3 = tf.nn.relu(tf.nn.conv2d(self.h_conv2, self.W_conv3, strides = [1, 1, 1, 1], padding = "VALID") + self.b_conv3)
    
        temporary = self.h_conv3.get_shape().as_list()
        dim = temporary[1]*temporary[2]*temporary[3]

    
        self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, dim])


        self.W_fc1 = tf.Variable(tf.truncated_normal([dim, 512],stddev=0.01))
        self.b_fc1 = tf.Variable(tf.zeros(512))

        self.W_fc2 = tf.Variable(tf.truncated_normal([512, 4],stddev=0.01))
        self.b_fc2 = tf.Variable(tf.zeros(4))

        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_conv3_flat, self.W_fc1) + self.b_fc1)

   
        self.out_fc3 = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2
     

   
        self.action_predictions = tf.reduce_sum(tf.multiply(self.out_fc3, self.actions), reduction_indices=1)

        # Calcualte the loss
        self.loss = tf.reduce_mean(tf.square(self.target - self.action_predictions))

        self.train_op = tf.train.AdamOptimizer(1e-6).minimize(self.loss)

    