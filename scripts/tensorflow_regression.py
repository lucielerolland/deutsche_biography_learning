import tensorflow as tf
import estimation as es
import numpy as np

x, y = es.build_x_and_y('../data/')

np.save('x.npy', x)

np.save('y.npy', y)

# x = np.load('x.npy')

# y = np.load('y.npy')

beta = tf.placeholder()