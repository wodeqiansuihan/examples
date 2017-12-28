import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

minist = input_data.read_data_sets("MNIST_Data",one_hot=True)

learning_rate = 0.1
epochs = 10000
batch_size = 128

n_inputs = 28
n_steps = 28
n_hidden_unit = 128


#input holder
#we use this pattern to get our data,but when we processing in the cell,we will expand the x to another model.
#we define x below
x = tf.placeholder(tf.float32,[None,28,28])
y = tf.placeholder(tf.float32,[None,10]) #how to explain this code.