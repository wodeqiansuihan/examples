import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_Data",one_hot=True)

print (mnist.train.images.shape,mnist.train.labels.shape)
print (mnist.test.images.shape,mnist.test.labels.shape)
print (mnist.validation.images.shape)