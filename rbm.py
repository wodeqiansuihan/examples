# -*- coding: utf-8 -*-

import sys
import numpy
import matplotlib.pylab as plt
import numpy as np
import random
from scipy.linalg import norm
import PIL.Image
from utils import *

class RBM(object):
    def __init__(self, input=None, n_visible=2, n_hidden=3, W=None, hbias=None, vbias=None, rng=None):

        self.n_visible = n_visible  # num of units in visible (input) layer
        self.n_hidden = n_hidden    # num of units in hidden layer

        if rng is None:
            rng = numpy.random.RandomState(1234)


        if W is None:
            a = 1. / n_visible
            initial_W = numpy.array(rng.uniform(  # initialize W uniformly(�������ʵ����-a-a֮��)
                low=-a,
                high=a,
                size=(n_visible, n_hidden)))

            W = initial_W

#hbias是正向传播的时候的那个bias.
# h 是hidden的意思
        if hbias is None:
            hbias = numpy.zeros(n_hidden)  # initialize h bias 0

#vbias 是反向传播的时候的那个 bias..
# v  是visible的意思.
        if vbias is None:
            vbias = numpy.zeros(n_visible)  # initialize v bias 0


        self.rng = rng
        self.input = input
        self.W = W
        self.hbias = hbias
        self.vbias = vbias


    def contrastive_divergence(self, lr=0.1, k=1, input=None):
        if input is not None:
            self.input = input

        ''' CD-ks�㷨 '''
        #mean 其实是样本正向传播后的概率了.
        #sample是在mean的基础上做了抽样,是抽样的样本.
        ph_mean, ph_sample = self.sample_h_given_v(self.input)

#马尔科夫链.
        chain_start = ph_sample


        for step in xrange(k):
            if step == 0:
                nv_means, nv_samples,nh_means, nh_samples = self.gibbs_hvh(chain_start)
            else:
                nv_means, nv_samples,nh_means, nh_samples = self.gibbs_hvh(nh_samples)

        # chain_end = nv_samples


#但是如何解释这种模型呢??为什么这么做可以呢?站在概率角度上,怎么理解这种东西呢??
        self.W += lr * (numpy.dot(self.input.T, ph_mean)
                        - numpy.dot(nv_samples.T, nh_means))
        self.vbias += lr * numpy.mean(self.input - nv_samples, axis=0)
        self.hbias += lr * numpy.mean(ph_mean - nh_means, axis=0)

        # cost = self.get_reconstruction_cross_entropy()
        # return cost

    def sample_h_given_v(self, v0_sample):
        h1_mean = self.propup(v0_sample)#正向传播的sigmoid后的结果.
        h1_sample = self.rng.binomial(size=h1_mean.shape,   # discrete: binomial(离散的二项分布)
                                       n=1,
                                       p=h1_mean)

        return [h1_mean, h1_sample]


    def sample_v_given_h(self, h0_sample):
        v1_mean = self.propdown(h0_sample)
        v1_sample = self.rng.binomial(size=v1_mean.shape,   # discrete: binomial
                                            n=1,
                                            p=v1_mean)

        return [v1_mean, v1_sample]

    def propup(self, v):
        pre_sigmoid_activation = numpy.dot(v, self.W) + self.hbias
        return sigmoid(pre_sigmoid_activation)

    def propdown(self, h):
        pre_sigmoid_activation = numpy.dot(h, self.W.T) + self.vbias
        return sigmoid(pre_sigmoid_activation)


    def gibbs_hvh(self, h0_sample):
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample)

        return [v1_mean, v1_sample,
                h1_mean, h1_sample]


    def get_reconstruction_cross_entropy(self):
        pre_sigmoid_activation_h = numpy.dot(self.input, self.W) + self.hbias
        sigmoid_activation_h = sigmoid(pre_sigmoid_activation_h)

        pre_sigmoid_activation_v = numpy.dot(sigmoid_activation_h, self.W.T) + self.vbias
        sigmoid_activation_v = sigmoid(pre_sigmoid_activation_v)

        cross_entropy =  - numpy.mean(
            numpy.sum(self.input * numpy.log(sigmoid_activation_v) +
            (1 - self.input) * numpy.log(1 - sigmoid_activation_v),
                      axis=1))

        return cross_entropy

    def reconstruct(self, v):
        h = sigmoid(numpy.dot(v, self.W) + self.hbias)
        reconstructed_v = sigmoid(numpy.dot(h, self.W.T) + self.vbias)
        return reconstructed_v




def readData(path):
    data = []
    for line in open(path, 'r'):
        ele = line.split(' ')
    tmp = []
    for e in ele:
        if e != '':
            tmp.append(float(e.strip(' ')))
    data.append(tmp)
    return data
def test_rbm(learning_rate=0.1, k=1, training_epochs=50):
#     data = numpy.array([[1,1,1,0,0,0],
#                         [1,0,1,0,0,0],
#                         [1,1,1,0,0,0],
#                         [0,0,1,1,1,0],
#                         [0,0,1,1,0,0],
#                         [0,0,1,1,1,0]])
    data = readData('data.txt')
    data = np.array(data)
    data = data.transpose()


    rng = numpy.random.RandomState(123)

    # construct RBM
#     rbm = RBM(input=data, n_visible=6, n_hidden=2, rng=rng)

    rbm = RBM(input=data, n_visible=784, n_hidden=2, rng=rng)
    # train
    for epoch in xrange(training_epochs):
        rbm.contrastive_divergence(lr=learning_rate, k=k)
        cost = rbm.get_reconstruction_cross_entropy()
        print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, cost


    # test
#     v = numpy.array([[1, 1, 0, 0, 0, 0],
#                      [0, 0, 0, 1, 1, 0]])

    v=data[1,:]

    print rbm.reconstruct(v)

if __name__ == "__main__":
    test_rbm()
	

import numpy
numpy.seterr(all='ignore')


def sigmoid(x):
    return 1. / (1 + numpy.exp(-x))


def dsigmoid(x):
    return x * (1. - x)

# def tanh(x):
#     return numpy.tanh(x)
# 
# def dtanh(x):
#     return 1. - x * x
# 
# def softmax(x):
#     e = numpy.exp(x - numpy.max(x))  # prevent overflow
#     if e.ndim == 1:
#         return e / numpy.sum(e, axis=0)
#     else:  
#         return e / numpy.array([numpy.sum(e, axis=1)]).T  # ndim = 2
# 
# 
# def ReLU(x):
#     return x * (x > 0)
# 
# def dReLU(x):
#     return 1. * (x > 0)