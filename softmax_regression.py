import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

with tf.Graph().as_default():
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32,shape = [None,784],name="x_holder")
        y = tf.placeholder(tf.float32,shape = [None,10],name="y_holder")

    with tf.name_scope("inference"):
        w = tf.Variable(tf.zeros([784,10]),name="weights")
        b = tf.Variable(tf.zeros([1,10]),name="bias")
        y_pred = tf.add(tf.matmul(x,w,name="muti"),b,name="add")

    with tf.name_scope("softmax"):
        y_prob = tf.nn.softmax(logits=y_pred)

    with tf.name_scope("loss"):
        loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_prob),axis = 1))

    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        trainop = optimizer.minimize(loss)

    with tf.name_scope("evalute"):
        correct_predict = tf.equal(tf.argmax(y_pred,1),tf.argmax(y,1))
        accrucy = tf.reduce_mean(tf.cast(correct_predict,tf.float32))

    writer = tf.summary.FileWriter(logdir="logs",graph=tf.get_default_graph())
    # writer.close()
    mnist = input_data.read_data_sets("MNIST_Data",one_hot=True)
    init = tf.global_variables_initializer()

    sess = tf.InteractiveSession()

    sess.run(init)

    for step in range(1000):
        batch_x,batch_y = mnist.train.next_batch(100)
        _,train_loss = sess.run([trainop,loss],feed_dict={x:batch_x,y:batch_y})

        print "train_step = ",step,"train_loss = ",train_loss