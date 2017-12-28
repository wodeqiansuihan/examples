import os
import tensorflow as tf





with tf.Graph().as_default():
    with tf.name_scope("input_layer"):
     x = tf.placeholder(tf.float32,name="x")
     y = tf.placeholder(tf.float32,name="y")

    w = tf.Variable(tf.zeros([1]),name="w")
    b = tf.Variable(tf.zeros([1]),name="b")
    y_pre = tf.add(tf.multiply(w,x,name="multiply"),b,name="add")

    loss = tf.reduce_mean(tf.pow((y_pre,y),2,name="pow"),name="mean")/2
    train= tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
    evaluate = tf.reduce_mean(tf.pow((y_pre, y), 2, name="pow"), name="mean") / 2


    # sess = with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # for step in xrange(100):
        # sess.run(train,feed_dict={x:})


    writer = tf.summary.FileWriter(logdir="logs",graph=tf.get_default_graph())
    writer.close()



