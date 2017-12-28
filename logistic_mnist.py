from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

minist_data = input_data.read_data_sets("MNIST_Data",one_hot = True)

x = tf.placeholder(tf.float32,shape=[None,784])
y_ = tf.placeholder(tf.float32,shape=[None,10])

w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,w) + b)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_predict = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_predict,tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch = minist_data.train.next_batch(50)
        sess.run(train_step,feed_dict={x:batch[0],y_:batch[1]})
        if i%20 == 0:
            print sess.run(accuracy,feed_dict={x:minist_data.test.images,y_:minist_data.test.labels})




