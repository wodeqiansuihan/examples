import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#整体的思路是这样的：
#我们每次的输入是一个 28×28 的图片。
#然后每一个时间节点上输入是一个 1*28 的vector。
#所以n_time = 28

#mnist data 
mnist = input_data.read_data_sets("MNIST_Data",one_hot = True)

#hyperparamters
learning_rate = 0.01
epochs = 10000
batchsize = 128
display_step = 10#这个意思是每隔10步显示一下。

n_inputs = 28#28个输入，这个才是每一个时间节点上的输入，是一个 1*28 的vector。
n_steps = 28 #time steps。应该就是有多少个时间节点。。有28个时间节点。
#我插，莫非这个是层数？应该不是层数.
n_hidder_unit = 80#这个是干什么呢？hidden_unit 不是应该和输入是一样的数量吗？应该是随意设置的，主要大于等于n_inputs 就可以了。
n_class = 10

#holder input 
#还是之前的说法，我们这里holder 接受的变量，其实是整个传过来的样本集，我们是用整个矩阵运算，而不是一个一个循环去处理。。
x = tf.placeholder(tf.float32,[None,n_steps,n_inputs])#我们怎么样构建这样一个数据呢？
y = tf.placeholder(tf.float32,[None,n_class])

#如果weights和bias是下面这种定义方法，那么我们怎么样反向更新呢？？
#weights和bias定义                              [28,80]                                                     [80,10]
weights = {'in':tf.Variable(tf.random_normal([n_inputs,n_hidder_unit])),'out':tf.Variable(tf.random_normal([n_hidder_unit,n_class]))}
bias = {'in':tf.Variable(tf.Constant(0.1,shape=[n_hidder_unit,])),'out':tf.Variable(tf.Constant(0.1,shape=[n_class,]))}

def RNN(x,wights,bias):
    #x是一个图像集，28*28*batch_size ，因为分集的思想。
    #我们要把它转化到二维上，（128*28,28）。
    #那么我们的时间节点设置，就应该是28了。


    #关键就是怎么样转换到并行运算上去??
    x = tf.reshape(x,[-1,n_inputs])
    x_in = tf.matmul(x,weights['in']) + bias['in']
    x_in = tf.reshape(x_in,[-1,n_steps,n_hidder_unit])#这里是在构建所需要的数据。我插,为什么要这么构建数据?不是很麻烦吗?
    
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidder_unit, forget_bias=1.0, 
                                            state_is_tuple=True)
    _init_state = lstm_cell.zero_state(batchsize,dtype=tf.float32)
    outputs,states = tf.nn.dynamic_rnn(lstm_cell,x_in,initial_state=_init_state,time_major=False)
    
    
    results = None
    return results


pred = RNN(x,weights,bias)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.initialize_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step*batchsize < train_size:
        batch_xs,batch_ys = mnist.train.next_batch(batchsize)
        batch_xs = batch_xs.reshape([batchsize,n_steps,n_inputs])
        sess.run([train_op],feed_dict={x:batch_xs,y:batch_ys})
        
        if step % 20 == 0:
            pass
        step += 1