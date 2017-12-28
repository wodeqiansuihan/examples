import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#�����˼·�������ģ�
#����ÿ�ε�������һ�� 28��28 ��ͼƬ��
#Ȼ��ÿһ��ʱ��ڵ���������һ�� 1*28 ��vector��
#����n_time = 28

#mnist data 
mnist = input_data.read_data_sets("MNIST_Data",one_hot = True)

#hyperparamters
learning_rate = 0.01
epochs = 10000
batchsize = 128
display_step = 10#�����˼��ÿ��10����ʾһ�¡�

n_inputs = 28#28�����룬�������ÿһ��ʱ��ڵ��ϵ����룬��һ�� 1*28 ��vector��
n_steps = 28 #time steps��Ӧ�þ����ж��ٸ�ʱ��ڵ㡣����28��ʱ��ڵ㡣
#�Ҳ壬Ī������ǲ�����Ӧ�ò��ǲ���.
n_hidder_unit = 80#����Ǹ�ʲô�أ�hidden_unit ����Ӧ�ú�������һ����������Ӧ�����������õģ���Ҫ���ڵ���n_inputs �Ϳ����ˡ�
n_class = 10

#holder input 
#����֮ǰ��˵������������holder ���ܵı�������ʵ�����������������������������������������㣬������һ��һ��ѭ��ȥ������
x = tf.placeholder(tf.float32,[None,n_steps,n_inputs])#������ô����������һ�������أ�
y = tf.placeholder(tf.float32,[None,n_class])

#���weights��bias���������ֶ��巽������ô������ô����������أ���
#weights��bias����                              [28,80]                                                     [80,10]
weights = {'in':tf.Variable(tf.random_normal([n_inputs,n_hidder_unit])),'out':tf.Variable(tf.random_normal([n_hidder_unit,n_class]))}
bias = {'in':tf.Variable(tf.Constant(0.1,shape=[n_hidder_unit,])),'out':tf.Variable(tf.Constant(0.1,shape=[n_class,]))}

def RNN(x,wights,bias):
    #x��һ��ͼ�񼯣�28*28*batch_size ����Ϊ�ּ���˼�롣
    #����Ҫ����ת������ά�ϣ���128*28,28����
    #��ô���ǵ�ʱ��ڵ����ã���Ӧ����28�ˡ�


    #�ؼ�������ô��ת��������������ȥ??
    x = tf.reshape(x,[-1,n_inputs])
    x_in = tf.matmul(x,weights['in']) + bias['in']
    x_in = tf.reshape(x_in,[-1,n_steps,n_hidder_unit])#�������ڹ�������Ҫ�����ݡ��Ҳ�,ΪʲôҪ��ô��������?���Ǻ��鷳��?
    
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