import tensorflow as tf
import numpy as np
import math
import argparse
import os.path
import six
from pylab import *

parse=argparse.ArgumentParser(description='super parameter')
parse.add_argument('-b',dest='BATCH_SIZE',default=10,required=False)
parse.add_argument('-l',dest='learning_rate',default=0.01,required=False)
parse.add_argument('-e',dest='num_epochs',default=100,required=False)
args=parse.parse_args()

BATCH_SIZE=args.BATCH_SIZE
learning_rate=args.learning_rate
num_epochs=args.num_epochs

#BATCH_SIZE=10
#learning_rate=0.01

features=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
features_num=len(features)
'''
data=np.fromfile('./test.txt',sep=' ')              np.fromfile导出的数据是拉长后的一维数据
data=np.reshape(data.shape[0]//features_num,features_num)
'''
data=np.loadtxt('./test.txt')
def normalization(data):
    maxmium=np.amax(data,axis=0)
    minimum=np.amin(data,axis=0)
    means=np.mean(data,axis=0)
    for i in six.moves.range(features_num-1):
        if i!=3:
            data[:,i]=(data[:,i]-means[i])/(maxmium[i]-minimum[i]+0.0001)
    return data

ratio=0.8
data=normalization(data)
offset=math.floor(ratio*data.shape[0])
train_data=data[:offset]
test_data=data[offset:]

def get_batch(data,batch_size=BATCH_SIZE):
    #input_queue=tf.train.slice_input_producer([data[:,:-1],data[:,-1:]],shuffle=True,capacity=100,num_epochs=num_epochs)
    input_queue=tf.train.slice_input_producer([data[:,:-1],data[:,-1:]],shuffle=True)
    batch_x,batch_y=tf.train.shuffle_batch(input_queue,batch_size=BATCH_SIZE,capacity=100,min_after_dequeue=20)
    return batch_x,batch_y

x=tf.placeholder(dtype=tf.float32,shape=[None,13],name='x')
y=tf.placeholder(dtype=tf.float32,shape=[None,1],name='y')

#weights=tf.get_variable('weight',shape=[13,1],dtype=tf.float32,initializer=tf.random_normal_initializer())
#bias=tf.get_variable('bias',shape=[1],dtype=tf.float32,initializer=tf.random_normal_initializer())
weights=tf.Variable(tf.truncated_normal([13,1],stddev=0.1,dtype=tf.float32))
bias=tf.Variable(tf.truncated_normal([1],stddev=0.1,dtype=tf.float32))
y_predict=tf.add(tf.matmul(x,weights),bias)

train_loss=tf.reduce_mean(tf.square(y_predict-y))
test_loss=tf.reduce_mean(tf.square(y_predict-y))
opt=tf.train.GradientDescentOptimizer(learning_rate).minimize(train_loss)

train_x,train_y=get_batch(train_data)
test_x,test_y=get_batch(test_data)

init=tf.global_variables_initializer()
local_init=tf.local_variables_initializer()

gpu_options=tf.GPUOptions(allow_growth=True)
config=tf.ConfigProto(gpu_options=gpu_options)

with tf.Session(config=config) as sess:
    sess.run(init)
    sess.run(local_init)
    print(weights,bias)
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess,coord)
    epoch=0
    loss_plt=[]
    '''
    try:
        while not coord.should_stop():
            train_x_input,train_y_truth=sess.run([train_x,train_y])
            sess.run(opt,feed_dict={x:train_x_input,y:train_y_truth})
            if epoch%2==0:
                test_x_input,test_y_truth=sess.run([test_x,test_y])
                test_loss_value=sess.run(test_loss,feed_dict={x:test_x_input,y:test_y_truth})
                print('epoch:%d eval loss is:%f' % (epoch,test_loss_value))
            epoch+=1
    except tf.errors.OutOfRangeError:
        print('----training end-----')
    finally:
        coord.request_stop()
        print('----program end-----')
    coord.join(threads)
    '''
    while epoch<num_epochs:
        train_x_input,train_y_truth=sess.run([train_x,train_y])
        _,loss=sess.run([opt,train_loss],feed_dict={x:train_x_input,y:train_y_truth})
        if epoch%10==0:
            test_x_input,test_y_truth=sess.run([test_x,test_y])
            test_loss_value=sess.run(test_loss,feed_dict={x:test_x_input,y:test_y_truth})
            print('epoch:%d eval loss is:%f' % (epoch,test_loss_value))
        loss_plt.append(loss)
        epoch+=1
    print('----program end-----')
    epoch_plt=[epoch for epoch in range(num_epochs)]
    scatter(epoch_plt,loss_plt)
    show()
    coord.request_stop()
    coord.join(threads)

        