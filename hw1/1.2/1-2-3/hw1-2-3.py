import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
# number 1 to 10 data
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
session = tf.Session(config=config)

Fs = 10000
f = 5
sample = 10000
x_g = np.arange(sample)

x_g=(x_g+1)/10000
# np.random.shuffle(x)
# print(x[0:10])
y_g = np.sin(5* np.pi * x_g)/(5*np.pi*x_g)

x_g=np.reshape(x_g,(10000,1))
# print(x_g)

y_g=np.reshape(y_g,(10000,1))
# print(y_g)
# x2=np.arange(sample)
# y2=np.sign(np.sin(2*np.pi*5*x))

# print(tf.__version__)
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# Create the model

xs = tf.placeholder(tf.float32, [None, 1],name='x_input')
ys = tf.placeholder(tf.float32, [None, 1],name='y_true')

W_1 = tf.Variable(tf.truncated_normal([1, 32]),name='d_1')
b_1 = tf.Variable(tf.zeros([32]),name='b_1')
W_2 = tf.Variable(tf.truncated_normal([32, 32]),name='d_2')
b_2 = tf.Variable(tf.zeros([32]),name='b_2')
W_3 = tf.Variable(tf.truncated_normal([32, 1]),name='fc1')
b_3 = tf.Variable(tf.zeros([1]),name='b_fc1')


y1=tf.nn.relu(tf.matmul(xs, W_1) + b_1)
y2=tf.nn.relu(tf.matmul(y1, W_2) + b_2)
y = (tf.matmul(y2, W_3) + b_3)


# Define loss and optimizer



# cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(y), reduction_indices=[1]))
cross_entropy_loss = tf.reduce_mean(tf.square(y-ys))
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, ys))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_loss)


grad=tf.gradients(cross_entropy_loss,tf.trainable_variables())
# print(len(grad))
summed_squares = [tf.reduce_mean(tf.square(g)) for g in grad]
grad_norm_loss=tf.sqrt(tf.reduce_mean(summed_squares))
# print(grad_norm_loss)
# for i in range(len(grad)):
# grad_norm_loss=tf.norm(grad[0])

train_step2 = tf.train.AdamOptimizer(1e-4).minimize(grad_norm_loss)


# grad_norm_loss2=tf.norm(tf.gradients(cross_entropy_loss,tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))
# print(grad_norm_loss2)
# # for i in range(len(grad)):
# # grad_norm_loss=tf.norm(grad[0])
#
# train_step3 = tf.train.AdamOptimizer(1e-4).minimize(grad_norm_loss2)
he = tf.hessians(cross_entropy_loss, tf.trainable_variables())
e0 = tf.self_adjoint_eigvals(tf.reshape(he[0], [32, 32]))
e1 = tf.self_adjoint_eigvals(he[1])
e2 = tf.self_adjoint_eigvals(tf.reshape(he[2], [32*32, 32*32]))
e3 = tf.self_adjoint_eigvals(he[3])
e4 = tf.self_adjoint_eigvals(tf.reshape(he[4], [32, 32]))
e5 = tf.self_adjoint_eigvals(he[5])
re=[]
count=0
exit_flag=False
for times in range(110):
    with tf.Session() as sess:
    # important step
        tf.global_variables_initializer().run()
        # exit_flag=False

        for i in range(12000):
            d=i%100
            batch_xs=x_g[d*100:(d+1)*100,:]
            batch_ys =y_g[d*100:(d+1)*100,:]
            # print(batch_xs)
            sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
            if i % 1000 == 0:
                # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(ys, 1))
                # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                a=(sess.run([cross_entropy_loss], feed_dict={xs: batch_xs, ys: batch_ys}))
                # print(times,a)
        # print('done1')


        for i in range(10000):
            d=i%1
            batch_xs=x_g[d*10000:(d+1)*10000,:]
            batch_ys = y_g[d*10000:(d+1)*10000,:]

            ggg = sess.run(grad_norm_loss, feed_dict={xs: batch_xs, ys: batch_ys})
            if(exit_flag):
                # print('ya')
                exit_flag = False
                break
            else:
                if(ggg>0.005):
                    _,gg=sess.run([train_step2,grad_norm_loss], feed_dict={xs: batch_xs, ys: batch_ys})
                    # if i % 500 == 0:
                        # print(i)
                else:

                    l, g_loss, eig0, eig1, eig2, eig3, eig4, eig5 = (
                    sess.run([cross_entropy_loss, grad_norm_loss, e0, e1, e2, e3, e4, e5], feed_dict={xs: batch_xs,
                                                                                                      ys: batch_ys}))
                    min_ratio=0
                    # print(len(eig0) + len(eig0 - 1) + len(eig2) + len(eig3) + len(eig4) + len(eig5))
                    lens=len(eig0)+len(eig0-1)+len(eig2)+len(eig3)+len(eig4)+len(eig5)
                    for t0 in range(len(eig0)):
                        if (eig0[t0]) > 0:
                            min_ratio = min_ratio + 1
                    for t1 in range(len(eig1)):
                        if (eig1[t1]) > 0:
                            min_ratio = min_ratio + 1
                    for t2 in range(len(eig2)):
                        if (eig2[t2]) > 0:
                            min_ratio = min_ratio + 1
                    for t3 in range(len(eig3)):
                        if (eig3[t3]) > 0:
                            min_ratio = min_ratio + 1
                    for t4 in range(len(eig4)):
                        if (eig4[t4]) > 0:
                            min_ratio = min_ratio + 1
                    for t5 in range(len(eig5)):
                        if (eig5[t5]) > 0:
                            min_ratio = min_ratio + 1
                    # print(eig0[0:100])
                    # print(i,'hi')
                    re.append([l,g_loss,min_ratio/lens])
                    # print(l, g_loss, min_ratio/lens)
                    count=count+1
                    # print(count)
                    # print('why')
                    # count=0
                    exit_flag=True


            # sess.run(train_step3, feed_dict={xs: batch_xs, ys: batch_ys})
            # if i % 50 == 0:
            #     # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(ys, 1))
            #     # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            #
            #     l=(sess.run([grad_norm_loss], feed_dict={xs:batch_xs,ys: batch_ys}))
            #
            #     print(l)
            #     # print(eig1)
            # if i % 500 == 0:
            #     # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(ys, 1))
            #     # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            #
            #     l,g_loss,eig0,eig1,eig2,eig3,eig4,eig5=(sess.run([cross_entropy_loss,grad_norm_loss,e0,e1,e2,e3,e4,e5], feed_dict={xs:batch_xs,
            #                                   ys: batch_ys}))
            #     min_ratio=0
            #     print(len(eig0)+len(eig0-1)+len(eig2)+len(eig3)+len(eig4)+len(eig5))
            #     for t0 in range(len(eig0)):
            #         if (eig0[t0])>0:
            #             min_ratio=min_ratio+1
            #     for t1 in range(len(eig1)):
            #         if (eig1[t1])>0:
            #             min_ratio=min_ratio+1
            #     for t2 in range(len(eig2)):
            #         if (eig2[t2])>0:
            #             min_ratio=min_ratio+1
            #     for t3 in range(len(eig3)):
            #         if (eig3[t3])>0:
            #             min_ratio=min_ratio+1
            #     for t4 in range(len(eig4)):
            #         if (eig4[t4])>0:
            #             min_ratio=min_ratio+1
            #     for t5 in range(len(eig5)):
            #         if (eig5[t5])>0:
            #             min_ratio=min_ratio+1
            #     # print(eig0[0:100])
            #     # print(eig1)
            #     # print(l,g_loss,min_ratio)
            #
            #     # print ((he[0]))
            #     # print(e)
            #     # array = np.array(e)
            #     # print(array)
# print(re)
data=np.array(re)
# print(data.shape)
np.save('hw1-2-3-666.npy',data)


plt.xlabel('minimum_ratio')
plt.ylabel('loss')

plt.scatter(data[0:100,2], data[0:100,0])
plt.savefig('loss-minimum_ratio.png')