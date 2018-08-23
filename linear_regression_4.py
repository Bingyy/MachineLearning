import tensorflow as tf 
import numpy as np 

# 定义参数
learning_rate = 0.01
training_epoches = 1000
display_step = 50

# 训练数据
train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1]) 
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])

n_samples = train_X.shape[0]

# 模型参数:用tf.Variable来封装
W = tf.Variable(np.random.randn(), name="Weights")
b = tf.Variable(np.random.randn(), name="bias")

# 数据待填充
x = tf.placeholder(dtype=tf.float32) # shape不定的目的是未来后面方便而已
y = tf.placeholder(dtype=tf.float32) # 用于填充真实的label

# 定义模型
pred = W * x + b

# 定义loss和用到的优化器
loss = tf.reduce_sum(tf.pow(pred - y, 2)) / (2 * n_samples)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print("开始训练啦~")
    for epoch in range(training_epoches):
        for(_x,_y) in zip(train_X,train_Y):
            sess.run(optimizer, feed_dict={x:_x, y:_y})
        if epoch % display_step == 0:
            l = sess.run(loss, feed_dict={x:train_X, y:train_Y})
            print("Loss is: ", l)


