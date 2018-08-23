import tensorflow as tf 
import numpy as np 

# 定义超参数
learning_rate = 0.01
training_epoches = 1000

display_step = 50

train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1]) 
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])

n_samples = train_X.shape[0]

# 定义模型参数
W = tf.Variable(np.random.randn(), name="Weights")
b = tf.Variable(np.random.randn(), name="bias")

# 制作模型
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

pred = W * X + b # 拟合函数

loss = tf.reduce_sum(tf.pow(pred - Y,2)) / (2 * n_samples)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for epoch in range(training_epoches):
		for (x,y) in zip(train_X, train_Y):
			sess.run(optimizer, feed_dict={X:x, Y:y})

		if epoch % 50 == 0:
			loss_res = sess.run(loss, feed_dict={X:train_X, Y:train_Y})
			print("Loss", loss_res)









