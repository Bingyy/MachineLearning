import tensorflow as tf 
import numpy as np 
from load_ubyte_image import *

# mnist = 
train_data_filename = "./datasets/mnist/train-images-idx3-ubyte"
train_label_filename = "./datasets/mnist/train-labels-idx1-ubyte"

test_data_filename = "./datasets/mnist/t10k-images-idx3-ubyte"
test_label_filename = "./datasets/mnist/t10k-labels-idx1-ubyte"

imgs, data_head = loadImageSet(train_data_filename)
# 这里的label是60000个数字，需要转成one-hot编码
labels, labels_head = loadLabelSet(train_label_filename)


test_images, test_images_head = loadImageSet(test_data_filename)
test_labels, test_labels_head = loadLabelSet(test_label_filename)


def encode_one_hot(labels):
	num = labels.shape[0]
	res = np.zeros((num,10))
	for i in range(num):
		res[i,labels[i]] = 1 # labels[i]表示0，1，2，3，4，5，6，7，8，9,则对应的列是1，这就是One-Hot编码
	return res

# 定义参数
learning_rate = 0.01
training_epoches = 25
bacth_size = 100 # mini-batch
display_step = 1

# tf graph input
x = tf.placeholder(tf.float32, [None, 784]) # 28 * 28 = 784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 ==> 10 classes

# 模型参数
W = tf.Variable(tf.zeros([784,10])) # tf.truncated_normal()
b = tf.Variable(tf.zeros([10]))

# 构建模型
pred = tf.nn.softmax(tf.matmul(x, W) + b)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.clip_by_value(pred,1e-8,1.0)), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

res = encode_one_hot(labels)

print("res", res)

total_batch = int(data_head[1] / bacth_size)
print("total_batch:", total_batch)

with tf.Session() as sess:
	sess.run(init)

	for epoch in range(training_epoches):
		avg_loss = 0.
		total_batch = int(data_head[1] / bacth_size) # data_head[1]是图片数量

		for i in range(total_batch):

			batch_xs = imgs[i * bacth_size : (i + 1) * bacth_size, 0:784]
			batch_ys = res[i * bacth_size : (i + 1) * bacth_size, 0:10]

			_, l = sess.run([optimizer, loss], feed_dict={x: batch_xs, y: batch_ys})

			# print("loss is: ", l)
			# print("Weights is: ", sess.run(W))

			# 计算平均损失
			avg_loss += l / total_batch

		if epoch % display_step == 0:
			print("Epoch:", '%04d' % (epoch), "loss=", "{:.9f}".format(avg_loss))

	print("Optimization Done!")

	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	print("Accuracy:", accuracy.eval({x: test_images, y: encode_one_hot(test_labels)}))





