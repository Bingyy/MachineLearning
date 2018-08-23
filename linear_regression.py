import numpy as np 
import tensorflow as tf 

# 从最简单的一次方程开始推导
# y = W * x + b

# Parameters
learning_rate = 0.01
training_epoches = 10000
display_step = 50

# 准备训练数据
# asarray不会拷贝元数据，相当于指针
train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1]) 
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])

n_samples = train_X.shape[0]

# 定义训练时填充数据，这里的X和Y
X = tf.placeholder(dtype=tf.float32)
Y = tf.placeholder(dtype=tf.float32) # label

# 定义模型参数
W = tf.Variable(np.random.randn(),name="Weights") # tf.Variable第一个参数是值，所以它自己本身不用指定shape，只是一个封装
b = tf.Variable(np.random.randn(), name="bias")

# 定义模型
pred = tf.add(tf.multiply(X,W),b) # tf.multiply(W,X) + b

# 定义损失函数
mse = 0.5 * tf.reduce_sum(tf.pow(pred - Y,2)) / n_samples
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mse)

# 初始化所有变量
init = tf.global_variables_initializer()

# 执行计算图
with tf.Session() as sess:
    sess.run(init) # 真正执行初始化

    # 开始训练：就是要不停计算mse，而mse要依赖输入数据
    for epoch in range(training_epoches):
        for (x,y) in zip(train_X, train_Y): # 每次填充一个样例（x,y）,可是怎么计算总体损失函数呢
            print("Shape of x", x.shape)
            # optimizer需要的数据是什么，就填什么，这里的X是类型不定的，所以一会是单个数字，一会是数组
            sess.run(optimizer, feed_dict={X:x, Y:y}) # 这里的内层函数填充的是单个数字而非数组啊，而optimizer依赖的是数组对吧
            result = sess.run(pred,feed_dict={X:x, Y:y})
            print("Predicton is: ",result) # 单个值

        if epoch % display_step == 0:
            # 这里计算mse，需要全部样例的参与，因此是X:train_X, Y:train_Y
            res = sess.run(mse, feed_dict={X:train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch), "cost=", "{:.9f}".format(res), \
                "W=", sess.run(W), "b=", sess.run(b))


    print("DONE!")            

