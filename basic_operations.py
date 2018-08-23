import tensorflow as tf 

# 常数，这种构建方法定义的是计算图结点
a = tf.constant(2)
b = tf.constant(3)

# 加载默认图
with tf.Session() as sess:
	print("a=2, b=3")
	s = sess.run(a+b)
	m = a * b
	print("求和为：" % s)
	print("求积为："% m)

# 动态运行时注入数据
a = tf.placeholder(dtype=tf.float32, shape=[1,])
b = tf.placeholder(dtype=tf.float32, shape=[1,])

# 定义一些操作，因为数据是动态输入，还不是计算结点，这里特别定义
add = tf.add(a,b)
mul = tf.multiply(a, b)

with tf.Session() as sess:
	res_add = sess.run(add, feed_dict={a: 2, b: 3})
	res_mul = sess.run(mul, feed_dict={a: 2, b: 3})

	print(res_add, res_mul)

#### 矩阵计算 ####

# constant op
matrix_1 = tf.constant([[3., 3.]]) # 1 x 2
# constant op
matrix_2 = tf.constant([[2.],[2.]]) # 2 x 1

# 矩阵乘法操作
product = tf.matmul(matrix_1, matrix_2)

with tf.Session() as sess:
	result = sess.run(product)
	print(result) # ==> 3x2 + 3x2 = 12.

### 可以从命令行获取参数，然后解析，调用tf.placeholder动态填充调用 ###










