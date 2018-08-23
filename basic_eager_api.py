'''
eager可以使得不用显式通过tf.Session来执行
'''
import numpy as np 
import tensorflow as tf 
import tensorflow.contrib.eager as tfe 

# 设置Eager API
print("Setting eager mode")
tfe.enable_eager_execution()

# 定义常数tensor
a = tf.constant(2)
print("a = %i" % a) # 注意这就直接执行了，没通过tf.Session

b = tf.constant(3)
print("b = %i" % b)

c = a + b
print("a + b = %i" % c)

d = a * b
print("a * b = %i" % d)

# 与Numpy兼容
print("混合操作Tensor和Numpy数组")
a = tf.constant([[2.,1.],
				[1.,0.]],dtype=tf.float32)

print("Tensor:\n a = %s" % a)
b = np.array([[3.,0.],
			  [5.,1.]], dtype=tf.float32)

print("NumpyArray:\n b = %s" %b)

# 执行计算而不用tf.Session()
print("Running operations, without tf.Session")

c = a + b
print("a + b = %s" %c)

d = tf.matmul(a, b)
print("a * b = %s" % d)

print("循环")
for i in range(a.shape[0]):
	for j in range(a.shape[1]):
		print(a[i][j])



