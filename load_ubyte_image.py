import numpy as np 
import struct
import cv2

def loadImageSet(filename):
	binfile = open(filename, 'rb') # 读取二进制文件
	buffers = binfile.read()

	head = struct.unpack_from('>IIII', buffers, 0) # 读取前四个整数，返回一个元组

	offset = struct.calcsize('>IIII') # 定位到data开始的位置
	imageNum = head[1] # 拿到图片数量
	width = head[2]
	height = head[3]

	bits = imageNum * width * height
	bitsString = '>' + str(bits) + 'B' # fmt格式：'>47040000B'

	imgs = struct.unpack_from(bitsString, buffers, offset) # 取data数据，返回一个元组

	binfile.close()

	imgs = np.reshape(imgs, [imageNum, width * height]) # reshape为[60000,784]型的数组

	return imgs, head

def loadLabelSet(filename):
	binfile = open(filename, 'rb') # 读取二进制文件
	buffers = binfile.read()

	head = struct.unpack_from('>II', buffers, 0) # 读取label文件前两个整形数

	labelNum = head[1]
	offset = struct.calcsize('>II') # 定位到label数据开始的位置
	numString = '>' + str(labelNum) + 'B' # fmt格式：'>60000B'
	labels = struct.unpack_from(numString, buffers, offset) # 取label数据

	binfile.close()

	labels = np.reshape(labels, [labelNum])

	return labels, head


def main():
	train_data_filename = "./datasets/mnist/train-images-idx3-ubyte"
	train_label_filename = "./datasets/mnist/train-labels-idx1-ubyte"

	test_data_filename = "./datasets/mnist/t10k-images-idx3-ubyte"
	test_label_filename = "./datasets/mnist/t10k-labels-idx1-ubyte"


	imgs, data_head = loadImageSet(train_data_filename)

	print(type(imgs))
	print("images_array", imgs)
	print(imgs.shape)

	# 随机取出10个图像的像素点数据，可视化来看一看
	for i in range(10):
		idx = np.random.randint(6000)
		pick_one_image = np.reshape(imgs[idx,:],[28,28]) # 某一行的所有列就是一个图片的像素值
		cv2.imwrite("./datasets/test"+ str(i) + ".jpg", pick_one_image)

	print("data_head: ", data_head)

	##### 现在看一看labels数据 ######
	labels, labels_head = loadLabelSet(train_label_filename)
	print("labels_head",labels_head)
	print(type(labels))
	print("labels_shape",labels.shape)

	print("label: ", labels[0])

	print("part of labels", labels[1:10])

if __name__ == "__main__":
	main()

