
# coding: utf-8

# In[2]:


# 导入必要的处理包
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[3]:


# 导入数据
filename = './data/iris.data.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(filename, names=names) # 这个数据集没有头部，手动指定即可


# In[12]:


print(dataset.head())


# ### 现在开始对数据进行审查，加深对数据的了解。
# 
# 牵涉到如下几个维度：
# 
# - 数据的维度
# - 数据自身
# - 所有的数据特征
# - 数据的分布情况

# In[6]:


print(dataset.shape)


# In[10]:


# 查看数据自身
print(dataset.head(10))


# In[11]:


# 统计数据描述数据
print(dataset.describe())


# In[14]:


print(dataset.groupby('class').size())


# 可以看出数据的分布很均匀，**如果分布不均匀，则会影响到模型的准确度。** 如果不均匀，则需要对数据进行处理，使得数据达到相对均匀的状态。方法有：
# 
# - 扩大数据样本
# - 数据的重新采样
# - 生成人工样本
# - 异常检测，变化检测
# 

# ### 数据可视化
# 
# 图表分成两大类：
# - 单变量图表：理解每个特征属性
# - 多变量图表：理解不同特征属性之间的关系

# In[17]:


dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()


# In[18]:


dataset.hist()


# In[21]:


# 多变量图表
scatter_matrix(dataset) # 这个工具很好用，单变量的直方图 + 变量间的散点分布图
pyplot.show()


# ### 算法评估
# 
# 使用不同的算法来创建模型，并评估它们的准确度。主要有如下几个步骤：
# 
# - 分离出评估数据集
# - 10折交叉评估验证算法模型
# - 生成6个不同的模型来预测新数据
# - 选择最优模型
# 
# 

# In[22]:


# 分离数据集
array = dataset.values


# In[28]:


X = array[:,0:4] # 输入特征,0-1-2-3
Y = array[:, 4]
validation_size = 0.2
seed = 7 # 随机数种子
X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y, test_size=validation_size, random_state=seed)


# In[30]:


X_train.shape


# In[32]:


Y_train.shape


# ### 使用6种模型
# 线性算法：
# 
# - LR，线性回归
# - LDA，线性判别分析
# 
# 非线性算法:
# - KNN，k近邻
# - CART，分类与回归树
# - NB，贝叶斯分类器
# - SVM，支持向量机

# In[33]:


models = {}
models['LR'] = LogisticRegression()
models['LDA'] = LinearDiscriminantAnalysis()
models['KNN'] = KNeighborsClassifier()
models['CART'] = DecisionTreeClassifier()
models['NB'] = GaussianNB()
models['SVM'] = SVC()


# In[35]:


# 算法评估
results = []
for key in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(models[key], X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    print('%s: %f (%f)' % (key, cv_results.mean(), cv_results.std()))


# In[36]:


# 绘图比较
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(models.keys())
pyplot.show()


# In[37]:


# 使用评估数据集评估算法
svm = SVC()
svm.fit(X=X_train, y=Y_train)
predictions = svm.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

