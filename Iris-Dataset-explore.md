

```python
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
```


```python
# 导入数据
filename = './data/iris.data.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(filename, names=names) # 这个数据集没有头部，手动指定即可
```


```python
print(dataset.head())
```

       sepal-length  sepal-width  petal-length  petal-width        class
    0           5.1          3.5           1.4          0.2  Iris-setosa
    1           4.9          3.0           1.4          0.2  Iris-setosa
    2           4.7          3.2           1.3          0.2  Iris-setosa
    3           4.6          3.1           1.5          0.2  Iris-setosa
    4           5.0          3.6           1.4          0.2  Iris-setosa


### 现在开始对数据进行审查，加深对数据的了解。

牵涉到如下几个维度：

- 数据的维度
- 数据自身
- 所有的数据特征
- 数据的分布情况


```python
print(dataset.shape)
```

    (150, 5)



```python
# 查看数据自身
print(dataset.head(10))
```

       sepal-length  sepal-width  petal-length  petal-width        class
    0           5.1          3.5           1.4          0.2  Iris-setosa
    1           4.9          3.0           1.4          0.2  Iris-setosa
    2           4.7          3.2           1.3          0.2  Iris-setosa
    3           4.6          3.1           1.5          0.2  Iris-setosa
    4           5.0          3.6           1.4          0.2  Iris-setosa
    5           5.4          3.9           1.7          0.4  Iris-setosa
    6           4.6          3.4           1.4          0.3  Iris-setosa
    7           5.0          3.4           1.5          0.2  Iris-setosa
    8           4.4          2.9           1.4          0.2  Iris-setosa
    9           4.9          3.1           1.5          0.1  Iris-setosa



```python
# 统计数据描述数据
print(dataset.describe())
```

           sepal-length  sepal-width  petal-length  petal-width
    count    150.000000   150.000000    150.000000   150.000000
    mean       5.843333     3.054000      3.758667     1.198667
    std        0.828066     0.433594      1.764420     0.763161
    min        4.300000     2.000000      1.000000     0.100000
    25%        5.100000     2.800000      1.600000     0.300000
    50%        5.800000     3.000000      4.350000     1.300000
    75%        6.400000     3.300000      5.100000     1.800000
    max        7.900000     4.400000      6.900000     2.500000



```python
print(dataset.groupby('class').size())
```

    class
    Iris-setosa        50
    Iris-versicolor    50
    Iris-virginica     50
    dtype: int64


可以看出数据的分布很均匀，**如果分布不均匀，则会影响到模型的准确度。** 如果不均匀，则需要对数据进行处理，使得数据达到相对均匀的状态。方法有：

- 扩大数据样本
- 数据的重新采样
- 生成人工样本
- 异常检测，变化检测


### 数据可视化

图表分成两大类：
- 单变量图表：理解每个特征属性
- 多变量图表：理解不同特征属性之间的关系


```python
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()
```





```python
dataset.hist()
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x115f1f748>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1161c5400>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x1165a1080>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1165dcbe0>]],
          dtype=object)





```python
# 多变量图表
scatter_matrix(dataset) # 这个工具很好用，单变量的直方图 + 变量间的散点分布图
pyplot.show()
```


![png](output_12_0.png)


### 算法评估

使用不同的算法来创建模型，并评估它们的准确度。主要有如下几个步骤：

- 分离出评估数据集
- 10折交叉评估验证算法模型
- 生成6个不同的模型来预测新数据
- 选择最优模型




```python
# 分离数据集
array = dataset.values
```


```python
X = array[:,0:4] # 输入特征,0-1-2-3
Y = array[:, 4]
validation_size = 0.2
seed = 7 # 随机数种子
X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y, test_size=validation_size, random_state=seed)
```


```python
X_train.shape
```




    (120, 4)




```python
Y_train.shape
```




    (120,)



### 使用6种模型
线性算法：

- LR，线性回归
- LDA，线性判别分析

非线性算法:
- KNN，k近邻
- CART，分类与回归树
- NB，贝叶斯分类器
- SVM，支持向量机


```python
models = {}
models['LR'] = LogisticRegression()
models['LDA'] = LinearDiscriminantAnalysis()
models['KNN'] = KNeighborsClassifier()
models['CART'] = DecisionTreeClassifier()
models['NB'] = GaussianNB()
models['SVM'] = SVC()
```


```python
# 算法评估
results = []
for key in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(models[key], X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    print('%s: %f (%f)' % (key, cv_results.mean(), cv_results.std()))
```

    LR: 0.966667 (0.040825)
    LDA: 0.975000 (0.038188)
    KNN: 0.983333 (0.033333)
    CART: 0.975000 (0.038188)
    NB: 0.975000 (0.053359)
    SVM: 0.991667 (0.025000)



```python
# 绘图比较
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(models.keys())
pyplot.show()
```




```python
# 使用评估数据集评估算法
svm = SVC()
svm.fit(X=X_train, y=Y_train)
predictions = svm.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
```

    0.9333333333333333
    [[ 7  0  0]
     [ 0 10  2]
     [ 0  0 11]]
                     precision    recall  f1-score   support
    
        Iris-setosa       1.00      1.00      1.00         7
    Iris-versicolor       1.00      0.83      0.91        12
     Iris-virginica       0.85      1.00      0.92        11
    
        avg / total       0.94      0.93      0.93        30
    

