
# coding: utf-8

# 对数据进行理解的最快最有效的方式是：数据的可视化。

# ### 单一图表
#     
# - 直方图
# - 密度图
# - 箱线图
# 
# #### 直方图
# 
# 通过直方图可以非常直观地看出每个属性的分布状况：高斯分布，指数分布还是偏态分布。

# In[4]:


from pandas import read_csv
import matplotlib.pyplot as plt

filename = 'data/pima_data.csv'
# names = ['Number of times pregnant', 
#          'Plasma glucose concentration a 2 hours in an oral glucose tolerance test', 
#          'Diastolic blood pressure (mm Hg)',
#          'Triceps skin fold thickness (mm)',
#          '2-Hour serum insulin (mu U/ml)',
#          'Body mass index (weight in kg/(height in m)^2)',
#          'Diabetes pedigree function',
#          'Age (years)',
#          'Class variable (0 or 1)'
#         ]
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names) # 手动指定头部


# In[6]:


# 直方图
data.hist()
plt.show()


# #### 密度图
# 也是用于显示数据分布的图表，类似于对直方图进行抽象，用平滑的曲线来描述数据的分布。

# In[7]:


data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
plt.show()


# #### 箱线图
# 
# 用于显示数据分布， 中位线 + 上下四分数线 + 上下边缘线。

# In[9]:


data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
plt.show()


# ### 多重图表
# 
# 主要是两种图表：
# 
# - 相关矩阵图
# - 散点矩阵图
# 
# #### 相关矩阵图
# 
# 用于展示两个不同属性相互影响的程度。把所有的属性两两影响的关系展示出来的图就是相关矩阵图。
# 
# #### 散点矩阵图
# 
# 两组数据构成多个坐标点，两两特征之间的数据散点图组合成一个散点矩阵图。

# In[11]:


# 相关矩阵图
from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np

filename = 'data/pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names) 

correlations = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()


# In[12]:


# 散点矩阵图

from pandas.plotting import scatter_matrix
scatter_matrix(data)
plt.show()


# 这个使用是非常简单的，三行代码即可。

# ### 总结
# 
# 结合前面的7种审查数据的武器 + 这里讲到的数据可视化的方法，现在拿到一个CSV数据集，我们就可以迅速对数据集进行审查，然后加深对数据的理解，这个过程中解题的思路也会慢慢清晰。
# 
# 
