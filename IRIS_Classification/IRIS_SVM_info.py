
print("python_02!!!_____Garbage")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris
import pandas as pd

#：Iris Setosa，Iris Versicolour，Iris Virginica。每类收集了50个样本，因此这个数据集一共包含了150个样本。

columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels'] # As per the iris dataset information

# Load the data
df = pd.read_csv('iris.data', names=columns) # Pandas 读取一个名为 'iris.data' 的 CSV 文件 // names=columns：这个参数是用来自定义列名的
df.head() # 看前5行数据
print(df.head())
df.describe()  # 显示每列的统计信息（均值、最大、最小等）
print(df.describe())

sns.pairplot(df, hue='Class_labels')
# plt.show()

data = df.values #提取整个 DataFrame 中的所有数据，并转成 NumPy 的二维数组。
print(data)
print(len(data)) # 150行 因为  Iris 数据集一共有 150 条样本，每一行是一个样本，每一列是一个特征
            # #如 [5.7 2.6 3.5 1.0 'Iris-versicolor']

X = data[:,0:4]    # 把4个属性拿走了，
Y = data[:,4]       # 把花的名字拿走了

