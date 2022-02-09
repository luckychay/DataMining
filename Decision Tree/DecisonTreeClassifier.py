# -*- coding: utf-8 -*-
import numpy as np 
import pandas 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
import os

os.environ["PATH"] += os.pathsep + 'D:/ProgrammFiles/Graphviz/bin/'#在此添加环境变量
#第一步：读入数据
f = open("E:\改训\第二学期\人工智能\数据挖掘\作业\李莎莎老师\决策树实验\settle.csv")
my_data = pandas.read_csv(f, delimiter=",")
featureNames = list(my_data.columns.values)[2:]#获取特征名
targetNames = my_data["settle"].unique().tolist()#获取类标
X = my_data.drop(my_data.columns[[0,1]], axis=1).values#去掉类别获得X
y = my_data["settle"]#类别信息

#第二步：划分测试集和训练集
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.2, random_state=3)

#第三步：初始化决策树
settleTree = DecisionTreeClassifier(criterion="entropy" ,max_depth=5)

#第四步：训练一棵决策树
settleTree.fit(X_trainset,y_trainset)

#第五步：对测试集数据进行分类
predTree =settleTree.predict(X_testset)

#第六步：测试得分
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

#第七步：将生成的决策树存储在"skulltree.png"
dot_data = StringIO()
filename = "settletree.png"
out=tree.export_graphviz(settleTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)

#第八步：从文件中读取决策树并可视化
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')