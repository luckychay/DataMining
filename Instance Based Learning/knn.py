#-*-coding:utf-8 -*-
from sklearn import datasets  
#导入K近邻算法类                   
from sklearn.neighbors import KNeighborsClassifier 
#导入模型选择中的数据集划分函数
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd
import time  

#第一步 获取数据集
mushroom=pd.read_excel('E:\改训\第二学期\人工智能\数据挖掘\作业\李莎莎老师\基于实例的学习实验\基于实例的学习实验\mushrooms1.xlsx')
mushroom_y=mushroom['class']
mushroom_y=np.array(mushroom_y)
del mushroom['class']
mushroom_x=mushroom 
mushroom_x=np.array(mushroom_x)

#第二步 切分训练集与测试集
X_trainset, X_testset, y_trainset, y_testset = train_test_split(mushroom_x, mushroom_y, test_size=0.3, random_state=3)
p=1
P=[]
times=[]
accuracy=[]
while p<=10:
#定义一个knn分类器对象
    start=time.clock()
    
    knn = KNeighborsClassifier(algorithm='kd_tree',leaf_size=12,n_neighbors=1,p=p) 
    
    #调用该对象的训练方法，主要接收两个参数：训练数据集及其样本标签
    knn.fit(X_trainset, y_trainset)  
    
    #调用该对象的测试方法，主要接收一个参数：测试数据集
    mushroom_y_predict = knn.predict(X_testset) 
    
    #计算各测试样本基于概率的预测
    probility=knn.predict_proba(X_testset)  
    
    #计算与最后一个测试样本距离最近的5个点，返回的是这些样本的序号组成的数组
#    neighborpoint=knn.kneighbors(X_testset,5,True)
    
    #调用该对象的打分方法，计算出准确率
    score=knn.score(X_testset,y_testset,sample_weight=None)
    
    end=time.clock()
    
    accuracy.append(score)
    P.append(p)
    times.append(end-start)
    p+=1
##输出测试的结果
#print('mushroom_y_predict = ')  
#print(mushroom_y_predict)  
#
##输出原始测试数据集的正确标签，以方便对比
#print('y_testset = ')
#print(y_testset)    
#
##输出准确率计算结果
#print ('Accuracy:')
#print (score)  
#
#print ('neighborpoint of last test sample:')
#print (neighborpoint)
# 
#print ('probility:')
#print (probility)
import matplotlib.pyplot as plt
plt.plot(P, times, marker='o', mec='r', mfc='w',label='leaf_size和times曲线图')
plt.xlabel("P") #X轴标签
plt.ylabel("times") #Y轴标签
plt.title("The relationship P and times") #标题

#plt.plot(P, accuracy, marker='o', mec='y', mfc='w')
#plt.xlabel("P") #X轴标签
#plt.ylabel("accuracy") #Y轴标签
#plt.title("The relationship between P and accurary") #标题

