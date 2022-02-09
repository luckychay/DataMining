# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 11:10:59 2019

@author: lss
"""

# coding=utf-8
 
import numpy as np
import matplotlib.pyplot as plt
 
# 二维正态分布数据，数据中每个实例有两个维度，可以看作两个特征
count = 2000;#实例个数
mean1 = [0,0]#实例在不同特征上的均值
cov1 = [[1,5],[5,10]]#实例不同特征的协方差矩阵
data = np.random.multivariate_normal(mean1,cov1,count)

# 绘制散点图
#x,y=data.T
#plt.grid(linestyle="--")
#plt.scatter(x,y,c='b',marker='+')
#plt.savefig("pca data")
#plt.show()

from numpy.linalg import eig
cov_matrix=np.dot(data.T,data)/(count-1) #数据的协方差矩阵
values,vecs=eig(cov_matrix)#对协方差矩阵进行特征值分解
print("Eigenvalues:",values)
print("Eigenvectors:",vecs[:,0],vecs[:,1])

data_new=np.dot(data,vecs[:,1])#数据的协方差矩阵
plt.hist(data_new,bins=80,facecolor='blue')
plt.savefig("直方图")
plt.show()
