# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 12:06:12 2019

@author: chan
"""

# OneR算法实现
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
# 加载iris数据集
dataset = load_iris()
# 加载iris数据集中的data数组（数据集的特征）
X = dataset.data
# 加载iris数据集中的target数组（数据集的类别）
y_true = dataset.target
# 计算每一项特征的平均值
attribute_means = X.mean(axis=0)
# 与平均值比较，大于等于的为“1”，小于的为“0”.将连续性的特征值变为离散性的类别型。
#x = np.array(X >= attribute_means, dtype="int")

   #等宽离散
#x=np.zeros(shape=(150,4))#创建一个空矩阵
#for i in range(X.shape[1]):#取出X的列数
#    temp=pd.cut(X[:,i],10,labels=False)#labels设置为False返回一个列向量
#    x[:,i]=temp
#x=np.zeros(shape=(150,1))
#x[:,0]=pd.cut(X[:,0],4,labels=None) 
  
#等频离散
#x=np.zeros(shape=(150,4))#创建一个空矩阵 
#for i in range(X.shape[1]):#取出X的列数
#    data=pd.Series(X[:,i])
#    temp=pd.qcut(data,10,labels=False,duplicates='drop')#labels设置为False返回一个列向量,duplicates要设置成drop，去重复桶边界
#    x[:,i]=temp 
#x=np.zeros(shape=(150,1))
#x[:,0]=pd.qcut(X[:,0],2,labels=range(2),duplicates='drop')  
#基于误差的离散
#x=np.zeros(shape=(150,1))
#dic=zip(X[:,0],y_true)#zip函数
#new_list=sorted(list(dic))
#l=len(new_list)
#bp_list=[]
#bp=None
#f_num,s_num,t_num=0,0,0
#flag=new_list[0][1]
#for i in range(l):
#   if new_list[i][1]!=flag:
#      bp=(new_list[i][0]+new_list[i-1][0])/2.0#除出来要为小数
#      bp_list.append(bp)
#      flag=new_list[i][1]
#print(bp_list)
#k=0
#for j in range(len(bp_list)):
#    while new_list[k][0]<bp_list[j]:
#        if new_list[k][1]==0:
#            f_num+=1
#        if new_list[k][1]==1:
#            s_num+=1
#        if new_list[k][1]==2:
#            t_num+=1
#        k+=1
#    if max(f_num,s_num,t_num)<3:
#        bp_list[j]=1#避免列表长度动态变化，将应该被删除的断点置1
#    else:
#        f_num,s_num,t_num=0,0,0
#while 1 in bp_list:
#   bp_list.remove(1)#删除所有无用的断点
#bp_list.insert(0,4.29)
#bp_list.append(7.91)
#bp_list1=[]
#for m in bp_list: 
#    if not m in bp_list1: 
#        bp_list1.append(m)
#x[:,0]=pd.cut(X[:,0],bp_list1,labels=range(len(bp_list1)-1))#
#print(temp)

    
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y_true, random_state=14)
from operator import itemgetter
from collections import defaultdict
# 找到一个特征下的不同值的所属的类别。
def train_feature_class(x, y_true, feature_index, feature_values):
    num_class = defaultdict(int)
    for sample, y in zip(x, y_true):
        if sample[feature_index] == feature_values:
            num_class[y] += 1
    # 进行排序，找出最多的类别。按从大到小排列
    sorted_num_class = sorted(num_class.items(), key=itemgetter(1), reverse=True)
    most_frequent_class = sorted_num_class[0][0]
    error = sum(value_num for class_num , value_num in sorted_num_class if class_num != most_frequent_class)
    return most_frequent_class, error
# print train_feature_class(x_train, y_train, 0, 1)
# 接着定义一个以特征为自变量的函数，找出错误率最低的最佳的特征，以及该特征下的各特征值所属的类别。
def train_feature(x, y_true, feature_index):
    n_sample, n_feature = x.shape
    assert 0 <= feature_index < n_feature
    value = set(x[:, feature_index])
    predictors = {}
    errors = []
    for current_value in value:
        most_frequent_class, error = train_feature_class(x, y_true, feature_index, current_value)
        predictors[current_value] = most_frequent_class
        errors.append(error)
    total_error = sum(errors)
    return predictors, total_error
# 找到所有特征下的各特征值的类别，格式就如：{0：({0: 0, 1: 2}, 41)}首先为一个字典，字典的键是某个特征，字典的值由一个集合构成，这个集合又是由一个字典和一个值组成，字典的键是特征值，字典的值为类别，最后一个单独的值是错误率。
all_predictors = {feature: train_feature(x_train, y_train, feature) for feature in range(x_train.shape[1])}
# print all_predictors
# 筛选出每个特征下的错误率出来
errors = {feature: error for feature, (mapping, error) in all_predictors.items()}
# 对错误率排序，得到最优的特征和最低的错误率，以此为模型和规则。这就是one Rule（OneR）算法。
best_feature, best_error = sorted(errors.items(), key=itemgetter(1), reverse=False)[0]
# print "The best model is based on feature {0} and has error {1:.2f}".format(best_feature, best_error)
print(all_predictors[best_feature][0])
# 建立模型
model = {"feature": best_feature, "predictor": all_predictors[best_feature][0]}
# print model
# 开始测试——对最优特征下的特征值所属类别进行分类。
def predict(x_test, model):
    feature = model["feature"]
    predictor = model["predictor"]
    y_predictor = np.array([predictor[int(sample[feature])] for sample in x_test])
    return y_predictor

y_predictor = predict(x_test, model)
print(y_predictor)
# 在这个最优特征下，各特征值的所属类别与测试数据集相对比，得到准确率。
accuracy = np.mean(y_predictor == y_test) * 100
print("The test accuracy is {0:.2f}%".format(accuracy))

from sklearn.metrics import classification_report

print(classification_report(y_test, y_predictor))
