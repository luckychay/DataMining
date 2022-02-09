# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 00:11:12 2019

@author: lss
"""

# coding=utf-8 
import numpy as np
from sklearn.svm import SVC 
from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#载入数据
features_matrix=np.load('D:\改训\第二学期\人工智能\数据挖掘\作业\李莎莎老师\邮件分类实验\\features_matrix.npy')#读取npy文件
labels=np.load('D:\改训\第二学期\人工智能\数据挖掘\作业\李莎莎老师\邮件分类实验\\labels.npy')
test_features_matrix=np.load('D:\改训\第二学期\人工智能\数据挖掘\作业\李莎莎老师\邮件分类实验\\test_features_matrix.npy')
test_labels=np.load('D:\改训\第二学期\人工智能\数据挖掘\作业\李莎莎老师\邮件分类实验\\test_labels.npy')
X=np.append(features_matrix,test_features_matrix,axis=0)
y=np.append(labels,test_labels,axis=0)
t=0
f=0
for i in range(len(y)):
    if y[i]==0:
        f+=1
    else:
        t+=1
print('正例个数：',t)
print('反例个数：',f)
print('正反样例比为',t/f)
if t/f>=3 or f/t>=1/3:
    print('数据集为不均衡数据集')
    from imblearn.under_sampling import RandomUnderSampler
    ros = RandomUnderSampler(random_state=0)
    X,y=ros.fit_sample(X,y)
#分割训练集和测试集
#X_trainset, X_testset, y_trainset, y_testset = train_test_split(X,y, test_size=0.3, random_state=3)
#svc = SVC(kernel="linear", C=1) 
#svc.fit(X_trainset, y_trainset) 
#y_predicted=svc.predict(X_testset)
#print("Accuracy without decomposition: ", metrics.accuracy_score(y_testset, y_predicted))


#pca = PCA(n_components=20)
#pca.fit(X)
#X_new=pca.transform(X)
#X_trainset, X_testset, y_trainset, y_testset = train_test_split(X_new,y, test_size=0.3, random_state=3)
#svc = SVC(kernel="linear", C=0.01) 
#svc.fit(X_trainset, y_trainset) 
#y_predicted=svc.predict(X_testset)
#print("Accuracy with PCA: ", metrics.accuracy_score(y_testset, y_predicted))
#采用PCA方法进行降维

#采用LDA方法进行降维
lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(X,y)
X_new = lda.transform(X)
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X_new,y,test_size=0.3,random_state=2) # 
svc = SVC(kernel="linear", C=1) 
svc.fit(X_trainset, y_trainset) 
y_predicted=svc.predict(X_testset)
print("Accuracy with LDA: ", metrics.accuracy_score(y_testset, y_predicted))
from sklearn.metrics import classification_report
print()
print(classification_report(y_testset, y_predicted))