# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 17:24:03 2019

@author: chan
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# OneR算法实现
import numpy as np
import pandas as pd
# 加载数据集
#Your code here
df=pd.read_csv('recipes.csv')
y_true=df['country']
y_true=np.array(y_true)#转化为array
del df['country']
x=df
x=np.array(x)


 
#==============================================================================
 #划分数据集
 #train_test_split函数用于将矩阵随机划分为训练子集和测试子集，并返回划分好的训练集测试集样本和训练集测试集标签。
 
 #格式：
 #X_train,X_test, y_train, y_test =cross_validation.train_test_split(train_data,train_target,test_size=0.3, random_state=0)
 
 #参数解释：
 #train_data：被划分的样本特征集
 #train_target：被划分的样本标签
 #test_size：如果是浮点数，在0-1之间，表示样本占比；如果是整数的话就是样本的数量
 #random_state：是随机数的种子。
 #随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
 #随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则：
 #种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。
#==============================================================================

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y_true,random_state=14)
x_train=x
y_train=y_true
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
# print all_predictors[best_feature][0]
# 建立模型
model = {"feature": best_feature, "predictor": all_predictors[best_feature][0]}
print(model)
# print model
# 开始测试——对最优特征下的特征值所属类别进行分类。
def predict(x_test, model):
    feature = model["feature"]
    predictor = model["predictor"]
    y_predictor = np.array([predictor[(sample[feature])] for sample in x_test])
    return y_predictor

y_predictor = predict(x_test, model)
# print y_predictor
# 在这个最优特征下，各特征值的所属类别与测试数据集相对比，得到准确率。
accuracy = np.mean(y_predictor == y_test) * 100
print("The test accuracy is {0:.2f}%".format(accuracy))

from sklearn.metrics import classification_report

print(classification_report(y_test, y_predictor))