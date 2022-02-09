import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE#递归特征消除
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from pylab import *                 #支持中文
mpl.rcParams['font.sans-serif'] = ['SimHei']



def make_Dictionary(root_dir):
    all_words = []
    emails = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]
    for mail in emails:
        with open(mail) as m:
            for line in m:
                words = line.split()
                all_words += words
    dictionary = Counter(all_words)

    list_to_remove = list(dictionary)

    for item in list_to_remove:
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:   
            del dictionary[item]
    

    return dictionary



def extract_features(mail_dir):
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files),len(dictionary)))
    train_labels = np.zeros(len(files))
    count = 0
    docID = 0
    for fil in files:
      with open(fil) as fi:
        for i,line in enumerate(fi):
          if i == 2:
            words = line.split()
            for word in words:
              wordID = 0
              for i,d in enumerate(dictionary):
                if d[0] == word:
                  wordID = i
                  features_matrix[docID,wordID] = words.count(word)
        train_labels[docID] = 0
        filepathTokens = fil.split('/')
        lastToken = filepathTokens[len(filepathTokens) - 1]
        if "spmsg" in lastToken:
            train_labels[docID] = 1
            count = count + 1
        docID = docID + 1
    return features_matrix, train_labels



TRAIN_DIR = "./train-mails"
TEST_DIR = "./test-mails"



#print('根据训练数据，生成一个词汇表...')
#dictionary = make_Dictionary(TRAIN_DIR)
#print('词汇表维度:%d'%len(dictionary))
#print()
#
#print("第二步：加载训练数据，抽取特征和标签...")
#features_matrix, labels = extract_features(TRAIN_DIR)
#np.save('features_matrix.npy',features_matrix)#将feature_matrix存为npy格式
#np.save('labels.npy',labels)
#
#print("第三步：加载测试数据，抽取特征和标签...")
#test_features_matrix, test_labels = extract_features(TEST_DIR)
#np.save('test_features_matrix.npy',test_features_matrix)
#np.save('test_labels.npy',test_labels)


score=0
a=0.1
print('初始化模型...')
#model=MultinomialNB(alpha=a, class_prior=None,fit_prior=True)
#model = svm.SVC(gamma='auto')

#model=DecisionTreeClassifier(criterion="gini")

#==============================================================================
# print('将文本转化为TFIDF表示...')
# train_files=[]
# test_files=[]
# for file in os.walk(TRAIN_DIR):
#     train_files=file[2]
# for file in os.walk(TEST_DIR):
#     test_files=file[2]
#     
# corpus=[]
# for file in train_files:
#     f = open(TRAIN_DIR+'/'+file,"r")   #设置文件对象
#     st = f.read()     #将txt文件的所有内容读入到字符串str中
#     corpus.append(st)
#     f.close()
# for file in test_files:
#     f = open(TEST_DIR+'/'+file,"r")   #设置文件对象
#     st = f.read()     #将txt文件的所有内容读入到字符串str中
#     corpus.append(st)
#     f.close()
# from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf_vec = TfidfVectorizer() 
# tfidf_matrix = tfidf_vec.fit_transform(corpus).toarray()
# 
# print('读取训练数据和测试数据...')
# features_matrix=tfidf_matrix[:100]
# test_features_matrix=tfidf_matrix[100:389]
# labels=np.load('D:\改训\第二学期\人工智能\数据挖掘\作业\李莎莎老师\邮件分类实验\\labels.npy')
# test_labels=np.load('D:\改训\第二学期\人工智能\数据挖掘\作业\李莎莎老师\邮件分类实验\\test_labels.npy')
# 
# print('训练模型...')
# model.fit(features_matrix,labels)
# 
# print('测试模型...')
# predicted_labels=model.predict(test_features_matrix)
# print()
# print('alpha','\t','accuracy_score')
# print(round(a,2),'\t',accuracy_score(test_labels,predicted_labels))
#==============================================================================

print('读取数据...')
features_matrix=np.load('D:\改训\第二学期\人工智能\数据挖掘\作业\李莎莎老师\邮件分类实验\\features_matrix.npy')#读取npy文件
test_features_matrix=np.load('D:\改训\第二学期\人工智能\数据挖掘\作业\李莎莎老师\邮件分类实验\\test_features_matrix.npy')
labels=np.load('D:\改训\第二学期\人工智能\数据挖掘\作业\李莎莎老师\邮件分类实验\\labels.npy')
test_labels=np.load('D:\改训\第二学期\人工智能\数据挖掘\作业\李莎莎老师\邮件分类实验\\test_labels.npy')
#X=np.append(features_matrix,test_features_matrix,axis=0)
#Y=np.append(labels,test_labels,axis=0)
    
    
    #print('进行特征选择...')
    #svc = SVC(kernel="linear", C=0.01)#选择支持向量机作为基模型
    #rfe1 = RFE(estimator=svc, n_features_to_select=500, step=10)#设置选择特征数为500，每次消除的特征数为10
    #rfe1.fit(X, Y)
    #X=rfe1.transform(X)#执行特征选择
    
    #print('重新划分训练集合测试集...')
    #from sklearn.model_selection import train_test_split
    #x_train, x_test, y_train, y_test = train_test_split(X, Y,random_state=14)

model = KNeighborsClassifier(n_neighbors=48)
model.fit(features_matrix,labels)
predicted_labels=model.predict(test_features_matrix)
print()
#print('alpha','\t','accuracy_score')
print('accuracy_score:',accuracy_score(test_labels,predicted_labels))

from sklearn.metrics import classification_report
print()
print(classification_report(test_labels,predicted_labels))




