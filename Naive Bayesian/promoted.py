import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
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

print('第一步：根据训练数据，生成一个词汇表...')
dictionary = make_Dictionary(TRAIN_DIR)
print('词汇表维度:%d'%len(dictionary))
print()

print("第二步：加载训练数据，抽取特征和标签...")
features_matrix, labels = extract_features(TRAIN_DIR)
np.save('features_matrix.npy',features_matrix)#将feature_matrix存为npy格式
np.save('labels.npy',labels)

print("第三步：加载测试数据，抽取特征和标签...")
test_features_matrix, test_labels = extract_features(TEST_DIR)
np.save('test_features_matrix.npy',test_features_matrix)
np.save('test_labels.npy',test_labels)


score=[]
figure=[]
'''
cpit=0.0003
i=1
cp=[]
while i<=7422:
    cpit=cpit*i/6
    cp.append(cpit)
    i=i+1
cp=[0.1,0.2,0.3,0.04]
'''
a=0.1
#print('第四步：初始化贝叶斯模型,并开始循环')
print('alpha','\t','accuracy_score')
while a<=1:
    model=MultinomialNB(alpha=a, class_prior=None,fit_prior=True)
    
    features_matrix=np.load('features_matrix.npy')#读取npy文件
    labels=np.load('labels.npy')
    model.fit(features_matrix,labels)
    #print(test_labels)
    
    test_features_matrix=np.load('test_features_matrix.npy')
    test_labels=np.load('test_labels.npy')
    predicted_labels=model.predict(test_features_matrix)
    #print(predicted_labels)
    print(round(a,2),'\t',accuracy_score(test_labels,predicted_labels))
    b=accuracy_score(test_labels,predicted_labels)
    c=round(b,6)
    score.append(b)
    figure.append(c)
    a=a+0.1
#print(score)
import matplotlib.pyplot as plt
alpha=['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']
plt.plot(alpha, score, marker='o', mec='r', mfc='w',label=u'alpha和score曲线图')
plt.xlabel("alpha") #X轴标签
plt.ylabel("accury_score") #Y轴标签
plt.title("The relationship between alpha and accuracy_score") #标题
for x, y in zip(alpha, figure):
    plt.text(x, y+0.0004, str(y), ha='center', va='bottom', fontsize=7)
plt.savefig('The figure.jpg')
plt.show()



