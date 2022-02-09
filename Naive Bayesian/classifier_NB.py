import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB#多项式朴素贝叶斯
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


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

#x=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
#y=[]
#for i in x: 
#    print(i)
print('第一步：根据训练数据，生成一个词汇表...')
dictionary = make_Dictionary(TRAIN_DIR)
print('词汇表维度:%d'%len(dictionary))
print()

print("第二步：加载训练数据，抽取特征和标签...")
#    features_matrix, labels = extract_features(TRAIN_DIR)
#    np.save('features_matrix.npy',features_matrix)#将feature_matrix存为npy格式
#    np.save('labels.npy',labels)
features_matrix=np.load('features_matrix.npy')
labels=np.load('labels.npy')
print("第三步：加载测试数据，抽取特征和标签...") 
#    test_features_matrix, test_labels = extract_features(TEST_DIR)
#    np.save('test_features_matrix.npy',test_features_matrix)
#    np.save('test_labels.npy',test_labels)
test_features_matrix=np.load('test_features_matrix.npy')
test_labels=np.load('test_labels.npy')

print('第四步：初始化贝叶斯模型,并设置参数')
clf = MultinomialNB(alpha=0.1,class_prior=[0.1,0.9],fit_prior=False)
#    clf.__init__()
#    clf.set_params(alpha=i, class_prior=[0.3,0.7], fit_prior=True)
print("第五步：依据训练集特征和标签，训练朴素贝叶斯模型...")
clf.fit(features_matrix, labels)

print('第六步：将测试数据特征输入贝叶斯模型，预测邮件类别...')
predicted_labels=clf.predict(test_features_matrix)
  
print("分类结束，分类准确率：")
print(accuracy_score(test_labels,predicted_labels))
#y.append(accuracy_score(test_labels,predicted_labels))
#print(y)
#plt.plot(x, y, 'bo-')
#plt.xlabel('alpha')
#plt.ylabel('Y')
#plt.title("accuracy_score and alpha")
#plt.show()

