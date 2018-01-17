from sklearn import svm
import pylab as pl
import numpy as np
from time import time
from datetime import datetime
#from __future__ import print_function
import logging
import ssl

import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from  sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC

import pandas as pd
#------------------------------      simple eg
# x = [[2,0],[1,1],[2,3]]
# y = [0,0,1]
# clf = svm.SVC(kernel='linear')
# clf.fit(x,y)
# print(clf)
# print(clf.support_vectors_)
# print(clf.support_)        #  支持向量的索引
# print(clf.n_support_)      #  找出每个类里面的支持向量的个数
# print(clf.predict([[1,2]]))


#  ----------------------------        complicate eg
#
# np.random.seed(0)
# x = np.r_[np.random.randn(20,2) -[2,2],np.random.randn(20,2)+[2,2]]   # np.r_按row来组合array，np.c_按colunm来组合array
# y = [0] * 20 + [1] * 20
# clf = svm.SVC(kernel='linear')
# clf.fit(x,y)
# print(x[clf.support_])
# print(clf.support_vectors_)
# w = clf.coef_[0]
# a = -w[0]/w[1]
# xx = np.linspace(-5,5)
# yy = a * xx - (clf.intercept_[0]/w[1])
# print('..............')
# print(clf.support_vectors_)
# b = clf.support_vectors_[0]
# print('LLLll{{{{{{{{')
# print(b)
# yy_down = a * xx + (b[1] - a * b[0])
# b = clf.support_vectors_[-1]
# print(',,,,,,,,,,,')
# print(b)
# yy_up = a * xx + (b[1] - a * b[0])
#
# print('>>>>>>>>>>>')
# print('w',w)
# print('a',a)
# print('support_vectors :',clf.support_vectors_)
# print('clf.coef :',clf.coef_)
#
# pl.plot(xx,yy,'k--')
# pl.plot(xx,yy_down,'k--')
# pl.plot(xx,yy_up,'k--')
#
#
# pl.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],s = 80,facecolors = 'none')
# pl.scatter(x[:,0],x[:,1],c = y,cmap = pl.cm.Paired)
#
# pl.axis('tight')
# pl.show()


#------------------------------------人脸识别

# print(datetime.now())
# print(time())
import PIL
#                           日志的配置函数  http://blog.csdn.net/fengleieee/article/details/54862877
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s') # %(asctime)s: 打印日志的时间, %(message)s: 打印日志信息
ssl._create_default_https_context = ssl._create_unverified_context   #   为了81行下载数据集
lfw_people = fetch_lfw_people(min_faces_per_person=70,resize=0.4)
n_samples,h,w = lfw_people.images.shape
print(n_samples,h,w)
x = lfw_people.data
print(x.shape)
n_features = x.shape[1]

print(n_features)

y = lfw_people.target
print(y)
target_names = lfw_people.target_names
n_classes = target_names.shape[0]
print(target_names)
print('Total dataset size:')
print("n_samples:%d" % n_samples)
print("n_features:%d" % n_features)
print("n_classes:%d" % n_classes)

#   拆分数据，分成训练集和测试集,有个现成的函数，通过调用train_test_split；来分成两部分
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.25)

#   数据降维，因为特征值的维度还是比较高
n_components = 150

print('Extracing the top %d eigenfaces from %d faces'%(n_components,n_features))
t0 = time()            #计算出打印每一步需要的时间
pca = RandomizedPCA(n_components = n_components,whiten= True).fit(X_train)
print('done in %0.3fs'%(time()-t0))

#                    提取人脸图片中的特征值
eigenfaces = pca.components_.reshape((n_components,h,w))
print(';;;;;;;;;;;;;;;;')
print(eigenfaces.shape)
t0 = time()
print(X_train.shape)
#                特征量中训练集所有的特征向量通过pca转换成更低维的矩阵
X_train_pca = pca.transform(X_train)
print(X_train_pca.shape)
X_test_pca = pca.transform(X_test)
print('done in %0.3fs'%(time()-t0))


#              Fitting the classifier to the training set
t0 = time()

#             c为权重，对错误进行惩罚，根据降维之后的数据结合分类器进行分类
#            gamma为核函数的不同表现，表示有多少特征能够被表示，表示比例
param_grid = {'C':[1e3,5e3,1e4,5e4,1e5],
              'gamma':[0.0001,0.0005,0.001,0.005,0.01,0.1]}

#              建立分类器模型，找出表现最好的核函数
clf = GridSearchCV(SVC(kernel='rbf',class_weight='balanced'),param_grid)
#              训练模型
clf = clf.fit(X_train_pca,y_train)
print('done in %0.3fs '%(time()-t0))
print(clf.best_estimator_)

#               测试集预测
t0 = time()
y_pred = clf.predict(X_train_pca)
print('/>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
print(y_pred)
print('done in %0.3fs '%(time() - t0))

#              ???????????
# print(classification_report(y_test,y_pred,target_names = target_names))
# print(confusion_matrix(y_test,y_pred,labels=range(n_classes)))


# 把数据可视化的可以看到，把需要打印的图打印出来
def plot_gallery(images,titles,h,w,n_row = 3,n_col = 4):
    plt.figure(figsize=(1.8 * n_col,2.4 * n_row))
    plt.subplots_adjust(bottom = 0,left = .01,right = 0.99,top = 0.90,hspace = 0.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row,n_col,i + 1)
        plt.imshow(images[i].reshape((h,w)),cmap = plt.cm.gray)
        plt.title(titles[i],size = 12)
        plt.xticks(())
        plt.yticks(())

 # 把预测的函数归类标签和实际函数归类标签
def title(y_pred,y_test,target_names,i):
    pred_name = target_names[y_pred[i]].rsplit('',1)[-1]
    true_name = target_names[y_test[i]].rsplit('',1)[-1]
    return 'predicted:%s\ntrue:    %s'%(pred_name,true_name)

#    把预测出来的人名存起来
prediction_titles = [title(y_pred,y_test,target_names,i)
                     for i in range(y_pred.shape[0])]


plot_gallery(X_test,prediction_titles,h,w)

eigenface_titles = ['eigenface %d '% i for i in range(eigenfaces.shape[0])]

plot_gallery(eigenfaces,eigenface_titles,h,w)
plt.show()
















