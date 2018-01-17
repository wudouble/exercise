import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO
import numpy as np
import pandas as pd
from pandas import DataFrame,Series


# path = r'/Users/wss/Desktop/ex1.csv'
# data_origin = open(path,'rt')
# data = csv.reader(data_origin)
# header = next(data)
# for i in data:
#     print(i)

# print(header)
# data = pd.read_csv('/Users/wss/Desktop/ex1.csv')
# print(data)
# head = data.headers()
# print(head)
# feadturelist = []
# labellist = []
# for row in data:
#     labellist.append(row[len(row) - 1])
#     #print(row)
#     row_dict = {}
#     for i in range(1,len(row) - 1):
#         row_dict[header[i]] = row[i]
#     feadturelist.append(row_dict)
# # print(feadturelist)
# #类DictVectorizer可用于将表示为标准Python dict对象列表的要素数组转换为scikit-learn估计量使用的NumPy / SciPy表示。
# vec = DictVectorizer()
#-------------------------------------------------哑变量
# dummy_X = vec.fit_transform(feadturelist).toarray()
# print(dummy_X)
# print(vec.get_feature_names())
# # print(str(dummy))
# lb = preprocessing.LabelBinarizer()       #    标签二值化
# dummy_Y = lb.fit_transform(labellist)
# print('>>>>>>>>>',dummy_Y)
#
# clf = tree.DecisionTreeClassifier(criterion='entropy')
# clf = clf.fit(dummy_X,dummy_Y)
# print('clf : '+str(clf))


#=======---------------------------------logistic regreeion
from numpy import genfromtxt
import numpy as np
from sklearn import datasets,linear_model

'''
path = r'/Users/wss/Desktop/logits.csv'
deliverdata = genfromtxt(path,delimiter=',')
print(deliverdata)

x = deliverdata[:,:-1]
y = deliverdata[:,-1]
print(x)
print(y)

regres = linear_model.LinearRegression()
regres.fit(x,y)
print('coefficients',regres.coef_)
print('intercept',regres.intercept_)
x_pre = [102,6]
# y_predict = regres.predict(x_pre)
# print(y_predict)

'''






