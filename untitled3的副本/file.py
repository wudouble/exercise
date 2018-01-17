# '''f = open('g.txt','r')   #   读取子目录
# fd = f.read(90)
# print(fd)
# f.close()
#
#
# f1 = open(fn,'r')
# fd1 = f1.read()
# print(fd1)
# '''
#
#
# a=[98,98,98,83,65,72,79,76,75,94,91,77,63,83,89,69,64,78,63,86,91,72,71,72,70,80,65,70,62,74,71,76]
# def get_mode(arr):
#     li = []
#     li_appear = dict((x,arr.count(x)) for x in arr)
#     if max(li_appear.values()) == 1:
#         return
#     else:
#         for (x,y) in li_appear.items():
#             if y == max(li_appear.values()):
#                 li.append(x)
#                 print(y)
#     return li
# print(get_mode(a))
import pandas
from datetime import datetime
from datetime import timedelta
# i = 0
# # print(timedelta(7,5000))
# print(datetime(2014-12-06 02))
# b = datetime(2014, 12, 5)-timedelta(7,5000)
# print(b)
# print(datetime.now())
# for i in range(1,7):
#     print(i)
# for i_days in range(0, 31, 7):
#
#     start_date = datetime(2014, 12, 5)
#
#     vali_date = start_date - timedelta(i_days)
#     print(i_days,vali_date)
# import numpy as np
# a = 12
# b = 'sd'
# dict = {}
# if not dict.__contains__((a,b)):
#     dict[(a, b)] = [0, 0, 0, 0]
# print(dict)
# a1 = [3,4,2]
# a2 = [1,2,3]
# #ss = a1 - a2
# #s = np.linalg.norm(a1-a2])
# def fun(a,b):
#     return a-b
# print(list(map(fun,a1,a2)))
# a11 = np.asmatrix(a1)
# print(a11)
# a22 = np.asmatrix(a2)
# print(a22)
# print(a11 - a22)
# print(np.linalg.norm(np.array(a1) - np.array(a2)))
# print(np.array(a1) - np.array(a2))
# import pandas as pd
# path = '/Users/wss/Desktop/'
# data = pd.read_csv(path + 'kaggle_bike_competition_train.csv',header= 0,error_bad_lines=False)
# # print(data.head())
# temp = pd.DatetimeIndex(data['datetime'])
# # print(temp[:10])
# data['date'] = temp.date
# data['time'] = temp.time
# print(data.head())
# data['hour'] = pd.to_datetime(data.time,format="%H:%M:%S")
# data['hour'] = pd.Index(data['hour']).hour
# print(data[:5])
# data['day_of_week'] = pd.DatetimeIndex(data['date']).dayofweek
# data['number_of_days'] = (data.date - data.date[0]).astype('timedelta64[D]')
# byday = data.groupby('day_of_week')
# # print(byday['casual'].sum().reset_index())
# # print(byday['registered'].sum().reset_index())
# byday=data['casual'].groupby(data['day_of_week']).sum().reset_index()
# print(byday)
#encoding:utf-8

#---------------------------          http://blog.csdn.net/han_xiaoyang/article/details/49797143

path = '/Users/wss/Desktop/py/'
def set_ch():
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['FangSong']
    mpl.rcParams['axes.unicode_minus'] = False
set_ch()
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
#                    显示中文
from matplotlib.font_manager import FontManager, FontProperties
def getChineseFont():
    return FontProperties(fname='/System/Library/Fonts/PingFang.ttc')

data_train = pd.read_csv(path + 'train.csv')
print(data_train.columns)
# pd.set_option("display.width",5)
# print(data_train.info())
print(data_train[:10])
print(data_train.describe())

import matplotlib.pyplot as plt
import matplotlib as mpl
fig = plt.figure()
fig.set(alpha = 0.3)
plt.subplot2grid((2,3),(0,0))
data_train.Survived.value_counts().plot(kind = 'bar')
plt.title(u"获救情况",fontproperties=getChineseFont())
plt.ylabel(u"人数",fontproperties=getChineseFont())

plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind = 'bar')
plt.title(u"获救情况",fontproperties=getChineseFont())
plt.ylabel(u"乘客等级分布",fontproperties=getChineseFont())

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived,data_train.Age)
plt.title(u"获救情况",fontproperties=getChineseFont())
plt.grid(b=True,which='major',axis='y')
plt.ylabel(u"年龄",fontproperties=getChineseFont())

plt.subplot2grid((2,3),(1,0),colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind = 'kde')
data_train.Age[data_train.Pclass == 2].plot(kind = 'kde')
data_train.Age[data_train.Pclass == 3].plot(kind = 'kde')
plt.title(u"各等级的乘客年龄分布",fontproperties=getChineseFont())
plt.ylabel(u"密度",fontproperties=getChineseFont())
plt.xlabel("年龄",fontproperties=getChineseFont())
plt.legend(('First class', 'Second class','Third class'),loc = 0)

plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind = 'bar')
plt.title("各登船口岸上船人数",fontproperties=getChineseFont())
plt.ylabel("人数",fontproperties=getChineseFont())
# plt.show()

fig = plt.figure()
fig.set(alpha = 0.2)

Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df = pd.DataFrame({'un_saved':Survived_0, 'saved':Survived_1})
df.plot(kind = 'bar',stacked = True)
plt.title(u"各乘客等级的获救情况",fontproperties=getChineseFont())
plt.xlabel(u"乘客等级",fontproperties=getChineseFont())
plt.ylabel(u"人数",fontproperties=getChineseFont())
# plt.show()

#------------------------------   插值法

from sklearn.ensemble import RandomForestRegressor

def set_missing_age(df):
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    y = known_age[:,0]
    x = known_age[:,1:]

    rfr = RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
    rfr.fit(x,y)
    predict_age = rfr.predict(unknown_age[:, 1::])
    df.loc[(df.Age.isnull()),'Age'] = predict_age
    return df,rfr

def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()),'Cabin'] = 'Yes'
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df
data_train,rfr = set_missing_age(data_train)
data_train = set_Cabin_type(data_train)
print(data_train[:50])


dummies_Cabin = pd.get_dummies(data_train['Cabin'],prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'],prefix='Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'],prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

df = pd.concat([data_train,dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
print('________________________________')
print(df[:10])

#                   Age和Fare两个属性  变化幅度太大  进行标准化
from sklearn.preprocessing import StandardScaler
# import sklearn.preprocessing as preprocessing

# scaler = preprocessing.StandardScaler()
# age_scale_param = scaler.fit(df['Age'])
# df['Age_scaled'] = scaler.fit_transform(df['Age'],age_scale_param)
# fare_scale_param = scaler.fit(df['Fare'])
# df['Fare_scaled'] = scaler.fit_transform(df['Fare'],fare_scale_param)

scaler = StandardScaler()
ss_age = (df['Age'] - df['Age'].min())/(df['Age'].max() - df['Age'].min())
ss_age.columns = ['scaled_age']
ss_fare = (df['Fare'] - df['Fare'].min())/(df['Fare'].max() - df['Fare'].min())
df1 = df.drop(['Age','Fare'],axis = 1,inplace=True)
df = pd.concat([df,ss_age,ss_fare],axis=1)

# df['Fare_scaled'] = scaler.fit_transform(df['Fare'], fare_scale_param)
print('+++++++++++++++++++++++++++++')
print(df[:10])
print('___________________________')
# print(df[:10])

from sklearn import linear_model
print(df.info())
train_df = df.filter(regex='Survived|Age|Fare|SibSp|Parch|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
print(train_df.info())
train_np = train_df.as_matrix()
y = train_np[:,0]
x = train_np[:,1:]
clf = linear_model.LogisticRegression(C = 1.0,penalty='l1',tol=1e-6)
clf.fit(x,y)

print("''''''''''''''''''''''''''''''''''''")
print(pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)}))

from sklearn import cross_validation

print('p;;;;;;;;;;;;;;;;;;;;;;;;;;;;')
print(cross_validation.cross_val_score(clf,x,y))

from sklearn.ensemble import BaggingRegressor
bagging_clf = BaggingRegressor(clf,n_estimators=20,max_samples=0.8,max_features=1.0,bootstrap=True,bootstrap_features=False,n_jobs=-1)
bagging_clf.fit(x,y)












