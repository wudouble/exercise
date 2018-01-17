import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR
import seaborn as sns
from sklearn import metrics


path = '/Users/wss/Desktop/py/第11周/'
data = pd.read_excel(path + 'bankloan.xls')
x = data.ix[:,:8].as_matrix()
y =data.ix[:,8].as_matrix()

#                 建立随机逻辑回归模型，筛选变量
'''
rlr = RLR()
rlr.fit(x,y)
rlr.get_support()
print(rlr.get_support())
print('通过随机逻辑回归模型筛选特征结束。')
print(u'有效特征为：%s' % ','.join(data.ix[:,rlr.get_support()]))
print('有效特征为：%s' % rlr.scores_(x,y))

x = data.ix[:,rlr.get_support()].as_matrix()
print(x)

#                 建立逻辑回归模型
lr = LR()
lr.fit(x,y)
print('平均正确率：%s' % lr.score(x,y))
'''
x=pd.DataFrame([1.5,2.8,4.5,7.5,10.5,13.5,15.1,16.5,19.5,22.5,24.5,26.5])
y=pd.DataFrame([7.0,5.5,4.6,3.6,2.9,2.7,2.5,2.4,2.2,2.1,1.9,1.8])
print(x)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x,y)
#plt.show()

linreg = LR()
linreg.fit(x,y)
print('Coefficients: \n', linreg.coef_)
y_pred = linreg.predict(x)
print('MSE :',metrics.mean_squared_error(y_pred,y))
print('Variance score: %.2f' % linreg.score(x,y))

#                           多项式模型
x1 = x
x2 = x **2
x1['x2'] = x2
print(x1)
reglr = LR()
reglr.fit(x1,y)
print('Coefficients: \n', reglr.coef_)
y_pred1 = reglr.predict(x1)
print('MSE :',metrics.mean_squared_error(y_pred1,y))

#                            对数模型
x_log = np.log(x)
reglr_log = LR()
reglr_log.fit(x_log,y)
print(">>>>>>>>>>>")

print('Coefficients: \n', reglr_log.coef_)
y_pred_log = reglr_log.predict(x_log)
print(' %s MSE :',metrics.mean_squared_error(y_pred_log,y))

#                         指数模型

y1 = np.log(y)
print(y1)
print("?????????")
#y2 = pd.DataFrame(np.log(y))
#print(y2)
linreg = LR()
linreg.fit(pd.DataFrame(x[0]),y1)

# The coefficients
print('Coefficients: \n', linreg.coef_)

y_pred = linreg.predict(pd.DataFrame(x[0]))
# The mean square error
print ("MSE:",metrics.mean_squared_error(y1,y_pred))

#                         幂函数
lr_mi = LR()
lr_mi.fit(x_log,y1)
y_mi = lr_mi.predict(x_log)
print('L>>>>>>>>>>>>>>>')
print('Coefficients: \n', lr_mi.coef_)
print ("MSE:",metrics.mean_squared_error(y1,y_mi))






