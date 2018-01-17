#encoding:utf-8
import matplotlib.pyplot as plt
from pandas import Series,DataFrame
import pandas as pd
from numpy.random import randn
import numpy as np
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.dates as mdates

#                http://blog.csdn.net/ywjun0919/article/details/8692018
#                http://blog.sina.com.cn/s/blog_b09d460201019c10.html
path = '/Users/wss/Desktop/py/第7周/data/'

plt.subplot(122)
plt.plot([1,2,3,4,5,4,5,6,7])
plt.plot([4,3,2,1],[1,2,3,4])
plt.show()

plt.figure()
x = randn(10)
y = randn(10)
plt.subplot(2,3,1)
plt.plot(x,y)
plt.subplot(232)
plt.barh(x,y)
plt.subplot(233)
plt.bar(x,y)
plt.subplot(234)
plt.scatter(x,y)
plt.subplot(235)
plt.boxplot(x)   #   只有一个参数
plt.subplot(236)
plt.hist(x)     #   只有一个参数
plt.show()

x = randn(10)
y = randn(10)
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(212)
plt.plot(y.cumsum(),'r--')
plt.plot(x.cumsum(),'k--')
_ = ax1.hist(randn(100),bins=90,color='k',alpha = 0.3)
ax2.scatter(np.arange(30),np.arange(30) + 3 * randn(30))
plt.show()
plt.close('all')

#                     http://blog.csdn.net/suzyu12345/article/details/50662226
fig,axes = plt.subplot(2,2,sharex = True,sharey = True)
for i in range(2):
    for j in range(2):
        axes[i,j].hist(randn(50),bins= 50,color = 'K',alpha = 0.5)
plt.subplots_adjust(wspace = 0,hspace = 0)

#设置标题、轴标签、刻度以及刻度标签
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(500),'ko--')
ticks = ax.set_xticks([0,250,500,750,1000])
lables = ax.set_xticklabels(['one','two','three','four','five'],
                            rotation = 30,fontsize = 'small')
ax.set_title("MY PLOT")
ax.set_xlabel('Stages')
ax.set_ylabel('Values')
plt.show()

#     添加图例
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(randn(1000).cumsum(),'k',label = 'one')
ax.plot(randn(1000).cumsum(),'g--',label = 'two')
ax.plot(randn(1000).cumsum(),'r.',label = 'three')
ax.legend(loc = 0)
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)

data = pd.read_csv(path + 'spx.csv',index_col= 0,parse_dates=True)
#print data[:10]
spx = data['SPX']
print spx[:10]
# plt.plot(spx,'k--')   或者
spx.plot(ax = ax,style = 'k--')

crisis_data = [
    (datetime(2007, 10, 11), 'Peak of bull market'),
    (datetime(2008, 3, 12), 'Bear Stearns Fails'),
    (datetime(2008, 9, 15), 'Lehman Bankruptcy')
]
for date,label in crisis_data:
    ax.annotate(label,xy=(date,spx.asof(date) + 50),
                xytext = (date,spx.asof(date)+200),
                arrowprops = dict(facecolor = 'red'),
                horizontalalignment = 'left',
                verticalalignment = 'bottom')
ax.set_xlim(['1/1/2007','1/1/2011'])
ax.set_ylim([600,1800])
ax.set_title("Important dates in 2008-2009 financial crisis")
plt.show()

#           http://blog.csdn.net/u013524655/article/details/41291715
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
rect = plt.Rectangle((0.2,0.75),0.4,0.15,color='k',alpha = 0.3)  #  起点坐标，长，宽，颜色，透明度
cir = plt.Circle((0.7,0.2),0.15,color='b',alpha = 0.3)           # 圆心坐标
pgon = plt.Polygon([[0.15, 0.15], [0.35, 0.4], [0.2, 0.6]],      #  三个点的坐标
                   color='g', alpha=0.5)
ax.add_patch(rect)
ax.add_patch(cir)
ax.add_patch(pgon)
#------------------------------------------------图标的保存到磁盘
fig.savefig(path + 'figpath.svg')
fig.savefig( path + 'figpath.png',dpi = 400,bbox_inches = 'tight')
#      
from io import BytesIO
buffer = BytesIO()
plt.savefig(buffer)
plot_data = buffer.getvalue()

#                           pandas中的绘图函数
plt.close('all')
s = Series(np.random.rand(10),index=np.arange(0,100,10))
s.plot()
print s
df = DataFrame(np.random.randn(10, 4).cumsum(1),
               columns=['A', 'B', 'C', 'D'],
               index=np.arange(0, 100, 10))
df.plot(kind = 'kde')
print df
fig ,axes = plt.subplots(2,1)
data = Series(np.random.rand(16),index=list('abcdefghijklmnop'))
data.plot(kind = 'bar',ax = axes[0],color = 'k',alpha = 0.7)
data.plot(kind = 'barh',ax = axes[1],color = 'k',alpha = 0.7)
plt.show()

np.random.seed(12)
df = DataFrame(np.random.rand(24).reshape(6,4),
                index = ['one', 'two', 'three', 'four', 'five', 'six'],
                columns = pd.Index(['A', 'B', 'C', 'D'], name='Genus'))
df.plot(kind = 'barh')
df.plot(kind = 'bar',stacked = True)
plt.show()

#               柱形图
data = pd.read_csv(path + 'tips.csv')
#print data.day[:10]
#print data['size'][:10]
data1 = pd.crosstab(data.day,data['size'])
#print data1
data1 = data1.ix[:,2:5]
#print data1
data_pict = data1.div(data1.sum(1).astype(float),axis = 0)   #df.div(val,axis) df中的每一项除以val，axis=0/1 指定行或者列 如果val是一个list，则对应每行除以list对应的元素
print data_pict
data_pict.plot(kind = 'bar',stacked = True)
plt.show()

#              直方图、密度图
data = pd.read_csv(path + 'tips.csv')
data['tip_pct'] = data.tip/data.total_bill
print data[:10]
data['tip_pct'].hist(bins=50)
data['tip_pct'].plot(kind = 'kde')
plt.show()


comp1 = np.random.normal(0, 1, size=200)  # N(0, 1)
comp2 = np.random.normal(10, 2, size=200)  # N(10, 4)
values = Series(np.concatenate([comp1, comp2]))

values.hist(bins=100, alpha=0.3, color='k', normed=True)
values.plot(kind='kde', style='k--')
plt.show()

#                       散点图
data = pd.read_csv(path + 'macrodata.csv')
print data[:5]
data1 = data[['cpi','m1','tbilrate','unemp']]
print '---------'
print data1[:4]
trans_data1 = np.log(data1).diff().dropna()   #   计算出对数差，然后去na
#print np.log(data1)[:5]
print trans_data1[:5]
plt.scatter(trans_data1['m1'], trans_data1['unemp'])
plt.title('Changes in log %s vs. log %s' % ('m1', 'unemp'))

pd.scatter_matrix(trans_data1,diagonal = 'kde',color = 'r',alpha = 0.3)
plt.show()















































































