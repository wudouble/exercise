#encoding:utf-8
from numpy import *
import pandas as pd
import seaborn as sns
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RandomizedLogisticRegression as RLR
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame,Series


'''
path = '/Users/wss/Desktop/py/第11周/'
data = pd.read_excel(path + 'bankloan.xls')
x = data.ix[:,:8].as_matrix()
y =data.ix[:,8].as_matrix()
rlr = RLR(selection_threshold=0.5)
rlr.fit(x,y)
rlr.get_support()
print(rlr.get_support())
print('通过随机逻辑回归模型筛选特征结束。')
print u'有效特征为：%s' % ','.join(data.columns[rlr.get_support()])
#x = data[data.columns[rlr.get_support()]].as_matrix()
from datetime import datetime
from datetime import timedelta
from dateutil.parser import parse
import numpy as np
from pandas import DataFrame,Series
import pandas as pd
from pandas.tseries.offsets import Hour,Minute,Day,MonthEnd
import matplotlib.pyplot as plt
'''
path = '/Users/wss/Desktop/py/第12周/'
'''
now = datetime.now()
print(now)
print(now.year,'\n',now.month,'\n',now.day)
print(datetime(2017,3,23))
delta = datetime(2003,7,1)-datetime(2002,1,3)
print(delta.days,delta.seconds)


start = datetime(2012,11,3)
m = start + timedelta(12,10000)     #    天 和 秒
print(m)


# 字符串转日期
stamp = datetime(2012,11,3,23,3,4)
print(stamp)
print(type(stamp))
print(str(stamp))
print(type(str(stamp)))
time_s = '2023/3/2'
print(type(time_s))
time_d = datetime.strptime(time_s,"%Y/%m/%d")
print(time_d,type(time_d))


dates = ['7/6/2011', '8/6/2011']
li = []
li = [datetime.strptime(x,"%d/%m/%Y") for x in dates]
#print(li)


#                             dateutil可以解析几乎所有人类能够理解的日期表示形式
p = parse('11/12/2023',dayfirst = True)
print(p)
print(parse('Jan 31, 1997 10:45 PM'))

#                pandas 中的时间序列
#               pandas通常是用于处理成组日期的，不管这些日期是DataFrame的轴索引还是列。to_datetime方法可以解析多种不同的日期表示形式。
print(pd.to_datetime(dates))
idx = pd.to_datetime(dates + [None])
print(idx)
print(idx[2])
print(pd.isnull(idx))


dates = [datetime(2011, 1, 2), datetime(2011, 1, 5), datetime(2011, 1, 7),
         datetime(2011, 1, 8), datetime(2011, 1, 10), datetime(2011, 1, 12)]
ts = Series(np.random.randn(6), index=dates)
print(ts)
print(type(ts))
print(ts.index)
print(ts[::2])
print(ts + ts[::2])    #    NaT（Not a Time）是pandas中时间戳数据的NA值
print(ts.index.dtype)
print(ts.index[0])

#索引、选取与子集构造
temp = ts.index[2]
print(ts[temp])
print(ts['20110102'])
print(ts['2011/1/2'])

index = pd.date_range('2011/1/2',periods=1000)
print(index)
long_ts = Series(np.random.randn(1000),index=index)
print(long_ts)
print('..........................')
print(long_ts['2013'])
print('///////////////////////////////')
print(long_ts[datetime(2011,1,7):])   #       输出此日期之后的值
print(long_ts['2011':'2011/1/7'])      #      时间索引切片
print('\n')
print('...................,,,,,,,,,,,,////////////////')
print(long_ts.truncate(after='2011/1/23',before='2011/1/12'))  #   truncate()截取
print(long_ts['2011/1/12':'2011/1/23'])


dates = pd.date_range('1/1/2000', periods=100, freq='W-WED') #   表示每个星期三    freq：string或DateOffset，
# 默认值是’D’，表示以自然日为单位，这个参数用来指定计时单位，比如’5H’表示每隔5个小时计算一次
# print(dates)
long_df = DataFrame(np.random.randn(100, 4),
                    index=dates,
                    columns=['Colorado', 'Texas', 'New York', 'Ohio'])
print(long_df)
print('?????????????????????')
print(long_df.ix['5-2001'])

#Note: pandas时间序列series的index必须是DatetimeIndex不能是Index，
#也就是如果index是从dataframe列中转换来的，其类型必须是datetime类型，不能是date、object等等其它类型！否则在很多处理包中会报错。

dates = pd.DatetimeIndex(['1/1/2000', '1/2/2000', '1/2/2000', '1/2/2000',
                          '1/3/2000'])
S_date = Series(np.arange(5),index=dates)
print(S_date)
print(S_date.index.is_unique)
print(S_date['2000/1/2'])
grouped = S_date.groupby(S_date.index).count()
print(grouped)
grouped1 = S_date.groupby(level=0).count()
print(grouped1)



#
#                          日期范围、频率与移动
dates = [datetime(2011, 1, 2), datetime(2011, 1, 5), datetime(2011, 1, 7),
         datetime(2011, 1, 8), datetime(2011, 1, 10), datetime(2011, 1, 12)]
ts = Series(np.random.randn(6), index=dates)
print(ts)
print(ts.resample('D').mean())
da = pd.date_range('1/1/2000', '12/1/2000', freq='BM')
print(da)
print(pd.date_range('5/2/2012 12:56:31', periods=5,normalize=True))  #产生一组被规范化午夜的时间戳，可用normalize实现


hour = Hour(4)     #传入一个整数即可定义偏移量的倍数
print(hour)
data = pd.date_range('1/1/2000', '1/3/2000 23:59',freq= hour)
print(data)
data1 = pd.date_range('1/1/2000',periods= 10 ,freq='1h30min')
print(data1)
rng = pd.date_range('1/1/2012', '9/1/2012', freq='WOM-3FRI')
print(list(rng))


ts = Series(np.random.randn(4),
            index=pd.date_range('1/1/2000', periods=4, freq='M'))

print(ts.shift(2))
print(ts.shift(1))
print(ts/ts.shift(1) - 1)
print(ts)
print(ts.shift(2,freq='3D'))


now = datetime(2017,1,30)
print(now)
print(now + 3*Day())
print(now + MonthEnd(4))
offset = MonthEnd()
print(offset.rollback(now))
print(offset.rollforward(now))
ts = Series(np.random.randn(20),
            index=pd.date_range('1/15/2000', periods=20, freq='4d'))
print(ts)
print(ts.groupby(offset.rollforward).count())
print(ts.resample('M').mean())
print(ts.resample('M',how = 'mean'))

'''
data = pd.read_csv(path + 'stock_px.csv',usecols=[0,1,2,3],parse_dates=True,index_col=0)      #用作行索引的列编号或者列名
print(data.head(10))
#print(data[['AAPL','MSFT']].head())

'''
rng=pd.date_range('1/1/2000',periods=12,freq='T')
ts=Series(np.arange(12),index=rng)
print(ts)
print('5min,closed=left')
print(ts.resample('5min',closed='left').sum())
print('5min,closed=right')
print(ts.resample('5min',closed='right').sum())
print('5min,closed=left,label=right  ***********************')
print(ts.resample('5min',closed='left',label='right').sum())
print('5min,closed=left,label=left  ***********************')
print(ts.resample('5min',closed='left',label='left').sum())
print('5min,closed=right,label=left   &&&&&&&&&&&&&&&&&&&&&&&')
print(ts.resample('5min',closed='right',label='left').sum())
print('5min,closed=right,label=left   &&&&&&&&&&&&&&&&&&&&&&&')
print(ts.resample('5min',closed='right',label='right').sum())
'''
data_px = data.resample('B').ffill()    #    数据的填充方式
# print(data_px.head(10))
# print(data_px.info())
# data_px['AAPL'].plot()
# plt.show()
df = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
                'data1': range(6)})
print df
print '.....>>>>>>>>>>',pd.get_dummies(df)
print '[[[[[[['
dummies = pd.get_dummies(df['key'], prefix='pkey')
print "---->>>>",dummies
df_with_dummy = df[['data1']].join(dummies)
print df_with_dummy
























