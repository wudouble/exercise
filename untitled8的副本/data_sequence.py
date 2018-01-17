#encoding:utf-8
from datetime import datetime
from datetime import timedelta
from dateutil.parser import parse
import numpy as np
from pandas import DataFrame,Series
import pandas as pd
from pandas.tseries.offsets import Hour,Minute,Day,MonthEnd
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib
matplotlib.matplotlib_fname()
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA

path = '/Users/wss/Desktop/py/第12周/'
now = datetime.now()
print(now)
'''
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
'''
data = pd.read_csv(path + 'stock_px.csv',usecols=[0,1,2,3,4],parse_dates=True,index_col=0)      #用作行索引的列编号或者列名
print(data.head(10))
#print(data[['AAPL','MSFT']].head())
'''
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
'''
data_px = data.resample('B').ffill()    #    数据的填充方式
print(data_px.head(10))
print(data_px.info())
#data_px['AAPL'].plot()
#plt.show()
#data_px.ix['2009'].plot()
#plt.show()
data_px['AAPL'].ix['01-2011':'03-2011'].plot()
#plt.show()
data_Q = data_px['AAPL'].resample('Q-DEC').ffill()
print(data_Q.head())
#data_Q.ix['2009'].plot()
#plt.show()
#data_px.AAPL.plot()
#plt.show()
print(pd.rolling_mean(data_px.AAPL,250))
#pd.rolling_mean(data_px.AAPL,250).plot()
#plt.show()
print('**************************************************')
data_min = pd.rolling_mean(data_px.AAPL,250,min_periods=10)
#data_min.plot()
#plt.show()
#expanding_mean = lambda x:rolling_mean(x,len(x),min_periods=1)
#print(data_px().apply(expanding_mean))
'''
'''
fig,axes = plt.subplots(nrows = 2,ncols = 1,sharex = True,sharey = True,figsize = (12,7))
aapl_px = data_px.AAPL['2005':'2009']
ma60 = pd.rolling_mean(aapl_px,60,min_periods = 50)
ewma60 = pd.ewma(aapl_px,span = 60)

aapl_px.plot(style = 'k-',ax = axes[0])
ma60.plot(style = 'k--',ax = axes[0])
aapl_px.plot(style = 'k-',ax = axes[1])
ewma60.plot(style = 'k--',ax = axes[1])
axes[0].set_title('Simple MA')
axes[1].set_title('Exponentially-weighted MA')
#plt.show()
'''
'''
print(data_px.head())
pct_data = data_px.pct_change(fill_method='pad')
print(pct_data.head())
#                       下面即为pct_change（）函数的实现方法
#pct_data_equal = data_px/data_px.shift(1)-1
#print(pct_data_equal.head())
cor = pd.rolling_corr(pct_data.AAPL,pct_data.SPX,125,min_periods = 100)
#cor.plot()

corr = pd.rolling_corr(pct_data.ix[:,[0,1,2]],pct_data.SPX,125,min_periods = 100)
#corr.plot()
#plt.show()
pct_data_three = pct_data[['AAPL','MSFT','XOM']]
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
print(pct_data_three.head())
score_at_2percent = lambda x:percentileofscore(x,0.02)      #percentileofscore是指，0.02在x中的位置是x中的百分比
result = pd.rolling_apply(pct_data_three.AAPL,250,score_at_2percent)
print(result.head(260))
result.plot()
plt.show()


rng = pd.date_range('1/1/2000',periods = 10000000,freq = '10ms')
ts = Series(np.random.randn(len(rng)),index = rng)
print(ts.head())
print (ts.resample('15min').ohlc())


data = pd.read_excel(path + 'arima_data.xls',index_col=u'日期')
print(data.head())
data = pd.DataFrame(data,dtype=np.float64)                      #  转化成浮点型
print(data.head())

plt.rcParams['font.sans-serif'] = ['SimHei']                 #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False                   # 用来正常显示正负号
plt.rcParams.update({
    'font.family':'sans-serif',
    'font.sans-serif':['Liberation Sans'],
    })
#data.plot()
#plt.show()
'''
# plot_acf(data)                                               #自相关图
# plt.show()
'''
print(ADF(data[u'销量']))       # ADF检验可以得到单位根检验统计量对应的p值，若此值显著大于0.05，则该序列非平稳
#返回值依次为adf、pvalue、usedlag、nobs、critical values、icbest、regresults、resstore

data_D = data.diff().dropna()
data_D.columns = [u'销量差分']
#data_D.rename(columns = {u'销量':u'销量差分'},inplace=True)
print(data_D.head())
data_D.plot()

plot_acf(data_D)
plot_pacf(data_D)                            #偏自相关图
plt.show()

print(ADF(data_D[u'销量差分']))               #ADF:平稳性检验

print(acorr_ljungbox(data_D,lags=1))
pmax = int(len(data_D)/10) #一般阶数不超过length/10
qmax = int(len(data_D)/10)
bic_matrix = []
for p in range(pmax+1):
    temp = []
    for q in range(qmax+1):
        try:
            temp.append(ARIMA(data, (p, 1, q)).fit().bic)
        except:
            temp.append(None)
    bic_matrix.append(temp)
bic_matrix = pd.DataFrame(bic_matrix)
p,q = bic_matrix.stack().idxmin()
print(p,q)
model = ARIMA(data, (0,1,1)).fit()
print(model)

'''














