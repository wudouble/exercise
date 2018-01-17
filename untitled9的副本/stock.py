#encoding:UTF-8
import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib.pyplot import plot
from matplotlib.pyplot import show


cp,trans= np.loadtxt('/Users/wss/Desktop/py/data.csv',usecols=(6,7),delimiter=',',unpack=True)
#print cp,trans
# VMAP(Volume Weighted Average Price ):确定时段中的总交易金额除以相应的总交易量
vmap = np.average(cp,weights=trans)
print vmap
# 算术平均价：
mean = np.average(cp)
print mean,cp.mean()
#TWAP(The Time Weighted Average Price ):
w = np.arange(len(cp))
twap = np.average(cp,weights=w)
print twap
hignPrice,lowPrice = np.loadtxt('/Users/wss/Desktop/py/data.csv',usecols=(4,5),delimiter=',',unpack=True)
print "highest : ", np.max(hignPrice)
print "lowest :",np.min(lowPrice)
print "range of hignPrice:",np.ptp(hignPrice)    # 返回数组最大值和最小值之间的差值
print "range of lowPrice:",np.ptp(lowPrice)
print "*******"
cp1 = np.loadtxt('/Users/wss/Desktop/py/data.csv',usecols=(6,),delimiter=',',unpack=True)
print (cp==cp1).all()       #   比较两个array是否相等
# 输出中位数
print "median : ",np.median(cp1)
# 用排序的方法验证以上的中位数
sort_cp = np.msort(cp1)
print sort_cp
n = len(cp1)
if n/2 !=0:
    middle = sort_cp[n/2]
else :
    middle = (sort_cp[n/2]+sort_cp[(n/2)-1])
print middle
#  方差
variance = np.var(cp1)
print variance

#   股票收益率
#arr = np.array([1,2,4,3,5,6])
#print np.diff(arr,n=2,axis = 0)                    n代表迭代次数
#print arr
diff = np.diff(cp1)
#print cp1[:-1]
yields = diff/cp1[:-1]
print yields
#  收益率的标准差 R=(b-a)/a
print np.std(yields)

#   对数收益率   r=lnb-lna = ln(1+R) = ln(b/a)
lndiff = np.diff(np.log(cp1))
print lndiff
lndiff1 = np.log(1 + yields)
print lndiff1
print (lndiff==lndiff1).all()

#  输出收益率为正的情况:
positive_yields = np.where(yields > 0)
print "index of ",positive_yields
#  年波动率  对数收益率的标准差除以其均值，再除以交易日倒数的平方根，交易日通常取252天
st_lndiff = np.std(lndiff)
bodonglv_Year = (st_lndiff/np.mean(lndiff))/np.sqrt(1.0/252)
print bodonglv_Year
bodonglv_Month = bodonglv_Year * np.sqrt(1.0/12)  #月波动率转为年波动率一般都是乘以根号12
print bodonglv_Month

#    日期分析
def datetrans(s):
   time = datetime.datetime.strptime(s,'%d-%m-%Y').isoweekday()  # .weekday()则返回0—6
   return time
date,cp2 = np.loadtxt('/Users/wss/Desktop/py/data.csv',delimiter=',',usecols=(1,6),converters={1:datetrans},unpack=True)
print date
#  把周一到周五的所有收盘价找到并求均值
arr = np.zeros(5)
for i in range(1,6):
    index = np.where(date == i)
    temp = np.take(cp2,index)    #b中元素作为索引，查找a中对应元素：np.take(a,b)
    avg = np.mean(temp)
    print "weekday ",i,"close_Price ",temp,"average ",avg
    arr[i] =avg
#      按周压缩数据
def datetrans(s):
   time = datetime.datetime.strptime(s,'%d-%m-%Y').isoweekday()  # .weekday()则返回0—6
   return time
dates,op,hp,lp,cp = np.loadtxt('/Users/wss/Desktop/py/data.csv',delimiter=',',usecols=(1,3,4,5,6),converters={1:datetrans},unpack=True)
print dates
###   由于数据有缺失  为了方便计算就只取前三周
dates = dates[:16]
# find first_Monday
frist_Monday = np.ravel(np.where(dates == 1))[0]
print frist_Monday
#  find Last_Friday
last_Friday = np.ravel(np.where(dates == 5))[-1]
print last_Friday

week_index = np.arange(frist_Monday,last_Friday+1)
print week_index

week_index_seprate = np.split(week_index,3)
print week_index_seprate

def summaris(a,o,h,l,c):
    openPrice = o[a[0]]
    highest_price = np.max(np.take(h,a))
    lowest_price = np.min(np.take(l,a))
    friday_close = c[a[-1]]
    return "AAPL ",openPrice,highest_price,lowest_price,friday_close
weeksummarize = np.apply_along_axis(summaris,1,week_index_seprate,op,hp,lp,cp)
print weeksummarize

#     真实波动幅度均值 ATR
#真实波动幅度：是以下三个波动幅度的最大值
#1. 当天最高点和最低点间的距离
#2. 前一天收盘价和当天最高价间的距离，或
#3. 前天收盘价和当天最低价间的距离
hp,lp,cp = np.loadtxt('/Users/wss/Desktop/py/data.csv',delimiter=',',usecols=(4,5,6),unpack=True)
cp1 =cp[1:]
hp = hp[1:]
lp = lp[1:]
previouscp = cp[:-1]
print len(cp),len(cp1),len(previouscp)
tr = np.maximum(hp-lp,hp-previouscp,previouscp-lp)
print len(tr)
print tr
#  atr
atr = np.zeros(len(tr))
atr[0] = np.mean(tr)
n = len(tr)
for i in range(1,n):
    atr[i] = (n - 1)* atr[i-1] + tr[i]
    atr[i] /= n
print atr
'''
#  移动平均
'''
n = 10   #  移动的n值
weights = np.ones(n)/n
print weights
cp = np.loadtxt('/Users/wss/Desktop/py/data.csv',delimiter=',',usecols=(6,),unpack=True)
# 要从卷积运算中取出与原数组重叠的区域
sma = np.convolve(weights,cp)[n-1:-n+1]
print "======"
print np.convolve(weights,cp)
print sma
t = np.arange(n-1,len(cp))
plot(t,cp[n-1:],'g-.',lw = 1.0)     #http://www.360doc.com/content/15/0113/23/16740871_440559122.shtml
plot(t,sma,'r--',lw = 4.0)
show()

plt.xlabel("Date")

plt.ylabel("Close Price")

plt.title(u"Simple Moving Average")


plt.annotate('before convolve', xy=(12.8, 363), xytext=(15, 363),

arrowprops=dict(facecolor='black',shrink=0.005))

plt.annotate('after convolve', xy=(15, 358), xytext=(17, 358),

arrowprops=dict(facecolor='black',shrink=0.005))


#  指数移动平均线 EXPMA
n = 10
print "指定的间隔内返回均匀间隔的数字",np.linspace(-1,0,6,retstep=True)
weight1 = np.exp(np.linspace(-1,0,n))
weight1 /= weight1.sum()
print weight1
cp = np.loadtxt('/Users/wss/Desktop/py/data.csv',delimiter=',',usecols=(6,),unpack=True)
exp_ma = np.convolve(weight1,cp)[n-1:-n+1]
print exp_ma
print len(cp),len(exp_ma)
t = np.arange(n-1,len(cp))
plot(t,cp[n-1:],"g-",lw = 1.0)
plot(t,exp_ma,"r-",lw = 2.0)
show()

#    布林带
#中轨：简单移动平均线
#上轨：比简单移动平均线高两倍标准差的距离，标准差是简单移动平均线的标准差
#下轨：比简单移动平均线低两倍标准差的距离，标准差是简单移动平均线的标准差
n = 5
cp = np.loadtxt('/Users/wss/Desktop/py/data.csv',delimiter=',',usecols=(6,),unpack=True)
arr = np.ones(n)
weight = arr/n
print weight
sma = np.convolve(weight,cp)[n-1:-n+1]
print len(sma),sma
print cp

dev = []
for i in range(n-1,len(cp)):
    average = np.zeros(n)
    average.fill(sma[i-n+1])           #以便于后面的减法
    temp = cp[i-n+1:i+1]
    temp_Vari = temp - average
    temp_Vari = temp_Vari ** 2
    varice = np.mean(temp_Vari)    # 方差
    deviation = np.sqrt(varice)
    dev.append(2*deviation)
upperB = sma + dev
lowerB = sma - dev
print len(sma),sma
print len(upperB),upperB
print len(lowerB),lowerB

# 检验数据是否全部都落在上轨和下轨内
between_band = np.where((cp[n-1:]<upperB)&(cp[n-1:]>lowerB))
print between_band
print np.ravel(between_band)
print "ration in B ",len(np.ravel(between_band))/len(cp[n-1:])

#  绘图
t = np.arange(n-1,len(cp))
plot(t,cp[n-1:],'g-.')
plot(t,sma,'b-')
plot(t,upperB,'r-.')
plot(t,lowerB,'r-.')
show()
a = np.arange(1,8)
print a
print "Clipped :",a.clip(3,6)      # clip返回一个修剪过的数组，小于等于给定最小值的设为给定最小值，反之亦然,长度与原数组一样
print a
print "Compressed : ",a.compress(a>5)   #返回一个给定筛选条件后的数组，返回长度不一样

print a.prod()   # 全部的阶乘
print a.cumprod()    #   累积的阶乘














