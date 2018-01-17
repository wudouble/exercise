#encoding:utf-8
from pandas import Series,DataFrame
import pandas as pd
import numpy as np
import statsmodels.api as sm
from StringIO import StringIO
import matplotlib.pyplot as plt
from scipy import stats as ss


path = '/Users/wss/Desktop/py/第8周/data/'

df = DataFrame({'key1':['a','a','b','b','a'],
                'key2': ['one', 'two', 'one', 'two', 'one'],
                'data1':np.random.randn(5),
                'data2':np.random.randn(5)})

print df
grouped = df['data1'].groupby(df['key1'])
print grouped.mean()
means = df['data1'].groupby([df['key1'],df['key2']]).mean()
print means
print means.unstack()
states = np.array(['Ohio', 'California', 'California', 'Ohio', 'Ohio'])
years = np.array([2005, 2005, 2006, 2005, 2006])
print df['data1'].groupby([states,years]).mean()
print df.groupby([df['key1'], df['key2']]).mean()
print df.groupby([df['key1'], df['key2']]).size()

#                         分组迭代
df = DataFrame({'key1':['a','a','b','b','a'],
                'key2': ['one', 'two', 'one', 'two', 'one'],
                'data1':np.random.randn(5),
                'data2':np.random.randn(5)})
print df
for name,group in df.groupby(['key1']):
    print name
    print '*******'
    print group
print '????????????????'
for (k1,k2),group in df.groupby(['key1','key2']):
    print (k1,k2)
    print group

print list(df.groupby(['key1','key2']))
print '>>>>>>>>>'
print dict(list(df.groupby(['key1','key2'])))[('a', 'one')]
print '**************'
print list(df.groupby(df.dtypes,axis=1))[1]

print list(df.groupby(['key1'])['data1'])
print list(df.data1.groupby(df.key1))
print '<<<<<<<<<<<<<<<'
print list(df['data1'].groupby(df['key1']))         #   与上面的表达功能一致
print '&&&&&&&&&&&&&&&&&&&&&&&&&&&'
print list(df[['data1']].groupby(df['key1']))

people = DataFrame(np.random.randn(5,5),
                columns=['a', 'b', 'c', 'd', 'e'],
                index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis'])
print people
people.ix[2:3,['b','c']] = np.nan
print people

map = {'a': 'red', 'b': 'red', 'c': 'blue',
       'd': 'blue', 'e': 'red', 'f' : 'orange'}
print map
li = []
for x in people.index:
    li.append(len(x))
print li
print 'ZZzzzzz'
print people.groupby(map,axis=1).sum()
#print people.groupby(map,axis=0).sum()
S_map = Series(map)
print people.groupby(S_map,axis=1).count()
print len(people.index)
print people.groupby(li).count()
l=[len(x) for x in people.index]
print people.groupby(l).count()
print people.groupby(len).count()
key_list = ['one', 'two', 'one', 'on', 'one']
print people.groupby([len,key_list]).count()    #    两个列表一一对应 进行分组


#                      按 索引级别 分组
Mu_column= pd.MultiIndex.from_arrays([['US', 'US', 'US', 'JP', 'JP'],
                                    [1, 3, 5, 1, 3]], names=['city', 'tensor'])
DF_mu = DataFrame(np.random.randn(4,5),columns=Mu_column)
print DF_mu
print DF_mu.groupby(level='tensor',axis=1).count()

#                 数据聚合

df = DataFrame({'key1':['a','a','b','b','a'],
                'key2': ['one', 'two', 'one', 'two', 'one'],
                'data1':np.random.randn(5),
                'data2':np.random.randn(5)})
print df
grouped = df.groupby('key1')['data1'].quantile(0.9)
print grouped
def peak_to_peak(arr):
    return arr.max()-arr.min()
print grouped.agg(peak_to_peak)
print grouped.describe()

#              面向列的多函数应用
def peak_to_peak(arr):
    return arr.max()-arr.min()
data = pd.read_csv(path + 'tips.csv')
print data[:6]
data['tip_dic'] = data['tip'] / data['total_bill']
print data[:6]
group = data.groupby(['sex','smoker'])['tip_dic']
print group.agg(['mean','count',peak_to_peak])
print group.agg({'foo':np.mean,'dd':sum})
function = ['mean','count','sum']
result = data.groupby(['sex','smoker'])['tip_dic', 'total_bill'].agg(function)
print result
print '/??????????????'
print data.groupby(['sex','smoker']).mean()
print data.groupby(['sex','smoker'],as_index=False).mean()  #聚合数据都需要唯一的分组键组成的索引，但也可以通过向groupby传入as_index=False以禁用该功能
grouped = data.groupby(['sex', 'smoker'])
print grouped.agg({'tip_dic' : ['min', 'max', 'mean', 'std'],
             'size' : 'sum'})


data = DataFrame({'key1':np.random.randn(1000),
                'key2':np.random.randn(1000)})
print data[:10]
data1 = pd.cut(data.key1,4)
print data1[:10]
def definegroup(arr):
    return {'min':arr.min(),'max':arr.max(),'mean':arr.mean(),'count':arr.count()}
group = data.key2.groupby(data1)
print group.apply(definegroup).unstack()
qdata = pd.qcut(data.key1,4,labels=False)      #   label=false即可值获取分位数的编号
qgroup = data.key2.groupby(qdata).apply(definegroup).unstack()
print qgroup


#                      用特定于分组的值填充缺失值    
data = Series(np.random.randn(6))
print data
data[::2] = np.nan
print data
data1 = data.fillna(data.mean())   #用一个固定值或由数据集本事衍生出来的值去填充na值
print 'ŁKKKKKKK'
print data1

states = ['Ohio', 'New York', 'Vermont', 'Florida',
          'Oregon', 'Nevada', 'California', 'Idaho']
group_key = ['East'] * 4 + ['West'] * 4
print group_key
data = Series(np.random.randn(8),index=states)
data.ix[['New York','Florida','California']] = np.nan
print data
group = data.groupby(group_key).mean()
print group
#              填充na
fill_nan = lambda x:x.fillna(x.mean())
group1 = data.groupby(group_key).apply(fill_nan).mean()
print group1
print 'KKKK<<<<<<<<<<<<<<<'
fill_values = {'East': 0.5, 'West': -1}
fill_func = lambda g: g.fillna(fill_values[g.name])

print data.groupby(group_key).apply(fill_func)


#                         随机采样和排列
suits=['H','S','C','D']
num = (list(range(1,11))+[10]*3)*4
print num
name = ['A'] + list(range(2,11)) + ['J','Q','K']
print name
arr = []
for  x in ['H','S','C','D']:
    arr.extend(x + str(nu) for nu in name)
deck = Series(num,index=arr)
print '>>>>>>>>>>>>>'
print deck[:13]
def sampling(a,n = 5):
    return a.take(np.random.permutation(len(a))[:n])
print sampling(deck)
get_suit = lambda card:card[0] 
Sample_number = deck.groupby(get_suit,group_keys=False).apply(sampling,n=2)    #group_keys=False 不显示分组关键字
print Sample_number


#       计算分组平均值
df=DataFrame({'category':['a','a','a','a','b','b','b','b'],'data':np.random.randn(8),
              'weights':np.random.rand(8)})
print df
group = df.groupby('category')
fun = lambda x: np.average(x['data'],weights=x['weights'])
print group.apply(fun)
data = pd.read_csv(path + 'stock_px.csv',parse_dates= True,index_col= 0)
print data[:4]
ret = data.pct_change().dropna()
print ret[:4]
spx_cor = lambda x:x.corrwith(x['SPX'])         #    计算相关系数   Series和DataFrame都可以用corr和cov方法分别计算相关系数和协方差
print data.groupby(lambda y:y.year).apply(spx_cor)
ss = lambda x:x['AAPL'].corr(x['MSFT'])
print data.groupby(lambda y:y.year).apply(ss)
print '?????????????????'
#                   最小二乘法
def relation(data,xvar,yvar):
    y = data[yvar]
    x = data[xvar]
    x['intercept'] = 1.
    result = sm.OLS(y,x).fit()
    return result.params
print data.groupby(lambda y:y.year).apply(relation,['SPX'],'AAPL')

#====                           透视表 http://python.jobbole.com/81212/

#                            pivot_table
tips = pd.read_csv(path + 'tips.csv')
print tips[:10]
print '??????????????'
print tips.pivot_table(index=['sex','smoker'],aggfunc=np.sum)
print tips.pivot_table(['total_bill','tip'],index=['sex','day'],columns=['smoker'])
print tips.pivot_table(['total_bill','tip'],index=['sex','day'],columns=['smoker'],margins = True)
print '>>>>>>>>>>>>>>>>>'
#                            pd.crosstab
print pd.crosstab([tips['day'],tips['smoker']],[tips['sex'],tips['size']],margins=True)

data = """Sample    Gender    Handedness
1    Female    Right-handed
2    Male    Left-handed
3    Female    Right-handed
4    Male    Right-handed
5    Male    Left-handed
6    Male    Right-handed
7    Female    Right-handed
8    Female    Left-handed
9    Male    Right-handed
10    Female    Right-handed"""

data1 = pd.read_table(StringIO(data),sep='\s+')
print data1
print pd.crosstab(data1.Gender,data1.Handedness,margins=True)



#                                          eg:2012联邦选举委员会数据分析  http://blog.csdn.net/yisuoyanyv/article/details/75799289
data = pd.read_csv(path + 'P00000001-ALL.csv')
print data.head()
print '>>>>>>>>>>>>>'
print data.ix[123456]
#unique_candicate = list(set(data.cand_nm))
unique_candicate = data.cand_nm.unique()
print unique_candicate
print unique_candicate[2]
parties={'Bachmann, Michelle':'Republican',
         'Cain, Herman':'Republican',
         'Gingrich, Newt':'Republican',
         'Johnson, Gary Earl':'Republican',
         'McCotter, Thaddeus G':'Republican',
         'Obama, Barack':'Democrat',
         'Paul, Ron':'Republican',
         'Pawlenty, Timothy':'Republican',
         'Perry, Rick':'Republican',
         "Roemer, Charles E. 'Buddy' III":'Republican',
         'Romney, Mitt':'Republican',
         'Cain, Herman':'Republican',
         'Santorum, Rick':'Republican'
         }
print "*****************"
print parties
print data.cand_nm[123456:123461]
print data.cand_nm[123456:123461].map(parties)
data['Parties'] = data.cand_nm.map(parties)
print data.Parties.value_counts()   #  //统计重复重现的数据的个数。返回以数据作为key，以重复个数为value的对象。
print sum(np.where(data.contb_receipt_amt < 0))

print (data.contb_receipt_amt > 0).value_counts()
#   #为了简化分析过程，我们限定该数据集只有正的出资额
data = data[data.contb_receipt_amt > 0]
print (data.contb_receipt_amt > 0).value_counts()
print (data.cand_nm == 'Romney, Mitt').value_counts()
data_sub = data[data.cand_nm.isin(['Romney, Mitt','Obama, Barack'])]
print data_sub.head()
print '???????????????????'
print data.contbr_occupation.unique()
job_num =data.contbr_occupation.value_counts().head()    #  计算各个职业对应的人数
print job_num
emp_mapping={'INFORMATION REQUESTED PER BEST EFFORTS':'NOT PROVIDED',
             'INFORMATION REQUESTED':'NOT PROVIDED',
             'SELF':'SELF-EMPLOYED',}
f = lambda x : emp_mapping.get(x,x)                     #如果没有提供相关映射，则会返回x
data.contbr_occupation = data.contbr_occupation.map(f)
data.contbr_employer = data.contbr_employer.map(f)
sa = data.pivot_table(['contb_receipt_amt'],index=['contbr_occupation'],columns=['Parties'],aggfunc='sum')
print '::::::::::::::::::::::::::::'
print sa[:10]
sa1 = sa[sa.sum(1)> 2000000]
print '<<<<<<<<<<<<<<<<<<<'
print sa1

#sa1.plot(kind = 'barh')
#plt.show()

#                          统计对Obama和Romney总出资额最高的职业和企业
#data11 = data[['Obama, Barack','Romney, Mitt']]
#                   way 1
sss = data.pivot_table(['contb_receipt_amt'],index=['contbr_occupation'],columns=['cand_nm'],aggfunc='sum')
print '.............................'
sssa =sss.sum(0)
sorted(sssa)
print sssa.sort_values(ascending=False)
df =DataFrame(sssa)
#                  way2
def top_group(arr,key,n = 5):
    grouped = arr.groupby(key)['contb_receipt_amt'].sum()
    return grouped.sort_values(ascending=False)[:n]
group1 = data_sub.groupby('cand_nm')
group2 = group1.apply(top_group,'contbr_occupation')
print(group2)

#                            对出资额分组
#              利用cut函数根据出资额的大小将数据离散化到多个面元中
#data1 = pd.cut(data['contb_receipt_amt'],4)
#print data1
bins =np.array([0,1,10,100,1000,10000,100000,1000000,10000000])
cut_group = pd.cut(data_sub.contb_receipt_amt,bins)
print cut_group.head()
#            然后根据候选人姓名以及面元标签对数据进行分组：
group_1 = data_sub.groupby(['cand_nm',cut_group])
print group_1.size().unstack(0)

v_data = group_1.contb_receipt_amt.sum().unstack(0)
#    相当于 下面的 print
#print data_sub.pivot_table(['contb_receipt_amt'],index=[cut_group],columns='cand_nm',aggfunc='sum')
print v_data
percent_data = v_data.div(v_data.sum(axis =1),axis = 0)
print 'PPppppppppppppppppppppp'
print percent_data
percent_data[:-2].plot(kind = 'barh',stacked = True)   #排除了最大的两个面元，因为这些不是个人捐赠
#plt.show()

#                          根据州统计赞助信息
stat_group = data_sub.pivot_table(['contb_receipt_amt'],index=['contbr_st'],columns=['cand_nm'],aggfunc='sum').fillna(0)
print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
print stat_group
total = stat_group[stat_group.sum(1) > 100000]
print total.head(10)
#                  对各行除以总赞助额，就会得到各候选人在各州的总赞助额比例
percent_total = total.div(total.sum(axis = 1),axis = 0)
print percent_total[:10]

































