#encoding:UTF-8
from __future__ import division
import pandas as pd
import numpy as np
import os
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange    #  导入拉格朗日插值函数
import patsy
import json
import re
'''
#----------------------------------拉格朗日插值法
path = '/Users/wss/Desktop/py/第6周/data/'

x=[1,2,3,4,7]
y=[5,7,10,3,9]
a=lagrange(x,y)
print a
print(a(1),a(2),a(3))     #    a() 表示得到对应函数值

print(a[0],a[2],a[3])  #  a[] 表示得到系数

print a(4),a(5),a(7)


path = '/Users/wss/Desktop/py/第6周/data/'
pd = pd.read_excel(path + 'catering_sale.xls')
print pd
print pd.columns
pd[u'销量'][(pd[u'销量'] < 400)|(pd[u'销量'] > 5000)] = None
#print pd

def ployinterp_column(s,n,k = 5):
    data_value = s[list(range(n, n+ 2 * k +1))]
   # data_value = s[list(range(n-k,n)):list(range(n+1,n+k+1))]
    data_value = data_value[data_value.notnull()]
    return lagrange(data_value.index,list(data_value))(n)
print pd.columns
for i in pd.columns:
    for j in range(len(pd)):
        if (pd[i].isnull())[j]:
            pd[i][j] = ployinterp_column(pd[i],j)
pd.to_excel(path + 'refy.xls')


###-------------------------数据连接
###--------------------------DataFrame 连接

df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                 'data1': range(7)})
df2 = DataFrame({ 'key': ['a', 'b', 'd'],
                 'data2': range(3)})
print df1
print df2
print pd.merge(df1,df2)                   #  和下面的两种合并方法达到的效果一致
print pd.merge(df1,df2,on='key')
print pd.merge(df1,df2,how='inner')
print pd.merge(df1,df2,how='outer')

df3 = DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                 'data1': range(7)})
df4 = DataFrame({'rkey': ['a', 'b', 'd'],
                 'data2': range(3)})
print df3
print df4
print pd.merge(df3,df4,left_on='lkey',right_on='rkey')
#print pd.merge(df3,df4,on =['lkey','rkey'])

left = DataFrame({'key1': ['foo', 'foo', 'bar'],
                  'key2': ['one', 'two', 'one'],
                  'lval': [1, 2, 3]})
right = DataFrame({'key1': ['foo', 'foo', 'bar', 'bar'],
                   'key2': ['one', 'one', 'one', 'two'],
                   'rval': [4, 5, 6, 7]})
print left
print right
print pd.merge(left, right, on=['key1', 'key2'], how='outer')
print pd.merge(left, right, on=['key1', 'key2'], how='inner')
print pd.merge(left, right, on='key1')
print pd.merge(left, right, on='key1', suffixes=('_left', '_right'))      #  suffixes  用于追加到重叠列名的末尾    默认 _x,_y

##                            索引上的合并
left1 = DataFrame({'key': ['a', 'b', 'a', 'a', 'b', 'c'],'value': range(6)})
right1 = DataFrame({'group_val': [3.5, 7]}, index=['a', 'b'])
print left1
print right1
print pd.merge(left1, right1, left_on='key', right_index=True)
print pd.merge(left1, right1, left_on='key', right_index=True, how='outer')

lefth = DataFrame({'key1': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
                   'key2': [2000, 2001, 2002, 2001, 2002],
                   'data': np.arange(5.)})
righth = DataFrame(np.arange(12).reshape((6, 2)),
                   index=[['Nevada', 'Nevada', 'Ohio', 'Ohio', 'Ohio', 'Ohio'],
                          [2001, 2000, 2000, 2000, 2001, 2002]],
                   columns=['event1', 'event2'])
print lefth
print righth
print pd.merge(lefth, righth, left_on=['key1', 'key2'], right_index=True)
print pd.merge(lefth, righth, left_on=['key1', 'key2'],right_index=True, how='outer')

left2 = DataFrame([[1., 2.], [3., 4.], [5., 6.]], index=['a', 'c', 'e'],
                 columns=['Ohio', 'Nevada'])
right2 = DataFrame([[7., 8.], [9., 10.], [11., 12.], [13, 14]],
                   index=['b', 'c', 'd', 'e'], columns=['Missouri', 'Alabama'])
print '-----------------'
print left2
print right2
print pd.merge(left2, right2, how='outer', left_index=True, right_index=True)

#                                                    join 连接
print left2.join(right2, how='outer')
#print pd.merge(left2,right2,on='key')
print left1
print right1
print left1.join(right1, on='key')

another = DataFrame([[7., 8.], [9., 10.], [11., 12.], [16., 17.]],
                    index=['a', 'c', 'e', 'f'], columns=['New York', 'Oregon'])
print left2
print another
print left2.join([right2, another])
print left2.join([right2, another], how='outer')

#=================================轴向连接     numpy: concatenate()函数    pandas :concat() 函数
arr = np.arange(12).reshape((3, 4))
print arr
arr1 = np.concatenate((arr,arr),axis=1)
print arr1


s1 = Series([0, 1], index=['a', 'b'])
s2 = Series([2, 3, 4], index=['c', 'd', 'e'])
s3 = Series([5, 6], index=['f', 'g'])
s4 = pd.concat([s1 * 5, s3])
print s1
print s2
print s3
print 's4->'
print s4
print "---------"
print pd.concat([s1,s2,s3])                      #  按列合并   横着
print pd.concat([s1,s2,s3],axis=1)               #  按行合并
print pd.concat([s1,s2,s3],ignore_index=True)  #生成纵轴上的并集，索引会自动生成新的一列
print "======="
print pd.concat([s1, s4], axis=1)
print pd.concat([s1, s4], axis=1,join='inner')
print pd.concat((s1,s4),axis=1,join_axes=[['a','c','b','e']])
result =  pd.concat([s1, s1, s3], keys=['one', 'two', 'three'])     #  keys()   形成层次化索引
print result

print result.unstack()                                   #非堆叠操作（unstack）
print pd.concat([s1, s2, s3], axis=1, keys=['one', 'two', 'three'])

df1 = DataFrame(np.arange(6).reshape(3, 2), index=['a', 'b', 'c'],
                columns=['one', 'two'])
df2 = DataFrame(5 + np.arange(4).reshape(2, 2), index=['a', 'c'],
                columns=['three', 'four'])
print df1
print df2
print pd.concat([df1, df2], axis=0, keys=['level1', 'level2']).unstack()
print pd.concat([df1, df2], axis=1, keys=['level1', 'level2']).unstack()
print pd.concat({'level_1':df1,'level_2':df2})
print "//////////"
print pd.concat([df1, df2], axis=1, keys=['level1', 'level2'])
print pd.concat([df1, df2], axis=1,levels=['level1','level2'])
print pd.concat([df1, df2], axis=1, keys=['level1', 'level2'],names=['upper', 'lower'])

'''
'''
s1 = pd.Series([0,1,2],index = ['a','b','c'])

s2 = pd.Series([2,3,4],index = ['c','f','e'])

s3 = pd.Series([4,5,6],index = ['c','f','g'])

print pd.concat([s1,s2,s3])
print pd.concat([s1,s2,s3],ignore_index=True)
print pd.concat([s1,s2,s3],axis=1)
print pd.concat([s1,s2,s3],axis=1,join = 'inner')
print pd.concat([s1,s2,s3],axis = 1,join = 'outer')


#                            合并缺失值
a = Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan],
           index=['f', 'e', 'd', 'c', 'b', 'a'])
b = Series(np.arange(len(a), dtype=np.float64),
           index=['f', 'e', 'd', 'c', 'b', 'a'])
print a
print b
b[-4] = np.nan
print b
print a[0]
print np.where(pd.isnull(a))
np.where(pd.isnull(a),b,a)
print "-----------------"
print a
print b
t = b[:-2].combine_first(a[2:])
print t
df1 = DataFrame({'a': [1., np.nan, 5., np.nan],
                 'b': [np.nan, 2., np.nan, 6.],
                 'c': range(2, 18, 4)})
df2 = DataFrame({'a': [5., 4., np.nan, 3., 7.],
                 'b': [np.nan, 3., 4., 6., 8.]})
print df1.combine_first(df2)


#======================================    数据重塑 # unstack操作的是最内层的（stack也是） 
#                                               当我们传入的分层级别的编号或名称，同样可以对其他级别进行unstack 操作

data = DataFrame(np.arange(6).reshape((2, 3)),
                 index=pd.Index(['Ohio', 'Colorado'], name='state'),
                 columns=pd.Index(['one', 'two', 'three'], name='number'))
print data
print "-------------------"
result =  data.stack()                         #   把列转化为行
print '=================='
print data.stack().unstack()              #     把行转化成列
print '??????????????????'
print data.stack().unstack(0)
print data.stack().unstack('state')
s1 = Series([0, 1, 2, 3], index=['a', 'b', 'c', 'd'])
s2 = Series([4, 5, 6], index=['c', 'd', 'e'])
data2 = pd.concat([s1, s2], keys=['one', 'two'])
print s1
print s2
print data2
print data2.unstack()
print '---------'
print data2.unstack().stack()
print '>>>>>>>>>>>>>'
print data2.unstack().stack(dropna = False)            #   dropna = False  表示  保留值为 nan 的
print data2.unstack(0)                                 #   参数表示 进行宽数据转化时 要变成行的 列号
df = DataFrame({'left': result, 'right': result + 5},
               columns=pd.Index(['left', 'right'], name='side'))
print '------------------<<<<<<<<<<'
print df
print df.stack()
print ',,,,,,,,,'
print df.unstack(0)
print '............'
print df.unstack('state').stack('side')
'''
##========================================                   长宽数据格式 转换
path = '/Users/wss/Desktop/py/第6周/data/'
data = pd.read_csv(path + 'macrodata.csv')
print data [:10]
periods = pd.PeriodIndex(year=data.year, quarter=data.quarter, name='date')
print periods
data = DataFrame(data.to_records(),
                 columns=pd.Index(['realgdp', 'infl', 'unemp'], name='item'),
                 index=periods.to_timestamp('D', 'end'))
print data
ldata = data.stack().reset_index().rename(columns={0: 'value'})
print ldata
wdata = ldata.pivot('date','item','value')
print wdata
ldata['value2'] = np.random.randn(len(ldata))
print ldata
s_ldata =ldata[:10]
print s_ldata
print s_ldata.pivot('date','item')
print s_ldata.pivot('date','item')['value'][:5]
print s_ldata.set_index(['date','item']).unstack('item')    # 这个操作是完整的pivot 操作
print s_ldata.set_index(['date','item']).unstack()

'''

##=================================================       重复数据的移除
data = DataFrame({'k1': ['one'] * 3 + ['two'] * 4,
                  'k2': [1, 1, 2, 3, 3, 4, 4]})
print data
print data.duplicated()               #   返回bool型
print data.drop_duplicates()
data['v1'] = range(7)
print data
print data.drop_duplicates(['k1'])        #   指定某一列 去重
print data.drop_duplicates(['k1', 'k2'])   #  默认保留 第一次出现的值
print data.drop_duplicates(['k1', 'k2'],keep='last')        #    设置为保留最后一次出现的值

#--------------------------------------利用函数或映射进行数据转换
data = DataFrame({'food': ['bacon', 'pulled pork', 'bacon', 'Pastrami',
                           'corned beef', 'Bacon', 'pastrami', 'honey ham',
                           'nova lox'],
                  'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
print data
meat_to_animal = {
  'bacon': 'pig',
  'pulled pork': 'pig',
  'pastrami': 'cow',
  'corned beef': 'cow',
  'honey ham': 'pig',
  'nova lox': 'salmon'
}
print data['food']
data1 = map(str.lower,data['food'])
data['animal'] = data['food'].map(str.lower).map(meat_to_animal)  #   先转换成小写 再映射
print data
print data['food'].map(lambda x:meat_to_animal[x.lower()])      #   等价于上面的

#---------------------------------数据的标准化
data = pd.read_excel(path + 'normalization_data.xls',header=None)
print data
data1 = (data - data.min())/(data.max()-data.min())     #   最小-最大规范化
print data1
data2 = (data - data.mean())/data.std()                #零-均值规范化
print data2
data3 = data/10**np.ceil(np.log10(data.abs().max()))    #小数定标规范化  np.ceil(x):返回不小于x的最小整数


#-----------------------------------数据的替换
data = Series([1., -999., 2., -999., -1000., 3.])
print data
print data.replace(-999, np.nan)
print data.replace([-999, -1000], np.nan)
print data.replace([-999, -1000], [np.nan, 0])
print data.replace({-999: np.nan, -1000: 0})

#-----------------------------------轴索引 重命名
data = DataFrame(np.arange(12).reshape((3, 4)),
                 index=['Ohio', 'Colorado', 'New York'],
                 columns=['one', 'two', 'three', 'four'])
print data
data.index = data.index.map(str.upper)                   #  改横索引的形式
print data
data = data.rename(index=str.title,columns=str.upper)    #str.title 首字母大写
print data
print '--------------'
print data
data1 = data.rename(index={'Ohio': 'INDIANA'},       #     替换 横列 的 索引名
            columns={'THREE': 'peekaboo'})
print data1
_ = data.rename(index={'Ohio': 'INDIANAppppppp'}, inplace=True)    #   对原数据结构改变
print data

#-----------------------------------离散化  cat() http://blog.csdn.net/g_66_hero/article/details/73065698
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]        # 切割成什么
cats = pd.cut(ages, bins)
print cats
print cats.codes                       #  表明  数组的元素 各被标记为 什么类
print cats.categories                   #   有哪些类
print pd.value_counts(cats)            # 各个类 包含几个
print pd.cut(ages, [18, 26, 36, 61, 100], right=False)     #  左开右闭 的区间
group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
print pd.cut(ages, bins, labels=group_names)
data = np.random.rand(20)
print data
print pd.cut(data, 4, precision=2)

#---------------------------cut()函数划分得到的面元，每个面元的数量不同。而qcut()可以保证每个面元的数量相同，且每个面元的区间大小不等。
data = np.random.randn(1000)             # Normally distributed
cats = pd.qcut(data, 4)                  # Cut into quartiles
print cats
print pd.value_counts(cats)
print '----------------'
print pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.])

np.random.seed(12345)
data = DataFrame(np.random.randn(1000, 4))
print data.describe()                     # describe()函数对于数据的快速统计汇总
col = data[3]
col[np.abs(col) > 3]
print data[(np.abs(data) > 3).any(1)]

#------------------------------------使用permutation()函数可以创建一个随机顺序的数组。
#-------------------------------------使用take()函数可以采用新的索引次序

df = DataFrame(np.arange(5 * 4).reshape((5, 4)))
sampler = np.random.permutation(5)
print sampler
print df
print df.take(sampler)

bag = np.array([5, 7, -1, 6, 4])
sampler = np.random.randint(0, len(bag), size=10)               # 随机数产生，可以 进行重抽样
print ';;;;;;;;;'
print sampler
draws = bag.take(sampler)
print draws


#--------------------------  哑变量
df = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
                'data1': range(6)})
print df
print pd.get_dummies(df)
print '[[[[[[['
dummies = pd.get_dummies(df['key'], prefix='key')
print dummies
df_with_dummy = df[['data1']].join(dummies)
print df_with_dummy

s = pd.Series(list('abca'))
print s
print pd.get_dummies(s)
df1 = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b','a','c'],'C': [1, 2, 3]})
print df
print pd.get_dummies(df1, prefix='key')          #  加 前缀
print pd.get_dummies(pd.Series(list('abcaa')))
print '[[[[[[[[[[[[[[[[[['


#                       复杂的  自定义哑变量的情况
mnames = ['movie_id', 'title', 'genres']
movies1= pd.read_table(path + 'movies.dat')
print movies1.head(10)
print "****************"
movies = pd.read_table(path + 'movies.dat',sep= "::",names=mnames,header=None)
print movies.head(10)
#         genre_iter 为一个生成器 ()后，不是生成一个tuple，而是生成一个generator
genre_iter = (set(x.split('|')) for x in movies.genres)
for x in  movies.genres[:10]:
    print "&&&&  >>>> ",set(x.split('|'))
#print set.union(genre_iter)
genres = sorted(set.union(*genre_iter))
print genres
dummies = DataFrame(np.zeros((len(movies),len(genres))),columns=genres)
#print dummies[:10]
for i,gen in enumerate(movies.genres):      #迭代每一部电影并将dummies各行的项设置为1
    print gen.split('|')
    print
    print  "*******",i,dummies.ix[i,gen.split('|')]
    dummies.ix[i,gen.split('|')] = 1
print dummies[:10]
print "???????????????????"
print dummies.add_prefix('Genre_')
movies_windic = movies.join(dummies.add_prefix('Genre_'))    #   添加 dummies 到 movies 里
print ">>>>>>>>>>>>>>>"
print movies_windic[:10]
print ">>>>>>>>>>>>,<<<<<<<<<<<,:::::::",movies_windic.ix[0]

#                       线损率属性构造
data = pd.read_excel(path+'electricity_data.xls')
print data
data[u'线损率'] = (data[u'供入电量'] - data[u'供出电量'])/data[u'供入电量']
print data
data.to_excel(path + 'redify_electricity.xls',index = False)



#================================pd.get_dummies  &  pd.cut  结合使用
np.random.randint(12345)
arr = np.random.rand(10)
print arr
bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
print pd.cut(arr,bins)
print pd.get_dummies(pd.cut(arr,bins))


val = ' a,b,  guido '
#s_val = val.split(',')
#print s_val
#li = []
#for x in s_val:
 #   li.append(x.strip())
#print li
#    等价于
li = [x.strip() for x in val.split(',')]
print li
#================                     =###pandas中矢量化的字符串函数

data = {'Dave': 'dave@google.com', 'Steve': 'steve@gmail.com',
        'Rob': 'rob@gmail.com', 'Wes': np.nan}
data = Series(data)
print data
print data.isnull()
print data.str.contains('gmail')        #  判断是否包含   gmail
pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
print " -------------->"
print data.str.findall(pattern,flags =re.IGNORECASE)
matc = data.str.match(pattern,flags = re.IGNORECASE)
print matc
print matc.str.get(1)
print matc.str[0]
print data.str[:5]


#=====================================================eg:https://www.cnblogs.com/virusolf/p/6231748.html

data = json.load(open(path+'foods-2011-10-03.json'))
print len(data)
print type(data)
print data[0].keys()
print data[0]['nutrients']
print data[0]['nutrients'][0]
nutrients = DataFrame(data[0]['nutrients'])
print nutrients
info_keys = ['description','group','id','manufacturer']
info = DataFrame(data,columns=info_keys)
print "?????????????????"
print info[:10]
print pd.value_counts(info.group)
nutrients_whole = []
for x in data:
    fnuts = DataFrame(x['nutrients'])
    fnuts['id'] = x['id']
    nutrients_whole.append(fnuts)
#print nutrients_whole[:10]
nutrients_whole1 = pd.concat(nutrients_whole,ignore_index=True)  #  通过连接方式  把list 整合成DataFrame  ,ignore_index=True:遇到两张表的列字段本来就不一样，但又想将两个表合并，其中无效的值用nan来表示
print type(nutrients_whole)
print len(nutrients_whole)
print '>>>>>>>>>>>>>>'
print type(nutrients_whole1)
print nutrients_whole1[:10]
print nutrients_whole1.duplicated().sum()      #        去重
print len(nutrients_whole1)
nutrients_whole_clean = nutrients_whole1.drop_duplicates()
print nutrients_whole_clean[:10]
print len(nutrients_whole_clean)
new_name = {'description':'food',
            'group':'fgroup'}
info = info.rename(columns = new_name,copy = False)  #由于nutrients_whole_clean 与info有重复的名字，所以需要重命名一下info
print info[:10]
print "//////////////////"
print nutrients_whole_clean.ix[:10,:]
data1 = pd.merge(nutrients_whole_clean,info,on='id')
data2 = pd.merge(nutrients_whole_clean,info,on='id',how='outer')      #   连接两个Dataframe
print "*******************************"
print data2.columns
print data1[:10]
print data2[:10]
print data1.ix[30000]
result = data2.groupby(['food','fgroup'])['value'].quantile(0.5)
print "<<<<<<<<<<<<<<<<<<"
print result
result1 = data2.groupby(['description','group']).head()
print result1
#result['Zinc, Zn'].order().plot(kind='barh')
#plt.show()
print "??????????????"
#print result1.xs
get_maximum = lambda x : x.xs(x.value.idxmax())  #最大最小值的索引值
get_minimum = lambda x:x.xs(x.value.idxmin())
print result1.columns
max_foods = result1.apply(get_maximum)[['value','food']]
print max_foods
max_foods.food = max_foods.food.str[:50]
'''
