#encoding:UTF-8
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import csv
import xlrd,xlwt
import json
import requests

#  Series

obj = Series(range(10))         #  类似 dict
print obj
print obj.values
print obj.index
obj1 =Series(range(5),index=['a','b','c','d','e'])
print obj1
print obj1.index
print obj1['b']       #  索引找值
print obj1[obj1 > 2]
obj1['a'] = 100                #  可以通过索引赋值
print obj1
print obj1 **2                 #  改变value 索引不变
print np.exp(obj1)
print 'g'in obj1               #  判断索引是否存在与Series中

dict1 = {'a':1,'b':3,'v':0,'p':9}
print dict1
obj1 = Series(dict1)
print obj1
#dict2 = {'c':8,'f':6,'a':2}
li = ['c','f','a','b']
obj2 = Series(dict1,index=li)     #  重新通过索引去匹配value
print obj2
print pd.isnull(obj2)             #  value 为NaN 返回True
print pd.notnull(obj2)            #   value 为非空 返回True
print obj1 + obj2
obj2.name = "kokoa"
obj2.index.name = "char"
#  obj2.values.name = "value"     不存在
print obj2

#  Dataframe
dict2 = {"name":['a','b','c','d'],
         "Age" :[12,24,46,67],
         "sex" :['男','女','女','男']
         }
print dict2
frame = DataFrame(dict2)
print frame
print DataFrame(frame,columns=['sex','Age','name'])   #  改 列的顺序
print frame['sex']       #  此结构就类似与Series
print frame.Age          #  获取列
print frame
print frame.ix[2]        #  获取 行
obj3 = Series([00,22],index=[0,2])
frame['new'] = obj3
print frame
frame['new1'] = frame.sex=='男'     #  用逻辑判断 添加列
print frame

del frame['new']    #  删除某列
print frame


dict3 = {'aa':{'a1':1,'a2':2,'a3':3},'bb':{'b1':4,'b2':5}}    #  外键 作为列，内键 为 行名
frame1 = DataFrame(dict3)
print dict3
print frame1.T
print frame1
pdata = {'aa':frame1['aa'][:-1],'bb':frame1['bb'][:4]}
frame2 = DataFrame(pdata)
frame2.index.name = "Name"               #  设置整个的 行名
frame2.columns.name = "whea"             # 设置 整个的列名
print frame2
print type(frame2.values)


index = pd.Index(np.arange(4))
print index
obj3 = Series([1.2,1.3,1.4,1.5],index=index)
print obj3
print obj3.index is index
print 4 in obj3.index                   #  逻辑判断某个值是否在index中
frame4 = DataFrame(obj3,columns=['new'])
print frame4
print 'new' in frame4.columns            #  逻辑判断某个值是否为列名
print 1.2 in frame4.columns


df = pd.read_csv('/Users/wss/Desktop/py/第5周/data/ex1.csv',header=None)
print df
pd1 = pd.read_table('/Users/wss/Desktop/py/第5周/data/ex1.csv')
print pd1
pd2 = pd.read_table('/Users/wss/Desktop/py/第5周/data/ex1.csv',sep=',')
print pd2

pd3 = pd.read_csv('/Users/wss/Desktop/py/第5周/data/ex2.csv',names=['a','b','c','d','message'])  #   添加列名
print pd3
pd4 = pd.read_csv('/Users/wss/Desktop/py/第5周/data/ex2.csv',names=['a','b','c','d','message'],index_col='message')         #  把某一列作为行的索引
print pd4
pd5 = pd.read_csv('/Users/wss/Desktop/py/第5周/data/csv_mindex.csv')
print pd5
pd6 = pd.read_csv('/Users/wss/Desktop/py/第5周/data/csv_mindex.csv',index_col=['key1','key2'])
print pd6
pd4 = pd.read_csv('/Users/wss/Desktop/py/第5周/data/ex3.csv')
print pd4
print list(open('/Users/wss/Desktop/py/第5周/data/ex3.csv'))
pd7 = pd.read_csv('/Users/wss/Desktop/py/第5周/data/ex3.csv',sep='\s+')
print pd7
pd8 = pd.read_csv('/Users/wss/Desktop/py/第5周/data/ex4.csv')
print pd8
pd9 = pd.read_csv('/Users/wss/Desktop/py/第5周/data/ex4.csv',skiprows=[1,2,3])
print pd9

pd10 = pd.read_csv('/Users/wss/Desktop/py/第5周/data/ex5.csv')
print pd10
print pd.isnull(pd10)
pd11 = pd.read_csv('/Users/wss/Desktop/py/第5周/data/ex5.csv',na_values=['NULL'])
print pd11

pd = pd.read_csv('/Users/wss/Desktop/py/第5周/data/ex6.csv')
print pd

print pd.read_csv('/Users/wss/Desktop/py/第5周/data/ex6.csv',nrows = 10)    #  指定输出前十行
print pd.read_csv('/Users/wss/Desktop/py/第5周/data/ex6.csv',chunksize = 1000)
print '**',

#   文件写出

pd10 = pd.read_csv('/Users/wss/Desktop/py/第5周/data/ex5.csv')
print pd10
pd10.to_csv('/Users/wss/Desktop/py/第5周/data/OOO.csv')        #   写出文件
pd10.to_csv('/Users/wss/Desktop/py/第5周/data/OO1.csv',sep="|")
pd10.to_csv('/Users/wss/Desktop/py/第5周/data/OO1.csv',na_rep="??")   #  替换缺失值
pd10.to_csv('/Users/wss/Desktop/py/第5周/data/OO2.csv',na_rep="??",index=False,header=False,columns=['a','b','c'])    #  去掉行列标签

dat = pd.date_range('6/6/2000',periods=7)    #时间序列数据的索引
print dat
ts = Series(np.arange(7),index=dat)
print ts
ts.to_csv('/Users/wss/Desktop/py/第5周/data/OO3.csv')
print Series.from_csv('/Users/wss/Desktop/py/第5周/data/OO3.csv',parse_dates=True)  #指定含有时间数据信息的列

#  手工处理分隔符格式
f = open('/Users/wss/Desktop/py/第5周/data/ex7.csv')
print f
reader = csv.reader(f)
print reader
for x in reader:
    print x

lines = list(csv.reader(open('/Users/wss/Desktop/py/第5周/data/ex7.csv')))
print lines[0],lines[1:]
header, value = lines[0],lines[1:]
print zip(*value)
dict1 = {h : v for h,v in zip(header,zip(*value))}
dic = Series(dict1)
dic.to_csv('/Users/wss/Desktop/py/第5周/data/UU.csv')


#  excel  文件的处理
path = '/Users/wss/Desktop/py/第5周/data/'
wb = xlwt.Workbook(encoding = 'utf-8')    #   为了防止写入中文报错
print wb
wb.add_sheet('frist sheet',cell_overwrite_ok=True) # cell_overwrite_ok=True 表示可以重复写
wb.get_active_sheet()  #获取当前激活的表
ws1 = wb.get_sheet(0)
print ws1
ws2 = wb.add_sheet('second sheet')
ws3 = wb.add_sheet('third sheet')
ws3.write(0,0,'表示')
ws3.write(1,1,'获取')
data = np.arange(64).reshape(8,8)
print data.shape[0],data.shape[1]   #   表示获取每个维度的大小
print data
ws1.write(0,0,100)    #   表示0行0列写入100
for x in range(data.shape[0]):
    for y in range(data.shape[1]):
        ws1.write(y,x,data[x,y])
        ws2.write(y,x,data[y,x])
wb.save(path + 'wss.xls')

#   excel 的读取

path = '/Users/wss/Desktop/py/第5周/data/'
get_excel = xlrd.open_workbook(path+'wss.xls')
print get_excel.sheet_names()
sheet_1 = get_excel.sheet_by_name('frist sheet')   # 根据sheet名称获取sheet内容
sheet_2 = get_excel.sheet_by_index(2)              # 根据sheet索引获取sheet内容
print sheet_1.name
print sheet_2.name
print get_excel.sheet_names()[2]
print sheet_2.ncols,sheet_2.nrows
ce = sheet_2.cell(0,0)                   #  获得单元格
print ce.value.encode('utf-8'),ce.ctype                   #  单元格的值以及类型
print sheet_1.col_values(3,start_rowx=0,end_rowx=4)       #  不包含end 那行或列
print sheet_1.row_values(2,start_colx=0,end_colx=3)
for r in range(sheet_1.nrows):
    for c in range(sheet_1.ncols):
        print '%d'%sheet_1.cell(r,c).value,                #'%i'%  字符类型转换
    print

obj = """
{"name": "Wes",
 "places_lived": ["United States", "Spain", "Germany"],
 "pet": null,
 "siblings": [{"name": "Scott", "age": 25, "pet": "Zuko"},
              {"name": "Katie", "age": 33, "pet": "Cisco"}]
}


##        json
result = json.loads(obj)
print result
asjson = json.dumps(result)
print asjson
print result['siblings']
df = DataFrame(result['siblings'])
print df

#                       pickle
fram = pd.read_csv('/Users/wss/Desktop/py/第5周/data/ex1.csv')
fram.to_pickle('/Users/wss/Desktop/py/第5周/data/pick.csv')
print pd.read_pickle('/Users/wss/Desktop/py/第5周/data/pick.csv')

url = 'https://api.github.com/repos/pydata/pandas/milestones/28/labels'
resp = requests.get(url)
print resp
data = json.loads('/Users/wss/Desktop/py/第5周/data/resp')
print DataFrame(data)


pd10 = pd.read_table('/Users/wss/Desktop/py/第5周/data/wss.xls',sep=',')
print pd10


path = '/Users/wss/Desktop/py/第5周/data/'
get_excel = xlrd.open_workbook(path+'SCOOP.xls')
jsonr = json.loads(get_excel)
df_elx = DataFrame(jsonr)

print get_excel.sheet_names()
#print get_excel.col_values(9,start_rowx=2,end_rowx=9)

path = '/Users/wss/Desktop/py/第6周/data/'

pp = pd.read_excel(path+'catering_sale.xls')
print pp

path = '/Users/wss/Desktop/py/第6周/data/'
get_excel = xlrd.open_workbook(path+'catering_sale.xls')
print get_excel.sheet_names()
jsonr = json.loads(get_excel)
df_elx = DataFrame(jsonr)
print df_elx

"""
dict2 ={"name":['a','b','c','d'],
         "Age" :[12,24,46,67],
         "sex" :['男','女','女','男'],
         }
dict2 = DataFrame(dict2,index =['ss','sa','sd','sx'])
print dict2
print dict2['Age']
print 'lllllllllll'
print dict2.loc[['ss','sa']]
print dict2.iloc[1:3,0:3]
print dict2.ix[1:3,['Age','sex']]


