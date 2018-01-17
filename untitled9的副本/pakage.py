#encoding:UTF-8
import numpy as np
from datetime import datetime
import sys
import os
import random
from numpy.random import randn
import matplotlib.pyplot as plt
from numpy.linalg import inv,qr
import requests
from pandas import Series,DataFrame
#import cv2
import time
import logging
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
'''
def npsum(n):
    a = np.arange(n)**2
    b = np.arange(n)**3
    #print (a)
    #print (b)
    return a + b
c = npsum(5)
print c
print(c[-3:])
start = datetime.now()
size = 1000
c = npsum(size)

delta = datetime.now()-start
print delta.microseconds,c[-2:]

a = np.arange(5)

print type(a),a.shape,a.dtype

m = np.array([np.arange(3),np.arange(3),np.arange(3)])
print m,m.shape,m.dtype,type(m)

t = np.zeros(10)
print t,t.dtype,t.shape

t1 = np.zeros((3,4))
print t1,t1.shape,t1.dtype

t = np.empty([4,3,2],dtype=int)
print t

def add(a):
    return a * 2
li = [1,2,3,4,5,6,7]
p = map(add,li)
print p


a = np.array(([1,2],[3,4]))
print a
print a[0,0]
print np.float(2)


try:
    print np.int(42.0+1.j)
except TypeError:
    print "TypeError"


arr = np.array([1,2,3,4,5,6])
print arr,arr.dtype
flo = arr.astype(np.float64)
print flo
flo1 = np.array([1.2,-2.3,4,5.9])
int_arr = flo1.astype(np.int64)
print int_arr


print np.array(7,dtype="D")

t = np.dtype("d")
print t.type
print t.char
print t.str


t = np.dtype('float64')
print t.char
print t.type
print t.str

arr = np.array([[1,2,3],[2,3,4]])
print arr
print arr.shape
print arr * arr
print arr[1]
a = np.arange(9)
print a
print a[1:4]
print a[:7:3]
s=slice(None,7,3)
print a[::-1],a[s]   
b = np.arange(24).reshape(2,3,4)
print b
print b[0]
print b[:,0,0]
print b[0,:,:]
print b[0,...]
print b[0,1,::2]
print b[...,1]
print b[:,1]
print("-----")
print b[0,:,1]
print b[0,::-1,1] #-----
print "-----------------"
print b[0,::2,-1]            #--------------::2 步长
print b[0,::3,-1]
s = slice(None,None,-1)
print b[s]
print b[(s,s,s)]
b =  np.arange(24).reshape(2,3,4)
print b
print b[0,::2,-1]  #  选取b中的第一个元素，-1 代表选最后一列，：：2代表步长为2
print b[...,1]

#-------------------------------------布尔索引
names = np.array(['Bob','Joe','will','Bob','will','joe'])
data = randn(6,4)  #  从正态分布中随机取数据
print names
print data
print "------------"
print data[names=='Bob']
print data[names=='Bob',1:]  # ，后面代表对列的选取
print "________________"
data[names!='joe']= 7
print data


arr = np.empty((8,4))
print arr
for i in range(8):
    arr[i]= i
print arr
print "============"
#========================================索引
arr = np.arange(32).reshape(8,4)
print arr
print arr[[1,4,5,2],[0,2,3,1]]
print arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]]   #对[1, 5, 7, 2]]中选出的行按照[0, 3, 1, 2]的列排列
print arr[np.ix_([1, 5, 7, 2],[0, 3, 1, 2])]   #  和上一行达到的效果一样

arr = np.arange(15).reshape(5,3)
print arr.ravel('F')
print(arr)
print arr.ravel()     # 默认按行排
print '====='
print arr.reshape(-1)
print arr.T.reshape(-1)
print arr
print arr.flatten('F')   #  按列排
print arr.T                  #  -----              数组转置

a = np.arange(9).reshape(3,3)
b = 2 * a
print a
print b
print np.hstack((a,b))
print np.concatenate((a,b),axis=1)
print np.vstack((a,b))
print np.concatenate((a,b),axis=0)

a = [[1],[2],[3]]
b = [[4],[5],[6]]
print np.hstack((a,b))
print np.column_stack((a,b))
print np.vstack((a,b))
print np.row_stack((a,b))
print np.dstack((a,b))


a = np.arange(8).reshape(4,2)
print a
b = np.hsplit(a,2)
print b
c = np.vsplit(a,4)
print c

#   数组属性
b = np.arange(24).reshape(2,12)
print b
print b.size       #  总的元素的个数
print b.ndim       #   数组的维度
print b.itemsize   #  单个元素占的字节数
print b.nbytes     # b 占的总空间

print b.flat       #   访问b的数据 也可以更改
print b.flat[19]
b.flat[19]=90
print b.flat[[3,6]]
print b
c = np.array([1+4.j,3+ 2.j])
print c.real             # 输出复数的实部
print c.imag             # 输出复数的虚部
print "-----"
# 数组的转换
print c.astype(int)
print c.astype('complex')
print c.tolist()
print c.tostring()
print np.fromstring('23:54:09',sep=':',dtype=int)

#====================================numpy function

arr = np.arange(10)
print arr
print np.sqrt(arr)
print np.exp(arr)    # 求指数
x = randn(8)
y = randn(8)
print x
print y
print np.maximum(x,y)     #   对应元素取最大
print np.modf(x)           #  将其分类，成两个数组，一个为浮点型，一个为整形

points = np.arange(-5,5,0.01)
xs,ys = np.meshgrid(points,points)    #meshgrid函数通常在数据的矢量化上使用,meshgrid的作用适用于生成网格型数据，可以接受两个一维数组生成两个二维矩阵，对应两个数组中所有的(x,y)对
x = np.arange(5)
y = np.arange(2)
print np.meshgrid(x,y)
print xs,ys
z = np.sqrt(xs ** 2 + ys ** 2)
#    作图
plt.imshow(z,cmap=plt.cm.gray)
plt.colorbar()
plt.title("Image plot of $\sqrt{x^2+y^2}$for a grid of values")
plt.draw()

#===================================数组与逻辑条件的运算
x = np.array([1,2.3,4,2,4])
y = np.array([2,3.2,1.2,2,3])
condition = np.array([True,False,True,True,False])
result = [(z if c else p)
          for z,p,c in zip(x,y,condition)]
print result
result1 = np.where(condition,x,y)
print result1

arr = randn(4,4)
print arr
arr1 = np.where(arr > 0,2,-2)
print arr1
arr2 = np.where(arr > 0 ,2,arr)
print arr2

cond1 = [True,False,True,False,False]
cond2 = [True,True,False,True,False]
result = []
result1 = list()
for i in range(5):
    if cond1[i] and cond2[i]:
        result.append(0)
    elif cond1[i]:
        result.append(1)
    elif cond2[i]:
        result.append(2)
    else:result.append(3)
print result
#  等价于
np.where(cond1 and cond2,result1.append(0),
         np.where(cond1,result1.append(1),
                  np.where(cond2,result1.append(2),result1.append(3))))
print result1

#=========================================数学与统计方法
arr = np.random.randn(5,4)
print arr
print (arr > 0).sum()        #  返回arr中大于0 的元素的个数
print arr.mean()
print np.mean(arr)
print arr.mean(axis=1)  #   每一行的mean
print arr.mean(axis=0)   # 每一列的mean
arr1 = np.array([[1,2,3,5],[4,5,6,6],[7,8,9,7]])
print arr1
cond2 = np.array([True,True,False,True,False])
print "****"
print cond2.any()
print cond2.all()
##======================array.cumsum & array.cumprod  不能用axis
print arr1.cumsum()   #  🈚️每一个数字一次累加
print arr1.cumsum(0)  #  每一列依次累加
print arr1.cumsum(1)  #   每一行依次累加
print arr1.cumprod()  #  所有元素的累积积

#====================================  排序
arr = randn(10)
print arr
arr.sort()        #  直接使用sort 即直接在arr上更改
print arr
arr1 = randn(5,3)
print arr1
arr1.sort(1) #  每一行进行排序
print arr1
arr1.sort(0)
print arr1   #  每一列进行排序
arr = randn(4,5)
print arr
print np.sort(arr)   #  使用np.sort  不会改变数组本身
print arr

#===================================唯一化以及其他集合逻辑
names = np.array(['bu','bu','ak','ko','kk','lk','ko'])
a = np.unique(names)
print a[0]
print sorted(set(names))[0]
val = np.array([1,2,3,4,5,6,5,4,3])
print np.in1d(val,[3,4])       #val中的每一个值是否在[3,4]中，如果是，该值返回true,否则返回false.返回bool值

#============================================线性代数
position = 0
walk = []
steps = 1000
for i in xrange(steps):
    step = 1 if random.randint(0,1) else -1
    position += step
    walk.append(position)
npsteps = 1000
np.random.seed(12345)
draws = np.random.randint(0,2,npsteps)
npstep = np.where(draws>0,1,-1)
print npstep
walk1 = npstep.cumsum()
print walk1
print walk1.min()
print walk1.max()

nwalk = 5000
nstep = 1000
draws = np.random.randint(0,2,size=(nwalk,nstep))
print draws
nstep = np.where(draws > 0,-1,1)
walk2 = nstep.cumsum(1)
print walk2
flag = (np.abs(walk2 > 50)).any(1)
print flag
print flag.sum()
flag1 = (np.abs(walk2[flag]) > 50).argmax(1)
print flag1


matr = randn(4,5)
matr1 = randn(5,4)
mat = matr.dot(matr1)
print(mat)
q ,r = qr(mat)     
print q
print r

L=[('b',2),('a',1),('c',3),('d',4)]
t = sorted(L, cmp=lambda x,y:cmp(x[1],y[1]))

print sorted(L)

a = [1,2,3,4]
print a[4/2],a[4/2-1],(a[4/2]+a[4/2-1])/2

a = [[1],[2],[3]]
b = [[4],[5],[6]]
print np.hstack((a,b))
print np.column_stack((a,b))
print np.vstack((a,b))
print np.row_stack((a,b))
print np.dstack((a,b))
print "kk============="
d = [[1/3],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[5],[4]]
print d
print d[2:-2]
print d[-2:2]
li = range(10)
print li
print li[-2:-8:-2]

print np.eye(3,k=-2)      #   np中eye与identity的区别
print np.identity(3)


s1 = Series([0, 1], index=['a', 'b'])
s2 = Series([2, 3, 4], index=['c', 'd', 'e'])
s3 = Series([5, 6], index=['f', 'g'])
ss=pd.concat([s1, s2, s3])
st=pd.concat([s1,s2,s3],axis=1)
print s1
print s2
print s3
print ss
print st
dd = DataFrame(st)
print "pppppp"
print dd
print dd.xs('f')


Mu_column= pd.MultiIndex.from_arrays([['US', 'US', 'US', 'JP', 'JP'],
                                    [1, 3, 5, 1, 3]], names=['city', 'tensor'])
DF_mu = DataFrame(np.random.randn(4,5),columns=Mu_column)
print DF_mu
print DF_mu.groupby(level='tensor',axis=1).sum()
print DF_mu.groupby(level='tensor',axis=1).size()
print DF_mu.groupby(level='city',axis=1).size()
print DF_mu.groupby(level='tensor',axis=1).count()
print 'P?????????????'
print DF_mu.groupby(level='tensor',axis=0).sum()
print DF_mu.groupby(level='tensor',axis=0).size()
print DF_mu.groupby(level='city',axis=0).size()
print DF_mu.groupby(level='tensor',axis=0).count()

'''
b = np.arange(24).reshape(2,12)
print b
print b[:,1::]



