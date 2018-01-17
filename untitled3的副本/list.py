#-------------------------list
import random
'''li = [11,22,33,4,51,6,13]
i = 0
while i < len(li):
    print(li[i],end=" ")
    i+=1
print()
for x in li:
    print(x,end="/")
print()
for x in li:
    print("for",x)
    li.pop(0)
    print(li)
max = li[0]
# --------- for循环找max
for x in li:
    if max< x:
        max=x
print(max)
#-----------way1:while 找  max  与   second
index = 0
max = li[0]
while i < len(li):
    if li[i] > max:
        max = li[i]
        index = i
    i+=1

print("max",max,index)
i = 0
temp = 0
i=0
second = -1
while i < len(li) :
    if i != index :
        if li[i] > second :
            second = li[i]
    i+=1
print("second-max",second)   

# --------------------way 2: while---find  max and second-max
max = li[0]
second_max = li[0]
i = 0
while i < len(li):
    if max < li[i]:
        second_max = max
        max = li[i]
    else:
        if second_max < li[i]:
            second_max = li[i]
    i+=1
print("while  ",max,second_max)

#--------------------way 2:for----find  max and second-max
max = li[0]
second_max = li[0]
for x in li:
    if max < x:
        second_max = max
        max = x
    else:
        if second_max < x:
            second_max = x
print("for ",max,second_max)   
lis = ["ji",10,[29]]
print(lis + [10])
print(lis) 
li1 = list(range(1,11))    #--------range(start,end,步长）
li2 = list(range(10,20))
print(li1,len(li1))
print(li2,len(li2))
i = 0
while i < len(li1):
    li1[i]+=li2[i]
    i+=1
print(li1,type(li1))
if type(li1) == list:
    print("right")
#-----------------------打印多层列表
li = [1,[2,3],[4],[5],6,[[[7]],8]]
print(li)
i = 0
for x in li :
    if type(x) == list:
        for y in x:
            if type(y) == list:
                for z in y:
                    if type(z) == list:
                        for w in z:
                            if type(w) == list:
                                print("*",end=" ")
                            else:
                                print(w,end=" ")
                    else:
                        print(z,end=" ")
            else:
                print(y,end=" ")
    else:
        print(x,end=" ")


#-------------------------list 索引
li = [1,[2,3],[4],[5],6,[[[7]],8]]
print(li.index([[[7]],8]))
print(li.index(6))
li1 = li[li.index([4]):li.index([[[7]],8])]
li2 = li[0:2]
print(li1)
print(li2)
li3 = [1,32,2,3,4,5,6,7,8,9,10]
s = li3[li3.index(32)-len(li3):li3.index(10)-len(li3)]
s1 = s[-9:-1]
print(li3.index(10)-len(li3))
print(li3)
s2 = li3[-11:-1]
print(s2)
#-----------list  步长的方向一定要与start到end的方向一致：起点到终点指向x的负半轴 则步长为负数，切出来的为倒序
#                                                    起点到终点指向x的正半轴 则步长为正数，切出来的为正序
print(li3[0:11:2])         # 输出也能用步长
print(li3[-1:1:-2])
print(li3[::-1])

temp = (1,2,3,4,5,6)  # 元组tuple 元素只能只读，与string一样 不能被赋值
li = list(temp)
print(temp)
li[2] = 9            # 列表可以赋值
print(li)
#temp[2] = 9
#print(temp)
s = "josa sm ksl 8nk *mk"
s1 = list(s)
print(s1)
s2 = tuple(s1)
print(type(s2))
#------------------------s.split()  返回列表
ss = "nsj mkl lska k*j Ma"
sp = ss.split()
for x in sp:
    print(type(x),x)
print(sp)
spl = ss.split("s")
print(type(spl)) 

#----------------------list.insert()

temp = ("dsnaj,sj,skdx ,skla")
li = list(temp)
print(li)
li.insert(1,"jj")
print(li)
li1 = list(range(1,11))
li2 = list(range(2,5))
i = 0
#   顺叙插
while i < len(li2):
    li1.insert(5+i,li2[i])
    i+=1
#   逆序插
while i < len(li2):
    li1.insert(5,li2[len(li2)-1-i])
    i+=1
print(li1)
#-------------------list.append() &list.extend 动态扩长 尾部添加元素  前者以整体形式，后者把每个元素分别加入
s = [23,"osq"]
li1 = [2,3,4,5,6,7,8,9,[28,3,4],3,4]
li = [1,2,3,4,5,6,6]
li.append(s)
print(li)
li.extend(s)
print(li)
print(type(li[li.index("osq")]))
sds = li+li1
print(sds)
li.extend(li1)
print(li)
#------------------append实现extend

li = [1,2,3,4,5]
lis = "sjka"
i=0
while i<len(lis):
    li.append(lis[i])
    i+=1
print(li)
li.extend(4,9)
print(li)
'''
#--------------------list.pop()

'''li = [1,2,3,4,5,6,7,8,9]
print(li.pop(3))
# 删除重复元素4，只保留第一个4
li1 = [1,2,3,4,1,2,1,11,2,3,4,4,4,5,6]
i = 0
while i < len(li1):
    if li1[i] == 4:
        temp1 = li1[:i+1]
        temp2 = li1[i+1:]
        break
    i+=1
i = 0
while i < len(temp2):
    if temp2[i] == 4:
        temp2.pop(i)
        i-=1
    i+=1
li1 = temp1 + temp2
print(li1)
#删除所有的重复元素
#  方法一
li = [1,1,2,2,33,4,4,4,3,3,4,4,5,5,6,6,7,7,8,8]
i = 0
while i < len(li):
    if li.count(li[i]) >1:
        #pos = li.index(li[i])
        temp1 = li[:i+1]
        temp2 = li[i+1:]
        j = 0
        while j < len(temp2):
            if temp2[j] == li[i]:
                temp2.pop(j)
                j-=1
            j+=1
    li = temp1 + temp2
    i+=1
print(li)

#   方法二
li = [1,1,1,2,2,1,2,33,4,4,4,3,3,4,4,5,5,6,6,7,7,8,8]

i = 0
while i < len(li):
    x = li.index(li[i])
    #print(li.count(li[i]))
    while li.count(li[i]) > 1:
        li.pop(li.index(li[i],x+1))
       # i -= 1
    i+=1
print(li)

#      方法三
li = [1,1,1,2,2,1,2,33,4,4,4,3,3,4,4,5,5,6,6,7,7,8,8]
i = 0
while i < len(li):
    x = li.index(li[i])
    if li.count(li[i]) > 1:
        li.pop(li.index(li[i],x+1))
        i -= 1
    i+=1
print(li)
#     方法四
li = [1,1,1,2,2,1,2,33,4,4,4,3,3,4,4,5,5,6,6,7,7,8,8]
lit = []
i = 0
while i < len(li):
    if li[i] not in lit:
        lit.append(li[i])
    i += 1
print(lit)
#      方法 五
li = [1,1,1,2,2,1,2,33,4,4,4,3,3,4,4,5,5,6,6,7,7,8,8]
lit = []
i = 0
for x in li:
    if x not in lit:
        lit.append(x)
print(lit)
#      方法六 -------------------------------set() 对象为列表或者元组，返回没有重复值的列表或元组，顺序与之前的顺序不一定一样
li = [1,1,1,2,2,1,2,33,4,4,4,3,3,4,4,5,5,6,6,7,7,8,8]
lit = []
lit = list(set(li))
print(lit)

'''
#--------------------------list.remove()
'''li = [1,2,2,3,4,5,6,6,6,7,5,4,3]
i = 0
#while i <li.count(6) :               典型错误：每次remove都会影响li.count的值
#    li.remove(6)
 #   i+=1
#print(li)
x = li.count(6)
while i < x:
    li.remove(6)
    i+=1
print(li)

#===================================================
li = []
i = 0
while i < 10:
    x = random.randint(10,15)
    li.append(x)
    i += 1
print(li)
temp = []
for x in li:
    if x not in temp:
        temp.append(x)
print(temp)
for y in temp:
    if li.count(y) > 1:
        print(y,li.count(y))
'''
#-===================================== li.extend(): 单个插入，与li.appened 不一样
#====================================== li.insert()  具体位置插入
#----------------------==================排序
'''li = [1,3,2,4,6,5,8,7]
i = 0
while i < len(li):
    j=i+1
    while j < len(li):
        if li[i] < li[j]:
            temp = li[i]
            li[i] = li[j]
            li[j] = temp
        j += 1
    i += 1
print(li)
li.sort()
print(li)
'''


#   ------------------------------------------------mac 绘图中文显示问题
import numpy as np
import pandas as pd
import matplotlib
import pylab
from matplotlib.font_manager import FontManager, FontProperties
import subprocess
import matplotlib.pyplot as plt
from numpy.random import randn

def getChineseFont():
    return FontProperties(fname='/System/Library/Fonts/PingFang.ttc')

'''
if __name__ == '__main__':
    plot.title(u"我是道哥", fontproperties=getChineseFont())
    plot.ylabel(u"我觉得",fontproperties=getChineseFont())
    plot.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(randn(1000).cumsum(),'k',label = 'one')
ax.plot(randn(1000).cumsum(),'g--',label = 'two')
ax.plot(randn(1000).cumsum(),'r.',label = 'three')
ax.legend(loc = 4)
plt.show()
'''
b = np.arange(24).reshape(2,12)
print(b)
print(b[:,1::])