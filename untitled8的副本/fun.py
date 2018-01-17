import random

def prinfun(time):
    i = 0;
    while i < time:
        print("wss.hhhhhh")
        i +=1
prinfun(6)
def str_count(s,a):
    i = 0
    c = 0
    while i < len(s) - len(a)+1:
        if s[i:i+len(a)] == a:
            c += 1
        i += 1
    print(c)
s = "djaosdaosd"
#str_count(s,"sd")
def find_substring(s,a):
    i = 0
    global x                      #函数体中修改全剧变量的值 需要 global
    x += 1
    print(x)
    while i < len(s) - len(a)+1:
        if s[i:i+len(a)] == a:
           print(a,i)
           break
        i += 1
    else:
        print(-1)
x = 0
find_substring(s,"sd")

def find_secondMax(a):                 # 有点问题
    max = li[0]
    second_max = li[0]
    i = 0
    while i < len(li):
        if max < li[i]:
            second_max = max
            max = li[i]
        else:
            if second_max < li[i] and li[i] != max:
                second_max = li[i]
        i += 1
    print(max,second_max)
li = [121,11,121,31,31,43,121,52,2]
find_secondMax(li)

def remove_value(li,v):
    i = 0
    while i < len(li):
        if li[i] == v:
            left = li[:i]
            right = li[i+1:]
            li = left + right
            i -= 1
        i += 1
    print(li)
li = [1.2,2,3,4,5,2,2,3,2,1]
remove_value(li,2)

#------------------函数返回值--------------------------
def add(a,b):
    z = a + b
    return z + 10
he = add(1,2)
print(he)


#  ----------------交换两个数
def swap(a,b):
    global x,y           #  全局变量在局部中修改要申明
    t = a
    a = b
    b = t
    x = a
    y = b
    print(a,b)
x = 10
y = 11
swap(x,y)
print(x,y)

#------------------函数的多值返回
def swap2(a,b):
    return b,a
x = 10
y = 11
print(x,y)
x,y = swap2(x,y)
print(x,y)

#===================================函数的预设值
#==========-----------str.find()
def mystr_find(string,sub,start = 0,end = 0):
    i = start
    if end == -1:
        end = len(string)
    while i <= end - len(sub):
        if string[i:i+len(sub)] == sub:
            return i
        i += 1
    else:
        return -1

s = "jklas mka klas we"
a = mystr_find(s,"la",1,9)
print(a)

#====================================函数为形参
#平方
def f(x):
    return x*x
def s(f,t):
    i = 0
    while i < len(t):
        t[i] = f(t[i])
        i += 1
    print(t)
li = list(range(0,10))
s(f,li)

#立方    (m**n)  -------- pow(x,y,z)：这个是表示x的y次幂后除以z的余数

print(2**3)

def li(m,n):
   # return m ** n
   s = 1
   for x in range(1,n+1):
       s = s * m
   return s
def s(f,t,n):
    i = 0
    while i < len(t):
        t[i] = li(t[i],n)
        i += 1
    return t

lis = list(range(1,10))
print(lis)
lis = s(li,lis,3)
print(lis)

#------------------------列表对应元素的乘积

def mul(f,t,w):
    i = 0
    ltt = []
    while i < len(t):
        ltt.append(f(t[i],w[i]))
        i += 1
    return ltt

def m(a,b):
    return a * b
s = []
li1 = list(range(5,10))
li2 = list(range(11,16))
s = mul(m,li1,li2)
print(s)

#=================================      map  作用于每一个元素
li1 = list(range(5,10))
li2 = list(range(11,16))
def pp(x,y):
    return x * y

s1 = []
li1 = list(map(pp,li1,li2))
print(li1)
t = list(map(tuple,"1234"))
print(t)

#  -----三个等长列表取 最小值组成新的列表----
li = list(range(100,10000))
li1 = random.sample(li,10)
li2 = random.sample(li,10)
li3 = random.sample(li,10)
print(li1)
print(li2)
print(li3)
def inter_min(a,b,c):
    if a < b:
        if a < c:
            return a
    else:
        if b < c:
            return b
    return c

li4 = list(map(inter_min,li1,li2,li3))
print(li4)

###-------------------------------------map  and  zip  and  index
def f(x):
    return 2 * x +3
lx = list(range(3,9))
lll =list(zip(*[lx]*2))
print(lx)
print(lll)

ly = list(map(f,lx))
print(ly)
lt = list(zip(lx,ly))
print(lt)
print(ly[lx.index(5)])
ltt = list(zip(*lt))
print(ltt)
for x in ltt:
    if x[0] == 9:
        print(x[1])


def f(x):
    return chr(x)
asci = list(range(97,123))
asc = list(map(f,asci))
print(asc)
lz = list(zip(asc,asci))
print(lz)
print(ord('中'))
print(hex(32))    #转换一个整数对象为十六进制的字符串表示
print(oct(7))     #返回一个整数的八进制表示

#------------------    map  实现  zip
def mp(x,y):
    return (x,y)
li = list(range(1,10))
s = "abcdefghi"
zp = list(map(mp,li,s))
print(zp)

#=======================================   字典  dict{key:value}
d = {1:"a",2:"b",3:"c",4:"d",5:"e"}
print(type(d),d[2])
#   构造空字典
d1 = {}
d2 = dict()
print(d1,d2)
#    构造：{'a':97,'b':98,'c':99,...'z',122}
#     法一
li = list(range(97,123))
i = 0
while i < len(li):
    c = chr(li[i])
    d1[c] = li[i]
    i += 1
print(d1)

d1.keys()
d1.values()
print(d1.keys())
print(d1.values())
#    法二

li1 = list(range(97,123))
key = list(map(chr,li1))
def dd(x,y):
    return {x:y}
d2 = list(map(dd,key,li1))
print(d2)
d3 = {}
for x in d2:
    key1 = x.keys()
    value1 = x.values()
   # print(type(key1),type(value1),key1,value1,list(key1),list(value1))
    #print(key1,value1)
    sa = list(key1)
    sb = list(value1)
    d3[sa[0]] = sb[0]
print(d3)
print(d3.keys())
print(d3.values())

#      方法三
def double(x,y):
     return [x,y]
    #return x,y
    #return (x,y)
li = list(range(97,123))
key = map(chr,li)
li2 = map(double,key,li)
d = {}
d = dict(li2)                 #   dict()  函数
print(d)

#   方法四

li = list(range(97,123))
key = map(chr,li)
tuple = zip(key,li)
d = dict()
d = dict(tuple)
print(d)

#--------------遍历字典
#  法一
key_value = list(d.keys())
for x in key_value:
     print(x,d[x])
#-----------------------dict.items()  以元组形式取出 key，value 顺序可以打乱
#   法二
tupl = list(d.items())
print("____________",tupl)
for x in tupl:
     print(x[0],x[1])

#--------------------通过value值  找key值
#    方法一  重新构造字典
key1 = d.keys()
value1 = d.values()
newd = dict(zip(value1,key1))
print(newd)
print(newd[100])
#   方法二
i = 0
tupl = list(d.items())
for x,y in tupl:
     #print(x,y)
     if tupl[i][1] == 100:
          print(tupl[i][0])
     i += 1

print(d.get("a","????"))
print(d.setdefault("?","hjjj"))    #  dict.setdefault 会把没有查到的添加到 dict中
print(d.get("？","????"))
print(d)

#===================================dict 实现大数据的快速检索
li = list()
for x in range(0,1000000):
     k = random.randint(100000,10000000)
     li.append(k)

d = dict()
i = 0
key = random.sample(li,100000)
for x in range(0,100000):
     val = list()
     name = "mingzi" + str(x)
     year = random.randint(1990,2000)
     if x%3 ==1:
          sex = "famale"
     else :
          sex = "male"
     if x % 5 ==1:
          race = "meng"
     else:
          race = "han"
     val = [name,year,sex,race]
     d.setdefault(key[x],val)
     i += 1
dk = d.keys()
c = 0
s = 0
for x in dk:
     if d[x][1] == 1995:
          c += 1
          if d[x][2]== "famale":
               s += 1

print(s * 1.0/c)

#---------------------------------------dict 查询公交路线 无换乘
sa = "aa bb cc dd ff ed sd ak sf"
st = sa.split()
t = range(1,len(st)+2)
d = dict(zip(t,st))
print(d)
bename = "bb"
endname = "sd"
for x in d.keys():
     if d[x] == bename:
          start = x
     if d[x] == endname:
          end = x
for x in list(d.keys())[start-1:end]:
     print(d[x],"---->",end= "")
print(d[end])

#---------------------------------------dict 查询公交路线 多条路线  无换乘
line_a = "aa bb cc dd ff ee gg sd ii"
line_b = "sx bb fs ss dx fd sd we re"
line_c = "as sd bb ew fk nm sd"

bename = "bb"
endname = "sd"

a_value = line_a.split()
num_a = range(1,len(a_value)+1)
temp_a =zip(num_a,a_value)
d_a = dict(temp_a)
a_key = d_a.keys()
print(d_a)

b_value = line_b.split()
num_b = range(1,len(a_value)+1)
temp_b =zip(num_b,b_value)
d_b = dict(temp_b)
b_key = d_b.keys()
print(d_b)

c_value = line_c.split()
num_c = range(1,len(c_value)+1)
temp_c =zip(num_c,c_value)
d_c = dict(temp_c)
c_key = d_c.keys()
print(d_c)

whole_D = {}
whole_L = []
whole_L.append(d_a)
whole_L.append(d_b)
whole_L.append(d_c)
print(whole_L)
for x in whole_L:
     print(x)
     for y in x.keys():
          if x[y] == bename:
               start = y
          if x[y] == endname:
               end = y
     print(start, end)
     print(list(x.values()))
     for w in list(x.values())[start-1:end-1]:
          print(w,"----->",end = "")
     print(list(x.values())[end-1])
     print()

#---------------------------------------dict 查询公交路线 有换乘->两次
line_a = "aa bb cc dd ee ff ft hh kk"
line_b = "ss mm ll ff ii gg gk pp"
line_c = "sl ml gg fl ik us gs ps"
bename = "bb"
endname = "gs"
li_a = line_a.split()
li_b = line_b.split()
li_c = line_c.split()
if bename in line_a:
    bename_index = li_a.index(bename)
    print("the begain station num is ",bename_index)
if endname in line_b:
    endname_index = li_b.index(endname)
    print("the begain station num is ",endname_index)
def find_change(a,b):
    linenum = []
    for x in a:
        if x in b:
           comstaion = x
           x_index1 = a.index(x)
           x_index2 = b.index(x)
           temp = (comstaion ,x_index1,x_index2)
           linenum.append(temp)
    return linenum
find_change(li_a,li_b)
find_change(li_b,li_c)
print("-------------")
find_change(li_a,[bename,endname])
find_change(li_b,[bename,endname])
find_change(li_c,[bename,endname])
d = []
d.append(li_a)
d.append(li_b)
d.append(li_c)
b = []    #记录起点所在路线
c = []    #记录终点所在路线   b,c与d中的序号是一一对应的
i = 0
for x in d:
    px = find_change(x,[bename,endname])
    for y in px:
        if y[0] == bename:
            b.append(i)
        if y[0] == endname:
            c.append(i)
    i += 1
print("_____")
print(b,c)
for x in b:
    for y in c:
        print(x,y)
        ret = find_change(d[x],d[y])
        if len(ret) >= 1:
            print("need only once change")
            print(ret)
        if len(ret) == 0:
            print("need more than once")
            ret1 = find_change(d[x],d[3-x-y])
            print(x,3-x-y,ret1)
            ret2 = find_change(d[y],d[3-x-y])
            print(y,3-x-y,ret2)

#---------------------------------------dict 查询公交路线 有换乘->三次
line_string = """aa bb ct uu dd yy fg
tt yy ww mm cc bx hh
ss dd zz xx
ii mm bc gg
tx cx bx xx
pp cx gg nn ee"""
be_name = "bb"
end_name = "ee"
li = []
#li.append(line_string.split())
def split_string(s):
    li_info = s.split("\n")
    i = 0
    for x in li_info:
        #print(x.split())
        li.append(x.split())
        print("line"+str(i),li[i])
        i += 1
    return li
li_1 = split_string(line_string)

begain = []
end = []
change = []
bex = []
endx = []
def common_station(a,b):
    li_com = []
    for x in a:
        if x in b:
            xa_index = a.index(x)
            xb_index = b.index(x)
            temp = (x,xa_index,xb_index)
            li_com.append(temp)
    if len(li_com) == 0:
        return [(-1,-1,-1)]
    return li_com
def change_station(line1,line2):
    i = 0
    for x in line1:
        px = common_station(x,line2)
       # print(px)
        #print("----")
        #print(px[0])
        #print("+++++")
        for p in px:
            if px[0][0] == be_name:
               begain.append(i)
            elif px[0][0] == end_name:
               end.append(i)
            else:
               change.append(i)
            i += 1
    print(begain,end,change)
change_station(li_1,[be_name,end_name])
def find_begain_connect(a,b):
    lll = []
    for x in a :
        for y in b :
            temp1 = common_station(li_1[x],li_1[y])
            if temp1[0][0] != -1:
                lll.append(y)
                print(x,y,temp1)
    return lll
bex = find_begain_connect(begain,change)
endx = find_begain_connect(end,change)
print("begain connect ",bex)
print("end connect ",endx)
finall = find_begain_connect(bex,endx) 
#_____________________________________exercise
s = "aa bb cc dd aa d dss aa kj"
sub = "aa"
s1 = s.split()

#------------------------------------
def max(m,*b):
    print(b)
    for x in b:
        if x > m:
            m = x
    print(m)
max(2,34,56,78,98,54,4,5,)
def add(**kv):
    print(kv)
add(n1 = 1,n2 = 2,n3 = 3,n4 = 4,n5 = 5,n6 =7)

#===========================================lambda 函数
li = [1,2,3,4,5,6,7,8,9,0]
fun = lambda y : y + 100
k = map(fun,li)
print(list(k))

f = lambda x,y : [x+10,y+10]
print(f(1,2))














