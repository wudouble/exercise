li = ['a','b','c','d','e','f','f','g','k']
'''
print(li)
tset = tuple(li)
print(tset)
print(set(tset))                #   集合是乱序的
liset = set(li)
print(liset)
d = dict(a=1,b=2,c=3,d=1)
dset = set(d)
dset1 = set(d.values())
print(dset)
#print(dset1[0])               #  集合不支持索引
sset = set("123456")
print(sset)
for x in sset:                #  集合可以通过迭代取出
    print(x)
i = 0
#while i < len(sset):          #   不行  因为不支持索引
#    print(sset[i])
 #   i += 1

ss = {1,2,3,4,5}
print(ss)
print(type(ss))

li1 = range(1,9)
li2 = range(5,10)
set1 = set(li1)
set2 = set(li2)
print(set1)
print(set2)
print(set1.intersection(set2))  # 交集
print(set1)
print("*****************")
print(set1.intersection_update(set2))       #   改变set1 的值
print(set1)
print(set1.union(set2))                    # 并集  可以多个一起求并集
print(set1.difference(set2))               # 差集
print(set2.difference(set1))
t1 = set1.intersection(set2)
if len(t1) == len(set1):            # 拿出的元素为非重复的元素，不包括原始有重复的数据
    print("set2 all in set1")
else:
    print("not the same")

def isin(a,b):
    c = 0
    for x in a:
        if x in b:
            c += 1
    if c == len(a):
        print('in ')
    else:print("not in")
isin(li1,li2)

#      两个元组的不同元素
tup1 = tuple("1234356")
tup2 = tuple("45679480")
print(tup2.__add__(tup1))         #    元组是整个元组一起添加的
print(tup1)
print(tup2)
set1 = set(tup1)
set2 = set(tup2)

for x in set2:
    set1.add(x)            # set.add() 单个元素添加  不能整个一起添加
print("============")
print(set1)
print(set1,set2)
s = set1.union(set2) - set1.intersection(set2)
print(s)
d = list()
for x in tup1:
    if x not in tup2:
        if x not in d:
            d.append(x)
print(d)


#               找出字典里面有重复值的键
dic = {'a':1,'b':2,'c':3,'d':4,'e':3,'f':3,'g':2}
print(dic)
key = list(dic.keys())
value =list(dic.values())
print(key,value)
set_value = list(set(value))
print(set_value)
for x in set_value:
    if value.count(x) >1:
        for y in dic:
            if dic[y] == x:
                print(y,x)
print(value)
for x in value:
    if value.count(x) ==1:
        value.remove(x)                      #   作删除运算时 得使用while 循环  因为for 无法i-=1，会存在遗漏
replicat_value = list(set(value))
lis = []
for x in replicat_value:
    for y in dic:
        if dic[y] == x:
            temp =(y,x)
            lis.append(temp)
print(dict(lis))

#   删除 a 中含有的 b
#    方法一
a = [1,2,3,3,3,4,5,5,6,6,4]
b = [1,2,3,5]
for x in b:
    while x in a:
        a.remove(x)
print(a)
c = 0
#  方法二
for x in b:
    c = a.count(x)
    for y in range(c):
        a.remove(x)
print(a)

tup1 = tuple("1234356")
tup2 = tuple("45679480")
print(tup2.__add__(tup1))

'''
li1 = range(1,9)
li2 = range(5,10)
l1 = set(li1)
l2 = set(li2)
print(l1)
print(l2)
print(set.union(*l2))
print(set.union(l1,l2))





















