# 输出三角矩阵
import re
i=1
j=1
#print(" ")
''''# while j<10:
    i=1
    while i<10:
        if i==j:
           print( i*j,end=" ")
        else:
            print(" -",end="")
        i+=1
    print("")
    j+=1
    
#-----------water flower---------
i=1
while i<10:
    j=0
    while j<10:
        k=0
        while k<10:
           if i**3+j**3+k**3==i*100+j*10+k:
             print(i*100+j*10+k )
           k += 1
        j += 1
    i+=1
i=0
#----------------------------输出三角形
print(int(19/2)+1)
while i<10:
    j=1
    midum=int(20/2)+1
    while j<21:
        if j==midum or j-midum<=i and midum-j<=i :
           print("*",end="")
        else:
            print(" ",end="")
        j+=1
    i+=1
    print("")

#-----------string------------
s0="""nfbd
dnjfikc"""  #------三重引号支持换行输出--------
s1="njk"
print(s0)
s2="What did Tom say?"
'''
'''print(s2, s3)
print(s3)
ss="dfbhdsn\"dhs"  #-----双引号里面输出双引号得加转义字符\
print(ss)
s4="hello world\bh\bb\bj\bn\bj\b"   #----\b前面的字符不输出
print(s4)
i=1
#while i<100:
 #   print(str(i),end=" ")
  #  i+=1
s1="aa"
s2="bb"
i=len(s2)
print(s1+s2,i) 
if s1>s2:
    print("T")
else:
    print("F")
s3="sdhfjaa"
if s1 in s3:          #------判断字符串的包含关系-------
    print("Ture")
else:
    print("False")
i=0
while i<=100:
    print("wss"+str(i))
    i+=1
import random
x=random.randint(100000,1000000)
i=1
while i<20:
    s=random.randint(10000,100000)
    s=str(s)
    if "4" in s and "7" in s:
        print(s)
    else:
        print(s,"F")
    i+=1
j=0
while j<26:
    if j<10:
        print(chr(j+97)+"0"+str(j+1),end="")
    else:
        print(chr(j+97)+str(j+1),end="")
    j+=1
print("")
print(chr(90))
print(ord("a"))
#---------------chr(97)=a,ord("a")=97:字符和整型的转换---------

i=0

while i<len(x):
    print(str(ord(x[i]))+x[i] ,end=" ")
    i += 1

#----------------字符串的匹配与位置识别-----------------
x="hell zipuedu"
b="zipu"
if b in x:                       #------匹配
    print("yesh")
j=0
print(len(x),len(b))
while j<= len(x)-len(b):          #---------位置识别
    i=0
    count=0
    while i<len(b):
  #  if x[j+0]==b[0] and x[j+1]==b[1] and x[j+2]==b[2] and x[j+3]==b[3]:
   #     print(j)
      if x[j+i]==b[i]:
          count +=1
      i+=1
    if count==len(b):
        print(j,x[j],x[j+len(b)-1])
    j+=1
print("")

#  ------输出字符串之间的字符---------------------
x="neifijefhnenfjsnfi"
head="ne"
tail="fi"
j=0
preh=-1   # 比用preh=0好  因为有可能当head是头一个字符时后面正好 preh=h1
h1=0
t1=0
while j<=len(x)-len(head):
    ch=0
    ct=0
    i=0
    while i<len(head):
        if x[j+i]==head[i]:
            ch+=1
        i+=1
    i=0
    while i<len(tail):
        if x[j+i]==tail[i]:
            ct+=1
        i+=1
    if ch==len(head):
        h1=j
        print(x[j+1],"head="+str(h1),end=" "),
    if ct==len(tail):
        t1=j
        print(x[j+1],"tail="+str(t1),end=" "),
        if h1<t1 and preh!=h1:   #  排除了共用一个head的情况,
            k=h1
            sub=""
            while k<t1+len(tail):
                 sub+=x[k]
                 k+=1
            preh=h1
            print(sub)
    j+=1                       

#---------------------字符串的切块     slice:一次性输出一段   eg:s[1:10]--------------------------
x="hek sdladlanjks skdla"
#print(x[4:10],x[11:15],x[:3],x[11:],x[:])
i=0
sub="dla"
while i<len(x)-len(sub)+1:
    if x[i:i+len(sub)]==sub:
        x=x[:i]+x[i+len(sub):]
        i-=1          #   必须得减掉 因为有可能遇到sub正好相邻的情况 如果不减就会执行下面的i+=1，遍历字符串就会漏掉切块后的x[i]
        print(i,x)
    i+=1
print(x)  

#-----------------------str()函数---------------------------

x=" Hek sdladlanjks skdla   "
i=0
sub=""
while i<len(x):
    if x[i].isupper():                 # isupper() 找到第一个大写字母的位置
        print(x[i])
    if x[i].islower():                 # islower()  找到第一个小写字母的位置
        print(x[i],end="")
    if x[i].isspace():                 # isspace()  找到第一个空格的位置
        print("")  
        print(i)
    else:
        sub+=x[i]
    i+=1
print(sub)


x=" Hek sdladlanjks skdla   ji"       #  只删除首尾空格，中间空格保留
i=0
head=-1
tail=-1
sub=""
while i<len(x):
    if not x[i].isspace():
        head=i
        break
    i+=1
i=len(x)-1
while i>=0:
    if not x[i].isspace():
        tail=i
        break
    i-=1
print(head,tail)
print(x[head:tail+1])
i=0
while i<len(x):
    if i>=head and i<=tail:
       sub+=x[i]
    i+=1
print(sub)

#---------- ---------------------- str[].find:从左到右的第一个值／str[].rfind：从右到左的第一个
sub="hello jeapedu.com "*4
s="eape"
a=-1
b=-1
i=0
c=-1
a=sub.find("aa")
c=sub.find("aa",11)
b=sub.rfind("aa")
#print(a,b,c)
print(sub.count(s))                        #sub.count() 子串出现的次数
pos=-4
while i<sub.count(s):
    pos =sub.find(s,pos+len(s))
    print(pos)
    i+=1

s="http://hdcjndsj.comsnjahttp://sabjhasndan.comhttp://bsahb.com"
s1="http://hdcjndsj.comsnjahttp://sabjhasndan.comhttp://bsahb"
head="http://"
tail=".com"
print(len(s),s[56],len(s1))
i=0
j=0
flag=-1
while i<len(s):
    if s[i:i+len(head)]==head:
        j=i
        while j<len(s):
            if s[j:j+len(tail)]==tail:
                flag=j
                break
            j+=1
        print(i,j,s[i:j+len(tail)])
    i+=1

#---------上面例题的另一解法-------------------
s="http://hdcjndsj.comsnjahttp://sabjhasndan.comhttp://bsahb.com"
head="http://"
tail=".com"
posh=-len(head)
post = -len(tail)
i=0
while i<s.count(head):
    posh = s.find(head,post+len(tail))
    post = s.find(tail,posh+len(head))
    print(posh,post)
    print(s[posh:post+len(tail)])
    i+=1

s = "ndjs njkn sk"
print(s.find("dj"))
print(s.index("dj"))    # s.find 与s.index的区别
print(s.find("dkj"))
#print(s.index("dkj"))

#------------------------------------------str.replace
s = "snjd jk ksdn" * 4
print(s)
t = s.replace("nj","KN")
print(t)
t1 = s.replace("sn","SSHJK",3)
print(t1)
#----------------------------------------str.strip    
s = "    njkndka123 4353l l lms kjn"
t=s.strip()
print(s)
print("|"+ t + "*")
s1 = "njknnnnnn53l l lms kjn"
t1 = s1.strip("njk")             #  首尾字符在（）里的都删除，
print(t1)
tt = s1.strip("n").strip("j")
print("tt   "+tt)
s2 = "12332nfjkwnfw mdl 98321231232"
t2 = s2.strip("123")
print(t2)
#   具体实现str.strip()
s = "#$%^*@ndjsad nkjs23 fdkj 89#@$%*"
sub = "#$%^*@"
i = 0
pre = 0
tail = 0
while i < len(s):
    if s[i] not in sub:
        pre = i
        break
    i+=1
print(s[pre:])
s= s[pre:]
i = len(s)-1
while i >=0:
    if s[i] not in sub:
        tail = i + 1
        break
    i-=1
print(s[:tail])


print("nbdjs")

# ======================str.count(x,start,end)
# ============             字符的添加与替换

val = 'a,b,  guido'
val.split(',')
pieces = [x.strip() for x in val.split(',')]
print(pieces)
a1,a2,a3 = pieces
print(a1)
print(a2)
print(a3)
a4 = a1 + '::' + a2 + '::'+ a3      # 字符串的连接
print(a4)
a5 = "::".join(pieces)
print(a5)
'''
#---------------------     正则表达式
text = 'foo bar\tbaz\tquz'
print(text)
print(re.split('\s+',text))    # \s 代表空格
regex = re.compile('\s+')
print(regex.split(text))
print(">>>>>>>")
print(regex.findall(text))
for x in regex.finditer(text):
    print(x)

text = """Dave dave@google.com
Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com
"""
pattern = r'[a-z\d._%+-]+@[a-z\d.-]+\.[a-z]{2,4}'
regex = re.compile(pattern,flags=re.IGNORECASE)    # re.IGNORECASE 的作用是使正则表达式对大小写不敏感
print(regex.findall(text))
m = regex.search(text)
print('//////////')
print(m)
print(text[m.start():m.end()])
print(regex.match(text))
print(regex.sub('REDACTED', text))
regex = re.compile(r"""
    (?P<username>[A-Z0-9._%+-]+)
    @
    (?P<domain>[A-Z0-9.-]+)
    \.
    (?P<suffix>[A-Z]{2,4})""", flags=re.IGNORECASE|re.VERBOSE)
m = regex.match('wesm@bright.net')
print(m)
print(m.groupdict())





























