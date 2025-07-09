#to generate a random number using random

import random
l=[]
for i in range(0,100):
    l.append(random.randint(100,150))

print("the mean is ",sum(l)/len(l))    

print(l.sort())
n=len(l)
print("the median is",(l[50]+l[51])/2)
