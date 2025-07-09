#to generate a random number using random

def most_frequent(List):
    return max(set(List), key=List.count)

List = [2, 1, 2, 2, 1, 3]
print(most_frequent(List))

import random
l=[]
for i in range(0,100):
    l.append(random.randint(100,150))

print("the mean is ",sum(l)/len(l))    

print(l.sort())
n=len(l)
print("the median is",(l[50]+l[51])/2)

print("the mode is ",most_frequent(l))