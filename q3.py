list1=input("enter  list1").split()
list2=input("enter list2").split()

count=0
for i in range(0,len(list1)):
      if(list1[i] in list2):
           count=count+1


print("the number of common elements are",count)


