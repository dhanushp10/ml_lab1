r=int(input("enter number of row"))
c=int(input("enter number of column"))

m=[]

for i in range(r):         
    a =[]
    for j in range(c):    
         a.append(int(input()))
    m.append(a)

for i in range(r):
    for j in range(c):
        print(m[i][j], end = " ")
    print()

# finding transpose by swapping (i,j)
for i in range(0,r):
    for j in range(0,c):
        if(i<=j):
           m[i][j],m[j][i]=m[j][i],m[i][j]

print("the transpose of a matrix is ")
for i in range(0,r):
    for j in range(0,c):
        print(m[i][j],end=" ")   
    print()         