#matrix multiplication

size_a=(input("enter size of matrix a")).split()
size_b=(input("enter size of matrix b")).split()

# s_a=int(list(size_a))

m1=[]
m2=[]
ans=[[0,0],[0,0]]

if (int(size_a[1])==int(size_b[0])):
     print("enter matrix a\n")
     for i in range(int(size_a[0])):
        a=[]
        for j in range(int(size_a[1])):
            a.append(int(input()))
        m1.append(a)    

     print("enter matrix b\n")
     for i in range(int(size_b[0])):
        b=[]
        for j in range(int(size_b[1])):
            b.append(int(input()))
        m2.append(b)

     for i in range(int(size_a[0])):
         for j in range(int(size_b[1])):
             for a in range(int(size_a[1])):
                 ans[i][j]=ans[i][j]+m1[i][a]*m2[a][j]
                 


else:
    print("It is not a valid operation")

for i in range(0,int(size_a[0])):
    for j in range(0,int(size_b[1])):
        print(ans[i][j],end=" ")   
    print()             