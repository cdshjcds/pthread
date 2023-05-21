import numpy as np
N,M = 5000,5000
x = np.random.randint(low=-10,high=10,size=N)
A = np.random.randint(low=-10,high=10,size=(M,N))
b = np.matmul(A,x[:,np.newaxis])

Ab = np.concatenate((A,b),axis=1)
np.savetxt('data.txt',Ab,fmt='%d',delimiter=' ')

with open('data.txt') as f:
    data = f.read()

data = str(M)+" "+str(N+1)+"\n"+data

with open('Project2//Project2//data.txt','w') as f:
    f.write(data)

print(x)
