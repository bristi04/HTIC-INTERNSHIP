import numpy as np

def matrix_mul(A,B):
    res=np.zeros(3,3)
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                res+=A[i][k]+B[k][j]
    return res

A=[[2,3,4],[4,2,1],[7,8,5]]
B=[[4,5,6],[8,8,4],[2,7,7]]
print(matrix_mul(A,B))
