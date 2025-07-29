import numpy as np

def matrix_vec(A,B):
    res=np.zeros(len(A),len(B[0]))
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                 res[i][j]+=A[i][j]*B[i][j]
    return res


A=np.array([2,3,4],[4,4,3],[9,7,4])
B=np.array([2,3,1])

print(matrix_vec(A,B))
