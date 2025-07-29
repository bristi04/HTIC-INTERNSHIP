import numpy as np

def dot_prod(A,B):
    res=0
    for i in range(len(A)):
        res+=A[i]*B[i]
    return res

A=[2,2,7]
B=[9,7,5]
print(dot_prod(A,B))
