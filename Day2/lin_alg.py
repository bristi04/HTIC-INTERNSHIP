import numpy as np

A=np.array([1,2,4],[4,4,4],[6,6,6])
B=np.array([3,2,3])

x=np.linalg.solve(A,B)
print("Solution: x =", x[0], ", y =", x[1])
