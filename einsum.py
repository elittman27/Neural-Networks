import numpy as np
np.random.seed(100)

## 6.1 (use numpy and einsum functions)
# a. For a random 5x5 matrix A, find its trace
A = np.random.rand(5, 5)

traceA = np.einsum('ii->', A)
traceA_np = np.trace(A)

print(traceA - traceA_np)

# b. For A, B, compute the matrix product
B = np.random.rand(5, 5)
C = np.einsum('ik, kj -> ij', A, B)
C_np = A @ B

print(C - C_np)

# c. For a batch of random matrices of shapes (3, 4, 5) and (3,5,6), compute matrix product (3,4,6)
D = np.random.rand(3, 4, 5)
E = np.random.rand(3, 5, 6)
F = np.einsum('ijk, ikh -> ijh', D, E)
F_np = D @ E
print(F - F_np)


