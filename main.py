import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp
# create a 2-D representation of the matrix
A = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 1],\
 [0, 0, 0, 2, 0, 0]])
C = np.array([[1, 5, 0, 8, 0, 0], [7, 0, 2, 9, 0, 1],\
 [0, 8, 0, 2, 3, 0]])
print("Dense matrix representation: \n", A)

# convert to sparse matrix representation
S = csr_matrix(A)
print("Sparse matrix: \n",S)

# convert back to 2-D representation of the matrix
B = S.todense()
print("Dense matrix: \n", B)
r=[1 ,2, 3]
c=[4 ,2,6]
v=[0,0,3]

m=sp.csr_matrix((v, (r, c)), shape=(4, 7))
m=sp.csr_matrix(m)
print(m)
m.eliminate_zeros()
print(np.shape(m))
r = [0, 1, 1, 2, 3]
c = [0, 1, 1, 2, 3]
v = [1, 2, 3, 4, 5]

# Get unique row and column indices
# Create sparse matrix from unique indices and values
m = csr_matrix((v, (r, c)), shape=(4, 4))

print(m.toarray())
print(np.shape(m))

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Remove first row
new_matrix = matrix[1:, :]

# Add first column as last column
#
print(matrix)
print()
print(new_matrix)
new_matrix = np.concatenate((new_matrix, new_matrix[:, 0].reshape(-1, 1)), axis=1)
print()
new_matrix=new_matrix[:,1:]
print(new_matrix)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
