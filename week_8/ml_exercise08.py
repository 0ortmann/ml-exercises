#!/usr/bin/python3

print('assignment 1: sparse matrices, implementations and properties')

print('''1 a) different implementations
    The different implementations come with different runtime properties and memory layouts. Each of them has (dis-) advantages over the others. Basically, sparse matrix implementations can be divided into two classes: 
        - easy to construct (insert, delete ..)
        - easy to access / operate with (traverse in order, multiply ..)
    CSR: "compressed sparse rows" - sparse matrices in this form are layouted as a 3-tuple of arrays in memory (A, IA, JA) for MxN matrix:
        A: holds all non-zero values (let K) in the order of appearance from left-right top-bottom (length = K)
        IA: holds a count of all non-zero values per row plus all previously seens non-zero values (length = M)
        JA: column index of each element in A (length = K)
    ''')

print('''1 b) operation performance
    Inserting elements is best with DOK, LIL and COO.
    - DOK: dict of keys: good for incremental construction + O(1) access
    - LIL: list of lists: row-based linked list. Good for construction, but insertion could take O(N) -> insert sorted elements.
    - COO: coordinate format (list of row-col-value tuples): good for insertion, keep values sorted in list (by row, then col). Good to transform to CSR/CSC
    Matrix Operations (Vector dot product, matrix multiplication) is best with CSR + CSC:
    - CSR: fast row slicing, slow col slicing. Efficient for traversal -> dot product / multiplicaiton
    - CSC: fast col slicing, slow row slicing. Efficient for traversal -> dot product / multiplicaiton
    ''')


import numpy as np
from scipy.sparse import csr_matrix, diags

print('''1 c) calculate sparsity
    The scipy baseclass for sparse matrices offers two functions 'get_shape()' and 'getnnz()', which can be used to calculate the sparsity of a matrix:
    ''')
orig_arr = np.array([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
A = csr_matrix(orig_arr)
shape, nnz = A.get_shape(), A.getnnz()
print('Original matrix data:\n{}\nCSR Matrix:\n{}\nshape: {}\nnon-zero-values: {}'.format(orig_arr, A, shape, nnz))
print('Sparsity = {}\n'.format(nnz / (shape[0] * shape[1])))

print('''1 d) random sparse tri-diagonal matrix
    Example matrix with n=10:''')

n = 10
diagonals = np.random.randint(1, n, size=(3, n))
A = diags(diagonals, [-1, 0, 1], shape=(n, n))
print(A.toarray())