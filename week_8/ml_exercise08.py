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