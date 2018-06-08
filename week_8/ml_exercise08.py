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