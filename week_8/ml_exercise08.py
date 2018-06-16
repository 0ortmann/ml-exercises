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
from scipy.sparse.linalg import spsolve

print('''1 c) calculate sparsity
    The scipy baseclass for sparse matrices offers two functions 'get_shape()' and 'getnnz()', which can be used to calculate the sparsity of a matrix:
    ''')
orig_arr = np.array([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
A = csr_matrix(orig_arr)
shape, nnz = A.get_shape(), A.getnnz()
print('Original matrix data:\n{}\nCSR Matrix:\n{}\nshape: {}\nnon-zero-values: {}'.format(orig_arr, A, shape, nnz))
print('Sparsity = {}\n'.format(nnz / (shape[0] * shape[1])))


## 1 d)
def rand_diag(n):
    diagonals = np.random.randint(1, n, size=(3, n))
    ## use csr format -> better for dot product in 1e)
    return diags(diagonals, [-1, 0, 1], shape=(n, n), format='csr')

print('''1 d) random sparse tri-diagonal matrix
    Example matrix with n=10:''')
n = 10
A = rand_diag(n)
print(A.toarray())

print('''1 e) solve linear equations:
    ''')

import time

print('Sparse matrix solving:')
for n in [10, 100, 1000, 10000]: #, 100000, 1000000, 10000000]:
    start = time.time()
    A = rand_diag(n)
    b = np.ones(n)
    spsolve(A, b)
    print('n={} --> {} seconds'.format(n, time.time()-start))
print('With n=10000000 I used more than 7G ram, this was the maximum I could solve on my machine. But it took only ~9.2 seconds.\n')

print('Dense matrix solving:')
for n in [10, 100, 1000]: #, 10000]:
    start = time.time()
    A = rand_diag(n)
    b = np.ones(n)
    np.linalg.solve(A.todense(), b)
    print('n={} --> {} seconds'.format(n, time.time()-start))

print('''It took more than 236 seconds (~4 minutes) to solve the dense matrix for n=10000 on my machine. For n=100000 I got a memory error when calling the '.todense()' function. The optimizations that can safely be made for sparse matrices have a huge impact on runtime.''')


print('\n\nassignment 2: "20 Newsgroups" data set\n')

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_newsgroups = fetch_20newsgroups(categories=categories, shuffle=True, random_state=41)

print('''2 a) general data properties:
    files: {}
    labels: {}
    classes: {}
    '''.format(len(twenty_newsgroups.filenames), twenty_newsgroups.target_names, np.unique(twenty_newsgroups.target)))

## 2 b)

count_vect = CountVectorizer()
word_counts = count_vect.fit_transform(twenty_newsgroups.data)

print('''2 b) tokenize data:
    - Found words: {}
    - Access word list: use count_vect.get_feature_names() for a full list of all words
    - Find index for given word: use count_vect.vocabulary_.get(...)'''.format(word_counts.shape[1]))


## 2 c)

classify_text = ['didactics are important for university courses', 'new smartphone released', 'apples or bananas?', 'apple mac book', 'chicken tikka plate']

def do_2c(word_counts):
    tfidf_transformer = TfidfTransformer()
    tfidf = tfidf_transformer.fit_transform(word_counts)

    nb_classifier = MultinomialNB().fit(tfidf, twenty_newsgroups.target)

    clsfy_counts = count_vect.transform(classify_text)
    clsfy_tfidf = tfidf_transformer.transform(clsfy_counts)

    predicted = nb_classifier.predict(clsfy_tfidf)
    return predicted

print('2 c) classify own text to the imported word-bags:')

predicted = do_2c(word_counts)
for doc, category in zip(classify_text, predicted):
    print('{} => {}'.format(doc, twenty_newsgroups.target_names[category]))

print('''\n2 d) Stopwords
    Stopwords are used to filter out common words with few meaning / impact on single search terms. Common stopwords are 'is, are, a, the, for, to, from....'.
    Redo 2 c) with enabled english stopwords:''')

count_vect = CountVectorizer(stop_words='english')
word_counts = count_vect.fit_transform(twenty_newsgroups.data)
print('Found words:', word_counts.shape[1])
predicted = do_2c(word_counts)
for doc, category in zip(classify_text, predicted):
    print('{} => {}'.format(doc, twenty_newsgroups.target_names[category]))

print('''\n2 e) TF-IDF
    Word counts only describe which words are used often. In longer texts, there are more words used in general than in short texts.
    With word frequencies it is more easy to get a grasp of the importance of a word for a the text it appears in. Considering the word frequency strips away the importance of the length of the document.
    This can even be improved by using TF-IDF (term frequency x inverse document frequency). IDF looks at a whole text corpus. Words that are frequent among a majority of texts are treated as being less informative, thus get a lower score. IDF is much less disturbing then stopwords. 
    ''')

print('''assignment 3: invoking script from 'http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html' with different parameters:

    - normal execution with --report --confusion_matrix:
      average detection precision is for all tested classifiers somewhere in the late 80s or even 90%. There seems to be a general confusion for approx. 40 documents of the 4th class, that are commonly misclassified as the 1st class.
    - use --filtered:
      After filtering headers, quotes and emails, the classification got worse. For most classifiers the precision is around 75%, which is clearly below the unfiltered processing. i.e. then kNN precision suffered, that classifier is completely broken with this parameter. It has only 26% precision.
    - use --all_categories:
      With the bigger corpus, prediction became a little less accurate. On average the classifiers were between 75% and 85% accurate. There is no clear winner / best classifier.
      The best four with each 86% are: Ridge, PassiveAggressive, SGDClassifier and LinearSVC. It appears for this kind of classification, kNN is performing least among all three test runs.''')

print('''Notes from presentation in class:
    The removal of email addresses leads to miss-classification, because the author is a very clear indicator for which category a text belongs to.
    All-categories: much confusion between atheism and christianity.
    Other people have tested training times and shit.
    In general: i did a poor job on explaining the 'whys' of the runs.''')