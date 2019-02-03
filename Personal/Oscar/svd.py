# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 11:24:52 2018

@author: Òscar
"""


# Singular-value decomposition. Tutorial from https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/
## Import
import numpy as np
from sklearn.decomposition import TruncatedSVD

from scipy.linalg import svd
# define a matrix
A = np.array([
	[1,2,3,4,5,6,7,8,9,10],
	[11,12,13,14,15,16,17,18,19,20],
	[21,22,23,24,25,26,27,28,29,30]])

# SVD
U, s, VT = svd(A)
#Calculate how many singular values we have to take into account -> 80-90%

sumv = np.sum(s)*0.85

cumsumv = np.cumsum(s)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

nearests=find_nearest(cumsumv,sumv)

itemindexs = np.where(cumsumv==nearests)
print(itemindexs)

## n_elements = valueof itemindexs

svd = TruncatedSVD(n_components=3
                  )
svd.fit(A)
result = svd.transform(A)
np.asarray(result)


print(result)
