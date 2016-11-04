# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg

    
#Approximate a matrix with a Toeplitz matrix
def toeplitz_projection(mat):
    
    rows, cols = mat.shape
    toeplitz = np.zeros(mat.shape, dtype=np.complex)
    row_id, col_id = np.indices(mat.shape)
    
    for offset in (range(-rows+1,0) + range(cols)):
        val = np.mean(mat.diagonal(offset))
        
        rid = row_id.diagonal(offset)
        cid = col_id.diagonal(offset)            

        toeplitz[rid, cid] = val*np.ones(len(mat.diagonal(offset)))
        
    return toeplitz
    
def hankel_projection(mat):
    
    return np.fliplr(toeplitz_projection(np.fliplr(mat)))
    
#builds a Toeplitz matrix from a vector
def to_toeplitz(coefficients, n_col):
    tc = coefficients[(n_col-1):]
    tr = coefficients[np.arange(n_col-1, -1, -1)]
    
    return linalg.toeplitz(tc, tr)
    
#builds a vector from a Toeplitz matrix
def to_measurements(toeplitz):
    n_col = toeplitz.shape[1]
    return np.concatenate((toeplitz[0,np.arange(n_col-1, -1, -1)], toeplitz[1:,0]))

#builds a block Toeplitz matrix from a matrix (each row is mapped to a different block)   
def to_block_toeplitz(matrix, n_col, weights=False):
    A0 = np.zeros((0,n_col), dtype=np.complex_)
    #build a block Toeplitz matrix
    for m in range(matrix.shape[0]):
        to = to_toeplitz(matrix[m,:], n_col)
        if weights:
            A0 = np.vstack((A0, np.exp(-m)*to))
        else:
            A0 = np.vstack((A0, to))
    
    return A0
