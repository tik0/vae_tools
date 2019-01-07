#!/usr/bin/python
import numpy as np


def windowing(A, L, S = 1, asarray = False):  # Window size = L, Stride (aka stepsize) = S
    ''' windowing: Given an NxM array, windowing will produce an Lx(M*(N-L)/S) array (w/o padding!!!)
    A: any np.array of dimension NxM
    L (int): size of window
    S (int): stride of window
    asarray (bool): Returns as single array instead of list of windowed sequences
    '''
    M = A.size[1]
    B = []
    for idx in np.arange(M):
        B.append(_windowing(A[idx,:], L, S))
    if asarray:
        return np.asarray(B)
    else:
        return B

def _windowing(a, L, S = 1 ):  # Window size = L, Stride (aka stepsize) = S
    ''' _windowing: Given an Nx1 array, windowing will produce an Lx((N-L)/S) array (w/o padding!!!)
    a: any np.array of dimension Nx1
    L (int): size of window
    S (int): stride of window
    '''
    nrows = ((len(a)-L)//S)+1
    n = a.strides[0] # steps along the zero dimension
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

def norm_min0_max1(A):
    ''' norm_min0_max1: Normalizes a NxM array along the N dimension to have minimum 0 and maximum 1 values per column
    A: Any np.array of NxM dimension
    '''
    A = A - np.expand_dims(np.amin(A,axis=1),1)
    A = A / np.expand_dims(np.amax(A,axis=1),1)
    return A

def split_train_test(X, labels, split = 0.99, shuffle = True):
    ''' split_train_test: Splits a NxM array X into a training set of size N(split)xM and testset of size N(1-split)xM
    A: Any np.array of NxM dimension
    labels: Any np.array of Nx1 dimension
    split (float): Percentage of training set
    shuffle (bool): Shuffle the sets
    '''
    idx = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(idx)
    train_size = np.int(len(X) * split)
    train = X[idx[:train_size],:]
    test = X[idx[train_size:],:]
    train_label = labels[idx[:train_size]]
    test_label = labels[idx[train_size:]]
    return train, test, train_label, test_label, idx