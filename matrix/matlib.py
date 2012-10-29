#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
matrix lib
"""

from __future__ import absolute_import

import random

from matrix import  matrix

__author__ = "Jiawang Liu <ljwsummer@gmail.com>"
__version__ = "1.0.0"

# custom matrix
def custom(m, n, elem):
    row = [str(elem)] * n
    raw_matrix = ';'.join([','.join(row)] * m)
    ret = matrix.Matrix(raw_matrix)
    return ret

# zero matrix   
def zeros(m, n = -1):
    if n == -1:
        n = m
    return custom(m, n, 0)

# a matrix whose elements are all 1
def ones(m, n = -1):
    if n == -1:
        n = m
    return custom(m, n, 1)

# random matrix, each element is a random number between 0 and 1
def rand(m, n = -1):
    if n == -1:
        n = m
    ret = zeros(m, n)
    for x in xrange(1, m+1):
        for y in xrange(1, n+1):
            ret.set(x, y, random.random())
    return ret

# diagonal matrix
def eye(m, n = -1):
    if n == -1:
        n = m
    ret = zeros(m, n)
    min = m if m < n else n
    for x in xrange(1, min+1):
        ret.set(x, x, 1)
    return ret

# custom diagonal matrix or return the diagonal element of a matrix
def diag(ins):
    if isinstance(ins, matrix.Matrix):
        min = ins.m if ins.m < ins.n else ins.n
        temp = []
        for x in xrange(1, min+1):
            temp.append(str(ins.get(x,x)))
        return matrix.Matrix(';'.join(temp))
    elif isinstance(ins, list):
        ret = zeros(len(ins))
        for x in xrange(1, ret.m+1):
            ret.set(x, x, ins[x-1])
        return ret
    else:
        print >> sys.stderr, 'ERROR the parameter must be a matrix or list.'
        
# trnaspose of matrix   
def trans(matrix):
    ret = zeros(matrix.n, matrix.m)
    for x in xrange(matrix.n):
        for y in xrange(matrix.m):
            ret.set(x, y, matrix.get(y, x))
    return ret

  