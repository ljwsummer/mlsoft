#!/usr/bin/python
# -*-coding:utf-8-*-

import os
from unit_tests import config
from common import reader
def setUp():
    pass

def tearDown():
    pass


def test_append_line():
    path = os.path.join(config.unit_tests_path, 'common/data/test_data.txt')
    data = reader.Data(path)
    m, n = data.X.shape
    data.append_line('1 qid:2 1:0.21 3:0.11 27:-0.32')
    m1, n1 = data.X.shape
    assert m1 == m + 1
    assert n1 == n
    assert data.Y[m, 0] == 1
    assert data.qid[m, 0] == 2
    assert data.X[m, 0] == 0.21
    assert data.X[m, 2] == 0.11
    assert data.X[m, 26] == -0.32
    for x in xrange(n):
        if x not in [0, 2, 26]:
            assert data.X[m, x] == 0


def test_append_file():
    path = os.path.join(config.unit_tests_path, 'common/data/test_data.txt')
    data = reader.Data(path)
    m, n = data.X.shape
    data.append_file(path)
    assert (data.X[:m] == data.X[m:]).all()
    assert (data.Y[:m] == data.Y[m:]).all()
    assert (data.qid[:m] == data.qid[m:]).all()

