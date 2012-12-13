# -*-coding:utf-8-*-

import os

from unit_tests import config
from mlsoft.common import reader

class TestData(object):

    def setup(self):
        self.path = os.path.join(config.test_data_path, 'test_data.txt')
        self.data = reader.Data(self.path)

    def teardown(self):
        del self.data

    def test_append_line(self):
        m, n = self.data.X.shape
        self.data.append_line('1 qid:2 1:0.21 3:0.11 27:-0.32')
        m1, n1 = self.data.X.shape
        assert m1 == m + 1
        assert n1 == n
        assert self.data.Y[m, 0] == 1
        assert self.data.qid[m, 0] == 2
        assert self.data.X[m, 0] == 0.21
        assert self.data.X[m, 2] == 0.11
        assert self.data.X[m, 26] == -0.32
        for x in xrange(n):
            if x not in [0, 2, 26]:
                assert self.data.X[m, x] == 0

    def test_append_file(self):
        m, n = self.data.X.shape
        self.data.append_file(self.path)
        assert (self.data.X[:m] == self.data.X[m:]).all()
        assert (self.data.Y[:m] == self.data.Y[m:]).all()
        assert (self.data.qid[:m] == self.data.qid[m:]).all()

