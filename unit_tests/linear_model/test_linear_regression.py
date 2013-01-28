#-*-coding:utf-8-*-

import os
import copy
import numpy as np

import mlsoft.linear_model.linear_regression as lr
from unit_tests import config

class TestLinearReg(object):

    def setup(self):
        self.conf = os.path.join(config.model_conf, 'model.cfg')
        self.lr = lr.LinearReg(self.conf)

    def teardown():
        del self.lr

    def test_train():
        self.lr.train()
        theta = copy.deepcopy(self.lr.theta)

    def test_predict():
        pass

    def test_save_model():
        pass

    def load_model():
        pass


