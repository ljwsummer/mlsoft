#-*-coding:utf-8-*-

import os
import copy
import numpy as np

import mlsoft.linear_model.linear_regression as lr
from unit_tests import config

class TestLinearReg(object):

    def setup(self):
        self.conf = os.path.join(config.model_conf, 'model.cfg')
        self.mat = lambda file : np.matrix(';'.join([x.strip() for x in open(file)]))
        self.model = self.mat(config.linear_reg_model)
        self.model_has_intercept = self.mat(config.linear_reg_model_has_intercept)
        self.predicts = self.mat(config.linear_reg_predicts)
        self.lr = lr.LinearReg(self.conf)

    def teardown(self):
        del self.lr

    def test_train(self):
        def test(lr, has_intercept, model):
            lr.intercept = has_intercept
            lr.train()
            for x in (model - lr.theta).flat:
                assert x < 1e-6
        test(self.lr, True, self.model_has_intercept)
        test(self.lr, False, self.model)

    def test_predict(self):
        self.lr.load_model(config.linear_reg_model)
        for x in (self.predicts - self.lr.predict(self.lr.data.X)).flat:
            assert x < 1e-6

    def test_save_model(self):
        import tempfile
        temp_model_file = tempfile.NamedTemporaryFile().name
        self.lr.save_model(temp_model_file)
        for x in (self.mat(temp_model_file) - self.lr.theta).flat:
            assert x < 1e-6

    def test_load_model(self):
        self.lr.load_model(config.linear_reg_model)
        for x in (self.model - self.lr.theta).flat:
            assert x < 1e-6


