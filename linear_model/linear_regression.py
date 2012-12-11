# -*-coding:utf-8-*-

import numpy as np
import commom.reader
import ConfigParser

class LinearReg(object):
    """linear regression"""

    def __init__(self, conf_file):
        self._load_conf(conf_file)
        self.alpha = self.conf.getfloat('linear_regression', 'alpha')
        data_file = self.conf.get('global', 'train_data')
        self._load_data(data_file)
        self.theta = np.matrix(';'.join([0.0] * self.data.Y.shape[0]))

    def _load_conf(self, conf_file):
        self.conf = ConfigParser.ConfigParser()
        self.conf.optionxform = str
        self.conf.read(conf_file)

    def _load_data(self, data_file):
        self.data = common.reader(data_file)

    def _compute_cost(self, X, Y, theta):
        m = X.shape[0]
        res = X * theta - Y
        j = sum(np.power(res, 2)) / m
        return j

    def _compute_gradient(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass


