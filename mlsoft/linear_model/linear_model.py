# -*-coding:utf-8-*-

__author__ = "Jiawang Liu (ljwsummer@gmail.com)"
__date__ = "2013/02/21 12:07"

import ConfigParser
import numpy as np
import mlsoft.common.reader

class Error(Exception):
    pass

class LinearModel(object):
    """base class of linear model"""

    def __init__(self, conf_file):
        if self.__class__ is LinearModel:
            raise Error("this is a abstract class")
        self._load_conf(conf_file)
        data_file = self.conf.get('global', 'train_data')
        self._load_data(data_file)
        self.m = self.data.X.shape[0]
        self.n = self.data.X.shape[1]
        self.theta = np.mat(np.zeros((self.n, 1)))
        self.costs = []

    def _load_conf(self, conf_file):
        self.conf = ConfigParser.ConfigParser()
        self.conf.optionxform = str
        self.conf.read(conf_file)

    def _load_data(self, data_file):
        self.data = mlsoft.common.reader.Data(data_file)

    def _compute_cost(self):
        pass

    def _compute_gradient(self):
        pass

    def _gradient_descent(self):
        pass

    def train(self):
        for x in xrange(self.iters):
            self._gradient_descent()
            cost = self._compute_cost(self.data.X, self.data.Y, self.theta)
            self.costs.append(cost)

    def predict(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass

